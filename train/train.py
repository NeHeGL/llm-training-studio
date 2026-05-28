"""
LLM Training Studio - Training Script (NeHe Productions)
Author: Jeff Molofee (aka NeHe) — 2026
Fine-tunes a base LLM with QLoRA (4-bit) or full BF16/FP16 via HuggingFace PEFT + TRL.

Automatically adapts to your GPU — uses the best features available for your card:
  - Blackwell (RTX 50xx, SM 12.x): 4-bit QLoRA, FP16 compute (BF16 cuBLAS broken in BNB on SM12)
  - Ampere / Ada / Hopper (RTX 30/40xx, A100, H100): 4-bit QLoRA with BF16 or full BF16
  - Turing / Volta / Pascal (RTX 20xx, GTX 16xx, older): 4-bit QLoRA with FP16
  - No GPU / CPU: warns and continues (very slow)

  The model you pick in the Training Studio determines what fits; the script adapts
  batch size, gradient checkpointing, and sequence length automatically.
  Rough 4-bit QLoRA capacity guide (model size is set in config.json / Studio):
    4–6 GB VRAM  → 1–3B models comfortably
    8  GB VRAM   → 7B models (tight — seq_len auto-reduced, grad-ckpt ON)
    12 GB VRAM   → 7–9B models
    16 GB VRAM   → up to ~13B with 4-bit (batch=2, grad-ckpt auto)
    24 GB VRAM   → up to ~30B with 4-bit, or 7B full BF16/FP16
    32 GB VRAM   → up to ~32B with 4-bit
    48+ GB VRAM  → 70B+ with 4-bit

Usage:
    python train\\train.py                          # auto-detects GPU, prompts to resume if checkpoint found
    python train\\train.py --fresh                  # wipe checkpoints and start from scratch
    python train\\train.py --resume popai_lora/checkpoint-NNN  # resume from specific checkpoint
    python train\\train.py --epochs 5               # more epochs
    python train\\train.py --batch 4                # bigger batch (watch VRAM)
    python train\\train.py --low-mem                # if crashing with OOM / eating all your RAM

Memory / RAM troubleshooting:
    --low-mem       Aggressive RAM saving mode. Forces gradient checkpointing ON,
                    batch size=1, sequence length capped at 1024, and tightens the
                    CUDA allocator. Roughly halves peak RAM at the cost of ~2× slower
                    training.  Use this first if you are running out of memory.
    --grad-ckpt     Enables gradient checkpointing only (saves ~3 GB VRAM, ~40% slower).
    --batch 1       Reduce per-device batch size (fewer activations held in RAM).
    --max_seq_len 512  Shorten sequences (biggest single RAM lever — each token costs memory).

After training, run:
    python train/generate_llm.py   # merge LoRA + export to models/<ModelName>/
"""

import argparse
import os
import sys
import shutil

# ── UTF-8 everywhere on Windows (must happen before any TRL/HF imports) ──────
# Re-launch with PYTHONUTF8=1 + -X utf8 if not already in UTF-8 mode.
# Both env var and flag are set for maximum compatibility across Python versions.
if sys.platform == "win32" and sys.flags.utf8_mode == 0:
    import subprocess
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Pass stdin=DEVNULL so the relaunched child also has no TTY attached,
    # ensuring GetConsoleMode() returns False in the child process.
    result = subprocess.run(
        [sys.executable, "-X", "utf8"] + sys.argv,
        env=env,
        stdin=subprocess.DEVNULL if not sys.stdin.isatty() else None,
    )
    sys.exit(result.returncode)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # Do NOT wrap sys.stdin — when launched by the server with stdin=DEVNULL,
    # wrapping the NUL device buffer can cause slow initialization or blocking.
    # GetConsoleMode() is used instead to detect interactive vs non-interactive.

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_ROOT  = os.path.join(ROOT, "models")

# ── Consolidate __pycache__ at project root ───────────────────────────────────
if not sys.pycache_prefix:
    sys.pycache_prefix = os.path.join(ROOT, "__pycache__")

import re as _re

def _safe_name(name):
    """Sanitise a name for use as a filesystem directory component."""
    return _re.sub(r'[^\w\-]', '_', name.strip())


def _load_config():
    """Load active model config from config.json + per-model config file.

    Supports two config formats:
      v1 (legacy): "models" is a list of dicts — reads directly from root config.json
      v2 (current): "models" is a list of name strings — reads per-model config from
                    models/{Name}/config.json
    """
    import json as _json

    cfg_path = os.path.join(ROOT, "config.json")
    default = {"name": "", "base_model": "Qwen/Qwen2.5-7B"}
    if not os.path.exists(cfg_path):
        return default
    try:
        cfg = _json.load(open(cfg_path, encoding="utf-8"))
        active_name = cfg.get("active", "")
        models = cfg.get("models", [])

        if not models:
            return default

        # v1 format: models list contains dicts
        if isinstance(models[0], dict):
            for m in models:
                if m.get("name") == active_name:
                    return m
            return models[0]

        # v2 format: models list contains name strings
        # Load the active model's per-model config file
        name_to_load = active_name if active_name in models else models[0]
        model_cfg_path = os.path.join(ROOT, "models", _safe_name(name_to_load), "config.json")
        if os.path.exists(model_cfg_path):
            return _json.load(open(model_cfg_path, encoding="utf-8"))

        # Per-model config missing — return a minimal default with the name
        return {"name": name_to_load, "base_model": "Qwen/Qwen2.5-7B"}
    except Exception as e:
        print(f"[train] Warning: could not load config: {e}")
        return default


# ── Power management (Windows) ────────────────────────────────────────────────
# Prevent the OS from sleeping, hibernating, or triggering a screensaver while
# training is running.  Uses SetThreadExecutionState via ctypes — no external
# process or administrator rights required.
# Previous powercfg values are read and saved before changing, then restored.

_saved_power_settings = {}   # populated by _prevent_sleep(), consumed by _restore_sleep()


def _read_powercfg_value(setting_name):
    """
    Read a single powercfg /query value by name (e.g. 'standby-timeout-ac').
    Returns an int (minutes) or None if not readable.
    Tries 'powercfg /query SCHEME_CURRENT' and parses the relevant line.
    Falls back to 'powercfg /export' + xml parse if needed.
    """
    import subprocess, re
    # Map our short names to the GUIDs powercfg uses, and to the /change keyword
    _CHANGE_MAP = {
        "standby-timeout-ac":   ("SUB_SLEEP", "STANDBYIDLE",    "standby-timeout-ac"),
        "standby-timeout-dc":   ("SUB_SLEEP", "STANDBYIDLE",    "standby-timeout-dc"),
        "hibernate-timeout-ac": ("SUB_SLEEP", "HIBERNATEIDLE",  "hibernate-timeout-ac"),
        "hibernate-timeout-dc": ("SUB_SLEEP", "HIBERNATEIDLE",  "hibernate-timeout-dc"),
        "monitor-timeout-ac":   ("SUB_VIDEO", "VIDEOIDLE",      "monitor-timeout-ac"),
    }
    if setting_name not in _CHANGE_MAP:
        return None
    try:
        result = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT"],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.splitlines()
        sub_key, idle_key, _ = _CHANGE_MAP[setting_name]
        ac_mode = "standby-timeout-ac" in setting_name or "hibernate-timeout-ac" in setting_name or setting_name == "monitor-timeout-ac"
        # Find the section matching idle_key then grab AC or DC value
        in_section = False
        ac_next = False
        dc_next = False
        for line in lines:
            if idle_key in line.upper():
                in_section = True
            if in_section:
                if "CURRENT AC POWER SETTING INDEX" in line.upper():
                    ac_next = True
                elif "CURRENT DC POWER SETTING INDEX" in line.upper():
                    dc_next = True
                m = re.search(r"0x([0-9a-fA-F]+)", line)
                if m and (ac_next or dc_next):
                    val_sec = int(m.group(1), 16)
                    val_min = val_sec // 60
                    if ac_next and ac_mode:
                        return val_min
                    if dc_next and not ac_mode:
                        return val_min
                    ac_next = dc_next = False
    except Exception:
        pass
    return None


def _prevent_sleep():
    """
    Read current powercfg timeouts, save them, then set all to 0.
    Also calls SetThreadExecutionState to block sleep at the kernel level.
    Safe no-op on non-Windows platforms.
    """
    global _saved_power_settings
    if sys.platform != "win32":
        return

    import subprocess

    # ── Read & save current values ────────────────────────────────────────────
    settings_to_save = [
        "standby-timeout-ac", "standby-timeout-dc",
        "hibernate-timeout-ac", "hibernate-timeout-dc",
        "monitor-timeout-ac",
    ]
    for s in settings_to_save:
        val = _read_powercfg_value(s)
        _saved_power_settings[s] = val  # may be None if unreadable
    saved_str = ", ".join(f"{k}={v}" for k, v in _saved_power_settings.items())
    print(f"[power] Saved power settings: {saved_str}")

    # ── SetThreadExecutionState — kernel-level sleep block ────────────────────
    try:
        import ctypes
        ES_CONTINUOUS        = ctypes.c_uint(0x80000000)
        ES_SYSTEM_REQUIRED   = ctypes.c_uint(0x00000001)
        ES_AWAYMODE_REQUIRED = ctypes.c_uint(0x00000040)
        flags = ctypes.c_uint(ES_CONTINUOUS.value | ES_SYSTEM_REQUIRED.value | ES_AWAYMODE_REQUIRED.value)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        print("[power] Sleep/hibernate prevention active (SetThreadExecutionState).")
    except Exception as e:
        print(f"[power] Could not set execution state: {e}")

    # ── Zero out powercfg timeouts ────────────────────────────────────────────
    try:
        cmds = [
            ["powercfg", "/change", "standby-timeout-ac",   "0"],
            ["powercfg", "/change", "standby-timeout-dc",   "0"],
            ["powercfg", "/change", "hibernate-timeout-ac", "0"],
            ["powercfg", "/change", "hibernate-timeout-dc", "0"],
            ["powercfg", "/change", "monitor-timeout-ac",   "0"],
        ]
        for c in cmds:
            subprocess.run(c, capture_output=True)
        print("[power] powercfg standby/hibernate timeouts set to 0.")
    except Exception as e:
        print(f"[power] powercfg adjustment skipped: {e}")


def _restore_sleep():
    """
    Restore the powercfg timeouts that were saved by _prevent_sleep().
    Falls back to safe defaults if values were not readable.
    Also clears SetThreadExecutionState.
    """
    if sys.platform != "win32":
        return

    import subprocess

    # ── Clear SetThreadExecutionState ─────────────────────────────────────────
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000))  # ES_CONTINUOUS only
        print("[power] Sleep/hibernate prevention cleared.")
    except Exception as e:
        print(f"[power] Could not restore execution state: {e}")

    # ── Restore saved powercfg values (fallback if None or 0) ────────────────
    # A saved value of 0 means either: (a) we couldn't read the original, or
    # (b) the user had already set it to 0 (never sleep) before training started.
    # In case (b) we would restore 0 → never sleep, which is correct.
    # But case (a) is indistinguishable from (b), so we use the fallback only
    # when val is None (unreadable).  0 is a valid user setting and is preserved.
    fallbacks = {
        "standby-timeout-ac":   30,
        "standby-timeout-dc":   15,
        "hibernate-timeout-ac": 180,
        "hibernate-timeout-dc": 60,
        "monitor-timeout-ac":   15,
    }
    restored = []
    try:
        for setting, fallback in fallbacks.items():
            val = _saved_power_settings.get(setting)
            if val is None:
                val = fallback   # couldn't read original — use safe default
            subprocess.run(
                ["powercfg", "/change", setting, str(val)],
                capture_output=True,
            )
            restored.append(f"{setting}={val}")
        print(f"[power] powercfg timeouts restored: {', '.join(restored)}")
    except Exception as e:
        print(f"[power] powercfg restore skipped: {e}")


# ── Fresh-start: wipe existing LoRA checkpoints ───────────────────────────────

def _checkpoint_base_model(checkpoint_path: str) -> str | None:
    """
    Read the base_model_name_or_path from a checkpoint's adapter_config.json.
    Returns the model name string, or None if the file is missing / unreadable.
    """
    import json as _json
    cfg = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(cfg):
        return None
    try:
        with open(cfg, encoding="utf-8") as f:
            data = _json.load(f)
        return data.get("base_model_name_or_path")
    except Exception:
        return None


def _read_training_info(lora_dir: str) -> dict:
    """
    Read the .training_info.json file from lora_dir.
    Returns a dict with keys 'model_name' and 'base_model', or {} if missing/unreadable.
    """
    import json as _json
    info_path = os.path.join(lora_dir, ".training_info.json")
    if not os.path.exists(info_path):
        return {}
    try:
        with open(info_path, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}


def _write_training_info(lora_dir: str, model_name: str, base_model: str):
    """
    Write (or update) .training_info.json in lora_dir with the current model identity.
    Called just before training starts so the file is always up-to-date.
    """
    import json as _json
    os.makedirs(lora_dir, exist_ok=True)
    info_path = os.path.join(lora_dir, ".training_info.json")
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            _json.dump({"model_name": model_name, "base_model": base_model}, f, indent=2)
    except Exception as e:
        print(f"[train] Warning: could not write .training_info.json: {e}")


def _wipe_lora_checkpoints(lora_dir):
    """Delete all checkpoint-* subdirs and final adapter files from lora_dir."""
    if not os.path.isdir(lora_dir):
        return
    removed = []
    for name in os.listdir(lora_dir):
        full = os.path.join(lora_dir, name)
        if name.startswith("checkpoint-") and os.path.isdir(full):
            shutil.rmtree(full, ignore_errors=True)
            removed.append(name)
    for fname in ("adapter_config.json", "adapter_model.safetensors",
                  "adapter_model.bin", "adapter_model.pt",
                  "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "chat_template.jinja",
                  "training_args.bin"):
        fp = os.path.join(lora_dir, fname)
        if os.path.exists(fp):
            os.remove(fp)
            removed.append(fname)
    if removed:
        print(f"[fresh] Removed {len(removed)} item(s) from {lora_dir}")
    else:
        print(f"[fresh] Nothing to remove in {lora_dir}  -  already clean.")


# ── Live progress callback ────────────────────────────────────────────────────

def _make_progress_callback(total_steps, epochs, dataset_size, batch, grad_accum):
    """
    Return a TransformersTrainerCallback that prints a compact progress line
    every logging_steps steps.  Works with both Trainer and SFTTrainer.
    """
    from transformers import TrainerCallback
    import time as _time

    class LiveProgressCallback(TrainerCallback):
        def __init__(self):
            self._start      = None
            self._last_step  = 0
            self._last_time  = None
            self._losses     = []

        def on_train_begin(self, args, state, control, **kwargs):
            self._start     = _time.time()
            self._last_time = self._start
            steps = state.max_steps or total_steps
            print(f"\n[progress] Training started  -  {steps} total steps "
                  f"({epochs} epoch(s), {dataset_size} examples, "
                  f"batch={batch}, grad_accum={grad_accum})")
            # Column widths: Step=6, Epoch=5, Loss=7, Bar=[20]+%=27, Elapsed=8, ETA=8, Speed=10
            print(f"[progress] {'Step':>6}  {'Epoch':>5}  {'Loss':>7}  "
                  f"{'Progress':<27}  {'Elapsed':>8}  {'ETA':>8}  {'Speed':>10}")
            print(f"[progress] {'─'*6}  {'─'*5}  {'─'*7}  "
                  f"{'─'*27}  {'─'*8}  {'─'*8}  {'─'*10}", flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            loss = logs.get("loss") or logs.get("train_loss")
            if loss is None:
                return
            self._losses.append(loss)

            now       = _time.time()
            elapsed   = now - self._start
            step      = state.global_step
            max_steps = state.max_steps or total_steps
            epoch     = state.epoch or 0.0

            # steps/sec from last interval
            dt  = now - self._last_time if self._last_time else elapsed
            ds  = step - self._last_step if step > self._last_step else 1
            sps = ds / dt if dt > 0 else 0.0
            self._last_step = step
            self._last_time = now

            pct     = step / max_steps if max_steps > 0 else 0.0
            bar_len = 20
            filled  = int(bar_len * pct)
            bar     = "█" * filled + "░" * (bar_len - filled)
            eta_sec = int((max_steps - step) / sps) if sps > 0 else 0

            def _fmt_time(secs):
                h, rem = divmod(int(secs), 3600)
                m, s   = divmod(rem, 60)
                return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

            speed_str = f"{sps:.2f} s/s" if sps < 1 else f"{1/sps:.2f} it/s"

            line = (
                f"\r[progress] {step:>6}  {epoch:>5.2f}  {loss:>7.4f}  "
                f"[{bar}]{pct:>5.1%}  "
                f"{_fmt_time(elapsed):>8}  {_fmt_time(eta_sec):>8}  {speed_str:>10}"
            )
            # Use \n terminator so each step arrives as its own SSE chunk immediately.
            # The leading \r causes server.py to classify it as a progress-bar update.
            print(line.ljust(120), flush=True)

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch  = round(state.epoch or 0)
            recent = self._losses[-10:] if self._losses else []
            avg    = sum(recent) / len(recent) if recent else 0.0
            best   = min(self._losses) if self._losses else 0.0
            # Move to new line before printing epoch summary
            print(f"\n[progress] ── Epoch {epoch} complete  "
                  f"avg_loss={avg:.4f}  best_loss={best:.4f}", flush=True)

        def on_train_end(self, args, state, control, **kwargs):
            elapsed = _time.time() - self._start
            h, rem  = divmod(int(elapsed), 3600)
            m, s    = divmod(rem, 60)
            best    = min(self._losses) if self._losses else 0.0
            print(f"\n[progress] Training finished in {h}h {m:02d}m {s:02d}s  "
                  f"best_loss={best:.4f}", flush=True)

    return LiveProgressCallback()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_info():
    """
    Return a dict with GPU capabilities so all decisions are made in one place.
    Keys:
      has_cuda     – bool
      bf16         – bool  (SM >= 8.0, i.e. Ampere / Ada / Hopper / Blackwell)
      is_blackwell – bool (SM >= 12.0, RTX 50xx series)
      vram_gb      – float (0.0 if no GPU)
      name         – str
      sm_major     – int
      sm_minor     – int
    """
    info = dict(has_cuda=False, bf16=False, is_blackwell=False,
                vram_gb=0.0, name="CPU", sm_major=0, sm_minor=0)
    try:
        import torch
        if not torch.cuda.is_available():
            return info
        info["has_cuda"] = True
        props = torch.cuda.get_device_properties(0)
        info["name"]     = props.name
        info["vram_gb"]  = props.total_memory / (1024 ** 3)
        info["sm_major"] = props.major
        info["sm_minor"] = props.minor
        info["bf16"]     = props.major >= 8          # Ampere+
        info["is_blackwell"] = props.major >= 12     # RTX 50xx / SM 12.x
    except Exception:
        pass
    return info


def _bf16_supported():
    return _gpu_info()["bf16"]


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_lora_target_modules(base_model_name: str) -> list:
    """
    Return the LoRA target module names appropriate for the base model family.
    Different architectures use different naming conventions.

    Catalog coverage (all open, no gating):
      Qwen2.5 (0.5B–72B), SmolLM2-1.7B  → default (q/k/v/o + gate/up/down)
      Microsoft Phi-2                     → phi-2 branch (dense + fc1/fc2)
      Microsoft Phi-3 Mini                → phi-3 branch (qkv_proj + gate/up/down)
      Mistral-7B-v0.1                     → mistral branch (q/k/v/o + gate/up/down)
      Falcon-7B                           → falcon branch (query_key_value + dense + mlp)
    """
    bm = base_model_name.lower()
    if "phi-2" in bm or "phi2" in bm:
        # Phi-2 uses fc1/fc2 for MLP and MHA with q/k/v/dense names
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    if "pythia" in bm or "gpt-neox" in bm or "neox" in bm:
        # Pythia / GPT-NeoX uses query_key_value (fused QKV) + dense + mlp dense layers
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    if "mistral" in bm or "mixtral" in bm:
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    # Default: Qwen2.5, SmolLM2, and most modern decoder-only transformers
    return ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train fine-tuned LLM model")
    p.add_argument("--fmt",         default="chatml", choices=["chatml", "alpaca"],
                   help="Dataset format (default: chatml)")
    p.add_argument("--epochs",      type=int,   default=3,
                   help="Number of training epochs (default: 3)")
    p.add_argument("--batch",       type=int,   default=2,
                   help="Per-device batch size (default: 2)")
    p.add_argument("--grad_accum",  type=int,   default=4,
                   help="Gradient accumulation steps (default: 4)")
    p.add_argument("--lr",          type=float, default=2e-4,
                   help="Learning rate (default: 2e-4)")
    p.add_argument("--max_seq_len", type=int,   default=2048,
                   help="Max sequence length (default: 2048)")
    p.add_argument("--lora-r",      dest="lora_r",     type=int,   default=16,
                   help="LoRA rank (default: 16)")
    p.add_argument("--lora-alpha",  dest="lora_alpha",  type=int,   default=16,
                   help="LoRA alpha (default: 16)")
    p.add_argument("--no-4bit",     dest="use_4bit", action="store_false",
                   help="Disable 4-bit quantisation (needs ~14 GB free VRAM)")
    p.add_argument("--grad-ckpt",   dest="grad_ckpt", action="store_true",
                   help="Force gradient checkpointing ON (saves ~3 GB VRAM, ~40%% slower). "
                        "Default: auto (ON if VRAM tight, OFF if VRAM sufficient).")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                   help="Force gradient checkpointing OFF (faster, uses ~3 GB more VRAM).")
    p.add_argument("--resume",      default=None,
                   help="Resume from checkpoint path (e.g. popai_lora/checkpoint-NNN)")
    p.add_argument("--fresh",       action="store_true",
                   help="Delete existing LoRA checkpoints and train from scratch")
    p.add_argument("--low-mem",     dest="low_mem", action="store_true",
                   help="Aggressive RAM saving: forces grad-ckpt ON, batch=1, "
                        "shorter seq-len (1024), and CPU-offloads the optimizer. "
                        "Slowest but uses the least RAM — use if you are crashing "
                        "due to out-of-memory errors.")
    p.add_argument("--eval-split",  dest="eval_split", action="store_true",
                   help="Hold out 5%% of the dataset for validation (overfitting check). "
                        "Default: OFF — train on 100%% of the data.")
    # GPU presets (optimised defaults):
    #   RTX 5070 Ti (15.9 GB): 4-bit, batch=2, grad_accum=4, NO grad-ckpt → ~12 GB, fastest
    #   RTX 4090 (24 GB):      --no-4bit --batch 4 --grad_accum 2
    #   RTX 3090 (24 GB):      --no-4bit --batch 2 --grad_accum 4
    #   RTX 4080 (16 GB):      4-bit, batch=2 (default)
    # grad_ckpt defaults to None so the auto-detect logic below can decide.
    # Pass --grad-ckpt or --no-grad-ckpt to override.
    p.set_defaults(use_4bit=True, grad_ckpt=None, fresh=False, low_mem=False)
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_dataset_path(fmt, dataset_dir):
    """Return path to the JSONL dataset file for the given format.

    dataset_dir must be the per-model dataset directory (models/{ModelName}/dataset/).
    """
    suffix = "alpaca" if fmt == "alpaca" else "chatml"
    path = os.path.join(dataset_dir, f"train_{suffix}.jsonl")
    if not os.path.exists(path):
        print(f"[train] Dataset not found: {path}")
        print(f"[train] Run 'Build Dataset' in the Training Studio first.")
        sys.exit(1)
    return path


def make_text_dataset(raw_dataset, fmt, tokenizer):
    """Convert each record to a single formatted text string for SFTTrainer.

    keep_in_memory=False + load_from_cache_file=False ensures the Arrow-backed
    dataset is memory-mapped from disk rather than copied into RAM, which is the
    single biggest source of RAM growth on large datasets.

    On Windows, the HuggingFace datasets library tries to rename a temp Arrow
    file over the existing cached file during .map().  Windows refuses this when
    the cache file is already memory-mapped by another process (e.g. a previous
    training run that was paused/refreshed), raising WinError 1224.  Passing a
    unique new_fingerprint forces datasets to write to a fresh cache path each
    run, completely avoiding the rename-over-locked-file conflict.
    """
    import time as _time
    _run_id = str(int(_time.time()))   # unique per training invocation

    if fmt == "chatml":
        def _apply(examples):
            texts = []
            for msgs in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)
                texts.append(text)
            return {"text": texts}
        return raw_dataset.map(
            _apply, batched=True,
            remove_columns=raw_dataset.column_names,
            keep_in_memory=False,       # stream from disk — don't load full dataset into RAM
            load_from_cache_file=False, # always re-map so stale cache doesn't bloat memory
            new_fingerprint=f"chatml_{_run_id}",  # unique path avoids WinError 1224 rename conflict
            desc="Applying ChatML template",
        )
    else:  # alpaca
        def _apply(examples):
            texts = []
            systems = examples.get("system", [""] * len(examples["instruction"]))
            for sys_p, instr, out in zip(systems,
                                         examples["instruction"],
                                         examples["output"]):
                text = (
                    f"<|im_start|>system\n{sys_p}<|im_end|>\n"
                    f"<|im_start|>user\n{instr}<|im_end|>\n"
                    f"<|im_start|>assistant\n{out}<|im_end|>"
                )
                texts.append(text)
            return {"text": texts}
        return raw_dataset.map(
            _apply, batched=True,
            remove_columns=raw_dataset.column_names,
            keep_in_memory=False,
            load_from_cache_file=False,
            new_fingerprint=f"alpaca_{_run_id}",  # unique path avoids WinError 1224 rename conflict
            desc="Applying Alpaca template",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Power management: prevent sleep/hibernate for the whole run ───────────
    _prevent_sleep()

    # ── Fresh / resume logic ──────────────────────────────────────────────────
    # The checkpoint scan is deferred until after LORA_OUT_MODEL is known (below).
    # --fresh is handled here but the actual wipe happens after LORA_BASE is set.
    # We set a flag so the wipe can run once the per-model path is computed.
    _do_fresh_wipe = args.fresh
    if args.fresh:
        print(f"[fresh] --fresh flag set  -  will clear LoRA checkpoints after paths are resolved...")
        args.resume = None

    # ── Package checks ────────────────────────────────────────────────────────
    missing = []
    for pkg in ("transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[train] Missing packages: {', '.join(missing)}")
        print("[train] Run:  pip install transformers peft trl accelerate bitsandbytes datasets")
        sys.exit(1)

    # Must be set before torch is imported — affects CUDA memory allocator behaviour.
    # expandable_segments avoids fragmentation OOM when large contiguous blocks are needed,
    # but is NOT supported on Windows — omitting it there to avoid a UserWarning.
    # max_split_size_mb=512 prevents the allocator from holding onto huge cached blocks,
    # which reduces peak RAM/VRAM usage at the cost of slightly more allocator overhead.
    if sys.platform == "win32":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    else:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

    import warnings
    # Suppress the "expandable_segments not supported on this platform" UserWarning.
    # This fires on Windows whenever PYTORCH_CUDA_ALLOC_CONF contains
    # expandable_segments:True (e.g. set by a system env var outside our control).
    warnings.filterwarnings(
        "ignore",
        message=".*expandable_segments.*not supported.*",
        category=UserWarning,
    )

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging as hf_logging
    from trl import SFTTrainer, SFTConfig

    # Suppress HuggingFace info/warning messages globally
    hf_logging.set_verbosity_error()

    # Also silence the trainer's internal Python logger (prints {'loss':...} lines)
    import logging
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("trl").setLevel(logging.ERROR)
    logging.getLogger("trl.trainer").setLevel(logging.ERROR)
    logging.getLogger("trl.trainer.sft_trainer").setLevel(logging.ERROR)
    # NOTE: Do NOT call logging.disable() globally — it breaks BNB's quantization
    # internal signalling and can leave Linear4bit weights as plain Parameters.

    cuda  = _has_cuda()
    bf16  = _bf16_supported()
    use_4bit = args.use_4bit and cuda

    # ── Load config early — needed for model name in auto-detect ─────────────
    _cfg = _load_config()
    BASE_MODEL = _cfg.get("base_model", "Qwen/Qwen2.5-7B")
    MODEL_NAME = _cfg.get("name", "Model")

    # ── Model-specific LoRA output directory ──────────────────────────────────
    # Directory name encodes BOTH the Studio model name AND the HuggingFace base
    # model (size + precision), e.g.:
    #
    #   popai_lora/PopAI_7b_4bit/checkpoint-500/
    #   popai_lora/PopAI_14b_4bit/checkpoint-200/
    #
    # This means switching the base model in config.json automatically creates a
    # fresh directory — stale checkpoints from the old model are never found.
    # Changing only the Studio name also creates a fresh directory.
    # No metadata files or compatibility checks are needed.
    import re as _re

    # Sanitise MODEL_NAME (strip path-illegal chars)
    _name_part = _re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', MODEL_NAME).strip(". ")
    if not _name_part:
        _name_part = "model"

    # Extract size + family from BASE_MODEL string (e.g. "EleutherAI/pythia-1.4b" → "1.4b", "pythia")
    # This matches the GGUF filename convention so LoRA folder and GGUF name are consistent.
    # ── Extract model family name ─────────────────────────────────────────────
    # Same logic as generate_llm.py's _model_family():
    #   "EleutherAI/pythia-1.4b" → "pythia"
    #   "Qwen/Qwen2.5-7B"        → "qwen2.5"
    #   "microsoft/phi-2"        → "phi-2"
    def _model_family(base_model):
        parts = base_model.split("/")
        model_part = parts[-1] if len(parts) > 1 else parts[0]
        name = _re.sub(r'[-_]?\d+\.?\d*[Bb][\w.]*.*$', '', model_part)
        name = _re.sub(r'[-_]?(instruct|chat|hf|base|it|v\d[\w.]*)$', '', name, flags=_re.IGNORECASE)
        name = name.strip('-_')
        # Preserve dots (qwen2.5), replace underscores with dashes
        name = name.lower().replace('_', '-')
        name = _re.sub(r'-+', '-', name).strip('-')
        return name

    # ── Extract size from model name, or compute from HF config ──────────────
    # Same logic as generate_llm.py:
    #   Priority 1: parse from model name string ("pythia-1.4b" → "1.4b")
    #   Priority 2: compute from model config.json (accurate param count)
    def _size_from_hf_config(model_id):
        """Compute size tag from HuggingFace model config (cached download)."""
        try:
            from huggingface_hub import hf_hub_download
            import json as _json
            cfg_path = hf_hub_download(model_id, "config.json")
            with open(cfg_path, encoding="utf-8") as _f:
                cfg = _json.load(_f)
        except Exception:
            return ""
        vocab    = cfg.get("vocab_size", 0)
        hidden   = cfg.get("hidden_size", 0)
        layers   = cfg.get("num_hidden_layers", 0)
        inter    = cfg.get("intermediate_size", 0)
        heads    = cfg.get("num_attention_heads", 0)
        kv_heads = cfg.get("num_key_value_heads", heads)
        head_dim = cfg.get("head_dim", hidden // heads if heads else 0)
        if not (hidden and layers):
            return ""
        q_size  = hidden * (heads * head_dim) if heads and head_dim else hidden * hidden
        kv_size = hidden * (kv_heads * head_dim) if kv_heads and head_dim else hidden * hidden
        attn    = q_size + 2 * kv_size + q_size
        mlp     = 3 * hidden * inter if inter else 2 * hidden * hidden
        norms   = 2 * hidden
        total   = (attn + mlp + norms) * layers + vocab * hidden
        if total >= 1e9:
            b = total / 1e9
            rounded = round(b, 1)
            tag = f"{int(rounded)}b" if rounded == int(rounded) else f"{rounded}b"
        elif total >= 1e6:
            tag = f"{round(total/1e6)}m"
        else:
            return ""
        return tag

    _size_match = _re.search(r'[-_/](\d+(?:\.\d+)?[Bb])', BASE_MODEL)
    if _size_match:
        _size_tag = _size_match.group(1).lower()
    else:
        # No size in model name — compute from HuggingFace config
        _size_tag = _size_from_hf_config(BASE_MODEL)
        if _size_tag:
            print(f"[train] Model size (from config): {_size_tag.upper()}")

    _family_name = _model_family(BASE_MODEL)
    # Combined size-family tag: "1.4b-pythia", "7b-qwen2.5", "2.7b-phi-2"
    _size_family_tag = f"{_size_tag}-{_family_name}" if (_size_tag and _family_name) else (_size_tag or _family_name or "base")

    # Directory name encodes the model name + base model identity:
    #
    #   {modelname}-{size}-{family}
    #
    MODEL_DIR      = os.path.join(MODELS_ROOT, _safe_name(_name_part))
    DATASET_DIR    = os.path.join(MODEL_DIR, "dataset")
    LORA_BASE      = os.path.join(MODEL_DIR, "lora")

    _dir_name = f"{_name_part.lower()}-{_size_family_tag}"
    LORA_OUT_MODEL = os.path.join(LORA_BASE, _dir_name)

    # ── Execute deferred --fresh wipe now that LORA_OUT_MODEL is known ───────
    if _do_fresh_wipe:
        _wipe_lora_checkpoints(LORA_OUT_MODEL)

    # ── One-time cleanup: remove legacy LoRA files from old train/popai_lora/ ─
    # Before the consolidated models/ layout, checkpoints lived in
    # train/popai_lora/{dir}/.  Silently remove stale legacy roots.
    _LEGACY_LORA_ROOT = os.path.join(ROOT, "train", "popai_lora")
    _LEGACY_ADAPTER_FILES = {
        "adapter_config.json", "adapter_model.safetensors",
        "adapter_model.bin", "adapter_model.pt",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "chat_template.jinja",
        "training_args.bin", "README.md", ".training_info.json",
    }
    if not args.fresh and os.path.isdir(_LEGACY_LORA_ROOT):
        _legacy_removed = []
        for _n in os.listdir(_LEGACY_LORA_ROOT):
            _full = os.path.join(_LEGACY_LORA_ROOT, _n)
            if _n.startswith("checkpoint-") and os.path.isdir(_full):
                shutil.rmtree(_full, ignore_errors=True)
                _legacy_removed.append(_n)
            elif os.path.isfile(_full) and _n in _LEGACY_ADAPTER_FILES:
                try:
                    os.remove(_full)
                    _legacy_removed.append(_n)
                except Exception:
                    pass
        if _legacy_removed:
            print(f"[train] Cleaned up {len(_legacy_removed)} legacy file(s) from {_LEGACY_LORA_ROOT} "
                  f"(moved to models/{_name_part}/lora/)")

    # ── Deferred checkpoint scan (now that LORA_OUT_MODEL is known) ───────────
    # Each named model gets its own subdirectory (popai_lora/{ModelName}_{size}/),
    # so checkpoints from a different model are invisible here by design.
    # As an additional safety net, we also check the base_model recorded in the
    # checkpoint's adapter_config.json — this catches the edge case where the user
    # keeps the same Studio model name but changes the HuggingFace base model ID.
    if not args.fresh and args.resume is None:
        existing_checkpoints = []
        if os.path.isdir(LORA_OUT_MODEL):
            for _name in os.listdir(LORA_OUT_MODEL):
                if _name.startswith("checkpoint-") and os.path.isdir(os.path.join(LORA_OUT_MODEL, _name)):
                    try:
                        _step = int(_name.split("-")[1])
                        existing_checkpoints.append((_step, _name))
                    except ValueError:
                        pass
        if existing_checkpoints:
            existing_checkpoints.sort(reverse=True)
            latest_step, latest_name = existing_checkpoints[0]
            latest_path = os.path.join(LORA_OUT_MODEL, latest_name)

            # Safety net: verify the checkpoint's base model matches current BASE_MODEL.
            # Handles the edge case: same Studio model name, different HuggingFace model.
            _ckpt_base = _checkpoint_base_model(latest_path)
            if _ckpt_base is not None and _ckpt_base != BASE_MODEL:
                print(
                    f"\n[train] ⚠️  BASE MODEL CHANGED — clearing incompatible checkpoint.\n"
                    f"[train]   Checkpoint was trained with : {_ckpt_base}\n"
                    f"[train]   Current base model is       : {BASE_MODEL}\n"
                    f"[train] Starting fresh..."
                )
                _wipe_lora_checkpoints(LORA_OUT_MODEL)
                # No checkpoint to resume from — fall through with args.resume = None
            else:
                print(f"[train] Found existing checkpoint: {latest_name} (step {latest_step})")
                try:
                    import ctypes as _ctypes
                    _STD_INPUT_HANDLE = _ctypes.c_ulong(-10 & 0xFFFFFFFF)
                    _handle = _ctypes.windll.kernel32.GetStdHandle(_STD_INPUT_HANDLE)
                    _mode   = _ctypes.c_ulong(0)
                    is_console = bool(_ctypes.windll.kernel32.GetConsoleMode(_handle, _ctypes.byref(_mode)))
                except Exception:
                    is_console = False
                if is_console:
                    print(f"[train] Options:")
                    print(f"[train]   [R] Resume from checkpoint (default)")
                    print(f"[train]   [F] Start fresh (deletes checkpoint)")
                    try:
                        choice = input("[train] Resume or Fresh? (R/f): ").strip().lower()
                    except EOFError:
                        choice = ""
                    if choice in ("f", "fresh"):
                        print(f"[train] Starting fresh  -  clearing checkpoints...")
                        _wipe_lora_checkpoints(LORA_OUT_MODEL)
                        args.resume = None
                    else:
                        print(f"[train] Resuming from {latest_name}")
                        args.resume = latest_path
                else:
                    print(f"[train] Auto-resuming from checkpoint-{latest_step}")
                    args.resume = latest_path

    # ── Auto-detect VRAM and scale settings ──────────────────────────────────
    vram_gb = 0
    if cuda:
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
        except Exception:
            vram_gb = 0

    # Estimate model parameter count from base model name (rough heuristic)
    _bm_lower = BASE_MODEL.lower()
    if any(x in _bm_lower for x in ["72b", "70b"]):
        model_params_b = 70
    elif any(x in _bm_lower for x in ["32b", "30b"]):
        model_params_b = 32
    elif any(x in _bm_lower for x in ["14b"]):
        model_params_b = 14
    elif any(x in _bm_lower for x in ["13b", "12b"]):
        model_params_b = 13
    elif any(x in _bm_lower for x in ["8b", "9b"]):
        model_params_b = 8
    elif any(x in _bm_lower for x in ["7b"]):
        model_params_b = 7
    elif any(x in _bm_lower for x in ["3b"]):
        model_params_b = 3
    elif any(x in _bm_lower for x in ["1.5b", "1b"]):
        model_params_b = 1
    elif any(x in _bm_lower for x in ["0.5b"]):
        model_params_b = 0.5
    else:
        model_params_b = 7  # default assumption

    # Rough VRAM needed for 4-bit weights (bytes_per_param ≈ 0.5 for nf4 + overhead)
    # Rule of thumb: model_gb_4bit ≈ model_params_b * 0.6
    model_vram_4bit = model_params_b * 0.6
    model_vram_fp16 = model_params_b * 2.0

    # Auto-scale batch size and gradient accumulation based on available VRAM
    # after subtracting model weight VRAM. Effective batch = batch * grad_accum = 8 target.
    # Only override if user didn't pass explicit --batch / --grad_accum
    _user_set_batch = "--batch" in sys.argv
    _user_set_accum = "--grad_accum" in sys.argv

    if not _user_set_batch:
        if vram_gb >= 40:
            args.batch = 4
        elif vram_gb >= 24:
            args.batch = 2
        elif vram_gb >= 16:
            args.batch = 2
        elif vram_gb >= 10:
            args.batch = 1
        else:
            args.batch = 1

    if not _user_set_accum:
        # Keep effective batch ~8
        args.grad_accum = max(1, 8 // args.batch)

    # Auto-scale seq_len down for tight VRAM situations (only when user hasn't set it explicitly).
    # Sequence length is the single biggest VRAM lever: VRAM ∝ seq_len² for attention.
    # On 8 GB with a 7B model, 2048 tokens will OOM; 1024 is safe.
    # On 6 GB or less (sub-4B models), 512 is safer still.
    _user_set_seq = "--max_seq_len" in sys.argv
    if not _user_set_seq and cuda and vram_gb > 0:
        headroom = vram_gb - model_vram_4bit if use_4bit else vram_gb - model_vram_fp16
        if headroom < 4:
            # Very tight: 6 GB card running 3B, or 8 GB running 7B
            if args.max_seq_len > 512:
                args.max_seq_len = 512
                print(f"[train] Auto: seq_len capped to 512 (only {headroom:.1f} GB free after model weights)")
        elif headroom < 8:
            # Tight: 8–12 GB card running 7B, etc.
            if args.max_seq_len > 1024:
                args.max_seq_len = 1024
                print(f"[train] Auto: seq_len capped to 1024 ({headroom:.1f} GB free after model weights)")
        # >= 8 GB headroom: keep user default (2048)

    # For large models (14B+) on large VRAM cards, device_map="auto" allows
    # spreading across multiple GPUs or using CPU offload for layers that don't fit.
    # For small/medium models on single GPU, {"": 0} is safer with BNB 4-bit.
    # Rule: if model fits in single GPU (leaving ≥4 GB headroom), use {"": 0}.
    # Otherwise use "auto" to allow multi-GPU or CPU offload.
    headroom_gb = 4.0
    if use_4bit:
        fits_single_gpu = (model_vram_4bit + headroom_gb) <= vram_gb
    else:
        fits_single_gpu = (model_vram_fp16 + headroom_gb) <= vram_gb

    if use_4bit:
        device_map_4bit = {"": 0} if fits_single_gpu else "auto"
    # Non-4bit path uses "auto" inline (no BNB Params4bit constraint)

    # ── --low-mem: override everything for minimum RAM usage ─────────────────
    # Applies: gradient checkpointing ON, batch=1, grad_accum=8, seq_len=1024,
    # and switches the optimizer to paged_adamw_32bit which still uses paging
    # (CPU offload for optimizer states) but is available without bitsandbytes.
    if args.low_mem:
        args.grad_ckpt   = True
        if not _user_set_batch:
            args.batch   = 1
        if not _user_set_accum:
            args.grad_accum = 8   # keep effective batch = 8
        if args.max_seq_len > 1024:
            args.max_seq_len = 1024
        print("[train] --low-mem mode: grad-ckpt=ON, batch=1, grad_accum=8, max_seq_len=1024")
        # Also cap the CUDA allocator split size more aggressively.
        # expandable_segments is not supported on Windows — omit it there.
        if sys.platform == "win32":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        # Tell HuggingFace datasets to never cache more than 200 MB in RAM
        os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", str(200 * 1024 * 1024))

    # Gradient checkpointing decision:
    #   args.grad_ckpt=True  → user forced ON via --grad-ckpt
    #   args.grad_ckpt=False → user forced OFF via --no-grad-ckpt
    #   args.grad_ckpt=None  → auto: ON only if VRAM is tight (< 8 GB headroom after model)
    # --low-mem always forces ON regardless.
    if args.low_mem:
        use_grad_ckpt = True
        grad_ckpt_reason = "(--low-mem mode)"
    elif args.grad_ckpt is True:
        use_grad_ckpt = True
        grad_ckpt_reason = "(user requested --grad-ckpt)"
    elif args.grad_ckpt is False:
        use_grad_ckpt = False
        grad_ckpt_reason = "(user requested --no-grad-ckpt)"
    elif vram_gb > 0 and (vram_gb - model_vram_4bit) < 8:
        # Auto: tight VRAM — turn ON to save ~3 GB activations
        use_grad_ckpt = True
        grad_ckpt_reason = f"(auto: tight VRAM — {vram_gb:.0f} GB total, ~{model_vram_4bit:.0f} GB model)"
    else:
        # Auto: plenty of VRAM — leave OFF for ~40% faster training
        use_grad_ckpt = False
        grad_ckpt_reason = f"(auto: sufficient VRAM — {vram_gb:.0f} GB total, ~{model_vram_4bit:.0f} GB model)"

    # ── Gather all GPU info now that torch is imported ────────────────────────
    _gi = _gpu_info()
    is_blackwell = _gi["is_blackwell"]

    print("=" * 60)
    print("  LLM Training Studio - Training (NeHe Productions)")
    print("=" * 60)
    _gpu_label = _gi["name"] if cuda else "CPU (slow!)"
    _sm_label  = f" (SM {_gi['sm_major']}.{_gi['sm_minor']})" if cuda else ""
    print(f"[train] GPU            : {_gpu_label}{_sm_label}")
    print(f"[train] VRAM           : {vram_gb:.1f} GB")
    print(f"[train] CUDA available : {cuda}")
    print(f"[train] BF16 supported : {bf16}  {'(disabled for 4-bit on Blackwell)' if bf16 and is_blackwell and use_4bit else ''}")
    print(f"[train] 4-bit loading  : {use_4bit}")
    print(f"[train] Epochs         : {args.epochs}")
    print(f"[train] Batch size     : {args.batch}")
    print(f"[train] Grad accum     : {args.grad_accum}  (effective batch = {args.batch * args.grad_accum})")
    print(f"[train] Grad ckpt      : {use_grad_ckpt}  {grad_ckpt_reason}")
    print(f"[train] Format         : {args.fmt}")

    dataset_path = get_dataset_path(args.fmt, dataset_dir=DATASET_DIR)
    print(f"[train] Dataset        : {dataset_path}")
    print(f"[train] Model name     : {MODEL_NAME}")
    print(f"[train] Base model     : {BASE_MODEL}")
    print(f"[train] LoRA output    : {LORA_OUT_MODEL}")
    print()

    if not cuda:
        print("[train] WARNING: No CUDA GPU detected. Training on CPU will be extremely slow.")
        print("[train] Consider using a machine with a CUDA-capable GPU.")
        inp = input("[train] Continue anyway? (y/N): ").strip().lower()
        if inp != "y":
            sys.exit(0)

    # ── 1. Validate model exists on HuggingFace ───────────────────────────────
    print(f"[train] Checking model exists: {BASE_MODEL}")
    try:
        from huggingface_hub import model_info
        model_info(BASE_MODEL)
        print(f"[train] Model found ✓")
    except Exception:
        print(f"\n[train] ERROR: Model '{BASE_MODEL}' was not found on HuggingFace.")
        print(f"[train] The model ID you specified in config.json does not exist.")
        print(f"[train] Check the exact model ID at: https://huggingface.co/models")
        print(f"[train] Then update 'base_model' in config.json and try again.")
        _restore_sleep()
        sys.exit(1)

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    print("[train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Ensure ChatML tokens are recognized as single special tokens.
    # Only add tokens that are NOT already in the vocabulary — adding tokens that
    # already exist causes the embedding table to be flagged as "resized" by PEFT,
    # which forces it to save the entire embedding matrix on every checkpoint save
    # instead of just the LoRA adapters (bloats checkpoints and triggers a warning).
    _existing_vocab = set(tokenizer.get_vocab().keys())
    special_tokens = [t for t in ["<|im_start|>", "<|im_end|>"] if t not in _existing_vocab]
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    # Note: We resize embeddings later after loading the model (only if vocab actually grew)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Base models (non-instruct) don't have a chat template set.
    # Inject a suitable chat template so apply_chat_template() works.
    if not getattr(tokenizer, "chat_template", None):
        _bm = BASE_MODEL.lower()
        if "phi-2" in _bm or "phi2" in _bm:
            # Phi-2 uses a simple Instruct/Output format with no special tokens
            print("[train] No chat template found  -  injecting Phi-2 Instruct/Output template.")
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ message['content'] + '\n\n' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ 'Instruct: ' + message['content'] + '\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ 'Output: ' + message['content'] + '\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ 'Output: ' }}"
                "{% endif %}"
            )
        else:
            # Default: standard ChatML format (Qwen, Mistral, LLaMA-3, etc.)
            print("[train] No chat template found  -  injecting standard ChatML template for base model.")
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{'<|im_start|>assistant\n'}}"
                "{% endif %}"
            )

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("[train] Loading base model (may take a few minutes on first run)...")
    if use_4bit:
        # Compute dtype for the bitsandbytes 4-bit dequantize matmul.
        # Must match the dtype used for non-quantized model parts (norms, embeddings)
        # to avoid expensive implicit casts on every forward/backward pass.
        # Use float16 (not bfloat16) because on Blackwell GPUs (RTX 50xx / SM 12.0),
        # cublasGemmEx with CUDA_R_16BF triggers CUBLAS_STATUS_INTERNAL_ERROR.
        # FP16 cuBLAS is fully supported on all CUDA architectures.
        bnb_compute_dtype = torch.float16
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = bnb_compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config = bnb_cfg,
            # dtype must match bnb_4bit_compute_dtype so non-quantized layers
            # (layer norms, embeddings, lm_head) are already in FP16. Without this,
            # they default to FP32 and every step pays a costly FP32↔FP16 cast,
            # which starves the GPU and causes the very low utilisation (0.02 s/s).
            dtype               = bnb_compute_dtype,
            # device_map={"": 0} forces all layers onto GPU 0 so BNB quantizes every
            # Linear to Params4bit. For models too large to fit a single GPU, "auto"
            # is used instead to allow multi-GPU or CPU offload.
            device_map          = device_map_4bit,
            trust_remote_code   = True,
        )

        # Critical for 4-bit stability: prepares layer norms and heads for training
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

        torch.cuda.empty_cache()
    else:
        dtype = torch.bfloat16 if bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype             = dtype if cuda else torch.float32,
            device_map        = "auto" if cuda else None,
            trust_remote_code = True,
        )

    # Resize embeddings to match what the checkpoint/adapter was trained with.
    #
    # Three cases:
    #   A) Resuming from checkpoint: must match the checkpoint's saved embedding size
    #      exactly, or PEFT will raise a size mismatch error on load.
    #      Read embed_tokens.weight shape from the checkpoint's adapter_model.safetensors.
    #   B) Fresh training, tokenizer grew: resize UP to len(tokenizer) (new tokens added).
    #   C) Fresh training, tokenizer didn't grow: no resize needed.
    #      (Qwen2.5 base has 152064 rows; Qwen2Tokenizer reports 151643 — don't shrink.)
    _base_vocab_size = model.get_input_embeddings().weight.shape[0]
    _tok_vocab_size  = len(tokenizer)
    _target_vocab    = None

    # Case A: resuming — read the saved embedding size from the checkpoint
    _resume_for_resize = args.resume  # may have been set by checkpoint scan above
    if _resume_for_resize and os.path.isdir(_resume_for_resize):
        try:
            import safetensors.torch as _st_resize
            _ckpt_st = os.path.join(_resume_for_resize, "adapter_model.safetensors")
            if os.path.isfile(_ckpt_st):
                _ckpt_tensors = _st_resize.load_file(_ckpt_st)
                for _k in _ckpt_tensors:
                    if "embed_tokens.weight" in _k or "lm_head.weight" in _k:
                        _target_vocab = _ckpt_tensors[_k].shape[0]
                        break
        except Exception:
            pass

    if _target_vocab is not None:
        # Case A: match checkpoint embedding size exactly (may be smaller than base model)
        if _target_vocab != _base_vocab_size:
            print(f"[train] Resizing token embeddings: {_base_vocab_size} → {_target_vocab} "
                  f"(matching checkpoint embedding shape for resume)")
            model.resize_token_embeddings(_target_vocab)
        else:
            print(f"[train] Token embedding size matches checkpoint ({_target_vocab}) ✓")
    elif _tok_vocab_size > _base_vocab_size:
        # Case B: fresh training, tokenizer grew (new special tokens added)
        print(f"[train] Resizing token embeddings: {_base_vocab_size} → {_tok_vocab_size} "
              f"(new special tokens added)")
        model.resize_token_embeddings(_tok_vocab_size)
    else:
        # Case C: fresh training, no resize needed (special tokens already in base vocab)
        print(f"[train] Token embedding size OK ({_base_vocab_size})  -  no resize needed "
              f"(tokenizer={_tok_vocab_size}; special tokens already in base vocab)")

    model.config.use_cache = False

    # ── 4. LoRA ───────────────────────────────────────────────────────────────
    #
    # Small models (≤1.5B) suffer from two forms of "bleeding":
    #   1. Language bleeding — base-model pre-training (e.g. Chinese in Qwen2.5)
    #      leaks into responses because a small model lacks the parameter budget
    #      to cleanly override it with a low-rank adapter.
    #   2. Repetition — the model repeats phrases because it hasn't learned a
    #      strong enough EOS/stop signal.
    #
    # Auto-scaling strategy for small models:
    #   • Higher LoRA rank (r=32 vs 16) — more expressive adapter, stronger
    #     suppression of base-model habits.  VRAM cost is minimal for tiny models.
    #   • Higher lora_alpha (64 vs 16) — alpha/r ratio of 2.0 gives a larger
    #     effective learning rate for the adapter weights, helping fine-tuning
    #     punch through deep-seated pre-training patterns.
    #   • Higher dropout (0.1 vs 0.05) — prevents the small adapter from simply
    #     memorising the base model's output distribution; forces better
    #     generalisation to the new persona.
    #
    # These values are only auto-applied when the user hasn't explicitly passed
    # --lora-r or --lora-alpha on the command line.
    _user_set_lora_r     = "--lora-r"     in sys.argv
    _user_set_lora_alpha = "--lora-alpha" in sys.argv

    _lora_r       = args.lora_r
    _lora_alpha   = args.lora_alpha
    _lora_dropout = 0.05

    if not _user_set_lora_r and not _user_set_lora_alpha and model_params_b <= 1.5:
        # Small model: boost rank, alpha, and dropout automatically
        _lora_r       = max(args.lora_r, 32)    # at least r=32
        _lora_alpha   = max(args.lora_alpha, _lora_r * 2)  # alpha = 2× rank
        _lora_dropout = 0.1
        print(f"[train] Small model ({model_params_b}B ≤ 1.5B) — auto-scaling LoRA:")
        print(f"[train]   r={_lora_r} (was {args.lora_r}), alpha={_lora_alpha} "
              f"(was {args.lora_alpha}), dropout={_lora_dropout} (was 0.05)")
        print(f"[train]   Higher rank+alpha helps override base-model language bleeding.")

    _lora_targets = _get_lora_target_modules(BASE_MODEL)
    print(f"[train] LoRA targets   : {_lora_targets}")
    lora_cfg = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = _lora_r,
        lora_alpha     = _lora_alpha,
        lora_dropout   = _lora_dropout,
        target_modules = _lora_targets,
        bias           = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Cast ALL non-quantized float params to FP16 (Blackwell / SM 12.x fix) ─
    #
    # Root cause of CUBLAS_STATUS_INTERNAL_ERROR with CUDA_R_16BF:
    #   cublasGemmEx on SM 12.x (Blackwell) crashes whenever either input matrix
    #   is BF16.  Two sources of stray BF16 tensors exist after model loading:
    #
    #   1. PEFT LoRA adapters — get_peft_model() creates lora_A/lora_B Parameters
    #      in bfloat16 because the CUDA device advertises SM ≥ 8.0 BF16 support,
    #      even though we passed torch_dtype=float16 to from_pretrained().
    #
    #   2. Tied / shared embedding weights — models such as Qwen2 store lm_head
    #      weights tied to the token embedding.  Some safetensors checkpoints save
    #      these in BF16 regardless of the requested torch_dtype, so after
    #      from_pretrained(..., torch_dtype=float16) a handful of embedding-related
    #      tensors remain in BF16.  During the backward pass the BF16 embedding
    #      gradient becomes the grad_output fed to the bitsandbytes backward matmul,
    #      which then calls cublasGemmEx with CUDA_R_16BF → crash.
    #
    # Fix: cast every non-quantized floating-point parameter that is in BF16 to
    # FP16.  BNB Params4bit objects are excluded (they are not nn.Parameter and
    # will not appear in named_parameters with requires_grad=True for the base
    # weights), so this cast is safe.
    if use_4bit and cuda:
        _bf16_params_found = 0
        for _name, _param in model.named_parameters():
            if _param.dtype == torch.bfloat16:
                _param.data = _param.data.to(torch.float16)
                _bf16_params_found += 1
        if _bf16_params_found:
            print(f"[train] Cast {_bf16_params_found} BF16 param(s) → FP16 "
                  f"(Blackwell cuBLAS BF16 workaround)")

    # ── 5. Dataset ────────────────────────────────────────────────────────────
    # keep_in_memory=False (default) means HuggingFace uses memory-mapped Arrow
    # files on disk instead of loading everything into RAM.  This is the single
    # biggest RAM saving — a 10 k-example dataset can occupy several GB in RAM
    # if accidentally kept in memory.
    print(f"[train] Applying {args.fmt.upper()} template to dataset...")
    raw     = load_dataset("json", data_files={"train": dataset_path},
                           split="train", keep_in_memory=False)
    dataset = make_text_dataset(raw, args.fmt, tokenizer)
    # Release the raw dataset immediately — only the formatted `dataset` is needed
    del raw
    import gc; gc.collect()
    sys.stderr.flush()
    sys.stdout.flush()
    print(f"[train] Dataset size   : {len(dataset)} examples")

    # ── 6. Training config ────────────────────────────────────────────────────
    # RTX 5070 Ti optimised:
    #   - 4-bit QLoRA: weights ~4.5 GB, activations ~7 GB → total ~12 GB (fits in 15.9 GB)
    #   - gradient_checkpointing OFF: saves 40% time, uses ~3 GB more VRAM (still fits)
    #   - bfloat16 compute: native on Blackwell SM12, fastest precision
    #   - paged_adamw_8bit: 8-bit optimizer states, saves ~1.5 GB vs fp32 adam
    #   - packing=False: avoids cross-sample contamination in fine-tuning
    # ── Validation split (5% of dataset held out for eval loss tracking) ─────
    # Axolotl, LLaMA-Factory, and Unsloth all use a validation split to detect
    # overfitting. Without it, the training loss can go to 0 while the model
    # memorises the training set. Require at least 20 examples before splitting.
    _total_size = len(dataset)
    # Validation split: only activated when --eval-split is passed (or checkbox checked in UI).
    # Default is OFF — train on 100% of the data, which is better for small curated datasets.
    if args.eval_split and _total_size >= 20:
        _val_size   = max(1, int(_total_size * 0.05))
        _train_size = _total_size - _val_size
        _split = dataset.train_test_split(test_size=_val_size, seed=42)
        train_dataset = _split["train"]
        eval_dataset  = _split["test"]
        print(f"[train] Train examples : {len(train_dataset)}  "
              f"(eval: {len(eval_dataset)} examples, 5% held out for overfitting check)")
    else:
        train_dataset = dataset
        eval_dataset  = None
        if args.eval_split and _total_size < 20:
            print(f"[train] Train examples : {len(train_dataset)}  (dataset too small for eval split — need 20+)")
        else:
            print(f"[train] Train examples : {len(train_dataset)}  (training on 100% of data)")

    steps_per_epoch = max(1, len(train_dataset) // (args.batch * args.grad_accum))
    # Warmup over 10% of total steps — matches Axolotl/LLaMA-Factory standard.
    # 10% warmup gives the optimizer more time to find a good initial direction,
    # reducing early training instability especially for small models.
    warmup = max(1, steps_per_epoch * args.epochs // 10)
    # Save a checkpoint roughly every 10% of an epoch, but clamp between 50–200 steps.
    # This ensures the first checkpoint arrives well before step 200 on typical datasets,
    # while avoiding excessive disk I/O on very large datasets.
    save_steps = max(50, min(200, steps_per_epoch // 10))

    # Enable TF32 on Ampere/Blackwell for faster matmuls with minimal precision loss.
    # TF32 uses the full FP32 exponent range but rounds the mantissa to 10 bits,
    # giving ~3× speedup on tensor cores vs strict FP32.
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Suppress PyTorch use_reentrant deprecation warning from HF/TRL internals
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*use_reentrant.*",
        category=UserWarning,
    )
    # Suppress PEFT's "save_embedding_layers set to True" warning.
    # This fires whenever the embedding table was resized (e.g. new ChatML tokens
    # added for models like Mistral that don't include them in the base vocab).
    # Setting save_embedding_layers=True is the *correct* behaviour in that case —
    # the warning is purely informational and clutters the training output.
    warnings.filterwarnings(
        "ignore",
        message=".*save_embedding_layers.*",
        category=UserWarning,
    )

    # Precision strategy for 4-bit QLoRA on Blackwell (RTX 50xx / SM 12.0):
    #
    # Problem 1: cublasGemmEx with CUDA_R_16BF crashes in the bitsandbytes backward
    #   on Blackwell — bnb_4bit_compute_dtype must be float16.
    # Problem 2: With bf16=True in SFTConfig the trainer produces BF16 grad_output,
    #   then bitsandbytes does .to(grad_output.dtype) → BF16 dequantized weight →
    #   BF16 matmul → same CUBLAS crash.
    # Problem 3: With fp16=True in SFTConfig, Accelerate's GradScaler hits the BF16
    #   LoRA parameters (which PEFT keeps in BF16) →
    #   NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda for BFloat16
    #
    # Solution: disable both fp16 and bf16 in SFTConfig (no AMP wrapper, no GradScaler).
    # The model already has torch_dtype=float16 for non-quantized layers, and the LoRA
    # adapters are cast to FP16 explicitly below. All arithmetic stays in FP16/FP32 —
    # no BF16 tensor ever reaches the bitsandbytes backward matmul.
    if use_4bit:
        # 4-bit QLoRA on Blackwell: disable all AMP wrappers.
        # fp16=True → GradScaler hits BF16 LoRA params → NotImplementedError
        # bf16=True → BF16 grad_output reaches BNB backward → CUBLAS_STATUS_INTERNAL_ERROR
        # With torch_dtype=float16 and bnb_4bit_compute_dtype=float16, all tensors are
        # FP16 or FP32 — training is stable without any AMP wrapper.
        train_fp16 = False
        train_bf16 = False
    else:
        # Full-precision path: BF16 on Blackwell (no GradScaler, safe), FP16 otherwise
        train_fp16 = cuda and not bf16
        train_bf16 = cuda and bf16

    sft_cfg = SFTConfig(
        output_dir                  = LORA_OUT_MODEL,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch,
        gradient_accumulation_steps = args.grad_accum,
        warmup_steps                = warmup,
        learning_rate               = args.lr,
        fp16                        = train_fp16,   # FP16 when 4-bit QLoRA (matches bnb compute dtype)
        bf16                        = train_bf16,   # BF16 only for full-precision runs on Blackwell
        logging_steps               = 10,
        disable_tqdm                = True,   # suppress HF built-in progress bar (we have our own)
        log_level                   = "error", # suppress HF info/warning log spam
        # Optimizer choice:
        #   --low-mem:      paged_adamw_32bit — pages optimizer states to CPU (lowest RAM)
        #   4-bit QLoRA:    adamw_torch — standard optimizer; paged_adamw_8bit has
        #                   significant per-step CPU↔GPU paging overhead on each update,
        #                   which can halve throughput on fast GPUs like the RTX 5070 Ti.
        #   full precision: adamw_torch — standard optimizer
        optim                       = ("paged_adamw_32bit" if args.low_mem
                                       else "adamw_torch"),
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        report_to                   = "none",
        save_strategy               = "steps",
        save_steps                  = save_steps,
        save_total_limit            = 2,
        # Evaluation strategy — run eval at the same cadence as checkpoints.
        # eval_strategy="steps" + eval_steps=save_steps means we get a validation
        # loss reading every time a checkpoint is saved. This lets us see if the
        # model is overfitting (train loss falling while eval loss rises) and
        # stop early if needed. Only active when eval_dataset is provided.
        eval_strategy               = "steps" if eval_dataset is not None else "no",
        eval_steps                  = save_steps if eval_dataset is not None else None,
        # load_best_model_at_end: when eval is enabled, reload the checkpoint with
        # the lowest eval loss at the end of training instead of keeping the last
        # checkpoint. This ensures the saved adapter represents the best-generalizing
        # weights rather than potentially overfit late-epoch weights.
        # Only enabled when we have an eval split — requires eval_strategy != "no".
        load_best_model_at_end      = eval_dataset is not None,
        metric_for_best_model       = "eval_loss" if eval_dataset is not None else None,
        greater_is_better           = False if eval_dataset is not None else None,
        gradient_checkpointing      = use_grad_ckpt,
        # use_reentrant=False avoids the float32 param cast that the legacy
        # reentrant checkpoint path performs, preventing OOM on 16 GB GPUs.
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        dataloader_num_workers      = 0,
        # pin_memory speeds up CPU→GPU data transfers by using pinned (page-locked)
        # host memory. Safe to enable since dataloader_num_workers=0.
        dataloader_pin_memory       = True,
        dataset_text_field          = "text",
        max_length                  = args.max_seq_len,
        packing                     = False,
        # NEFTune: adds small uniform noise to token embeddings during training.
        # Shown to improve instruction-following quality on fine-tuned models with
        # no extra compute cost. Alpha=5 is the standard recommended value.
        # Paper: https://arxiv.org/abs/2310.05914
        neftune_noise_alpha         = 5,
    )

    # ── 7. Trainer + progress callback ───────────────────────────────────────
    total_steps = max(1, len(train_dataset) // (args.batch * args.grad_accum)) * args.epochs
    progress_cb = _make_progress_callback(
        total_steps  = total_steps,
        epochs       = args.epochs,
        dataset_size = len(train_dataset),
        batch        = args.batch,
        grad_accum   = args.grad_accum,
    )

    # ── Memory-efficient SFTTrainer subclass (TRL 1.3.0 VRAM fix) ─────────────
    # TRL 1.3.0's SFTTrainer.compute_loss retains outputs.logits in VRAM after
    # the forward pass to compute per-token entropy and mean_token_accuracy logs.
    # For a 7B model (vocab=152 k) with batch=2 / seq=2048 this costs ~1.2 GB
    # extra — enough to OOM a 16 GB card.
    #
    # Fix: let SFTTrainer run its full compute_loss (so entropy + accuracy metrics
    # are still computed and logged correctly), then immediately delete the logits
    # from the returned outputs object and empty the CUDA cache.  Training quality,
    # gradient computation, and optimizer steps are completely unaffected — only
    # the temporary logit tensor is freed ~1 step earlier than TRL would.
    class _LeanSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(
                model, inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            loss, outputs = result
            # Free the logit tensor immediately — it was only needed for the
            # entropy / accuracy metric computation inside super().compute_loss().
            if hasattr(outputs, "logits") and outputs.logits is not None:
                del outputs.logits
                outputs.logits = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return (loss, outputs) if return_outputs else loss

        def create_model_card(self, *args, **kwargs):
            # TRL's SFTTrainer._save_checkpoint() calls create_model_card() which
            # tries to write a README.md into the LoRA output dir.  On Windows this
            # fails with OSError: [Errno 22] Invalid argument when the file already
            # exists as a Git-tracked file with LF line endings (Windows git lock
            # conflict).  We don't need the auto-generated model card — suppress it.
            pass

    trainer = _LeanSFTTrainer(
        model              = model,
        processing_class   = tokenizer,
        train_dataset      = train_dataset,
        eval_dataset       = eval_dataset,
        args               = sft_cfg,
        callbacks          = [progress_cb],
    )

    # Remove the default PrinterCallback that prints {'loss': ...} dicts to stdout
    from transformers import PrinterCallback
    trainer.remove_callback(PrinterCallback)

    # ── 8. Train ──────────────────────────────────────────────────────────────
    resume_ckpt = args.resume
    if resume_ckpt:
        if not os.path.isdir(resume_ckpt):
            # Try relative to ROOT
            resume_ckpt = os.path.join(ROOT, resume_ckpt)
        if os.path.isdir(resume_ckpt):
            print(f"\n[train] Resuming from checkpoint: {resume_ckpt}", flush=True)
        else:
            print(f"\n[train] WARNING: checkpoint not found at {resume_ckpt}, starting fresh", flush=True)
            resume_ckpt = None

    # ── Pre-remove README.md from LoRA output dir (Windows OSError workaround) ──
    _readme = os.path.join(LORA_OUT_MODEL, "README.md")
    if os.path.exists(_readme):
        try:
            os.remove(_readme)
            print(f"[train] Removed stale README.md from {LORA_OUT_MODEL}")
        except Exception as _e:
            print(f"[train] Warning: could not remove README.md: {_e}")

    print(f"\n[train] Starting training ({args.epochs} epoch(s))...")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    finally:
        # Always restore power settings, even if training crashes
        _restore_sleep()

    print("[train] Training complete!")

    # ── 9. Save LoRA adapter ──────────────────────────────────────────────────
    print(f"[train] Saving LoRA adapter to {LORA_OUT_MODEL}")

    from peft import PeftModel as _PeftModel
    _save_model = model if isinstance(model, _PeftModel) else trainer.model
    _save_model.save_pretrained(LORA_OUT_MODEL)
    tokenizer.save_pretrained(LORA_OUT_MODEL)

    # ── Verify the adapter was saved correctly ────────────────────────────────
    _adapter_cfg = os.path.join(LORA_OUT_MODEL, "adapter_config.json")
    if not os.path.exists(_adapter_cfg):
        # Newer TRL may have saved the adapter inside a checkpoint subdirectory.
        # Find the latest checkpoint and copy the adapter files to LORA_OUT_MODEL root.
        print(f"[train] WARNING: adapter_config.json not found at {LORA_OUT_MODEL}")
        _checkpoints = sorted(
            [d for d in os.listdir(LORA_OUT_MODEL)
             if d.startswith("checkpoint-") and
             os.path.exists(os.path.join(LORA_OUT_MODEL, d, "adapter_config.json"))],
            key=lambda x: int(x.split("-")[1])
        )
        if _checkpoints:
            _latest_ckpt = os.path.join(LORA_OUT_MODEL, _checkpoints[-1])
            print(f"[train] Copying adapter files from latest checkpoint: {_checkpoints[-1]}")
            import glob as _glob
            for _f in _glob.glob(os.path.join(_latest_ckpt, "adapter_*")):
                shutil.copy2(_f, LORA_OUT_MODEL)
                print(f"[train]   Copied: {os.path.basename(_f)}")
            for _tf in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                _src = os.path.join(_latest_ckpt, _tf)
                _dst = os.path.join(LORA_OUT_MODEL, _tf)
                if os.path.exists(_src) and not os.path.exists(_dst):
                    shutil.copy2(_src, _dst)
            print(f"[train] Adapter files recovered from checkpoint.")
        else:
            print(f"[train] ERROR: Could not find adapter_config.json anywhere in {LORA_OUT_MODEL}")
            print(f"[train] The model may not have saved correctly. Try re-running training.")
    else:
        print(f"[train] Adapter saved successfully (adapter_config.json found).")

    # ── Checkpoints are kept after training completes ────────────────────────
    # checkpoint-N/ subdirs remain on disk so that if training is run again
    # on the same model (without Reset), the trainer can resume from where it
    # left off — useful for adding more epochs.
    # Checkpoints are only wiped when the user explicitly presses Reset (which
    # calls _wipe_lora_checkpoints() in server.py before launching train.py
    # with --fresh), or when --fresh is passed on the command line.
    # The presence of adapter_config.json at the LORA_OUT_MODEL root is the
    # definitive signal that training completed 100% — checkpoints are incidental.
    _all_ckpts = [
        d for d in os.listdir(LORA_OUT_MODEL)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(LORA_OUT_MODEL, d))
    ]
    if _all_ckpts:
        print(f"[train] {len(_all_ckpts)} checkpoint dir(s) kept for future resume (use Reset to wipe).")

    print()
    print("=" * 60)
    print("  Training done! LoRA adapter saved.")
    print(f"  Next step: run  python generate_llm.py")
    print("  This merges the adapter into a full model for testing.")
    print("=" * 60)

    # ── Auto-generate if LLM_AUTO_GENERATE env var is set ────────────────────
    if os.environ.get("LLM_AUTO_GENERATE", "").strip().lower() in ("1", "true", "yes"):
        print("\n[train] LLM_AUTO_GENERATE=1  -  auto-running generate_llm.py...")
        import subprocess
        gen_py = os.path.join(ROOT, "train", "generate_llm.py")
        result = subprocess.run(
            [sys.executable, gen_py],
            cwd=ROOT, env=os.environ.copy(),
        )
        if result.returncode == 0:
            print("[train] Model exported successfully ✓")
        else:
            print(f"[train] generate_llm.py exited with code {result.returncode}")


if __name__ == "__main__":
    main()
