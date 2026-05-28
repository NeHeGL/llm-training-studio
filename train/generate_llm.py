"""
LLM Training Studio - Generate / Export Script (NeHe Productions)
Author: Jeff Molofee (aka NeHe) — 2026
Merges the trained LoRA adapter into the full base model and exports it
to models/<ModelName>/ so chat_test/server.py can load it immediately.

Pipeline:
    1. python train/server.py  → open Training Studio, import knowledge, build dataset
    2. python train/train.py   → fine-tune LoRA adapter  → models/<ModelName>/lora/
    3. python generate_llm.py  → merge & export → models/<ModelName>/gguf/
    4. python chat_test/server.py → launch chat UI at http://localhost:5000

Usage:
    python generate_llm.py                  # recommended: auto-selects best GGUF type for your GPU
    python generate_llm.py --no-gguf           # skip GGUF (SafeTensors only — Ollama/LM Studio won't load)
    python generate_llm.py --gguf-type q4_k_m  # override auto-selection with a specific type
    python generate_llm.py --no-bf16           # force fp16 instead of bfloat16
    python generate_llm.py --lora path/to/lora # custom LoRA path

Default output (in models/<ModelName>/gguf/):
    <modelname>.gguf   — The ONLY file you need. Share this. Load in Ollama or LM Studio.
    Modelfile          — Ollama Modelfile (points to the .gguf)

GGUF types (default: q4_k_m — override with --gguf-type):
    q4_k_m  ~4 GB   DEFAULT. Best balance of quality, size, and compatibility. Runs anywhere.
                     Two-step: SafeTensors → f16 GGUF → q4_k_m. llama-quantize installed by install.bat.
    q8_0    ~8 GB   Near-lossless quality. Single-step, no extra tools. Only if you need near-lossless.
    f16     ~14 GB  Lossless, full precision. Needs 16+ GB VRAM headroom just to load for inference.
    q4_0    ~4 GB   Fastest inference, lowest quality. Use only if speed is critical.

The .gguf is ONE self-contained file — weights + tokenizer + config — everything needed.
"""

import os
import re as _re
import sys
import shutil

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Power management (Windows) ────────────────────────────────────────────────
# Previous powercfg values are read and saved before changing, then restored.

_saved_power_settings = {}   # populated by _prevent_sleep(), consumed by _restore_sleep()


def _read_powercfg_value(setting_name):
    """
    Read a single powercfg /query value by name (e.g. 'standby-timeout-ac').
    Returns an int (minutes) or None if not readable.
    """
    import subprocess, re
    _CHANGE_MAP = {
        "standby-timeout-ac":   ("SUB_SLEEP", "STANDBYIDLE",   True),
        "standby-timeout-dc":   ("SUB_SLEEP", "STANDBYIDLE",   False),
        "hibernate-timeout-ac": ("SUB_SLEEP", "HIBERNATEIDLE", True),
        "hibernate-timeout-dc": ("SUB_SLEEP", "HIBERNATEIDLE", False),
        "monitor-timeout-ac":   ("SUB_VIDEO", "VIDEOIDLE",     True),
    }
    if setting_name not in _CHANGE_MAP:
        return None
    _, idle_key, ac_mode = _CHANGE_MAP[setting_name]
    try:
        result = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT"],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.splitlines()
        in_section = False
        ac_next = dc_next = False
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
    for s in ("standby-timeout-ac", "standby-timeout-dc",
              "hibernate-timeout-ac", "hibernate-timeout-dc", "monitor-timeout-ac"):
        _saved_power_settings[s] = _read_powercfg_value(s)
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
        for setting in ("standby-timeout-ac", "standby-timeout-dc",
                        "hibernate-timeout-ac", "hibernate-timeout-dc", "monitor-timeout-ac"):
            subprocess.run(["powercfg", "/change", setting, "0"], capture_output=True)
        print("[power] powercfg standby/hibernate timeouts set to 0.")
    except Exception as e:
        print(f"[power] powercfg adjustment skipped: {e}")


def _restore_sleep():
    """
    Restore the powercfg timeouts that were saved by _prevent_sleep().
    Falls back to safe defaults if values were not readable.
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

    # ── Restore saved powercfg values (fallback if None) ─────────────────────
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
                val = fallback
            subprocess.run(["powercfg", "/change", setting, str(val)], capture_output=True)
            restored.append(f"{setting}={val}")
        print(f"[power] powercfg timeouts restored: {', '.join(restored)}")
    except Exception as e:
        print(f"[power] powercfg restore skipped: {e}")

MODELS_ROOT = os.path.join(ROOT, "models")

# ── Consolidate __pycache__ at project root ───────────────────────────────────
if not sys.pycache_prefix:
    sys.pycache_prefix = os.path.join(ROOT, "__pycache__")

def _load_config():
    """Load active model config from config.json."""
    import json as _json
    cfg_path = os.path.join(ROOT, "config.json")
    default = {"name": "MyModel", "base_model": "Qwen/Qwen2.5-7B"}
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
        safe_n = _re.sub(r'[^\w\-]', '_', name_to_load.strip())
        model_cfg_path = os.path.join(ROOT, "models", safe_n, "config.json")
        if os.path.exists(model_cfg_path):
            return _json.load(open(model_cfg_path, encoding="utf-8"))

        # Per-model config missing — return a minimal default with the name
        return {"name": name_to_load, "base_model": "Qwen/Qwen2.5-7B"}
    except Exception as e:
        print(f"[generate] Warning: could not load config: {e}")
        return default

def _safe_name(name):
    return _re.sub(r'[^\w\-]', '_', name.strip())

_cfg       = _load_config()
AI_NAME    = _cfg.get("name", "MyModel")
BASE_MODEL = _cfg.get("base_model", "Qwen/Qwen2.5-7B")

# All per-model artifacts live under models/{ModelName}/
#   models/{ModelName}/lora/            — LoRA adapters (one subdir per size/rank combo)
#   models/{ModelName}/gguf/{subdir}/   — exported GGUF + Modelfile (one subdir per build)
#   models/{ModelName}/knowledge/       — source Q&A knowledge files
#   models/{ModelName}/dataset/         — built JSONL training files
#
# The LoRA subdir name encodes the model name + size + rank, e.g.:
#   models/TestAI/lora/TestAI_7b_r16_a16/
#
# The GGUF subdir mirrors the LoRA subdir name so each base-model/rank
# combination gets its own isolated output folder, e.g.:
#   models/TestAI/gguf/TestAI_7b_r16_a16/
#   models/TestAI/gguf/TestAI_14b_r16_a16/
#
# This guarantees that switching the base model in the Studio never causes
# a stale GGUF from a previous build to be mistaken for a fresh export.
#
# Scan models/{ModelName}/lora/ for the best available adapter subdir.
# Priority:
#   1. Any lora/{ModelName}_*/ subdir that contains adapter_config.json
#      AND whose base_model_name_or_path matches BASE_MODEL (current config)
#   2. Any lora/{ModelName}_*/ subdir that contains adapter_config.json
#      (fallback — any built adapter for this model name)
#   3. Legacy: adapter_config.json directly in train/popai_lora/ root
#   4. Default: lora/{ModelName}_base/ (will be reported as missing)
_name_part = _safe_name(AI_NAME)
_MODEL_DIR  = os.path.join(MODELS_ROOT, _name_part)
_LORA_BASE  = os.path.join(_MODEL_DIR, "lora")
_lora_dir_found = None
_lora_subdir_name = None   # just the dir basename, used to derive the GGUF subdir

def _read_lora_base_model(lora_dir):
    """Read base_model_name_or_path from lora_dir/adapter_config.json, or None."""
    import json as _json
    _acp = os.path.join(lora_dir, "adapter_config.json")
    try:
        with open(_acp, encoding="utf-8") as _f:
            return _json.load(_f).get("base_model_name_or_path", "")
    except Exception:
        return None

# Scan models/{name}/lora/ for a matching adapter subdir.
# Prefer an adapter whose base_model matches the currently configured BASE_MODEL.
if os.path.isdir(_LORA_BASE):
    _exact_match = None     # matches both name prefix AND base model
    _any_match   = None     # matches name prefix only (used as fallback)
    _any_mtime   = -1.0

    for _entry in os.listdir(_LORA_BASE):
        _full = os.path.join(_LORA_BASE, _entry)
        if not (os.path.isdir(_full)
                and (_entry.startswith(_name_part.lower() + "-") or _entry.startswith(_name_part + "_"))
                and os.path.exists(os.path.join(_full, "adapter_config.json"))):
            continue
        _lora_bm = _read_lora_base_model(_full)
        if _lora_bm and _lora_bm.strip().lower() == BASE_MODEL.strip().lower():
            # Exact match — prefer most-recently-modified if multiple exist
            _mtime = os.path.getmtime(_full)
            if _exact_match is None or _mtime > _any_mtime:
                _exact_match = (_entry, _full)
        # Track any matching subdir (most recently modified wins)
        _mtime = os.path.getmtime(_full)
        if _mtime > _any_mtime:
            _any_match = (_entry, _full)
            _any_mtime = _mtime

    if _exact_match:
        _lora_subdir_name, _lora_dir_found = _exact_match
    elif _any_match:
        _lora_subdir_name, _lora_dir_found = _any_match

# Legacy fallback: check old train/popai_lora/ location
_LEGACY_LORA_ROOT = os.path.join(ROOT, "train", "popai_lora")
if _lora_dir_found is None and os.path.isdir(_LEGACY_LORA_ROOT):
    for _entry in os.listdir(_LEGACY_LORA_ROOT):
        _full = os.path.join(_LEGACY_LORA_ROOT, _entry)
        if (os.path.isdir(_full)
                and _entry.startswith(_name_part + "_")
                and os.path.exists(os.path.join(_full, "adapter_config.json"))):
            _lora_dir_found = _full
            _lora_subdir_name = _entry
            break
    if _lora_dir_found is None and os.path.exists(os.path.join(_LEGACY_LORA_ROOT, "adapter_config.json")):
        _lora_dir_found = _LEGACY_LORA_ROOT
        _lora_subdir_name = None

if _lora_dir_found:
    LORA_DIR = _lora_dir_found
else:
    # Default (not yet trained, or subdir with no adapter yet)
    LORA_DIR = os.path.join(_LORA_BASE, f"{_name_part}_base")
    _lora_subdir_name = f"{_name_part}_base"

# GGUF files go flat into models/{ModelName}/gguf/ — the filenames are already
# unique per build (e.g. TestAI_7b_q4_k_m.gguf, TestAI_14b_q4_k_m.gguf).
# No subfolders needed — all GGUFs for a model coexist in one flat directory.
EXPORT_DIR = os.path.join(_MODEL_DIR, "gguf")
AI_COMPANY = "NeHe Productions"
AI_OWNER   = "Jeff Molofee"
AI_DOMAIN  = AI_NAME


def _gpu_info():
    """Return GPU capabilities dict. Safe when torch is not installed."""
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
        info["bf16"]     = props.major >= 8
        info["is_blackwell"] = props.major >= 12
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


# ── Local runtime detection + installation ────────────────────────────────────

def _detect_ollama():
    """
    Return the ollama executable path if Ollama is installed, else None.
    Checks PATH first, then common install locations.
    """
    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        candidates = [
            r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(
                os.environ.get("USERNAME", "")),
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            os.path.expanduser("~/.local/bin/ollama"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                ollama_exe = c
                break
    return ollama_exe or None


def _ollama_is_running(ollama_exe=None):
    """Return True if the Ollama daemon is reachable.

    Checks the Ollama REST API directly (http://localhost:11434) with a short
    HTTP timeout. This is faster and more reliable than running 'ollama list',
    which can trigger Windows service auto-start and block for many seconds.
    The ollama_exe parameter is accepted for backwards compatibility but unused.
    """
    import urllib.request, urllib.error
    try:
        req = urllib.request.urlopen(
            "http://localhost:11434",
            timeout=2,
        )
        req.close()
        return True
    except urllib.error.HTTPError:
        # Any HTTP error response still means the daemon is up and listening
        return True
    except Exception:
        # Connection refused, timeout, etc. — daemon not running
        return False


def _register_with_ollama(ollama_exe, model_name, modelfile_path, ollama_tag):
    """
    Register the model with Ollama using an already-written Modelfile.
    Returns (success, ollama_tag).
    """
    import subprocess

    # Skip registration if Ollama isn't running — it's optional, just a convenience
    if not _ollama_is_running(ollama_exe):
        print(f"[export] ℹ Ollama is installed but not running — skipping registration.", flush=True)
        print(f"[export]   To register manually once Ollama is running:", flush=True)
        print(f"[export]   ollama create {ollama_tag} -f {modelfile_path}", flush=True)
        return False, ollama_tag

    try:
        print(f"[export] Registering '{ollama_tag}' with Ollama...", flush=True)
        result = subprocess.run(
            [ollama_exe, "create", ollama_tag, "-f", modelfile_path],
            capture_output=True, text=True, timeout=300,
        )

        # Strip ANSI escape sequences from Ollama's terminal spinner output before
        # checking for errors — Ollama uses interactive terminal codes that look like
        # garbage when captured via pipe (e.g. "[?2026h[?25l[1G...").
        import re as _re_ansi
        _ansi_escape = _re_ansi.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\r')
        raw_out = (result.stdout or "") + (result.stderr or "")
        clean_out = _ansi_escape.sub('', raw_out).strip()

        if result.returncode == 0:
            return True, ollama_tag

        # Non-zero exit — but Ollama sometimes exits non-zero while still succeeding
        # (e.g. when registering from a directory). Verify by checking 'ollama list'.
        try:
            check = subprocess.run(
                [ollama_exe, "list"],
                capture_output=True, text=True, timeout=10,
            )
            if check.returncode == 0 and ollama_tag in check.stdout:
                # Model IS registered despite non-zero exit — treat as success
                print(f"[export] ✓ '{ollama_tag}' confirmed registered in Ollama.", flush=True)
                return True, ollama_tag
        except Exception:
            pass

        # Genuinely failed — print cleaned error
        err = clean_out
        if len(err) > 800:
            err = err[:800] + "\n  ... (truncated)"
        print(f"[export] ⚠ Ollama registration failed: {err}", flush=True)
        print(f"[export]   You can register manually once Ollama is running:", flush=True)
        print(f"[export]   ollama create {ollama_tag} -f {modelfile_path}", flush=True)
        return False, ollama_tag
    except subprocess.TimeoutExpired:
        print(f"[export] ⚠ Ollama registration timed out.", flush=True)
        return False, ollama_tag
    except Exception as e:
        print(f"[export] ⚠ Ollama registration error: {e}", flush=True)
        return False, ollama_tag


def _ensure_gguf_package():
    """Install the gguf pip package if not already present."""
    try:
        import gguf  # noqa: F401
        return True
    except ImportError:
        pass
    import subprocess
    print("[export] Installing 'gguf' package for GGUF conversion...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "gguf", "-q"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[export] ⚠ Failed to install gguf: {result.stderr.strip()}")
        return False
    return True


def _patch_gguf_metadata(gguf_path, model_name, system_prompt, base_model, ai_owner, ai_company):
    """
    Patch key metadata fields directly into an existing GGUF file so the
    model's identity and system prompt are self-contained in the file itself.

    Fields written (standard GGUF / llama.cpp keys):
      general.name            — model name shown by Ollama/LM Studio
      general.description     — one-line description
      general.author          — creator
      general.organization    — company
      general.base_model.0.name — the HF base model this was fine-tuned from
      tokenizer.chat_template — the Jinja2 ChatML template with system prompt
                                baked in as the default (matches tokenizer_config.json)

    The patch works by:
      1. Reading the existing GGUF header + all KV metadata
      2. Replacing / adding the target keys
      3. Writing a new GGUF file with the patched metadata + original tensor data

    This is done with the official 'gguf' Python package (same one used by
    convert_hf_to_gguf.py) so the output is always a valid GGUF file.
    """
    try:
        import struct, shutil as _shutil
        import gguf as _gguf_pkg

        print(f"[export] Patching GGUF metadata: {os.path.basename(gguf_path)}", flush=True)

        # ── Read the existing GGUF ────────────────────────────────────────────
        reader = _gguf_pkg.GGUFReader(gguf_path, "r")

        # Collect all existing KV pairs we want to KEEP (skip ones we will replace)
        _REPLACE_KEYS = {
            "general.name",
            "general.description",
            "general.author",
            "general.organization",
            "general.base_model.0.name",
            "tokenizer.chat_template",
        }

        # Build the chat template string — same as what we put in tokenizer_config.json.
        # The system prompt is the default; callers can still override it at runtime.
        _escaped = system_prompt.replace("\\", "\\\\").replace("'", "\\'")
        _bm_lower = base_model.lower()
        if "phi-2" in _bm_lower or "phi2" in _bm_lower:
            chat_template = (
                "{%- set ns = namespace(sys='" + _escaped + "') -%}"
                "{%- for message in messages -%}"
                "{%- if message['role'] == 'system' -%}{%- set ns.sys = message['content'] -%}{%- endif -%}"
                "{%- endfor -%}"
                "{{ ns.sys + '\\n\\n' }}"
                "{%- for message in messages -%}"
                "{%- if message['role'] == 'user' -%}{{ 'Instruct: ' + message['content'] + '\\n' }}"
                "{%- elif message['role'] == 'assistant' -%}{{ 'Output: ' + message['content'] + '\\n' }}"
                "{%- endif -%}"
                "{%- endfor -%}"
                "{%- if add_generation_prompt -%}{{ 'Output: ' }}{%- endif -%}"
            )
        else:
            chat_template = (
                "{%- set ns = namespace(sys='" + _escaped + "') -%}"
                "{%- for message in messages -%}"
                "{%- if message['role'] == 'system' -%}{%- set ns.sys = message['content'] -%}{%- endif -%}"
                "{%- endfor -%}"
                "<|im_start|>system\n{{ ns.sys }}<|im_end|>\n"
                "{%- for message in messages -%}"
                "{%- if message['role'] != 'system' -%}"
                "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                "{%- endif -%}"
                "{%- endfor -%}"
                "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
            )

        description = f"{model_name} — fine-tuned by {ai_owner} ({ai_company})"

        # ── Write patched GGUF to a temp file, then replace original ─────────
        import numpy as _np
        tmp_path = gguf_path + ".patching_tmp"
        try:
            # Extract the architecture string from the reader fields.
            # reader.fields["general.architecture"] is a ReaderField — use
            # .contents() to decode the UTF-8 string value.
            arch_field = reader.fields.get("general.architecture")
            arch_str = arch_field.contents() if arch_field is not None else "llama"

            writer = _gguf_pkg.GGUFWriter(tmp_path, arch=arch_str)

            # Copy all existing KV fields we are NOT replacing
            for key, field in reader.fields.items():
                # Skip internal GGUF header pseudo-fields (version, tensor_count, etc.)
                if key.startswith("GGUF."):
                    continue
                # Skip keys we are replacing and the arch (GGUFWriter adds it automatically)
                if key in _REPLACE_KEYS or key == "general.architecture":
                    continue
                try:
                    ftype = field.types[0]
                    if ftype == _gguf_pkg.GGUFValueType.STRING:
                        val = field.contents()
                        if val:
                            writer.add_string(key, val)
                    elif ftype == _gguf_pkg.GGUFValueType.UINT8:
                        writer.add_uint8(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.INT8:
                        writer.add_int8(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.UINT16:
                        writer.add_uint16(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.INT16:
                        writer.add_int16(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.UINT32:
                        writer.add_uint32(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.INT32:
                        writer.add_int32(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.FLOAT32:
                        writer.add_float32(key, float(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.UINT64:
                        writer.add_uint64(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.INT64:
                        writer.add_int64(key, int(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.FLOAT64:
                        writer.add_float64(key, float(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.BOOL:
                        writer.add_bool(key, bool(field.parts[-1][0]))
                    elif ftype == _gguf_pkg.GGUFValueType.ARRAY:
                        arr = field.contents()
                        if arr:
                            writer.add_array(key, arr)
                except Exception:
                    pass  # Skip fields we can't copy cleanly

            # Write our new / replacement metadata
            writer.add_string("general.name",              model_name)
            writer.add_string("general.description",       description)
            writer.add_string("general.author",            ai_owner)
            writer.add_string("general.organization",      ai_company)
            writer.add_string("general.base_model.0.name", base_model)
            writer.add_string("tokenizer.chat_template",   chat_template)

            # Register tensors with the writer.
            #
            # Shape convention — GGUFReader vs GGUFWriter:
            #
            #   GGUFReader stores tensor.shape = dims exactly as they appear in the
            #   file's tensor info table.  The GGUF spec stores dimensions in REVERSED
            #   order relative to numpy convention (innermost/last dim first).
            #   e.g. a weight matrix with numpy shape (vocab=50279, hidden=4096) is
            #   stored in the file as dims=[4096, 50279].
            #   GGUFReader._build_tensors reverses this back: np_dims=(50279, 4096).
            #
            #   GGUFWriter.write_ti_data_to_file ALSO reverses the shape it receives:
            #   it writes shape[n_dims-1-j] for j in range(n_dims).  So it expects
            #   the shape in numpy (un-reversed) order and reverses it when writing.
            #
            #   Therefore: tensor.shape.tolist() = file-native order = already reversed.
            #   Passing it directly to add_tensor_info() would cause GGUFWriter to
            #   reverse it AGAIN, producing the wrong dim order in the output file.
            #
            #   Fix: pass list(reversed(tensor.shape.tolist())) so that GGUFWriter's
            #   reversal puts the dims back into the correct file-native order.
            #
            # dtype sentinel:
            #   Passing np.uint8 as tensor_dtype triggers quant_shape_from_byte_shape()
            #   which expects the last dim to be a raw byte count — but our shape is
            #   in element units.  Passing np.float32 instead skips that conversion
            #   when raw_dtype is also provided.
            for tensor in reader.tensors:
                writer.add_tensor_info(
                    tensor.name,
                    list(reversed(tensor.shape.tolist())),  # numpy order (GGUFWriter reverses when writing)
                    _np.float32,            # sentinel: skip quant_shape_from_byte_shape
                    tensor.n_bytes,         # total raw bytes in the file
                    tensor.tensor_type,     # GGMLQuantizationType (the real quant type)
                )

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()

            # Write each tensor's raw data in order.
            # tensor.data is the mmap'd numpy view (uint8 for quantized types).
            # write_tensor_data() calls .tofile() on it and pads to alignment —
            # this is the correct, fully supported API approach.
            for tensor in reader.tensors:
                writer.write_tensor_data(tensor.data)

            writer.close()

            # Release the memory-mapped file before attempting to replace it.
            # GGUFReader has no close() method — it stores the mmap as reader.data
            # (a numpy.memmap).  On Windows, the OS keeps the file handle open as
            # long as any reference to the memmap exists, which causes os.replace()
            # to fail with [WinError 5] Access is denied.
            # Fix: explicitly delete reader.data and the reader, then call gc.collect()
            # to force CPython to release the mmap handle before the rename.
            import gc as _gc
            try:
                reader.data._mmap.close()   # numpy memmap exposes the raw mmap object
            except Exception:
                pass
            try:
                del reader.data             # drop the numpy memmap view
            except Exception:
                pass
            try:
                del reader                  # drop the GGUFReader (releases remaining refs)
            except Exception:
                pass
            _gc.collect()                   # force CPython to close the mmap handle now

            # Replace original with patched version.
            # On Windows, os.replace() can still fail if the handle hasn't been
            # released yet.  Fall back to an explicit delete + rename with retries.
            try:
                os.replace(tmp_path, gguf_path)
            except OSError:
                import time as _time
                for _attempt in range(5):
                    try:
                        if os.path.exists(gguf_path):
                            os.remove(gguf_path)
                        os.rename(tmp_path, gguf_path)
                        break
                    except OSError:
                        _time.sleep(0.5)
                else:
                    raise
            print(f"[export] ✅ GGUF metadata patched: name='{model_name}', system prompt baked in", flush=True)
            return True

        except Exception as e:
            # Clean up temp file on failure
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if hasattr(reader, "close"):
                try: reader.close()
                except Exception: pass
            raise e

    except Exception as e:
        print(f"[export] ⚠ GGUF metadata patch skipped: {e}", flush=True)
        print(f"[export]   The GGUF is still valid — identity is provided via the Modelfile instead.", flush=True)
        return False


def _convert_to_gguf(hf_model_dir, output_path, quantization="q8_0"):
    """
    Convert a HuggingFace SafeTensors model directory to a single GGUF file
    using llama.cpp's convert_hf_to_gguf.py script (downloaded on demand).

    Returns the path to the .gguf file on success, or None on failure.
    """
    import subprocess, urllib.request, tempfile

    # ── Ensure gguf package is available ──────────────────────────────────────
    if not _ensure_gguf_package():
        print("[export] ⚠ GGUF conversion skipped (gguf package unavailable).")
        return None

    # ── Locate or download convert_hf_to_gguf.py ─────────────────────────────
    # Look for a local copy first (user may have llama.cpp checked out)
    script_name = "convert_hf_to_gguf.py"
    local_script = os.path.join(ROOT, "tools", script_name)
    candidate_dirs = [
        os.path.join(ROOT, "tools"),
        os.path.join(ROOT, "llama.cpp"),
        os.path.expanduser("~/llama.cpp"),
    ]
    convert_script = None
    for d in candidate_dirs:
        candidate = os.path.join(d, script_name)
        if os.path.isfile(candidate):
            convert_script = candidate
            break

    if convert_script is None:
        # Download directly from the llama.cpp GitHub repo
        url = (
            "https://raw.githubusercontent.com/ggerganov/llama.cpp/"
            "master/convert_hf_to_gguf.py"
        )
        os.makedirs(os.path.join(ROOT, "tools"), exist_ok=True)
        print(f"[export] Downloading {script_name} from llama.cpp GitHub...")
        try:
            urllib.request.urlretrieve(url, local_script)
            convert_script = local_script
            print(f"[export] Saved to: {local_script}")
        except Exception as e:
            print(f"[export] ⚠ Could not download {script_name}: {e}")
            return None

    # ── Run the conversion ────────────────────────────────────────────────────
    # convert_hf_to_gguf.py only supports: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto
    # Types like q4_k_m and q4_0 require a two-step process:
    #   1. Convert to f16 GGUF with convert_hf_to_gguf.py
    #   2. Quantize to q4_k_m/q4_0 with llama-quantize (from llama.cpp release)
    # If llama-quantize is not installed, we keep the f16 GGUF as the output.
    _SUPPORTED_DIRECT = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
    # Map user-facing names to convert_hf_to_gguf.py --outtype values
    _DIRECT_OUTTYPE = {
        "f16":    "f16",
        "bf16":   "bf16",
        "q8_0":   "q8_0",
        "tq1_0":  "tq1_0",
        "tq2_0":  "tq2_0",
        "auto":   "auto",
        # These require llama-quantize post-processing:
        "q4_k_m": "f16",
        "q4_0":   "f16",
    }
    # llama-quantize type strings
    _QUANTIZE_TYPE = {
        "q4_k_m": "Q4_K_M",
        "q4_0":   "Q4_0",
    }
    needs_quantize = quantization not in _SUPPORTED_DIRECT
    outtype = _DIRECT_OUTTYPE.get(quantization, "f16")

    # If we need a post-quantize step, write f16 to a temp path first
    if needs_quantize:
        import tempfile
        f16_path = output_path.replace(
            f"_{quantization}.gguf", "_f16_tmp.gguf"
        ).replace(".gguf", "_f16_tmp.gguf") if not output_path.endswith(f"_{quantization}.gguf") else \
            output_path[: output_path.rfind(f"_{quantization}.gguf")] + "_f16_tmp.gguf"
        # Simpler approach: always put f16 temp file next to the final output
        f16_path = os.path.join(os.path.dirname(output_path), "_tmp_f16.gguf")
        convert_output = f16_path
        print(f"[export] Note: '{quantization}' requires two-step conversion.")
        print(f"[export]   Step 1: Convert SafeTensors → f16 GGUF (intermediate)")
        print(f"[export]   Step 2: Quantize f16 GGUF → {quantization} with llama-quantize")
    else:
        convert_output = output_path
        f16_path = None

    print(f"[export] Converting SafeTensors → GGUF ({outtype})...")
    print(f"[export]   Input : {hf_model_dir}")
    print(f"[export]   Output: {convert_output}")

    # Run convert_hf_to_gguf.py as a direct subprocess.
    # We suppress per-tensor INFO spam by setting PYTHONWARNINGS and passing a
    # wrapper -W flag — the convert script itself uses Python's logging module,
    # so we silence the noisy loggers via a small sitecustomize trick using env.
    # Note: exec() cannot be used here because convert_hf_to_gguf.py references
    # __file__ internally, which is not defined in an exec() context.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Suppress INFO-level log spam from hf-to-gguf and gguf loggers via env var
    env["HF_TO_GGUF_LOG_LEVEL"] = "WARNING"
    # Use a wrapper script that sets log levels then runs convert as __main__
    # convert_output is either the final path (direct types) or a temp f16 path
    wrapper = (
        "import logging, runpy, sys; "
        "logging.getLogger('hf-to-gguf').setLevel(logging.WARNING); "
        "logging.getLogger('gguf').setLevel(logging.WARNING); "
        "logging.getLogger('gguf.gguf_writer').setLevel(logging.WARNING); "
        "logging.getLogger('gguf.vocab').setLevel(logging.WARNING); "
        f"sys.argv = {[convert_script, hf_model_dir, '--outfile', convert_output, '--outtype', outtype]!r}; "
        f"runpy.run_path({convert_script!r}, run_name='__main__')"
    )
    cmd = [sys.executable, "-c", wrapper]
    try:
        # Launch the GGUF converter with piped stdout so it is NOT attached to
        # the parent console.  This prevents Windows from sending CTRL_CLOSE_EVENT
        # to the subprocess when the console window is closed, which was causing
        # the Fortran runtime (numpy/scipy) inside the converter to crash with
        # "forrtl: error (200): program aborting due to window-CLOSE event".
        # We stream each line back to our own stdout so progress bars still appear.
        sys.stdout.flush()
        kwargs = {}
        if sys.platform == "win32":
            # CREATE_NEW_PROCESS_GROUP isolates the child from Ctrl+C/Ctrl+Break
            # console events, preventing the converter from being killed when the
            # user hits Ctrl+C in the parent console.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **kwargs,
        )
        # Stream converter output line-by-line so tqdm progress bars appear live
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = proc.wait()
        sys.stdout.flush()
        if returncode != 0 or not os.path.isfile(convert_output):
            print(f"[export] ⚠ GGUF conversion failed (exit code {returncode}).", flush=True)
            return None

        # ── Step 2: Quantize with llama-quantize if needed ────────────────────
        if needs_quantize:
            quant_type = _QUANTIZE_TYPE.get(quantization, quantization.upper())
            # Look for llama-quantize in PATH and common install locations
            quantize_exe = shutil.which("llama-quantize") or shutil.which("llama_quantize")
            if not quantize_exe:
                llama_candidates = [
                    # Installed by install.bat into tools/llama/ — checked first
                    os.path.join(ROOT, "tools", "llama", "llama-quantize.exe"),
                    os.path.join(ROOT, "tools", "llama", "llama-quantize"),
                    os.path.join(ROOT, "llama.cpp", "build", "bin", "Release", "llama-quantize.exe"),
                    os.path.join(ROOT, "llama.cpp", "build", "bin", "llama-quantize.exe"),
                    os.path.join(ROOT, "llama.cpp", "build", "bin", "Release", "llama-quantize"),
                    os.path.join(ROOT, "llama.cpp", "build", "bin", "llama-quantize"),
                    os.path.expanduser("~/llama.cpp/build/bin/Release/llama-quantize.exe"),
                    os.path.expanduser("~/llama.cpp/build/bin/llama-quantize.exe"),
                    r"C:\Program Files\llama.cpp\llama-quantize.exe",
                    "/usr/local/bin/llama-quantize",
                ]
                for c in llama_candidates:
                    if os.path.isfile(c):
                        quantize_exe = c
                        break

            if quantize_exe:
                print(f"[export] Step 2: Quantizing f16 GGUF → {quantization} using llama-quantize...", flush=True)
                print(f"[export]   {quantize_exe} {convert_output} {output_path} {quant_type}", flush=True)
                try:
                    q_proc = subprocess.Popen(
                        [quantize_exe, convert_output, output_path, quant_type],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    for line in q_proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    q_ret = q_proc.wait()
                    if q_ret == 0 and os.path.isfile(output_path):
                        # Remove the intermediate f16 temp file
                        try:
                            os.remove(convert_output)
                        except Exception:
                            pass
                        size_gb = os.path.getsize(output_path) / 1e9
                        print(f"[export] ✅ GGUF quantization complete ({size_gb:.1f} GB): {output_path}", flush=True)
                        return output_path
                    else:
                        print(f"[export] ⚠ llama-quantize failed (exit {q_ret}) — keeping f16 GGUF instead.", flush=True)
                        # Fall through to f16 fallback below
                except Exception as qe:
                    print(f"[export] ⚠ llama-quantize error: {qe} — keeping f16 GGUF instead.", flush=True)
            else:
                print(f"[export] ℹ llama-quantize not found — keeping f16 GGUF (larger but lossless).", flush=True)
                print(f"[export]   To get a {quantization} GGUF later, install llama.cpp and run:", flush=True)
                print(f"[export]   llama-quantize {convert_output} {output_path} {quant_type}", flush=True)

            # llama-quantize unavailable or failed: rename f16 temp → adjusted output path
            # Update output_path to reflect the actual type (f16) so caller uses correct name
            f16_output = output_path.replace(f"_{quantization}.gguf", "_f16.gguf")
            try:
                os.rename(convert_output, f16_output)
                size_gb = os.path.getsize(f16_output) / 1e9
                print(f"[export] ✅ f16 GGUF saved ({size_gb:.1f} GB): {f16_output}", flush=True)
                return f16_output
            except Exception as re_err:
                # rename failed — return the temp path as-is
                size_gb = os.path.getsize(convert_output) / 1e9
                print(f"[export] ✅ f16 GGUF saved ({size_gb:.1f} GB): {convert_output}", flush=True)
                return convert_output
        else:
            size_gb = os.path.getsize(convert_output) / 1e9
            print(f"[export] ✅ GGUF conversion complete ({size_gb:.1f} GB): {convert_output}", flush=True)
            return convert_output
    except Exception as e:
        print(f"[export] ⚠ GGUF conversion error: {e}", flush=True)
        return None


def _write_modelfile(export_dir, gguf_path, model_name, system_prompt=""):
    """
    Write a Modelfile in export_dir named to match the GGUF it references.

    Since multiple GGUFs can coexist in the same flat gguf/ directory
    (e.g. TestAI_7b_q4_k_m.gguf and TestAI_14b_q4_k_m.gguf), each gets
    its own .ollama Modelfile so they never overwrite each other:
      TestAI_7b_q4_k_m.ollama   ← points to TestAI_7b_q4_k_m.gguf
      TestAI_14b_q4_k_m.ollama  ← points to TestAI_14b_q4_k_m.gguf

    If no GGUF path is provided (SafeTensors-only export), falls back to
    a generic "Modelfile" pointing at the export directory.

    The SYSTEM block is written so Ollama injects the correct identity prompt
    at inference time.  NOTE: Although the system prompt is also baked into the
    GGUF's tokenizer.chat_template by _patch_gguf_metadata() (for portability
    with llama.cpp / LM Studio), Ollama uses its own Go template engine and does
    NOT read the Jinja2 chat_template from the GGUF — it requires the SYSTEM
    block in the Modelfile to inject the system prompt at runtime.
    """
    import json as _json

    # Read temperature/params from generation_config if present
    gen_cfg_path = os.path.join(export_dir, "generation_config.json")
    temperature = 0.3
    top_p = 0.9
    repeat_penalty = 1.3
    num_predict = 512
    try:
        with open(gen_cfg_path, encoding="utf-8") as _f:
            gc = _json.load(_f)
        temperature = gc.get("temperature", temperature)
        top_p = gc.get("top_p", top_p)
        repeat_penalty = gc.get("repetition_penalty", repeat_penalty)
        num_predict = gc.get("max_new_tokens", num_predict)
    except Exception:
        pass

    from_line = gguf_path if gguf_path else export_dir
    # Use forward slashes — Ollama handles both on Windows
    from_line = from_line.replace("\\", "/")

    # ChatML template — must match the format used during fine-tuning.
    # Without this, Ollama uses {{ .Prompt }} (raw text) and messages are
    # not wrapped in <|im_start|>/<|im_end|> tokens.
    _chatml_template = (
        "{{- if .System }}<|im_start|>system\\n{{ .System }}<|im_end|>\\n{{ end }}"
        "{{- range .Messages }}<|im_start|>{{ .Role }}\\n{{ .Content }}<|im_end|>\\n{{ end }}"
        "<|im_start|>assistant\\n"
    )

    # Use the provided system_prompt; fall back to a minimal identity string.
    _system = system_prompt.strip() if system_prompt else f"You are {model_name}, a helpful AI assistant."

    lines = [
        f"FROM {from_line}",
        f'TEMPLATE "{_chatml_template}"',
        f'SYSTEM """{_system}"""',
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
        f"PARAMETER repeat_penalty {repeat_penalty}",
        f"PARAMETER num_predict {num_predict}",
        "PARAMETER stop <|im_end|>",
        "PARAMETER stop <|endoftext|>",
        "",
    ]

    # Name the Modelfile to match the GGUF stem with a .ollama extension so
    # multiple builds coexist in the same flat gguf/ directory:
    #   TestAI_7b_q4_k_m.gguf  →  TestAI_7b_q4_k_m.ollama
    #   TestAI_14b_q4_k_m.gguf →  TestAI_14b_q4_k_m.ollama
    if gguf_path:
        gguf_stem = os.path.splitext(os.path.basename(gguf_path))[0]
        modelfile_name = f"{gguf_stem}.ollama"
    else:
        modelfile_name = "Modelfile"

    modelfile_path = os.path.join(export_dir, modelfile_name)
    with open(modelfile_path, "w", encoding="utf-8") as _f:
        _f.write("\n".join(lines))
    print(f"[export] Modelfile written: {modelfile_path}")
    return modelfile_path


def _cleanup_orphaned_ollama_models(ollama_exe):
    """
    For every model registered in Ollama, check whether the GGUF file it was
    registered from still exists on disk.  If it does not, run 'ollama rm' to
    remove the stale entry.

    How we know the original GGUF path:
      When a model is exported, generate_llm.py writes a .ollama Modelfile next
      to the GGUF (e.g. models/TestAI/gguf/testai_7b_q4_k_m-qwen2-5.ollama)
      containing a FROM line that points to the absolute GGUF path.  That file
      is the authoritative record of "what path was specified at registration time".

    We scan every models/*/gguf/*.ollama (and legacy Modelfile) across the
    project, build a map of ollama_tag → gguf_path, then for each Ollama tag:
      • If we have a record and the GGUF exists  → leave it alone (healthy)
      • If we have a record and the GGUF is gone → remove from Ollama
      • If we have no record but the tag matches a known model name prefix
        (from our models/ folders or config.json) → Modelfile was deleted too,
        but the model is still ours → remove from Ollama

    Models installed by the user manually (qwen3, deepseek, mistral, etc.) will
    never match one of our model-name prefixes, so they are never touched.

    This function is only called when Ollama is already confirmed to be running
    (_ollama_is_running returned True), so no extra connectivity check is needed.
    """
    import subprocess, json as _json
    try:
        result = subprocess.run(
            [ollama_exe, "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return

        # Parse tag names from 'ollama list' output (skip header line).
        # Each line: "name:tag   id   size   modified"
        lines = result.stdout.strip().splitlines()
        ollama_tags = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                ollama_tags.append(parts[0].split(":")[0])   # strip ":latest"

        if not ollama_tags:
            return

        # ── Locate project paths ──────────────────────────────────────────────
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir   = os.path.join(project_root, "models")

        # ── Build: tag → gguf_path from every .ollama / Modelfile we wrote ───
        # Also build the set of safe-lowercase model-name prefixes so we can
        # identify tags whose Modelfiles have been deleted alongside the GGUFs.
        tag_to_gguf  = {}      # derived_tag → absolute gguf path
        our_prefixes = set()   # safe-lowercase model name prefixes we own

        if os.path.isdir(models_dir):
            for folder in os.listdir(models_dir):
                model_dir = os.path.join(models_dir, folder)
                if not os.path.isdir(model_dir):
                    continue

                # Prefix for matching (e.g. "TestAI" → "testai")
                our_prefixes.add(_re.sub(r'[^a-z0-9_\-\.]', '-', folder.lower()).strip('-'))

                # Collect Modelfiles: new .ollama per-build + legacy generic Modelfile
                mf_paths = []
                gguf_dir = os.path.join(model_dir, "gguf")
                if os.path.isdir(gguf_dir):
                    for fname in os.listdir(gguf_dir):
                        if fname == "Modelfile" or fname.endswith(".ollama"):
                            mf_paths.append(os.path.join(gguf_dir, fname))
                legacy = os.path.join(model_dir, "Modelfile")
                if os.path.isfile(legacy):
                    mf_paths.append(legacy)

                for mf in mf_paths:
                    try:
                        with open(mf, encoding="utf-8") as fh:
                            for line in fh:
                                line = line.strip()
                                if line.upper().startswith("FROM "):
                                    gguf_path = line[5:].strip().replace("/", os.sep)
                                    stem = os.path.splitext(os.path.basename(gguf_path))[0]
                                    dtag = _re.sub(r'[^a-z0-9_\-\.]', '-', stem.lower()).strip('-')
                                    tag_to_gguf[dtag] = gguf_path
                                    break
                    except Exception:
                        pass

        # ── Also pull model names from config.json (catches deleted folders) ─
        cfg_path = os.path.join(project_root, "config.json")
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg_data = _json.load(fh)
            for m in cfg_data.get("models", []):
                name = m if isinstance(m, str) else m.get("name", "")
                if name:
                    our_prefixes.add(_re.sub(r'[^a-z0-9_\-\.]', '-', name.lower()).strip('-'))
        except Exception:
            pass

        # ── Check every Ollama tag ─────────────────────────────────────────────
        def _remove(tag, reason):
            print(f"[export] 🧹 Removing orphaned Ollama model '{tag}' ({reason})...", flush=True)
            try:
                subprocess.run([ollama_exe, "rm", tag],
                               capture_output=True, text=True, timeout=30)
            except Exception as ex:
                print(f"[export]   ⚠ Could not remove '{tag}': {ex}", flush=True)

        for tag in ollama_tags:
            if tag in tag_to_gguf:
                # We know the GGUF path — check it still exists on disk
                if not os.path.isfile(tag_to_gguf[tag]):
                    _remove(tag, "GGUF file no longer on disk")
            else:
                # No Modelfile found — check if the tag belongs to one of our models
                # by seeing whether it starts with a model-name prefix we own.
                for prefix in our_prefixes:
                    if tag == prefix or tag.startswith(prefix + "-") or tag.startswith(prefix + "_"):
                        _remove(tag, "Modelfile removed — model no longer exported")
                        break
                # No prefix match → not our model, leave it alone

    except Exception as e:
        print(f"[export] ⚠ Ollama orphan cleanup skipped: {e}", flush=True)


def _install_to_local_runtimes(model_name, export_dir, gguf_path=None, system_prompt=""):
    """
    Detect Ollama and register the newly exported model with it.
    Prefers the GGUF file; falls back to the directory.
    The Ollama tag is derived from the GGUF filename so it includes the
    quantization type (e.g. popai_q4_k_m.gguf → popai-q4-k-m).
    Before registering, cleans up any orphaned Ollama models whose backing
    GGUF files no longer exist on disk.
    Prints a friendly summary at the end.
    Returns the ollama_tag used (or None if Ollama not found).
    """
    ollama_exe = _detect_ollama()
    if not ollama_exe:
        return None  # Ollama not installed — nothing to do

    # ── Clean up orphaned models before registering the new one ──────────────
    if _ollama_is_running(ollama_exe):
        _cleanup_orphaned_ollama_models(ollama_exe)

    # Derive the Ollama tag from the GGUF filename to include the quant type.
    # e.g. "popai_q4_k_m.gguf" → "popai-q4-k-m"
    # e.g. "popai_f16.gguf"    → "popai-f16"
    # Fall back to just model_name if no GGUF available.
    if gguf_path:
        gguf_stem = os.path.splitext(os.path.basename(gguf_path))[0]  # e.g. "popai_q4_k_m"
        tag_name = gguf_stem
    else:
        tag_name = model_name

    # Write an updated Modelfile pointing to the GGUF (or directory)
    modelfile_path = _write_modelfile(export_dir, gguf_path, model_name, system_prompt=system_prompt)

    # Derive the ollama tag (lowercase, safe chars only)
    ollama_tag = _re.sub(r'[^a-z0-9_\-\.]', '-', tag_name.lower()).strip('-')

    # Register with Ollama using the already-written Modelfile
    ok, ollama_tag = _register_with_ollama(ollama_exe, tag_name, modelfile_path, ollama_tag)
    if ok:
        print(flush=True)
        print(f"[export] ✅ Ollama was found on your system and '{ollama_tag}' has been", flush=True)
        print(f"[export]    configured as a model — you can now run it directly with:", flush=True)
        print(f"[export]    ollama run {ollama_tag}", flush=True)
    return ollama_tag if ok else None


def main():
    import argparse
    p = argparse.ArgumentParser(description=f"Merge {AI_NAME} LoRA adapter and export to GGUF")
    p.add_argument("--lora",      default=LORA_DIR,   help="LoRA adapter directory")
    p.add_argument("--out",       default=EXPORT_DIR, help="Export destination (default: models/<name>/)")
    p.add_argument("--no-bf16",   dest="use_bf16", action="store_false", help="Force fp16 instead of bfloat16")
    p.add_argument("--no-gguf",          dest="do_gguf", action="store_false",
                   help="Skip GGUF conversion entirely (SafeTensors only — Ollama/LM Studio won't load this)")
    p.add_argument("--keep-safetensors", dest="keep_safetensors", action="store_true",
                   help="Keep SafeTensors files alongside the .gguf. "
                        "Only needed if you want to use chat_test/server.py (the Python chat tester). "
                        "By default they are deleted after GGUF conversion to save ~15 GB.")
    p.add_argument("--gguf-type",        dest="gguf_type", default=None,
                   choices=["f16", "q8_0", "q4_k_m", "q4_0"],
                   help="GGUF quantization type. Auto-selected based on your GPU VRAM if not specified. "
                        "f16=lossless ~14 GB (16+ GB VRAM); q8_0=near-lossless ~8 GB (8-16 GB VRAM); "
                        "q4_k_m=good quality ~4 GB (4-8 GB VRAM); q4_0=smallest ~4 GB (CPU/low VRAM)")
    p.set_defaults(use_bf16=True, do_gguf=True, keep_safetensors=False)
    args = p.parse_args()

    # ── Auto-select GGUF type ─────────────────────────────────────────────────
    # q4_k_m is always the best default:
    #   - ~4 GB — runs on any hardware (4 GB VRAM, 8 GB RAM for CPU)
    #   - Barely perceptible quality loss vs f16/q8_0
    #   - What Ollama and llama.cpp recommend by default
    #   - f16 (~14 GB) needs 16+ GB VRAM headroom just to *load* for inference
    #     and leaves almost nothing for context — a poor experience even on high-end GPUs
    #   - q8_0 (~8 GB) is only useful if you specifically need near-lossless precision
    # Override with --gguf-type if you have a specific reason (e.g. q8_0 for eval work).
    # llama-quantize is installed automatically by install.bat into tools/llama/.
    if (args.gguf_type is None or args.gguf_type == "auto") and args.do_gguf:
        args.gguf_type = "q4_k_m"
        print(f"[export] GGUF type auto-selected : q4_k_m (~4 GB, best quality/size balance)")
        print(f"[export]   Two-step conversion: SafeTensors → f16 GGUF → q4_k_m GGUF")
        print(f"[export]   (llama-quantize installed by install.bat — no extra setup needed)")
    elif args.gguf_type is not None:
        if args.gguf_type in ("q4_k_m", "q4_0"):
            print(f"[export] GGUF type (manual)      : {args.gguf_type}")
            print(f"[export]   Two-step conversion: SafeTensors → f16 GGUF → {args.gguf_type} GGUF")
        else:
            print(f"[export] GGUF type (manual)      : {args.gguf_type} (single-step)")

    lora_dir   = os.path.abspath(args.lora)
    export_dir = os.path.abspath(args.out)

    # ── Prevent sleep during the long merge+save operation ────────────────────
    _prevent_sleep()

    print("=" * 60)
    print(f"  {AI_NAME} — LLM Export ({AI_COMPANY})")
    print(f"  Owner: {AI_OWNER}  |  Domain: {AI_DOMAIN}")
    print("=" * 60)

    # ── Validate LoRA adapter ─────────────────────────────────────────────────
    if not os.path.isdir(lora_dir):
        print(f"[export] ERROR: LoRA adapter not found at: {lora_dir}")
        print(f"[export] Train first:  python train\\train.py")
        sys.exit(1)

    if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        # Try to find the latest checkpoint subdirectory
        checkpoints = sorted(
            [d for d in os.listdir(lora_dir)
             if d.startswith("checkpoint-") and
             os.path.exists(os.path.join(lora_dir, d, "adapter_config.json"))],
            key=lambda x: int(x.split("-")[1])
        )
        if checkpoints:
            latest = checkpoints[-1]
            lora_dir = os.path.join(lora_dir, latest)
            print(f"[export] Using latest checkpoint: {lora_dir}")
        else:
            print(f"[export] ERROR: {lora_dir} is not a LoRA adapter (missing adapter_config.json)")
            print(f"[export] Train first:  python train\\train.py")
            sys.exit(1)

    print(f"[export] LoRA adapter  : {lora_dir}")
    print(f"[export] Base model    : {BASE_MODEL}")
    print(f"[export] Export to     : {export_dir}")

    # ── Validate model exists on HuggingFace ──────────────────────────────────
    print(f"[export] Checking model exists: {BASE_MODEL}")
    try:
        from huggingface_hub import model_info
        model_info(BASE_MODEL)
        print(f"[export] Model found ✓")
    except Exception:
        print(f"\n[export] ERROR: Model '{BASE_MODEL}' was not found on HuggingFace.")
        print(f"[export] The model ID you specified in config.json does not exist.")
        print(f"[export] Check the exact model ID at: https://huggingface.co/models")
        print(f"[export] Then update 'base_model' in config.json and try again.")
        _restore_sleep()
        sys.exit(1)

    # ── Check packages ────────────────────────────────────────────────────────
    missing = []
    for pkg in ("transformers", "peft", "accelerate", "sentencepiece"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[export] Missing packages: {', '.join(missing)}")
        print("[export] Run:  pip install transformers peft accelerate sentencepiece")
        sys.exit(1)

    import warnings
    # Suppress the "expandable_segments not supported on this platform" UserWarning.
    # Fires on Windows if PYTORCH_CUDA_ALLOC_CONF contains expandable_segments:True
    # (e.g. set by a system env var outside our control).
    warnings.filterwarnings(
        "ignore",
        message=".*expandable_segments.*not supported.*",
        category=UserWarning,
    )

    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Reduce CUDA allocator fragmentation — same as train.py.
    # expandable_segments is not supported on Windows — omit it there to avoid a UserWarning.
    if sys.platform == "win32":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    else:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

    gi   = _gpu_info()
    cuda = gi["has_cuda"]
    # For the merge operation we load the model in full precision (fp16 or bf16).
    # BF16 is safe here because there is NO BNB backward pass during merge —
    # merge_and_unload() just does fp32 addition of weight deltas.
    # Use bf16 on Ampere+ (best quality, smallest memory), fp16 on older GPUs.
    # --no-bf16 forces fp16 regardless (useful for very old cards or CPU-only).
    bf16 = gi["bf16"] and args.use_bf16
    dtype = torch.bfloat16 if bf16 else torch.float16

    _gpu_label = f"{gi['name']} (SM {gi['sm_major']}.{gi['sm_minor']})" if cuda else "CPU (slow)"
    print(f"[export] GPU           : {_gpu_label}")
    print(f"[export] VRAM          : {gi['vram_gb']:.1f} GB" if cuda else "[export] VRAM          : N/A")
    print(f"[export] dtype         : {'bfloat16' if bf16 else 'float16'}")
    print()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = lora_dir if os.path.exists(os.path.join(lora_dir, "tokenizer_config.json")) else BASE_MODEL
    print(f"[export] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # Load the system prompt from config so it can be baked into the chat template.
    # This is the correct way to embed identity — convert_hf_to_gguf.py reads the
    # tokenizer_config.json chat_template and stores it in the GGUF metadata, so the
    # system prompt is permanently part of the GGUF file without needing a Modelfile.
    import json as _jcfg2
    _sys_prompt = ""
    try:
        _sp_cfg_path = os.path.join(MODELS_ROOT, _safe_name(AI_NAME), "config.json")
        with open(_sp_cfg_path, encoding="utf-8") as _f2:
            _sys_prompt = _jcfg2.load(_f2).get("system_prompt", "").strip()
    except Exception:
        pass
    if not _sys_prompt:
        _sys_prompt = f"You are {AI_NAME}, a helpful AI assistant."
    print(f"[export] System prompt         : {_sys_prompt[:80]}{'...' if len(_sys_prompt) > 80 else ''}")

    # Build a ChatML template that bakes in the system prompt as the default.
    # When a caller passes no system message, this default fires automatically.
    # When a caller does pass a system message, that overrides the default.
    # This matches what published fine-tuned models on HuggingFace do — the system
    # prompt is baked into tokenizer_config.json and therefore into the GGUF itself,
    # so no Modelfile is needed for the model to know who it is.
    _bm = BASE_MODEL.lower()
    if "phi-2" in _bm or "phi2" in _bm:
        # Phi-2 uses Instruct:/Output: format, not ChatML
        _escaped = _sys_prompt.replace("\\", "\\\\").replace("'", "\\'")
        print("[export] Injecting Phi-2 chat template with system prompt.")
        tokenizer.chat_template = (
            "{% set ns = namespace(sys='" + _escaped + "') %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{% set ns.sys = message['content'] %}{% endif %}"
            "{% endfor %}"
            "{{ ns.sys + '\\n\\n' }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ 'Instruct: ' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ 'Output: ' + message['content'] + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ 'Output: ' }}"
            "{% endif %}"
        )
    else:
        # Standard ChatML — used by Qwen, Mistral, Pythia, SmolLM2, etc.
        # The system prompt is embedded as the default; callers can override it.
        _escaped = _sys_prompt.replace("\\", "\\\\").replace("'", "\\'")
        if getattr(tokenizer, "chat_template", None):
            print("[export] Replacing existing chat template with system-prompt-aware ChatML template.")
        else:
            print("[export] No chat template found — injecting ChatML template with system prompt.")
        tokenizer.chat_template = (
            "{% set ns = namespace(sys='" + _escaped + "') %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{% set ns.sys = message['content'] %}{% endif %}"
            "{% endfor %}"
            "<|im_start|>system\n{{ ns.sys }}<|im_end|>\n"
            "{% for message in messages %}"
            "{% if message['role'] != 'system' %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )

    # ── Base model (fp16/bf16 — can't merge from 4-bit) ───────────────────────
    # Use GPU if VRAM is sufficient to hold the full model (avoids the PEFT
    # offload_dir issue that occurs when device_map="auto" splits across GPU+CPU).
    # A 7B fp16/bf16 model is ~14 GB; add a 2 GB headroom buffer.
    # If VRAM is insufficient, fall back to CPU cleanly.
    _model_vram_needed_gb = 16  # conservative estimate for 7B fp16/bf16
    if cuda and gi["vram_gb"] >= _model_vram_needed_gb:
        _merge_device_map = "cuda:0"
        _merge_device_label = f"GPU ({gi['name']})"
    else:
        _merge_device_map = None   # CPU
        _merge_device_label = "CPU (insufficient VRAM for GPU merge)"

    print(f"[export] Loading base model in {'bfloat16' if bf16 else 'float16'} for merge...")
    # Check if base model is already cached locally
    _model_cached = False
    try:
        from huggingface_hub import try_to_load_from_cache
        _cached_check = try_to_load_from_cache(BASE_MODEL, "config.json")
        _model_cached = _cached_check is not None and _cached_check != "not_cached"
    except Exception:
        pass
    if _model_cached:
        print(f"[export] (Base model cached locally — no download needed)")
    else:
        # ANSI yellow so the user notices this will be a long download
        _YELLOW = "\033[33m"
        _RESET  = "\033[0m"
        print(f"{_YELLOW}[export] ⚠ Base model not cached — downloading ~15 GB (one-time, cached after){_RESET}", flush=True)
    print(f"[export] Merge device     : {_merge_device_label}")
    print(f"[export] Allocating model into memory — this takes 1-3 min, please wait...", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype             = dtype,
        device_map        = _merge_device_map,
        low_cpu_mem_usage = True,   # load shard-by-shard → halves peak RAM usage
        trust_remote_code = True,
    )
    print(f"[export] Base model loaded ✓", flush=True)

    # ── Resize embeddings to match the trained LoRA adapter ───────────────────
    # The LoRA adapter's embedding layer size is recorded in adapter_config.json
    # as modules_to_save or can be inferred from the saved embedding weights.
    # The safest approach: read the actual embedding size from the saved adapter
    # (adapter_model.safetensors contains embed_tokens.weight if it was resized).
    # Fall back to len(tokenizer) if the adapter doesn't include embedding weights.
    import json as _jcfg
    _adapter_cfg_path = os.path.join(lora_dir, "adapter_config.json")
    _adapter_emb_size = None
    try:
        import safetensors.torch as _st
        _adapter_st = os.path.join(lora_dir, "adapter_model.safetensors")
        if os.path.exists(_adapter_st):
            _adapter_tensors = _st.load_file(_adapter_st)
            # Check for resized embedding weights saved in the adapter
            for _k in _adapter_tensors:
                if "embed_tokens.weight" in _k or "lm_head.weight" in _k:
                    _adapter_emb_size = _adapter_tensors[_k].shape[0]
                    break
    except Exception:
        pass

    base_vocab = base_model.get_input_embeddings().weight.shape[0]
    tok_vocab  = len(tokenizer)
    # Use the adapter embedding size if found; otherwise use tokenizer vocab size
    target_vocab = _adapter_emb_size if _adapter_emb_size is not None else tok_vocab
    if base_vocab != target_vocab:
        if target_vocab > base_vocab:
            print(f"[export] Resizing token embeddings: {base_vocab} → {target_vocab} "
                  f"(adapter added new special tokens)")
        else:
            print(f"[export] Resizing token embeddings: {base_vocab} → {target_vocab} "
                  f"(matching trained adapter embedding shape)")
        base_model.resize_token_embeddings(target_vocab)
    else:
        print(f"[export] Token embedding size matches adapter ({target_vocab}) ✓")

    # ── Load & merge LoRA ─────────────────────────────────────────────────────
    print(f"[export] Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, lora_dir, device_map=_merge_device_map)

    print(f"[export] Merging LoRA weights into base model — please wait...", flush=True)
    # merge_and_unload() folds the adapter deltas into the base weights in-place
    # and returns a plain nn.Module — the adapter tensors are freed immediately.
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    print(f"[export] LoRA merge complete ✓", flush=True)

    # Explicitly free the PeftModel wrapper and base_model reference so Python
    # can reclaim that memory before the (large) save step below.
    del peft_model, base_model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    # ── Move merged model to CPU before saving ────────────────────────────────
    # save_pretrained() serializes tensors to CPU regardless of where they live.
    # If the model is on GPU, each tensor transfer holds the GIL while waiting
    # for CUDA — combined with a background thread this can deadlock the save.
    # Moving to CPU first lets save_pretrained() work entirely in RAM.
    if cuda and _merge_device_map == "cuda:0":
        print(f"[export] Moving model to CPU for serialization...", flush=True)
        merged_model = merged_model.cpu()
        torch.cuda.empty_cache()

    # ── Save merged model ─────────────────────────────────────────────────────
    try:
        # Remove only SafeTensors/config from a previous export — keep existing
        # .gguf files so builds for different sizes/quant types are preserved.
        # e.g. popai_7b_q4_k_m.gguf and popai_7b_q8_0.gguf coexist in the same folder.
        if os.path.isdir(export_dir):
            _st_exts = {".safetensors", ".bin", ".pt"}
            _st_names = {
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "tokenizer.model", "special_tokens_map.json", "generation_config.json",
                "chat_template.jinja", "model.safetensors.index.json",
            }
            _removed_old = []
            for _fn in os.listdir(export_dir):
                _fp = os.path.join(export_dir, _fn)
                if not os.path.isfile(_fp):
                    continue
                _ext = os.path.splitext(_fn)[1].lower()
                if _ext in _st_exts or _fn in _st_names:
                    try:
                        os.remove(_fp)
                        _removed_old.append(_fn)
                    except Exception:
                        pass
            if _removed_old:
                print(f"[export] Cleared {len(_removed_old)} stale SafeTensors/config files from {export_dir}")

        os.makedirs(export_dir, exist_ok=True)
        print(f"[export] Saving merged model → {export_dir}", flush=True)
        print(f"[export] Writing ~15 GB to disk — this takes several minutes, please wait...", flush=True)
        print(f"[export] *** Watch this folder for growing files: {export_dir} ***", flush=True)

        # Use 500 MB shards so ~30 small files appear one-by-one and you can
        # watch them grow in Explorer.  The default 5 GB shard writes a single
        # huge temp file that looks like nothing is happening for many minutes.
        merged_model.save_pretrained(
            export_dir,
            safe_serialization=True,
            max_shard_size="500MB",
        )
        tokenizer.save_pretrained(export_dir)
        print(f"[export] Model saved ✓", flush=True)

        # ── Write a corrected generation_config.json ──────────────────────────
        # Qwen2.5 ChatML uses <|im_end|> (token 151645) as the turn-end token,
        # but the base model's saved generation_config has eos_token_id=151643
        # (<|endoftext|>) only — so <|im_end|> is never treated as a stop and
        # the model generates garbage forever.  Overwrite with both IDs and
        # sensible sampling defaults so the exported model works out of the box.
        import json as _json
        gen_cfg_path = os.path.join(export_dir, "generation_config.json")
        gen_cfg = {
            "bos_token_id": 151643,
            "do_sample": True,
            "eos_token_id": [151643, 151645],   # <|endoftext|> + <|im_end|>
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
            "temperature": 0.3,
            "top_p": 0.9,
            "transformers_version": "5.7.0",
        }
        with open(gen_cfg_path, "w", encoding="utf-8") as _f:
            _json.dump(gen_cfg, _f, indent=2)
        print(f"[export] generation_config.json written with correct EOS tokens.")

        # Free the merged model once saved — helps if further steps follow
        del merged_model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        # Also release torch's internal memory allocator cache so the OS
        # can reclaim RAM before the GGUF converter subprocess loads the
        # SafeTensors back into memory.
        try:
            if cuda:
                # torch.cuda.memory.empty_cache() clears the CUDA allocator cache
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # Force CPython to release any remaining tensor memory pools
            import ctypes
            if sys.platform == "win32":
                ctypes.windll.ucrtbase._heapmin()
        except Exception:
            pass
        print(f"[export] Memory freed — ready for GGUF conversion.", flush=True)
    except Exception:
        # Restore power settings if save crashes, then re-raise
        _restore_sleep()
        raise

    # ── Size on disk (SafeTensors) ────────────────────────────────────────────
    size_gb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(export_dir)
        for fn in fns
    ) / 1e9

    # ── Convert to GGUF ───────────────────────────────────────────────────────
    # Ollama and LM Studio require a single .gguf file, not a HuggingFace
    # SafeTensors directory.  We convert here automatically so the exported
    # model can be loaded by either tool without any manual steps.
    #
    # Quantization choices (set via --gguf-type flag, default: q4_k_m):
    #   q4_k_m  ~  4 GB  — good quality, smallest file, runs on any hardware  ← default
    #   q8_0    ~  8 GB  — near-lossless, use if you have 8+ GB VRAM
    #   f16     ~ 14 GB  — lossless fp16, requires 16+ GB VRAM to run
    #   q4_0    ~  4 GB  — fastest but lowest quality
    gguf_path = None
    # Preserve original model name casing (e.g. "TestAI"), only strip path-illegal chars.
    # Quantization suffix is lowercased (e.g. "q4_k_m").
    # Result: TestAI-7b-q4_k_m-qwen2.5.gguf
    safe_name_for_file = _re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', AI_NAME).strip('. ').lower()
    if not safe_name_for_file:
        safe_name_for_file = "model"
    safe_name_lower = _re.sub(r'[^a-z0-9_\-]', '-', AI_NAME.lower()).strip('-')  # kept for ollama tag only
    gguf_type_suffix = _re.sub(r'[^a-z0-9_]', '_', (args.gguf_type or "f16").lower())

    # ── Determine model size for the GGUF filename ────────────────────────────
    # Strategy 1: read the saved config.json (always present after save_pretrained)
    #   and compute total parameters from architecture dimensions.
    # Strategy 2: fall back to parsing the model ID string if config is unreadable.
    # This produces a self-describing filename like popai_7b_q4_k_m.gguf.
    def _size_tag_from_config(cfg_path):
        """Return size tag (e.g. '1.4b_') from the exported model's config.json.

        Used only as a fallback when the HF cache is unavailable.
        Reads the saved config.json from the export directory to count
        params from architecture dimensions. Less accurate than safetensors
        but works when the model hasn't been fully cached yet.
        """
        import json as _json
        try:
            with open(cfg_path, encoding="utf-8") as _f:
                cfg = _json.load(_f)
        except Exception:
            return None
        vocab  = cfg.get("vocab_size", 0)
        hidden = cfg.get("hidden_size", 0)
        layers = cfg.get("num_hidden_layers", 0)
        inter  = cfg.get("intermediate_size", 0)
        heads  = cfg.get("num_attention_heads", 0)
        kv_h   = cfg.get("num_key_value_heads", heads)
        hd     = cfg.get("head_dim", hidden // heads if heads else 0)
        if not (hidden and layers):
            return None
        q  = hidden * (heads * hd) if heads and hd else hidden * hidden
        kv = hidden * (kv_h * hd) if kv_h and hd else hidden * hidden
        attn = q + 2 * kv + q
        arch = (cfg.get('architectures') or [''])[0]
        is_neox  = bool(cfg.get('use_parallel_residual'))
        is_phi12 = 'PhiForCausalLM' in arch
        mf   = 2 if (is_neox or is_phi12) else 3
        mlp  = mf * hidden * inter if inter else 2 * hidden * hidden
        embc = 2 if (is_neox or is_phi12) else 1
        total = (attn + mlp + 2 * hidden) * layers + embc * vocab * hidden
        if total < 1e6:
            return None
        b = total / 1e9
        if b >= 1000:
            return f"{int(b // 1000)}t_"
        r = round(b, 1)
        tag = f"{int(r)}b" if r == int(r) else f"{r}b"
        return tag + '_'

    # ── Determine model family and size for the GGUF filename ───────────────────
    # Import the canonical implementations from server.py — single source of truth.
    # Both server.py (pipeline checker) and generate_llm.py (exporter) use the
    # same functions so the filename built and the filename checked are identical.
    import sys as _sys
    _train_dir = os.path.dirname(os.path.abspath(__file__))
    if _train_dir not in _sys.path:
        _sys.path.insert(0, _train_dir)
    from server import _get_model_family as _get_model_family_local
    from server import _get_model_size_tag as _get_model_size_tag_from_server

    def _get_model_size_tag_local(base_model_id, export_dir):
        """Thin wrapper: try server.py's safetensors-based size tag first,
        fall back to config.json dimension formula if model not cached."""
        tag = _get_model_size_tag_from_server(base_model_id)
        if tag:
            return tag
        # Fallback: use the exported config.json + dimension formula
        _cfg_path = os.path.join(export_dir, 'config.json')
        return (_size_tag_from_config(_cfg_path) or '').rstrip('_') or None

    _family  = _get_model_family_local(BASE_MODEL)
    _size_str = _get_model_size_tag_local(BASE_MODEL, export_dir) or ''
    if _size_str:
        print(f"[export] Model size (from config) : {_size_str.upper()}", flush=True)
        print(f"[export] Model family             : {_family}", flush=True)
    else:
        print(f"[export] Model family             : {_family}", flush=True)

    _family_suffix = f"-{_family}" if _family else ''
    if _size_str:
        gguf_filename = f"{safe_name_for_file}-{_size_str}-{gguf_type_suffix}{_family_suffix}.gguf"
    else:
        gguf_filename = f"{safe_name_for_file}-{gguf_type_suffix}{_family_suffix}.gguf"
    gguf_out = os.path.join(export_dir, gguf_filename)

    if args.do_gguf:
        print()
        print(f"[export] Converting model to GGUF format ({args.gguf_type}) for Ollama / LM Studio...", flush=True)
        print(f"[export] Note: RAM will spike again as the converter reads the SafeTensors back", flush=True)
        print(f"[export]       into memory — this is normal and expected during GGUF conversion.", flush=True)
        gguf_path = _convert_to_gguf(export_dir, gguf_out, quantization=args.gguf_type)

        if gguf_path:
            gguf_size_gb = os.path.getsize(gguf_path) / 1e9
        else:
            print("[export] ⚠ GGUF conversion failed — SafeTensors files kept.")
            print("[export]   You can convert manually later with:")
            print(f"[export]   python tools/convert_hf_to_gguf.py {export_dir} --outfile {gguf_out} --outtype {args.gguf_type}")
            gguf_size_gb = 0.0
    else:
        print()
        print("[export] GGUF conversion skipped (--no-gguf). Ollama/LM Studio require a .gguf file.")
        gguf_size_gb = 0.0

    # ── Patch GGUF metadata — bake identity directly into the file ────────────
    # This writes general.name, general.author, general.description, and
    # tokenizer.chat_template (with system prompt as default) into the GGUF
    # metadata block so the file is self-contained with no Modelfile required.
    # Anyone receiving just the .gguf gets the full identity embedded.
    if gguf_path and os.path.isfile(gguf_path):
        _patch_gguf_metadata(
            gguf_path,
            model_name    = AI_NAME,
            system_prompt = _sys_prompt,
            base_model    = BASE_MODEL,
            ai_owner      = AI_OWNER,
            ai_company    = AI_COMPANY,
        )

    # ── Write Modelfile and register with Ollama BEFORE cleanup ───────────────
    # Must happen before SafeTensors are deleted so generation_config.json is
    # still present when _write_modelfile reads temperature/top_p from it.
    _install_to_local_runtimes(AI_NAME, export_dir, gguf_path=gguf_path, system_prompt=_sys_prompt)

    # ── Delete SafeTensors unless --keep-safetensors was passed ───────────────
    # By default the .gguf is all you need — it is fully self-contained.
    # SafeTensors are only required for chat_test/server.py (the Python tester).
    safetensor_deleted_gb = 0.0
    delete_safetensors = args.do_gguf and gguf_path and not args.keep_safetensors
    if delete_safetensors:
        print()
        print("[export] Removing SafeTensors files (keep only the .gguf)...")
        print("[export] Use --keep-safetensors to retain them for chat_test/server.py")
        # Keep only the .gguf files and all .ollama Modelfiles
        keep_exts = {".gguf", ".ollama"}
        removed = []
        for fname in os.listdir(export_dir):
            fpath = os.path.join(export_dir, fname)
            if os.path.isfile(fpath):
                # Keep .gguf files, .ollama Modelfiles, and the legacy generic Modelfile
                if os.path.splitext(fname)[1] in keep_exts:
                    continue
                if fname == "Modelfile":
                    continue
                try:
                    fsize = os.path.getsize(fpath)
                    os.remove(fpath)
                    safetensor_deleted_gb += fsize / 1e9
                    removed.append(fname)
                except Exception as e:
                    print(f"[export] ⚠ Could not remove {fname}: {e}")
        if removed:
            print(f"[export] Freed {safetensor_deleted_gb:.1f} GB ({len(removed)} files removed)")

    # ── LoRA adapter is kept permanently as a pipeline artifact ──────────────
    # The popai_lora/{model}/ folder is NOT deleted after export.
    # It serves as the permanent record that Step 2 (training) is complete,
    # and lets the pipeline_state endpoint correctly show training_done=True.
    # Users who want to reclaim the disk space can delete it manually.

    # ── Restore power settings now that all work is done ─────────────────────
    _restore_sleep()

    # Derive the actual Ollama tag from the GGUF filename (matches what _install_to_local_runtimes used)
    # e.g. "popai_7b_q4_k_m.gguf" → "popai-7b-q4-k-m"
    if gguf_path:
        gguf_stem  = os.path.splitext(os.path.basename(gguf_path))[0]
        ollama_tag = _re.sub(r'[^a-z0-9_\-\.]', '-', gguf_stem.lower()).strip('-')
    else:
        ollama_tag = _re.sub(r'[^a-z0-9_\-\.]', '-', AI_NAME.lower()).strip('-')

    # Emit a machine-readable tag line so the Studio UI can show the correct
    # 'ollama run <tag>' command in the result bar (instead of guessing from AI_NAME).
    print(f"[export] OLLAMA_TAG: {ollama_tag}", flush=True)

    print()
    print("=" * 60)
    print(f"  Export complete!")
    print(f"  {AI_NAME} model saved to : {export_dir}")
    if gguf_path:
        # Use the actual returned path — may differ from gguf_out if we fell back to f16
        actual_gguf_filename = os.path.basename(gguf_path)
        actual_gguf_size_gb  = os.path.getsize(gguf_path) / 1e9
        print(f"  GGUF file (shareable)  : {actual_gguf_filename}  ({actual_gguf_size_gb:.1f} GB)")
        print(f"  Ollama model name      : {ollama_tag}")
        print(f"  ✅ This is ONE self-contained file — weights + tokenizer + config.")
        print(f"     Share just this file. Recipients need only Ollama or LM Studio.")
        if delete_safetensors:
            print(f"  SafeTensors deleted    : freed {safetensor_deleted_gb:.1f} GB")
        else:
            print(f"  SafeTensors also kept  : {size_gb:.1f} GB  (for chat_test/server.py)")
        print()
        print(f"  → Load in Ollama       : ollama run {ollama_tag}")
        print(f"  → Load in LM Studio    : open {gguf_path}")
    else:
        print(f"  SafeTensors size       : {size_gb:.1f} GB")
        print(f"  ⚠ No .gguf produced — Ollama/LM Studio cannot load SafeTensors directly.")
        if args.do_gguf:
            print(f"  To convert manually (use q8_0 — directly supported):")
            print(f"  python tools/convert_hf_to_gguf.py {export_dir} --outfile <output>.gguf --outtype q8_0")
    print()
    if args.keep_safetensors or not args.do_gguf:
        print("  Launch the Python chat tester:")
        print("      python chat_test\\server.py")
        print("  Then open: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
