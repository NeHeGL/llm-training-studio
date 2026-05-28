"""
LLM Training Studio - Server (NeHe Productions)
Author: Jeff Molofee (aka NeHe) — 2026
AI Assistance: Cline (Claude Sonnet 4.5)

Serves studio.html and provides API to manage training examples + build dataset files.

Usage:
    python train/server.py          # starts on port 5001, opens browser

API:
    GET  /examples          — list all training examples
    POST /examples          — save all training examples (replaces file)
    POST /build             — write dataset/train_chatml.jsonl + dataset/train_alpaca.jsonl
    GET  /stats             — dataset statistics
"""

import json
import os
import random
import subprocess
import sys
import threading
import time
import webbrowser

# ── Consolidate __pycache__ at project root ───────────────────────────────────
# This mirrors what launch_web.bat / launch_app.bat do via PYTHONPYCACHEPREFIX.
# Setting sys.pycache_prefix here ensures the cache goes to the root even when
# scripts are run directly (e.g. from VS Code or the command line).
if not sys.pycache_prefix:
    sys.pycache_prefix = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__pycache__"
    )

# Only re-wrap stdio when running as a standalone script, not when imported as a module.
if sys.platform == "win32" and __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _check_and_install_dependencies():
    """First-run dependency check.

    Verifies that all packages in requirements.txt are importable.
    For any that are missing, auto-installs them via pip so the server
    starts correctly even after a fresh clone with no manual pip install step.

    PyTorch is NOT auto-installed here (it needs a CUDA-specific index URL).
    If torch is missing, we print a clear message pointing the user to install.bat.
    """  # noqa: D401
    import importlib

    # Map package names → import names (they differ for some packages)
    IMPORT_MAP = {
        "flask":           "flask",
        "flask-cors":      "flask_cors",
        "psutil":          "psutil",
        "transformers":    "transformers",
        "peft":            "peft",
        "accelerate":      "accelerate",
        "bitsandbytes":    "bitsandbytes",
        "datasets":        "datasets",
        "trl":             "trl",
        "sentencepiece":   "sentencepiece",
        "safetensors":     "safetensors",
        "huggingface_hub": "huggingface_hub",
        "gguf":            "gguf",
        "requests":        "requests",
        "tqdm":            "tqdm",
    }

    # Packages that must be present for the server to start at all
    CRITICAL = {"flask", "flask-cors", "psutil"}

    missing = []
    for pkg, import_name in IMPORT_MAP.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return  # all good

    print()
    print("=" * 60)
    print("  [setup] First-run dependency check")
    print("=" * 60)
    print(f"  Missing packages: {', '.join(missing)}")
    print()

    # PyTorch isn't in requirements.txt — handle separately
    torch_missing = False
    try:
        importlib.import_module("torch")
    except ImportError:
        torch_missing = True

    if torch_missing:
        print("  [setup] PyTorch is not installed.")
        print("          Run install.bat to auto-detect your CUDA version")
        print("          and install the correct PyTorch build.")
        print()

    # Auto-install non-torch missing packages from requirements.txt
    non_torch = [p for p in missing if p.lower() not in ("torch", "torchvision", "torchaudio")]
    if non_torch:
        print(f"  [setup] Auto-installing: {', '.join(non_torch)}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + non_torch,
                check=False,
            )
            if result.returncode == 0:
                print("  [setup] ✓ Packages installed successfully")
            else:
                print("  [setup] ✗ Some packages failed to install.")
                print("          Run: pip install -r requirements.txt")
        except Exception as e:
            print(f"  [setup] ✗ Auto-install failed: {e}")
            print("          Run: pip install -r requirements.txt")

    # Check critical packages are now available
    still_missing_critical = []
    for pkg in CRITICAL:
        import_name = IMPORT_MAP.get(pkg, pkg)
        try:
            importlib.import_module(import_name)
        except ImportError:
            still_missing_critical.append(pkg)

    if still_missing_critical:
        print()
        print(f"  [setup] CRITICAL packages still missing: {', '.join(still_missing_critical)}")
        print("          The server cannot start without them.")
        print("          Run: pip install -r requirements.txt")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print()

# Run dependency check before anything else (only when run as main script)
if __name__ == "__main__":
    _check_and_install_dependencies()

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT  = os.path.join(ROOT, "models")
CONFIG_FILE  = os.path.join(ROOT, "config.json")
MODEL_CONFIG = "config.json"   # per-model config filename inside models/{Name}/


# ── Per-model directory helpers ───────────────────────────────────────────────
# All build artifacts for a model live under models/{ModelName}/:
#   models/{ModelName}/knowledge/   ← source Q&A knowledge files
#   models/{ModelName}/dataset/     ← built JSONL training files
#   models/{ModelName}/lora/        ← LoRA adapter subdirs
#   models/{ModelName}/gguf/        ← exported GGUF + Modelfile


def _model_root(name):
    """Return the root directory for a named model: models/{SafeName}/"""
    return os.path.join(MODELS_ROOT, _safe_name(name))


def get_dataset_dir():
    """Return the per-model dataset directory: models/{ModelName}/dataset/"""
    return os.path.join(_model_root(get_active_name()), "dataset")


# ── Model Catalog ─────────────────────────────────────────────────────────────
# Each entry: model_id, params_b (billions), min_vram_4bit, min_vram_fp16,
#             quant ("4bit"|"fp16"|"fp32"), bf16_ok, description

MODEL_CATALOG = [
    # ── Sub-1B ────────────────────────────────────────────────────────────────
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "params_b": 0.5,
        "min_vram_4bit": 1,
        "min_vram_fp16": 2,
        "description": "Smallest Qwen — only for very limited VRAM. Limited knowledge.",
        "tag": "sub-1B",
    },
    # ── 1–2B ──────────────────────────────────────────────────────────────────
    {
        "id": "Qwen/Qwen2.5-1.5B",
        "params_b": 1.5,
        "min_vram_4bit": 2,
        "min_vram_fp16": 4,
        "description": "Qwen 1.5B — slightly more capable than average 1B models.",
        "tag": "1.5B",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B",
        "params_b": 1.7,
        "min_vram_4bit": 2,
        "min_vram_fp16": 4,
        "description": "SmolLM2 1.7B — Apache 2.0, no gating. Efficient and capable for its size.",
        "tag": "1.7B",
    },
    {
        "id": "EleutherAI/pythia-1.4b",
        "params_b": 1.4,
        "min_vram_4bit": 2,
        "min_vram_fp16": 4,
        "description": "Pythia 1.4B — Apache 2.0. Rock-solid GPT-NeoX architecture, no custom code, works everywhere.",
        "tag": "1.4B",
    },
    # ── 2–3B ──────────────────────────────────────────────────────────────────
    {
        "id": "microsoft/phi-2",
        "params_b": 2.7,
        "min_vram_4bit": 2,
        "min_vram_fp16": 6,
        "description": "Microsoft Phi-2 — MIT license. Trained on high-quality 'textbook' data. Strong reasoning for its size.",
        "tag": "2.7B",
    },
    {
        "id": "Qwen/Qwen2.5-3B",
        "params_b": 3,
        "min_vram_4bit": 3,
        "min_vram_fp16": 7,
        "description": "Qwen 3B — good balance of size and capability for fine-tuning.",
        "tag": "3B",
    },
    # ── 7B ────────────────────────────────────────────────────────────────────
    {
        "id": "Qwen/Qwen2.5-7B",
        "params_b": 7,
        "min_vram_4bit": 5,
        "min_vram_fp16": 15,
        "description": "Qwen 7B — recommended for 16 GB VRAM. Excellent quality with 4-bit QLoRA.",
        "tag": "7B ⭐",
    },
    {
        "id": "mistralai/Mistral-7B-v0.1",
        "params_b": 7,
        "min_vram_4bit": 5,
        "min_vram_fp16": 15,
        "description": "Mistral 7B v0.1 — Apache 2.0, no gating. Fast inference, strong instruction following.",
        "tag": "7B",
    },
    {
        "id": "EleutherAI/pythia-6.9b",
        "params_b": 6.9,
        "min_vram_4bit": 5,
        "min_vram_fp16": 15,
        "description": "Pythia 6.9B — Apache 2.0. Standard GPT-NeoX architecture, no custom code, very reliable for fine-tuning.",
        "tag": "6.9B",
    },
    # ── 14B ───────────────────────────────────────────────────────────────────
    {
        "id": "Qwen/Qwen2.5-14B",
        "params_b": 14,
        "min_vram_4bit": 9,
        "min_vram_fp16": 29,
        "description": "Qwen 14B — excellent for 24 GB cards. Significantly better than 7B.",
        "tag": "14B",
    },
    # ── 32B ───────────────────────────────────────────────────────────────────
    {
        "id": "Qwen/Qwen2.5-32B",
        "params_b": 32,
        "min_vram_4bit": 20,
        "min_vram_fp16": 65,
        "description": "Qwen 32B — near GPT-4 quality. Needs 24+ GB for 4-bit.",
        "tag": "32B",
    },
    # ── 72B ───────────────────────────────────────────────────────────────────
    {
        "id": "Qwen/Qwen2.5-72B",
        "params_b": 72,
        "min_vram_4bit": 42,
        "min_vram_fp16": 145,
        "description": "Qwen 72B — top open-source quality. Multi-GPU or cloud only.",
        "tag": "72B",
    },
]


# ── Multi-model Config I/O ────────────────────────────────────────────────────
#
# Config architecture (v2):
#   config.json (root)          — lightweight index: {"active": "TestAI", "models": ["PopAI", "TestAI"]}
#   models/{Name}/config.json   — full per-model config: name, base_model, system_prompt, train_settings
#
# This keeps each model self-contained in its own folder. Copying models/{Name}/
# to another machine is all that's needed to transfer a model's full setup.
#
# Backward compatibility: if the root config.json still contains the old
# "models": [{...}] list format, _migrate_config() converts it on first load.


def _safe_name(name):
    """Sanitize model name for use in folder/file paths."""
    import re
    return re.sub(r'[^\w\-]', '_', name.strip())


def _model_knowledge_dir(name):
    """Source knowledge files: models/{Name}/knowledge/"""
    return os.path.join(MODELS_ROOT, _safe_name(name), "knowledge")


def _model_lora_dir(name):
    """LoRA adapter directory: models/{Name}/lora/"""
    return os.path.join(MODELS_ROOT, _safe_name(name), "lora")


def _model_gguf_dir(name):
    """Exported GGUF directory: models/{Name}/gguf/"""
    return os.path.join(MODELS_ROOT, _safe_name(name), "gguf")


def _model_manual_file(name):
    """Manual Q&A entries: models/{Name}/knowledge/manual/manual_entries.txt"""
    return os.path.join(_model_knowledge_dir(name), "manual", "manual_entries.txt")


# ── Model identity helpers (read from HuggingFace cached config.json) ─────────
#
# These three functions are the single source of truth for:
#   1. The base model family name  (used in the GGUF filename suffix)
#   2. The model size tag          (used in the GGUF filename)
#   3. The complete GGUF filename  (used by both exporter and pipeline checker)
def _get_model_family(base_model_id):
    """Return the base model family string derived from the HuggingFace model name.
    Uses the same logic as generate_llm.py _model_family() to ensure the
    family suffix in GGUF filenames is consistent between exporter and checker.
    Examples:
        "microsoft/phi-2"             -> "phi-2"
        "Qwen/Qwen2.5-7B"            -> "qwen2.5"
        "EleutherAI/pythia-1.4b"     -> "pythia"
        "mistralai/Mistral-7B-v0.1"  -> "mistral"
        "HuggingFaceTB/SmolLM2-1.7B" -> "smollm2"
    """
    import re as _re
    # Take the last part of the HF model ID (after the /)
    name = base_model_id.split('/')[-1] if '/' in base_model_id else base_model_id
    # Strip trailing size tag and everything after it: -7B, -1.4b, -0.5B, etc.
    name = _re.sub(r'[-_]?\d+\.?\d*[Bb][\w.]*.*$', '', name)
    # Strip trailing noise: -instruct, -chat, -hf, -base, -it, -v0.1, etc.
    name = _re.sub(r'[-_]?(instruct|chat|hf|base|it|v\d[\w.]*)$', '', name, flags=_re.IGNORECASE)
    # Lowercase, underscores to dashes, collapse repeated dashes
    # Preserve dots so "Qwen2.5" -> "qwen2.5" (not "qwen2-5")
    name = name.lower().replace('_', '-')
    name = _re.sub(r'-+', '-', name).strip('-')
    return name
def _get_model_size_tag(base_model_id):
    """Return exact size tag (e.g. '1.4b', '7.6b') for a HuggingFace model ID.
    Reads the actual floating-point parameter count from the HF-cached
    safetensors shard headers — no architecture assumptions, no fudging.
    Only F16/BF16/F32/F64 tensors are counted; U8 attention masks and
    I32 index tensors are excluded as they are not model parameters.
    Supports both single-file (model.safetensors) and sharded models
    (model.safetensors.index.json + individual shards).
    Returns a lowercase tag like '1.4b', '7.6b', '0.5b', or None if
    the model is not cached locally.
    """
    import json as _json
    import struct as _struct
    _FLOAT_DTYPES = {'F16', 'BF16', 'F32', 'F64'}
    def _is_valid_path(p):
        return isinstance(p, str) and p not in ('not_cached', '')
    def _count_shard(path):
        count = 0
        with open(path, 'rb') as _f:
            _hlen = _struct.unpack('<Q', _f.read(8))[0]
            _hdr  = _json.loads(_f.read(_hlen))
        for _name, _info in _hdr.items():
            if _name == '__metadata__':
                continue
            if _info.get('dtype') not in _FLOAT_DTYPES:
                continue
            _shape = _info.get('shape', [])
            if _shape:
                _p = 1
                for _d in _shape:
                    _p *= _d
                count += _p
        return count
    def _params_to_tag(n):
        b = n / 1e9
        if b >= 1000:
            return f"{int(b // 1000)}t"
        r = round(b, 1)
        return f"{int(r)}b" if r == int(r) else f"{r}b"
    try:
        from huggingface_hub import try_to_load_from_cache, hf_hub_download
        # ── Strategy 1: model already cached locally ──────────────────────────
        # Single-file model
        sf = try_to_load_from_cache(base_model_id, 'model.safetensors')
        if _is_valid_path(sf):
            n = _count_shard(sf)
            if n > 0:
                return _params_to_tag(n)
        # Sharded model — read all shard headers from cache
        idx = try_to_load_from_cache(base_model_id, 'model.safetensors.index.json')
        if _is_valid_path(idx):
            with open(idx) as _f:
                _idx = _json.load(_f)
            shards = set(_idx.get('weight_map', {}).values())
            total = 0
            for shard in sorted(shards):
                sp = try_to_load_from_cache(base_model_id, shard)
                if _is_valid_path(sp):
                    total += _count_shard(sp)
            if total > 0:
                return _params_to_tag(total)
        # ── Strategy 2: model not cached — download index.json only (~50 KB) ──
        # The safetensors index file lists every tensor name → shard filename.
        # Its metadata.total_size field is the total byte size of all tensors.
        # Dividing by 2 gives the param count (assumes BF16, valid for all
        # modern models in the catalog). This downloads ~50 KB, not 15 GB.
        try:
            _idx_path = hf_hub_download(base_model_id, 'model.safetensors.index.json')
            with open(_idx_path) as _f:
                _idx2 = _json.load(_f)
            _total_bytes = int(_idx2.get('metadata', {}).get('total_size', 0))
            if _total_bytes > 0:
                # BF16 = 2 bytes per param (all catalog models use BF16)
                return _params_to_tag(_total_bytes // 2)
        except Exception:
            pass
        # ── Strategy 3: single-file model not cached — try downloading it ─────
        # (rare — most modern models are sharded; this covers small models like
        # pythia-1.4b which store weights in one file)
        try:
            _sf_path = hf_hub_download(base_model_id, 'model.safetensors')
            if _sf_path and os.path.isfile(_sf_path):
                n = _count_shard(_sf_path)
                if n > 0:
                    return _params_to_tag(n)
        except Exception:
            pass
    except Exception:
        pass
    return None

def _build_gguf_filename(model_name, base_model_id, gguf_type):
    """Build the exact GGUF filename that generate_llm.py produces.
    Format: {ModelName}_{size}_{quant}-{family}.gguf
    Example: PopAI_2.7b_q4_k_m-phi.gguf
    Single source of truth: both the exporter and the pipeline state checker
    call this function so the filename built and checked are always identical.
    """
    import re as _re
    # Strip path-illegal chars from model name and lowercase (mirrors generate_llm.py safe_name_for_file)
    safe_name = _re.sub(r'[<>:"/\\|?*]', '_', model_name).strip('. ').lower() or 'model'
    # quant suffix: lowercase, non-alnum/underscore -> underscore
    resolved_type = 'q4_k_m' if gguf_type in (None, 'auto') else gguf_type
    quant_suffix  = _re.sub(r'[^a-z0-9_]', '_', resolved_type.lower())
    # family and size from HF cached config
    family        = _get_model_family(base_model_id)
    size_tag      = _get_model_size_tag(base_model_id)
    family_suffix = f"-{family}" if family else ""
    if size_tag:
        return f"{safe_name}-{size_tag}-{quant_suffix}{family_suffix}.gguf"
    else:
        return f"{safe_name}-{quant_suffix}{family_suffix}.gguf"


def _model_config_file(name):
    """Per-model config file: models/{Name}/config.json"""
    return os.path.join(MODELS_ROOT, _safe_name(name), MODEL_CONFIG)


# ── Per-model config read/write ───────────────────────────────────────────────


def load_model_config(name):
    """Read models/{Name}/config.json. Returns a dict with model settings."""
    path = _model_config_file(name)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[studio] Warning: could not read {path}: {e}")
    # Defaults when no per-model config exists yet
    return {
        "name": name,
        "base_model": "Qwen/Qwen2.5-7B",
        "system_prompt": f"You are {name}, a helpful AI assistant.",
    }


def save_model_config(model_data):
    """Write a model's settings to models/{Name}/config.json."""
    name = model_data.get("name", "")
    if not name:
        return
    path = _model_config_file(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)
    print(f"[studio] Model config saved → {path}")


# ── Root config (index only) ──────────────────────────────────────────────────


def _rebuild_config_from_model_folders():
    """Scan models/ subfolders for config.json files and rebuild the root index.

    Called when root config.json is missing or unreadable. This means deleting
    config.json is safe — the server will automatically rediscover all models
    from their per-model config files and recreate the index.
    """
    print("[studio] Root config.json missing or unreadable — scanning models/ for per-model configs…")
    found_names = []
    if os.path.isdir(MODELS_ROOT):
        for entry in sorted(os.listdir(MODELS_ROOT)):
            model_cfg_path = os.path.join(MODELS_ROOT, entry, MODEL_CONFIG)
            if os.path.isfile(model_cfg_path):
                try:
                    with open(model_cfg_path, encoding="utf-8") as f:
                        m = json.load(f)
                    name = m.get("name", entry)
                    found_names.append(name)
                    print(f"[studio]   Found model: {name}")
                except Exception as e:
                    print(f"[studio]   Warning: could not read {model_cfg_path}: {e}")
    if found_names:
        cfg = {"active": found_names[0], "models": found_names}
        save_config(cfg)
        print(f"[studio] Rebuilt root config.json with models: {found_names}")
        return cfg
    # Truly fresh install — no models anywhere
    return None


def load_config():
    """Load the root config.json (index: active model + list of model names).

    Returns a dict: {"active": "TestAI", "models": ["PopAI", "TestAI"]}

    On first load, if the old all-in-one format is detected (models list contains
    dicts), _migrate_config() is called to split each model into its own file.

    If root config.json is missing or corrupt, _rebuild_config_from_model_folders()
    scans models/ subfolders and recreates the index automatically.
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                cfg = json.load(f)
            # Detect old format: "models" is a list of dicts
            if cfg.get("models") and isinstance(cfg["models"][0], dict):
                cfg = _migrate_config(cfg)
            return cfg
        except Exception as e:
            print(f"[studio] Warning: could not read config.json: {e}")
    # Root config missing or unreadable — try to rebuild from per-model folders
    rebuilt = _rebuild_config_from_model_folders()
    if rebuilt:
        return rebuilt
    # Truly fresh install — return a default so startup can create it
    return {"active": "TestAI", "models": ["TestAI"]}


def save_config(cfg):
    """Write the root config.json (index only — no model data)."""
    # Strip any accidentally included full model dicts from the models list
    models_val = cfg.get("models", [])
    if models_val and isinstance(models_val[0], dict):
        models_val = [m["name"] for m in models_val]
    index = {"active": cfg.get("active", ""), "models": models_val}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"[studio] Root config saved → {CONFIG_FILE}")


def _migrate_config(old_cfg):
    """One-time migration: split old all-in-one config.json into per-model files.

    Called automatically when load_config() detects the old format.
    Each model's full data is written to models/{Name}/config.json and the
    root config.json is rewritten as a lean index.
    """
    print("[studio] Migrating config.json → per-model config files…")
    model_names = []
    for m in old_cfg.get("models", []):
        name = m.get("name", "")
        if not name:
            continue
        # Write per-model config (only if one doesn't already exist)
        path = _model_config_file(name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)
            print(f"[studio]   → {path}")
        model_names.append(name)
    new_cfg = {"active": old_cfg.get("active", model_names[0] if model_names else ""), "models": model_names}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)
    print(f"[studio] Migration complete. Root config.json updated.")
    return new_cfg


def get_active_model(cfg=None):
    """Return the active model's full config dict (loaded from its per-model file)."""
    if cfg is None:
        cfg = load_config()
    active_name = cfg.get("active", "")
    if not active_name and cfg.get("models"):
        active_name = cfg["models"][0] if isinstance(cfg["models"][0], str) else cfg["models"][0].get("name", "")
    return load_model_config(active_name)


# ── Startup: init config + migrate if needed ──────────────────────────────────
_config = load_config()   # triggers migration if old format detected

# Write root config.json if it doesn't exist at all (very first run)
if not os.path.exists(CONFIG_FILE):
    # Create a default TestAI model
    _default_name = "TestAI"
    _default_model = {
        "name": _default_name,
        "base_model": "Qwen/Qwen2.5-7B",
        "system_prompt": f"You are {_default_name}, a helpful AI assistant.",
    }
    save_model_config(_default_model)
    _config = {"active": _default_name, "models": [_default_name]}
    save_config(_config)
    _default_kdir = _model_knowledge_dir(_default_name)
    os.makedirs(os.path.join(_default_kdir, "manual"), exist_ok=True)
    print(f"[studio] First run — created config files and training folder: {_default_kdir}")


def get_active_name():
    return load_config().get("active", "")


def get_knowledge_dir():
    return _model_knowledge_dir(get_active_name())


def get_manual_file():
    return _model_manual_file(get_active_name())


def get_system_prompt():
    return get_active_model().get("system_prompt", "You are a helpful AI assistant.")


# ── Manual entries I/O ────────────────────────────────────────────────────────


def load_manual():
    """Read manual_entries.txt and return list of {question, answer} dicts.

    Uses Q:/A: format. Both single-line and multi-line answers are supported.
    Pairs may be separated by blank lines OR run back-to-back with no gap.
    """
    manual_file = get_manual_file()
    if not os.path.exists(manual_file):
        return []
    import re
    with open(manual_file, encoding="utf-8") as f:
        content = f.read()
    pairs = []
    # Split on any newline immediately followed by Q:
    # Handles blank-line-separated pairs AND back-to-back pairs (no blank line)
    blocks = re.split(r'\n(?=Q:)', content.strip())
    for block in blocks:
        block = block.strip()
        if not block.startswith("Q:"):
            continue
        # re.DOTALL lets .+ span newlines so multi-line answers are fully captured
        m = re.match(r'^Q:\s*(.+?)\s*\nA:\s*(.+)$', block, re.DOTALL)
        if m:
            q = m.group(1).strip()
            a = m.group(2).strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
    return pairs


def save_manual(examples):
    """Write list of {question, answer} dicts to manual_entries.txt in Q:/A: format."""
    manual_file = get_manual_file()
    os.makedirs(os.path.dirname(manual_file), exist_ok=True)
    lines = []
    for ex in examples:
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()
        if q:
            lines.append(f"Q: {q}\nA: {a}")
    with open(manual_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
        if lines:
            f.write("\n")
    print(f"[studio] Saved {len(lines)} entries → {manual_file}")


# ── Dataset builder ────────────────────────────────────────────────────────────


def build_dataset(examples):
    """Write both ChatML and Alpaca format JSONL files into the per-model dataset dir."""
    dataset_dir = get_dataset_dir()
    os.makedirs(dataset_dir, exist_ok=True)

    system_prompt = get_system_prompt()
    chatml_records = []
    alpaca_records = []

    for ex in examples:
        q = ex.get("question", "").strip()
        a = ex.get("answer", "").strip()
        if not q or not a:
            continue

        chatml_records.append({
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": q},
                {"role": "assistant", "content": a},
            ]
        })
        alpaca_records.append({
            "instruction": q,
            "output":      a,
            "input":       "",
            "system":      system_prompt,
        })

    # Shuffle so training sees varied topics
    random.shuffle(chatml_records)
    random.shuffle(alpaca_records)

    chatml_path = os.path.join(dataset_dir, "train_chatml.jsonl")
    alpaca_path = os.path.join(dataset_dir, "train_alpaca.jsonl")

    with open(chatml_path, "w", encoding="utf-8") as f:
        for rec in chatml_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(alpaca_path, "w", encoding="utf-8") as f:
        for rec in alpaca_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[studio] Wrote {len(chatml_records)} records → {chatml_path}")
    print(f"[studio] Wrote {len(alpaca_records)} records → {alpaca_path}")
    return len(chatml_records)


# ── Flask app ──────────────────────────────────────────────────────────────────


def create_app():
    try:
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS
    except ImportError:
        print("[studio] Missing packages. Run:")
        print("  pip install flask flask-cors")
        sys.exit(1)

    app = Flask(__name__, static_folder=TRAIN_DIR)
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(TRAIN_DIR, 'studio.html')

    @app.route("/examples", methods=["GET"])
    def get_examples():
        return jsonify(load_manual())

    @app.route("/examples", methods=["POST"])
    def post_examples():
        data = request.get_json(force=True)
        if not isinstance(data, list):
            return jsonify({"error": "Expected a JSON array"}), 400
        # Validate each item
        clean = []
        for item in data:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer",   "")).strip()
            if q:
                clean.append({"question": q, "answer": a})
        save_manual(clean)
        return jsonify({"saved": len(clean)})

    @app.route("/build", methods=["POST"])
    def build():
        import shutil as _shutil
        if TRAIN_DIR not in sys.path:
            sys.path.insert(0, TRAIN_DIR)
        from import_knowledge import collect_all_files, parse_qa_file

        # ── Clear stale dataset files before writing fresh ones ───────────────
        # Only the JSONL output files are removed — not the source knowledge data.
        # This guarantees no old pairs bleed into the new build when knowledge
        # files have been added, removed, or edited since the last run.
        _dataset_dir = get_dataset_dir()
        if os.path.isdir(_dataset_dir):
            for _fn in os.listdir(_dataset_dir):
                if _fn.endswith(".jsonl"):
                    try:
                        os.remove(os.path.join(_dataset_dir, _fn))
                    except Exception:
                        pass
            print(f"[studio] Cleared stale JSONL files from {_dataset_dir}")

        all_pairs = []

        # 1. Scan the model's knowledge folder (e.g. models/MyModel/knowledge/)
        #    This is the primary source — it contains all knowledge files for this model.
        knowledge_dir = get_knowledge_dir()
        if os.path.isdir(knowledge_dir):
            for _rel, _abs in collect_all_files(knowledge_dir):
                # Skip manual_entries.txt — loaded separately below
                if os.path.basename(_abs) == "manual_entries.txt":
                    continue
                all_pairs.extend(parse_qa_file(_abs))

        # 2. Fall back to shared knowledge/ folder only if model has no knowledge files yet
        if not all_pairs:
            shared_knowledge_dir = os.path.join(ROOT, "knowledge")
            if os.path.isdir(shared_knowledge_dir):
                for _rel, _abs in collect_all_files(shared_knowledge_dir):
                    all_pairs.extend(parse_qa_file(_abs))

        # 3. Include manual entries
        manual = load_manual()
        all_pairs.extend(manual)

        # ── Sanity checks ─────────────────────────────────────────────────────
        warnings = []

        # Get seq_len from active model's train_settings (default 2048)
        try:
            cfg = load_config()
            active_model = get_active_model(cfg)
            seq_len = int(active_model.get("train_settings", {}).get("seq_len", 2048))
        except Exception:
            seq_len = 2048

        # Chars-per-token approximation: ~4 chars per token is a safe estimate
        chars_per_token = 4
        max_chars = seq_len * chars_per_token

        empty_q = 0
        empty_a = 0
        too_long = 0
        seen = {}          # key → first question text (truncated for display)
        dup_examples = []  # up to 5 example duplicate questions to show user
        duplicates = 0

        for pair in all_pairs:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()

            if not q:
                empty_q += 1
                continue
            if not a:
                empty_a += 1
                continue

            # Duplicate detection — normalize whitespace for comparison
            key = " ".join(q.lower().split())
            if key in seen:
                duplicates += 1
                if len(dup_examples) < 5:
                    dup_examples.append(q[:80] + ("…" if len(q) > 80 else ""))
            else:
                seen[key] = q

            # Sequence length check
            if len(q) + len(a) > max_chars:
                too_long += 1

        if empty_q > 0:
            warnings.append(f"{empty_q} pair(s) have an empty question and were skipped.")
        if empty_a > 0:
            warnings.append(f"{empty_a} pair(s) have an empty answer and were skipped.")
        if duplicates > 0:
            ex = "; ".join(f'"{e}"' for e in dup_examples)
            more = f" (and {duplicates - len(dup_examples)} more)" if duplicates > len(dup_examples) else ""
            warnings.append(f"{duplicates} duplicate question(s) detected — remove them to avoid overfitting. Examples: {ex}{more}")
        if too_long > 0:
            warnings.append(f"{too_long} pair(s) may exceed seq_len={seq_len} tokens (~{max_chars} chars) and could be truncated during training.")

        print(f"[studio] Sanity check: {len(all_pairs)} pairs, {duplicates} dupes, {empty_q} empty-Q, {empty_a} empty-A, {too_long} too-long")
        if dup_examples:
            print(f"[studio] Duplicate examples: {dup_examples}")

        count = build_dataset(all_pairs)
        _ddir = get_dataset_dir()
        return jsonify({
            "count":    count,
            "chatml":   os.path.join(_ddir, "train_chatml.jsonl"),
            "alpaca":   os.path.join(_ddir, "train_alpaca.jsonl"),
            "warnings": warnings,
        })

    @app.route("/stats", methods=["GET"])
    def stats():
        examples = load_manual()
        total = len(examples)
        if total == 0:
            return jsonify({"total": 0})
        avg_q = sum(len(e.get("question","")) for e in examples) // total
        avg_a = sum(len(e.get("answer",""))   for e in examples) // total
        return jsonify({"total": total, "avg_q_len": avg_q, "avg_a_len": avg_a})

    # Cache for knowledge stats so page loads don't re-scan 86 files every time
    _knowledge_cache = {}

    @app.route("/knowledge-stats", methods=["GET"])
    def knowledge_stats():
        """Fast cached knowledge stats — no re-scan on every page load."""
        try:
            active = get_active_name()
            if active in _knowledge_cache:
                return jsonify(_knowledge_cache[active])
            knowledge_dir = get_knowledge_dir()
            if TRAIN_DIR not in sys.path:
                sys.path.insert(0, TRAIN_DIR)
            from import_knowledge import collect_all_files, parse_qa_file
            if not os.path.isdir(knowledge_dir):
                result = {"added": 0, "files": 0, "total": 0, "avg_q": None, "avg_a": None}
            else:
                files = collect_all_files(knowledge_dir)
                all_pairs = []
                for _, abs_path in files:
                    all_pairs.extend(parse_qa_file(abs_path))
                manual = load_manual()
                all_combined = all_pairs + [{"question": e["question"], "answer": e["answer"]} for e in manual]
                avg_q = round(sum(len(p.get("question","")) for p in all_combined) / len(all_combined)) if all_combined else None
                avg_a = round(sum(len(p.get("answer","")) for p in all_combined) / len(all_combined)) if all_combined else None
                result = {"added": len(all_pairs), "files": len(files), "total": len(all_combined), "avg_q": avg_q, "avg_a": avg_a}
            _knowledge_cache[active] = result
            print(f"[studio] Knowledge scan (cached): {result['added']} pairs across {result['files']} files")
            return jsonify(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/import-knowledge", methods=["POST"])
    def import_knowledge_route():
        try:
            knowledge_dir = get_knowledge_dir()
            if TRAIN_DIR not in sys.path:
                sys.path.insert(0, TRAIN_DIR)
            from import_knowledge import collect_all_files, parse_qa_file
            if not os.path.isdir(knowledge_dir):
                return jsonify({"added": 0, "files": 0, "total": 0})
            files = collect_all_files(knowledge_dir)
            total_pairs = sum(len(parse_qa_file(abs_path)) for _, abs_path in files)
            manual = load_manual()
            # Invalidate cache so next GET re-scans
            active = get_active_name()
            _knowledge_cache.pop(active, None)
            print(f"[studio] Knowledge scan: {total_pairs} pairs across {len(files)} files")
            return jsonify({"added": total_pairs, "files": len(files), "total": total_pairs + len(manual)})
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/models", methods=["GET"])
    def get_models():
        cfg = load_config()
        # cfg["models"] is now a list of name strings
        return jsonify({"active": cfg.get("active", ""), "models": cfg.get("models", [])})

    @app.route("/models/select", methods=["POST"])
    def select_model():
        global _config
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        cfg = load_config()
        if name not in cfg.get("models", []):
            return jsonify({"error": f"Model '{name}' not found"}), 404
        cfg["active"] = name
        save_config(cfg)
        _config = cfg
        return jsonify({"active": name})

    @app.route("/models/add", methods=["POST"])
    def add_model():
        global _config
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        base_model = data.get("base_model", "Qwen/Qwen2.5-7B").strip()
        if not name:
            return jsonify({"error": "Name required"}), 400
        cfg = load_config()
        if name in cfg.get("models", []):
            return jsonify({"error": f"Model '{name}' already exists"}), 409
        # Write the per-model config file
        new_model = {
            "name": name,
            "base_model": base_model,
            "system_prompt": f"You are {name}, a helpful AI assistant.",
        }
        save_model_config(new_model)
        # Add to root index and make active
        cfg.setdefault("models", []).append(name)
        cfg["active"] = name
        save_config(cfg)
        _config = cfg
        # Create knowledge folder
        kdir = _model_knowledge_dir(name)
        os.makedirs(os.path.join(kdir, "manual"), exist_ok=True)
        print(f"[studio] Created training data folder: {kdir}")
        return jsonify({"added": name, "knowledge_dir": kdir})

    @app.route("/models/delete", methods=["POST"])
    def delete_model():
        global _config
        import shutil as _shutil
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        cfg = load_config()
        models = cfg.get("models", [])
        if len(models) <= 1:
            return jsonify({"error": "Cannot delete the last model"}), 400
        cfg["models"] = [m for m in models if m != name]
        if cfg.get("active") == name:
            cfg["active"] = cfg["models"][0]
        save_config(cfg)
        _config = cfg

        # ── Remove the model's entire folder (knowledge, dataset, lora, gguf) ─
        model_dir = _model_root(name)
        if os.path.isdir(model_dir):
            try:
                _shutil.rmtree(model_dir)
                print(f"[studio] Deleted model folder: {model_dir}")
            except Exception as e:
                print(f"[studio] Could not delete model folder {model_dir}: {e}")

        return jsonify({
            "deleted": name,
            "active":  cfg["active"],
        })

    @app.route("/models/detail", methods=["GET"])
    def model_detail():
        name = request.args.get("name", "").strip()
        cfg = load_config()
        if name not in cfg.get("models", []):
            return jsonify({"error": f"Model '{name}' not found"}), 404
        return jsonify(load_model_config(name))

    @app.route("/model-catalog", methods=["GET"])
    def model_catalog():
        """Return the full model catalog, annotated with fit status for the user's VRAM."""
        try:
            import torch
            vram = 0
            bf16 = False
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram = round(props.total_memory / (1024**3))
                bf16 = props.major >= 8
        except Exception:
            vram = 0
            bf16 = False

        result = []
        for m in MODEL_CATALOG:
            # vram == 0 means NO GPU detected — not "unknown".
            # Without a CUDA GPU, training requires CPU-only mode (extremely slow)
            # and bitsandbytes 4-bit is unavailable. Show all models as fitting
            # so the dropdown isn't empty, but annotate them as CPU-only.
            has_gpu = vram > 0
            fits_4bit  = not has_gpu or vram >= m["min_vram_4bit"]
            # For fp16/bf16 training, require BOTH:
            #   1. At least 24 GB VRAM — matches the UI's _setQuantOptionsForVram threshold.
            #      Small models (0.5B) technically fit fp16 on 16 GB but the UI disables
            #      fp16/bf16 below 24 GB so quant_mode must match.
            #   2. Model-specific VRAM requirement (min_vram_fp16 + 4 GB training buffer).
            # No GPU → fp16 never fits (bitsandbytes 4-bit won't work either, but at
            # least 4bit shows as the option — the user gets a clear "CPU only" warning).
            fits_fp16  = has_gpu and vram >= 24 and vram >= m["min_vram_fp16"] + 4
            if fits_fp16:
                quant_mode = "fp16" if not bf16 else "bf16"
                quant_label = "Full BF16 (fastest)" if bf16 else "Full FP16"
            elif fits_4bit and has_gpu:
                quant_mode = "4bit"
                quant_label = "4-bit QLoRA recommended"
            elif fits_4bit and not has_gpu:
                quant_mode = "4bit"
                quant_label = "CPU only (very slow — GPU required)"
            else:
                quant_mode = "too_large"
                quant_label = f"Needs {m['min_vram_4bit']} GB+ VRAM (4-bit)"

            result.append({
                "id":          m["id"],
                "params_b":    m["params_b"],
                "tag":         m["tag"],
                "description": m["description"],
                "min_vram_4bit": m["min_vram_4bit"],
                "min_vram_fp16": m["min_vram_fp16"],
                "fits":        fits_4bit,   # True = model fits (at least with 4-bit)
                "fits_fp16":   fits_fp16,
                "quant_mode":  quant_mode,
                "quant_label": quant_label,
            })
        # Include the recommended base model so the frontend can auto-select it
        # in the catalog dropdown (same logic as /system-info)
        recommended_base = None
        if vram >= 48:
            recommended_base = "Qwen/Qwen2.5-32B"
        elif vram >= 24:
            recommended_base = "Qwen/Qwen2.5-14B"
        elif vram >= 16:
            recommended_base = "Qwen/Qwen2.5-7B"
        elif vram >= 10:
            recommended_base = "Qwen/Qwen2.5-3B"
        elif vram >= 6:
            recommended_base = "microsoft/phi-2"
        elif vram > 0:
            recommended_base = "Qwen/Qwen2.5-0.5B"

        return jsonify({"vram_gb": vram, "bf16": bf16, "models": result, "recommended_base": recommended_base})

    @app.route("/models/update", methods=["POST"])
    def update_model():
        global _config
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        cfg = load_config()
        if name not in cfg.get("models", []):
            return jsonify({"error": f"Model '{name}' not found"}), 404
        # Load the per-model config, update fields, save back
        m = load_model_config(name)
        if "base_model" in data:
            m["base_model"] = str(data["base_model"])
        if "system_prompt" in data:
            m["system_prompt"] = str(data["system_prompt"])
        if "train_settings" in data:
            ts = data["train_settings"]
            if isinstance(ts, dict):
                m["train_settings"] = {
                    "quant":   str(ts.get("quant", "4bit")),
                    "epochs":  int(ts.get("epochs", 3)),
                    "batch":   int(ts.get("batch", 2)),
                    "lr":      str(ts.get("lr", "2e-4")),
                    "lora_r":  int(ts.get("lora_r", 16)),
                    "seq_len": int(ts.get("seq_len", 2048)),
                }
        save_model_config(m)
        return jsonify({"saved": True})

    @app.route("/pipeline-state", methods=["GET"])
    def pipeline_state():
        """Return which pipeline steps have been completed for the active model.

        Checks for the presence of key output files to determine progress:
          - dataset_built : models/<ModelName>/dataset/train_chatml.jsonl exists
          - training_done : models/<ModelName>/lora/<ModelName>_{size}_r{r}_a{a}/ has
                            adapter_config.json at its ROOT (not just checkpoint-N/ subdirs).
                            Prefers the subdir whose base_model matches the current config.
          - export_done   : models/<ModelName>/gguf/ contains a .gguf file whose name
                            includes the size tag of the currently configured base model
                            (e.g. "TestAI_7b_q4_k_m.gguf" matches size tag "7b").

        Each pipeline step keeps its output on disk as a permanent artifact:
          Step 1 writes JSONL files  → dataset_built = True
          Step 2 writes LoRA adapter → training_done = True (in models/<Name>/lora/{subdir}/)
          Step 3 writes GGUF file    → export_done   = True (in models/<Name>/gguf/)
        The UI uses these flags to show checkmarks on completed steps and unlock the next one.

        GGUF filenames encode both model name, size and quantization:
          models/<ModelName>/gguf/TestAI_7b_q4_k_m.gguf   <- 7B export
          models/<ModelName>/gguf/TestAI_14b_q4_k_m.gguf  <- 14B export (different size tag)

        This means switching the base model (7B → 14B) automatically marks export_done=False
        for the new base model — the 7B file doesn't match the 14B size tag.

        Migration: silently moves JSONL files from legacy locations (train/dataset/
        or train/dataset/<ModelName>/) into the current models/<ModelName>/dataset/.
        """
        active = get_active_name()
        safe_active = _safe_name(active)
        dest_dir = get_dataset_dir()   # models/<ModelName>/dataset/

        # ── Migration: move JSONL files from legacy locations ─────────────────
        legacy_roots = [
            os.path.join(ROOT, "train", "dataset"),                      # flat pre-v2
            os.path.join(ROOT, "train", "dataset", safe_active),         # per-model pre-v3
        ]
        for _legacy in legacy_roots:
            for _fname in ("train_chatml.jsonl", "train_alpaca.jsonl"):
                _src = os.path.join(_legacy, _fname)
                if os.path.isfile(_src):
                    os.makedirs(dest_dir, exist_ok=True)
                    _dst = os.path.join(dest_dir, _fname)
                    if not os.path.exists(_dst):
                        os.rename(_src, _dst)
                        print(f"[studio] Migrated {_fname} → {dest_dir}")

        # 1. Dataset built? Check the per-model dataset directory.
        chatml_path = os.path.join(dest_dir, "train_chatml.jsonl")
        dataset_built = os.path.isfile(chatml_path) and os.path.getsize(chatml_path) > 0

        # 2. Training done? Check models/<ModelName>/lora/ for a completed LoRA.
        #    Also checks legacy train/popai_lora/ and migrates any found LoRA subdir.
        #
        #    A COMPLETE LoRA has adapter_config.json at the ROOT of the lora subdir.
        #    checkpoint-N/ subdirs are intermediate saves — they do NOT count as done.
        #    train.py itself uses this same rule (see its "definitive signal" comment).
        #    Counting checkpoints as done would block the Train button for incomplete runs.
        #
        #    BASE MODEL MISMATCH: If the LoRA was trained on a different base model than
        #    the one currently configured, it is incompatible — treat as not trained.
        lora_base = _model_lora_dir(active)   # models/<ModelName>/lora/
        training_done = False
        base_model_mismatch = False
        _lora_found_sub = None   # path to the completed LoRA subdir (for mismatch check)

        # Check new location: models/<ModelName>/lora/
        # Prefer the subdir whose adapter_config.json base_model matches the
        # currently configured base model (so switching from 7B to 14B after
        # training both picks the 14B adapter, not the 7B one).
        _cfg_base_model = load_model_config(active).get("base_model", "")

        if os.path.isdir(lora_base):
            import json as _json
            _exact_lora_match = None   # subdir whose base_model matches config
            _any_lora_match   = None   # any completed subdir (fallback)

            for _entry in sorted(os.listdir(lora_base)):   # sorted for determinism
                _sub = os.path.join(lora_base, _entry)
                if not os.path.isdir(_sub):
                    continue
                _has_adapter = (
                    os.path.isfile(os.path.join(_sub, "adapter_config.json")) or
                    os.path.isfile(os.path.join(_sub, "adapter_model.safetensors"))
                )
                if not _has_adapter:
                    continue
                # Try to read the base model from this adapter
                _sub_bm = ""
                _acp = os.path.join(_sub, "adapter_config.json")
                if os.path.isfile(_acp):
                    try:
                        with open(_acp, "r", encoding="utf-8") as _f:
                            _sub_bm = _json.load(_f).get("base_model_name_or_path", "")
                    except Exception:
                        pass
                if _sub_bm.strip().lower() == _cfg_base_model.strip().lower():
                    _exact_lora_match = _sub   # matches currently configured model
                if _any_lora_match is None:
                    _any_lora_match = _sub     # first found (fallback)

            if _exact_lora_match:
                training_done = True
                _lora_found_sub = _exact_lora_match
            elif _any_lora_match:
                # Found an adapter but it doesn't match the current base model
                training_done = False
                base_model_mismatch = True
                _lora_found_sub = _any_lora_match  # still track it for GGUF check (export_done stays False)

        # Fall back to legacy train/popai_lora/<ModelName>_*/ location
        if not training_done and not base_model_mismatch:
            _legacy_lora = os.path.join(TRAIN_DIR, "popai_lora")
            if os.path.isdir(_legacy_lora):
                import json as _json
                prefix = safe_active.lower() + "_"
                for _entry in sorted(os.listdir(_legacy_lora)):
                    if not _entry.lower().startswith(prefix):
                        continue
                    _sub = os.path.join(_legacy_lora, _entry)
                    if not os.path.isdir(_sub):
                        continue
                    _has_adapter = (
                        os.path.isfile(os.path.join(_sub, "adapter_config.json")) or
                        os.path.isfile(os.path.join(_sub, "adapter_model.safetensors"))
                    )
                    if not _has_adapter:
                        continue
                    _sub_bm = ""
                    _acp = os.path.join(_sub, "adapter_config.json")
                    if os.path.isfile(_acp):
                        try:
                            with open(_acp, "r", encoding="utf-8") as _f:
                                _sub_bm = _json.load(_f).get("base_model_name_or_path", "")
                        except Exception:
                            pass
                    if _sub_bm.strip().lower() == _cfg_base_model.strip().lower():
                        training_done = True
                        _lora_found_sub = _sub
                        break
                    elif _lora_found_sub is None:
                        # Mismatch — note it but keep scanning for a match
                        base_model_mismatch = True
                        _lora_found_sub = _sub

        # 3. Export done?
        #    Build the exact filename that generate_llm.py produces for the currently
        #    selected quant type and check whether it exists in models/<Name>/gguf/.
        #    _build_gguf_filename() reads model_type and architecture dimensions from
        #    the HF cached config.json — same data generate_llm.py uses — so the
        #    name built here and the name written by the exporter are always identical.
        #    Changing the quant dropdown immediately changes the expected filename,
        #    so the button disables/enables without any file scanning or regex.
        _gguf_type_req = request.args.get('gguf_type', 'q4_k_m')
        export_done = False
        _matched_gguf = ""
        if not base_model_mismatch and training_done:
            gguf_base      = _model_gguf_dir(active)
            _expected_fn   = _build_gguf_filename(active, _cfg_base_model, _gguf_type_req)
            _expected_path = os.path.join(gguf_base, _expected_fn)
            if os.path.isfile(_expected_path):
                export_done   = True
                _matched_gguf = _expected_fn
        return jsonify({
            "dataset_built":       dataset_built,
            "training_done":       training_done,
            "export_done":         export_done,
            "gguf_name":           _matched_gguf,
            "base_model_mismatch": base_model_mismatch,
        })

    @app.route("/system-monitor", methods=["GET"])
    def system_monitor():
        """Live CPU, RAM, GPU utilization — polled every few seconds by the UI.

        VRAM and GPU utilization are read from the NVIDIA driver via nvidia-smi
        (with pynvml as a faster fallback when available).  This gives the true
        system-wide numbers that match Windows Task Manager / GPU-Z, regardless
        of whether a PyTorch model is loaded in this server process.

        torch.cuda.mem_get_info() is intentionally NOT used for VRAM because it
        only reports what PyTorch's own allocator has reserved in *this* process
        — it misses VRAM used by the training subprocess and idle driver overhead,
        producing misleadingly low numbers (e.g. 1.2 GB when 10+ GB are in use).
        """
        data = {}

        # ── CPU + system RAM (psutil) ─────────────────────────────────────────
        # cpu_percent(interval=None) returns 0.0 on the very first call because
        # there is no previous measurement to diff against.  Use interval=0.1 to
        # take a 100 ms blocking sample — accurate, and fine since this endpoint
        # is only polled every 3 seconds from the browser.
        try:
            import psutil
            data["cpu_pct"]      = psutil.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            data["ram_used_gb"]  = round(vm.used  / (1024**3), 1)
            data["ram_total_gb"] = round(vm.total / (1024**3), 1)
            data["ram_pct"]      = round(vm.percent, 1)
        except Exception:
            data["cpu_pct"]      = None
            data["ram_used_gb"]  = None
            data["ram_total_gb"] = None
            data["ram_pct"]      = None

        # ── GPU utilization + VRAM (driver-level, system-wide) ───────────────
        # Strategy: try pynvml first (fastest, no subprocess), then fall back to
        # nvidia-smi (always available on Windows with an NVIDIA driver).
        gpu_pct      = None
        vram_used_gb = None
        vram_total_gb = None
        vram_pct     = None

        gpu_temp_c   = None

        # 1. pynvml — reads the NVIDIA Management Library directly
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_pct       = util.gpu
            vram_used_gb  = round(mem.used  / (1024**3), 1)
            vram_total_gb = round(mem.total / (1024**3), 1)
            vram_pct      = round(mem.used / mem.total * 100, 1) if mem.total else 0
            try:
                gpu_temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
        except Exception:
            pass

        # 2. nvidia-smi — subprocess fallback; always correct, slightly slower
        if gpu_pct is None or vram_used_gb is None or gpu_temp_c is None:
            try:
                import subprocess as _sp
                result = _sp.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 4:
                        _gpu  = float(parts[0].strip())
                        _used = float(parts[1].strip())   # MiB
                        _tot  = float(parts[2].strip())   # MiB
                        _temp = float(parts[3].strip())   # °C
                        if gpu_pct is None:
                            gpu_pct = int(_gpu)
                        if vram_used_gb is None:
                            vram_used_gb  = round(_used / 1024, 1)
                            vram_total_gb = round(_tot  / 1024, 1)
                            vram_pct      = round(_used / _tot * 100, 1) if _tot else 0
                        if gpu_temp_c is None:
                            gpu_temp_c = int(_temp)
            except Exception:
                pass

        data["gpu_pct"]       = gpu_pct
        data["vram_used_gb"]  = vram_used_gb
        data["vram_total_gb"] = vram_total_gb
        data["vram_pct"]      = vram_pct
        data["gpu_temp_c"]    = gpu_temp_c
        return jsonify(data)

    @app.route("/system-info", methods=["GET"])
    def system_info():
        try:
            info = {}
            # CPU + RAM
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
                info["ram_gb"] = round(ram_gb)
            except Exception:
                # psutil not installed — try ctypes fallback on Windows
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                     ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                                     ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                                     ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                                     ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                    info["ram_gb"] = round(stat.ullTotalPhys / (1024**3))
                except Exception:
                    info["ram_gb"] = None
            # GPU + VRAM
            try:
                import torch
                if torch.cuda.is_available():
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    info["vram_gb"] = round(props.total_memory / (1024**3))
                    info["cuda_compute"] = f"{props.major}.{props.minor}"
                    info["cuda_version"] = torch.version.cuda or None
                    info["bf16"] = props.major >= 8
                else:
                    info["gpu_name"] = None
                    info["vram_gb"] = None
                    info["cuda_compute"] = None
                    info["bf16"] = False
            except Exception:
                info["gpu_name"] = None
                info["vram_gb"] = None
                info["cuda_compute"] = None
                info["bf16"] = False
            # Recommendations
            vram = info.get("vram_gb") or 0
            ram  = info.get("ram_gb")  or 0
            recs = []
            if vram == 0:
                recs.append({"level": "warn", "text": "No CUDA GPU detected — training on CPU will be extremely slow."})
            elif vram < 8:
                recs.append({"level": "warn", "text": f"Only {vram} GB VRAM — too little for 7B models. Use a smaller base model (e.g. 1B or 3B)."})
            elif vram < 12:
                recs.append({"level": "warn", "text": f"Only {vram} GB VRAM — stick to 3B models or smaller. See recommended settings below."})
            elif vram < 16:
                recs.append({"level": "info", "text": f"{vram} GB VRAM — 7B models may be tight. See recommended settings below."})
            elif vram < 24:
                recs.append({"level": "good", "text": f"{vram} GB VRAM — good for 7B models. See recommended settings below."})
            else:
                recs.append({"level": "good", "text": f"{vram} GB VRAM — can run larger models. See recommended settings below."})
            if ram and ram < 16:
                recs.append({"level": "warn", "text": f"Only {ram} GB system RAM — disable gradient checkpointing may cause OOM. Keep --grad-ckpt ON."})
            elif ram and ram < 32:
                recs.append({"level": "info", "text": f"{ram} GB system RAM — sufficient, but keep dataset size reasonable."})
            # Base model recommendation based on VRAM
            if vram >= 48:
                info["recommended_base"] = "Qwen/Qwen2.5-32B"
                info["recommended_alts"] = [
                    "Qwen/Qwen2.5-72B",
                    "mistralai/Mistral-7B-v0.1",
                ]
            elif vram >= 24:
                info["recommended_base"] = "Qwen/Qwen2.5-14B"
                info["recommended_alts"] = [
                    "mistralai/Mistral-7B-v0.1",
                    "Qwen/Qwen2.5-7B",
                ]
            elif vram >= 16:
                info["recommended_base"] = "Qwen/Qwen2.5-7B"
                info["recommended_alts"] = [
                    "mistralai/Mistral-7B-v0.1",
                    "EleutherAI/pythia-6.9b",
                ]
            elif vram >= 10:
                info["recommended_base"] = "Qwen/Qwen2.5-3B"
                info["recommended_alts"] = [
                    "microsoft/phi-2",
                    "HuggingFaceTB/SmolLM2-1.7B",
                ]
            elif vram >= 6:
                info["recommended_base"] = "microsoft/phi-2"
                info["recommended_alts"] = [
                    "Qwen/Qwen2.5-1.5B",
                    "HuggingFaceTB/SmolLM2-1.7B",
                ]
            elif vram > 0:
                info["recommended_base"] = "Qwen/Qwen2.5-0.5B"
                info["recommended_alts"] = ["HuggingFaceTB/SmolLM2-1.7B"]
            else:
                info["recommended_base"] = None
                info["recommended_alts"] = []
            info["recommendations"] = recs
            return jsonify(info)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e), "recommendations": [], "gpu_name": None, "vram_gb": None, "ram_gb": None, "recommended_base": None})

    # Track the chat server subprocess so we can detect if it's already running
    _chat_proc = {"proc": None}

    @app.route("/run/<script>", methods=["POST"])
    def run_script(script):
        scripts = {
            "train":    [sys.executable, os.path.join(TRAIN_DIR, "train.py")],
            "generate": [sys.executable, os.path.join(TRAIN_DIR, "generate_llm.py")],
        }
        # chat server is a long-running process — handled separately below
        if script == "chat":
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            # Use the .venv Python explicitly so llama_cpp and all packages are available
            venv_python = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
            if not os.path.isfile(venv_python):
                venv_python = sys.executable  # fallback if no venv found
            chat_cmd = [venv_python, os.path.join(ROOT, "chat_test", "server.py"), "--no-open"]
            try:
                # If a previous chat server is still running, leave it alone
                existing = _chat_proc.get("proc")
                if existing and existing.poll() is None:
                    # Already running — tell the frontend to focus the existing tab
                    return jsonify({"returncode": 0, "stdout": "[chat] Server already running at http://localhost:5000", "stderr": "", "chat_url": "http://localhost:5000"})
                # Launch as a background process (Popen, not run) — it never exits on its own
                proc = subprocess.Popen(
                    chat_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    cwd=ROOT,
                    env=env,
                )
                _chat_proc["proc"] = proc
                # Return the URL — frontend opens the tab (avoids duplicate windows)
                return jsonify({"returncode": 0, "stdout": "[chat] Server starting at http://localhost:5000 - loading model (may take 10-30 seconds)...", "stderr": "", "chat_url": "http://localhost:5000"})
            except Exception as e:
                return jsonify({"error": str(e)})

        if script not in scripts:
            return jsonify({"error": "Unknown script"}), 400
        cmd = scripts[script]
        # Ensure UTF-8 mode so TRL/HF imports don't hit Windows charmap errors
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=ROOT,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            return jsonify({
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-1500:],
            })
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Timed out after 5 minutes"})
        except Exception as e:
            return jsonify({"error": str(e)})

    # ── SSE streaming endpoint for train.py ───────────────────────────────────
    # Keeps the subprocess alive and pushes each line as a Server-Sent Event so
    # the browser can display live progress without waiting for the job to finish.

    _train_proc = {"proc": None}   # mutable container so the generator can close it
    # Broadcast queue: all SSE events are pushed here so reconnecting clients
    # can subscribe and receive live output from an already-running process.
    import queue as _queue_mod
    _train_broadcast_q = []   # list of subscriber Queue objects (one per client)
    _train_log_buffer  = []   # rolling replay buffer (last 500 events)

    def _train_publish(event_str):
        """Publish one SSE event string to all subscribers and the replay buffer."""
        _train_log_buffer.append(event_str)
        if len(_train_log_buffer) > 500:
            del _train_log_buffer[:-500]
        for q in list(_train_broadcast_q):
            try:
                q.put_nowait(event_str)
            except Exception:
                pass

    @app.route("/run/train/stream", methods=["GET"])
    def stream_train():
        from flask import Response, stream_with_context

        # If a train process is already running and this is NOT a reconnect request,
        # refuse to start a second one. The client sends ?reconnect=1 when it wants
        # to reattach to an existing stream after a page refresh.
        # This is a hard guard: only one training process may run at a time across
        # ALL models — training loads a full base LLM into VRAM, so two concurrent
        # runs (even for different models) will exhaust VRAM and crash both.
        is_reconnect = request.args.get('reconnect') == '1'
        existing_proc = _train_proc.get("proc")
        if not is_reconnect and existing_proc and existing_proc.poll() is None:
            # Already running — send a structured SSE error so the client can
            # display a clear message instead of silently failing.
            import json as _json
            def _already_running():
                yield f"data: {_json.dumps({'type': 'error', 'text': 'already_running'})}\n\n"
            return Response(
                stream_with_context(_already_running()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # -- Reconnect path: reattach to an already-running training process ----
        if is_reconnect and existing_proc and existing_proc.poll() is None:
            import json as _json
            sub_q = _queue_mod.Queue()
            _train_broadcast_q.append(sub_q)
            def _reconnect_stream():
                try:
                    # Replay buffered events so the client catches up
                    for evt in list(_train_log_buffer):
                        yield evt
                    # Then stream live events as they arrive
                    while True:
                        try:
                            item = sub_q.get(timeout=30)
                            yield item
                            try:
                                import json as _j
                                parsed = _j.loads(item[len('data: '):].strip())
                                if parsed.get('type') in ('done', 'error'):
                                    break
                            except Exception:
                                pass
                        except _queue_mod.Empty:
                            yield ': keepalive\n\n'
                finally:
                    try:
                        _train_broadcast_q.remove(sub_q)
                    except ValueError:
                        pass
            return Response(
                stream_with_context(_reconnect_stream()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Clear log buffer for a new training run
        _train_log_buffer.clear()

        def generate():
            env = os.environ.copy()
            # Force UTF-8 mode so train.py does NOT re-launch itself via subprocess.run.
            # Without these, train.py detects utf8_mode==0 and spawns a child process,
            # causing our Popen to capture only the short-lived wrapper (which exits
            # immediately), closing the SSE stream before any real output appears.
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            # Launch with -X utf8 so sys.flags.utf8_mode==1 inside train.py,
            # skipping the self-relaunch guard entirely.
            # Read saved train_settings from config and pass as CLI args
            _ts = get_active_model(load_config()).get("train_settings", {})
            _use_4bit = _ts.get("quant", "4bit") == "4bit"
            _epochs   = str(_ts.get("epochs", 3))
            _batch    = str(_ts.get("batch", 2))
            _lr       = str(_ts.get("lr", "2e-4"))
            _lora_r     = str(_ts.get("lora_r", 16))
            _seq_len    = str(_ts.get("seq_len", 2048))
            # eval_split: UI sends ?eval_split=1 in the stream URL when the checkbox is checked.
            # Also honour the saved setting in train_settings as a fallback.
            _eval_split = request.args.get('eval_split') == '1' or bool(_ts.get("eval_split", False))
            # Do NOT pass --fresh here — let train.py's checkpoint detection run.
            # If a checkpoint exists, train.py will auto-resume (EOFError path
            # triggers the default "resume" choice since the server has no TTY).
            # The user can click "Fresh Start" in the UI to wipe checkpoints first.
            cmd = [sys.executable, "-X", "utf8", "-u",
                   os.path.join(TRAIN_DIR, "train.py"),
                   "--epochs", _epochs,
                   "--batch",  _batch,
                   "--lr",     _lr,
                   "--lora-r", _lora_r,
                   "--max_seq_len", _seq_len,
            ]
            if not _use_4bit:
                cmd.append("--no-4bit")
            if _eval_split:
                cmd.append("--eval-split")

            is_fresh = request.args.get('fresh') == '1'
            if is_fresh:
                cmd.append('--fresh')
                # ── Wipe the active model's LoRA checkpoints before a fresh start ─
                # Only wipes checkpoint-N/ subdirs and adapter files from the specific
                # LoRA subdir that matches the current model name + base model.
                # Other LoRA subdirs (e.g. for a different base model size) are kept.
                # .gguf and .ollama files are NOT deleted — see note below.
                import shutil as _shutil_fs
                import re as _re_fresh
                _active_name = get_active_name()
                _safe_n = _safe_name(_active_name)
                _active_model_cfg = load_model_config(_active_name)
                _active_base = _active_model_cfg.get("base_model", "")

                # Compute the same subdir name that train.py would use:
                #   {modelname_lower}-{size}-{family}
                # This must match train.py's _dir_name computation exactly.
                _name_part = _re_fresh.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', _active_name).strip(". ").lower() or "model"

                def _fresh_model_family(base_model):
                    parts = base_model.split("/")
                    mp = parts[-1] if len(parts) > 1 else parts[0]
                    n = _re_fresh.sub(r'[-_]?\d+\.?\d*[Bb][\w.]*.*$', '', mp)
                    n = _re_fresh.sub(r'[-_]?(instruct|chat|hf|base|it|v\d[\w.]*)$', '', n, flags=_re_fresh.IGNORECASE)
                    n = n.strip('-_').lower().replace('_', '-')
                    return _re_fresh.sub(r'-+', '-', n).strip('-')

                _size_m = _re_fresh.search(r'[-_/](\d+(?:\.\d+)?[Bb])', _active_base)
                _size_t = _size_m.group(1).lower() if _size_m else ""
                _fam_t  = _fresh_model_family(_active_base)
                _size_fam = f"{_size_t}-{_fam_t}" if (_size_t and _fam_t) else (_size_t or _fam_t or "base")
                _lora_subdir_name = f"{_name_part}-{_size_fam}"

                _lora_base_dir   = _model_lora_dir(_active_name)
                _lora_target_dir = os.path.join(_lora_base_dir, _lora_subdir_name)

                # Wipe only the matching LoRA subdir contents (checkpoints + adapter files).
                # Other subdirs in lora/ (e.g. from a different base model) are preserved.
                _ADAPTER_FILES = {
                    "adapter_config.json", "adapter_model.safetensors",
                    "adapter_model.bin", "adapter_model.pt",
                    "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "chat_template.jinja",
                    "training_args.bin", "README.md", ".training_info.json",
                }
                if os.path.isdir(_lora_target_dir):
                    _fresh_removed = []
                    for _fn in os.listdir(_lora_target_dir):
                        _fp = os.path.join(_lora_target_dir, _fn)
                        if _fn.startswith("checkpoint-") and os.path.isdir(_fp):
                            try:
                                _shutil_fs.rmtree(_fp)
                                _fresh_removed.append(_fn)
                            except Exception as _e:
                                print(f"[studio] Could not delete checkpoint {_fp}: {_e}")
                        elif os.path.isfile(_fp) and _fn in _ADAPTER_FILES:
                            try:
                                os.remove(_fp)
                                _fresh_removed.append(_fn)
                            except Exception as _e:
                                print(f"[studio] Could not delete {_fp}: {_e}")
                    if _fresh_removed:
                        print(f"[studio] Fresh start — cleared {len(_fresh_removed)} item(s) from: {_lora_target_dir}")
                    else:
                        print(f"[studio] Fresh start — nothing to clear in: {_lora_target_dir}")
                else:
                    print(f"[studio] Fresh start — LoRA subdir not found (will be created by training): {_lora_target_dir}")

                # Note: .gguf and .ollama files in models/<ModelName>/gguf/ are NOT
                # deleted on fresh start. They are only replaced when generate_llm.py
                # produces a new export with the same filename. Deleting them here
                # would destroy a working export just because the user wants to retrain.

                # Legacy location: train/popai_lora/<ModelName>_*/
                _legacy_lora_base = os.path.join(TRAIN_DIR, "popai_lora")
                if os.path.isdir(_legacy_lora_base):
                    for _entry in os.listdir(_legacy_lora_base):
                        _sub = os.path.join(_legacy_lora_base, _entry)
                        if os.path.isdir(_sub) and _entry.lower().startswith(_safe_n.lower() + "_"):
                            try:
                                _shutil_fs.rmtree(_sub)
                                print(f"[studio] Fresh start — deleted legacy LoRA dir: {_sub}")
                            except Exception as _e:
                                print(f"[studio] Could not delete legacy LoRA dir {_sub}: {_e}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=ROOT,
                    env=env,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                _train_proc["proc"] = proc
                import json as _json
                import re as _re
                _evt = "data: {\"type\":\"start\"}\n\n"
                _train_publish(_evt)
                yield _evt
                for raw_line in proc.stdout:
                    # tqdm on Windows writes \r-separated frames in a single \n-terminated
                    # chunk.  When tqdm output is merged with stdout (stderr=STDOUT), a plain
                    # [train] log line printed before tqdm starts can end up in the same read
                    # chunk as the tqdm bar, e.g.:
                    #   "[train] Resuming from checkpoint: ...\rTokenizing: 100%|...|\n"
                    # Split on \r and classify each segment individually:
                    #   - Looks like a [tag] log line → 'line'  (permanent, new line in UI)
                    #   - Looks like a tqdm bar        → 'progress' (in-place update)
                    segments = raw_line.rstrip('\n').split('\r')
                    non_empty = [s for s in segments if s.strip()]
                    if not non_empty:
                        continue

                    if len(non_empty) == 1 and '\r' not in raw_line:
                        # Pure plain log line — no \r at all; always show as a
                        # permanent line (includes warnings, tracebacks, etc.)
                        safe = non_empty[0].rstrip()
                        _evt = f"data: {_json.dumps({'type': 'line', 'text': safe})}\n\n"
                        _train_publish(_evt)
                        yield _evt
                    else:
                        # Mixed chunk (tqdm / progress bar with embedded \r frames).
                        # The LAST segment is the final bar frame — show as progress
                        # (in-place update). Earlier segments that look like plain log
                        # lines are shown as permanent lines.
                        for i, seg in enumerate(non_empty):
                            safe = seg.strip()
                            if not safe:
                                continue
                            is_last = (i == len(non_empty) - 1)
                            is_log_line = bool(_re.match(r'^\[[\w\s]+\]', safe))
                            # Log-tagged lines are always permanent regardless of position.
                            # The last segment in a \r chunk is a progress bar update.
                            # Any other segment defaults to 'line' so it's not lost.
                            if is_log_line or not is_last:
                                evt_type = 'line'
                            else:
                                evt_type = 'progress'
                            _evt = f"data: {_json.dumps({'type': evt_type, 'text': safe})}\n\n"
                            _train_publish(_evt)
                            yield _evt
                proc.wait()
                _evt = f"data: {_json.dumps({'type': 'done', 'returncode': proc.returncode})}\n\n"
                _train_publish(_evt)
                yield _evt
            except Exception as exc:
                import json as _json
                _evt = f"data: {_json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
                _train_publish(_evt)
                yield _evt
            finally:
                p = _train_proc.get("proc")
                if p is not None and p.poll() is not None:
                    _train_proc["proc"] = None

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/run/train/stop", methods=["POST"])
    def stop_train():
        proc = _train_proc.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
            return jsonify({"stopped": True})
        return jsonify({"stopped": False, "reason": "not running"})

    @app.route("/run/train/status", methods=["GET"])
    def train_status():
        proc = _train_proc.get("proc")
        running = proc is not None and proc.poll() is None
        return jsonify({"running": running})

    # ── SSE streaming endpoint for generate_llm.py ────────────────────────────
    _generate_proc = {"proc": None}

    @app.route("/run/generate/stream", methods=["GET"])
    def stream_generate():
        from flask import Response, stream_with_context

        # Read GGUF options from query string — set by the UI export panel
        gguf_type         = request.args.get("gguf_type", "auto")       # auto|f16|q8_0|q4_k_m|q4_0
        keep_safetensors  = request.args.get("keep_safetensors", "0")   # 1 = keep

        def generate():
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            cmd = [sys.executable, "-X", "utf8", "-u",
                   os.path.join(TRAIN_DIR, "generate_llm.py")]
            # Pass GGUF type if the user specified one (not "auto")
            if gguf_type and gguf_type != "auto":
                cmd += ["--gguf-type", gguf_type]
            # Pass --keep-safetensors if requested
            if keep_safetensors == "1":
                cmd.append("--keep-safetensors")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=ROOT,
                    env=env,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                _generate_proc["proc"] = proc
                import json as _json
                yield "data: {\"type\":\"start\"}\n\n"

                for raw_line in proc.stdout:
                    segments = raw_line.rstrip('\n').split('\r')
                    non_empty = [s for s in segments if s.strip()]
                    if not non_empty:
                        continue
                    if len(non_empty) > 1 or '\r' in raw_line:
                        safe = non_empty[-1].strip()
                        if safe:
                            yield f"data: {_json.dumps({'type': 'progress', 'text': safe})}\n\n"
                    else:
                        safe = non_empty[0].rstrip()
                        yield f"data: {_json.dumps({'type': 'line', 'text': safe})}\n\n"

                proc.wait()
                yield f"data: {_json.dumps({'type': 'done', 'returncode': proc.returncode})}\n\n"
            except Exception as exc:
                import json as _json
                yield f"data: {_json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
            finally:
                _generate_proc["proc"] = None

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/run/generate/stop", methods=["POST"])
    def stop_generate():
        proc = _generate_proc.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
            return jsonify({"stopped": True})
        return jsonify({"stopped": False, "reason": "not running"})

    # ── Package management endpoints ──────────────────────────────────────────

    @app.route("/packages", methods=["GET"])
    def list_packages():
        """Return installed vs latest versions for all packages in requirements.txt.

        Uses 'pip index versions' (pip 21.2+) to fetch the latest PyPI version for
        each package without downloading anything.  Falls back to pip list --outdated
        if the requirements file cannot be parsed.
        """
        try:
            req_path = os.path.join(ROOT, "requirements.txt")
            # Parse package names from requirements.txt (skip comments / blank lines)
            packages = []
            if os.path.exists(req_path):
                import re as _re
                with open(req_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # Extract package name (before any version specifier)
                        m = _re.match(r'^([A-Za-z0-9_\-\.]+)', line)
                        if m:
                            packages.append(m.group(1).lower())

            # Also always include torch (not in requirements.txt)
            if "torch" not in packages:
                packages.insert(0, "torch")

            # Get installed versions via pip list (fast, no network)
            installed = {}
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--format=json"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    import json as _json
                    for pkg in _json.loads(result.stdout):
                        installed[pkg["name"].lower()] = pkg["version"]
            except Exception:
                pass

            # Get outdated packages via pip list --outdated (hits PyPI, ~5-10s)
            outdated = {}
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    import json as _json
                    for pkg in _json.loads(result.stdout):
                        outdated[pkg["name"].lower()] = pkg["latest_version"]
            except Exception:
                pass

            rows = []
            for name in packages:
                inst = installed.get(name, None)
                latest = outdated.get(name, inst)   # if not outdated, latest == installed
                is_missing  = inst is None
                is_outdated = (not is_missing) and (name in outdated)
                rows.append({
                    "name":      name,
                    "installed": inst or "not installed",
                    "latest":    latest or "unknown",
                    "outdated":  is_outdated,
                    "missing":   is_missing,
                })

            return jsonify({"packages": rows})
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/update/stream", methods=["GET"])
    def stream_update():
        """SSE endpoint — streams pip upgrade output live to the browser.

        Query params:
            packages  — comma-separated list of package names to upgrade.
                        If omitted, upgrades everything in requirements.txt plus torch.
            torch_index — PyTorch index URL (e.g. https://download.pytorch.org/whl/cu128).
                          Required when upgrading torch.
        """
        from flask import Response, stream_with_context

        pkg_arg     = request.args.get("packages", "").strip()
        torch_index = request.args.get("torch_index", "").strip()

        def generate():
            import json as _json

            # Resolve the list of packages to upgrade
            if pkg_arg:
                pkg_list = [p.strip() for p in pkg_arg.split(",") if p.strip()]
            else:
                # Default: everything in requirements.txt + torch
                pkg_list = []
                req_path = os.path.join(ROOT, "requirements.txt")
                if os.path.exists(req_path):
                    import re as _re
                    with open(req_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            m = _re.match(r'^([A-Za-z0-9_\-\.]+)', line)
                            if m:
                                pkg_list.append(m.group(1))
                if "torch" not in [p.lower() for p in pkg_list]:
                    pkg_list.insert(0, "torch")

            yield f"data: {_json.dumps({'type': 'start', 'packages': pkg_list})}\n\n"

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            # Separate torch from other packages — it needs a special --index-url
            torch_pkgs = [p for p in pkg_list if p.lower() in ("torch", "torchvision", "torchaudio")]
            other_pkgs = [p for p in pkg_list if p.lower() not in ("torch", "torchvision", "torchaudio")]

            def run_pip(cmd, label):
                """Run a pip command and stream every output line as an SSE event."""
                try:
                    yield f"data: {_json.dumps({'type': 'line', 'text': f'$ {chr(32).join(cmd)}'})}\n\n"
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        cwd=ROOT,
                        env=env,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    for raw_line in proc.stdout:
                        text = raw_line.rstrip("\r\n")
                        if text:
                            yield f"data: {_json.dumps({'type': 'line', 'text': text})}\n\n"
                    proc.wait()
                    if proc.returncode != 0:
                        yield f"data: {_json.dumps({'type': 'warn', 'text': f'[{label}] exited with code {proc.returncode}'})}\n\n"
                    else:
                        yield f"data: {_json.dumps({'type': 'ok', 'text': f'[{label}] done'})}\n\n"
                except Exception as exc:
                    yield f"data: {_json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

            # 1. Upgrade torch / torchvision / torchaudio (with index URL if provided)
            if torch_pkgs:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + torch_pkgs
                if torch_index:
                    cmd += ["--index-url", torch_index]
                yield from run_pip(cmd, "torch")

            # 2. Upgrade all other packages
            if other_pkgs:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + other_pkgs
                yield from run_pip(cmd, "packages")

            yield f"data: {_json.dumps({'type': 'done'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Server restart endpoint ───────────────────────────────────────────────

    @app.route("/restart", methods=["POST"])
    def restart_server():
        """Restart the server by spawning a new process then exiting.

        Two modes:

        Standalone mode (launch_web.bat / python train/server.py):
            sys.argv[0] == 'server.py' — safe to spawn a new server process
            then sys.exit(0) to release the port.  The browser waits 4 s before
            reloading, giving the new process time to bind.

        Desktop/embedded mode (launch_app.bat / launch_app.py):
            Flask runs as a daemon thread inside the PyQt6 process.
            sys.argv[0] == 'launch_app.py' — we must NOT spawn a new process or
            call sys.exit() (that would kill the desktop window).
            Instead we return {restarting: true, embedded: true} so the browser
            reloads immediately without waiting — the server never went away.
        """
        import threading as _threading

        # Detect embedded (desktop app) mode: __name__ != '__main__' means server.py
        # was imported as a module by launch_app.py, not run as the top-level script.
        _embedded = (__name__ != "__main__")

        if _embedded:
            # In desktop mode the Flask server is a daemon thread — we can't
            # restart the process.  Just tell the browser to reload immediately.
            print("[studio] Restart requested (desktop/embedded mode) — reloading UI only.")
            return jsonify({"restarting": True, "embedded": True})

        def _do_restart():
            import time as _time
            _time.sleep(0.4)   # let Flask finish sending the 200 response
            print("[studio] Restarting server…")
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.Popen(
                [sys.executable] + sys.argv,
                cwd=os.getcwd(),
                env=env,
            )
            sys.exit(0)

        _threading.Thread(target=_do_restart, daemon=True).start()
        return jsonify({"restarting": True, "embedded": False})

    # ── Static assets (CSS, JS, HTML pages) ──────────────────────────────────
    # Serve any .css, .js, or .html file directly from TRAIN_DIR.
    # This must come AFTER all API routes so API paths take priority.

    @app.route("/<path:filename>")
    def static_assets(filename):
        return send_from_directory(TRAIN_DIR, filename)

    return app


def _kill_port(port):
    """Kill any process already listening on the given port before we bind."""
    try:
        import subprocess, re
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        pids = set()
        for line in result.stdout.splitlines():
            # Match lines with LISTENING on the target port
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
        our_pid = os.getpid()
        for pid in pids:
            if pid == our_pid or pid == 0:
                continue
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5,
                )
                print(f"[studio] Killed stale process PID {pid} on port {port}")
            except Exception:
                pass
    except Exception as e:
        print(f"[studio] Port cleanup skipped: {e}")


def _kill_orphan_train():
    """Kill any train.py subprocesses left over from a previous server instance.

    When the server is stopped (Ctrl+C, Task Manager, launch.bat restart), the
    train.py subprocess it launched keeps running as an orphan on Windows because
    Windows does not automatically kill child processes when the parent exits.

    This function finds and terminates any python processes whose command line
    contains 'train.py', so the new server starts with a clean slate and
    /run/train/status correctly reports {running: false}.
    """
    try:
        import subprocess as _sp
        result = _sp.run(
            ["wmic", "process", "where",
             "name='python.exe' or name='python3.exe'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=8,
        )
        our_pid = os.getpid()
        killed = []
        for line in result.stdout.splitlines():
            # CSV columns: Node, CommandLine, ProcessId
            parts = line.split(',')
            if len(parts) < 3:
                continue
            cmdline = parts[1].strip()
            pid_str = parts[-1].strip()
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == our_pid:
                continue
            # Match train.py (but not server.py or generate_llm.py)
            if 'train.py' in cmdline and 'server.py' not in cmdline and 'generate_llm' not in cmdline:
                try:
                    _sp.run(["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True, timeout=5)
                    killed.append(pid)
                    print(f"[studio] Killed orphan train.py (PID {pid})")
                except Exception:
                    pass
        if not killed:
            print("[studio] No orphaned train.py processes found.")
    except Exception as e:
        print(f"[studio] Orphan train.py cleanup skipped: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Training Studio Server")
    parser.add_argument("--port",    type=int, default=5001)
    parser.add_argument("--no-open", dest="open_browser", action="store_false")
    parser.set_defaults(open_browser=True)
    args = parser.parse_args()

    # Kill any leftover server processes on this port before we start
    _kill_port(args.port)

    # Kill any orphaned train.py processes from a previous server session.
    # On Windows, child processes survive when the parent (server) is killed,
    # so /run/train/status would incorrectly report {running: true} on the new server.
    _kill_orphan_train()

    print("=" * 60)
    print("  LLM Training Studio - NeHe Productions")
    print("=" * 60)
    active = get_active_name()
    print(f"[studio] Active model  : {active}")
    print(f"[studio] Training data : {get_knowledge_dir()}")
    print(f"[studio] Dataset dir   : {get_dataset_dir()}")
    print(f"[studio] Studio URL    : http://localhost:{args.port}")
    print()

    if args.open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    app = create_app()
    # threaded=True is required for SSE streaming — without it Flask serves
    # one request at a time, so the long-running /run/train/stream response
    # blocks all other API calls (stats, stop, etc.) while training runs.
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    main()
