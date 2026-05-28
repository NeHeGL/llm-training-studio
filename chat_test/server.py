"""
LLM Chat Test Server (NeHe Productions)
Author: Jeff Molofee (aka NeHe) — 2026
Serves chat.html and provides /chat + /status API endpoints.

Automatically detects and loads the best available model format:
  1. GGUF file  (.gguf) — loaded via llama-cpp-python. Fast, no Ollama needed.
  2. SafeTensors directory — loaded via HuggingFace transformers (requires more VRAM/RAM).

Usage:
    python chat_test/server.py                   # auto-detects model
    python chat_test/server.py --model path/to/model.gguf
    python chat_test/server.py --model path/to/safetensors_dir
    python chat_test/server.py --no-4bit         # full fp16 (more VRAM, SafeTensors only)
    python chat_test/server.py --port 5000       # default port

Then open: http://localhost:5000
"""

import argparse
import os
import re
import sys
import threading
import webbrowser

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Consolidate __pycache__ at project root ───────────────────────────────────
if not sys.pycache_prefix:
    sys.pycache_prefix = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__pycache__"
    )

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_config():
    """Load active model config from config.json + per-model config file.

    Supports v1 (models list of dicts) and v2 (models list of name strings
    with per-model config in models/{Name}/config.json).
    """
    import json as _json
    cfg_path = os.path.join(ROOT, "config.json")
    default = {"name": "MyModel", "base_model": "Qwen/Qwen2.5-7B", "system_prompt": "You are a helpful AI assistant."}
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
        name_to_load = active_name if active_name in models else models[0]
        safe_n = re.sub(r'[^\w\-]', '_', name_to_load.strip())
        model_cfg_path = os.path.join(ROOT, "models", safe_n, "config.json")
        if os.path.exists(model_cfg_path):
            return _json.load(open(model_cfg_path, encoding="utf-8"))

        return {"name": name_to_load, "base_model": "Qwen/Qwen2.5-7B", "system_prompt": "You are a helpful AI assistant."}
    except Exception as e:
        print(f"[chat] Warning: could not load config: {e}")
        return default

_cfg = _load_config()

def _safe_name(n): return re.sub(r'[^\w\-]', '_', n.strip())

# ── System prompt + model paths from config ────────────────────────────────────
SYSTEM_PROMPT = _cfg.get("system_prompt", "You are a helpful AI assistant.")
_base_model   = _cfg.get("base_model", "Qwen/Qwen2.5-7B")
_model_name   = _cfg.get("name", "MyModel")
# GGUF and Modelfile now live in models/{name}/gguf/
# Legacy: models/{name}/ (pre-restructure installs)
_model_root   = os.path.join(ROOT, "models", _safe_name(_model_name))
_export_dir   = os.path.join(_model_root, "gguf")
if not os.path.isdir(_export_dir):
    # Fall back to legacy location if gguf/ subdir doesn't exist yet
    _export_dir = _model_root

MODEL_CANDIDATES = [
    _export_dir,
    os.path.join(ROOT, "train", "popai_lora"),
    _base_model,
]
BASE_MODEL_FOR_LORA = _base_model

# ── Global model state ─────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_ready     = False
_status_msg = "Loading model..."


def _gpu_info():
    """
    Return a dict with GPU capabilities so loading decisions are made correctly
    for every GPU architecture (Pascal through Blackwell, and CPU-only).
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
        info["bf16"]     = props.major >= 8          # Ampere+ supports BF16
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


# ── GGUF flag — set by find_model() / main() ──────────────────────────────────
_use_gguf = False   # True when loading a .gguf file via llama-cpp-python


def find_gguf_in_dir(directory):
    """Return the first .gguf file found in directory, or None."""
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if fname.lower().endswith(".gguf"):
            return os.path.join(directory, fname)
    return None


def find_model():
    """
    Return the best model path available:
    - A .gguf file inside the export dir (preferred — self-contained, fast)
    - The export dir itself (SafeTensors)
    - A LoRA adapter dir
    - A HuggingFace model ID string
    """
    # Check export dir for a .gguf file first
    gguf = find_gguf_in_dir(_export_dir)
    if gguf:
        return gguf
    for path in MODEL_CANDIDATES:
        if "/" in path and os.path.sep not in path:
            return path
        if os.path.isdir(path):
            # Also check inside this dir for a .gguf
            gguf = find_gguf_in_dir(path)
            if gguf:
                return gguf
            return path
    return None


def is_lora_adapter(path):
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def _install_llama_cpp():
    """
    Install llama-cpp-python for GGUF support.
    Strategy (fastest to slowest):
      1. Try a pre-built binary wheel from the llama-cpp-python releases page
         (no C++ compiler needed, installs in seconds).
      2. Fall back to CPU-only binary wheel via --prefer-binary.
      3. Last resort: source build (slow, needs C++ toolchain).
    """
    import subprocess

    # ── Step 1: pre-built CUDA wheel from the official releases ──────────────
    # The llama-cpp-python project publishes pre-built wheels for common CUDA
    # versions at: https://github.com/abetlen/llama-cpp-python/releases
    # Using --extra-index-url lets pip find them without compiling anything.
    # Try cu124 first (covers CUDA 12.4–12.8), then cu121 as fallback.
    print("[server] Installing llama-cpp-python (pre-built CUDA wheel)...")
    for cuda_tag in ("cu124", "cu121", "cu118"):
        cuda_index = f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag}"
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python",
             "--prefer-binary", "--extra-index-url", cuda_index, "-q"],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"[server] llama-cpp-python installed (CUDA wheel: {cuda_tag}).")
            return

    # ── Step 2: CPU-only pre-built wheel ─────────────────────────────────────
    print("[server] CUDA wheel not available  -  installing CPU-only llama-cpp-python...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "llama-cpp-python",
         "--prefer-binary", "-q"],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print("[server] llama-cpp-python installed (CPU-only wheel).")
        return

    # ── Step 3: source build as last resort ──────────────────────────────────
    print("[server] Binary wheel unavailable  -  compiling from source (this may take several minutes)...")
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "llama-cpp-python", "-q"],
        capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        # Final fallback: CPU source build
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python", "-q"],
            capture_output=True, text=True
        )


def load_model_gguf_bg(gguf_path):
    """Load a GGUF model using llama-cpp-python. No Ollama required."""
    global _model, _tokenizer, _ready, _status_msg
    try:
        try:
            from llama_cpp import Llama
        except ImportError:
            _status_msg = "Installing llama-cpp-python..."
            print("[server] llama-cpp-python not found, installing...")
            _install_llama_cpp()
            from llama_cpp import Llama

        gi = _gpu_info()
        # n_gpu_layers=-1 offloads all layers to GPU; 0 = CPU only
        n_gpu_layers = -1 if gi["has_cuda"] else 0
        gpu_label = f"{gi['name']} ({gi['vram_gb']:.1f} GB)" if gi["has_cuda"] else "CPU"

        _status_msg = f"Loading GGUF model on {gpu_label}..."
        print(f"[server] Loading GGUF: {gguf_path}")
        print(f"[server] Device: {gpu_label}  |  n_gpu_layers: {n_gpu_layers}")

        _model = Llama(
            model_path=gguf_path,
            n_ctx=4096,           # context window
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        # _tokenizer is not used for GGUF — llama-cpp handles tokenization internally
        _tokenizer = None
        _ready = True
        _status_msg = "Model ready (GGUF)"
        print("[server] GGUF model ready!")

    except Exception as e:
        _status_msg = f"Error loading GGUF model: {e}"
        print(f"[server] ERROR loading GGUF: {e}")


def generate_gguf(user_input, history, attachments=None, max_new_tokens=512, temperature=0.3, top_p=0.9):
    """Generate a response using the loaded llama-cpp-python GGUF model."""
    content = user_input
    if attachments:
        for att in attachments:
            if att.get("type") == "text":
                content += f"\n\n[Attached file: {att['name']}]\n{att['data']}"
            elif att.get("type") == "image":
                content += f"\n[Image attached: {att['name']} — text-only model]"

    # Build ChatML prompt manually (llama-cpp uses create_chat_completion which handles this)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": content})

    result = _model.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=1.1,
        stop=["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )
    reply = result["choices"][0]["message"]["content"].strip()
    return _sanitize_reply(reply)


def load_model_bg(model_path, use_4bit):
    global _model, _tokenizer, _ready, _status_msg
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cuda  = _has_cuda()
        bf16  = _bf16_supported()
        lora  = is_lora_adapter(model_path)
        base  = BASE_MODEL_FOR_LORA if lora else model_path

        tok_path = model_path
        if lora:
            tok_path = model_path if os.path.exists(
                os.path.join(model_path, "tokenizer_config.json")) else BASE_MODEL_FOR_LORA

        _status_msg = "Loading tokenizer..."
        print(f"[server] Loading tokenizer from: {tok_path}")
        _tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        _tokenizer.pad_token = _tokenizer.eos_token

        # Base models (non-instruct) don't have a chat template set.
        # Inject a suitable chat template so apply_chat_template() works.
        if not getattr(_tokenizer, "chat_template", None):
            _bm = base.lower()
            if "phi-2" in _bm or "phi2" in _bm:
                print("[server] No chat template found  -  injecting Phi-2 Instruct/Output template.")
                _tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                    "{{ message['content'] + '\\n\\n' }}"
                    "{% elif message['role'] == 'user' %}"
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
                print("[server] No chat template found  -  injecting standard ChatML template for base model.")
                _tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{'<|im_start|>assistant\\n'}}"
                    "{% endif %}"
                )

        _status_msg = "Loading model (this may take a minute)..."
        print(f"[server] Loading model from: {base}")

        if use_4bit and cuda:
            # Use float16 for 4-bit compute dtype on ALL GPU architectures.
            # Blackwell (SM 12.x, RTX 50xx): bfloat16 triggers CUBLAS_STATUS_INTERNAL_ERROR
            #   in the BNB dequantize matmul — float16 is safe on ALL architectures.
            # Ampere/Ada/Turing: float16 inference quality is identical to bfloat16.
            bnb_compute = torch.float16
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=bnb_compute,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                base,
                quantization_config=bnb_cfg,
                dtype=bnb_compute,   # match non-quantized layers to avoid cast overhead
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            dtype = torch.bfloat16 if bf16 and cuda else (torch.float16 if cuda else torch.float32)
            _model = AutoModelForCausalLM.from_pretrained(
                base, dtype=dtype, device_map="auto" if cuda else None, trust_remote_code=True)

        if lora:
            from peft import PeftModel
            _status_msg = "Loading LoRA adapter..."
            _model = PeftModel.from_pretrained(_model, model_path)
            if not (use_4bit and cuda):
                _model = _model.merge_and_unload()

        _model.eval()
        _ready = True
        _status_msg = "Model ready"
        print("[server] Model ready!")

    except Exception as e:
        _status_msg = f"Error loading model: {e}"
        print(f"[server] ERROR: {e}")


def generate(user_input, history, attachments=None, max_new_tokens=512, temperature=0.3, top_p=0.9):
    import torch

    # Build augmented message (include text file content if attached)
    content = user_input
    if attachments:
        for att in attachments:
            if att.get("type") == "text":
                content += f"\n\n[Attached file: {att['name']}]\n{att['data']}"
            elif att.get("type") == "image":
                content += f"\n[Image attached: {att['name']} — note: this model processes text only]"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": content})

    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt")
    if next(_model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Qwen2.5 ChatML uses <|im_end|> (token 151645) as the turn stop token
    # and <|endoftext|> (token 151643) as the document EOS.
    # The exported generation_config.json had eos_token_id=151643 only, which
    # meant <|im_end|> was never treated as a stop — the model ran forever.
    # Always pass both explicitly so generation stops correctly regardless of
    # what is saved in the model's generation_config.json.
    qwen_eos_ids = [151643, 151645]   # <|endoftext|>, <|im_end|>

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,            # always sample — greedy locks in repetition loops
            repetition_penalty=1.1,    # penalise already-seen tokens to break loops
            pad_token_id=151643,       # <|endoftext|>
            eos_token_id=qwen_eos_ids,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    reply = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _sanitize_reply(reply)


# ChatML / special tokens that can appear as literal strings in model output
# when skip_special_tokens=True doesn't catch them (e.g. the model outputs
# the token text rather than the token ID, or the tokenizer doesn't have them
# registered as special).  Ollama avoids this by doing string-level stop-
# sequence matching inside llama.cpp — we replicate that here.
_STOP_STRINGS = [
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|end|>",
    "<|eot_id|>",          # LLaMA-3 end-of-turn
    "<|start_header_id|>", # LLaMA-3 header start
    "<|end_header_id|>",   # LLaMA-3 header end
]

def _sanitize_reply(text: str) -> str:
    """
    Post-processing of model output to match what Ollama does internally:
    1. Truncate at the first stop-string (ChatML / special tokens as literal text).
    2. Truncate runaway repetition loops.
    3. Replace ISO country codes after location prepositions with full names.
    """
    # ── Step 1: truncate at any stop string (string-level, like Ollama) ──────
    # The model sometimes emits these as plain text rather than as token IDs,
    # especially when skip_special_tokens=True doesn't cover them all.
    for stop in _STOP_STRINGS:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    text = text.strip()

    # ── Step 2: strip any residual ChatML role lines from the start ───────────
    # e.g. "assistant\nHere is ..." → "Here is ..."
    text = re.sub(r'^(system|user|assistant)\s*\n', '', text, flags=re.IGNORECASE)
    text = text.strip()

    # ── Step 3: detect runaway repetition (any phrase repeated 6+ times) ─────
    if len(text) > 500:
        sample = text[:80].strip()
        if sample and text.count(sample) > 6:
            first_end = text.find(sample, len(sample))
            if first_end > 0:
                text = text[:first_end].rstrip()
                if not text:
                    return "(The model produced a repetitive response. Please try again.)"

    # ── Step 4: country code fixes ────────────────────────────────────────────
    text = re.sub(r'\bfrom\s+(?:KR|SK|FR)\b', 'from South Korea', text, flags=re.IGNORECASE)
    text = re.sub(r'\bin\s+(?:KR|SK)\b', 'in South Korea', text, flags=re.IGNORECASE)
    text = re.sub(r'\bKR\b', 'South Korea', text)
    text = re.sub(r'\bSK\b', 'South Korea', text)
    return text


# ── Flask app ──────────────────────────────────────────────────────────────────

def create_app():
    try:
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS
    except ImportError:
        print("[server] Missing packages. Run:")
        print("  pip install flask flask-cors")
        sys.exit(1)

    app = Flask(__name__, static_folder=TEST_DIR)
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(TEST_DIR, "chat.html")

    @app.route("/status")
    def status():
        return jsonify({"ready": _ready, "message": _status_msg, "model_name": _model_name})

    @app.route("/chat", methods=["POST"])
    def chat():
        if not _ready:
            return jsonify({"error": _status_msg}), 503

        data        = request.get_json(force=True)
        user_msg    = data.get("message", "").strip()
        history     = data.get("history", [])
        attachments = data.get("attachments", [])

        if not user_msg and not attachments:
            return jsonify({"error": "Empty message"}), 400

        try:
            # Route to GGUF or SafeTensors generate based on which loader was used
            if _use_gguf:
                reply = generate_gguf(user_msg, history, attachments)
            else:
                reply = generate(user_msg, history, attachments)
            return jsonify({"response": reply})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="LLM Chat Server")
    parser.add_argument("--model",   default=None, help="Model path (auto-detected if omitted)")
    parser.add_argument("--port",    type=int, default=5000)
    parser.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--no-open", dest="open_browser", action="store_false",
                        help="Don't auto-open browser")
    parser.set_defaults(use_4bit=True, open_browser=True)
    args = parser.parse_args()

    model_path = args.model or find_model()
    if not model_path:
        print("[server] No model found. Checked:")
        for c in MODEL_CANDIDATES:
            print(f"  {c}")
        print("\n[server] Train first:  python train/train.py")
        print("[server] Then export:  python generate_llm.py")
        sys.exit(1)

    global _use_gguf

    if os.path.isfile(model_path) and model_path.lower().endswith(".gguf"):
        # Explicit .gguf file path
        _use_gguf = True
    elif os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
        # If user passed a directory, check if it contains a .gguf — prefer it
        gguf_inside = find_gguf_in_dir(model_path)
        if gguf_inside:
            model_path = gguf_inside
            _use_gguf = True

    print("=" * 60)
    print("  LLM Chat Server - NeHe Productions")
    print("=" * 60)
    print(f"[server] Model     : {model_path}")
    print(f"[server] Format    : {'GGUF (llama-cpp-python)' if _use_gguf else 'SafeTensors (HuggingFace)'}")
    print(f"[server] Port      : {args.port}")
    print(f"[server] Chat UI   : http://localhost:{args.port}")
    print()

    # Load model in background thread so server starts immediately
    if _use_gguf:
        t = threading.Thread(target=load_model_gguf_bg, args=(model_path,), daemon=True)
    else:
        t = threading.Thread(target=load_model_bg, args=(model_path, args.use_4bit), daemon=True)
    t.start()

    if args.open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    app = create_app()
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
