"""Qwen2-VL-7B backend with the same API surface as llm_utils/gpt_request:

    gpt_response(text_prompt, system_prompt="")               -> str
    gptv_response(text_prompt, image_prompt, system_prompt="") -> str

Drop-in replacement for the GPT-4o Azure client used inside InstructNav,
intended for the "second backbone" experiment (cross-VLM consistency check).

Toggled by setting environment variable DRPN_BACKBONE=qwen2vl before launch;
the objnav_benchmark driver monkey-patches llm_utils.gpt_request when this
flag is on. When the flag is off, this module is never imported.

Design notes:
 - Loads Qwen2-VL-7B-Instruct once at process start (~10 s on RTX 5090) and
   reuses the global handle across all calls. Eager warmup pass to amortize.
 - Generation is greedy (do_sample=False) to mirror deterministic
   InstructNav-GPT4o evaluation. Per smoke trials:
       multimodal (panorama input) : ~1.0 s/call @ 60 tok/s
       text-only  (CHAINON)        : ~0.5 s/call @ 70 tok/s
 - Accepts the same `image_prompt` types as the GPT-4o helper: a file path,
   a numpy BGR array (OpenCV convention used by InstructNav panorama), or a
   PIL.Image. Internally normalizes to PIL RGB.
 - Output post-processing strips Qwen's chat-template echoes / system tokens
   so downstream regex parsing (Action/Landmark/Direction/Stop) keeps working.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image

_MODEL_DIR = os.environ.get(
    "QWEN2VL_MODEL",
    "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct",
)
_MAX_NEW_TOKENS = int(os.environ.get("QWEN2VL_MAX_NEW", "256"))
_GREEDY = os.environ.get("QWEN2VL_GREEDY", "1") == "1"

_lock = threading.Lock()
_proc = None
_mdl  = None
_initialized = False


def _is_main_process() -> bool:
    return os.environ.get("LOCAL_RANK", "0") in ("0", "")


def _ensure_loaded():
    """Lazy global load, thread-safe. Idempotent."""
    global _proc, _mdl, _initialized
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
        )
        t0 = time.time()
        print(f"[qwen-backend] loading processor from {_MODEL_DIR}", flush=True)
        _proc = AutoProcessor.from_pretrained(_MODEL_DIR)
        print(f"[qwen-backend] loading model fp16 on cuda:0 ...", flush=True)
        _mdl = Qwen2VLForConditionalGeneration.from_pretrained(
            _MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        _mdl.eval()
        torch.cuda.synchronize()
        print(
            f"[qwen-backend] ready in {time.time()-t0:.1f}s | "
            f"GPU mem alloc {torch.cuda.memory_allocated()/1e9:.1f} GB | "
            f"max_new={_MAX_NEW_TOKENS} greedy={_GREEDY}",
            flush=True,
        )
        _initialized = True


def _to_pil_rgb(image) -> Image.Image:
    """Normalize the heterogeneous image argument shape to a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        # InstructNav passes BGR (OpenCV convention) for panoramas
        if image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        if image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(image[:, :, [2, 1, 0]])  # BGRA-ish drop A
        return Image.fromarray(image)
    raise TypeError(f"Unsupported image type: {type(image)}")


@torch.inference_mode()
def _generate(messages, images) -> str:
    chat = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images:
        inputs = _proc(text=[chat], images=images, return_tensors="pt", padding=True)
    else:
        inputs = _proc(text=[chat], return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda:0", non_blocking=True) for k, v in inputs.items()}
    gen = _mdl.generate(
        **inputs,
        max_new_tokens=_MAX_NEW_TOKENS,
        do_sample=not _GREEDY,
        pad_token_id=_proc.tokenizer.eos_token_id,
    )
    inp_len = inputs["input_ids"].shape[1]
    txt = _proc.batch_decode(gen[:, inp_len:], skip_special_tokens=True)[0]
    return txt.strip()


def gpt_response(text_prompt: str, system_prompt: str = "") -> str:
    """Text-only completion. Drop-in for llm_utils.gpt_request.gpt_response."""
    _ensure_loaded()
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": text_prompt})
    return _generate(msgs, images=[])


def gptv_response(text_prompt: str, image_prompt, system_prompt: str = "") -> str:
    """Multimodal completion. Drop-in for llm_utils.gpt_request.gptv_response."""
    _ensure_loaded()
    pil = _to_pil_rgb(image_prompt)
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image", "image": pil},
            {"type": "text",  "text": text_prompt},
        ],
    })
    return _generate(msgs, images=[pil])


def warmup():
    """Optional eager warmup -- helps amortize the first call's CUDA graph build."""
    _ensure_loaded()
    try:
        _ = gpt_response("Say 'ready'.", system_prompt="You are a test fixture.")
    except Exception as e:
        print(f"[qwen-backend][warn] warmup gpt_response failed: {e}", flush=True)
    try:
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        _ = gptv_response("Describe the image in one word.", dummy)
    except Exception as e:
        print(f"[qwen-backend][warn] warmup gptv_response failed: {e}", flush=True)
    print("[qwen-backend] warmup complete.", flush=True)


def describe() -> dict:
    return {
        "backbone": "qwen2vl-7b-instruct",
        "model_dir": _MODEL_DIR,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "greedy": _GREEDY,
        "initialized": _initialized,
    }
