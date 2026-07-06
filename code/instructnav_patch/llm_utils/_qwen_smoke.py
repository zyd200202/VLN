"""Latency smoke test for Qwen2-VL-7B as InstructNav VLM backbone.

Measures end-to-end latency for a single multimodal query (vision + text)
to project the total time for paired N=150 x 4 degradations x 2 arms.
"""
import os, time, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct"

print("[smoke] importing torch / transformers...", flush=True)
t0 = time.time()
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
print(f"[smoke]   done in {time.time()-t0:.1f}s. torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

print("[smoke] loading processor...", flush=True)
t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL)
print(f"[smoke]   done in {time.time()-t0:.1f}s", flush=True)

print("[smoke] loading model in fp16 on cuda:0...", flush=True)
t0 = time.time()
mdl = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    attn_implementation="sdpa",
)
mdl.eval()
print(f"[smoke]   done in {time.time()-t0:.1f}s | GPU mem alloc: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# fake an InstructNav-style multimodal call: 4-view panorama choice
import numpy as np
from PIL import Image

dummy = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
sys_p = "You are a robot navigator. Choose 0/1/2/3 to point to a chair."
text_p = "Given the 4-direction views, return only an integer 0-3."

messages = [
    {"role": "system", "content": sys_p},
    {"role": "user", "content": [
        {"type": "image", "image": dummy},
        {"type": "text",  "text": text_p},
    ]},
]
chat_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[chat_text], images=[dummy], return_tensors="pt", padding=True).to("cuda:0")

print("[smoke] warmup pass (1 token)...", flush=True)
with torch.inference_mode():
    _ = mdl.generate(**inputs, max_new_tokens=4, do_sample=False)
torch.cuda.synchronize()

print("[smoke] running 3 typed latency trials (max_new_tokens=64, do_sample=False)...", flush=True)
for trial in range(3):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = mdl.generate(**inputs, max_new_tokens=64, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    txt = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"[smoke]   trial{trial}: dt={dt:.2f}s | gen_tokens={gen_tokens} | tok/s={gen_tokens/dt:.1f} | out='{txt.strip()[:80]}'", flush=True)

print("[smoke] running 2 trials with max_new_tokens=1000 (InstructNav default)...", flush=True)
for trial in range(2):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = mdl.generate(**inputs, max_new_tokens=1000, do_sample=False)
    torch.cuda.synchronize(); dt = time.time() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    print(f"[smoke]   trial{trial}: dt={dt:.2f}s | gen_tokens={gen_tokens} | tok/s={gen_tokens/dt:.1f}", flush=True)

print(f"[smoke] final GPU mem alloc: {torch.cuda.memory_allocated()/1e9:.1f} GB / max: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
print("[smoke] done.", flush=True)
