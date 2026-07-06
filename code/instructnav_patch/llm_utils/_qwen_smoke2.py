"""Realistic InstructNav prompt latency test on Qwen2-VL-7B.

Mimics the actual nav prompts (CHAINON for reasoning, GPT4V for direction choice)
to project total compute for paired N=150 x 4 deg x 2 arm runs.
"""
import os, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL = "/root/autodl-tmp/models/Qwen2-VL-7B-Instruct"

print("[smoke2] loading model...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL)
mdl = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="sdpa",
)
mdl.eval()
print(f"[smoke2] loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ---- Prompt 1: GPT4V style (multimodal direction choice) ----
GPT4V_PROMPT = (
    "You are an indoor navigation agent. I give you a panoramic observation image, complete "
    "navigation instruction and the sub-instruction you should execute now. Direction 1 and 11 "
    "are ahead, Direction 5 and 7 are back, Direction 3 is to the right, and Direction 9 is to "
    "the left. Please carefully analyze visual information in each direction and judge which "
    "direction is most suitable for next movement according to the act and landmark mentioned "
    "in the sub-instruction. You answer should follow \"Thinking Process\" and \"Judgement\". "
    "In the \"Judgement: \" field, you should only write down direction ID you choose. If you "
    "think you have arrived the destination, you can answer \"Stop\" in the \"Judgement: \" "
    "field. Note that the \"Direction 5\" and \"Direction 7\" are the directions you just came "
    "from. Generally, the direction with more navigation landmarks in the complete navigation "
    "instruction is better."
)
gpt4v_user = (
    "Complete instruction: Find the chair. Sub-instruction: Approach the chair in the dining room."
)
# 12-view panorama tile -> a wide image (typical InstructNav passes a 12-strip)
panorama = Image.fromarray((np.random.rand(384, 384 * 3, 3) * 255).astype("uint8"))

msgs = [
    {"role": "system", "content": GPT4V_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": panorama},
        {"type": "text",  "text": gpt4v_user},
    ]},
]
chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[chat], images=[panorama], return_tensors="pt").to("cuda:0")
print(f"[smoke2] GPT4V prompt: input_ids len={inputs.input_ids.shape[1]}", flush=True)

print("[smoke2] warmup...", flush=True)
with torch.inference_mode():
    _ = mdl.generate(**inputs, max_new_tokens=4, do_sample=False)
torch.cuda.synchronize()

# realistic max_new_tokens=512 (Thinking Process is verbose)
print("[smoke2] GPT4V style (max_new=512, deterministic):", flush=True)
times = []
for trial in range(3):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = mdl.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=proc.tokenizer.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t0
    gen = out.shape[1] - inputs.input_ids.shape[1]
    times.append(dt)
    txt = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"  trial{trial}: dt={dt:.2f}s | gen={gen} | tok/s={gen/dt:.1f} | out preview='{txt.strip()[:120].replace(chr(10),' ')}'", flush=True)
gpt4v_avg = sum(times)/len(times)
print(f"  >> avg GPT4V latency: {gpt4v_avg:.2f}s/call", flush=True)

# ---- Prompt 2: CHAINON style (long text reasoning, no image) ----
CHAINON = (
    "You are a wheeled mobile robot working in an indoor environment. And you are required to "
    "finish a navigation task indicated by a human instruction in a new house. Your task is to "
    "make a navigation plan ... [truncated for test] ... formatted as Answer={'Reason':<...>, "
    "'Action':<...>, 'Landmark':<...>, 'Flag':<...>}."
)
chainon_user = (
    "<Navigation Instruction>: Find the chair.\n"
    "<Previous Plan>: Explore - Living_Room - Approach - Sofa\n"
    "<Semantic Clue>: rooms=[living_room, kitchen, bedroom], objects=[sofa(0.4m), tv(2.1m), table(1.5m), bed(5.7m), microwave(8.2m)]"
)
msgs2 = [
    {"role": "system", "content": CHAINON},
    {"role": "user", "content": chainon_user},
]
chat2 = proc.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
inputs2 = proc(text=[chat2], return_tensors="pt").to("cuda:0")
print(f"\n[smoke2] CHAINON prompt: input_ids len={inputs2.input_ids.shape[1]}", flush=True)

print("[smoke2] CHAINON (max_new=512, deterministic):", flush=True)
times = []
for trial in range(3):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = mdl.generate(**inputs2, max_new_tokens=512, do_sample=False, pad_token_id=proc.tokenizer.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t0
    gen = out.shape[1] - inputs2.input_ids.shape[1]
    times.append(dt)
    txt = proc.batch_decode(out[:, inputs2.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"  trial{trial}: dt={dt:.2f}s | gen={gen} | tok/s={gen/dt:.1f} | out preview='{txt.strip()[:120].replace(chr(10),' ')}'", flush=True)
chainon_avg = sum(times)/len(times)
print(f"  >> avg CHAINON latency: {chainon_avg:.2f}s/call", flush=True)

# ---- Project total time ----
# InstructNav typically: per step ~ 1 CHAINON + (occasionally) 1 GPT4V vote
# Episode length ~ 100-300 steps in HM3D ObjectNav (avg ~ 200)
# We assume: per ep ~ 30 CHAINON calls + 30 GPT4V calls (rough)
calls_per_ep_chainon = 30
calls_per_ep_gpt4v   = 30
sec_per_ep = calls_per_ep_chainon * chainon_avg + calls_per_ep_gpt4v * gpt4v_avg
print(f"\n[smoke2] projected per-ep VLM time @ ({calls_per_ep_chainon}xCHAINON + {calls_per_ep_gpt4v}xGPT4V) = {sec_per_ep:.0f}s", flush=True)
for label, neps in [("1 arm x 1 deg x N=150", 150),
                    ("2 arms x 1 deg x N=150", 300),
                    ("2 arms x 4 deg x N=150 (full)", 1200),
                    ("2 arms x 2 deg x N=150 (low_light + motion_blur)", 600)]:
    h = neps * sec_per_ep / 3600
    print(f"  {label:50s}: {h:.1f} h", flush=True)

print(f"\n[smoke2] GPU mem peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
