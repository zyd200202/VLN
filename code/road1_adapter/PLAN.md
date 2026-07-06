# Road 1 Pilot — Degradation-Aware GLEE Head Fine-Tune

## Goal

Move the paper from "diagnosis + uniform negative results on inference-time
patches" to "diagnosis + uniform negative results + one positive datapoint
on the training-side direction (Road 1)" — without doing a full retraining
of InstructNav.

The minimum claim we want to support: degradation-aware fine-tuning of the
**GLEE detection head** (frozen SwinL backbone) recovers a measurable
fraction of the 26-pt reach-to-success gap that no inference-time arm
moved, **on the same paired N=150 protocol against the same baseline**.

## Why this is the right minimum experiment

1. **Cheap**: only the GLEE transformer decoder head (~tens of MB of
   parameters) is updated; SwinL backbone + CLIP text encoder stay frozen.
2. **No extra data**: we re-render HM3D `train` scenes via Habitat (NEVER
   touch the `val` episodes used in the main paper).
3. **Plug-in**: the new checkpoint loads through the existing
   `initialize_glee` code path with zero changes to InstructNav.
4. **Paper-tight**: the result lands in the exact same paired-N=150 cell
   that REVOKE/CRV/MUAP/FUSE/DXCV occupy, so the comparison is on the same
   episodes — no methodological wiggle.

## Pipeline

### Stage A — Data generation (offline, ~1.5h)

Walk through `train` split episodes in Habitat, log per step:

* `clean_rgb`  (H, W, 3) uint8
* `degraded_rgb` (low-light sev 4 applied deterministically)
* GLEE teacher outputs on `clean_rgb`: boxes, masks, classes, scores

Target: ~30 episodes × ~50 steps = ~1500 (clean, degraded, pseudo-label)
tuples. Stored as a single `.pt` shard at `/root/autodl-tmp/DRPhysNav/road1/data.pt`
(estimated ~3-5 GB).

The pseudo-labels are the **clean** GLEE outputs at threshold 0.10 — the
teacher's view of "what's actually there".

### Stage B — Head fine-tune (~3h on RTX 5090, head-only)

Initialise GLEE-SwinL from the existing 1.4 GB checkpoint. Freeze:

* `backbone.*` (Swin-L)
* `lang_encoder.*` (CLIP text)

Trainable:

* `transformer_decoder.*` (mask DINO decoder head)
* (optionally) `pixel_decoder` last 1-2 layers

Loss on each batch:
```
L = lambda_cls * KL(student_cls_logits(degraded), teacher_cls_logits(clean))
  + lambda_mask * BCE(student_mask(degraded), teacher_mask(clean))
  + lambda_box * L1(student_box(degraded), teacher_box(clean))
```
λ_cls = 1.0, λ_mask = 5.0, λ_box = 2.0.

Optimiser: AdamW, lr=1e-5, weight_decay=0.05, ~2000 steps, batch=4.

Stop criterion: validation pseudo-label-match score (top-1 class agreement
on a held-out 100-frame split) saturates or starts to drop.

Save: `glee_da_head_iter{NNN}.pth` checkpoints every 500 steps.

### Stage C — Plug-in & eval (~12h on N=150)

Add an env variable `DRPN_GLEE_DA_CKPT=/path/to/glee_da_head.pth` that, if
set, loads the DA head on top of the frozen base GLEE. Modify
`initialize_glee` to optionally apply this overlay.

Then run, paired against the **same** N=150 baseline used in Table 1:
```
ROAD1 + low_light sev 4 seed 0  ->  u_ROAD1_low_light_s4_seed0_N150.csv
```
McNemar exact + bootstrap CI as for the other arms.

### Stage D — Paper integration

1. **Tab 1**: add a final row
   ```
   Road 1 pilot   GLEE-DA head     Δ_R1   p_R1   150
   ```
2. **Sec.\ Probing the Multimodal Direction** -> rename to
   **"Probing the Training-Side Direction"** with two paragraphs:
   one on the DXCV inference-time multimodal probe (existing, null result),
   one on the GLEE-DA Road-1 pilot (new, positive result hopefully).
3. **Conclusion**: replace "we view degradation-aware training as the most
   promising next step" with "a head-only Road-1 fine-tune of the GLEE
   detector lifts paired SR by Δ_R1 (McNemar p=p_R1), confirming the
   direction; full backbone-and-policy retraining is the natural extension."
4. **Appendix**: add a HP + data-gen section.

## What it costs

| Stage | Time | GPU | Risk |
|-------|------|-----|------|
| Data gen | ~1.5h | RTX 5090 | low — same Habitat env as paper |
| Head fine-tune | ~3h | RTX 5090 | medium — head fine-tune may not converge cleanly |
| Eval (N=150) | ~12h | RTX 5090 | low — same protocol as unified queue |
| Paper update | ~1h | — | low |
| **Total** | **~17.5h** | single GPU | acceptable |

## Worst-case fallback

If GLEE training won't converge in 3h (mask DINO is finicky), fall back to:

**Plan B**: a smaller adapter. Insert two trainable rank-32 LoRA blocks on
the last 2 transformer decoder layers, train only those (~5M params). This
is even cheaper and more likely to land a small but real Δ.

If even that fails, **Plan C**: don't fine-tune GLEE at all; instead fine-tune
a small **feature-level restoration** head that maps degraded SwinL
features back toward clean SwinL features, and inject it as an adapter
between backbone and head. This is 1 single MLP, ~1h to train.

## Sequencing

```
[currently running queue]   <-- DO NOT INTERRUPT
       ↓ (~Mon evening / Tue morning)
[wait_then_road1.sh detects queue exit]
       ↓
[Stage A: gen data]      ~1.5h
       ↓
[Stage B: head fine-tune] ~3h
       ↓
[Stage C: paired N=150 eval] ~12h
       ↓
[fill paper + recompile]
```

Hand-off file when this is approved: `/root/autodl-tmp/DRPhysNav/road1/wait_then_road1.sh`
will be set up to:
1. block until `pgrep -f objnav_benchmark.py` returns empty,
2. then sequentially fire Stage A -> B -> C with logs to `road1/road1.log`.
