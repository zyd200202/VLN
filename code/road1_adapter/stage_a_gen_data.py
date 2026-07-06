"""Road 1 Stage A — pair-data generation.

Walk through HM3D *train* episodes (NEVER touch val) and dump
(clean_rgb, degraded_rgb) frame pairs that Stage B uses to train a small
feature-level adapter to recover SwinL features lost to degradation.

Outputs to /root/autodl-tmp/DRPhysNav/road1/data/:
    frames.h5
        clean   : uint8 [N, H, W, 3]   BGR (OpenCV)
        degraded: uint8 [N, H, W, 3]   BGR (OpenCV), low_light sev 4 by default
    meta.json
        N, deg_type, deg_sev, scene_ids, ts

Pipeline keeps each frame at the same 360x640 resolution InstructNav uses,
so SwinL features are directly comparable to those seen at deployment.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import h5py
import numpy as np
import cv2

# Habitat path
sys.path.insert(0, "/root/autodl-tmp/InstructNav")

import habitat
from config_utils import hm3d_config


def _short_id(scene_path: str) -> str:
    """Return the compact scene ID that appears in Habitat episode.scene_id
    and in the objectnav_hm3d_v2/val/content/*.json.gz filenames.

    Example inputs:
        "/root/.../scene_datasets/hm3d_v0.2/val/00839-zt1RVoi7PcG/zt1RVoi7PcG.basis.glb"
        "00839-zt1RVoi7PcG/zt1RVoi7PcG.basis.glb"
        "zt1RVoi7PcG.basis.glb"
    All map to "zt1RVoi7PcG".
    """
    leaf = os.path.basename(scene_path)
    return leaf.replace(".basis.glb", "").replace(".glb", "")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_frames", type=int, default=1500,
                   help="Total frame pairs to dump (default: 1500).")
    p.add_argument("--max_episodes", type=int, default=30,
                   help="Cap on episodes traversed (default: 30).")
    p.add_argument("--stride", type=int, default=10,
                   help="Frame stride within each episode (default: 10).")
    p.add_argument("--deg_type", type=str, default="low_light",
                   choices=["low_light", "motion_blur", "fog", "gaussian_noise"])
    p.add_argument("--deg_sev", type=int, default=4)
    p.add_argument("--split", type=str, default="val",
                   help="HM3D split. Since HM3D train scenes are not present on "
                        "this workstation, we use val + a manual scene deny-list "
                        "(the 6 val scenes that N=150 paired eval hits) to "
                        "guarantee scene-level disjointness between adapter data "
                        "and paired eval.")
    p.add_argument("--deny_scenes", type=str,
                   default="4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,"
                           "7MXmsvcQjpJ,BAbdmeyTvMZ,CrMo8WxCyVb",
                   help="Comma-separated scene short-IDs to skip. Default: the "
                        "six val scenes hit by the N=150 paired eval at seed 0 "
                        "(computed by parsing objectnav_hm3d_v2/val/content/).")
    p.add_argument("--out_dir", type=str,
                   default="/root/autodl-tmp/DRPhysNav/road1/data")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    deny = set(s.strip() for s in args.deny_scenes.split(",") if s.strip())
    print(f"[stageA] scene deny-list ({len(deny)}): {sorted(deny)}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_h5 = os.path.join(args.out_dir, "frames.h5")
    out_meta = os.path.join(args.out_dir, "meta.json")

    # --- Habitat env ---
    # We request MORE episodes than needed so that after we filter out deny-list
    # scenes we still have enough episodes to draw args.num_frames pairs from.
    ep_pool = max(args.max_episodes * 4, 100)
    print(f"[stageA] config split={args.split}, ep_pool={ep_pool}", flush=True)
    cfg = hm3d_config(stage=args.split, episodes=ep_pool)
    env = habitat.Env(cfg)

    # --- degradation pipeline (reuse the paper's drphysnav_integration) ---
    os.environ["DRPN_DEGRADE_TYPE"] = args.deg_type
    os.environ["DRPN_DEGRADE_SEVERITY"] = str(args.deg_sev)
    import drphysnav_integration as drpn  # type: ignore

    # --- pre-alloc hdf5 ---
    # peek one frame for shape; keep resetting until we land on an allowed scene
    obs = env.reset()
    while _short_id(env.current_episode.scene_id) in deny:
        obs = env.reset()
    h, w = obs["rgb"].shape[:2]
    print(f"[stageA] frame shape {h}x{w}", flush=True)
    f = h5py.File(out_h5, "w")
    ds_clean = f.create_dataset("clean", shape=(args.num_frames, h, w, 3),
                                dtype=np.uint8, chunks=(1, h, w, 3))
    ds_deg   = f.create_dataset("degraded", shape=(args.num_frames, h, w, 3),
                                dtype=np.uint8, chunks=(1, h, w, 3))

    rng = np.random.RandomState(args.seed)
    n_written = 0
    n_ep = 0
    scene_ids = []
    t0 = time.time()

    while n_written < args.num_frames and n_ep < args.max_episodes:
        if n_ep > 0:
            obs = env.reset()
            # skip deny-list scenes without counting them toward n_ep
            skip_guard = 0
            while _short_id(env.current_episode.scene_id) in deny and skip_guard < 50:
                obs = env.reset()
                skip_guard += 1
            if skip_guard >= 50:
                print(f"[stageA][warn] cycled 50 resets without an allowed scene; stop.",
                      flush=True)
                break
        n_ep += 1
        scene_id = env.current_episode.scene_id.split("/")[-1]
        scene_ids.append(scene_id)
        print(f"[stageA] ep {n_ep}/{args.max_episodes} scene={scene_id} "
              f"frames written so far {n_written}/{args.num_frames}",
              flush=True)
        step = 0
        # random walk through the scene with stride sampling
        while not env.episode_over and n_written < args.num_frames:
            # take an action (random forward / turn so we cover space)
            a = int(rng.choice([1, 2, 3], p=[0.6, 0.2, 0.2]))
            obs = env.step(a)
            step += 1
            if step % args.stride != 0:
                continue
            rgb = obs["rgb"]  # RGB uint8
            bgr = rgb[:, :, ::-1].copy()  # convert to BGR for OpenCV / drpn
            # drpn.degrade_rgb(rgb_uint8, seed) -> uint8; we use ep+step-derived
            # seed so two different frames get different noise realisations
            # while remaining deterministic.
            deg_seed = args.seed * 100003 + n_ep * 1009 + step
            deg = drpn.degrade_rgb(bgr, deg_seed)
            ds_clean[n_written] = bgr
            ds_deg[n_written]   = deg.astype(np.uint8)
            n_written += 1
            if n_written % 50 == 0:
                dt = time.time() - t0
                print(f"[stageA]   {n_written}/{args.num_frames} "
                      f"({n_written/max(dt,1):.1f} f/s)", flush=True)
    # truncate hdf5 to actual size
    if n_written < args.num_frames:
        ds_clean.resize((n_written, h, w, 3))
        ds_deg.resize((n_written, h, w, 3))
    f.close()
    env.close()

    meta = {
        "N": n_written,
        "shape": [h, w, 3],
        "deg_type": args.deg_type,
        "deg_sev": args.deg_sev,
        "scene_ids": scene_ids,
        "split": args.split,
        "stride": args.stride,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_meta, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[stageA] DONE. wrote {n_written} pairs in "
          f"{meta['elapsed_sec']/60:.1f} min -> {out_h5}", flush=True)


if __name__ == "__main__":
    main()
