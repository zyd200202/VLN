"""Render photoreal top-down (bird's-eye) RGB crops for the trajectory
vignette episodes, aligned to world coordinates.

For each episode we:
  1. load the HM3D scene in habitat_sim with an ORTHOGRAPHIC RGB sensor
     looking straight down (-Y),
  2. clip the ceiling with the near plane (camera high above the floor,
     near = cam_y - (floor_y + CEIL_M)),
  3. calibrate px-per-meter empirically by shifting the camera 1 m in +X
     and cross-correlating the two renders,
  4. save the render + a JSON with (center_x, center_z, px_per_m) so the
     figure script can map world coords -> pixels.

Run inside the habitat conda env:
  python render_topdown_rgb.py
"""
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim  # noqa: E402

RUNS = Path("/root/autodl-tmp/DRPhysNav/runs/traj_vignette")
OUT = RUNS / "topdown_rgb"
OUT.mkdir(exist_ok=True)
SCENES_ROOT = Path("/root/autodl-tmp/navwork/data/versioned_data/hm3d-0.2/hm3d/val")

EPISODES = [2, 4, 6, 7, 8]
RES = 1400            # square render resolution
CEIL_M = 2.0          # keep geometry up to this height above the floor
CAM_ELEV = 50.0       # camera height above floor (m); near plane cuts the rest


def find_scene(glb_name):
    hits = list(SCENES_ROOT.glob(f"*/{glb_name}"))
    assert hits, f"scene {glb_name} not found under {SCENES_ROOT}"
    return str(hits[0])


def load_ep_meta(ep):
    for line in open(RUNS / "tv_B0_low_light_s4_seed0_traj.jsonl"):
        d = json.loads(line)
        if d["ep"] == ep:
            return d
    raise KeyError(ep)


def make_sim(scene_glb, ortho_scale, near):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.enable_physics = False

    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "ortho_rgb"
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    spec.resolution = [RES, RES]
    spec.position = [0.0, 0.0, 0.0]
    spec.orientation = [-np.pi / 2, 0.0, 0.0]   # pitch down
    spec.ortho_scale = ortho_scale
    spec.near = near
    spec.far = 200.0

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def render_at(sim, x, z, cam_y):
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.array([x, cam_y, z], dtype=np.float32)
    state.rotation = np.quaternion(1, 0, 0, 0)
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()["ortho_rgb"]
    return obs[..., :3].copy()


def calibrate_px_per_m(sim, x, z, cam_y):
    """Shift camera +1 m in X, cross-correlate column profiles for px shift."""
    a = render_at(sim, x, z, cam_y).astype(float).mean(axis=2)
    b = render_at(sim, x + 1.0, z, cam_y).astype(float).mean(axis=2)
    # +X shift moves content left in the image by s pixels (columns)
    pa = a.mean(axis=0) - a.mean()
    pb = b.mean(axis=0) - b.mean()
    corr = np.correlate(pa, pb, mode="full")
    shift = np.argmax(corr) - (len(pa) - 1)
    return abs(float(shift))


def main():
    meta_out = {}
    for ep in EPISODES:
        d = load_ep_meta(ep)
        scene = find_scene(d["scene"])
        traj = np.asarray(d["traj_world"], float)
        goals = np.asarray(d["goals_world"], float)
        floor_y = float(np.median(traj[:, 1]))

        # crop bounds in world XZ over all three arms + goals
        xs, zs = [goals[:, 0]], [goals[:, 2]]
        for arm in ("B0", "RES", "ORACLEGATE"):
            for line in open(RUNS / f"tv_{arm}_low_light_s4_seed0_traj.jsonl"):
                r = json.loads(line)
                if r["ep"] == ep:
                    t = np.asarray(r["traj_world"], float)
                    xs.append(t[:, 0]); zs.append(t[:, 2])
                    break
        xs = np.concatenate(xs); zs = np.concatenate(zs)
        cx, cz = (xs.min() + xs.max()) / 2, (zs.min() + zs.max()) / 2
        span = max(xs.max() - xs.min(), zs.max() - zs.min()) + 4.0  # margin

        # habitat-sim ortho camera shows 1/ortho_scale world units
        # vertically (verified by calibration); px_per_m is still measured
        # empirically below.
        guess = 1.0 / span
        cam_y = floor_y + CAM_ELEV
        near = CAM_ELEV - CEIL_M

        sim = make_sim(scene, guess, near)
        try:
            ppm = calibrate_px_per_m(sim, cx, cz, cam_y)
            img = render_at(sim, cx, cz, cam_y)
        finally:
            sim.close()

        np.save(OUT / f"ep{ep:03d}_rgb.npy", img)
        meta_out[str(ep)] = dict(center_x=cx, center_z=cz, px_per_m=ppm,
                                 floor_y=floor_y, res=RES, span_m=span,
                                 scene=d["scene"])
        print(f"ep{ep}: scene={d['scene']} center=({cx:.2f},{cz:.2f}) "
              f"span={span:.1f}m px/m={ppm:.2f}")

    with open(OUT / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    print("wrote", OUT / "meta.json")


if __name__ == "__main__":
    main()
