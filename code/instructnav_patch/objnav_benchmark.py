import os
# --- VLM backbone switch (must happen BEFORE objnav_agent is imported, because
# the pyc binds gpt_response / gptv_response into its own globals at import time). ---
if os.environ.get("DRPN_BACKBONE", "").lower() in ("qwen2vl", "qwen2_vl", "qwen"):
    import sys as _sys, types as _types
    from llm_utils import qwen_backend as _qwen
    _qwen._ensure_loaded()
    _stub = _types.ModuleType("llm_utils.gpt_request")
    _stub.gpt_response  = _qwen.gpt_response
    _stub.gptv_response = _qwen.gptv_response
    _sys.modules["llm_utils.gpt_request"] = _stub
    print("[DRPhysNav] VLM backbone = Qwen2-VL-7B (sys.modules['llm_utils.gpt_request'] redirected).", flush=True)

# --- Road-1 feature adapter overlay (also pre-import, because objnav_agent.pyc
# binds initialize_glee at import time). Activated by DRPN_ROAD1_ADAPTER=<ckpt>. ---
if os.environ.get("DRPN_ROAD1_ADAPTER", ""):
    import sys as _sys
    from cv_utils import glee_detector as _gd
    _orig_init_glee = _gd.initialize_glee
    def _patched_init_glee(*a, **kw):
        m = _orig_init_glee(*a, **kw)
        try:
            from llm_utils import road1_adapter as _ra
            _ra.maybe_activate_from_env(m)
        except Exception as _e:
            print(f"[DRPhysNav][road1][warn] adapter activation failed: {_e}", flush=True)
        return m
    _gd.initialize_glee = _patched_init_glee
    print(f"[DRPhysNav] Road-1 adapter overlay armed: {os.environ['DRPN_ROAD1_ADAPTER']}",
          flush=True)

import habitat
import argparse
import csv
from tqdm import tqdm
from config_utils import hm3d_config,mp3d_config
from mapping_utils.transform import habitat_camera_intrinsic
from mapper import Instruct_Mapper
from objnav_agent import HM3D_Objnav_Agent
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
def write_metrics(metrics,path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=500)
    parser.add_argument("--split",type=str,default="val_mini")
    parser.add_argument("--out_csv",type=str,default="objnav_hm3d.csv")
    parser.add_argument("--save_traj",type=int,default=0)
    parser.add_argument("--mapper_resolution",type=float,default=0.05)
    parser.add_argument("--path_resolution",type=float,default=0.2)
    parser.add_argument("--path_scale",type=int,default=5)
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    args = get_args()
    os.makedirs("./tmp", exist_ok=True)
    import drphysnav_integration as drpn
    import torch, random as _random
    import numpy as np
    _random.seed(drpn.SEED); np.random.seed(drpn.SEED); torch.manual_seed(drpn.SEED)
    torch.cuda.manual_seed_all(drpn.SEED)
    print("[DRPhysNav] condition:", drpn.describe(), "| split:", args.split,
          "| seed:", drpn.SEED, flush=True)
    # --- Component B: pluggable strong restorer (PromptIR, NeurIPS'23 all-in-one blind restoration) ---
    # Replaces drpn.restore_rgb with the pretrained PromptIR network so every downstream restore call
    # (blanket RESTORE arm, FUSE restored path, DARE router) uses the learned restorer instead of the
    # handcrafted CLAHE/DCP/NLM pipeline. Honors drpn.USE_RESTORE; falls back to handcrafted on error.
    if os.environ.get("DRPN_RESTORER", "").lower() == "promptir":
        import promptir_restore as _pir
        _orig_restore_hc = drpn.restore_rgb
        def _pir_restore(bgr=None):
            if not drpn.USE_RESTORE:
                return bgr
            try:
                return _pir.restore_bgr(bgr)
            except Exception as _e:
                print("[PromptIR][WARN] inference failed -> handcrafted fallback:", _e, flush=True)
                return _orig_restore_hc(bgr)
        drpn.restore_rgb = _pir_restore
        try:
            _pir._load()   # eager load -> fail fast + warmup before the episode loop
        except Exception as _e:
            print("[PromptIR][WARN] eager load failed:", _e, flush=True)
        print("[DRPhysNav] RESTORER=PromptIR (NeurIPS'23 all-in-one) active", flush=True)

    # --- Reliability-gated Directional Decision Arbitration (DDA) ---
    # InstructNav fuses the navigation value map as 0.25*semantic + 0.25*action + 0.25*gpt4v + 0.25*history.
    # The gpt4v term is the only one read off the (degraded) RGB panorama; under visual degradation it is
    # unreliable. DDA trusts it less when the current frame reliability r is low and redistributes its weight
    # to image-quality-independent cues (semantic memory, commonsense action affordance, history geometry).
    # On clean frames (r=1) it reduces exactly to the original 0.25/0.25/0.25/0.25 -> baseline-identical.
    if os.environ.get("DRPN_USE_DDA", "0") == "1":
        import numpy as _np
        _dda_level = int(os.environ.get("DRPN_DDA_LEVEL", "1"))    # 1=global r; 2=per-direction r_d + consistency
        _dda_min = float(os.environ.get("DRPN_DDA_MIN_RGATE", "1.0"))  # L1: arbitrate only when r < this
        # L2 hook: stash per-direction reliability (reliability of the view GPT4V actually selected).
        if _dda_level >= 2:
            _orig_qgptv = HM3D_Objnav_Agent.query_gpt4v
            def _dda_query_gpt4v(self):
                ans = _orig_qgptv(self)
                try:
                    self.mapper._dda_dir_rels = [float(drpn.frame_reliability(im)) for im in self.temporary_images]
                    self.mapper._dda_dir_idx = int(ans)
                except Exception as _e:
                    self.mapper._dda_dir_rels = []; self.mapper._dda_dir_idx = None
                return ans
            HM3D_Objnav_Agent.query_gpt4v = _dda_query_gpt4v
        def _dda_affordance_map(self, action, target_class, gpt4v_pcd, complete_flag=False, failure_mode=False):
            if failure_mode:
                obstacle_affordance = self.get_obstacle_affordance()
                aff = self.get_action_affordance('Explore')
                aff = _np.clip(aff, 0.1, 1.0); aff[obstacle_affordance == 0] = 0
                return aff, self.visualize_affordance(aff)
            if complete_flag:
                aff = self.get_semantic_affordance([target_class], threshold=0.1)
                return aff, self.visualize_affordance(aff)
            obstacle_affordance = self.get_obstacle_affordance()
            semantic_affordance = self.get_semantic_affordance([target_class], threshold=1.5)
            action_affordance = self.get_action_affordance(action)
            gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
            history_affordance = self.get_trajectory_affordance()
            if _dda_level >= 2:
                # per-direction reliability: reliability of the view GPT4V selected
                rels = getattr(self, "_dda_dir_rels", None); idx = getattr(self, "_dda_dir_idx", None)
                if rels and idx is not None and 0 <= idx < len(rels):
                    r_d = float(rels[idx])
                else:
                    r_d = float(getattr(self, "uap_reliability", 1.0))
                r_d = min(1.0, max(0.0, r_d))
                # cross-cue consistency: do robust cues corroborate gpt4v's chosen spot?
                robust = semantic_affordance + action_affordance + history_affordance
                rmax = float(robust.max()) + 1e-6
                gpeak = int(_np.argmax(gpt4v_affordance))
                consistency = min(1.0, max(0.0, float(robust[gpeak]) / rmax))
                trust = r_d + (1.0 - r_d) * consistency
                w_g = 0.25 * trust; w_o = 0.25 + (0.25 - w_g) / 3.0
            else:
                r = float(getattr(self, "uap_reliability", 1.0)); r = min(1.0, max(0.0, r))
                if r >= _dda_min:
                    w_g = 0.25; w_o = 0.25
                else:
                    w_g = 0.25 * r; w_o = 0.25 + (0.25 - w_g) / 3.0
            aff = w_o*semantic_affordance + w_o*action_affordance + w_g*gpt4v_affordance + w_o*history_affordance
            aff = _np.clip(aff, 0.1, 1.0); aff[obstacle_affordance == 0] = 0
            return aff, self.visualize_affordance(aff/(aff.max()+1e-6))
        Instruct_Mapper.get_objnav_affordance_map = _dda_affordance_map
        print("[DRPhysNav] DDA enabled: level=%d (1=global r, 2=per-direction r_d + consistency)" % _dda_level, flush=True)

    # --- Confirm-or-REVOKE recovery (commitment-layer module 3) ---
    # Diagnosis (N=300): once InstructNav sets found_goal=True it beelines to plan_position and STOPS
    # there irreversibly (step() never re-plans while found_goal is True -> bytecode-confirmed). Under
    # degradation the agent commits at low reliability (r 0.95->0.41), 76% of commits are wrong, and it
    # sticks ~12 steps barely approaching (dist improvement 1.44m clean -> 0.52m low_light). Unlike RES/M1g
    # (perception layer, 0 oracle headroom), this targets the commit/stop policy (reach 0.62 vs SR 0.36 ->
    # ~0.26 headroom). Mechanism mirrors ConsistNav's bounded recovery/failover (same-stack +8~11% SR).
    # Signal is SELF-supervised (agent's own distance to its OWN committed target, never the true goal):
    # if a committed agent fails to reduce move_distance to plan_position for K steps AND is still far,
    # the commitment is unreachable/bad -> revoke (found_goal=False), blacklist that spot, resume search;
    # veto re-committing within COOLDOWN metres of a blacklisted spot. Bounded by MAX revokes/episode.
    if os.environ.get("DRPN_USE_REVOKE", "0") == "1":
        import numpy as _np
        import atexit as _atexit
        _rk_K    = int(os.environ.get("DRPN_REVOKE_STALL_K", "6"))      # committed steps w/o progress -> revoke
        _rk_eps  = float(os.environ.get("DRPN_REVOKE_EPS", "0.15"))     # min move_distance drop (m) counted as progress
        _rk_far  = float(os.environ.get("DRPN_REVOKE_FAR", "0.8"))      # only revoke if still >this far from own target (not a legit arrival)
        _rk_max  = int(os.environ.get("DRPN_REVOKE_MAX", "2"))          # max revokes per episode (loop guard)
        _rk_cool = float(os.environ.get("DRPN_REVOKE_COOLDOWN", "2.0")) # m: veto re-commit within this of a blacklisted spot
        _rk_stat = {"revoke": 0, "veto": 0, "commit_eps": 0}
        _orig_step_rk  = HM3D_Objnav_Agent.step
        _orig_reset_rk = HM3D_Objnav_Agent.reset
        _orig_mkplan_rk = HM3D_Objnav_Agent.make_plan
        def _rk_dist(self):
            return float(_np.sqrt(_np.square(_np.array(self.mapper.current_position) - _np.array(self.plan_position)).sum()))
        def _rk_reset(self, *a, **k):
            r = _orig_reset_rk(self, *a, **k)
            self._rk_best = None; self._rk_stall = 0; self._rk_n = 0; self._rk_black = []; self._rk_was_committed = False
            return r
        def _rk_make_plan(self, *a, **k):
            out = _orig_mkplan_rk(self, *a, **k)
            try:
                if getattr(self, "found_goal", False) and getattr(self, "_rk_black", None):
                    tgt = _np.array(self.plan_position)
                    if any(float(_np.sqrt(_np.square(tgt - _np.array(b)).sum())) < _rk_cool for b in self._rk_black):
                        self.found_goal = False
                        _rk_stat["veto"] += 1
                        print("[DRPhysNav][REVOKE] veto re-commit within %.1fm of a blacklisted spot -> keep searching" % _rk_cool, flush=True)
            except Exception:
                pass
            return out
        def _rk_step(self, *a, **k):
            try:
                if getattr(self, "found_goal", False):
                    if not self._rk_was_committed:
                        self._rk_was_committed = True; self._rk_best = None; self._rk_stall = 0
                        _rk_stat["commit_eps"] += 1
                    md = _rk_dist(self)
                    if self._rk_best is None or md < self._rk_best - _rk_eps:
                        self._rk_best = md; self._rk_stall = 0
                    else:
                        self._rk_stall += 1
                    if self._rk_stall >= _rk_K and md > _rk_far and self._rk_n < _rk_max:
                        self._rk_black.append(_np.array(self.plan_position))
                        self.found_goal = False
                        self._rk_best = None; self._rk_stall = 0; self._rk_n += 1
                        self._rk_was_committed = False
                        _rk_stat["revoke"] += 1
                        print("[DRPhysNav][REVOKE] stuck commit (md=%.2f stall>=%d) -> revoke #%d, blacklist + resume search" % (md, _rk_K, self._rk_n), flush=True)
                else:
                    self._rk_was_committed = False
            except Exception:
                pass
            return _orig_step_rk(self, *a, **k)
        def _rk_report():
            print("[DRPhysNav][REVOKE] SUMMARY revokes=%d vetoes=%d committed_eps=%d" % (_rk_stat["revoke"], _rk_stat["veto"], _rk_stat["commit_eps"]), flush=True)
            if _rk_stat["revoke"] == 0 and _rk_stat["veto"] == 0:
                print("[DRPhysNav][REVOKE][WARN] never fired -> arm is a NO-OP vs B0 (check thresholds K/far/cooldown)!", flush=True)
        _atexit.register(_rk_report)
        HM3D_Objnav_Agent.reset = _rk_reset
        HM3D_Objnav_Agent.step = _rk_step
        HM3D_Objnav_Agent.make_plan = _rk_make_plan
        print("[DRPhysNav] REVOKE enabled: stall_K=%d eps=%.2f far=%.2f max=%d cooldown=%.1f" % (_rk_K, _rk_eps, _rk_far, _rk_max, _rk_cool), flush=True)

    # --- Close-Range recall-triggered Commit (CRV) ---
    # Root cause traced through the .pyc (N=300): the COMMIT is VLM-gated --
    #   self.found_goal = bool(self.chainon_answer['Flag'])  (objnav_agent.make_plan @180)
    # GLEE detection only feeds the affordance (where-to-go) map, never the commit, and the GLEE threshold
    # is ALREADY reliability-adaptive: mapper.update -> perceive(rgb, drpn.thf_glee_threshold(uap_reliability)).
    # That is exactly why THF/MUAP capped at ~+0.04: "lower the threshold" was already in place and never
    # touched the VLM verdict. Failure decomposition (motiv, low_light & gaussian_noise): of the near-miss
    # failures (trajectory passes <=1.0m from the goal yet the episode fails), ~60-70% are cases where the
    # agent is right next to the target but the VLM refuses to flag 'found' on the degraded panorama -- a
    # RECALL failure at the commit layer, not a wrong commitment (so RES/M1g/REVOKE structurally cannot fix
    # them). CRV attacks this untouched lever: when the VLM declines to commit AND a low-threshold GLEE pass
    # sees the TARGET class with a LARGE mask (=close) PERSISTING over M frames under low reliability, it
    # forces Flag=True so the normal found_goal=True planner beelines to the target.
    # Precision guards (avoid REVOKE-style self-harm): target-class only, large-mask proximity, M-frame
    # temporal vote, a confidence floor, and r<gate -> on clean input the VLM commits anyway so CRV is a no-op.
    if os.environ.get("DRPN_USE_CRV", "0") == "1":
        import numpy as _np
        import atexit as _atexit
        _cv_relgate = float(os.environ.get("DRPN_CRV_REL_GATE", "0.85"))  # only fire when frame r < this (clean no-op)
        _cv_thr     = float(os.environ.get("DRPN_CRV_CONF",     "0.10"))  # low detection threshold to recover the faint target
        _cv_floor   = float(os.environ.get("DRPN_CRV_FLOOR",    "0.15"))  # min confidence of the recovered target to trust it
        _cv_area    = float(os.environ.get("DRPN_CRV_AREA",     "9000"))  # min target mask pixels (proxy for "close enough")
        _cv_M       = int(os.environ.get("DRPN_CRV_FRAMES",     "2"))     # consecutive frames required (temporal vote)
        _cv_max     = int(os.environ.get("DRPN_CRV_MAX",        "3"))     # max forced commits per episode (loop guard)
        _cv_center  = float(os.environ.get("DRPN_CRV_CENTER",  "0.0"))    # spatial guard: target mask centroid must lie
                                                                          # within central [c,1-c] of width (in-front/reachable).
                                                                          # 0.0 disables; 0.18 keeps central ~64%.
        _cv_diag    = os.environ.get("DRPN_CRV_DIAG", "0") == "1"          # log GLEE recall on EVERY Flag=False frame
        # diag aggregates: vlm_no = #(VLM Flag=False, low-r) frames; seen = target detected at all; bins by mask area
        _cv_stat    = {"fire": 0, "vlm_no": 0, "seen": 0,
                       "a0": 0, "a2500": 0, "a5000": 0, "a9000": 0, "amax": 0.0}
        _orig_qc_cv    = HM3D_Objnav_Agent.query_chainon
        _orig_reset_cv = HM3D_Objnav_Agent.reset
        def _cv_reset(self, *a, **k):
            r = _orig_reset_cv(self, *a, **k)
            self._crv_hits = 0; self._crv_n = 0
            return r
        def _cv_target(self):
            try:
                g = self._goal_category()
            except Exception:
                g = self.env.current_episode.object_category
            return str(g or "").lower()
        def _cv_query_chainon(self, *a, **k):
            ans = _orig_qc_cv(self, *a, **k)
            try:
                if (not ans) or ans.get("Flag"):
                    self._crv_hits = 0
                    return ans
                m = getattr(self, "mapper", None)
                if m is None:
                    return ans
                r = float(getattr(m, "uap_reliability", 1.0))
                if r >= _cv_relgate or getattr(self, "_crv_n", 0) >= _cv_max:
                    self._crv_hits = 0
                    return ans
                img = getattr(m, "current_rgb", None)
                if img is None:
                    return ans
                goal = _cv_target(self)
                cls, masks, confs, _ = m.object_percevior.perceive(img, confidence_threshold=_cv_thr)
                # raw_*  : best target detection anywhere in frame (honest recall denominator)
                # best_* : best target detection that ALSO passes the central-band spatial guard (fire candidate)
                raw_a = 0.0; raw_c = 0.0; best_a = 0.0; best_c = 0.0
                for c, mk, s in zip(cls, masks, confs):
                    cl = str(c).lower()
                    if not (goal and (goal in cl or cl in goal)):
                        continue
                    mka = _np.asarray(mk)
                    a = float(mka.sum())
                    if a <= 0:
                        continue
                    if a > raw_a:
                        raw_a = a; raw_c = float(s)
                    centered = True
                    if _cv_center > 0 and mka.ndim == 2:
                        cols = _np.where(mka.any(axis=0))[0]
                        if cols.size:
                            cx = float(cols.mean()) / mka.shape[1]
                            centered = (_cv_center <= cx <= 1.0 - _cv_center)
                    if centered and a > best_a:
                        best_a = a; best_c = float(s)
                # --- recall diagnostic: this is a VLM-declined, low-reliability frame ---
                _cv_stat["vlm_no"] += 1
                if raw_a > 0:
                    _cv_stat["seen"] += 1
                    _cv_stat["a0"] += 1
                    if raw_a >= 2500: _cv_stat["a2500"] += 1
                    if raw_a >= 5000: _cv_stat["a5000"] += 1
                    if raw_a >= 9000: _cv_stat["a9000"] += 1
                    if raw_a > _cv_stat["amax"]: _cv_stat["amax"] = raw_a
                if _cv_diag:
                    print("[DRPhysNav][CRV][DIAG] vlm_no r=%.2f goal='%s' raw_area=%.0f centered_area=%.0f conf=%.2f"
                          % (r, goal, raw_a, best_a, best_c), flush=True)
                if best_a >= _cv_area and best_c >= _cv_floor:
                    self._crv_hits = getattr(self, "_crv_hits", 0) + 1
                else:
                    self._crv_hits = 0
                if self._crv_hits >= _cv_M:
                    ans["Flag"] = True
                    self._crv_hits = 0
                    self._crv_n = getattr(self, "_crv_n", 0) + 1
                    _cv_stat["fire"] += 1
                    print("[DRPhysNav][CRV] force-commit: target '%s' area=%.0f conf=%.2f r=%.2f (#%d this ep)"
                          % (goal, best_a, best_c, r, self._crv_n), flush=True)
            except Exception:
                pass
            return ans
        def _cv_report():
            s = _cv_stat
            print("[DRPhysNav][CRV] SUMMARY forced_commits=%d" % s["fire"], flush=True)
            print(("[DRPhysNav][CRV][RECALL] vlm_declined_lowR_frames=%d  target_seen(any)=%d (%.0f%%)  "
                   "area>=2500=%d  >=5000=%d  >=9000=%d  max_area=%.0f")
                  % (s["vlm_no"], s["seen"], (100.0*s["seen"]/max(1,s["vlm_no"])),
                     s["a2500"], s["a5000"], s["a9000"], s["amax"]), flush=True)
            if s["fire"] == 0:
                print("[DRPhysNav][CRV][WARN] never fired -> NO-OP vs B0 (lower AREA/CONF/FRAMES or raise REL_GATE)!", flush=True)
        _atexit.register(_cv_report)
        HM3D_Objnav_Agent.reset = _cv_reset
        HM3D_Objnav_Agent.query_chainon = _cv_query_chainon
        print("[DRPhysNav] CRV enabled: rel_gate=%.2f conf=%.2f floor=%.2f area=%.0f frames=%d max=%d center=%.2f"
              % (_cv_relgate, _cv_thr, _cv_floor, _cv_area, _cv_M, _cv_max, _cv_center), flush=True)

    # --- Road 2: Depth Cross-Modal Commitment Verifier (DXCV) ---
    # Motivation (paper Sec. 5 root cause): under RGB degradation, the VLM commits at frame reliability
    # r=0.41 (vs 0.95 clean) and 76% of commits are wrong. Five families of inference-time interventions
    # (RES, M1g, REVOKE, CRV, MUAP, ROUTER, FUSE) all fail (Tab 3, all p>0.05). The paper's hypothesis is
    # that degraded RGB carries irreducible uncertainty, so perception-side patches cannot recover the
    # missing discriminative signal -- the only ways out are degradation-aware training OR multimodal
    # sensing with degradation-invariant channels.
    #
    # DXCV is the minimal experimental probe of the *multimodal sensing* claim: the Habitat depth sensor
    # is physically invariant to low-light / blur / fog / noise (it is a synthetic geometric channel in
    # sim). When the VLM proposes Flag=True at a low-reliability frame, DXCV requires CROSS-MODAL agreement
    # between the VLM verdict and depth-geometry sanity on the GLEE target mask:
    #   (a) depth_median(target_mask) in [d_min, d_max]   (close enough, not behind a wall)
    #   (b) depth_iqr(target_mask) <= d_iqr_max           (mask covers a coherent single object)
    #   (c) GLEE detects the goal class at all with conf >= c_floor
    # If any check fails, DXCV vetoes the commit (Flag=True -> False), letting the agent keep exploring.
    # On clean frames (r >= rel_gate) DXCV is a NO-OP, so the clean-side comparison is baseline-identical.
    #
    # Honest expectation: if irreducible-uncertainty is the right root cause, DXCV will NOT recover SR --
    # depth can VETO bad commits but cannot inject the missing recognition signal. A null result here is
    # *predicted* by the paper and strengthens the conclusion that retraining is required, not optional.
    # A positive result would partially validate the multimodal-sensing direction.
    if os.environ.get("DRPN_USE_DXCV", "0") == "1":
        import numpy as _np
        import atexit as _atexit
        _dx_relgate = float(os.environ.get("DRPN_DXCV_REL_GATE", "0.85"))  # only gate when r<this (clean no-op)
        _dx_dmin    = float(os.environ.get("DRPN_DXCV_DMIN",    "0.30"))   # min target depth (m); below = sensor artifact
        _dx_dmax    = float(os.environ.get("DRPN_DXCV_DMAX",    "3.50"))   # max target depth (m); above = too far to trust
        _dx_iqr     = float(os.environ.get("DRPN_DXCV_IQR_MAX", "0.60"))   # max depth IQR within mask (m); above = sees through
        _dx_thr     = float(os.environ.get("DRPN_DXCV_CONF",    "0.10"))   # low GLEE threshold to recover faint target
        _dx_floor   = float(os.environ.get("DRPN_DXCV_FLOOR",   "0.12"))   # min target conf to call detection trustworthy
        _dx_area    = float(os.environ.get("DRPN_DXCV_AREA",    "2500"))   # min mask area (px) to evaluate geometry
        _dx_log     = os.environ.get("DRPN_DXCV_LOG", "0") == "1"
        _dx_stat    = {"checked": 0, "vetoed": 0, "no_det": 0, "geom_fail": 0,
                       "dmed_sum": 0.0, "dmed_n": 0,
                       "iqr_sum": 0.0,  "iqr_n": 0}
        _orig_qc_dx    = HM3D_Objnav_Agent.query_chainon
        _orig_reset_dx = HM3D_Objnav_Agent.reset
        def _dx_reset(self, *a, **k):
            r = _orig_reset_dx(self, *a, **k)
            self._dx_n = 0
            return r
        def _dx_goal(self):
            try:
                g = self._goal_category()
            except Exception:
                g = self.env.current_episode.object_category
            return str(g or "").lower()
        def _dx_query_chainon(self, *a, **k):
            ans = _orig_qc_dx(self, *a, **k)
            try:
                if (not ans) or (not ans.get("Flag", False)):
                    return ans  # VLM says don't commit -> nothing to veto
                m = getattr(self, "mapper", None)
                if m is None:
                    return ans
                r = float(getattr(m, "uap_reliability", 1.0))
                if r >= _dx_relgate:
                    return ans  # clean frame -> baseline-identical
                img = getattr(m, "current_rgb", None)
                dep = getattr(m, "current_depth", None)
                if img is None or dep is None:
                    return ans
                dep = _np.asarray(dep)
                if dep.ndim == 3 and dep.shape[-1] == 1:
                    dep = dep[..., 0]
                goal = _dx_goal(self)
                cls, masks, confs, _ = m.object_percevior.perceive(img, confidence_threshold=_dx_thr)
                _dx_stat["checked"] += 1
                # find best target-class detection that is large enough to evaluate geometry on
                best_a = 0.0; best_mk = None; best_c = 0.0
                for c, mk, s in zip(cls, masks, confs):
                    cl = str(c).lower()
                    if not (goal and (goal in cl or cl in goal)):
                        continue
                    mka = _np.asarray(mk).astype(bool)
                    a = float(mka.sum())
                    if a < _dx_area:
                        continue
                    if float(s) < _dx_floor:
                        continue
                    if a > best_a:
                        best_a = a; best_mk = mka; best_c = float(s)
                if best_mk is None:
                    # VLM committed but GLEE finds no trustworthy target -> veto (consistent with paper's
                    # 'commits at low reliability' finding: the VLM is alone in seeing the target)
                    _dx_stat["no_det"] += 1
                    _dx_stat["vetoed"] += 1
                    ans["Flag"] = False
                    if _dx_log:
                        print("[DRPhysNav][DXCV] veto: VLM Flag=True but no trustworthy GLEE target (r=%.2f goal='%s')"
                              % (r, goal), flush=True)
                    return ans
                # depth-geometry sanity on the target mask
                if best_mk.shape != dep.shape[:2]:
                    # mismatched resolution -> can't apply geometry, skip the veto (safe default)
                    return ans
                vals = dep[best_mk]
                vals = vals[_np.isfinite(vals) & (vals > 0)]
                if vals.size < 50:
                    return ans
                d_med = float(_np.median(vals))
                q75, q25 = float(_np.percentile(vals, 75)), float(_np.percentile(vals, 25))
                d_iqr = q75 - q25
                _dx_stat["dmed_sum"] += d_med; _dx_stat["dmed_n"] += 1
                _dx_stat["iqr_sum"]  += d_iqr; _dx_stat["iqr_n"]  += 1
                geom_ok = (_dx_dmin <= d_med <= _dx_dmax) and (d_iqr <= _dx_iqr)
                if not geom_ok:
                    _dx_stat["geom_fail"] += 1
                    _dx_stat["vetoed"] += 1
                    ans["Flag"] = False
                    self._dx_n = getattr(self, "_dx_n", 0) + 1
                    if _dx_log:
                        print("[DRPhysNav][DXCV] veto: depth-geom fail d_med=%.2f d_iqr=%.2f conf=%.2f r=%.2f goal='%s'"
                              % (d_med, d_iqr, best_c, r, goal), flush=True)
                elif _dx_log:
                    print("[DRPhysNav][DXCV] pass: d_med=%.2f d_iqr=%.2f conf=%.2f r=%.2f goal='%s'"
                          % (d_med, d_iqr, best_c, r, goal), flush=True)
            except Exception as _e:
                if _dx_log:
                    print("[DRPhysNav][DXCV][WARN] exception in gate: %s" % _e, flush=True)
            return ans
        def _dx_report():
            s = _dx_stat
            tot = max(1, s["checked"])
            print(("[DRPhysNav][DXCV] SUMMARY checks=%d vetoed=%d (%.0f%%) "
                   "no_det_veto=%d geom_veto=%d  mean_d_med=%.2fm  mean_d_iqr=%.2fm")
                  % (s["checked"], s["vetoed"], 100.0*s["vetoed"]/tot,
                     s["no_det"], s["geom_fail"],
                     s["dmed_sum"]/max(1,s["dmed_n"]), s["iqr_sum"]/max(1,s["iqr_n"])), flush=True)
            if s["vetoed"] == 0:
                print("[DRPhysNav][DXCV][WARN] never vetoed -> NO-OP vs B0 (loosen thresholds or raise REL_GATE)!", flush=True)
        _atexit.register(_dx_report)
        HM3D_Objnav_Agent.reset = _dx_reset
        HM3D_Objnav_Agent.query_chainon = _dx_query_chainon
        print(("[DRPhysNav] DXCV enabled (Road 2 multimodal probe): rel_gate=%.2f "
               "depth in [%.2f,%.2f]m iqr<=%.2fm conf>=%.2f area>=%.0f")
              % (_dx_relgate, _dx_dmin, _dx_dmax, _dx_iqr, _dx_floor, _dx_area), flush=True)

    # --- Task-Aware Restoration Gating (TARG) ---
    # Diagnosis of why blanket restoration (RES) fails: it is tuned for human-perceived quality, so it
    # (1) pushes frames OOD for the clean-trained detector, (2) injects new artifacts, (3) is open-loop
    # (never verifies it actually helped this frame), (4) is content-agnostic -- net gain ~= 0.
    # TARG makes restoration a per-frame *candidate* that must prove DOWNSTREAM task utility. For each
    # frame under degradation we run GLEE on BOTH the degraded frame and its restored version, score each
    # by a task utility U = a*target_conf + b*detection_mass + g*memory_consistency, and feed the winner
    # to mapping/perception. Restored is adopted only if U(res) > U(deg) + margin, so TARG defaults to the
    # degraded frame (baseline-safe) unless restoration *demonstrably* helps recognition. On high-reliability
    # frames it skips arbitration entirely -> baseline-identical and no extra GLEE cost.
    #   solves (1)/(4): utility is measured by the detector on the target, not human quality;
    #   solves (2):     per-frame degraded-vs-restored selection rejects artifact-harmed frames;
    #   solves (3):     closed-loop -- every adoption is verified by a second GLEE pass.
    if os.environ.get("DRPN_USE_TARG", "0") == "1":
        import numpy as _np
        from mapping_utils.preprocess import preprocess_image as _pp
        from cv_utils.image_percevior import GLEE_Percevior as _GLEE
        _t_alpha  = float(os.environ.get("DRPN_TARG_ALPHA",  "1.0"))   # weight: target-class confidence
        _t_beta   = float(os.environ.get("DRPN_TARG_BETA",   "0.3"))   # weight: recognizable-detection mass
        _t_gamma  = float(os.environ.get("DRPN_TARG_GAMMA",  "0.2"))   # weight: consistency with object memory
        _t_tau    = float(os.environ.get("DRPN_TARG_TAU",    "0.30"))  # detection-mass confidence floor
        _t_margin = float(os.environ.get("DRPN_TARG_MARGIN", "0.05"))  # tolerance band around U_deg vs U_res
        _t_default= os.environ.get("DRPN_TARG_DEFAULT", "deg").lower() # "deg": adopt res only if U_res>U_deg+margin (conservative);
        #                                                               "res": adopt res by default, reject only if U_res<U_deg-margin
        #                                                               (correct when restoration is mostly beneficial, e.g. low_light)
        _t_relgate= float(os.environ.get("DRPN_TARG_REL_GATE","0.9"))  # arbitrate only when frame r < this
        _t_capN   = int(os.environ.get("DRPN_TARG_CAP_N",    "5"))     # cap detections counted in mass
        _t_log    = os.environ.get("DRPN_TARG_LOG", "0") == "1"
        # ORACLE upper bound: instead of the no-reference utility U, use the CLEAN frame (available in
        # sim) as judge. Per frame, pick whichever of {degraded, restored} reproduces the clean frame's
        # GLEE detections best. This is the ceiling of ANY per-frame deg/res gate. If it does not beat
        # blanket RES, per-frame gating is a dead end regardless of the signal (settles root cause A vs B).
        _t_oracle = os.environ.get("DRPN_TARG_ORACLE", "0") == "1"
        if _t_oracle:
            import cv2 as _cv2
            _orig_degrade = drpn.degrade_rgb            # capture the pre-degradation (clean) frame
            def _oracle_degrade(rgb_uint8=None, seed=None):
                try: drpn._oracle_clean = rgb_uint8.copy()
                except Exception: drpn._oracle_clean = None
                return _orig_degrade(rgb_uint8, seed=seed)
            drpn.degrade_rgb = _oracle_degrade
            def _oracle_match(ref_cls, ref_conf, cand_cls, cand_conf):
                # confidence-weighted class recall of the clean(reference) detections by the candidate
                if len(ref_cls) == 0: return -1.0       # clean sees nothing -> no signal (sentinel)
                cand = {}
                for c, s in zip(cand_cls, cand_conf):
                    k = str(c).lower(); cand[k] = max(cand.get(k, 0.0), float(s))
                return sum(min(float(s), cand.get(str(c).lower(), 0.0)) for c, s in zip(ref_cls, ref_conf))
        # (1) content-hash memoized perceive: lets the winner's GLEE result be reused by the subsequent
        #     original update() -> exactly 2 GLEE forwards per arbitrated frame instead of 3.
        _orig_perceive = _GLEE.perceive
        def _targ_perceive(self, image, confidence_threshold=0.25, area_threshold=2500):
            cache = getattr(self, "_targ_cache", None)
            if cache is None:
                return _orig_perceive(self, image, confidence_threshold, area_threshold)
            key = (image.shape, float(confidence_threshold), hash(image.tobytes()))
            if key in cache:
                return cache[key]
            out = _orig_perceive(self, image, confidence_threshold, area_threshold)
            cache[key] = out
            return out
        _GLEE.perceive = _targ_perceive
        # (2) stash the readable goal class on the mapper each episode (utility needs the target name),
        #     and dump per-episode arbitration stats.
        _orig_reset = HM3D_Objnav_Agent.reset
        def _targ_reset(self):
            m = getattr(self, "mapper", None)
            if m is not None and getattr(m, "_targ_stat_arb", 0) > 0:
                print("[TARG] ep stats: frames=%d arbitrated=%d adopted_res=%d (%.0f%%) meanU_deg=%.3f meanU_res=%.3f"
                      % (getattr(m,"_targ_stat_n",0), m._targ_stat_arb, m._targ_stat_res,
                         100.0*m._targ_stat_res/max(1,m._targ_stat_arb),
                         m._targ_stat_sumUd/max(1,m._targ_stat_arb), m._targ_stat_sumUr/max(1,m._targ_stat_arb)),
                      flush=True)
            _orig_reset(self)
            try:
                self.mapper._targ_goal_class = self._goal_category()
            except Exception:
                self.mapper._targ_goal_class = self.env.current_episode.object_category
            self.mapper._targ_stat_n = 0; self.mapper._targ_stat_arb = 0; self.mapper._targ_stat_res = 0
            self.mapper._targ_stat_sumUd = 0.0; self.mapper._targ_stat_sumUr = 0.0
        HM3D_Objnav_Agent.reset = _targ_reset
        def _targ_utility(mapper, classes, confidences):
            goal = str(getattr(mapper, "_targ_goal_class", "") or "").lower()
            max_t = 0.0
            for c, s in zip(classes, confidences):
                cl = str(c).lower()
                if goal and (goal in cl or cl in goal):
                    max_t = max(max_t, float(s))
            confs = sorted([float(s) for s in confidences if float(s) >= _t_tau], reverse=True)[:_t_capN]
            mass = (sum(confs) / float(_t_capN)) if _t_capN > 0 else 0.0
            mem = set(str(e['class']).lower() for e in getattr(mapper, "object_entities", []))
            det = [str(c).lower() for c in classes]
            cons = (sum(1 for d in det if d in mem) / len(det)) if (det and mem) else 0.0
            return _t_alpha*max_t + _t_beta*mass + _t_gamma*cons
        # (3) the gate itself: dual-path perception + utility arbitration, then delegate to original update.
        _orig_update = Instruct_Mapper.update
        def _targ_update(self, rgb, depth, position, rotation):
            self._targ_stat_n = getattr(self, "_targ_stat_n", 0) + 1
            if _t_oracle:  # ----- ORACLE-GATE upper bound: judge by the clean frame, not U -----
                clean = getattr(drpn, "_oracle_clean", None)
                if clean is None:
                    return _orig_update(self, rgb, depth, position, rotation)
                # reconstruct exact mapper-space frames (mapper sees cvtColor(obs, BGR2RGB))
                clean_m = _cv2.cvtColor(clean, _cv2.COLOR_BGR2RGB)
                deg_obs = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)   # invert -> obs-space degraded
                _sv = drpn.USE_RESTORE
                try:
                    drpn.USE_RESTORE = True; res_obs = drpn.restore_rgb(deg_obs)
                except Exception:
                    res_obs = deg_obs
                finally:
                    drpn.USE_RESTORE = _sv
                res_m = _cv2.cvtColor(res_obs, _cv2.COLOR_BGR2RGB)
                self._targ_cache = {}
                try:
                    cc, _, cf, _ = self.object_percevior.perceive(clean_m)
                    cd, _, fd, _ = self.object_percevior.perceive(rgb)
                    cr, _, fr, _ = self.object_percevior.perceive(res_m)
                    s_deg = _oracle_match(cc, cf, cd, fd); s_res = _oracle_match(cc, cf, cr, fr)
                except Exception:
                    self._targ_cache = None
                    return _orig_update(self, rgb, depth, position, rotation)
                self._targ_stat_arb = getattr(self, "_targ_stat_arb", 0) + 1
                if s_res > s_deg:
                    chosen = res_m; self._targ_stat_res = getattr(self, "_targ_stat_res", 0) + 1
                else:
                    chosen = rgb
                if _t_log:
                    print("[TARG-ORACLE] s_deg=%.3f s_res=%.3f -> %s" %
                          (s_deg, s_res, "RES" if chosen is res_m else "DEG"), flush=True)
                try:
                    return _orig_update(self, chosen, depth, position, rotation)
                finally:
                    self._targ_cache = None
            try:
                r = float(drpn.frame_reliability(rgb))
            except Exception:
                r = 1.0
            if r >= _t_relgate:  # frame already reliable -> no arbitration, baseline-identical
                return _orig_update(self, rgb, depth, position, rotation)
            _saved = drpn.USE_RESTORE  # force restoration regardless of the global RES flag
            try:
                drpn.USE_RESTORE = True
                x_res = drpn.restore_rgb(rgb.copy())
            except Exception:
                x_res = rgb
            finally:
                drpn.USE_RESTORE = _saved
            self._targ_cache = {}
            try:
                cd, _, fd, _ = self.object_percevior.perceive(_pp(rgb))
                cr, _, fr, _ = self.object_percevior.perceive(_pp(x_res))
                u_deg = _targ_utility(self, cd, fd); u_res = _targ_utility(self, cr, fr)
            except Exception:
                self._targ_cache = None
                return _orig_update(self, rgb, depth, position, rotation)
            self._targ_stat_arb = getattr(self, "_targ_stat_arb", 0) + 1
            self._targ_stat_sumUd = getattr(self, "_targ_stat_sumUd", 0.0) + u_deg
            self._targ_stat_sumUr = getattr(self, "_targ_stat_sumUr", 0.0) + u_res
            if _t_default == "res":   # default-restore: keep restored unless it demonstrably HURTS utility
                if u_res < u_deg - _t_margin:
                    chosen = rgb
                else:
                    chosen = x_res; self._targ_stat_res = getattr(self, "_targ_stat_res", 0) + 1
            else:                     # default-degraded (conservative): adopt restored only if it clearly helps
                if u_res > u_deg + _t_margin:
                    chosen = x_res; self._targ_stat_res = getattr(self, "_targ_stat_res", 0) + 1
                else:
                    chosen = rgb
            if _t_log:
                print("[TARG] r=%.2f U_deg=%.3f U_res=%.3f -> %s" %
                      (r, u_deg, u_res, "RES" if chosen is x_res else "DEG"), flush=True)
            try:
                return _orig_update(self, chosen, depth, position, rotation)  # perceive(chosen) -> cache hit
            finally:
                self._targ_cache = None
        Instruct_Mapper.update = _targ_update
        print("[DRPhysNav] TARG enabled: a=%.2f b=%.2f g=%.2f tau=%.2f margin=%.2f default=%s rel_gate=%.2f capN=%d"
              % (_t_alpha,_t_beta,_t_gamma,_t_tau,_t_margin,_t_default,_t_relgate,_t_capN), flush=True)

    # ===================================================================================
    # DARE -- Degradation-Aware Restoration decision layer (diagnosis-driven 3 components).
    #   A) Router            : blind (type,severity) estimate -> sign-map -> bypass | restore
    #   B) Specialized restore: route to the restoration tuned for that degradation type
    #   C) Temporal          : chunk window + hysteresis so the route does not jitter per-frame
    # Wired by REPLACING drpn.restore_rgb, which agent.update_trajectory calls UNCONDITIONALLY
    # every frame -> the layer is guaranteed to run (no silent no-op). Variants:
    #   DRPN_ROUTER_ORACLE=1 : route by the TRUE (type,sev) -> per-condition routing upper bound.
    #   DRPN_MIX=1           : randomize (type,sev) per episode from DRPN_MIX_SPEC (unknown to agent).
    # ===================================================================================
    if os.environ.get("DRPN_USE_ROUTER", "0") == "1":
        import numpy as _np
        import cv2 as _cv2
        import atexit as _atexit
        _ro_oracle = os.environ.get("DRPN_ROUTER_ORACLE", "0") == "1"
        _ro_chunk  = int(os.environ.get("DRPN_ROUTER_CHUNK", "5"))      # hysteresis window (C)
        _ro_log    = os.environ.get("DRPN_ROUTER_LOG", "0") == "1"
        # low_light severity proxy thresholds (blind, via mean luma): darker => more severe.
        _ro_ll_med  = float(os.environ.get("DRPN_ROUTER_LL_MED", "95"))       # median luma<this => low_light (measured: ll<=81, others>=110)
        _ro_sigma   = float(os.environ.get("DRPN_ROUTER_SIGMA", "18"))        # noise gate (measured: noise~48, all others<2)
        _ro_lap     = float(os.environ.get("DRPN_ROUTER_LAP", "120"))          # blur gate
        _ro_dark    = float(os.environ.get("DRPN_ROUTER_DARK", "120"))         # fog gate
        _ro_restore_fog = os.environ.get("DRPN_ROUTER_RESTORE_FOG", "1") == "1"  # tentative until sweep confirms
        drpn._router_stats = {}
        drpn._router_hist = []
        if not hasattr(drpn, "_cur_true_type"):
            drpn._cur_true_type = drpn.DEGRADE_TYPE
            drpn._cur_true_sev  = drpn.DEGRADE_SEVERITY

        _ro_dump = os.environ.get("DRPN_ROUTER_DUMP", "0") == "1"
        drpn._ro_dumpn = 0
        def _ro_blind(bgr):
            """Blind degradation (type, severity) estimate from no-reference features."""
            g = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
            luma = float(g.mean()); std = float(g.std())
            med  = float(_np.median(g)); p10 = float(_np.percentile(g, 10))
            darkfrac = float((g < 50).mean())
            sigma = float(drpn._noise_sigma(bgr))
            lap = float(_cv2.Laplacian(g, _cv2.CV_64F).var())
            dark = float(drpn._dark_channel(bgr).mean())
            if _ro_dump and drpn._ro_dumpn < 80:
                drpn._ro_dumpn += 1
                print("[RODUMP] true=%s/s%s | luma=%.1f med=%.1f p10=%.1f darkfrac=%.2f std=%.1f sigma=%.1f lap=%.0f dark=%.1f"
                      % (getattr(drpn,"_cur_true_type","?"), getattr(drpn,"_cur_true_sev","?"),
                         luma, med, p10, darkfrac, std, sigma, lap, dark), flush=True)
            # MEASURED on HM3D (median over a real episode, see runs/rodump):
            #   low_light/s4 : med~26  sigma~0.4  lap~115           (gamma crushes bulk -> low median, LOW sigma)
            #   gauss_noise/s4: med~153 sigma~48   lap~46000         (genuine heavy noise -> very high sigma)
            #   motion_blur/s4: med~107 sigma~0.3  lap~51            (blur -> low lap, low sigma)
            # => cleanly separable by sigma (noise) then median-luma (low_light). Do NOT use dark-channel
            #    for low_light: gaussian noise drives dark-channel->0 and would be misrouted to low_light.
            if sigma > _ro_sigma:                       typ = "gaussian_noise" # only genuine heavy noise
            elif med < _ro_ll_med:                      typ = "low_light"      # bulk dark
            elif std < 40 and dark > _ro_dark:          typ = "fog"
            elif lap < _ro_lap:                         typ = "motion_blur"
            else:                                       typ = "clean"
            if typ == "low_light":
                sev = 4 if med < 40 else 2          # logging only; severity is scene-confounded, not used for gating
            elif typ == "clean":
                sev = 0
            else:
                sev = 4
            return typ, sev

        def _ro_should_restore(typ, sev):
            """sign-map policy (root cause #1): restore only where it helps.
            Blind severity is scene-confounded (ll s1 med=67 > s2 med=81), so we cannot reliably
            bypass mild low_light; we restore ALL detected low_light. Net positive: s2/s4 gain ~+0.07,
            s1 cost only ~-0.04 and is a minority of the mix. Noise/blur (unrecoverable/harmful) bypass."""
            if typ == "low_light":     return True
            if typ == "fog":           return _ro_restore_fog   # std<40 & dark>120 (dehaze)
            return False                                        # noise / blur / clean -> bypass

        def _ro_specialized(bgr, typ, sev):
            """component B: route to the restoration specialized for this degradation type."""
            if typ == "low_light":
                out = bgr
                _s = float(drpn._noise_sigma(out))
                if _s >= _ro_sigma:                             # dark+noisy: denoise FIRST so CLAHE doesn't amplify noise
                    out = drpn._restore_denoise(out, _s)
                out = drpn._restore_lowlight(out)
                if drpn._noise_sigma(out) < _ro_sigma:          # only sharpen if not noisy
                    out = drpn._restore_sharpen(out)
                return out
            if typ == "fog":
                return drpn._restore_dehaze(bgr)
            return bgr

        def _router_restore(bgr=None):
            # decide degradation (oracle uses ground truth from mixed/fixed config)
            if _ro_oracle:
                typ = getattr(drpn, "_cur_true_type", drpn.DEGRADE_TYPE) or "clean"
                sev = int(getattr(drpn, "_cur_true_sev", drpn.DEGRADE_SEVERITY) or 0)
                if typ in ("", "none"): typ = "clean"
            else:
                typ, sev = _ro_blind(bgr)
            want = _ro_should_restore(typ, sev)
            # component C: hysteresis over a chunk window (majority vote)
            drpn._router_hist.append(1 if want else 0)
            if len(drpn._router_hist) > _ro_chunk:
                drpn._router_hist.pop(0)
            decision = sum(drpn._router_hist) * 2 > len(drpn._router_hist)
            if decision:
                out = _ro_specialized(bgr, typ, sev); key = "restore_%s" % typ
            else:
                out = bgr; key = "bypass_%s" % typ
            drpn._router_stats[key] = drpn._router_stats.get(key, 0) + 1
            if _ro_log:
                print("[ROUTER] %s typ=%s sev=%s want=%s -> %s"
                      % ("ORACLE" if _ro_oracle else "BLIND", typ, sev, want,
                         "RESTORE" if decision else "BYPASS"), flush=True)
            return out

        drpn.restore_rgb = _router_restore     # unconditional call site -> guaranteed to run
        drpn.USE_RESTORE = True

        def _ro_report():
            st = getattr(drpn, "_router_stats", {})
            tot = sum(st.values())
            print("[DRPhysNav][ROUTER] branch hit-counts (total frames=%d):" % tot, flush=True)
            for k in sorted(st): print("    %-22s %d" % (k, st[k]), flush=True)
            if tot == 0:
                print("[DRPhysNav][ROUTER][WARN] router NEVER invoked -- wiring/no-op problem!", flush=True)
            elif not any(k.startswith("restore_") for k in st):
                print("[DRPhysNav][ROUTER][WARN] restore branch NEVER fired -- check sign-map/thresholds!", flush=True)
            elif not any(k.startswith("bypass_") for k in st):
                print("[DRPhysNav][ROUTER][WARN] bypass branch NEVER fired -- check sign-map/thresholds!", flush=True)
        _atexit.register(_ro_report)
        print("[DRPhysNav] ROUTER (DARE) enabled: oracle=%s chunk=%d ll_med<%.0f noise_sigma>%.0f restore_fog=%s"
              % (_ro_oracle, _ro_chunk, _ro_ll_med, _ro_sigma, _ro_restore_fog), flush=True)

    # ===================================================================================
    # FUSE -- uncertainty-weighted dual-path detection FUSION (DARE component C, grounded).
    #   Borrowed methods:
    #     * Perception Matters (2024): inverse-uncertainty weighted map aggregation
    #       -> each observation contributes weighted by its (in)confidence, not binary.
    #     * Robust Bayesian Semantic Mapping (2023): confidence-weighted fusion +
    #       overconfidence regularization (beta) to reject confident-but-wrong observations.
    #     * Multi-view Bayesian sensor fusion: noisy-OR combination of independent positives.
    #   WHY (settles the dead-end): the oracle per-frame gate proved ORACLEGATE <= RES, i.e.
    #   *choosing* one of {degraded, restored} is capped by the better single frame. FUSION uses
    #   BOTH: the degraded path is kept verbatim (fused detections are a SUPERSET of B0 -> never
    #   regress), while the restored (pixel-aligned, same-viewpoint) path adds (a) noisy-OR
    #   corroboration on agreeing detections and (b) reliability-weighted, overconfidence-
    #   regularized recovery of objects only it can see. The fused evidence is the UNION, so the
    #   ceiling is >= the better single frame -> structurally escapes the per-frame ceiling.
    #   Wired by replacing GLEE_Percevior.perceive (called every frame by mapper.update) -> the
    #   layer is guaranteed to run; per-branch counters + atexit report guard against silent no-op.
    #     DRPN_FUSE_ORACLE=1 : trust restored fully (beta=0, r_r=1) -> fusion upper bound.
    # ===================================================================================
    if os.environ.get("DRPN_USE_FUSE", "0") == "1":
        import numpy as _np
        import cv2 as _cv2
        import atexit as _atexit
        from cv_utils.image_percevior import GLEE_Percevior as _GLEE_F
        _f_beta     = float(os.environ.get("DRPN_FUSE_BETA", "0.10"))    # overconfidence reg on restored-only adds
        _f_iou      = float(os.environ.get("DRPN_FUSE_IOU", "0.5"))      # mask IoU to treat two dets as same object
        _f_relfloor = float(os.environ.get("DRPN_FUSE_REL_FLOOR", "0.2"))
        _f_idt      = float(os.environ.get("DRPN_FUSE_IDENTITY_MAD", "1.0"))  # MAD<this => restore is a no-op (bypass) => skip fusion
        _f_oracle   = os.environ.get("DRPN_FUSE_ORACLE", "0") == "1"     # trust restored fully -> fusion ceiling
        _f_log      = os.environ.get("DRPN_FUSE_LOG", "0") == "1"
        drpn._fuse_stats = {"frames": 0, "deg_dets": 0, "res_corroborate": 0, "res_added": 0, "res_rejected": 0, "bypass_identity": 0}
        _f_guard = {"in": False}
        _orig_perceive_f = _GLEE_F.perceive

        def _f_rel(img_rgb):
            try:
                r = float(drpn.frame_reliability(_cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)))
            except Exception:
                r = 1.0
            return min(1.0, max(_f_relfloor, r))

        def _f_iou_fn(a, b):
            a = a > 0; b = b > 0
            inter = float(_np.logical_and(a, b).sum())
            uni = float(_np.logical_or(a, b).sum())
            return inter / uni if uni > 0 else 0.0

        def _fuse_perceive(self, image, confidence_threshold=0.25, area_threshold=2500):
            if _f_guard["in"]:                      # reentrancy guard (restored-path call)
                return _orig_perceive_f(self, image, confidence_threshold, area_threshold)
            _f_guard["in"] = True
            res_rgb = None
            try:
                cls_d, m_d, cf_d, vis = _orig_perceive_f(self, image, confidence_threshold, area_threshold)
                try:
                    _sv = drpn.USE_RESTORE; drpn.USE_RESTORE = True
                    deg_bgr = _cv2.cvtColor(image, _cv2.COLOR_RGB2BGR)
                    res_bgr = drpn.restore_rgb(deg_bgr)
                    drpn.USE_RESTORE = _sv
                    # If restoration is a no-op / bypass (restored ~= degraded), skip the restored path
                    # entirely: fused == degraded == B0. Prevents self-corroboration (double-counting the
                    # SAME frame via noisy-OR), which would otherwise break the "bypass == B0" guarantee.
                    if (res_bgr is None or res_bgr.shape != deg_bgr.shape or
                            float(_np.abs(res_bgr.astype(_np.int16) - deg_bgr.astype(_np.int16)).mean()) < _f_idt):
                        res_rgb = None; cls_r, m_r, cf_r = [], [], []
                        drpn._fuse_stats["bypass_identity"] += 1
                    else:
                        res_rgb = _cv2.cvtColor(res_bgr, _cv2.COLOR_BGR2RGB)
                        cls_r, m_r, cf_r, _ = _orig_perceive_f(self, res_rgb, confidence_threshold, area_threshold)
                except Exception:
                    cls_r, m_r, cf_r = [], [], []
            finally:
                _f_guard["in"] = False

            r_r  = 1.0 if _f_oracle else (_f_rel(res_rgb) if res_rgb is not None else 0.0)
            beta = 0.0 if _f_oracle else _f_beta

            # primary path = degraded frame, kept verbatim -> fused detections superset of B0 (no regression)
            fused_cls = [c for c in cls_d]
            fused_m   = [_np.asarray(m) for m in m_d]
            fused_cf  = [float(s) for s in cf_d]
            drpn._fuse_stats["frames"]   += 1
            drpn._fuse_stats["deg_dets"] += len(fused_cls)

            for c, m, s in zip(cls_r, m_r, cf_r):
                m = _np.asarray(m); cl = str(c).lower(); se = float(s)
                best, bj = -1.0, -1
                for j in range(len(fused_cls)):
                    if str(fused_cls[j]).lower() != cl:
                        continue
                    iou = _f_iou_fn(m, fused_m[j])
                    if iou > best:
                        best, bj = iou, j
                if best >= _f_iou and bj >= 0:
                    # multi-view corroboration: noisy-OR of two independent positive detections
                    prev = fused_cf[bj]
                    fused_cf[bj] = float(min(0.999, 1.0 - (1.0 - prev) * (1.0 - se * r_r)))
                    drpn._fuse_stats["res_corroborate"] += 1
                else:
                    # restored-only recovery: reliability-weighted + overconfidence-regularized
                    cf_eff = se * r_r - beta
                    if cf_eff >= confidence_threshold:
                        fused_cls.append(c); fused_m.append(m); fused_cf.append(float(cf_eff))
                        drpn._fuse_stats["res_added"] += 1
                    else:
                        drpn._fuse_stats["res_rejected"] += 1

            if _f_log:
                print("[FUSE] deg=%d res=%d -> fused=%d r_r=%.2f (corrob=%d add=%d rej=%d)"
                      % (len(cls_d), len(cls_r), len(fused_cls), r_r,
                         drpn._fuse_stats["res_corroborate"], drpn._fuse_stats["res_added"],
                         drpn._fuse_stats["res_rejected"]), flush=True)
            cls_out = _np.array(fused_cls, dtype=object) if fused_cls else _np.array([])
            return cls_out, fused_m, fused_cf, vis

        _GLEE_F.perceive = _fuse_perceive

        def _f_report():
            st = drpn._fuse_stats
            print("[DRPhysNav][FUSE] frames=%d deg_dets=%d bypass_identity=%d | restored path: corroborate=%d add=%d reject=%d"
                  % (st["frames"], st["deg_dets"], st["bypass_identity"], st["res_corroborate"], st["res_added"], st["res_rejected"]),
                  flush=True)
            if st["frames"] == 0:
                print("[DRPhysNav][FUSE][WARN] perceive NEVER called -- wiring/no-op problem!", flush=True)
            elif st["res_added"] + st["res_corroborate"] == 0:
                print("[DRPhysNav][FUSE][WARN] restored path NEVER contributed -- check restore_rgb/thresholds!", flush=True)
        _atexit.register(_f_report)
        print("[DRPhysNav] FUSE (DARE-C) enabled: beta=%.2f iou=%.2f rel_floor=%.2f oracle=%s"
              % (_f_beta, _f_iou, _f_relfloor, _f_oracle), flush=True)

    # --- Mixed-degradation evaluation: randomize (type,severity) per episode (unknown to agent) ---
    if os.environ.get("DRPN_MIX", "0") == "1":
        import numpy as _np
        _mix_spec = os.environ.get("DRPN_MIX_SPEC", "low_light:1,2,4;gaussian_noise:4;motion_blur:4")
        _mix_pool = []
        for _part in _mix_spec.split(";"):
            _part = _part.strip()
            if not _part: continue
            _t, _sevs = _part.split(":")
            for _s in _sevs.split(","):
                _mix_pool.append((_t.strip(), int(_s)))
        drpn._mix_ep = 0
        _orig_reset_mix = HM3D_Objnav_Agent.reset
        def _mix_reset(self):
            _rng = _np.random.RandomState(drpn._mix_ep + drpn.SEED * 100003)
            _t, _s = _mix_pool[_rng.randint(len(_mix_pool))]
            drpn.DEGRADE_TYPE = _t; drpn.DEGRADE_SEVERITY = _s; drpn._DEGRADE_ON = True
            drpn._cur_true_type = _t; drpn._cur_true_sev = _s
            drpn._router_hist = []     # reset hysteresis per episode
            drpn._mix_ep += 1
            return _orig_reset_mix(self)
        HM3D_Objnav_Agent.reset = _mix_reset
        print("[DRPhysNav] MIX enabled: pool=%s" % (_mix_pool,), flush=True)

    habitat_config = hm3d_config(stage=args.split,episodes=args.eval_episodes)
    habitat_env = habitat.Env(habitat_config)
    habitat_mapper = Instruct_Mapper(habitat_camera_intrinsic(habitat_config),
                                    pcd_resolution=args.mapper_resolution,
                                    grid_resolution=args.path_resolution,
                                    grid_size=args.path_scale)
    habitat_agent = HM3D_Objnav_Agent(habitat_env,habitat_mapper)
    import wandb_logger
    wandb_logger.init(args.split, args.eval_episodes, args.out_csv)
    evaluation_metrics = []
    # --- Top-down trajectory logging (DRPN_TRAJ_LOG=1): per-step agent world position,
    # goal positions, and per-episode top-down occupancy map, for the multi-arm
    # trajectory-comparison figure. Real runs only; no effect on agent behaviour. ---
    _traj_on = os.environ.get("DRPN_TRAJ_LOG", "0") == "1"
    if _traj_on:
        import json as _tj_json
        import numpy as _tj_np
        from habitat.utils.visualizations import maps as _tj_maps
        _traj_jsonl = args.out_csv.replace(".csv", "_traj.jsonl")
        _traj_mapdir = os.path.join(os.path.dirname(args.out_csv), "traj_maps")
        os.makedirs(_traj_mapdir, exist_ok=True)
        print("[DRPhysNav] TRAJ_LOG enabled -> %s" % _traj_jsonl, flush=True)
    for i in tqdm(range(args.eval_episodes)):
        habitat_agent.reset()
        if _traj_on:
            _tj_pts = [[float(x) for x in habitat_env.sim.get_agent_state().position]]
            try:
                _tj_map = _tj_maps.get_topdown_map_from_sim(habitat_env.sim, map_resolution=1024)
            except Exception as _e:
                print("[DRPhysNav][traj][warn] topdown map failed ep%d: %s" % (i, _e), flush=True)
                _tj_map = None
        habitat_agent.make_plan()
        while not habitat_env.episode_over and habitat_agent.episode_steps < 495:
            habitat_agent.step()
            if _traj_on:
                _tj_pts.append([float(x) for x in habitat_env.sim.get_agent_state().position])
        if args.save_traj:
            try:
                habitat_agent.save_trajectory("./tmp/episode-%d/"%i)
            except Exception as e:
                print("save_trajectory skipped:", e, flush=True)
        evaluation_metrics.append({'success':habitat_agent.metrics['success'],
                                'spl':habitat_agent.metrics['spl'],
                                'distance_to_goal':habitat_agent.metrics['distance_to_goal'],
                                'object_goal':habitat_agent.instruct_goal})
        write_metrics(evaluation_metrics,path=args.out_csv)
        if _traj_on:
            _ep = habitat_env.current_episode
            _goals = [[float(x) for x in g.position] for g in _ep.goals]
            _rec = {"ep": i, "scene": os.path.basename(_ep.scene_id),
                    "episode_id": str(_ep.episode_id),
                    "goal": habitat_agent.instruct_goal,
                    "success": habitat_agent.metrics['success'],
                    "spl": habitat_agent.metrics['spl'],
                    "distance_to_goal": habitat_agent.metrics['distance_to_goal'],
                    "traj_world": _tj_pts, "goals_world": _goals}
            if _tj_map is not None:
                _dims = (_tj_map.shape[0], _tj_map.shape[1])
                try:
                    _rec["traj_grid"] = [[int(v) for v in _tj_maps.to_grid(p[2], p[0], _dims, sim=habitat_env.sim)]
                                         for p in _tj_pts]
                    _rec["goals_grid"] = [[int(v) for v in _tj_maps.to_grid(g[2], g[0], _dims, sim=habitat_env.sim)]
                                          for g in _goals]
                except Exception as _e:
                    print("[DRPhysNav][traj][warn] to_grid failed ep%d: %s" % (i, _e), flush=True)
                _mapf = os.path.join(_traj_mapdir, "ep%03d_map.npy" % i)
                if not os.path.exists(_mapf):
                    _tj_np.save(_mapf, _tj_map)
                _rec["map_file"] = _mapf
            with open(_traj_jsonl, "a") as _tf:
                _tf.write(_tj_json.dumps(_rec) + "\n")
        if os.environ.get("DRPN_MIX", "0") == "1" or os.environ.get("DRPN_USE_ROUTER", "0") == "1":
            import json as _json
            _dlog = args.out_csv.replace(".csv", "_deg.jsonl")
            with open(_dlog, "a") as _df:
                _df.write(_json.dumps({"ep": i,
                                       "success": habitat_agent.metrics['success'],
                                       "spl": habitat_agent.metrics['spl'],
                                       "deg_type": getattr(drpn, "_cur_true_type", drpn.DEGRADE_TYPE),
                                       "deg_sev": getattr(drpn, "_cur_true_sev", drpn.DEGRADE_SEVERITY)}) + "\n")
        wandb_logger.log_episode(i, habitat_agent.metrics,
                                 motiv_steps=getattr(habitat_agent, "_motiv_log", None),
                                 instruct_goal=habitat_agent.instruct_goal)
        if drpn.MOTIV_LOG:
            import json
            mlog = args.out_csv.replace(".csv", "_motiv.jsonl")
            with open(mlog, "a") as mf:
                mf.write(json.dumps({"ep": i, "success": habitat_agent.metrics['success'],
                                     "spl": habitat_agent.metrics['spl'],
                                     "goal": habitat_agent.instruct_goal,
                                     "steps": habitat_agent._motiv_log}) + "\n")
        if drpn.M1_LOG:
            import json, statistics
            deg = habitat_agent._m1_psnr_deg; res = habitat_agent._m1_psnr_res
            m1log = args.out_csv.replace(".csv", "_m1.jsonl")
            with open(m1log, "a") as mf:
                mf.write(json.dumps({"ep": i, "success": habitat_agent.metrics['success'],
                                     "psnr_deg": (sum(deg)/len(deg)) if deg else None,
                                     "psnr_res": (sum(res)/len(res)) if res else None,
                                     "n_frames": len(deg)}) + "\n")
    wandb_logger.finish()
