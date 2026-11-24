"""Phase 1.5 Quantitative Verification Suite (3D Multi-Zone ESC + Wafer Thermal Control).

Runs three self-contained quantitative verification experiments on the `medium` 3D FEniCSx FEM
and `medium` Coupled 3D Wafer FVM + ESC ROM Surrogate:

  1. Exact Forward vs Forward+Backward (12D Adjoint) 1-Step Timing:
       - 20 warmed-up runs (batch=1) of Surrogate Forward-only
       - 20 warmed-up runs (batch=1) of Surrogate Forward + 12D Adjoint Backward
       - 5 warmed-up runs of same-case `medium` 3D FEniCSx FEM nonlinear solve
       - Computes single-solve speedup (T_fem / T_fwd) and 12D central-FD gradient
         1-step speedup (24 * T_fem / T_fwd_bwd).

  2. Large-Scale Independent Closed-Loop Batch Validation (n_A >= 60):
       - Reuses LOCKED calibration thresholds from Week 2 (S_0_thresh=107.154,
         S_probe_thresh=170.972) with ZERO re-tuning on test cases.
       - 87 independent test cases (RNG seed 20261125, fresh delta_cal ~ U(-0.2, 0.2)):
           * 72 recoverable in-model cases (30% no-leak, 70% beta_leak ~ U(0.70, 0.92))
           * 5 severe unrecoverable leak cases (beta_leak ~ U(0.15, 0.38))
           * 4 active H_block=1 cases (2 OEM He flow >= 20 sccm, 2 missing readback)
           * 6 OOD hidden-fault cases (2 azimuthal plasma hotspots, 2 shifted r_leak=128 mm,
             2 wafer warpage d(r))
       - Freezes A_pre and u* strictly BEFORE post-hoc `medium` 3D FEM evaluation with
         TRUE HIDDEN FAULT and original specs (no pre-approval margins).
       - Reports exact P(V=1 | A_pre=1) and one-sided 95% Clopper-Pearson upper bound.

  3. Operable Envelope 2D Phase Map (alpha_peak vs beta_leak):
       - Fixed Q_plasma=500 W, T_target=40.0 C, theta_leak=225 deg, r_leak=0.138 m,
         nominal u_0 (P_in=12, P_out=14 Torr), leak interlock P_out <= 14.0 for beta < 0.90.
       - 7 x 9 = 63 grid points over alpha_peak in [-0.20, 0.40] x beta_leak in [0.20, 1.00].
       - Saves publication-grade 2D phase map to `results/benchmarks/phase15_operable_envelope.png`.

Outputs:
  - `results/benchmarks/phase15_verification_suite.json`
  - `results/benchmarks/phase15_operable_envelope.png`
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
import torch

from benchmarks.week2_3d_closed_loop_eval import (
    BASELINE_P_IN,
    BASELINE_P_OUT,
    BASELINE_Q_HEATERS,
    EvalCase,
    grade_posthoc_fem,
    make_cond,
)
from src.fem3d.build_esc_3d_mesh import ESC3DGeometrySpec
from src.fem3d.esc_3d_fem_solver import ESC3DFEMSolver, ESC3DOperatingCondition
from src.inverse.stage1_3d_diagnosis import (
    HeFlowModelParams,
    Stage1Result,
    build_polar_sensor_interp_matrix,
    build_sensarray_65_coords,
    eval_he_flow_proxy_torch,
    generate_synthetic_observation,
    run_stage1_diagnosis,
)
from src.inverse.stage2_3d_recovery import (
    RecoverySpec,
    _heater_areas_cm2_torch,
    eval_constraints_torch,
    run_stage2_recovery,
)
from src.solvers.coupled_3d_wafer_fvm import Coupled3DWaferFVMSolver, SolveOptions


# LOCKED calibration thresholds from Week 2 (do NOT re-tune on test cases!)
LOCKED_S_0_THRESH: float = 107.154
LOCKED_S_PROBE_THRESH: float = 170.972


def clopper_pearson_upper_95(k: int, n: int) -> float | None:
    """One-sided 95% Clopper-Pearson upper confidence bound for binomial rate k/n."""
    if n <= 0:
        return None
    if k == 0:
        return float(1.0 - 0.05 ** (1.0 / n))
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(0.95, k + 1, n - k))


# ======================================================================
# EXPERIMENT 1: Exact Forward vs Forward+Backward (12D Adjoint) Timing
# ======================================================================
def run_experiment1_timing(
    fem_solver: ESC3DFEMSolver,
    fvm_solver: Coupled3DWaferFVMSolver,
    spec: RecoverySpec,
    geom: ESC3DGeometrySpec,
    opts: SolveOptions,
) -> dict:
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Forward vs Forward+Backward (12D Adjoint) 1-Step Timing")
    print("=" * 72)

    cond = make_cond(
        total_plasma_power_W=500.0,
        plasma_peaking_factor=0.25,
        beta_leak=0.70,
        leak_theta_center_deg=225.0,
    )
    u_vec = np.array(list(BASELINE_Q_HEATERS) + [BASELINE_P_IN, BASELINE_P_OUT], dtype=np.float64)
    heater_areas_cm2 = _heater_areas_cm2_torch(geom, fvm_solver.device)
    Q_t = torch.as_tensor(u_vec[:10], dtype=torch.float64, device=fvm_solver.device)
    p_in_t = torch.as_tensor(u_vec[10], dtype=torch.float64, device=fvm_solver.device)
    p_out_t = torch.as_tensor(u_vec[11], dtype=torch.float64, device=fvm_solver.device)

    # Warm-up surrogate
    for _ in range(3):
        with torch.no_grad():
            _ = fvm_solver.solve(
                Q_heaters=Q_t, p_in=p_in_t, p_out=p_out_t,
                beta_leak=0.70, Q_plasma=500.0, alpha_peak=0.25,
                opts=opts, theta_leak_deg=225.0,
            )
        u_w = torch.tensor(u_vec, dtype=torch.float64, device=fvm_solver.device, requires_grad=True)
        _, g_norm, m = eval_constraints_torch(
            solver=fvm_solver, Q_heaters=u_w[:10], p_in=u_w[10], p_out=u_w[11],
            Q_plasma=500.0, alpha_peak=0.25, beta_leak=0.70, theta_leak_deg=225.0,
            spec=spec, heater_areas_cm2=heater_areas_cm2, opts=opts, smooth=True,
        )
        g_stack = torch.stack(list(g_norm.values()))
        loss = 1.5 * (0.08 * torch.logsumexp(g_stack / 0.08, dim=0)) + 4.0 * m["var_A_eff"]
        loss.backward()

    # (a) 20 warmed-up runs: Surrogate Forward-only
    n_surr = 20
    fwd_times_ms: list[float] = []
    for _ in range(n_surr):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fvm_solver.solve(
                Q_heaters=Q_t, p_in=p_in_t, p_out=p_out_t,
                beta_leak=0.70, Q_plasma=500.0, alpha_peak=0.25,
                opts=opts, theta_leak_deg=225.0,
            )
        fwd_times_ms.append((time.perf_counter() - t0) * 1e3)

    # (b) 20 warmed-up runs: Surrogate Forward + 12D Adjoint Backward (full Stage-2 step)
    fwd_bwd_times_ms: list[float] = []
    for _ in range(n_surr):
        t0 = time.perf_counter()
        u_w = torch.tensor(u_vec, dtype=torch.float64, device=fvm_solver.device, requires_grad=True)
        _, g_norm, m = eval_constraints_torch(
            solver=fvm_solver, Q_heaters=u_w[:10], p_in=u_w[10], p_out=u_w[11],
            Q_plasma=500.0, alpha_peak=0.25, beta_leak=0.70, theta_leak_deg=225.0,
            spec=spec, heater_areas_cm2=heater_areas_cm2, opts=opts, smooth=True,
        )
        g_stack = torch.stack(list(g_norm.values()))
        loss = (
            1.5 * (0.08 * torch.logsumexp(g_stack / 0.08, dim=0))
            + 4.0 * m["var_A_eff"]
            + 2.0 * (m["T_mean_eff_C"] - spec.T_target_C).pow(2)
        )
        loss.backward()
        _ = u_w.grad.detach().cpu().numpy()
        fwd_bwd_times_ms.append((time.perf_counter() - t0) * 1e3)

    # (c) 5 warmed-up runs: same-case medium 3D FEniCSx FEM nonlinear solve
    _ = fem_solver.solve(cond)  # warm-up
    n_fem = 5
    fem_times_ms: list[float] = []
    for _ in range(n_fem):
        t0 = time.perf_counter()
        _ = fem_solver.solve(cond)
        fem_times_ms.append((time.perf_counter() - t0) * 1e3)

    t_fwd_med = float(np.median(fwd_times_ms))
    t_fwd_p95 = float(np.percentile(fwd_times_ms, 95))
    t_fb_med = float(np.median(fwd_bwd_times_ms))
    t_fb_p95 = float(np.percentile(fwd_bwd_times_ms, 95))
    t_fem_med = float(np.median(fem_times_ms))
    t_fem_p95 = float(np.percentile(fem_times_ms, 95))

    speedup_single = t_fem_med / t_fwd_med
    speedup_12d_cfd = (24.0 * t_fem_med) / t_fb_med
    speedup_12d_ffd = (13.0 * t_fem_med) / t_fb_med

    res = {
        "n_surrogate_runs": n_surr,
        "n_fem_runs": n_fem,
        "surrogate_forward_only_ms": {
            "median": round(t_fwd_med, 3),
            "p95": round(t_fwd_p95, 3),
            "mean": round(float(np.mean(fwd_times_ms)), 3),
            "min": round(float(np.min(fwd_times_ms)), 3),
        },
        "surrogate_forward_backward_12d_ms": {
            "median": round(t_fb_med, 3),
            "p95": round(t_fb_p95, 3),
            "mean": round(float(np.mean(fwd_bwd_times_ms)), 3),
            "min": round(float(np.min(fwd_bwd_times_ms)), 3),
        },
        "fem_3d_medium_ms": {
            "median": round(t_fem_med, 2),
            "p95": round(t_fem_p95, 2),
            "mean": round(float(np.mean(fem_times_ms)), 2),
            "min": round(float(np.min(fem_times_ms)), 2),
        },
        "single_solve_speedup": round(speedup_single, 2),
        "grad_12d_central_fd_speedup": round(speedup_12d_cfd, 2),
        "grad_12d_forward_fd_speedup": round(speedup_12d_ffd, 2),
    }

    print(
        f"  Surrogate Forward-only      : median={t_fwd_med:.2f} ms, p95={t_fwd_p95:.2f} ms\n"
        f"  Surrogate Fwd+Bwd (12D Adj) : median={t_fb_med:.2f} ms, p95={t_fb_p95:.2f} ms\n"
        f"  3D FEM medium nonlinear     : median={t_fem_med:.1f} ms, p95={t_fem_p95:.1f} ms\n"
        f"  Single-solve speedup        : {speedup_single:.1f}x\n"
        f"  12D Central-FD grad speedup : {speedup_12d_cfd:.1f}x  (Forward-FD: {speedup_12d_ffd:.1f}x)",
        flush=True,
    )
    return res


# ======================================================================
# EXPERIMENT 2: Large-Scale Independent Closed-Loop Batch (n_A >= 60)
# ======================================================================
def run_experiment2_batch(
    fem_solver: ESC3DFEMSolver,
    fvm_solver: Coupled3DWaferFVMSolver,
    sensor_xy: np.ndarray,
    W_sensor: torch.Tensor,
    spec: RecoverySpec,
    geom: ESC3DGeometrySpec,
    opts: SolveOptions,
    flow_params: HeFlowModelParams,
) -> dict:
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Large-Scale Independent Closed-Loop Batch Validation")
    print(f"  LOCKED Thresholds: S_0_thresh={LOCKED_S_0_THRESH}, S_probe_thresh={LOCKED_S_PROBE_THRESH}")
    print("=" * 72, flush=True)

    t_b0 = time.perf_counter()
    rng_gen = np.random.default_rng(20261125)
    rng_obs = np.random.default_rng(20261125 + 1000)

    cases: list[EvalCase] = []

    # 1) 72 recoverable in-model cases (upfront fixed N=72 to ensure n_A >= 60 without optional stopping)
    #    ~30% no-leak (22 cases) + ~70% mild-to-moderate localized leak (50 cases)
    n_inmodel_noleak = 22
    n_inmodel_leak = 50
    for i in range(n_inmodel_noleak):
        q_pl = float(rng_gen.uniform(460.0, 540.0))
        alpha = float(rng_gen.uniform(-0.10, 0.30))
        cases.append(
            EvalCase(
                name=f"inm_nl_{i + 1:02d}",
                category="in_model",
                cond=make_cond(
                    total_plasma_power_W=q_pl,
                    plasma_peaking_factor=alpha,
                    beta_leak=1.0,
                    leak_theta_center_deg=225.0,
                ),
            )
        )
    for i in range(n_inmodel_leak):
        q_pl = float(rng_gen.uniform(460.0, 540.0))
        alpha = float(rng_gen.uniform(-0.10, 0.30))
        beta = float(rng_gen.uniform(0.70, 0.92))
        theta = float(rng_gen.uniform(0.0, 360.0))
        cases.append(
            EvalCase(
                name=f"inm_lk_{i + 1:02d}",
                category="in_model",
                cond=make_cond(
                    total_plasma_power_W=q_pl,
                    plasma_peaking_factor=alpha,
                    beta_leak=beta,
                    leak_theta_center_deg=theta,
                ),
            )
        )

    # 2) 5 severe unrecoverable leak cases (beta_leak ~ U(0.15, 0.38))
    for i in range(5):
        q_pl = float(rng_gen.uniform(480.0, 520.0))
        alpha = float(rng_gen.uniform(0.00, 0.30))
        beta = float(rng_gen.uniform(0.15, 0.38))
        theta = float(rng_gen.uniform(0.0, 360.0))
        cases.append(
            EvalCase(
                name=f"sev_lk_{i + 1:02d}",
                category="in_model_unrec",
                cond=make_cond(
                    total_plasma_power_W=q_pl,
                    plasma_peaking_factor=alpha,
                    beta_leak=beta,
                    leak_theta_center_deg=theta,
                ),
            )
        )

    # 3) 4 active H_block=1 cases (2 OEM He Flow High >= 20 sccm + 2 missing readback)
    cases.extend([
        EvalCase("hblk_heflow_01", "h_block", make_cond(p_he_out_torr=16.5, beta_leak=0.08, leak_theta_center_deg=225.0)),
        EvalCase("hblk_heflow_02", "h_block", make_cond(p_he_out_torr=17.0, beta_leak=0.05, leak_theta_center_deg=90.0)),
        EvalCase("hblk_readbk_01", "h_block", make_cond(beta_leak=0.60, leak_theta_center_deg=180.0), readback_ok=False),
        EvalCase("hblk_readbk_02", "h_block", make_cond(beta_leak=0.80, leak_theta_center_deg=300.0), readback_ok=False),
    ])

    # 4) 6 OOD hidden-fault cases (2 hotspots, 2 shifted r_leak=128 mm, 2 wafer warpage)
    cases.extend([
        EvalCase("ood_hotspot_01", "ood", make_cond(plasma_hotspot_amp=0.45, plasma_hotspot_r_m=0.135, plasma_hotspot_theta_deg=225.0)),
        EvalCase("ood_hotspot_02", "ood", make_cond(plasma_hotspot_amp=0.40, plasma_hotspot_r_m=0.135, plasma_hotspot_theta_deg=60.0)),
        EvalCase("ood_rleak_01", "ood", make_cond(beta_leak=0.30, r_leak_m=0.128, leak_theta_center_deg=200.0)),
        EvalCase("ood_rleak_02", "ood", make_cond(beta_leak=0.35, r_leak_m=0.128, leak_theta_center_deg=45.0)),
        EvalCase("ood_warpage_01", "ood", make_cond(d_gap_warpage_m=25e-6)),
        EvalCase("ood_warpage_02", "ood", make_cond(d_gap_warpage_m=30e-6)),
    ])

    records: list[dict] = []
    n_total = len(cases)
    for idx, c in enumerate(cases, start=1):
        delta_cal = float(rng_obs.uniform(-flow_params.delta_cal_halfwidth, flow_params.delta_cal_halfwidth))
        fem_0 = fem_solver.solve(c.cond)
        obs_0 = generate_synthetic_observation(
            fem_0,
            c.cond,
            sensor_xy,
            rng_obs,
            delta_cal=delta_cal,
            flow_params=flow_params,
            pressure_tracking_ok=c.pressure_tracking_ok,
            readback_ok=c.readback_ok,
        )

        def make_probe_fn(cond_base: ESC3DOperatingCondition, d_cal: float):
            def _probe(p_out_probe: float):
                cond_p = replace(cond_base, p_he_out_torr=float(p_out_probe))
                fem_p = fem_solver.solve(cond_p)
                return generate_synthetic_observation(
                    fem_p,
                    cond_p,
                    sensor_xy,
                    rng_obs,
                    delta_cal=d_cal,
                    flow_params=flow_params,
                    pressure_tracking_ok=c.pressure_tracking_ok,
                    readback_ok=c.readback_ok,
                )
            return _probe

        s1 = run_stage1_diagnosis(
            obs_0=obs_0,
            probe_obs_fn=make_probe_fn(c.cond, delta_cal),
            solver=fvm_solver,
            sensor_xy=sensor_xy,
            W_sensor=W_sensor,
            opts=opts,
            flow_params=flow_params,
            S_0_thresh=LOCKED_S_0_THRESH,
            S_probe_thresh=LOCKED_S_PROBE_THRESH,
        )

        u_0_case = np.array(
            list(c.cond.heater_powers_W) + [c.cond.p_he_in_torr, c.cond.p_he_out_torr],
            dtype=np.float64,
        )
        s2 = run_stage2_recovery(
            stage1=s1,
            u_0=u_0_case,
            solver=fvm_solver,
            spec=spec,
            geom=geom,
            opts=opts,
        )

        # FREEZE A_pre and u* strictly BEFORE post-hoc 3D FEM evaluation!
        frozen_A_pre = int(s2.A_pre)
        frozen_u_star = s2.u_star.copy()

        # Post-hoc 3D FEM with u* and TRUE HIDDEN FAULT (reuse fem_0 when u* == u_0)
        if np.allclose(frozen_u_star, u_0_case, atol=1e-9):
            fem_recov = fem_0
            q_post = c.cond.heater_powers_W
        else:
            cond_recov = replace(
                c.cond,
                heater_powers_W=tuple(float(x) for x in frozen_u_star[:10]),
                p_he_in_torr=float(frozen_u_star[10]),
                p_he_out_torr=float(frozen_u_star[11]),
            )
            fem_recov = fem_solver.solve(cond_recov)
            q_post = cond_recov.heater_powers_W

        posthoc = grade_posthoc_fem(fem_recov, q_post, spec, geom)

        rec = {
            "name": c.name,
            "category": c.category,
            "true_params": {
                "Q_plasma": round(c.cond.total_plasma_power_W, 2),
                "alpha_peak": round(c.cond.plasma_peaking_factor, 4),
                "beta_leak": round(c.cond.beta_leak, 4),
                "theta_leak_deg": round(c.cond.leak_theta_center_deg, 2),
                "r_leak_m": c.cond.r_leak_m,
                "hotspot_amp": c.cond.plasma_hotspot_amp,
                "d_gap_warpage_m": c.cond.d_gap_warpage_m,
                "delta_cal": round(delta_cal, 4),
            },
            "H_block": s1.H_block,
            "S_0": None if math.isnan(s1.S_0) else round(s1.S_0, 2),
            "S_probe": None if s1.S_probe is None else round(s1.S_probe, 2),
            "stage1_pass": s1.stage1_pass,
            "leak_detected": s1.leak_detected,
            "A_pre": frozen_A_pre,
            "rho_bar": None if math.isnan(s2.rho_bar) else round(s2.rho_bar, 4),
            "hold_reason": s2.hold_reason,
            "posthoc_fem": {
                "V": posthoc["V"],
                "violations": posthoc["violations"],
                "T_mean_eff_C": round(posthoc["T_mean_eff_C"], 3),
                "dT_max_min_eff_C": round(posthoc["dT_max_min_eff_C"], 3),
                "T_3sigma_eff_C": round(posthoc["T_3sigma_eff_C"], 3),
                "max_grad_wafer_eff_C_per_mm": round(posthoc["max_grad_wafer_eff_C_per_mm"], 4),
                "max_grad_esc_C_per_mm": round(posthoc["max_grad_esc_C_per_mm"], 4),
                "Q_cool_W": round(posthoc["Q_cool_W"], 2),
                "max_qdiff_W_cm2": round(posthoc["max_qdiff_W_cm2"], 3),
            },
            "opt_time_s": round(s2.opt_time_s, 3),
        }
        records.append(rec)

        if idx % 10 == 0 or c.category != "in_model" or (frozen_A_pre == 1 and posthoc["V"] == 1) or idx == n_total:
            n_A_so_far = sum(r["A_pre"] for r in records)
            n_V_so_far = sum(1 for r in records if r["A_pre"] == 1 and r["posthoc_fem"]["V"] == 1)
            print(
                f"  [{idx:02d}/{n_total}] {c.name:16s} [{c.category:14s}] | H={s1.H_block} | "
                f"A_pre={frozen_A_pre} | V={posthoc['V']} | dT={posthoc['dT_max_min_eff_C']:.2f}C | "
                f"cum A_pre={n_A_so_far}, V|A={n_V_so_far} | {s2.hold_reason}",
                flush=True,
            )

    n_in_model = sum(1 for r in records if r["category"] == "in_model")
    n_unrec = sum(1 for r in records if r["category"] == "in_model_unrec")
    n_hblock = sum(1 for r in records if r["category"] == "h_block")
    n_ood = sum(1 for r in records if r["category"] == "ood")

    n_A = sum(r["A_pre"] for r in records)
    n_A_inmodel = sum(r["A_pre"] for r in records if r["category"] == "in_model")
    n_unrec_held = sum(1 for r in records if r["category"] == "in_model_unrec" and r["A_pre"] == 0)
    n_hblock_caught = sum(1 for r in records if r["category"] == "h_block" and r["H_block"] == 1 and r["A_pre"] == 0)
    n_ood_blocked = sum(1 for r in records if r["category"] == "ood" and r["A_pre"] == 0)
    n_V1_given_A1 = sum(1 for r in records if r["A_pre"] == 1 and r["posthoc_fem"]["V"] == 1)

    p_viol = (n_V1_given_A1 / n_A) if n_A > 0 else None
    cp_ub_95 = clopper_pearson_upper_95(n_V1_given_A1, n_A)

    summary = {
        "locked_S_0_thresh": LOCKED_S_0_THRESH,
        "locked_S_probe_thresh": LOCKED_S_PROBE_THRESH,
        "rng_seed": 20261125,
        "n_test_total": n_total,
        "n_in_model": n_in_model,
        "n_in_model_unrec": n_unrec,
        "n_h_block": n_hblock,
        "n_ood": n_ood,
        "n_A_pre": n_A,
        "n_A_pre_in_model": n_A_inmodel,
        "approval_rate_in_model": round(n_A_inmodel / n_in_model, 4),
        "severe_leak_hold_rate": round(n_unrec_held / n_unrec, 4),
        "h_block_catch_rate": round(n_hblock_caught / n_hblock, 4),
        "ood_block_rate": round(n_ood_blocked / n_ood, 4),
        "n_V1_given_A1": n_V1_given_A1,
        "P_V1_given_A1": None if p_viol is None else round(p_viol, 5),
        "clopper_pearson_95_upper": None if cp_ub_95 is None else round(cp_ub_95, 5),
        "runtime_s": round(time.perf_counter() - t_b0, 2),
    }
    print(f"\n  Batch Summary: {json.dumps(summary, indent=2)}", flush=True)
    return {"summary": summary, "cases": records}


# ======================================================================
# EXPERIMENT 3: Operable Envelope 2D Phase Map (alpha_peak vs beta_leak)
# ======================================================================
def _compute_laplace_sigma_d(
    d_vec: np.ndarray,
    u_0_nom: np.ndarray,
    fvm_solver: Coupled3DWaferFVMSolver,
    W_sensor: torch.Tensor,
    opts: SolveOptions,
    flow_params: HeFlowModelParams,
    sigma_T_eff: float = 0.20,
) -> dict[str, float]:
    """Compute exact Laplace/Gauss-Newton posterior std sigma_d at nominal d_vec."""
    Q_htr_t = torch.as_tensor(u_0_nom[:10], dtype=torch.float64, device=fvm_solver.device)
    p_in_0 = float(u_0_nom[10])
    p_out_0 = float(u_0_nom[11])

    def _eval_pred(x: np.ndarray) -> tuple[np.ndarray, float]:
        with torch.no_grad():
            out = fvm_solver.solve(
                Q_heaters=Q_htr_t,
                p_in=p_in_0,
                p_out=p_out_0,
                beta_leak=float(x[2]),
                Q_plasma=float(x[0]),
                alpha_peak=float(x[1]),
                opts=opts,
                theta_leak_deg=float(x[3]),
            )
            T_p = (W_sensor @ out["T_wafer_C"]).cpu().numpy()
            m_p = float(
                eval_he_flow_proxy_torch(
                    torch.tensor(float(x[2]), dtype=torch.float64, device=fvm_solver.device),
                    p_out_0,
                    flow_params,
                ).item()
            )
        return T_p, m_p

    T_0, m_0 = _eval_pred(d_vec)
    excess = max(m_0 - flow_params.m_base_sccm, 0.0)
    sigma_m = math.hypot(flow_params.sigma_m_est_base_sccm, flow_params.sigma_m_est_rel * excess)

    eps_d = np.array([1.0, 5e-3, 5e-3, 1.0], dtype=np.float64)
    n_s = W_sensor.shape[0]
    J_r = np.zeros((n_s + 1, 4), dtype=np.float64)
    for j in range(4):
        xp = d_vec.copy()
        step = eps_d[j]
        if j == 2 and xp[2] + step > 1.0:
            step = -step
        xp[j] += step
        T_p, m_p = _eval_pred(xp)
        J_r[:n_s, j] = (T_p - T_0) / (step * sigma_T_eff)
        J_r[-1, j] = (m_p - m_0) / (step * sigma_m)

    prior_std = np.array([150.0, 0.5, 0.4, 90.0], dtype=np.float64)
    H_gn = J_r.T @ J_r + np.diag(1.0 / (prior_std ** 2))
    cov_d = np.linalg.inv(H_gn)
    sig_vec = np.sqrt(np.maximum(np.diag(cov_d), 1e-12))
    return {
        "Q_plasma": float(sig_vec[0]),
        "alpha_peak": float(sig_vec[1]),
        "beta_leak": float(sig_vec[2]),
        "theta_leak_deg": float(sig_vec[3]),
    }


def _solve_envelope_point_warm(
    stage1: Stage1Result,
    u_0_nom: np.ndarray,
    u_warm: np.ndarray | None,
    solver: Coupled3DWaferFVMSolver,
    spec: RecoverySpec,
    geom: ESC3DGeometrySpec,
    opts: SolveOptions,
) -> tuple[float, float, np.ndarray, str]:
    """Warm-started Stage 2 solver identical in physics/bounds/rho_bar to run_stage2_recovery."""
    heater_areas_cm2 = _heater_areas_cm2_torch(geom, solver.device)
    d_hat = stage1.d_hat
    Q_pl_hat = float(d_hat["Q_plasma"])
    alpha_hat = float(d_hat["alpha_peak"])
    beta_hat = float(d_hat["beta_leak"])
    theta_hat = float(d_hat["theta_leak_deg"])

    p_out_hi = float(u_0_nom[11]) if stage1.leak_detected else spec.P_max_torr
    lo_phys = np.array([0.0] * 10 + [spec.P_min_torr, spec.P_min_torr], dtype=np.float64)
    hi_phys = np.array(
        [spec.Q_inner_max_W, spec.Q_inner_max_W] + [spec.Q_edge_max_W] * 8 + [spec.P_max_torr, p_out_hi],
        dtype=np.float64,
    )
    u_scales = np.array([200.0, 200.0] + [100.0] * 8 + [10.0, 10.0], dtype=np.float64)
    bounds_z = [(lo / s, hi / s) for lo, hi, s in zip(lo_phys, hi_phys, u_scales)]
    u_0_clipped = np.clip(u_0_nom, lo_phys, hi_phys)

    def _eval_obj_and_grad(z: np.ndarray, focus_feasibility: bool = False) -> tuple[float, np.ndarray]:
        u = np.clip(z * u_scales, lo_phys, hi_phys)
        u_t = torch.tensor(u, dtype=torch.float64, device=solver.device, requires_grad=True)
        _, g_norm, m = eval_constraints_torch(
            solver=solver, Q_heaters=u_t[:10], p_in=u_t[10], p_out=u_t[11],
            Q_plasma=Q_pl_hat, alpha_peak=alpha_hat, beta_leak=beta_hat, theta_leak_deg=theta_hat,
            spec=spec, heater_areas_cm2=heater_areas_cm2, opts=opts, smooth=True,
        )
        g_stack = torch.stack(list(g_norm.values()))
        tau_rho = 0.04 if focus_feasibility else 0.08
        rho_lse = tau_rho * torch.logsumexp(g_stack / tau_rho, dim=0)
        u_0_t = torch.as_tensor(u_0_clipped, dtype=torch.float64, device=solver.device)
        u_s_t = torch.as_tensor(u_scales, dtype=torch.float64, device=solver.device)
        reg = torch.sum(((u_t - u_0_t) / u_s_t).pow(2))

        if focus_feasibility:
            loss = (
                4.0 * rho_lse
                + 2.0 * m["var_A_eff"]
                + 1.0 * (m["T_mean_eff_C"] - spec.T_target_C).pow(2)
                + 0.005 * reg
            )
        else:
            viol_pen = torch.sum(torch.nn.functional.softplus((g_stack + 0.05) * 15.0).pow(2)) / 225.0
            loss = (
                1.5 * rho_lse
                + 8.0 * viol_pen
                + 4.0 * m["var_A_eff"]
                + 2.0 * (m["T_mean_eff_C"] - spec.T_target_C).pow(2)
                + 0.015 * reg
            )
        loss.backward()
        return float(loss.item()), u_t.grad.detach().cpu().numpy() * u_scales

    def _eval_exact_at(u_vec: np.ndarray, d_dict: dict[str, float]):
        with torch.no_grad():
            u_t = torch.as_tensor(u_vec, dtype=torch.float64, device=solver.device)
            _, g_norm, _ = eval_constraints_torch(
                solver=solver, Q_heaters=u_t[:10], p_in=u_t[10], p_out=u_t[11],
                Q_plasma=d_dict["Q_plasma"], alpha_peak=d_dict["alpha_peak"],
                beta_leak=d_dict["beta_leak"], theta_leak_deg=d_dict["theta_leak_deg"],
                spec=spec, heater_areas_cm2=heater_areas_cm2, opts=opts, smooth=False,
            )
            g_f = {k: float(v.item()) for k, v in g_norm.items()}
            return max(g_f.values()), g_f

    u_seed_balanced = np.clip(
        np.array([40.0, 120.0] + [75.0] * 8 + [12.0, min(12.0, p_out_hi)], dtype=np.float64),
        lo_phys, hi_phys,
    )
    starts = [np.clip(u_warm, lo_phys, hi_phys), u_seed_balanced] if u_warm is not None else [u_seed_balanced, u_0_clipped]

    best_u = u_0_clipped.copy()
    best_rho = float("inf")
    best_g: dict[str, float] = {}

    for u_init in starts:
        res_a = minimize(
            lambda z: _eval_obj_and_grad(z, focus_feasibility=False),
            u_init / u_scales,
            method="SLSQP",
            jac=True,
            bounds=bounds_z,
            options={"maxiter": 50, "ftol": 1e-8},
        )
        u_cand = np.clip(res_a.x * u_scales, lo_phys, hi_phys)
        rho_c, g_c = _eval_exact_at(u_cand, d_hat)
        if rho_c < best_rho:
            best_u, best_rho, best_g = u_cand, rho_c, g_c
        if best_rho < -0.15:
            break

    if best_rho > -0.05:
        res_b = minimize(
            lambda z: _eval_obj_and_grad(z, focus_feasibility=True),
            best_u / u_scales,
            method="SLSQP",
            jac=True,
            bounds=bounds_z,
            options={"maxiter": 40, "ftol": 1e-8},
        )
        u_cand = np.clip(res_b.x * u_scales, lo_phys, hi_phys)
        rho_c, g_c = _eval_exact_at(u_cand, d_hat)
        if rho_c < best_rho:
            best_u, best_rho, best_g = u_cand, rho_c, g_c

    # Robust upper bound rho_bar (exact same formula as run_stage2_recovery)
    sig = stage1.sigma_d
    active_axes: list[tuple[str, float, float, float]] = [
        ("Q_plasma", min(1.5 * sig["Q_plasma"], 15.0), 250.0, 800.0),
        ("alpha_peak", min(1.5 * sig["alpha_peak"], 0.08), -0.6, 0.9),
    ]
    if stage1.leak_detected or beta_hat < 0.96:
        active_axes.append(("beta_leak", min(1.5 * sig["beta_leak"], 0.05), 0.05, 1.0))
        active_axes.append(("theta_leak_deg", min(1.5 * sig["theta_leak_deg"], 10.0), -360.0, 720.0))

    worst_g = dict(best_g)
    for name, half_w, lo_d, hi_d in active_axes:
        for sign in (-1.0, 1.0):
            sc = dict(d_hat)
            val = d_hat[name] + sign * half_w
            sc[name] = float(val % 360.0) if name == "theta_leak_deg" else float(np.clip(val, lo_d, hi_d))
            _, g_sc = _eval_exact_at(best_u, sc)
            for k, v in g_sc.items():
                if v > worst_g[k]:
                    worst_g[k] = v

    u_star_t = torch.as_tensor(best_u, dtype=torch.float64, device=solver.device)
    d_tensors = {
        "Q_plasma": torch.tensor(Q_pl_hat, dtype=torch.float64, device=solver.device, requires_grad=True),
        "alpha_peak": torch.tensor(alpha_hat, dtype=torch.float64, device=solver.device, requires_grad=True),
        "beta_leak": torch.tensor(beta_hat, dtype=torch.float64, device=solver.device, requires_grad=True),
        "theta_leak_deg": torch.tensor(theta_hat, dtype=torch.float64, device=solver.device, requires_grad=True),
    }
    _, g_norm_diff, _ = eval_constraints_torch(
        solver=solver, Q_heaters=u_star_t[:10], p_in=u_star_t[10], p_out=u_star_t[11],
        Q_plasma=d_tensors["Q_plasma"], alpha_peak=d_tensors["alpha_peak"],
        beta_leak=d_tensors["beta_leak"], theta_leak_deg=d_tensors["theta_leak_deg"],
        spec=spec, heater_areas_cm2=heater_areas_cm2, opts=opts, smooth=True,
    )
    active_names = [a[0] for a in active_axes]
    active_half_w = {a[0]: a[1] for a in active_axes}
    active_t_list = [d_tensors[nm] for nm in active_names]
    h_frac = 0.25

    c_cov: dict[str, float] = {}
    for k, g_val in g_norm_diff.items():
        if k == "qdiff":
            c_cov[k] = float(worst_g[k])
            continue
        grads = torch.autograd.grad(g_val, active_t_list, retain_graph=True, allow_unused=True)
        cov_sq = sum(
            (float(g_d.item()) * h_frac * active_half_w[nm]) ** 2
            for nm, g_d in zip(active_names, grads)
            if g_d is not None
        )
        c_cov[k] = float(worst_g[k] + math.sqrt(cov_sq))

    rho_bar = max(c_cov.values())
    binding_k = max(c_cov, key=lambda k: c_cov[k])
    return float(best_rho), float(rho_bar), best_u, binding_k


def plot_operable_envelope(
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    rho_bar_grid: np.ndarray,
    out_path: Path,
) -> None:
    """Save publication-grade 2D Operable Envelope Phase Map (alpha_peak vs beta_leak)."""
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=180)
    A, B = np.meshgrid(alpha_vals, beta_vals, indexing="ij")  # shape (7, 9)

    vmin = float(min(-0.6, np.min(rho_bar_grid)))
    vmax = float(max(0.6, np.max(rho_bar_grid)))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    levels = np.linspace(vmin, vmax, 33)
    cf = ax.contourf(A, B, rho_bar_grid, levels=levels, cmap="RdYlGn_r", norm=norm, extend="both")

    # Subtle secondary iso-contours
    cs_sub = ax.contour(
        A, B, rho_bar_grid,
        levels=[-0.4, -0.2, 0.2, 0.5, 1.0],
        colors="#333333", linewidths=0.7, linestyles="--", alpha=0.65,
    )
    ax.clabel(cs_sub, fmt="%.1f", fontsize=8, inline=True)

    # Bold recovery boundary rho_bar = 0
    cs_zero = ax.contour(A, B, rho_bar_grid, levels=[0.0], colors="black", linewidths=2.6)
    ax.clabel(cs_zero, fmt={0.0: "ρ̄ = 0 (Boundary)"}, fontsize=9.5, inline=True)

    # Discrete grid markers
    op_mask = rho_bar_grid <= 0.0
    ax.scatter(
        A[op_mask], B[op_mask],
        c="#0b6623", edgecolors="white", s=46, linewidths=0.9, zorder=5,
        label="Operable / Pre-Approved (ρ̄ ≤ 0)",
    )
    ax.scatter(
        A[~op_mask], B[~op_mask],
        c="#b22222", marker="X", edgecolors="white", s=58, linewidths=0.7, zorder=5,
        label="Hold / Unrecoverable (ρ̄ > 0)",
    )

    # Region text callouts
    ax.text(
        0.05, 0.91,
        "OPERABLE / PRE-APPROVED\n(ρ̄ ≤ 0)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#073b14",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.80, edgecolor="#0b6623"),
    )
    ax.text(
        0.05, 0.34,
        "HOLD / UNRECOVERABLE\n(ρ̄ > 0)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#7a1111",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.80, edgecolor="#b22222"),
    )

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Robust Normalized Margin  ρ̄(u*)   [≤ 0: Feasible, > 0: Violated]", fontsize=10)
    cbar.ax.axhline(0.0, color="black", lw=2.0)

    ax.set_xlabel("Plasma Radial Peaking Factor  α_peak  [-]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Localized Edge He Leak Retention  β_leak  [-]", fontsize=11, fontweight="bold")
    ax.set_xlim(alpha_vals[0] - 0.02, alpha_vals[-1] + 0.02)
    ax.set_ylim(beta_vals[0] - 0.03, beta_vals[-1] + 0.03)
    ax.set_xticks(alpha_vals)
    ax.set_yticks(beta_vals)
    ax.grid(True, linestyle=":", alpha=0.35, color="#444444")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)

    ax.set_title(
        "3D Multi-Zone ESC Safe Recovery Operable Envelope (Al2O3, 10-Zone Heater + Dual He)\n"
        "Fixed: Q_plasma=500 W, T_target=40.0°C, θ_leak=225°, r_leak=138 mm\n"
        "u0=(P_in=12, P_out=14 Torr), P_out ≤ 14 Torr interlock (β_leak < 0.90)",
        fontsize=9.5, fontweight="bold", pad=10,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_experiment3_envelope(
    fvm_solver: Coupled3DWaferFVMSolver,
    W_sensor: torch.Tensor,
    spec: RecoverySpec,
    geom: ESC3DGeometrySpec,
    opts: SolveOptions,
    flow_params: HeFlowModelParams,
    out_png: Path,
) -> dict:
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Operable Envelope 2D Phase Map (alpha_peak x beta_leak)")
    print("=" * 72, flush=True)

    t_e0 = time.perf_counter()
    alpha_vals = np.round(np.linspace(-0.20, 0.40, 7), 4)
    beta_vals = np.round(np.linspace(0.20, 1.00, 9), 4)
    u_0_nom = np.array(list(BASELINE_Q_HEATERS) + [12.0, 14.0], dtype=np.float64)

    n_a, n_b = len(alpha_vals), len(beta_vals)
    rho_bar_grid = np.zeros((n_a, n_b), dtype=np.float64)
    rho_nom_grid = np.zeros((n_a, n_b), dtype=np.float64)
    binding_grid: list[list[str]] = [["" for _ in range(n_b)] for _ in range(n_a)]

    # Sweep each alpha row from beta=1.00 down to 0.20 with warm-starting
    for ia, alpha in enumerate(alpha_vals):
        u_warm: np.ndarray | None = None
        for ib in range(n_b - 1, -1, -1):
            beta = float(beta_vals[ib])
            d_vec = np.array([500.0, float(alpha), beta, 225.0], dtype=np.float64)
            sigma_d = _compute_laplace_sigma_d(
                d_vec, u_0_nom, fvm_solver, W_sensor, opts, flow_params
            )
            s1_clean = Stage1Result(
                H_block=0,
                d_hat={
                    "Q_plasma": 500.0,
                    "alpha_peak": float(alpha),
                    "beta_leak": beta,
                    "theta_leak_deg": 225.0,
                },
                sigma_d=sigma_d,
                S_0=0.0,
                S_probe=0.0,
                stage1_pass=True,
                leak_detected=bool(beta < 0.90),
                hold_reason="OK",
                pm_review=False,
                readback_ok=True,
            )
            rho_nom, rho_bar, u_star, binding_k = _solve_envelope_point_warm(
                stage1=s1_clean,
                u_0_nom=u_0_nom,
                u_warm=u_warm,
                solver=fvm_solver,
                spec=spec,
                geom=geom,
                opts=opts,
            )
            rho_nom_grid[ia, ib] = rho_nom
            rho_bar_grid[ia, ib] = rho_bar
            binding_grid[ia][ib] = binding_k
            u_warm = u_star

        op_betas = [float(beta_vals[ib]) for ib in range(n_b) if rho_bar_grid[ia, ib] <= 0.0]
        min_b_str = f"{min(op_betas):.2f}" if op_betas else "None"
        row_str = " ".join(f"{rho_bar_grid[ia, ib]:+5.2f}" for ib in range(n_b))
        print(f"  alpha={alpha:+5.2f} | min_recov_beta={min_b_str:>4s} | rho_bar: [{row_str}]", flush=True)

    # Interpolated exact beta_crit(alpha) where rho_bar = 0 along each alpha slice
    boundary_by_alpha: dict[str, float | None] = {}
    for ia, alpha in enumerate(alpha_vals):
        row = rho_bar_grid[ia, :]
        beta_crit = None
        for ib in range(n_b - 1):
            if row[ib] > 0.0 and row[ib + 1] <= 0.0:
                frac = row[ib] / (row[ib] - row[ib + 1])
                beta_crit = float(beta_vals[ib] + frac * (beta_vals[ib + 1] - beta_vals[ib]))
                break
        if beta_crit is None and row[0] <= 0.0:
            beta_crit = float(beta_vals[0])
        boundary_by_alpha[f"{alpha:+.2f}"] = None if beta_crit is None else round(beta_crit, 4)

    operable_mask = rho_bar_grid <= 0.0
    n_operable = int(np.sum(operable_mask))
    valid_crits = [v for v in boundary_by_alpha.values() if v is not None]
    min_recov_beta_discrete = float(np.min(np.broadcast_to(beta_vals[None, :], rho_bar_grid.shape)[operable_mask])) if n_operable > 0 else None
    min_recov_beta_interp = min(valid_crits) if valid_crits else None
    max_recov_beta_interp = max(valid_crits) if valid_crits else None

    plot_operable_envelope(alpha_vals, beta_vals, rho_bar_grid, out_png)

    res = {
        "fixed_conditions": {
            "Q_plasma_W": 500.0,
            "T_target_C": spec.T_target_C,
            "theta_leak_deg": 225.0,
            "r_leak_m": 0.138,
            "u_0_P_in_torr": 12.0,
            "u_0_P_out_torr": 14.0,
            "leak_interlock_rule": "P_out <= 14.0 Torr whenever beta_leak < 0.90",
        },
        "alpha_peak_vals": [float(x) for x in alpha_vals],
        "beta_leak_vals": [float(x) for x in beta_vals],
        "n_grid_points": int(n_a * n_b),
        "n_operable": n_operable,
        "operable_fraction": round(n_operable / (n_a * n_b), 4),
        "min_recoverable_beta_leak_discrete": min_recov_beta_discrete,
        "min_recoverable_beta_leak_interp": min_recov_beta_interp,
        "max_boundary_beta_leak_interp": max_recov_beta_interp,
        "beta_crit_by_alpha": boundary_by_alpha,
        "rho_bar_grid": [[round(float(v), 4) for v in row] for row in rho_bar_grid],
        "rho_nom_grid": [[round(float(v), 4) for v in row] for row in rho_nom_grid],
        "binding_constraint_grid": binding_grid,
        "plot_path": str(out_png),
        "runtime_s": round(time.perf_counter() - t_e0, 2),
    }
    print(
        f"  Operable points: {n_operable}/{n_a * n_b} ({100.0 * n_operable / (n_a * n_b):.1f}%) | "
        f"min recoverable beta_leak: {min_recov_beta_discrete} (interp: {min_recov_beta_interp}) | "
        f"saved plot: {out_png}",
        flush=True,
    )
    return res


def main() -> None:
    t_all = time.perf_counter()
    refinement = "medium"
    geom = ESC3DGeometrySpec()
    spec = RecoverySpec(T_target_C=40.0)
    flow_params = HeFlowModelParams()
    opts = SolveOptions(T_chiller_C=20.0, T_wall_C=60.0)

    print(f"=== Phase 1.5 Quantitative Verification Suite (refinement={refinement}) ===", flush=True)
    fem_solver = ESC3DFEMSolver(refinement=refinement, geom=geom)
    fvm_solver = Coupled3DWaferFVMSolver(refinement=refinement, ceramic_type="Al2O3")
    sensor_xy, _, _ = build_sensarray_65_coords()
    W_sensor = build_polar_sensor_interp_matrix(fvm_solver, sensor_xy)

    out_dir = PROJECT_ROOT / "results" / "benchmarks"
    out_json = out_dir / "phase15_verification_suite.json"
    out_png = out_dir / "phase15_operable_envelope.png"

    exp1 = run_experiment1_timing(fem_solver, fvm_solver, spec, geom, opts)
    exp2 = run_experiment2_batch(fem_solver, fvm_solver, sensor_xy, W_sensor, spec, geom, opts, flow_params)
    exp3 = run_experiment3_envelope(fvm_solver, W_sensor, spec, geom, opts, flow_params, out_png)

    b_sum = exp2["summary"]
    overall_pass = bool(
        (b_sum["n_A_pre"] >= 60)
        and (b_sum["n_V1_given_A1"] == 0)
        and (b_sum["clopper_pearson_95_upper"] is not None and b_sum["clopper_pearson_95_upper"] < 0.05)
        and (b_sum["h_block_catch_rate"] == 1.0)
        and (b_sum["ood_block_rate"] == 1.0)
        and (b_sum["severe_leak_hold_rate"] == 1.0)
        and (exp3["n_operable"] > 0)
    )

    payload = {
        "overall_pass": overall_pass,
        "total_runtime_s": round(time.perf_counter() - t_all, 2),
        "experiment1_timing": exp1,
        "experiment2_batch_validation": exp2,
        "experiment3_operable_envelope": exp3,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 72)
    print(f"PHASE 1.5 SUITE COMPLETE | overall_pass={overall_pass} | total_runtime={payload['total_runtime_s']} s")
    print(f"Saved JSON: {out_json}")
    print(f"Saved PNG : {out_png}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
