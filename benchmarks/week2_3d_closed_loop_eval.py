"""Week 2 Closed-Loop 3D Evaluation: Calibration, Independent Test Table & Showcase Plot.

Runs on the 3D FEniCSx FEM (`medium` mesh) + Coupled 3D Wafer FVM Surrogate:
  - Part A (Calibration Set, 8 paired nominal/in-model runs):
      Locks S_0_thresh and S_probe_thresh on independent calibration data.
  - Part B (Independent Test Set, 15 independent runs with fresh random drift/noise):
      * 9 in-model cases (nominal, radial plasma peaking shifts, mild-to-severe
        localized azimuthal edge He leaks at multiple angles)
      * 2 active H_block=1 cases (extreme He flow >= 20 sccm alarm, missing readback)
      * 4 OOD hidden-fault cases (2 localized azimuthal plasma hotspots with normal He flow,
        shifted r_leak=132 mm, and wafer warpage d(r))
  - Freezes A_pre and u* BEFORE running independent 3D FEniCSx FEM with u* and
    the TRUE HIDDEN FAULT to grade post-hoc V (original specs, NO pre-approval margins).
  - Computes Approval Rate, P(V=1 | A_pre=1) with one-sided 95% Clopper-Pearson upper bound,
    saves JSON to `results/benchmarks/week2_3d_closed_loop_eval.json`, and saves the 3-panel
    300 mm 2D wafer colormap to `results/benchmarks/week2_3d_recovery_showcase.png`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import beta as beta_dist, chi2

from src.fem3d.build_esc_3d_mesh import ESC3DGeometrySpec
from src.fem3d.esc_3d_fem_solver import ESC3DFEMSolver, ESC3DOperatingCondition
from src.inverse.stage1_3d_diagnosis import (
    HeFlowModelParams,
    build_polar_sensor_interp_matrix,
    build_sensarray_65_coords,
    generate_synthetic_observation,
    run_stage1_diagnosis,
)
from src.inverse.stage2_3d_recovery import (
    ADJACENT_HEATER_PAIRS,
    RecoverySpec,
    run_stage2_recovery,
)
from src.solvers.coupled_3d_wafer_fvm import Coupled3DWaferFVMSolver, SolveOptions


# Baseline 10-zone recipe tuned near T_target = 40.0 C at nominal 500 W plasma, 14 Torr He
BASELINE_Q_HEATERS: tuple[float, ...] = (80.0, 220.0) + (65.0,) * 8
BASELINE_P_IN: float = 14.0
BASELINE_P_OUT: float = 14.0


@dataclass(frozen=True)
class EvalCase:
    name: str
    category: str          # "in_model" | "h_block" | "ood"
    cond: ESC3DOperatingCondition
    pressure_tracking_ok: bool = True
    readback_ok: bool = True
    is_showcase: bool = False


def make_cond(**overrides) -> ESC3DOperatingCondition:
    defaults = dict(
        ceramic_type="Al2O3",
        total_plasma_power_W=500.0,
        plasma_peaking_factor=0.25,
        p_he_in_torr=BASELINE_P_IN,
        p_he_out_torr=BASELINE_P_OUT,
        beta_leak=1.0,
        r_leak_m=0.138,
        leak_theta_center_deg=225.0,
        heater_powers_W=BASELINE_Q_HEATERS,
        T_chiller_C=20.0,
        T_wall_C=60.0,
    )
    defaults.update(overrides)
    return ESC3DOperatingCondition(**defaults)


def compute_fem_surface_max_grad_C_per_mm(
    xy: np.ndarray,
    T_vals: np.ndarray,
    r_max_m: float,
    grid_step_m: float = 0.003,
) -> float:
    """Compute max |grad T| [C/mm] on a 3D FEM surface over r <= r_max_m via 3 mm grid."""
    lin = LinearNDInterpolator(xy, T_vals)
    xs = np.arange(-r_max_m, r_max_m + 1e-9, grid_step_m)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    T_grid = lin(gx, gy)
    dT_dx = (T_grid[2:, 1:-1] - T_grid[:-2, 1:-1]) / (2.0 * grid_step_m)
    dT_dy = (T_grid[1:-1, 2:] - T_grid[1:-1, :-2]) / (2.0 * grid_step_m)
    rr_in = np.hypot(gx[1:-1, 1:-1], gy[1:-1, 1:-1])
    grad_mag = np.hypot(dT_dx, dT_dy) * 1e-3  # C/mm
    valid = (rr_in <= r_max_m - 0.5 * grid_step_m) & np.isfinite(grad_mag)
    return float(np.nanmax(grad_mag[valid]))


def compute_max_qdiff_np(Q_heaters: tuple[float, ...], geom: ESC3DGeometrySpec) -> float:
    areas_cm2 = np.asarray([a * 1e4 for a in geom.heater_zone_areas], dtype=np.float64)
    q_flux = np.asarray(Q_heaters, dtype=np.float64) / areas_cm2
    return float(max(abs(q_flux[i] - q_flux[j]) for i, j in ADJACENT_HEATER_PAIRS))


def grade_posthoc_fem(
    fem_res: dict,
    Q_heaters: tuple[float, ...],
    spec: RecoverySpec,
    geom: ESC3DGeometrySpec,
) -> dict:
    """Grade post-hoc V on 3D FEM top surface (NO pre-approval margins!)."""
    T_mean = float(fem_res["T_mean_eff_C"])
    dT_span = float(fem_res["dT_max_min_eff_C"])
    T_3sig = float(fem_res["T_3sigma_eff_C"])
    Q_cool = float(fem_res["Q_cool_W"])

    grad_w = compute_fem_surface_max_grad_C_per_mm(
        fem_res["wafer_top_xy"], fem_res["T_wafer_top_C"], r_max_m=spec.r_eff_m
    )
    grad_esc = compute_fem_surface_max_grad_C_per_mm(
        fem_res["esc_top_xy"], fem_res["T_esc_top_C"], r_max_m=geom.r_esc - 0.001
    )
    qdiff = compute_max_qdiff_np(Q_heaters, geom)

    violations: list[str] = []
    if abs(T_mean - spec.T_target_C) > spec.limit_mean_err_C:
        violations.append(f"|T_mean-T_tgt|={abs(T_mean - spec.T_target_C):.3f}>{spec.limit_mean_err_C}")
    if dT_span > spec.limit_dT_span_C:
        violations.append(f"dT={dT_span:.3f}>{spec.limit_dT_span_C}")
    if T_3sig > spec.limit_3sigma_C:
        violations.append(f"3sig={T_3sig:.3f}>{spec.limit_3sigma_C}")
    if grad_w > spec.limit_wafer_grad_C_per_mm:
        violations.append(f"grad_w={grad_w:.3f}>{spec.limit_wafer_grad_C_per_mm}")
    if grad_esc > spec.limit_esc_grad_C_per_mm:
        violations.append(f"grad_esc={grad_esc:.3f}>{spec.limit_esc_grad_C_per_mm}")
    if Q_cool > spec.limit_Q_cool_W:
        violations.append(f"Q_cool={Q_cool:.1f}>{spec.limit_Q_cool_W}")
    if qdiff > spec.limit_qdiff_W_cm2:
        violations.append(f"qdiff={qdiff:.3f}>{spec.limit_qdiff_W_cm2}")

    return {
        "V": 1 if violations else 0,
        "violations": violations,
        "T_mean_eff_C": T_mean,
        "dT_max_min_eff_C": dT_span,
        "T_3sigma_eff_C": T_3sig,
        "max_grad_wafer_eff_C_per_mm": grad_w,
        "max_grad_esc_C_per_mm": grad_esc,
        "Q_cool_W": Q_cool,
        "max_qdiff_W_cm2": qdiff,
    }


def clopper_pearson_upper_95(k: int, n: int) -> float | None:
    """One-sided 95% Clopper-Pearson upper confidence bound for binomial rate k/n."""
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(0.95, k + 1, n - k))


def plot_3panel_showcase(
    fem_faulty: dict,
    T_pred_2d: np.ndarray,
    solver: Coupled3DWaferFVMSolver,
    fem_recovered: dict,
    d_hat: dict[str, float],
    out_path: Path,
) -> None:
    """Save 3-panel 300 mm 2D wafer colormap: Faulty 3D FEM -> Surrogate -> 3D FEM Re-Verification."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=160)
    grid_n = 220
    xs = np.linspace(-0.150, 0.150, grid_n)
    gx, gy = np.meshgrid(xs, xs)
    rr = np.hypot(gx, gy)
    mask_out = rr > 0.150

    def interp_to_disk(xy: np.ndarray, vals: np.ndarray) -> np.ndarray:
        lin = LinearNDInterpolator(xy, vals)
        Z = lin(gx, gy)
        Z[mask_out] = np.nan
        return Z

    Z_faulty = interp_to_disk(fem_faulty["wafer_top_xy"], fem_faulty["T_wafer_top_C"])
    Z_recov = interp_to_disk(fem_recovered["wafer_top_xy"], fem_recovered["T_wafer_top_C"])

    # Surrogate 2D polar grid -> disk grid (include center & periodic wrap via xy points)
    surr_xy = np.column_stack([
        solver.xx_flat.detach().cpu().numpy(),
        solver.yy_flat.detach().cpu().numpy(),
    ])
    Z_surr = interp_to_disk(surr_xy, T_pred_2d.reshape(-1))

    # Common colormap limits spanning all 3 panels so the thermal flattening is immediately visible
    vmin = float(min(np.nanmin(Z_faulty), np.nanmin(Z_surr), np.nanmin(Z_recov)))
    vmax = float(max(np.nanmax(Z_faulty), np.nanmax(Z_surr), np.nanmax(Z_recov)))
    levels = np.linspace(vmin, vmax, 36)

    # Surrogate effective metrics for panel title
    r_c = solver.r_centers.detach().cpu().numpy()
    eff_m = r_c <= 0.147 + 1e-9
    T_s_eff = T_pred_2d[eff_m, :].reshape(-1)
    A_s_eff = solver.cell_areas.detach().cpu().numpy()[eff_m, :].reshape(-1)
    w_s = A_s_eff / np.sum(A_s_eff)
    T_s_mean = float(np.sum(w_s * T_s_eff))
    dT_s = float(np.max(T_s_eff) - np.min(T_s_eff))
    sig3_s = float(3.0 * math.sqrt(np.sum(w_s * (T_s_eff - T_s_mean) ** 2)))

    panels = [
        (
            Z_faulty,
            "1. Faulty 3D FEM (u0, Edge Leak 225°)",
            f"Mean={fem_faulty['T_mean_eff_C']:.2f}°C | dT={fem_faulty['dT_max_min_eff_C']:.2f}°C | 3σ={fem_faulty['T_3sigma_eff_C']:.2f}°C",
        ),
        (
            Z_surr,
            f"2. Surrogate Prediction (u*, β̂={d_hat['beta_leak']:.2f}, θ̂={d_hat['theta_leak_deg']:.0f}°)",
            f"Mean={T_s_mean:.2f}°C | dT={dT_s:.2f}°C | 3σ={sig3_s:.2f}°C",
        ),
        (
            Z_recov,
            "3. Independent 3D FEM Re-Verification (u*)",
            f"Mean={fem_recovered['T_mean_eff_C']:.2f}°C | dT={fem_recovered['dT_max_min_eff_C']:.2f}°C | 3σ={fem_recovered['T_3sigma_eff_C']:.2f}°C",
        ),
    ]

    th_circ = np.linspace(0.0, 2.0 * np.pi, 240)
    cf = None
    for ax, (Z, title, subtitle) in zip(axes, panels):
        cf = ax.contourf(gx * 1e3, gy * 1e3, Z, levels=levels, cmap="inferno", extend="both")
        ax.plot(150.0 * np.cos(th_circ), 150.0 * np.sin(th_circ), "k-", lw=1.2)
        ax.plot(147.0 * np.cos(th_circ), 147.0 * np.sin(th_circ), "w--", lw=0.9, alpha=0.85, label="r=147mm EE")
        ax.set_aspect("equal")
        ax.set_xlim(-158, 158)
        ax.set_ylim(-158, 158)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")

    fig.subplots_adjust(right=0.90, wspace=0.25, top=0.84, bottom=0.12)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.018, 0.68])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label("Wafer Top Temperature [°C]", fontsize=10)
    fig.suptitle(
        "Week 2 3D Closed-Loop Safe Inverse Recovery Showcase (Al2O3 10-Zone ESC, Medium 3D FEM)",
        fontsize=12.5,
        fontweight="bold",
        y=0.97,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    refinement = "medium"
    geom = ESC3DGeometrySpec()
    spec = RecoverySpec(T_target_C=40.0)
    flow_params = HeFlowModelParams()
    opts = SolveOptions(T_chiller_C=20.0, T_wall_C=60.0)

    print(f"=== Week 2 3D Closed-Loop Evaluation (refinement={refinement}) ===")
    fem_solver = ESC3DFEMSolver(refinement=refinement, geom=geom)
    fvm_solver = Coupled3DWaferFVMSolver(refinement=refinement, ceramic_type="Al2O3")
    sensor_xy, _, _ = build_sensarray_65_coords()
    W_sensor = build_polar_sensor_interp_matrix(fvm_solver, sensor_xy)

    u_0_default = np.array(list(BASELINE_Q_HEATERS) + [BASELINE_P_IN, BASELINE_P_OUT], dtype=np.float64)

    # ------------------------------------------------------------------
    # PART A: Calibration Set (8 paired nominal / in-model runs)
    # ------------------------------------------------------------------
    cal_cases = [
        EvalCase("cal_01_nominal", "in_model", make_cond()),
        EvalCase("cal_02_peak_center", "in_model", make_cond(total_plasma_power_W=520.0, plasma_peaking_factor=0.40)),
        EvalCase("cal_03_peak_edge", "in_model", make_cond(total_plasma_power_W=480.0, plasma_peaking_factor=-0.05)),
        EvalCase("cal_04_leak_mild_0", "in_model", make_cond(beta_leak=0.75, leak_theta_center_deg=20.0)),
        EvalCase("cal_05_leak_mod_110", "in_model", make_cond(beta_leak=0.55, leak_theta_center_deg=110.0)),
        EvalCase("cal_06_leak_mod_200", "in_model", make_cond(beta_leak=0.45, leak_theta_center_deg=200.0)),
        EvalCase("cal_07_leak_sev_280", "in_model", make_cond(beta_leak=0.30, leak_theta_center_deg=280.0)),
        EvalCase("cal_08_leak_mix_340", "in_model", make_cond(total_plasma_power_W=535.0, plasma_peaking_factor=0.35, beta_leak=0.40, leak_theta_center_deg=340.0)),
    ]

    rng_cal = np.random.default_rng(101)
    cal_S0_vals: list[float] = []
    cal_Sprobe_vals: list[float] = []
    cal_records: list[dict] = []

    print("\n--- Part A: Calibration Set (8 paired in-model runs) ---")
    for c in cal_cases:
        delta_cal = float(rng_cal.uniform(-flow_params.delta_cal_halfwidth, flow_params.delta_cal_halfwidth))
        fem_0 = fem_solver.solve(c.cond)
        obs_0 = generate_synthetic_observation(
            fem_0, c.cond, sensor_xy, rng_cal, delta_cal=delta_cal, flow_params=flow_params
        )

        def make_probe_fn(cond_base: ESC3DOperatingCondition, d_cal: float, rng_local: np.random.Generator):
            def _probe(p_out_probe: float):
                cond_p = replace(cond_base, p_he_out_torr=float(p_out_probe))
                fem_p = fem_solver.solve(cond_p)
                return generate_synthetic_observation(
                    fem_p, cond_p, sensor_xy, rng_local, delta_cal=d_cal, flow_params=flow_params
                )
            return _probe

        s1 = run_stage1_diagnosis(
            obs_0=obs_0,
            probe_obs_fn=make_probe_fn(c.cond, delta_cal, rng_cal),
            solver=fvm_solver,
            sensor_xy=sensor_xy,
            W_sensor=W_sensor,
            opts=opts,
            flow_params=flow_params,
            S_0_thresh=None,
            S_probe_thresh=None,
        )
        cal_S0_vals.append(s1.S_0)
        cal_Sprobe_vals.append(float(s1.S_probe))
        cal_records.append({
            "name": c.name,
            "S_0": s1.S_0,
            "S_probe": s1.S_probe,
            "d_hat": s1.d_hat,
            "delta_cal": delta_cal,
        })
        print(
            f"  {c.name:20s} | S_0={s1.S_0:6.1f} | S_probe={s1.S_probe:6.1f} | "
            f"beta_hat={s1.d_hat['beta_leak']:.3f} (true {c.cond.beta_leak:.2f}) | "
            f"th_hat={s1.d_hat['theta_leak_deg']:5.1f}°"
        )

    # Lock thresholds strictly from Calibration Set (chi2 99.5% floor + empirical 1.35x safety factor)
    chi2_floor = float(chi2.ppf(0.995, df=66))
    S_0_thresh = max(max(cal_S0_vals) * 1.35, chi2_floor)
    S_probe_thresh = max(max(cal_Sprobe_vals) * 1.35, chi2_floor)
    print(
        f"\nLocked Thresholds from Calibration Set: "
        f"S_0_thresh = {S_0_thresh:.2f} (max cal {max(cal_S0_vals):.1f}), "
        f"S_probe_thresh = {S_probe_thresh:.2f} (max cal {max(cal_Sprobe_vals):.1f})"
    )

    # ------------------------------------------------------------------
    # PART B: Independent Test Set (15 independent cases)
    # ------------------------------------------------------------------
    test_cases = [
        # 8 Recoverable In-Model Cases (nominal, plasma shifts, mild-to-moderate localized edge leaks)
        EvalCase("test_01_nominal", "in_model", make_cond()),
        EvalCase("test_02_peak_mod", "in_model", make_cond(total_plasma_power_W=515.0, plasma_peaking_factor=0.30)),
        EvalCase("test_03_peak_edge", "in_model", make_cond(total_plasma_power_W=470.0, plasma_peaking_factor=-0.10)),
        EvalCase("test_04_power_high", "in_model", make_cond(total_plasma_power_W=550.0, plasma_peaking_factor=0.15)),
        EvalCase("test_05_leak_mild_45", "in_model", make_cond(beta_leak=0.80, leak_theta_center_deg=45.0)),
        EvalCase("test_06_leak_mod_135", "in_model", make_cond(beta_leak=0.70, leak_theta_center_deg=135.0)),
        EvalCase("test_07_leak_mod_315", "in_model", make_cond(total_plasma_power_W=490.0, plasma_peaking_factor=0.15, beta_leak=0.72, leak_theta_center_deg=315.0)),
        EvalCase("test_08_leak_225_show", "in_model", make_cond(total_plasma_power_W=500.0, plasma_peaking_factor=0.25, beta_leak=0.70, leak_theta_center_deg=225.0), is_showcase=True),
        # 1 Physically Unrecoverable Severe Edge Leak (Stage 1 passes, Stage 2 safely Holds via rho_bar > 0)
        EvalCase("test_09_leak_sev_hold", "in_model_unrec", make_cond(beta_leak=0.25, leak_theta_center_deg=225.0)),
        # 2 Active H_block = 1 Cases
        EvalCase("test_10_hblock_heflow", "h_block", make_cond(p_he_out_torr=16.5, beta_leak=0.08, leak_theta_center_deg=225.0)),
        EvalCase("test_11_hblock_readback", "h_block", make_cond(beta_leak=0.60, leak_theta_center_deg=180.0), readback_ok=False),
        # 4 OOD Hidden-Fault Cases
        EvalCase("test_12_ood_hotspot_225", "ood", make_cond(plasma_hotspot_amp=0.45, plasma_hotspot_r_m=0.135, plasma_hotspot_theta_deg=225.0)),
        EvalCase("test_13_ood_hotspot_60", "ood", make_cond(plasma_hotspot_amp=0.40, plasma_hotspot_r_m=0.135, plasma_hotspot_theta_deg=60.0)),
        EvalCase("test_14_ood_shift_rleak", "ood", make_cond(beta_leak=0.30, r_leak_m=0.128, leak_theta_center_deg=200.0)),
        EvalCase("test_15_ood_warpage", "ood", make_cond(d_gap_warpage_m=25e-6)),
    ]

    rng_test = np.random.default_rng(2026)
    test_records: list[dict] = []
    showcase_bundle: dict = {}

    print("\n--- Part B: Independent Test Set (15 independent runs) ---")
    for c in test_cases:
        delta_cal = float(rng_test.uniform(-flow_params.delta_cal_halfwidth, flow_params.delta_cal_halfwidth))
        fem_0 = fem_solver.solve(c.cond)
        obs_0 = generate_synthetic_observation(
            fem_0,
            c.cond,
            sensor_xy,
            rng_test,
            delta_cal=delta_cal,
            flow_params=flow_params,
            pressure_tracking_ok=c.pressure_tracking_ok,
            readback_ok=c.readback_ok,
        )

        def make_probe_fn_t(cond_base: ESC3DOperatingCondition, d_cal: float, rng_local: np.random.Generator):
            def _probe(p_out_probe: float):
                cond_p = replace(cond_base, p_he_out_torr=float(p_out_probe))
                fem_p = fem_solver.solve(cond_p)
                return generate_synthetic_observation(
                    fem_p,
                    cond_p,
                    sensor_xy,
                    rng_local,
                    delta_cal=d_cal,
                    flow_params=flow_params,
                    pressure_tracking_ok=c.pressure_tracking_ok,
                    readback_ok=c.readback_ok,
                )
            return _probe

        s1 = run_stage1_diagnosis(
            obs_0=obs_0,
            probe_obs_fn=make_probe_fn_t(c.cond, delta_cal, rng_test),
            solver=fvm_solver,
            sensor_xy=sensor_xy,
            W_sensor=W_sensor,
            opts=opts,
            flow_params=flow_params,
            S_0_thresh=S_0_thresh,
            S_probe_thresh=S_probe_thresh,
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

        # FREEZE A_pre and u* BEFORE running independent 3D FEM with TRUE HIDDEN FAULT!
        frozen_A_pre = int(s2.A_pre)
        frozen_u_star = s2.u_star.copy()

        # Post-hoc independent 3D FEM evaluation with u* and TRUE HIDDEN FAULT
        cond_recov = replace(
            c.cond,
            heater_powers_W=tuple(float(x) for x in frozen_u_star[:10]),
            p_he_in_torr=float(frozen_u_star[10]),
            p_he_out_torr=float(frozen_u_star[11]),
        )
        fem_recov = fem_solver.solve(cond_recov)
        posthoc = grade_posthoc_fem(fem_recov, cond_recov.heater_powers_W, spec, geom)

        if c.is_showcase:
            showcase_bundle = {
                "fem_faulty": fem_0,
                "T_pred_2d": s2.T_wafer_pred_2d_C,
                "fem_recovered": fem_recov,
                "d_hat": s1.d_hat,
                "before": {
                    "T_mean_eff_C": float(fem_0["T_mean_eff_C"]),
                    "dT_max_min_eff_C": float(fem_0["dT_max_min_eff_C"]),
                    "T_3sigma_eff_C": float(fem_0["T_3sigma_eff_C"]),
                },
                "after": {
                    "T_mean_eff_C": float(fem_recov["T_mean_eff_C"]),
                    "dT_max_min_eff_C": float(fem_recov["dT_max_min_eff_C"]),
                    "T_3sigma_eff_C": float(fem_recov["T_3sigma_eff_C"]),
                },
            }

        rec = {
            "name": c.name,
            "category": c.category,
            "H_block": s1.H_block,
            "S_0": None if math.isnan(s1.S_0) else round(s1.S_0, 2),
            "S_probe": None if s1.S_probe is None else round(s1.S_probe, 2),
            "stage1_pass": s1.stage1_pass,
            "leak_detected": s1.leak_detected,
            "pm_review": s1.pm_review,
            "d_hat": {k: round(v, 4) for k, v in s1.d_hat.items()},
            "A_pre": frozen_A_pre,
            "decision_label": s2.decision_label,
            "hold_reason": s2.hold_reason,
            "rho_bar": None if math.isnan(s2.rho_bar) else round(s2.rho_bar, 4),
            "p_out_upper_bound_torr": s2.p_out_upper_bound_torr,
            "u_star": [round(float(x), 3) for x in frozen_u_star],
            "before_fem": {
                "T_mean_eff_C": round(float(fem_0["T_mean_eff_C"]), 3),
                "dT_max_min_eff_C": round(float(fem_0["dT_max_min_eff_C"]), 3),
                "T_3sigma_eff_C": round(float(fem_0["T_3sigma_eff_C"]), 3),
            },
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
        test_records.append(rec)

        s0_str = "  N/A " if rec["S_0"] is None else f"{rec['S_0']:6.1f}"
        sp_str = "  N/A " if rec["S_probe"] is None else f"{rec['S_probe']:6.1f}"
        print(
            f"  {c.name:24s} [{c.category:8s}] | H={s1.H_block} | S0={s0_str} | Sp={sp_str} | "
            f"A_pre={frozen_A_pre} | V={posthoc['V']} | "
            f"dT: {fem_0['dT_max_min_eff_C']:.2f}->{posthoc['dT_max_min_eff_C']:.2f}°C | "
            f"3s: {fem_0['T_3sigma_eff_C']:.2f}->{posthoc['T_3sigma_eff_C']:.2f}°C | {s2.decision_label} ({s2.hold_reason})"
        )

    # ------------------------------------------------------------------
    # Summary Statistics & Clopper-Pearson Upper Bound
    # ------------------------------------------------------------------
    n_total = len(test_records)
    n_inmodel = sum(1 for r in test_records if r["category"] == "in_model")
    n_hblock = sum(1 for r in test_records if r["category"] == "h_block")
    n_ood = sum(1 for r in test_records if r["category"] == "ood")

    n_A = sum(r["A_pre"] for r in test_records)
    n_A_inmodel = sum(r["A_pre"] for r in test_records if r["category"] == "in_model")
    n_hblock_caught = sum(1 for r in test_records if r["category"] == "h_block" and r["H_block"] == 1 and r["A_pre"] == 0)
    n_ood_blocked = sum(1 for r in test_records if r["category"] == "ood" and r["A_pre"] == 0)
    n_viol_given_A = sum(1 for r in test_records if r["A_pre"] == 1 and r["posthoc_fem"]["V"] == 1)

    p_viol_given_A = (n_viol_given_A / n_A) if n_A > 0 else None
    cp_upper_95 = clopper_pearson_upper_95(n_viol_given_A, n_A)

    summary = {
        "refinement": refinement,
        "T_target_C": spec.T_target_C,
        "S_0_thresh": round(S_0_thresh, 3),
        "S_probe_thresh": round(S_probe_thresh, 3),
        "n_calibration": len(cal_cases),
        "n_test_total": n_total,
        "n_in_model": n_inmodel,
        "n_h_block": n_hblock,
        "n_ood": n_ood,
        "n_A_pre": n_A,
        "approval_rate_overall": round(n_A / n_total, 4),
        "approval_rate_in_model": round(n_A_inmodel / n_inmodel, 4),
        "h_block_catch_rate": round(n_hblock_caught / n_hblock, 4),
        "ood_block_rate": round(n_ood_blocked / n_ood, 4),
        "n_V1_given_A1": n_viol_given_A,
        "P_V1_given_A1": None if p_viol_given_A is None else round(p_viol_given_A, 4),
        "clopper_pearson_95_upper": None if cp_upper_95 is None else round(cp_upper_95, 4),
        "showcase_before_after": {
            "before": showcase_bundle.get("before", {}),
            "after": showcase_bundle.get("after", {}),
        },
        "total_runtime_s": round(time.perf_counter() - t_start, 2),
        "overall_pass": bool(
            (n_viol_given_A == 0)
            and (n_A_inmodel >= n_inmodel - 1)
            and (n_hblock_caught == n_hblock)
            and (n_ood_blocked == n_ood)
            and (n_A >= 6)
        ),
    }

    out_json = PROJECT_ROOT / "results" / "benchmarks" / "week2_3d_closed_loop_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "calibration": cal_records, "test_cases": test_records},
            f,
            indent=2,
        )

    out_png = PROJECT_ROOT / "results" / "benchmarks" / "week2_3d_recovery_showcase.png"
    if showcase_bundle and showcase_bundle.get("T_pred_2d") is not None:
        plot_3panel_showcase(
            fem_faulty=showcase_bundle["fem_faulty"],
            T_pred_2d=showcase_bundle["T_pred_2d"],
            solver=fvm_solver,
            fem_recovered=showcase_bundle["fem_recovered"],
            d_hat=showcase_bundle["d_hat"],
            out_path=out_png,
        )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {out_json}")
    print(f"Saved PNG : {out_png}")


if __name__ == "__main__":
    main()
