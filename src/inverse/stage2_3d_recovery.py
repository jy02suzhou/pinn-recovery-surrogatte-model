"""Stage 2 (3D): 10-Zone Azimuthal Safe Inverse Recovery & Robust Pre-Approval (A_pre).

Implements v1.7 Stage 2 for the 300 mm 3D Multi-Zone ESC + Wafer system:
  - 12D control vector u = (Q_1..Q_10, P_in, P_out) in hardware bounds U_HW:
      Q_1, Q_2   in [0, 400] W
      Q_3..Q_10  in [0, 200] W
      P_in       in [3.0, 20.0] Torr
      P_out      in [3.0, P_out_0] Torr if leak_detected else [3.0, 20.0] Torr
  - Product wafer quality specs on effective area r <= 147 mm with conservative
    FVM-to-3D-FEM-top margins (delta_T, delta_grad):
      1. |T_mean_eff - T_target| + delta_T       <= 0.5 C
      2. (max T_eff - min T_eff) + 2*delta_T     <= 1.5 C
      3. 3*std_A(T_eff) + 3*delta_T              <= 1.0 C
      4. max_grad_wafer_eff + delta_grad         <= 0.3 C/mm
  - Equipment protection constraints:
      5. max_grad_esc_full (r <= 155 mm)         <= 1.5 C/mm
      6. Independent bottom chiller Q_cool       <= 1800.0 W
      7. Adjacent heater flux diff |q''_i-q''_j| <= 2.5 W/cm^2
  - Optimization via SciPy SLSQP with exact PyTorch Adjoint gradients (smooth LogSumExp).
  - Robust upper bound rho_bar(u*) over finite uncertainty scenarios around d_hat plus
    local Adjoint gradient covering bound L_hat * h.
  - Freezes A_pre = 1{H_block==0, stage1_pass, readback_ok, rho_bar <= 0} and u*
    strictly BEFORE any independent 3D FEM evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import minimize
import torch

from src.fem3d.build_esc_3d_mesh import ESC3DGeometrySpec
from src.inverse.stage1_3d_diagnosis import Stage1Result
from src.solvers.coupled_3d_wafer_fvm import Coupled3DWaferFVMSolver, SolveOptions


# Adjacent heater zone index pairs (0-indexed: 0=Z1, 1=Z2, 2..9=Z3..Z10 edge sectors)
ADJACENT_HEATER_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 2),
)


@dataclass(frozen=True)
class RecoverySpec:
    """Product quality & equipment safety specifications for Stage 2."""
    T_target_C: float = 40.0
    r_eff_m: float = 0.147

    # Conservative FVM-to-3D-FEM-top margins (1.5x medium MAE=0.041C; ONLY for pre-approval, NEVER post-hoc V)
    delta_T_C: float = 0.06
    delta_grad_C_per_mm: float = 0.04

    # Product quality limits on r <= 147 mm
    limit_mean_err_C: float = 0.50
    limit_dT_span_C: float = 1.50
    limit_3sigma_C: float = 1.00
    limit_wafer_grad_C_per_mm: float = 0.30

    # Equipment protection limits
    limit_esc_grad_C_per_mm: float = 1.50
    limit_Q_cool_W: float = 1800.0
    limit_qdiff_W_cm2: float = 2.50

    # Hardware bounds U_HW
    Q_inner_max_W: float = 400.0
    Q_edge_max_W: float = 200.0
    P_min_torr: float = 3.0
    P_max_torr: float = 20.0


@dataclass
class Stage2Result:
    """Frozen pre-approval decision A_pre and optimal 12D recipe u*."""
    A_pre: int
    decision_label: str
    u_star: np.ndarray                     # Shape (12,): [Q_1..Q_10, P_in, P_out]
    Q_heaters_star_W: tuple[float, ...]    # Length 10
    p_in_star_torr: float
    p_out_star_torr: float
    p_out_upper_bound_torr: float
    rho_nominal: float
    rho_bar: float
    constraints_nominal: dict[str, float]
    constraints_worst: dict[str, float]
    predicted_metrics: dict[str, float]
    T_wafer_pred_2d_C: np.ndarray | None
    opt_time_s: float
    hold_reason: str


def _heater_areas_cm2_torch(geom: ESC3DGeometrySpec, device: torch.device) -> torch.Tensor:
    areas_cm2 = [a * 1e4 for a in geom.heater_zone_areas]
    return torch.as_tensor(areas_cm2, dtype=torch.float64, device=device)


def compute_max_qdiff_torch(
    Q_heaters: torch.Tensor,
    heater_areas_cm2: torch.Tensor,
    smooth_tau: float | None = None,
) -> torch.Tensor:
    """Compute max adjacent heater heat-flux density difference |q''_i - q''_j| [W/cm^2]."""
    q_flux = Q_heaters / heater_areas_cm2
    diffs = []
    for i, j in ADJACENT_HEATER_PAIRS:
        d = q_flux[i] - q_flux[j]
        diffs.append(torch.sqrt(d * d + 1e-10) if smooth_tau is not None else torch.abs(d))
    diff_vec = torch.stack(diffs)
    if smooth_tau is None:
        return torch.max(diff_vec)
    return smooth_tau * torch.logsumexp(diff_vec / smooth_tau, dim=0)


def _eval_polar_grad_mag_C_per_mm(
    T_2d: torch.Tensor,
    r_centers: torch.Tensor,
    dtheta: float,
    r_max_m: float | None = None,
) -> torch.Tensor:
    """Compute polar gradient magnitude |grad T| [C/mm] at radial interior faces."""
    dr_c = (r_centers[1:] - r_centers[:-1])[:, None]              # (n_r - 1, 1) [m]
    dT_dr = (T_2d[1:, :] - T_2d[:-1, :]) / dr_c                   # [C/m]

    T_next = torch.roll(T_2d, shifts=-1, dims=1)
    T_prev = torch.roll(T_2d, shifts=1, dims=1)
    dT_dth_cell = (T_next - T_prev) / (2.0 * r_centers[:, None] * dtheta)
    dT_dth_face = 0.5 * (dT_dth_cell[:-1, :] + dT_dth_cell[1:, :])

    grad_mag_C_per_mm = torch.sqrt(dT_dr.pow(2) + dT_dth_face.pow(2) + 1e-12) * 1e-3
    if r_max_m is not None:
        r_mid = 0.5 * (r_centers[:-1] + r_centers[1:])
        mask = r_mid <= r_max_m + 1e-9
        grad_mag_C_per_mm = grad_mag_C_per_mm[mask, :]
    return grad_mag_C_per_mm.reshape(-1)


def eval_constraints_torch(
    solver: Coupled3DWaferFVMSolver,
    Q_heaters: torch.Tensor,
    p_in: torch.Tensor,
    p_out: torch.Tensor,
    Q_plasma: torch.Tensor | float,
    alpha_peak: torch.Tensor | float,
    beta_leak: torch.Tensor | float,
    theta_leak_deg: torch.Tensor | float,
    spec: RecoverySpec,
    heater_areas_cm2: torch.Tensor,
    opts: SolveOptions = SolveOptions(),
    smooth: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate all 7 product + equipment constraints g_i(u, d) <= 0 and normalized g_i / s_i."""
    out = solver.solve(
        Q_heaters=Q_heaters,
        p_in=p_in,
        p_out=p_out,
        beta_leak=beta_leak,
        Q_plasma=Q_plasma,
        alpha_peak=alpha_peak,
        opts=opts,
        theta_leak_deg=theta_leak_deg,
    )
    T_2d = out["T_wafer_2d_C"]                                     # (n_r, n_theta)
    r_eff_mask = solver.r_centers <= spec.r_eff_m + 1e-9
    T_eff = T_2d[r_eff_mask, :].reshape(-1)
    A_eff = solver.cell_areas[r_eff_mask, :].reshape(-1)
    w_eff = A_eff / torch.sum(A_eff)

    T_mean = torch.sum(w_eff * T_eff)
    var_A = torch.sum(w_eff * (T_eff - T_mean).pow(2))
    std_A = torch.sqrt(var_A + (1e-8 if smooth else 0.0))

    if smooth:
        tau_T = 0.05
        T_max = tau_T * torch.logsumexp(T_eff / tau_T, dim=0)
        T_min = -tau_T * torch.logsumexp(-T_eff / tau_T, dim=0)
        mean_err = torch.sqrt((T_mean - spec.T_target_C).pow(2) + 1e-6)
    else:
        T_max = torch.max(T_eff)
        T_min = torch.min(T_eff)
        mean_err = torch.abs(T_mean - spec.T_target_C)

    dT_span = T_max - T_min

    grad_w_vec = _eval_polar_grad_mag_C_per_mm(
        T_2d, solver.r_centers, solver.dtheta, r_max_m=spec.r_eff_m
    )
    T_esc_full_2d = out["T_esc_full_C"].reshape(len(solver.r_esc_centers), solver.n_theta)
    grad_esc_vec = _eval_polar_grad_mag_C_per_mm(
        T_esc_full_2d, solver.r_esc_centers, solver.dtheta, r_max_m=None
    )

    if smooth:
        tau_gw = 0.015
        tau_ge = 0.05
        max_grad_w = tau_gw * torch.logsumexp(grad_w_vec / tau_gw, dim=0)
        max_grad_esc = tau_ge * torch.logsumexp(grad_esc_vec / tau_ge, dim=0)
        max_qdiff = compute_max_qdiff_torch(Q_heaters, heater_areas_cm2, smooth_tau=0.05)
    else:
        max_grad_w = torch.max(grad_w_vec)
        max_grad_esc = torch.max(grad_esc_vec)
        max_qdiff = compute_max_qdiff_torch(Q_heaters, heater_areas_cm2, smooth_tau=None)

    Q_cool = out["Q_cool_W"]

    # Raw constraint values g_i <= 0 (with conservative pre-approval margins on wafer specs)
    g_raw = {
        "mean_err": mean_err + spec.delta_T_C - spec.limit_mean_err_C,
        "dT_span": dT_span + 2.0 * spec.delta_T_C - spec.limit_dT_span_C,
        "T_3sigma": 3.0 * std_A + 3.0 * spec.delta_T_C - spec.limit_3sigma_C,
        "wafer_grad": max_grad_w + spec.delta_grad_C_per_mm - spec.limit_wafer_grad_C_per_mm,
        "esc_grad": max_grad_esc - spec.limit_esc_grad_C_per_mm,
        "Q_cool": Q_cool - spec.limit_Q_cool_W,
        "qdiff": max_qdiff - spec.limit_qdiff_W_cm2,
    }
    scales = {
        "mean_err": spec.limit_mean_err_C,
        "dT_span": spec.limit_dT_span_C,
        "T_3sigma": spec.limit_3sigma_C,
        "wafer_grad": spec.limit_wafer_grad_C_per_mm,
        "esc_grad": spec.limit_esc_grad_C_per_mm,
        "Q_cool": spec.limit_Q_cool_W,
        "qdiff": spec.limit_qdiff_W_cm2,
    }
    g_norm = {k: g_raw[k] / scales[k] for k in g_raw}

    metrics = {
        "T_mean_eff_C": T_mean,
        "dT_max_min_eff_C": dT_span,
        "T_3sigma_eff_C": 3.0 * std_A,
        "var_A_eff": var_A,
        "max_grad_wafer_eff_C_per_mm": max_grad_w,
        "max_grad_esc_C_per_mm": max_grad_esc,
        "Q_cool_W": Q_cool,
        "max_qdiff_W_cm2": max_qdiff,
        "T_wafer_2d_C": T_2d,
    }
    return g_raw, g_norm, metrics


def run_stage2_recovery(
    stage1: Stage1Result,
    u_0: np.ndarray,
    solver: Coupled3DWaferFVMSolver,
    spec: RecoverySpec = RecoverySpec(),
    geom: ESC3DGeometrySpec = ESC3DGeometrySpec(),
    opts: SolveOptions = SolveOptions(),
) -> Stage2Result:
    """Run Stage 2 10-zone safe inverse recovery and freeze A_pre & u*."""
    t0 = time.perf_counter()
    u_0 = np.asarray(u_0, dtype=np.float64).copy()
    p_out_0 = float(u_0[11])
    p_out_hi = p_out_0 if stage1.leak_detected else spec.P_max_torr

    # If H_block == 1 or Stage 1 failed, immediately freeze Hold (A_pre = 0)
    if stage1.H_block == 1 or not stage1.stage1_pass or not stage1.readback_ok:
        label = "Hold / PM Review" if stage1.pm_review else "Unverified / Hold"
        return Stage2Result(
            A_pre=0,
            decision_label=label,
            u_star=u_0.copy(),
            Q_heaters_star_W=tuple(float(x) for x in u_0[:10]),
            p_in_star_torr=float(u_0[10]),
            p_out_star_torr=float(u_0[11]),
            p_out_upper_bound_torr=p_out_hi,
            rho_nominal=float("nan"),
            rho_bar=float("nan"),
            constraints_nominal={},
            constraints_worst={},
            predicted_metrics={},
            T_wafer_pred_2d_C=None,
            opt_time_s=time.perf_counter() - t0,
            hold_reason=stage1.hold_reason,
        )

    heater_areas_cm2 = _heater_areas_cm2_torch(geom, solver.device)
    d_hat = stage1.d_hat
    Q_pl_hat = float(d_hat["Q_plasma"])
    alpha_hat = float(d_hat["alpha_peak"])
    beta_hat = float(d_hat["beta_leak"])
    theta_hat = float(d_hat["theta_leak_deg"])

    # Hardware bounds U_HW (with leak interlock freezing P_out <= P_out_0)
    lo_phys = np.array([0.0] * 10 + [spec.P_min_torr, spec.P_min_torr], dtype=np.float64)
    hi_phys = np.array(
        [spec.Q_inner_max_W, spec.Q_inner_max_W] + [spec.Q_edge_max_W] * 8 + [spec.P_max_torr, p_out_hi],
        dtype=np.float64,
    )
    u_scales = np.array([200.0, 200.0] + [100.0] * 8 + [10.0, 10.0], dtype=np.float64)
    bounds_z = [(lo / s, hi / s) for lo, hi, s in zip(lo_phys, hi_phys, u_scales)]
    u_0_clipped = np.clip(u_0, lo_phys, hi_phys)

    def _eval_obj_and_grad(z: np.ndarray, focus_feasibility: bool = False) -> tuple[float, np.ndarray]:
        u = np.clip(z * u_scales, lo_phys, hi_phys)
        u_t = torch.tensor(u, dtype=torch.float64, device=solver.device, requires_grad=True)
        _, g_norm, m = eval_constraints_torch(
            solver=solver,
            Q_heaters=u_t[:10],
            p_in=u_t[10],
            p_out=u_t[11],
            Q_plasma=Q_pl_hat,
            alpha_peak=alpha_hat,
            beta_leak=beta_hat,
            theta_leak_deg=theta_hat,
            spec=spec,
            heater_areas_cm2=heater_areas_cm2,
            opts=opts,
            smooth=True,
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
            # Smooth barrier on any positive constraint + uniformity objective
            viol_pen = torch.sum(torch.nn.functional.softplus((g_stack + 0.05) * 15.0).pow(2)) / 225.0
            loss = (
                1.5 * rho_lse
                + 8.0 * viol_pen
                + 4.0 * m["var_A_eff"]
                + 2.0 * (m["T_mean_eff_C"] - spec.T_target_C).pow(2)
                + 0.015 * reg
            )

        loss.backward()
        grad_z = u_t.grad.detach().cpu().numpy() * u_scales
        return float(loss.item()), grad_z

    # Stage 2a: Multi-start SLSQP (u_0 + balanced P_in/P_out seed)
    u_seed_balanced = np.clip(
        np.array([40.0, 120.0] + [75.0] * 8 + [12.0, min(12.0, p_out_hi)], dtype=np.float64),
        lo_phys,
        hi_phys,
    )

    def _eval_exact_at(u_vec: np.ndarray, d_dict: dict[str, float]):
        with torch.no_grad():
            u_t = torch.as_tensor(u_vec, dtype=torch.float64, device=solver.device)
            g_raw, g_norm, m = eval_constraints_torch(
                solver=solver,
                Q_heaters=u_t[:10],
                p_in=u_t[10],
                p_out=u_t[11],
                Q_plasma=d_dict["Q_plasma"],
                alpha_peak=d_dict["alpha_peak"],
                beta_leak=d_dict["beta_leak"],
                theta_leak_deg=d_dict["theta_leak_deg"],
                spec=spec,
                heater_areas_cm2=heater_areas_cm2,
                opts=opts,
                smooth=False,
            )
            g_norm_f = {k: float(v.item()) for k, v in g_norm.items()}
            rho_val = max(g_norm_f.values())
            return rho_val, g_norm_f, m

    best_u = u_0_clipped.copy()
    best_rho = float("inf")
    best_g_nom: dict[str, float] = {}
    best_metrics = None

    for u_init in (u_seed_balanced, u_0_clipped):
        res_a = minimize(
            lambda z: _eval_obj_and_grad(z, focus_feasibility=False),
            u_init / u_scales,
            method="SLSQP",
            jac=True,
            bounds=bounds_z,
            options={"maxiter": 60, "ftol": 1e-8},
        )
        u_cand = np.clip(res_a.x * u_scales, lo_phys, hi_phys)
        rho_cand, g_cand, m_cand = _eval_exact_at(u_cand, d_hat)
        if rho_cand < best_rho:
            best_u, best_rho, best_g_nom, best_metrics = u_cand, rho_cand, g_cand, m_cand
        # Early exit if already comfortably feasible
        if best_rho < -0.15:
            break

    u_star, rho_nom, g_nom_dict, metrics_nom = best_u, best_rho, best_g_nom, best_metrics
    if rho_nom > -0.05:
        res_b = minimize(
            lambda z: _eval_obj_and_grad(z, focus_feasibility=True),
            u_star / u_scales,
            method="SLSQP",
            jac=True,
            bounds=bounds_z,
            options={"maxiter": 45, "ftol": 1e-8},
        )
        u_cand = np.clip(res_b.x * u_scales, lo_phys, hi_phys)
        rho_cand, g_cand_dict, metrics_cand = _eval_exact_at(u_cand, d_hat)
        if rho_cand < rho_nom:
            u_star, rho_nom, g_nom_dict, metrics_nom = u_cand, rho_cand, g_cand_dict, metrics_cand

    # ------------------------------------------------------------------
    # Robust Upper Bound rho_bar(u*) over finite uncertainty scenarios
    # around d_hat + local Adjoint gradient covering bound L_hat * h
    # ------------------------------------------------------------------
    sig = stage1.sigma_d
    # 1.5-sigma uncertainty box around d_hat along active identifiable axes
    active_axes: list[tuple[str, float, float, float]] = [
        ("Q_plasma", min(1.5 * sig["Q_plasma"], 15.0), 250.0, 800.0),
        ("alpha_peak", min(1.5 * sig["alpha_peak"], 0.08), -0.6, 0.9),
    ]
    if stage1.leak_detected or beta_hat < 0.96:
        active_axes.append(("beta_leak", min(1.5 * sig["beta_leak"], 0.05), 0.05, 1.0))
        active_axes.append(("theta_leak_deg", min(1.5 * sig["theta_leak_deg"], 10.0), -360.0, 720.0))

    scenarios: list[dict[str, float]] = [dict(d_hat)]
    for name, half_w, lo_d, hi_d in active_axes:
        for sign in (-1.0, 1.0):
            sc = dict(d_hat)
            val = d_hat[name] + sign * half_w
            if name == "theta_leak_deg":
                sc[name] = float(val % 360.0)
            else:
                sc[name] = float(np.clip(val, lo_d, hi_d))
            scenarios.append(sc)

    # Worst-case normalized constraint across finite scenario grid
    worst_g_norm = dict(g_nom_dict)
    for sc in scenarios[1:]:
        _, g_sc, _ = _eval_exact_at(u_star, sc)
        for k, v in g_sc.items():
            if v > worst_g_norm[k]:
                worst_g_norm[k] = v

    # Local Adjoint gradient covering bound L_hat_i * h at d_hat for PDE constraints
    u_star_t = torch.as_tensor(u_star, dtype=torch.float64, device=solver.device)
    Q_pl_t = torch.tensor(Q_pl_hat, dtype=torch.float64, device=solver.device, requires_grad=True)
    alpha_t = torch.tensor(alpha_hat, dtype=torch.float64, device=solver.device, requires_grad=True)
    beta_t = torch.tensor(beta_hat, dtype=torch.float64, device=solver.device, requires_grad=True)
    theta_t = torch.tensor(theta_hat, dtype=torch.float64, device=solver.device, requires_grad=True)
    d_tensors = {
        "Q_plasma": Q_pl_t,
        "alpha_peak": alpha_t,
        "beta_leak": beta_t,
        "theta_leak_deg": theta_t,
    }
    _, g_norm_diff, _ = eval_constraints_torch(
        solver=solver,
        Q_heaters=u_star_t[:10],
        p_in=u_star_t[10],
        p_out=u_star_t[11],
        Q_plasma=Q_pl_t,
        alpha_peak=alpha_t,
        beta_leak=beta_t,
        theta_leak_deg=theta_t,
        spec=spec,
        heater_areas_cm2=heater_areas_cm2,
        opts=opts,
        smooth=True,
    )

    active_names = [a[0] for a in active_axes]
    active_half_w = {a[0]: a[1] for a in active_axes}
    active_t_list = [d_tensors[nm] for nm in active_names]
    h_frac = 0.25  # Covering half-step factor for uncovered box corners

    constraints_with_cov: dict[str, float] = {}
    for k, g_val in g_norm_diff.items():
        if k == "qdiff":
            # Purely analytical in Q_heaters, independent of d
            constraints_with_cov[k] = float(worst_g_norm[k])
            continue
        grads = torch.autograd.grad(
            g_val, active_t_list, retain_graph=True, allow_unused=True
        )
        cov_sq = 0.0
        for nm, g_d in zip(active_names, grads):
            if g_d is not None:
                cov_sq += (float(g_d.item()) * h_frac * active_half_w[nm]) ** 2
        constraints_with_cov[k] = float(worst_g_norm[k] + math.sqrt(cov_sq))

    rho_bar = max(constraints_with_cov.values())

    # Freeze A_pre strictly BEFORE any independent 3D FEM evaluation!
    A_pre = int((stage1.H_block == 0) and stage1.stage1_pass and stage1.readback_ok and (rho_bar <= 0.0))
    if A_pre == 1:
        decision_label = "Pre-Approved"
        hold_reason = "OK"
    else:
        decision_label = "Unverified / Hold"
        binding_k = max(constraints_with_cov, key=lambda k: constraints_with_cov[k])
        hold_reason = f"rho_bar={rho_bar:+.3f} > 0 (binding: {binding_k})"

    pred_metrics = {
        "T_mean_eff_C": float(metrics_nom["T_mean_eff_C"].item()),
        "dT_max_min_eff_C": float(metrics_nom["dT_max_min_eff_C"].item()),
        "T_3sigma_eff_C": float(metrics_nom["T_3sigma_eff_C"].item()),
        "max_grad_wafer_eff_C_per_mm": float(metrics_nom["max_grad_wafer_eff_C_per_mm"].item()),
        "max_grad_esc_C_per_mm": float(metrics_nom["max_grad_esc_C_per_mm"].item()),
        "Q_cool_W": float(metrics_nom["Q_cool_W"].item()),
        "max_qdiff_W_cm2": float(metrics_nom["max_qdiff_W_cm2"].item()),
    }

    return Stage2Result(
        A_pre=A_pre,
        decision_label=decision_label,
        u_star=u_star,
        Q_heaters_star_W=tuple(float(x) for x in u_star[:10]),
        p_in_star_torr=float(u_star[10]),
        p_out_star_torr=float(u_star[11]),
        p_out_upper_bound_torr=p_out_hi,
        rho_nominal=float(rho_nom),
        rho_bar=float(rho_bar),
        constraints_nominal=g_nom_dict,
        constraints_worst=constraints_with_cov,
        predicted_metrics=pred_metrics,
        T_wafer_pred_2d_C=metrics_nom["T_wafer_2d_C"].detach().cpu().numpy(),
        opt_time_s=time.perf_counter() - t0,
        hold_reason=hold_reason,
    )
