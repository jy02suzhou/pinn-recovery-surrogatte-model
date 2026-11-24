"""Stage 1 (3D): Active Equipment Inhibit (H_block), 65-Point SensArray Diagnosis & Pre-Refit Gate.

Implements v1.7 Stage 1 for the 300 mm 3D Multi-Zone ESC + Wafer system:
  1. 65-point 2D SensArray EtchTemp wafer coordinates (1 center + 5 concentric rings,
     with 32 sensors densely placed in the outer edge band r in [135, 147] mm).
  2. Synthetic equipment observation generator from 3D FEM top-surface temperatures
     (sigma_T = 0.2 C) + deliberately model-mismatched He leak flow:
       - Ground truth: m_true = m_base + kappa_nl * (1 - beta^2) * P_out^2 * (1 + delta_cal) + eps_m
       - Stage 1 proxy: m_hat  = m_base + kappa_lin * (1 - beta) * P_out
  3. Active Equipment Inhibit Gate H_block (checked FIRST before probe or optimization):
     blocks on OEM He Flow High alarm (>= 20 sccm), pressure tracking fault, or missing readback.
  4. Stage 1 estimation of d_hat = (Q_plasma, alpha_peak, beta_leak, theta_leak_deg) via
     fast 8-sector azimuthal coarse scan + continuous L-BFGS-B refinement with exact PyTorch
     Adjoint gradients from Coupled3DWaferFVMSolver.
  5. 2-Operating-Point Difference Pre-Refit Gate (S_probe): keeping d_hat FIXED (no refit!),
     evaluates a safe downward probe P_out_probe = max(3.0, P_out_0 - 3.0) Torr and grades
     the difference residual (Delta_y - Delta_f(d_hat)) to catch OOD hidden faults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import minimize
import torch

from src.fem3d.esc_3d_fem_solver import ESC3DOperatingCondition
from src.solvers.coupled_3d_wafer_fvm import Coupled3DWaferFVMSolver, SolveOptions


@dataclass(frozen=True)
class HeFlowModelParams:
    """Parameters for ground-truth nonlinear He leak flow vs Stage 1 linear proxy."""
    m_base_sccm: float = 0.80
    kappa_nl: float = 0.082              # Ground-truth nonlinear viscous coefficient
    kappa_lin: float = 1.15              # Deliberately mismatched linear proxy coefficient
    sigma_m_true_sccm: float = 0.15      # Measurement noise std [sccm]
    delta_cal_halfwidth: float = 0.20    # Uniform calibration drift U(-0.2, +0.2)
    oem_high_flow_alarm_sccm: float = 20.0
    normal_flow_upper_sccm: float = 1.80
    sigma_m_est_base_sccm: float = 1.20  # Conservative Stage 1 flow std [sccm]
    sigma_m_est_rel: float = 0.25        # Relative flow mismatch allowance in Stage 1


@dataclass(frozen=True)
class EquipmentObservation:
    """Equipment-available measurements at a single operating point."""
    T_sensors_C: np.ndarray              # Shape (65,)
    m_he_sccm: float
    Q_heaters_W: tuple[float, ...]       # Length 10
    p_in_torr: float
    p_out_torr: float
    pressure_tracking_ok: bool = True
    readback_ok: bool = True


@dataclass
class Stage1Result:
    """Structured output of Stage 1 diagnosis and Pre-Refit Gate."""
    H_block: int
    d_hat: dict[str, float]
    sigma_d: dict[str, float]
    S_0: float
    S_probe: float | None
    stage1_pass: bool
    leak_detected: bool
    hold_reason: str
    pm_review: bool
    readback_ok: bool
    S_0_T: float = 0.0
    S_0_m: float = 0.0
    S_probe_T: float | None = None
    S_probe_m: float | None = None
    m_he_0_sccm: float = 0.0
    m_he_probe_sccm: float | None = None
    P_out_probe_torr: float | None = None
    max_abs_res_0_C: float = 0.0
    max_abs_res_probe_C: float | None = None


def build_sensarray_65_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 65-point 2D SensArray EtchTemp sensor coordinates on the 300 mm wafer.

    Layout:
      - 1 center sensor at r = 0 mm
      - Ring 1:  8 sensors at r = 45 mm
      - Ring 2: 12 sensors at r = 88 mm
      - Ring 3: 12 sensors at r = 118 mm
      - Ring 4: 16 sensors at r = 138 mm (outer edge band)
      - Ring 5: 16 sensors at r = 146 mm (outer edge band, staggered by 11.25 deg)
      Total = 65 sensors, with 32 sensors in r in [135, 147] mm.

    Returns:
        (xy [65, 2] in meters, r [65] in meters, theta_deg [65] in [0, 360))
    """
    pts_xy = [(0.0, 0.0)]
    pts_r = [0.0]
    pts_th = [0.0]

    rings = [
        (0.045, 8, 0.0),
        (0.088, 12, 15.0),
        (0.118, 12, 0.0),
        (0.138, 16, 0.0),
        (0.146, 16, 11.25),
    ]
    for radius, count, offset_deg in rings:
        for k in range(count):
            th_deg = (offset_deg + k * (360.0 / count)) % 360.0
            th_rad = math.radians(th_deg)
            pts_xy.append((radius * math.cos(th_rad), radius * math.sin(th_rad)))
            pts_r.append(radius)
            pts_th.append(th_deg)

    return (
        np.asarray(pts_xy, dtype=np.float64),
        np.asarray(pts_r, dtype=np.float64),
        np.asarray(pts_th, dtype=np.float64),
    )


def build_polar_sensor_interp_matrix(
    solver: Coupled3DWaferFVMSolver,
    sensor_xy: np.ndarray,
) -> torch.Tensor:
    """Build exact differentiable (65, N) bilinear interpolation matrix from FVM polar cells to sensors."""
    n_s = sensor_xy.shape[0]
    r_c = solver.r_centers.detach().cpu().numpy()       # (n_r,)
    th_c = solver.theta_centers.detach().cpu().numpy()  # (n_theta,) in (0, 2pi)
    n_r, n_th = solver.n_r, solver.n_theta
    dth = solver.dtheta

    W = np.zeros((n_s, solver.N), dtype=np.float64)
    for k in range(n_s):
        x, y = float(sensor_xy[k, 0]), float(sensor_xy[k, 1])
        r = math.hypot(x, y)
        if r <= r_c[0]:
            # Center point: average of innermost ring
            W[k, :n_th] = 1.0 / n_th
            continue

        # Radial interpolation indices & weights
        if r >= r_c[-1]:
            j0 = j1 = n_r - 1
            wr = 0.0
        else:
            j1 = int(np.searchsorted(r_c, r))
            j0 = j1 - 1
            wr = (r - r_c[j0]) / (r_c[j1] - r_c[j0])

        # Periodic azimuthal interpolation indices & weights
        th = math.atan2(y, x) % (2.0 * math.pi)
        u = (th - th_c[0]) / dth
        l0 = int(math.floor(u)) % n_th
        l1 = (l0 + 1) % n_th
        wt = u - math.floor(u)

        W[k, j0 * n_th + l0] += (1.0 - wr) * (1.0 - wt)
        W[k, j0 * n_th + l1] += (1.0 - wr) * wt
        W[k, j1 * n_th + l0] += wr * (1.0 - wt)
        W[k, j1 * n_th + l1] += wr * wt

    return torch.as_tensor(W, dtype=torch.float64, device=solver.device)


def interpolate_fem_to_sensors(fem_res: dict, sensor_xy: np.ndarray) -> np.ndarray:
    """Interpolate 3D FEM wafer top-surface temperatures onto the 65 sensor coordinates."""
    xy = fem_res["wafer_top_xy"]
    T_top = fem_res["T_wafer_top_C"]
    lin = LinearNDInterpolator(xy, T_top)
    vals = lin(sensor_xy[:, 0], sensor_xy[:, 1])
    nan_mask = np.isnan(vals)
    if np.any(nan_mask):
        nn = NearestNDInterpolator(xy, T_top)
        vals[nan_mask] = nn(sensor_xy[nan_mask, 0], sensor_xy[nan_mask, 1])
    return np.asarray(vals, dtype=np.float64)


def eval_he_flow_true_sccm(
    beta_leak: float,
    p_out_torr: float,
    delta_cal: float,
    flow_params: HeFlowModelParams = HeFlowModelParams(),
) -> float:
    """Ground-truth nonlinear viscous He leak flow [sccm] (before measurement noise)."""
    leak_term = flow_params.kappa_nl * (1.0 - beta_leak ** 2) * (p_out_torr ** 2) * (1.0 + delta_cal)
    return float(flow_params.m_base_sccm + max(leak_term, 0.0))


def eval_he_flow_proxy_torch(
    beta_leak: torch.Tensor,
    p_out_torr: torch.Tensor | float,
    flow_params: HeFlowModelParams = HeFlowModelParams(),
) -> torch.Tensor:
    """Stage 1 deliberately mismatched linear He leak flow proxy [sccm]."""
    return flow_params.m_base_sccm + flow_params.kappa_lin * (1.0 - beta_leak) * p_out_torr


def generate_synthetic_observation(
    fem_res: dict,
    cond: ESC3DOperatingCondition,
    sensor_xy: np.ndarray,
    rng: np.random.Generator,
    delta_cal: float,
    sigma_T_C: float = 0.20,
    flow_params: HeFlowModelParams = HeFlowModelParams(),
    pressure_tracking_ok: bool = True,
    readback_ok: bool = True,
) -> EquipmentObservation:
    """Generate noisy 65-sensor temperatures + model-mismatched He flow from 3D FEM output."""
    T_clean = interpolate_fem_to_sensors(fem_res, sensor_xy)
    T_noisy = T_clean + rng.normal(0.0, sigma_T_C, size=T_clean.shape)

    m_clean = eval_he_flow_true_sccm(
        beta_leak=cond.beta_leak,
        p_out_torr=cond.p_he_out_torr,
        delta_cal=delta_cal,
        flow_params=flow_params,
    )
    m_noisy = max(0.0, m_clean + float(rng.normal(0.0, flow_params.sigma_m_true_sccm)))

    return EquipmentObservation(
        T_sensors_C=T_noisy,
        m_he_sccm=m_noisy,
        Q_heaters_W=tuple(float(q) for q in cond.heater_powers_W),
        p_in_torr=float(cond.p_he_in_torr),
        p_out_torr=float(cond.p_he_out_torr),
        pressure_tracking_ok=pressure_tracking_ok,
        readback_ok=readback_ok,
    )


def _sigma_m_est(m_meas: float, flow_params: HeFlowModelParams) -> float:
    excess = max(m_meas - flow_params.m_base_sccm, 0.0)
    return float(math.hypot(flow_params.sigma_m_est_base_sccm, flow_params.sigma_m_est_rel * excess))


def run_stage1_diagnosis(
    obs_0: EquipmentObservation,
    probe_obs_fn: Callable[[float], EquipmentObservation],
    solver: Coupled3DWaferFVMSolver,
    sensor_xy: np.ndarray,
    W_sensor: torch.Tensor,
    opts: SolveOptions = SolveOptions(),
    flow_params: HeFlowModelParams = HeFlowModelParams(),
    sigma_T_C: float = 0.20,
    sigma_model_T_C: float = 0.08,
    S_0_thresh: float | None = None,
    S_probe_thresh: float | None = None,
    probe_dp_torr: float = 9.0,
) -> Stage1Result:
    """Run Stage 1: H_block check -> 8-sector scan + Adjoint L-BFGS-B -> 2-point Pre-Refit Gate."""
    # ------------------------------------------------------------------
    # 0. Active Equipment Inhibit H_block (checked FIRST before probe/opt)
    # ------------------------------------------------------------------
    hblock_reasons: list[str] = []
    if obs_0.m_he_sccm >= flow_params.oem_high_flow_alarm_sccm:
        hblock_reasons.append(
            f"OEM He Flow High alarm ({obs_0.m_he_sccm:.2f} >= {flow_params.oem_high_flow_alarm_sccm:.1f} sccm)"
        )
    if not obs_0.pressure_tracking_ok:
        hblock_reasons.append("He backside pressure tracking fault")
    if not obs_0.readback_ok:
        hblock_reasons.append("Missing heater/pressure readback")

    if hblock_reasons:
        return Stage1Result(
            H_block=1,
            d_hat={"Q_plasma": 500.0, "alpha_peak": 0.25, "beta_leak": 1.0, "theta_leak_deg": 0.0},
            sigma_d={"Q_plasma": 50.0, "alpha_peak": 0.2, "beta_leak": 0.5, "theta_leak_deg": 180.0},
            S_0=float("nan"),
            S_probe=None,
            stage1_pass=False,
            leak_detected=bool(obs_0.m_he_sccm >= flow_params.oem_high_flow_alarm_sccm),
            hold_reason="H_block: " + "; ".join(hblock_reasons),
            pm_review=True,
            readback_ok=obs_0.readback_ok,
            m_he_0_sccm=obs_0.m_he_sccm,
        )

    # ------------------------------------------------------------------
    # 1. Setup observation weights & Stage 1 coarse azimuthal scan
    # ------------------------------------------------------------------
    sigma_T_eff = math.hypot(sigma_T_C, sigma_model_T_C)
    sigma_m_0 = _sigma_m_est(obs_0.m_he_sccm, flow_params)

    Q_htr_t = torch.as_tensor(obs_0.Q_heaters_W, dtype=torch.float64, device=solver.device)
    p_in_0 = float(obs_0.p_in_torr)
    p_out_0 = float(obs_0.p_out_torr)
    T_obs_0_t = torch.as_tensor(obs_0.T_sensors_C, dtype=torch.float64, device=solver.device)

    def eval_s0_loss_and_grad(x_vec: np.ndarray, need_grad: bool = True):
        Q_pl = torch.tensor(float(x_vec[0]), dtype=torch.float64, device=solver.device, requires_grad=need_grad)
        alpha = torch.tensor(float(x_vec[1]), dtype=torch.float64, device=solver.device, requires_grad=need_grad)
        beta = torch.tensor(float(x_vec[2]), dtype=torch.float64, device=solver.device, requires_grad=need_grad)
        theta = torch.tensor(float(x_vec[3]), dtype=torch.float64, device=solver.device, requires_grad=need_grad)

        out = solver.solve(
            Q_heaters=Q_htr_t,
            p_in=p_in_0,
            p_out=p_out_0,
            beta_leak=beta,
            Q_plasma=Q_pl,
            alpha_peak=alpha,
            opts=opts,
            theta_leak_deg=theta,
        )
        T_pred = W_sensor @ out["T_wafer_C"]
        m_pred = eval_he_flow_proxy_torch(beta, p_out_0, flow_params)

        r_T = (T_pred - T_obs_0_t) / sigma_T_eff
        r_m = (m_pred - obs_0.m_he_sccm) / sigma_m_0
        s0_T = torch.sum(r_T.pow(2))
        s0_m = r_m.pow(2)
        loss = s0_T + s0_m

        if not need_grad:
            return (
                float(loss.item()),
                float(s0_T.item()),
                float(s0_m.item()),
                T_pred.detach().cpu().numpy(),
                float(m_pred.item()),
            )

        loss.backward()
        grad = np.array(
            [
                float(Q_pl.grad.item()),
                float(alpha.grad.item()),
                float(beta.grad.item()),
                float(theta.grad.item()),
            ],
            dtype=np.float64,
        )
        return float(loss.item()), grad

    # Initial guess for beta from He flow proxy
    beta_from_flow = float(
        np.clip(
            1.0 - max(obs_0.m_he_sccm - flow_params.m_base_sccm, 0.0) / (flow_params.kappa_lin * max(p_out_0, 1.0)),
            0.12,
            1.0,
        )
    )

    # Candidate 0: no-leak baseline + fast 16-angle (8-sector + boundaries) azimuthal scan
    leak_candidates: list[tuple[float, np.ndarray]] = []
    with torch.no_grad():
        x_noleak = np.array([500.0, 0.25, 1.0, 225.0], dtype=np.float64)
        l_nl, _, _, _, _ = eval_s0_loss_and_grad(x_noleak, need_grad=False)

        beta_scan = min(beta_from_flow, 0.70) if beta_from_flow < 0.96 else 0.70
        for idx_a in range(16):
            th_sec = 22.5 * idx_a
            x_cand = np.array([500.0, 0.25, beta_scan, th_sec], dtype=np.float64)
            l_c, _, _, _, _ = eval_s0_loss_and_grad(x_cand, need_grad=False)
            leak_candidates.append((l_c, x_cand))

    leak_candidates.sort(key=lambda item: item[0])
    # Always refine BOTH the best leak sector candidate AND the no-leak candidate
    # so L-BFGS-B never gets trapped at beta=1.0 (where d/dtheta = 0)
    starts: list[np.ndarray] = [leak_candidates[0][1], x_noleak]

    # Scale variables for well-conditioned L-BFGS-B
    scales = np.array([100.0, 0.5, 0.5, 45.0], dtype=np.float64)
    best_x = starts[0].copy()
    best_val = float("inf")

    for x_init in starts:
        th0 = float(x_init[3])
        bounds_phys = [
            (250.0, 800.0),
            (-0.6, 0.9),
            (0.05, 1.0),
            (th0 - 65.0, th0 + 65.0),
        ]
        bounds_s = [(lo / s, hi / s) for (lo, hi), s in zip(bounds_phys, scales)]

        def obj_scaled(z: np.ndarray):
            x = z * scales
            val, g = eval_s0_loss_and_grad(x, need_grad=True)
            return val, g * scales

        res = minimize(
            obj_scaled,
            x_init / scales,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_s,
            options={"maxiter": 30, "ftol": 1e-10, "gtol": 1e-6},
        )
        if float(res.fun) < best_val:
            best_val = float(res.fun)
            best_x = res.x * scales

    best_x[3] = float(best_x[3] % 360.0)
    with torch.no_grad():
        S_0, S_0_T, S_0_m, T_pred_0, m_pred_0 = eval_s0_loss_and_grad(best_x, need_grad=False)
    max_abs_res_0 = float(np.max(np.abs(obs_0.T_sensors_C - T_pred_0)))

    # ------------------------------------------------------------------
    # 2. Laplace / Gauss-Newton Posterior Uncertainty sigma_d at d_hat
    # ------------------------------------------------------------------
    # Finite-difference Jacobian of weighted residual r(d) (66 x 4): only 4 extra solves (~0.16 s)
    eps_d = np.array([1.0, 5e-3, 5e-3, 1.0], dtype=np.float64)
    J_r = np.zeros((len(sensor_xy) + 1, 4), dtype=np.float64)
    with torch.no_grad():
        for j in range(4):
            xp = best_x.copy()
            step = eps_d[j]
            if j == 2 and xp[2] + step > 1.0:
                step = -step
            xp[j] += step
            _, _, _, T_p, m_p = eval_s0_loss_and_grad(xp, need_grad=False)
            J_r[: len(sensor_xy), j] = (T_p - T_pred_0) / (step * sigma_T_eff)
            J_r[-1, j] = (m_p - m_pred_0) / (step * sigma_m_0)

    # Weak physical prior precision to keep covariance SPD when beta ~ 1 (theta unidentifiable)
    prior_std = np.array([150.0, 0.5, 0.4, 90.0], dtype=np.float64)
    H_gn = J_r.T @ J_r + np.diag(1.0 / (prior_std ** 2))
    cov_d = np.linalg.inv(H_gn)
    sig_vec = np.sqrt(np.maximum(np.diag(cov_d), 1e-12))

    d_hat = {
        "Q_plasma": float(best_x[0]),
        "alpha_peak": float(best_x[1]),
        "beta_leak": float(best_x[2]),
        "theta_leak_deg": float(best_x[3]),
    }
    sigma_d = {
        "Q_plasma": float(sig_vec[0]),
        "alpha_peak": float(sig_vec[1]),
        "beta_leak": float(sig_vec[2]),
        "theta_leak_deg": float(sig_vec[3]),
    }

    # ------------------------------------------------------------------
    # 3. 2-Operating-Point Difference Pre-Refit Gate (S_probe)
    #    Keep d_hat FIXED (NO refit!). Safe downward probe in [3.0, P_out_0].
    # ------------------------------------------------------------------
    p_out_probe = max(3.0, p_out_0 - float(probe_dp_torr))
    obs_probe = probe_obs_fn(p_out_probe)

    if not obs_probe.pressure_tracking_ok or not obs_probe.readback_ok:
        return Stage1Result(
            H_block=1,
            d_hat=d_hat,
            sigma_d=sigma_d,
            S_0=S_0,
            S_probe=None,
            stage1_pass=False,
            leak_detected=bool((obs_0.m_he_sccm > flow_params.normal_flow_upper_sccm) or (d_hat["beta_leak"] < 0.90)),
            hold_reason="H_block: Probe operating point tracking/readback fault",
            pm_review=True,
            readback_ok=False,
            S_0_T=S_0_T,
            S_0_m=S_0_m,
            m_he_0_sccm=obs_0.m_he_sccm,
            m_he_probe_sccm=obs_probe.m_he_sccm,
            P_out_probe_torr=p_out_probe,
            max_abs_res_0_C=max_abs_res_0,
        )

    with torch.no_grad():
        out_probe = solver.solve(
            Q_heaters=Q_htr_t,
            p_in=p_in_0,
            p_out=p_out_probe,
            beta_leak=d_hat["beta_leak"],
            Q_plasma=d_hat["Q_plasma"],
            alpha_peak=d_hat["alpha_peak"],
            opts=opts,
            theta_leak_deg=d_hat["theta_leak_deg"],
        )
        T_pred_probe = (W_sensor @ out_probe["T_wafer_C"]).cpu().numpy()
        m_pred_probe = float(
            eval_he_flow_proxy_torch(
                torch.tensor(d_hat["beta_leak"], dtype=torch.float64),
                p_out_probe,
                flow_params,
            ).item()
        )

    delta_T_obs = obs_probe.T_sensors_C - obs_0.T_sensors_C
    delta_T_pred = T_pred_probe - T_pred_0
    res_dT = delta_T_obs - delta_T_pred

    delta_m_obs = obs_probe.m_he_sccm - obs_0.m_he_sccm
    delta_m_pred = m_pred_probe - m_pred_0
    res_dm = delta_m_obs - delta_m_pred

    # Difference-gate noise + model-mismatch covariance
    sigma_dT = math.sqrt(2.0 * (sigma_T_C ** 2) + (0.08 ** 2))
    sigma_dm = math.sqrt(2.0 * (flow_params.sigma_m_true_sccm ** 2) + (0.45 * abs(delta_m_pred)) ** 2 + 0.35 ** 2)

    S_probe_T = float(np.sum((res_dT / sigma_dT) ** 2))
    S_probe_m = float((res_dm / sigma_dm) ** 2)
    S_probe = S_probe_T + S_probe_m
    max_abs_res_probe = float(np.max(np.abs(res_dT)))

    # ------------------------------------------------------------------
    # 4. Gate Decision & Leak Flag
    # ------------------------------------------------------------------
    leak_detected = bool((obs_0.m_he_sccm > flow_params.normal_flow_upper_sccm) or (d_hat["beta_leak"] < 0.90))
    hold_reasons: list[str] = []
    if S_0_thresh is not None and S_0 > S_0_thresh:
        hold_reasons.append(f"S_0={S_0:.1f} > {S_0_thresh:.1f}")
    if S_probe_thresh is not None and S_probe > S_probe_thresh:
        hold_reasons.append(f"S_probe={S_probe:.1f} > {S_probe_thresh:.1f} (Pre-Refit Gate)")

    stage1_pass = len(hold_reasons) == 0
    hold_reason = "; ".join(hold_reasons) if hold_reasons else "OK"

    return Stage1Result(
        H_block=0,
        d_hat=d_hat,
        sigma_d=sigma_d,
        S_0=S_0,
        S_probe=S_probe,
        stage1_pass=stage1_pass,
        leak_detected=leak_detected,
        hold_reason=hold_reason,
        pm_review=False,
        readback_ok=obs_0.readback_ok and obs_probe.readback_ok,
        S_0_T=S_0_T,
        S_0_m=S_0_m,
        S_probe_T=S_probe_T,
        S_probe_m=S_probe_m,
        m_he_0_sccm=obs_0.m_he_sccm,
        m_he_probe_sccm=obs_probe.m_he_sccm,
        P_out_probe_torr=p_out_probe,
        max_abs_res_0_C=max_abs_res_0,
        max_abs_res_probe_C=max_abs_res_probe,
    )
