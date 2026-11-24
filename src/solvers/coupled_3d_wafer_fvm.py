"""PyTorch float64 2D Polar (r, theta) Differentiable Wafer FVM + 3D Linear ESC Coupled Solver.

Couples:
  1. Cached Mode A 3D Linear ESC Unit-Load ROM (.npz from src/rom/esc_3d_unit_load.py):
       T_esc(r, theta) = T_chiller + G_heaters @ Q_heaters + G_q @ q_gap
     with exact two-way Schur-complement elimination:
       (I + G_q * h_eff[None, :]) T_esc = T_esc_0 + G_q @ (h_eff * T_w)
       <=>  (T_w - T_esc) = (I + G_q * h_eff[None, :])^{-1} (T_w - T_esc_0)
  2. 2D Polar (r, theta) Finite-Volume Method (FVM) on the 300 mm Silicon Wafer:
       - Temperature-dependent k_Si(T) = 148 * (300 / T)^1.3 W/(m*K)
       - Distance-weighted harmonic-mean radial face conductance, periodic azimuthal conductance
       - Top & outer-rim vacuum radiation eps_Si * sigma * (T^4 - T_wall^4)
       - Exact area-normalized 2D plasma heat flux q''_plasma(r, theta)
       - Newton-Raphson with in-domain backtracking line search (T_w > 200 K, never clamped)
  3. Exact Adjoint Implicit Differentiation via custom torch.autograd.Function:
       - Solves (dF/dT_w)^T lambda = grad_T_w ONCE per backward pass
       - Computes exact VJP wrt (Q_heaters, p_in, p_out, beta_leak, Q_plasma, alpha_peak)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.physics.materials_and_he import (
    GAMMA_HE,
    K0_HE,
    LAMBDA0_HE,
    MATERIALS,
    P0_HE_TORR,
    PR_HE,
    STEFAN_BOLTZMANN,
    T0_HE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_f64_tensor(
    val: float | int | np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.to(device=device, dtype=torch.float64)
    return torch.as_tensor(val, dtype=torch.float64, device=device)


def quintic_smoothstep_torch(xi: torch.Tensor) -> torch.Tensor:
    """Compact-support C^2 quintic Hermite smoothstep on [0, 1]."""
    t = torch.clamp(xi, 0.0, 1.0)
    return t.pow(3) * (10.0 + t * (-15.0 + 6.0 * t))


def eval_he_pressure_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    p_in: torch.Tensor | float,
    p_out: torch.Tensor | float,
    beta_leak: torch.Tensor | float,
    r_leak: torch.Tensor | float = 0.138,
    theta_leak_deg: torch.Tensor | float = 225.0,
    hw_deg: float = 35.0,
    tw_deg: float = 10.0,
    r_split: float = 0.110,
    w_s: float = 0.002,
) -> torch.Tensor:
    """Differentiable C^2 compact-support backside He pressure P(r, theta) [Torr]."""
    r = torch.hypot(xx, yy)
    s_r = quintic_smoothstep_torch((r - (r_split - 0.5 * w_s)) / w_s)
    p_base = (1.0 - s_r) * p_in + s_r * p_out

    l_r = quintic_smoothstep_torch((r - r_leak) / w_s)
    if hw_deg >= 180.0:
        l_theta = torch.ones_like(r)
    else:
        theta_deg = torch.remainder(torch.rad2deg(torch.atan2(yy, xx)), 360.0)
        dtheta = torch.abs(torch.remainder(theta_deg - theta_leak_deg + 180.0, 360.0) - 180.0)
        tw = max(float(tw_deg), 1e-3)
        l_theta = 1.0 - quintic_smoothstep_torch((dtheta - (hw_deg - tw)) / tw)

    l_3d = l_r * l_theta
    return (1.0 - l_3d) * p_base + l_3d * (beta_leak * p_out)


def h_eff_torch(
    P_torr: torch.Tensor,
    d_gap: float = 10e-6,
    T_g: float = 315.0,
    h_solid: float = 50.0,
    alpha_T: float = 0.35,
) -> torch.Tensor:
    """Differentiable Kennard-Yovanovich rarefied He gap + solid contact conductance [W/(m^2*K)]."""
    P_safe = torch.clamp(P_torr, min=1e-12)
    mfp = LAMBDA0_HE * (P0_HE_TORR / P_safe) * (T_g / T0_HE)
    tac_factor = 2.0 * (2.0 - alpha_T) / alpha_T
    gas_factor = (2.0 * GAMMA_HE / (1.0 + GAMMA_HE)) / PR_HE
    M = tac_factor * gas_factor * mfp
    return h_solid + K0_HE / (d_gap + M)


def eval_plasma_flux_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    cell_areas: torch.Tensor,
    Q_plasma: torch.Tensor | float,
    alpha_peak: torch.Tensor | float,
    hotspot_amp: torch.Tensor | float = 0.0,
    hotspot_r: float = 0.135,
    hotspot_theta_deg: float = 225.0,
    hotspot_sigma: float = 0.025,
    R_w: float = 0.150,
) -> torch.Tensor:
    """Differentiable 2D plasma heat flux [W/m^2] normalized so sum(q * cell_areas) == Q_plasma."""
    r = torch.hypot(xx, yy)
    rho = torch.clamp(r / R_w, 0.0, 1.0)
    shape = 1.0 + alpha_peak * (1.0 - 2.0 * rho.pow(2))

    has_hotspot = isinstance(hotspot_amp, torch.Tensor) or abs(float(hotspot_amp)) > 1e-12
    if has_hotspot:
        th0 = math.radians(float(hotspot_theta_deg))
        x0 = float(hotspot_r) * math.cos(th0)
        y0 = float(hotspot_r) * math.sin(th0)
        dist2 = (xx - x0).pow(2) + (yy - y0).pow(2)
        shape = shape + hotspot_amp * torch.exp(-0.5 * dist2 / (float(hotspot_sigma) ** 2))

    shape = torch.clamp(shape, min=0.0)
    raw_int = torch.sum(shape * cell_areas)
    return shape * (Q_plasma / raw_int)


@dataclass(frozen=True)
class SolveOptions:
    """Fixed non-differentiated physical & numerical options for a coupled 3D FVM solve."""
    r_leak_m: float = 0.138
    theta_leak_deg: float = 225.0
    hw_deg: float = 35.0
    tw_deg: float = 10.0
    r_split_m: float = 0.110
    w_s_m: float = 0.002
    d_gap_m: float = 10e-6
    T_g_K: float = 315.0
    h_solid: float = 50.0
    alpha_T: float = 0.35
    hotspot_amp: float = 0.0
    hotspot_r_m: float = 0.135
    hotspot_theta_deg: float = 225.0
    hotspot_sigma_m: float = 0.025
    T_chiller_C: float = 20.0
    T_wall_C: float = 60.0
    include_radiation: bool = True
    nonlinear_k_si: bool = True
    tol: float = 1e-11
    max_iter: int = 30
    T_min_K: float = 200.0


class _WaferImplicitSolveFn(torch.autograd.Function):
    """Custom autograd Function for exact Adjoint Implicit Differentiation of T_w*(theta)."""

    @staticmethod
    def forward(
        ctx: Any,
        Q_heaters: torch.Tensor,
        p_in: torch.Tensor,
        p_out: torch.Tensor,
        beta_leak: torch.Tensor,
        Q_plasma: torch.Tensor,
        alpha_peak: torch.Tensor,
        theta_leak_deg: torch.Tensor,
        solver: "Coupled3DWaferFVMSolver",
        opts: SolveOptions,
    ) -> torch.Tensor:
        with torch.no_grad():
            T_w_star, J_star = solver._solve_newton_with_jacobian(
                Q_heaters=Q_heaters.detach(),
                p_in=p_in.detach(),
                p_out=p_out.detach(),
                beta_leak=beta_leak.detach(),
                Q_plasma=Q_plasma.detach(),
                alpha_peak=alpha_peak.detach(),
                theta_leak_deg=theta_leak_deg.detach(),
                opts=opts,
            )

        ctx.save_for_backward(
            T_w_star,
            J_star,
            Q_heaters.detach(),
            p_in.detach(),
            p_out.detach(),
            beta_leak.detach(),
            Q_plasma.detach(),
            alpha_peak.detach(),
            theta_leak_deg.detach(),
        )
        ctx.solver = solver
        ctx.opts = opts
        return T_w_star

    @staticmethod
    def backward(ctx: Any, grad_T_w: torch.Tensor):
        (
            T_w_star,
            J_star,
            Q_heaters,
            p_in,
            p_out,
            beta_leak,
            Q_plasma,
            alpha_peak,
            theta_leak_deg,
        ) = ctx.saved_tensors
        solver: Coupled3DWaferFVMSolver = ctx.solver
        opts: SolveOptions = ctx.opts

        needs = ctx.needs_input_grad[:7]
        if not any(needs):
            return (None, None, None, None, None, None, None, None, None)

        # Solve adjoint linear system ONCE: (dF/dT_w)^T lambda = grad_T_w
        lam = torch.linalg.solve(J_star.transpose(-2, -1), grad_T_w)

        raw_inputs = [Q_heaters, p_in, p_out, beta_leak, Q_plasma, alpha_peak, theta_leak_deg]
        req_inputs: list[torch.Tensor] = []
        active_indices: list[int] = []
        diff_inputs: list[torch.Tensor] = []
        for idx, (ten, need) in enumerate(zip(raw_inputs, needs)):
            if need:
                t_req = ten.detach().clone().requires_grad_(True)
                req_inputs.append(t_req)
                active_indices.append(idx)
                diff_inputs.append(t_req)
            else:
                diff_inputs.append(ten.detach())

        with torch.enable_grad():
            F_val = solver._eval_residual_differentiable(
                T_w=T_w_star.detach(),
                Q_heaters=diff_inputs[0],
                p_in=diff_inputs[1],
                p_out=diff_inputs[2],
                beta_leak=diff_inputs[3],
                Q_plasma=diff_inputs[4],
                alpha_peak=diff_inputs[5],
                theta_leak_deg=diff_inputs[6],
                opts=opts,
            )
            vjps = torch.autograd.grad(
                outputs=F_val,
                inputs=req_inputs,
                grad_outputs=-lam,
                allow_unused=True,
            )

        out_grads: list[torch.Tensor | None] = [None] * 7
        for idx, g in zip(active_indices, vjps):
            out_grads[idx] = g

        return (*out_grads, None, None)


class Coupled3DWaferFVMSolver:
    """PyTorch float64 2D Polar FVM Wafer + Mode A 3D Linear ESC Coupled Solver."""

    def __init__(
        self,
        rom_path: str | Path | None = None,
        refinement: str = "medium",
        ceramic_type: str = "Al2O3",
        n_r: int = 24,
        n_theta: int = 36,
        n_knots: int = 10,
        k_max: int = 4,
        t_wafer_m: float = 0.000775,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.t_wafer_m = float(t_wafer_m)
        self.k_si_ref = float(MATERIALS["Si"].k_ref)
        self.eps_si = float(MATERIALS["Si"].emissivity)
        self.sigma = float(STEFAN_BOLTZMANN)

        if rom_path is None:
            rom_path = (
                PROJECT_ROOT
                / "results"
                / "rom"
                / f"esc3d_rom_{ceramic_type}_{refinement}_nr{n_r}_nt{n_theta}_nk{n_knots}_km{k_max}.npz"
            )
        self.rom_path = Path(rom_path)
        if not self.rom_path.exists():
            raise FileNotFoundError(f"Mode A 3D ESC ROM file not found: {self.rom_path}")

        data = np.load(self.rom_path)
        self.ceramic_type = str(data["ceramic_type"])
        self.refinement = str(data["refinement"])
        self.n_r = int(data["n_r"])
        self.n_theta = int(data["n_theta"])
        self.N = self.n_r * self.n_theta

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(arr, dtype=np.float64), dtype=torch.float64, device=self.device)

        self.r_faces = _t(data["r_faces"])                  # (n_r + 1,)
        self.r_centers = _t(data["r_centers"])              # (n_r,)
        self.dr = _t(data["dr"])                            # (n_r,)
        self.dtheta = float(data["dtheta"])
        self.theta_centers = _t(data["theta_centers"])      # (n_theta,)
        self.xx = _t(data["xx"])                            # (n_r, n_theta)
        self.yy = _t(data["yy"])                            # (n_r, n_theta)
        self.xx_flat = self.xx.reshape(-1)                  # (N,)
        self.yy_flat = self.yy.reshape(-1)                  # (N,)
        self.cell_areas = _t(data["cell_areas"])            # (n_r, n_theta)
        self.cell_areas_flat = self.cell_areas.reshape(-1)  # (N,)
        self.r_wafer_m = float(self.r_faces[-1].item())

        self.G_heaters = _t(data["G_heaters"])                    # (N, 10)
        self.G_q = _t(data["G_q"])                                # (N, N)
        self.c_heaters_cool = _t(data["c_heaters_cool"])          # (10,)
        self.c_q_cool = _t(data["c_q_cool"])                      # (N,)
        self.G_heaters_esc_full = _t(data["G_heaters_esc_full"])  # (N_esc_full, 10)
        self.G_q_esc_full = _t(data["G_q_esc_full"])              # (N_esc_full, N)
        self.r_esc_centers = _t(data["r_esc_centers"])            # (n_r + 2,)

        # Precompute 2D Polar FVM geometric factors
        # Radial interior faces j = 0 .. n_r - 2 between cell j and j + 1 at r_faces[j + 1]
        r_int_faces = self.r_faces[1:-1]                              # (n_r - 1,)
        dr_center = self.r_centers[1:] - self.r_centers[:-1]          # (n_r - 1,)
        d_plus = r_int_faces - self.r_centers[:-1]                    # (n_r - 1,)
        d_minus = self.r_centers[1:] - r_int_faces                    # (n_r - 1,)
        self.w_r_left = (d_plus / dr_center)[:, None]                 # (n_r - 1, 1)
        self.w_r_right = (d_minus / dr_center)[:, None]               # (n_r - 1, 1)
        self.g_r = ((r_int_faces * self.dtheta * self.t_wafer_m) / dr_center)[:, None]  # (n_r - 1, 1)

        # Azimuthal periodic faces between (j, l) and (j, (l + 1) % n_theta)
        self.g_theta = ((self.dr * self.t_wafer_m) / (self.r_centers * self.dtheta))[:, None]  # (n_r, 1)

        # Outer rim radiation face area per azimuthal cell at j = n_r - 1
        self.edge_face_area = self.r_wafer_m * self.dtheta * self.t_wafer_m

        # Precompute flat index maps for fast vectorized analytical Jacobian assembly
        j_r = torch.arange(self.n_r - 1, device=self.device)[:, None]
        l_all = torch.arange(self.n_theta, device=self.device)[None, :]
        self._rad_a = (j_r * self.n_theta + l_all).reshape(-1)
        self._rad_b = ((j_r + 1) * self.n_theta + l_all).reshape(-1)

        j_all = torch.arange(self.n_r, device=self.device)[:, None]
        l_next = (l_all + 1) % self.n_theta
        self._th_a = (j_all * self.n_theta + l_all).reshape(-1)
        self._th_b = (j_all * self.n_theta + l_next).reshape(-1)

        self._diag_idx = torch.arange(self.N, device=self.device)
        self._outer_idx = (self.n_r - 1) * self.n_theta + torch.arange(self.n_theta, device=self.device)
        self._eye_N = torch.eye(self.N, dtype=torch.float64, device=self.device)

    def _k_si(self, T_2d: torch.Tensor, nonlinear_k_si: bool) -> torch.Tensor:
        if nonlinear_k_si:
            return self.k_si_ref * (300.0 / T_2d).pow(1.3)
        return torch.full_like(T_2d, self.k_si_ref)

    def _wafer_conduction_and_rad(
        self,
        T_flat: torch.Tensor,
        opts: SolveOptions,
    ) -> torch.Tensor:
        """Compute net outward conduction + radiation power [W] for all N wafer cells."""
        T_2d = T_flat.reshape(self.n_r, self.n_theta)
        k = self._k_si(T_2d, opts.nonlinear_k_si)

        res = torch.zeros_like(T_2d)

        # 1. Radial interior faces (distance-weighted harmonic mean)
        k_l = k[:-1, :]
        k_r = k[1:, :]
        k_face_r = (k_l * k_r) / (self.w_r_right * k_l + self.w_r_left * k_r)
        Q_r = self.g_r * k_face_r * (T_2d[:-1, :] - T_2d[1:, :])
        res[:-1, :] = res[:-1, :] + Q_r
        res[1:, :] = res[1:, :] - Q_r

        # 2. Azimuthal periodic faces (equal-distance harmonic mean)
        k_next = torch.roll(k, shifts=-1, dims=1)
        T_next = torch.roll(T_2d, shifts=-1, dims=1)
        k_face_th = (2.0 * k * k_next) / (k + k_next)
        Q_th = self.g_theta * k_face_th * (T_2d - T_next)
        res = res + Q_th - torch.roll(Q_th, shifts=1, dims=1)

        # 3. Vacuum radiation (top surface + outer wafer rim)
        if opts.include_radiation:
            T_wall_K = opts.T_wall_C + 273.15
            q_rad = self.eps_si * self.sigma * (T_2d.pow(4) - T_wall_K ** 4)
            res = res + self.cell_areas * q_rad
            res[-1, :] = res[-1, :] + self.edge_face_area * q_rad[-1, :]

        return res.reshape(-1)

    def _wafer_conduction_and_rad_jacobian(
        self,
        T_flat: torch.Tensor,
        opts: SolveOptions,
    ) -> torch.Tensor:
        """Assemble exact N x N analytical Jacobian of wafer conduction + radiation [W/K]."""
        T_2d = T_flat.reshape(self.n_r, self.n_theta)
        k = self._k_si(T_2d, opts.nonlinear_k_si)
        if opts.nonlinear_k_si:
            dk_dT = -1.3 * k / T_2d
        else:
            dk_dT = torch.zeros_like(k)

        J = torch.zeros((self.N, self.N), dtype=torch.float64, device=self.device)

        # 1. Radial interior faces
        k_a, k_b = k[:-1, :], k[1:, :]
        dk_a, dk_b = dk_dT[:-1, :], dk_dT[1:, :]
        dT_r = T_2d[:-1, :] - T_2d[1:, :]
        k_f = (k_a * k_b) / (self.w_r_right * k_a + self.w_r_left * k_b)
        dkf_dka = self.w_r_left * (k_f / k_a).pow(2)
        dkf_dkb = self.w_r_right * (k_f / k_b).pow(2)

        dQ_dTa = (self.g_r * (k_f + dkf_dka * dk_a * dT_r)).reshape(-1)
        dQ_dTb = (self.g_r * (-k_f + dkf_dkb * dk_b * dT_r)).reshape(-1)

        J.index_put_((self._rad_a, self._rad_a), dQ_dTa, accumulate=True)
        J.index_put_((self._rad_a, self._rad_b), dQ_dTb, accumulate=True)
        J.index_put_((self._rad_b, self._rad_a), -dQ_dTa, accumulate=True)
        J.index_put_((self._rad_b, self._rad_b), -dQ_dTb, accumulate=True)

        # 2. Azimuthal periodic faces (w_a = w_b = 0.5)
        k_next = torch.roll(k, shifts=-1, dims=1)
        dk_next = torch.roll(dk_dT, shifts=-1, dims=1)
        T_next = torch.roll(T_2d, shifts=-1, dims=1)
        dT_th = T_2d - T_next
        k_f_th = (2.0 * k * k_next) / (k + k_next)
        dkf_dka_th = 0.5 * (k_f_th / k).pow(2)
        dkf_dkb_th = 0.5 * (k_f_th / k_next).pow(2)

        dQ_dTa_th = (self.g_theta * (k_f_th + dkf_dka_th * dk_dT * dT_th)).reshape(-1)
        dQ_dTb_th = (self.g_theta * (-k_f_th + dkf_dkb_th * dk_next * dT_th)).reshape(-1)

        J.index_put_((self._th_a, self._th_a), dQ_dTa_th, accumulate=True)
        J.index_put_((self._th_a, self._th_b), dQ_dTb_th, accumulate=True)
        J.index_put_((self._th_b, self._th_a), -dQ_dTa_th, accumulate=True)
        J.index_put_((self._th_b, self._th_b), -dQ_dTb_th, accumulate=True)

        # 3. Radiation diagonal terms
        if opts.include_radiation:
            dq_rad_dT = 4.0 * self.eps_si * self.sigma * T_flat.pow(3)
            J[self._diag_idx, self._diag_idx] += self.cell_areas_flat * dq_rad_dT
            J[self._outer_idx, self._outer_idx] += self.edge_face_area * dq_rad_dT[self._outer_idx]

        return J

    def _eval_residual_differentiable(
        self,
        T_w: torch.Tensor,
        Q_heaters: torch.Tensor,
        p_in: torch.Tensor,
        p_out: torch.Tensor,
        beta_leak: torch.Tensor,
        Q_plasma: torch.Tensor,
        alpha_peak: torch.Tensor,
        theta_leak_deg: torch.Tensor,
        opts: SolveOptions,
    ) -> torch.Tensor:
        """Differentiable coupled wafer residual F(T_w, theta) [W] for Adjoint VJP."""
        P_torr = eval_he_pressure_torch(
            self.xx_flat,
            self.yy_flat,
            p_in=p_in,
            p_out=p_out,
            beta_leak=beta_leak,
            r_leak=opts.r_leak_m,
            theta_leak_deg=theta_leak_deg,
            hw_deg=opts.hw_deg,
            tw_deg=opts.tw_deg,
            r_split=opts.r_split_m,
            w_s=opts.w_s_m,
        )
        h_eff = h_eff_torch(
            P_torr,
            d_gap=opts.d_gap_m,
            T_g=opts.T_g_K,
            h_solid=opts.h_solid,
            alpha_T=opts.alpha_T,
        )
        q_plasma = eval_plasma_flux_torch(
            self.xx_flat,
            self.yy_flat,
            self.cell_areas_flat,
            Q_plasma=Q_plasma,
            alpha_peak=alpha_peak,
            hotspot_amp=opts.hotspot_amp,
            hotspot_r=opts.hotspot_r_m,
            hotspot_theta_deg=opts.hotspot_theta_deg,
            hotspot_sigma=opts.hotspot_sigma_m,
            R_w=self.r_wafer_m,
        )

        T_chiller_K = opts.T_chiller_C + 273.15
        T_esc_0 = T_chiller_K + self.G_heaters @ Q_heaters
        M_esc = self._eye_N + self.G_q * h_eff[None, :]
        dT_gap = torch.linalg.solve(M_esc, T_w - T_esc_0)
        q_gap = h_eff * dT_gap

        cond_rad = self._wafer_conduction_and_rad(T_w, opts)
        return cond_rad + self.cell_areas_flat * (q_gap - q_plasma)

    def _solve_newton_with_jacobian(
        self,
        Q_heaters: torch.Tensor,
        p_in: torch.Tensor,
        p_out: torch.Tensor,
        beta_leak: torch.Tensor,
        Q_plasma: torch.Tensor,
        alpha_peak: torch.Tensor,
        theta_leak_deg: torch.Tensor,
        opts: SolveOptions,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve coupled nonlinear wafer FVM via Newton-Raphson + in-domain backtracking."""
        P_torr = eval_he_pressure_torch(
            self.xx_flat,
            self.yy_flat,
            p_in=p_in,
            p_out=p_out,
            beta_leak=beta_leak,
            r_leak=opts.r_leak_m,
            theta_leak_deg=theta_leak_deg,
            hw_deg=opts.hw_deg,
            tw_deg=opts.tw_deg,
            r_split=opts.r_split_m,
            w_s=opts.w_s_m,
        )
        h_eff = h_eff_torch(
            P_torr,
            d_gap=opts.d_gap_m,
            T_g=opts.T_g_K,
            h_solid=opts.h_solid,
            alpha_T=opts.alpha_T,
        )
        q_plasma = eval_plasma_flux_torch(
            self.xx_flat,
            self.yy_flat,
            self.cell_areas_flat,
            Q_plasma=Q_plasma,
            alpha_peak=alpha_peak,
            hotspot_amp=opts.hotspot_amp,
            hotspot_r=opts.hotspot_r_m,
            hotspot_theta_deg=opts.hotspot_theta_deg,
            hotspot_sigma=opts.hotspot_sigma_m,
            R_w=self.r_wafer_m,
        )

        # Precompute exact Schur-complement ESC elimination matrix ONCE
        T_chiller_K = opts.T_chiller_C + 273.15
        T_esc_0 = T_chiller_K + self.G_heaters @ Q_heaters
        M_esc = self._eye_N + self.G_q * h_eff[None, :]
        lu, pivots = torch.linalg.lu_factor(M_esc)
        M_esc_inv = torch.linalg.lu_solve(lu, pivots, self._eye_N)

        # Constant gap-coupling Jacobian: d(A * q_gap)/dT_w = diag(A * h_eff) @ M_esc^{-1}
        J_gap = (self.cell_areas_flat * h_eff)[:, None] * M_esc_inv
        Q_plasma_cell = self.cell_areas_flat * q_plasma

        q_ref = max(abs(float(Q_plasma.item())) / self.N, 1e-3)

        def eval_res(T_vec: torch.Tensor) -> torch.Tensor:
            dT_gap = M_esc_inv @ (T_vec - T_esc_0)
            q_gap = h_eff * dT_gap
            return self._wafer_conduction_and_rad(T_vec, opts) + self.cell_areas_flat * q_gap - Q_plasma_cell

        # Initial guess from exact zero-conduction equilibrium
        T_w = torch.clamp(T_esc_0 + q_plasma / torch.clamp(h_eff, min=1.0), min=opts.T_min_K + 10.0)
        res = eval_res(T_w)
        rel_inf = float(torch.max(torch.abs(res)).item()) / q_ref

        for _ in range(opts.max_iter):
            J = self._wafer_conduction_and_rad_jacobian(T_w, opts) + J_gap
            if rel_inf < opts.tol:
                return T_w, J

            dT = torch.linalg.solve(J, -res)
            res_norm = float(torch.linalg.norm(res).item())

            # In-domain backtracking line search: strictly maintain T_w > T_min_K (NEVER clamp!)
            alpha = 1.0
            while float(torch.min(T_w + alpha * dT).item()) <= opts.T_min_K:
                alpha *= 0.5
                if alpha < 1e-10:
                    raise RuntimeError("Newton line search failed to stay in domain T_w > 200 K.")

            T_cand = T_w + alpha * dT
            res_cand = eval_res(T_cand)
            cand_norm = float(torch.linalg.norm(res_cand).item())
            while cand_norm > (1.0 - 1e-4 * alpha) * res_norm and alpha > 1e-6:
                alpha *= 0.5
                T_cand = T_w + alpha * dT
                res_cand = eval_res(T_cand)
                cand_norm = float(torch.linalg.norm(res_cand).item())

            T_w = T_cand
            res = res_cand
            rel_inf = float(torch.max(torch.abs(res)).item()) / q_ref

        if rel_inf >= opts.tol:
            raise RuntimeError(
                f"Coupled 3D Wafer FVM Newton-Raphson did not converge: "
                f"||r_w / q_ref||_inf = {rel_inf:.3e} >= {opts.tol:.1e}"
            )
        J = self._wafer_conduction_and_rad_jacobian(T_w, opts) + J_gap
        return T_w, J

    def solve(
        self,
        Q_heaters: torch.Tensor | np.ndarray | tuple[float, ...],
        p_in: torch.Tensor | float = 12.0,
        p_out: torch.Tensor | float = 12.0,
        beta_leak: torch.Tensor | float = 1.0,
        Q_plasma: torch.Tensor | float = 500.0,
        alpha_peak: torch.Tensor | float = 0.25,
        opts: SolveOptions = SolveOptions(),
        theta_leak_deg: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Solve coupled 3D ESC + 2D Polar Wafer FVM with exact end-to-end autograd."""
        Q_htr_t = _as_f64_tensor(Q_heaters, self.device)
        p_in_t = _as_f64_tensor(p_in, self.device)
        p_out_t = _as_f64_tensor(p_out, self.device)
        beta_t = _as_f64_tensor(beta_leak, self.device)
        Q_pl_t = _as_f64_tensor(Q_plasma, self.device)
        alpha_t = _as_f64_tensor(alpha_peak, self.device)
        theta_t = _as_f64_tensor(
            opts.theta_leak_deg if theta_leak_deg is None else theta_leak_deg,
            self.device,
        )

        T_w = _WaferImplicitSolveFn.apply(
            Q_htr_t, p_in_t, p_out_t, beta_t, Q_pl_t, alpha_t, theta_t, self, opts
        )

        # Differentiable reconstruction of ESC fields and independent energy balances
        P_torr = eval_he_pressure_torch(
            self.xx_flat,
            self.yy_flat,
            p_in=p_in_t,
            p_out=p_out_t,
            beta_leak=beta_t,
            r_leak=opts.r_leak_m,
            theta_leak_deg=theta_t,
            hw_deg=opts.hw_deg,
            tw_deg=opts.tw_deg,
            r_split=opts.r_split_m,
            w_s=opts.w_s_m,
        )
        h_eff = h_eff_torch(
            P_torr,
            d_gap=opts.d_gap_m,
            T_g=opts.T_g_K,
            h_solid=opts.h_solid,
            alpha_T=opts.alpha_T,
        )
        T_chiller_K = opts.T_chiller_C + 273.15
        T_esc_0 = T_chiller_K + self.G_heaters @ Q_htr_t
        M_esc = self._eye_N + self.G_q * h_eff[None, :]
        dT_gap = torch.linalg.solve(M_esc, T_w - T_esc_0)
        T_esc_top = T_w - dT_gap
        q_gap = h_eff * dT_gap

        T_esc_full = T_chiller_K + self.G_heaters_esc_full @ Q_htr_t + self.G_q_esc_full @ q_gap

        # Independent bottom chiller cooling [W] from Mode A ROM Robin integrals
        Q_cool_W = torch.dot(self.c_heaters_cool, Q_htr_t) + torch.dot(self.c_q_cool, q_gap)
        Q_gap_W = torch.dot(self.cell_areas_flat, q_gap)

        if opts.include_radiation:
            T_wall_K = opts.T_wall_C + 273.15
            q_rad_top = self.eps_si * self.sigma * (T_w.pow(4) - T_wall_K ** 4)
            Q_rad_top_W = torch.dot(self.cell_areas_flat, q_rad_top)
            Q_rad_edge_W = self.edge_face_area * torch.sum(q_rad_top[self._outer_idx])
        else:
            zero = torch.zeros((), dtype=torch.float64, device=self.device)
            Q_rad_top_W = zero
            Q_rad_edge_W = zero

        Q_heaters_total_W = torch.sum(Q_htr_t)
        wafer_imbalance_W = Q_pl_t - (Q_gap_W + Q_rad_top_W + Q_rad_edge_W)
        esc_imbalance_W = (Q_heaters_total_W + Q_gap_W) - Q_cool_W
        global_imbalance_W = (Q_pl_t + Q_heaters_total_W) - (Q_cool_W + Q_rad_top_W + Q_rad_edge_W)

        return {
            "T_wafer_K": T_w,
            "T_wafer_C": T_w - 273.15,
            "T_wafer_2d_C": (T_w - 273.15).reshape(self.n_r, self.n_theta),
            "T_esc_top_K": T_esc_top,
            "T_esc_top_C": T_esc_top - 273.15,
            "T_esc_full_K": T_esc_full,
            "T_esc_full_C": T_esc_full - 273.15,
            "q_gap_W_m2": q_gap,
            "h_eff_W_m2K": h_eff,
            "P_he_torr": P_torr,
            "Q_gap_W": Q_gap_W,
            "Q_cool_W": Q_cool_W,
            "Q_rad_top_W": Q_rad_top_W,
            "Q_rad_edge_W": Q_rad_edge_W,
            "wafer_imbalance_W": wafer_imbalance_W,
            "esc_imbalance_W": esc_imbalance_W,
            "global_imbalance_W": global_imbalance_W,
        }
