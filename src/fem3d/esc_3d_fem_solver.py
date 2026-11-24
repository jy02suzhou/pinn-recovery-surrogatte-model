"""3D Multi-Domain FEniCSx Thermal Solver for 300 mm Plasma Etch ESC & Wafer.

Solves steady-state 3D coupled nonlinear heat transfer across:
  1. Silicon Wafer (R_w = 150 mm, t_w = 775 um):
       - Temperature-dependent k_Si(T) = 148 * (300/T)^1.3 W/(m*K)
       - Top: Area-normalized 2D plasma flux q''_plasma(r, theta) + radiation T^4
       - Outer edge: Vacuum radiation T^4
  2. Rarefied He Micro-Gap (d = 10 um, virtual orthotropic layer k_z = h_eff*t_gap, k_xy = 1e-6):
       - C^2 Quintic Hermite dual-zone pressure + localized azimuthal edge leak beta_leak
       - Kennard-Yovanovich rarefied conductance h_eff(P(r,theta), T_g)
  3. ESC Ceramic Dielectric (R_esc = 155 mm, t_cer = 2.0 mm, Al2O3 vs AlN):
       - 10 independent embedded heater zones (2 inner rings + 8 azimuthal 45° edge sectors)
  4. Al Cooling Base (R_base = 160 mm, t_base = 15.0 mm):
       - Bottom chiller linear Robin BC: -k_Al * dT/dn = h_chiller * (T - T_chiller)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from src.physics.materials_and_he import MATERIALS, STEFAN_BOLTZMANN, h_eff_interface

from .build_esc_3d_mesh import (
    BND_CHILLER_BOTTOM,
    BND_ESC_TOP_INTERFACE,
    BND_WAFER_EDGE,
    BND_WAFER_TOP,
    ESC3DGeometrySpec,
    N_EDGE_SECTORS,
    N_HEATER_ZONES,
    TAG_AL_BASE,
    TAG_CERAMIC_BULK,
    TAG_HE_GAP,
    TAG_HEATER_EDGE_START,
    TAG_HEATER_Z1,
    TAG_HEATER_Z2,
    TAG_WAFER,
    build_esc_3d_gmsh,
)


def quintic_smoothstep(xi: np.ndarray) -> np.ndarray:
    """Compact-support C^2 quintic Hermite smoothstep on [0, 1]."""
    t = np.clip(xi, 0.0, 1.0)
    return t ** 3 * (10.0 + t * (-15.0 + 6.0 * t))


@dataclass
class ESC3DOperatingCondition:
    """3D operating parameters, 10-zone heater recipe, and localized (r, theta) faults."""
    ceramic_type: str = "Al2O3"
    total_plasma_power_W: float = 500.0
    plasma_peaking_factor: float = 0.25       # Radial peaking (+ center, - edge)

    # Optional localized azimuthal plasma hot-spot (area-normalized to keep total_plasma_power_W exact)
    plasma_hotspot_amp: float = 0.0           # Relative amplitude (e.g. 0.40 = +40% local peak)
    plasma_hotspot_r_m: float = 0.135         # Hotspot radial center [m]
    plasma_hotspot_theta_deg: float = 225.0   # Hotspot angle [deg] (225° = lower-left)
    plasma_hotspot_sigma_m: float = 0.025     # Gaussian spatial width [m]

    # Dual-zone backside He pressure [Torr] (in [3.0, 20.0] Torr hardware bounds)
    p_he_in_torr: float = 12.0
    p_he_out_torr: float = 12.0
    r_he_split_m: float = 0.110               # Split radius between inner and outer He zone [m]
    w_smooth_m: float = 0.002                 # C^2 transition width [m] (2.0 mm)

    # Localized azimuthal edge He leak: P_edge = beta_leak * P_out
    beta_leak: float = 1.0                    # 1.0 = no leak, 0.05..1.0 = leak retention ratio
    r_leak_m: float = 0.138                   # Must satisfy r_leak_m + w_smooth_m <= 0.150 m
    leak_theta_center_deg: float = 225.0      # Angular center of localized seal leak [deg]
    leak_theta_halfwidth_deg: float = 35.0    # >= 180° for full axisymmetric ring leak
    leak_theta_smooth_deg: float = 10.0       # C^2 angular transition width [deg]

    d_he_gap_m: float = 10e-6
    d_gap_warpage_m: float = 0.0              # Optional OOD wafer bow: d_gap(r) = d_he_gap_m + d_gap_warpage_m * (r/R_w)^2
    h_solid_contact: float = 50.0
    alpha_T: float = 0.35

    # 10-zone heater powers [W]: (Z1_center, Z2_mid, Z3..Z10_edge_sectors_0..7)
    heater_powers_W: tuple[float, ...] = (80.0, 200.0) + (55.0,) * N_EDGE_SECTORS

    # Chiller & chamber ambient BCs
    T_chiller_C: float = 20.0
    h_chiller: float = 2000.0
    T_wall_C: float = 60.0
    include_radiation: bool = True
    nonlinear_k_si: bool = True


def eval_he_pressure_3d(x: np.ndarray, y: np.ndarray, cond: ESC3DOperatingCondition) -> np.ndarray:
    """Evaluate C^2 compact-support backside He pressure P(r, theta) [Torr]."""
    r = np.hypot(x, y)
    s_r = quintic_smoothstep((r - (cond.r_he_split_m - 0.5 * cond.w_smooth_m)) / cond.w_smooth_m)
    p_base = (1.0 - s_r) * cond.p_he_in_torr + s_r * cond.p_he_out_torr

    if cond.beta_leak >= 1.0 - 1e-12:
        return p_base

    l_r = quintic_smoothstep((r - cond.r_leak_m) / cond.w_smooth_m)
    if cond.leak_theta_halfwidth_deg >= 180.0:
        l_theta = np.ones_like(r)
    else:
        theta_deg = np.degrees(np.arctan2(y, x)) % 360.0
        dtheta = np.abs(((theta_deg - cond.leak_theta_center_deg + 180.0) % 360.0) - 180.0)
        hw = cond.leak_theta_halfwidth_deg
        tw = max(cond.leak_theta_smooth_deg, 1e-3)
        # 1 inside |dtheta| <= hw - tw, smooth C^2 drop to 0 at |dtheta| >= hw
        l_theta = 1.0 - quintic_smoothstep((dtheta - (hw - tw)) / tw)

    l_3d = l_r * l_theta
    return (1.0 - l_3d) * p_base + l_3d * (cond.beta_leak * cond.p_he_out_torr)


def eval_plasma_flux_3d_unnorm(
    x: np.ndarray, y: np.ndarray, cond: ESC3DOperatingCondition, R_w: float = 0.150
) -> np.ndarray:
    """Evaluate unnormalized 2D plasma heat flux shape q_raw(x, y)."""
    r = np.hypot(x, y)
    rho = np.clip(r / R_w, 0.0, 1.0)
    shape = 1.0 + cond.plasma_peaking_factor * (1.0 - 2.0 * rho ** 2)
    if abs(cond.plasma_hotspot_amp) > 1e-12:
        th0 = np.radians(cond.plasma_hotspot_theta_deg)
        x0 = cond.plasma_hotspot_r_m * np.cos(th0)
        y0 = cond.plasma_hotspot_r_m * np.sin(th0)
        dist2 = (x - x0) ** 2 + (y - y0) ** 2
        shape = shape + cond.plasma_hotspot_amp * np.exp(-0.5 * dist2 / (cond.plasma_hotspot_sigma_m ** 2))
    return np.maximum(shape, 0.0)


class ESC3DFEMSolver:
    """3D FEniCSx Multi-Domain Thermal Solver for 300 mm Multi-Zone ESC & Wafer."""

    def __init__(
        self,
        refinement: str = "coarse",
        geom: ESC3DGeometrySpec = ESC3DGeometrySpec(),
        degree: int = 1,
    ) -> None:
        self.refinement = refinement
        self.geom = geom
        self.degree = degree

        msh_path = PROJECT_ROOT / "results" / "mesh" / f"esc_3d_{refinement}.msh"
        if not msh_path.exists():
            build_esc_3d_gmsh(refinement=refinement, msh_path=msh_path, geom=geom)

        mesh_data = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, 0, gdim=3)
        self.domain = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        self.facet_tags = mesh_data.facet_tags

        self.V = fem.functionspace(self.domain, ("Lagrange", degree))
        self.W_dg = fem.functionspace(self.domain, ("DG", 0))

        self.dx = ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)

        # Cell centroids (x, y, z) for DG0 coefficient assignment
        tdim = self.domain.topology.dim
        num_cells = self.domain.topology.index_map(tdim).size_local
        self.domain.topology.create_connectivity(tdim, 0)
        c2v = self.domain.topology.connectivity(tdim, 0)
        coords = self.domain.geometry.x
        cell_mid = np.zeros((num_cells, 3), dtype=float)
        for c in range(num_cells):
            verts = c2v.links(c)
            cell_mid[c] = coords[verts].mean(axis=0)
        self.cell_mid = cell_mid

        self.heater_tags = (TAG_HEATER_Z1, TAG_HEATER_Z2) + tuple(
            TAG_HEATER_EDGE_START + s for s in range(N_EDGE_SECTORS)
        )
        self.all_tags = (TAG_AL_BASE, TAG_CERAMIC_BULK, TAG_HE_GAP, TAG_WAFER) + self.heater_tags
        self.tag_to_cells = {t: self.cell_tags.find(t) for t in self.all_tags}

        # Compute exact discrete mesh volumes for each heater zone so integral q_dot dV == Q_i exactly
        one_fn = fem.Constant(self.domain, 1.0)
        self.heater_mesh_volumes = tuple(
            float(fem.assemble_scalar(fem.form(one_fn * self.dx(t)))) for t in self.heater_tags
        )

        # Precompute wafer top, wafer bottom, and ESC top DOF indices
        dof_xyz = self.V.tabulate_dof_coordinates()
        _, _, _, _, z_esc_top, z_waf_bot, z_waf_top = self.geom.z_levels
        r_dof = np.hypot(dof_xyz[:, 0], dof_xyz[:, 1])

        self.wafer_top_idx = np.where(np.isclose(dof_xyz[:, 2], z_waf_top, atol=1e-5))[0]
        self.wafer_bot_idx = np.where(np.isclose(dof_xyz[:, 2], z_waf_bot, atol=1e-5))[0]
        self.esc_top_idx = np.where(
            np.isclose(dof_xyz[:, 2], z_esc_top, atol=1e-5) & (r_dof <= self.geom.r_esc + 1e-4)
        )[0]
        self.wafer_top_xy = dof_xyz[self.wafer_top_idx, :2].copy()
        self.esc_top_xy = dof_xyz[self.esc_top_idx, :2].copy()

    def solve(self, cond: ESC3DOperatingCondition) -> dict:
        """Solve the steady-state 3D coupled thermal field."""
        if len(cond.heater_powers_W) != N_HEATER_ZONES:
            raise ValueError(f"Expected {N_HEATER_ZONES} heater powers, got {len(cond.heater_powers_W)}")
        if cond.r_leak_m + cond.w_smooth_m > self.geom.r_wafer + 1e-9:
            raise ValueError("Leak transition violates r_leak_m + w_smooth_m <= R_wafer!")

        t0 = time.perf_counter()
        T_chiller_K = cond.T_chiller_C + 273.15
        T_wall_K = cond.T_wall_C + 273.15
        k_cer = MATERIALS[cond.ceramic_type].k_ref
        k_al = MATERIALS["Al"].k_ref
        k_si_ref = MATERIALS["Si"].k_ref
        eps_si = MATERIALS["Si"].emissivity

        # 1. Assign anisotropic thermal conductivities (k_xy, k_z) in DG0
        k_xy_dg = fem.Function(self.W_dg)
        k_z_dg = fem.Function(self.W_dg)

        k_xy_dg.x.array[self.tag_to_cells[TAG_AL_BASE]] = k_al
        k_z_dg.x.array[self.tag_to_cells[TAG_AL_BASE]] = k_al

        for t in (TAG_CERAMIC_BULK,) + self.heater_tags:
            k_xy_dg.x.array[self.tag_to_cells[t]] = k_cer
            k_z_dg.x.array[self.tag_to_cells[t]] = k_cer

        gap_cells = self.tag_to_cells[TAG_HE_GAP]
        gx = self.cell_mid[gap_cells, 0]
        gy = self.cell_mid[gap_cells, 1]
        p_gap = eval_he_pressure_3d(gx, gy, cond)
        if abs(cond.d_gap_warpage_m) > 1e-15:
            d_gap_eval = cond.d_he_gap_m + cond.d_gap_warpage_m * (np.hypot(gx, gy) / self.geom.r_wafer) ** 2
        else:
            d_gap_eval = cond.d_he_gap_m
        h_gap = h_eff_interface(
            P_torr=p_gap,
            d_gap=d_gap_eval,
            T_g=315.0,
            h_solid=cond.h_solid_contact,
            alpha_T1=cond.alpha_T,
            alpha_T2=cond.alpha_T,
        )
        k_xy_dg.x.array[gap_cells] = 1e-6
        k_z_dg.x.array[gap_cells] = h_gap * self.geom.t_gap_mesh

        waf_cells = self.tag_to_cells[TAG_WAFER]
        k_xy_dg.x.array[waf_cells] = k_si_ref
        k_z_dg.x.array[waf_cells] = k_si_ref

        # 2. Heater volumetric heat generation (exact discrete power preservation)
        q_vol_dg = fem.Function(self.W_dg)
        q_vol_dg.x.array[:] = 0.0
        for h_tag, P_z, V_z in zip(self.heater_tags, cond.heater_powers_W, self.heater_mesh_volumes):
            q_vol_dg.x.array[self.tag_to_cells[h_tag]] = P_z / V_z

        # 3. Area-normalized 2D plasma heat flux on wafer top
        q_plasma_fn = fem.Function(self.V)
        q_plasma_fn.interpolate(lambda x: eval_plasma_flux_3d_unnorm(x[0], x[1], cond, self.geom.r_wafer))
        raw_int = float(fem.assemble_scalar(fem.form(q_plasma_fn * self.ds(BND_WAFER_TOP))))
        if raw_int > 0.0:
            q_plasma_fn.x.array[:] *= cond.total_plasma_power_W / raw_int

        # 4. Variational form
        v = ufl.TestFunction(self.V)
        non_wafer_tags = (TAG_AL_BASE, TAG_CERAMIC_BULK, TAG_HE_GAP) + self.heater_tags
        dx_non_wafer = sum((self.dx(t) for t in non_wafer_tags[1:]), self.dx(non_wafer_tags[0]))
        dx_heaters = sum((self.dx(t) for t in self.heater_tags[1:]), self.dx(self.heater_tags[0]))

        def aniso_flux(u_expr):
            return (
                k_xy_dg * (u_expr.dx(0) * v.dx(0) + u_expr.dx(1) * v.dx(1))
                + k_z_dg * u_expr.dx(2) * v.dx(2)
            )

        if not cond.nonlinear_k_si and not cond.include_radiation:
            u = ufl.TrialFunction(self.V)
            a = aniso_flux(u) * self.dx + cond.h_chiller * u * v * self.ds(BND_CHILLER_BOTTOM)
            L = (
                q_vol_dg * v * dx_heaters
                + q_plasma_fn * v * self.ds(BND_WAFER_TOP)
                + cond.h_chiller * T_chiller_K * v * self.ds(BND_CHILLER_BOTTOM)
            )
            problem = LinearProblem(
                a,
                L,
                bcs=[],
                petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-11},
                petsc_options_prefix="esc3d_lin_",
            )
            uh = problem.solve()
        else:
            uh = fem.Function(self.V)
            uh.x.array[:] = T_chiller_K + 35.0
            k_si_ufl = k_si_ref * (300.0 / uh) ** 1.3 if cond.nonlinear_k_si else k_si_ref

            F = (
                aniso_flux(uh) * dx_non_wafer
                + k_si_ufl * ufl.inner(ufl.grad(uh), ufl.grad(v)) * self.dx(TAG_WAFER)
                + cond.h_chiller * (uh - T_chiller_K) * v * self.ds(BND_CHILLER_BOTTOM)
                - q_vol_dg * v * dx_heaters
                - q_plasma_fn * v * self.ds(BND_WAFER_TOP)
            )
            if cond.include_radiation:
                q_rad = eps_si * STEFAN_BOLTZMANN * (uh ** 4 - T_wall_K ** 4)
                F += q_rad * v * (self.ds(BND_WAFER_TOP) + self.ds(BND_WAFER_EDGE))

            problem = NonlinearProblem(
                F,
                uh,
                bcs=[],
                petsc_options={
                    "snes_type": "newtonls",
                    "snes_linesearch_type": "bt",
                    "snes_rtol": 1e-10,
                    "snes_atol": 1e-10,
                    "snes_max_it": 20,
                    "ksp_type": "cg",
                    "pc_type": "gamg",
                    "ksp_rtol": 1e-10,
                },
                petsc_options_prefix="esc3d_nl_",
            )
            problem.solve()

        solve_time_s = time.perf_counter() - t0

        # 5. Global energy balance verification
        Q_plasma_W = float(fem.assemble_scalar(fem.form(q_plasma_fn * self.ds(BND_WAFER_TOP))))
        Q_heaters_W = float(fem.assemble_scalar(fem.form(q_vol_dg * dx_heaters)))
        Q_cool_W = float(fem.assemble_scalar(fem.form(cond.h_chiller * (uh - T_chiller_K) * self.ds(BND_CHILLER_BOTTOM))))
        if cond.include_radiation:
            q_rad_f = eps_si * STEFAN_BOLTZMANN * (uh ** 4 - T_wall_K ** 4)
            Q_rad_top_W = float(fem.assemble_scalar(fem.form(q_rad_f * self.ds(BND_WAFER_TOP))))
            Q_rad_edge_W = float(fem.assemble_scalar(fem.form(q_rad_f * self.ds(BND_WAFER_EDGE))))
        else:
            Q_rad_top_W = 0.0
            Q_rad_edge_W = 0.0
        imbalance_W = (Q_plasma_W + Q_heaters_W) - (Q_cool_W + Q_rad_top_W + Q_rad_edge_W)

        T_top_C = uh.x.array[self.wafer_top_idx] - 273.15
        T_bot_C = uh.x.array[self.wafer_bot_idx] - 273.15
        T_esc_C = uh.x.array[self.esc_top_idx] - 273.15

        r_top = np.hypot(self.wafer_top_xy[:, 0], self.wafer_top_xy[:, 1])
        eff_mask = r_top <= 0.147 + 1e-6
        T_eff = T_top_C[eff_mask]

        return {
            "wafer_top_xy": self.wafer_top_xy,
            "esc_top_xy": self.esc_top_xy,
            "T_wafer_top_C": T_top_C,
            "T_wafer_bot_C": T_bot_C,
            "T_esc_top_C": T_esc_C,
            "T_mean_eff_C": float(np.mean(T_eff)),
            "dT_max_min_eff_C": float(np.max(T_eff) - np.min(T_eff)),
            "T_3sigma_eff_C": float(3.0 * np.std(T_eff)),
            "Q_plasma_W": Q_plasma_W,
            "Q_heaters_W": Q_heaters_W,
            "Q_cool_W": Q_cool_W,
            "Q_rad_top_W": Q_rad_top_W,
            "Q_rad_edge_W": Q_rad_edge_W,
            "energy_imbalance_W": imbalance_W,
            "solve_time_s": solve_time_s,
            "n_dofs": int(self.V.dofmap.index_map.size_global),
        }
