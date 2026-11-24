"""3D Day 1 Benchmark: 3D FEniCSx Multi-Zone ESC Solver Verification & Visualization.

Verifies:
  1. Conformal 3D mesh import into FEniCSx 0.11.0 (coarse & medium)
  2. Global 3D energy balance |Q_in - Q_out|
  3. Wall-clock solve times (linear unit-load & nonlinear coupled)
  4. 300 mm wafer 2D top-surface colormap across:
       (a) Nominal operation
       (b) Localized lower-left edge He seal leak (theta=225°, beta_leak=0.25)
       (c) Localized azimuthal edge heater compensation (Zones 7-8 reduced, P_out locked)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fem3d.esc_3d_fem_solver import ESC3DFEMSolver, ESC3DOperatingCondition


def main() -> None:
    print("=" * 84)
    print(" [3D Day 1 Benchmark] 3D Multi-Zone ESC (10 Heaters) FEniCSx Thermal Solver")
    print("=" * 84)

    summary = {}
    demo_results = {}

    for ref in ("coarse", "medium"):
        solver = ESC3DFEMSolver(refinement=ref, degree=1)

        # Case 1: Nominal balanced 10-zone recipe
        cond_nom = ESC3DOperatingCondition(
            ceramic_type="Al2O3",
            total_plasma_power_W=500.0,
            plasma_peaking_factor=0.25,
            p_he_in_torr=12.0,
            p_he_out_torr=12.0,
            beta_leak=1.0,
            heater_powers_W=(80.0, 200.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0),
        )
        # Warmup + timed nonlinear
        res_nom = solver.solve(cond_nom)

        # Case 2: Localized lower-left edge He seal leak (theta=225°, beta_leak=0.25 -> 3 Torr at edge)
        cond_leak = ESC3DOperatingCondition(
            ceramic_type="Al2O3",
            total_plasma_power_W=500.0,
            plasma_peaking_factor=0.25,
            p_he_in_torr=12.0,
            p_he_out_torr=12.0,
            beta_leak=0.25,
            r_leak_m=0.138,
            leak_theta_center_deg=225.0,
            leak_theta_halfwidth_deg=35.0,
            heater_powers_W=(80.0, 200.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0),
        )
        res_leak = solver.solve(cond_leak)

        # Case 3: Manual azimuthal compensation (turn down Sectors 4 & 5 covering 180-270°, P_out locked!)
        cond_comp = ESC3DOperatingCondition(
            ceramic_type="Al2O3",
            total_plasma_power_W=500.0,
            plasma_peaking_factor=0.25,
            p_he_in_torr=12.0,
            p_he_out_torr=12.0,
            beta_leak=0.25,
            r_leak_m=0.138,
            leak_theta_center_deg=225.0,
            leak_theta_halfwidth_deg=35.0,
            heater_powers_W=(80.0, 200.0, 55.0, 55.0, 55.0, 55.0, 15.0, 15.0, 55.0, 55.0),
        )
        res_comp = solver.solve(cond_comp)

        # Linear solve timing (Mode A unit-load proxy)
        cond_lin = ESC3DOperatingCondition(
            ceramic_type="Al2O3",
            total_plasma_power_W=500.0,
            include_radiation=False,
            nonlinear_k_si=False,
        )
        res_lin = solver.solve(cond_lin)

        hotspot_spike = float(np.max(res_leak["T_wafer_top_C"] - res_nom["T_wafer_top_C"]))
        summary[ref] = {
            "n_dofs": res_nom["n_dofs"],
            "lin_solve_s": res_lin["solve_time_s"],
            "nonlin_solve_s": res_leak["solve_time_s"],
            "nom_energy_imbalance_W": res_nom["energy_imbalance_W"],
            "leak_energy_imbalance_W": res_leak["energy_imbalance_W"],
            "nom_dT_eff_C": res_nom["dT_max_min_eff_C"],
            "leak_dT_eff_C": res_leak["dT_max_min_eff_C"],
            "comp_dT_eff_C": res_comp["dT_max_min_eff_C"],
            "local_hotspot_spike_C": hotspot_spike,
        }
        print(
            f" Mesh={ref:>6} | DOFs={res_nom['n_dofs']:6d} | "
            f"Lin={res_lin['solve_time_s']*1e3:6.1f}ms | Nonlin={res_leak['solve_time_s']*1e3:6.1f}ms | "
            f"|Imbal|={abs(res_leak['energy_imbalance_W']):.2e}W | "
            f"Spike=+{hotspot_spike:.2f}°C | dT_eff: {res_nom['dT_max_min_eff_C']:.2f} -> "
            f"{res_leak['dT_max_min_eff_C']:.2f} -> {res_comp['dT_max_min_eff_C']:.2f}°C"
        )
        if ref == "medium":
            demo_results = {"nom": res_nom, "leak": res_leak, "comp": res_comp}

    print("=" * 84)
    out_dir = PROJECT_ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "day1_3d_fem_benchmark.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot 3-panel 300 mm wafer top-surface temperature colormap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    panels = [
        ("nom", "Nominal (Uniform 12 Torr He)"),
        ("leak", "Localized Edge Leak (225°, 3 Torr)"),
        ("comp", "Azimuthal Heater Comp (Sectors 4-5↓)"),
    ]
    all_T = np.concatenate([demo_results[k]["T_wafer_top_C"] for k, _ in panels])
    vmin, vmax = float(np.percentile(all_T, 1)), float(np.max(all_T))

    for ax, (key, title) in zip(axes, panels):
        res = demo_results[key]
        x_mm = res["wafer_top_xy"][:, 0] * 1e3
        y_mm = res["wafer_top_xy"][:, 1] * 1e3
        T_c = res["T_wafer_top_C"]
        tri = mtri.Triangulation(x_mm, y_mm)
        cf = ax.tricontourf(tri, T_c, levels=28, cmap="inferno", vmin=vmin, vmax=vmax)
        circle = plt.Circle((0, 0), 150.0, fill=False, color="white", lw=1.2)
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlim(-155, 155)
        ax.set_ylim(-155, 155)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(
            f"{title}\n"
            f"Mean={res['T_mean_eff_C']:.2f}°C, ΔT={res['dT_max_min_eff_C']:.2f}°C, 3σ={res['T_3sigma_eff_C']:.2f}°C",
            fontsize=10,
        )
    fig.colorbar(cf, ax=axes, label="Wafer Top Temperature [°C]", shrink=0.9)
    png_path = out_dir / "day1_3d_wafer_demo.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f" Saved JSON -> {json_path}")
    print(f" Saved Plot -> {png_path}")


if __name__ == "__main__":
    main()
