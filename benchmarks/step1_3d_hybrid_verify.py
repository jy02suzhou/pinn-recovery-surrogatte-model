"""Step 1-B Verification: 3D Hybrid (Mode A Linear ESC + PyTorch 2D Polar Wafer FVM) vs 3D FEniCSx FEM.

Verifies on `medium` (and `coarse`) 3D meshes:
  1. Field accuracy vs 3D FEniCSx FEM on Nominal and Localized 225° Edge Leak (beta_leak=0.25):
       - Max & MAE error on T_wafer (interpolated onto Polar FVM centers) and T_esc_top
  2. Conservation laws:
       - Wafer energy imbalance < 1e-10 W
       - Independent ESC-to-chiller energy imbalance < 1e-6 W
  3. Gradient verification (on Localized 225° Edge Leak):
       (a) Implementation check: Adjoint vs same-FVM central FD on
           (Q_1, Q_7, p_out, beta_leak, Q_plasma, alpha_peak) -> rel err < 1e-4 (or abs < 1e-6)
       (b) Physics sensitivity validation: FVM Adjoint vs 3D FEniCSx FEM central FD on
           (Q_1, Q_7, p_out, beta_leak) -> report relative error (%)
  4. Wall-clock timing:
       - Median & p95 inference time (batch 1, 20 runs after warmup) vs 3D FEM solve time (speedup)
  5. Saves full JSON report to results/benchmarks/step1_3d_hybrid_verify.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fem3d.esc_3d_fem_solver import ESC3DFEMSolver, ESC3DOperatingCondition
from src.solvers.coupled_3d_wafer_fvm import Coupled3DWaferFVMSolver, SolveOptions


def interp_2d_to_fvm(xy_src: np.ndarray, val_src: np.ndarray, xx_dst: np.ndarray, yy_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate scattered 2D FEM surface values onto Polar FVM cell centers."""
    pts_dst = np.column_stack([xx_dst.ravel(), yy_dst.ravel()])
    lin = LinearNDInterpolator(xy_src, val_src)
    out = lin(pts_dst)
    nan_mask = np.isnan(out)
    if np.any(nan_mask):
        nn = NearestNDInterpolator(xy_src, val_src)
        out[nan_mask] = nn(pts_dst[nan_mask])
    return out


def make_cond(
    Q_heaters: tuple[float, ...],
    p_in: float = 12.0,
    p_out: float = 12.0,
    beta_leak: float = 1.0,
    Q_plasma: float = 500.0,
    alpha_peak: float = 0.25,
) -> ESC3DOperatingCondition:
    return ESC3DOperatingCondition(
        ceramic_type="Al2O3",
        total_plasma_power_W=Q_plasma,
        plasma_peaking_factor=alpha_peak,
        p_he_in_torr=p_in,
        p_he_out_torr=p_out,
        beta_leak=beta_leak,
        r_leak_m=0.138,
        leak_theta_center_deg=225.0,
        leak_theta_halfwidth_deg=35.0,
        leak_theta_smooth_deg=10.0,
        heater_powers_W=Q_heaters,
    )


def scalar_objective_torch(T_wafer_C: torch.Tensor, cell_areas_flat: torch.Tensor, T_target_C: float = 55.0) -> torch.Tensor:
    """Area-weighted MSE loss against T_target_C [°C^2] (sensitive to both mean and spatial profile)."""
    w = cell_areas_flat / torch.sum(cell_areas_flat)
    return torch.sum(w * (T_wafer_C - T_target_C).pow(2))


def scalar_objective_np(T_wafer_C: np.ndarray, cell_areas_flat: np.ndarray, T_target_C: float = 55.0) -> float:
    w = cell_areas_flat / np.sum(cell_areas_flat)
    return float(np.sum(w * (T_wafer_C - T_target_C) ** 2))


def verify_refinement(refinement: str) -> dict:
    print(f"\n{'=' * 86}")
    print(f" [Step 1-B] 3D Hybrid (Mode A + 2D Polar FVM) vs 3D FEniCSx FEM | mesh = {refinement}")
    print(f"{'=' * 86}")

    fem_solver = ESC3DFEMSolver(refinement=refinement, degree=1)
    fvm_solver = Coupled3DWaferFVMSolver(refinement=refinement, ceramic_type="Al2O3", device="cpu")
    opts = SolveOptions()

    xx_np = fvm_solver.xx_flat.cpu().numpy()
    yy_np = fvm_solver.yy_flat.cpu().numpy()
    areas_np = fvm_solver.cell_areas_flat.cpu().numpy()

    Q_nom = (80.0, 200.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0)

    # 1. Field accuracy & conservation on Nominal and Localized 225° Edge Leak (beta_leak=0.25)
    cases = {
        "nominal": {"beta_leak": 1.0},
        "edge_leak_225": {"beta_leak": 0.25},
    }
    case_reports: dict[str, dict] = {}

    for cname, cfg in cases.items():
        cond = make_cond(Q_nom, beta_leak=cfg["beta_leak"])
        res_fem = fem_solver.solve(cond)
        with torch.no_grad():
            res_fvm = fvm_solver.solve(
                Q_heaters=Q_nom,
                p_in=12.0,
                p_out=12.0,
                beta_leak=cfg["beta_leak"],
                Q_plasma=500.0,
                alpha_peak=0.25,
                opts=opts,
            )

        T_waf_fem_on_fvm = interp_2d_to_fvm(res_fem["wafer_top_xy"], res_fem["T_wafer_top_C"], xx_np, yy_np)
        T_esc_fem_on_fvm = interp_2d_to_fvm(res_fem["esc_top_xy"], res_fem["T_esc_top_C"], xx_np, yy_np)

        T_waf_fvm = res_fvm["T_wafer_C"].cpu().numpy()
        T_esc_fvm = res_fvm["T_esc_top_C"].cpu().numpy()

        err_waf = np.abs(T_waf_fvm - T_waf_fem_on_fvm)
        err_esc = np.abs(T_esc_fvm - T_esc_fem_on_fvm)

        waf_imbal = abs(float(res_fvm["wafer_imbalance_W"].item()))
        esc_imbal = abs(float(res_fvm["esc_imbalance_W"].item()))
        glob_imbal = abs(float(res_fvm["global_imbalance_W"].item()))

        case_reports[cname] = {
            "T_wafer_max_err_C": float(np.max(err_waf)),
            "T_wafer_mae_C": float(np.mean(err_waf)),
            "T_esc_top_max_err_C": float(np.max(err_esc)),
            "T_esc_top_mae_C": float(np.mean(err_esc)),
            "wafer_imbalance_W": waf_imbal,
            "esc_imbalance_W": esc_imbal,
            "global_imbalance_W": glob_imbal,
            "Q_cool_fvm_W": float(res_fvm["Q_cool_W"].item()),
            "Q_cool_fem_W": float(res_fem["Q_cool_W"]),
            "fem_solve_time_s": float(res_fem["solve_time_s"]),
            "fem_n_dofs": int(res_fem["n_dofs"]),
        }
        print(
            f" [{cname:13s}] T_waf max={np.max(err_waf):.4f}°C (MAE={np.mean(err_waf):.4f}°C) | "
            f"T_esc max={np.max(err_esc):.4f}°C (MAE={np.mean(err_esc):.4f}°C) | "
            f"|imbal_waf|={waf_imbal:.2e}W | |imbal_esc|={esc_imbal:.2e}W"
        )

    # 2. Gradcheck (a): Adjoint vs same-FVM central FD on Localized 225° Edge Leak
    Q_t = torch.tensor(Q_nom, dtype=torch.float64, requires_grad=True)
    p_in_t = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
    p_out_t = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
    beta_t = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    Q_pl_t = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)
    alpha_t = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    res_ad = fvm_solver.solve(Q_t, p_in_t, p_out_t, beta_t, Q_pl_t, alpha_t, opts=opts)
    loss_ad = scalar_objective_torch(res_ad["T_wafer_C"], fvm_solver.cell_areas_flat)
    loss_ad.backward()

    adjoint_grads = {
        "Q_1": float(Q_t.grad[0].item()),
        "Q_7": float(Q_t.grad[6].item()),
        "p_out": float(p_out_t.grad.item()),
        "beta_leak": float(beta_t.grad.item()),
        "Q_plasma": float(Q_pl_t.grad.item()),
        "alpha_peak": float(alpha_t.grad.item()),
    }

    def eval_fvm_loss(Q_vec, p_in=12.0, p_out=12.0, beta=0.25, Q_pl=500.0, alpha=0.25) -> float:
        with torch.no_grad():
            r = fvm_solver.solve(Q_vec, p_in, p_out, beta, Q_pl, alpha, opts=opts)
            return float(scalar_objective_torch(r["T_wafer_C"], fvm_solver.cell_areas_flat).item())

    fvm_fd_specs = [
        ("Q_1", 1e-4),
        ("Q_7", 1e-4),
        ("p_out", 1e-4),
        ("beta_leak", 1e-5),
        ("Q_plasma", 1e-3),
        ("alpha_peak", 1e-5),
    ]
    gradcheck_impl: dict[str, dict] = {}
    all_impl_pass = True

    print("\n [Gradcheck (a)] Adjoint vs same-FVM Central FD (Localized 225° Edge Leak):")
    for name, eps in fvm_fd_specs:
        if name == "Q_1":
            qp, qm = list(Q_nom), list(Q_nom)
            qp[0] += eps
            qm[0] -= eps
            fd = (eval_fvm_loss(qp) - eval_fvm_loss(qm)) / (2.0 * eps)
        elif name == "Q_7":
            qp, qm = list(Q_nom), list(Q_nom)
            qp[6] += eps
            qm[6] -= eps
            fd = (eval_fvm_loss(qp) - eval_fvm_loss(qm)) / (2.0 * eps)
        elif name == "p_out":
            fd = (eval_fvm_loss(Q_nom, p_out=12.0 + eps) - eval_fvm_loss(Q_nom, p_out=12.0 - eps)) / (2.0 * eps)
        elif name == "beta_leak":
            fd = (eval_fvm_loss(Q_nom, beta=0.25 + eps) - eval_fvm_loss(Q_nom, beta=0.25 - eps)) / (2.0 * eps)
        elif name == "Q_plasma":
            fd = (eval_fvm_loss(Q_nom, Q_pl=500.0 + eps) - eval_fvm_loss(Q_nom, Q_pl=500.0 - eps)) / (2.0 * eps)
        elif name == "alpha_peak":
            fd = (eval_fvm_loss(Q_nom, alpha=0.25 + eps) - eval_fvm_loss(Q_nom, alpha=0.25 - eps)) / (2.0 * eps)
        else:
            raise ValueError(name)

        ad = adjoint_grads[name]
        abs_err = abs(ad - fd)
        rel_err = abs_err / max(abs(fd), 1e-12)
        ok = (rel_err < 1e-4) or (abs_err < 1e-6)
        all_impl_pass = all_impl_pass and ok
        gradcheck_impl[name] = {
            "adjoint": ad,
            "fvm_fd": fd,
            "abs_err": abs_err,
            "rel_err": rel_err,
            "pass": ok,
        }
        print(
            f"   {name:10s} | Adjoint={ad:+.8e} | FVM_FD={fd:+.8e} | "
            f"rel={rel_err:.2e} | {'PASS' if ok else 'FAIL'}"
        )

    # 3. Gradcheck (b): FVM Adjoint vs 3D FEniCSx FEM central FD on (Q_1, Q_7, p_out, beta_leak)
    def eval_fem_loss(Q_vec, p_out=12.0, beta=0.25) -> float:
        r = fem_solver.solve(make_cond(tuple(Q_vec), p_out=p_out, beta_leak=beta))
        T_on_fvm = interp_2d_to_fvm(r["wafer_top_xy"], r["T_wafer_top_C"], xx_np, yy_np)
        return scalar_objective_np(T_on_fvm, areas_np)

    fem_fd_specs = [
        ("Q_1", 1.0),
        ("Q_7", 1.0),
        ("p_out", 0.2),
        ("beta_leak", 0.01),
    ]
    gradcheck_fem: dict[str, dict] = {}
    print("\n [Gradcheck (b)] FVM Adjoint vs 3D FEniCSx FEM Central FD:")
    for name, eps in fem_fd_specs:
        if name == "Q_1":
            qp, qm = list(Q_nom), list(Q_nom)
            qp[0] += eps
            qm[0] -= eps
            fd = (eval_fem_loss(qp) - eval_fem_loss(qm)) / (2.0 * eps)
        elif name == "Q_7":
            qp, qm = list(Q_nom), list(Q_nom)
            qp[6] += eps
            qm[6] -= eps
            fd = (eval_fem_loss(qp) - eval_fem_loss(qm)) / (2.0 * eps)
        elif name == "p_out":
            fd = (eval_fem_loss(Q_nom, p_out=12.0 + eps) - eval_fem_loss(Q_nom, p_out=12.0 - eps)) / (2.0 * eps)
        elif name == "beta_leak":
            fd = (eval_fem_loss(Q_nom, beta=0.25 + eps) - eval_fem_loss(Q_nom, beta=0.25 - eps)) / (2.0 * eps)
        else:
            raise ValueError(name)

        ad = adjoint_grads[name]
        rel_pct = 100.0 * abs(ad - fd) / max(abs(fd), 1e-12)
        gradcheck_fem[name] = {
            "fvm_adjoint": ad,
            "fem_fd": fd,
            "rel_err_pct": rel_pct,
        }
        print(f"   {name:10s} | FVM_Adj={ad:+.6e} | 3D_FEM_FD={fd:+.6e} | rel={rel_pct:.3f}%")

    # 4. Timing: median & p95 inference time (batch 1, 20 runs after warmup) vs 3D FEM
    Q_bench = torch.tensor(Q_nom, dtype=torch.float64)
    p_in_b = torch.tensor(12.0, dtype=torch.float64)
    p_out_b = torch.tensor(12.0, dtype=torch.float64)
    beta_b = torch.tensor(0.25, dtype=torch.float64)
    Q_pl_b = torch.tensor(500.0, dtype=torch.float64)
    alpha_b = torch.tensor(0.25, dtype=torch.float64)

    with torch.no_grad():
        for _ in range(3):
            fvm_solver.solve(Q_bench, p_in_b, p_out_b, beta_b, Q_pl_b, alpha_b, opts=opts)

        times_s = []
        for _ in range(20):
            t0 = time.perf_counter()
            fvm_solver.solve(Q_bench, p_in_b, p_out_b, beta_b, Q_pl_b, alpha_b, opts=opts)
            times_s.append(time.perf_counter() - t0)

    med_ms = float(np.median(times_s) * 1e3)
    p95_ms = float(np.percentile(times_s, 95) * 1e3)
    fem_ms = float(case_reports["edge_leak_225"]["fem_solve_time_s"] * 1e3)
    speedup = fem_ms / med_ms

    print(
        f"\n [Timing] Hybrid FVM median={med_ms:.2f} ms, p95={p95_ms:.2f} ms | "
        f"3D FEM={fem_ms:.1f} ms | Speedup={speedup:.1f}x"
    )

    energy_pass = all(
        (rep["wafer_imbalance_W"] < 1e-10) and (rep["esc_imbalance_W"] < 1e-6)
        for rep in case_reports.values()
    )
    overall_pass = bool(energy_pass and all_impl_pass)

    return {
        "refinement": refinement,
        "cases": case_reports,
        "gradcheck_impl": gradcheck_impl,
        "gradcheck_fem": gradcheck_fem,
        "timing": {
            "fvm_median_ms": med_ms,
            "fvm_p95_ms": p95_ms,
            "fem_solve_ms": fem_ms,
            "speedup": speedup,
            "n_runs": 20,
        },
        "checks": {
            "energy_balance_pass": energy_pass,
            "gradcheck_impl_pass": all_impl_pass,
            "overall_pass": overall_pass,
        },
    }


def main() -> None:
    report = {}
    for ref in ("coarse", "medium"):
        report[ref] = verify_refinement(ref)

    out_dir = PROJECT_ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step1_3d_hybrid_verify.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n Saved verification report -> {out_path}")
    all_ok = all(r["checks"]["overall_pass"] for r in report.values())
    print(f" OVERALL STATUS: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
