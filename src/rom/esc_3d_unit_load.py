"""Mode A (3D): Exact Linear 3D ESC Unit-Load Response Extractor (FEniCSx 0.11.0).

Extracts the exact linear thermal response of the 3D ESC sub-domain
(Al Cooling Base + ESC Ceramic + 10 Embedded Heaters, WITHOUT wafer/He gap)
by assembling the 3D ESC stiffness matrix ONCE and solving for:
  - 10 unit heater loads (Q_i = 1 W, i = 1..10)
  - M = N_rb + (N_rb - 1) * 2 * K_max polar Fourier-Radial interface flux basis
    functions psi_alpha(r, theta) applied as Neumann load on BND_ESC_TOP_INTERFACE

For every load, records:
  1. ESC top surface temperature T_ESC(r_j, theta_l) on the 2D Polar FVM grid
  2. Independent bottom chiller Robin cooling integral Q_cool [W] over r <= 160 mm
     (so global ESC-to-chiller energy balance is verified independently!)

Results are cached to `.npz` so subsequent PyTorch runs load in < 10 ms.
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

from src.fem3d.build_esc_3d_mesh import (
    BND_CHILLER_BOTTOM,
    BND_ESC_TOP_INTERFACE,
    ESC3DGeometrySpec,
    N_EDGE_SECTORS,
    N_HEATER_ZONES,
    TAG_AL_BASE,
    TAG_CERAMIC_BULK,
    TAG_HEATER_EDGE_START,
    TAG_HEATER_Z1,
    TAG_HEATER_Z2,
    build_esc_3d_gmsh,
)
from src.physics.materials_and_he import MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_polar_fvm_grid(
    n_r: int = 24,
    n_theta: int = 36,
    r_wafer: float = 0.150,
) -> dict[str, np.ndarray]:
    """Build non-uniform radial (edge-clustered) x uniform angular 2D Polar FVM grid."""
    # Clustered radial faces: coarse [0, 105] mm, medium [105, 130] mm, fine [130, 150] mm
    n1 = max(int(round(n_r * 0.35)), 4)
    n2 = max(int(round(n_r * 0.25)), 4)
    n3 = n_r - n1 - n2
    r_faces = np.concatenate([
        np.linspace(0.0, 0.105, n1, endpoint=False),
        np.linspace(0.105, 0.132, n2, endpoint=False),
        np.linspace(0.132, r_wafer, n3 + 1),
    ])
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    dr = r_faces[1:] - r_faces[:-1]

    dtheta = 2.0 * np.pi / n_theta
    theta_centers = (np.arange(n_theta) + 0.5) * dtheta

    # 2D flattened cell coordinates (row-major: j * n_theta + l)
    rr, tt = np.meshgrid(r_centers, theta_centers, indexing="ij")
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)

    # Exact cylindrical cell areas A_{j,l} = 0.5 * (r_{j+1/2}^2 - r_{j-1/2}^2) * dtheta
    area_r = 0.5 * (r_faces[1:] ** 2 - r_faces[:-1] ** 2) * dtheta
    cell_areas = np.repeat(area_r[:, None], n_theta, axis=1)

    return {
        "n_r": n_r,
        "n_theta": n_theta,
        "r_faces": r_faces,
        "r_centers": r_centers,
        "dr": dr,
        "dtheta": dtheta,
        "theta_centers": theta_centers,
        "rr": rr,
        "tt": tt,
        "xx": xx,
        "yy": yy,
        "cell_areas": cell_areas,
    }


def make_radial_basis_knots(r_wafer: float = 0.150, n_knots: int = 10) -> np.ndarray:
    """Non-uniform radial hat-function knots clustered near r_split=110mm and edge [130,150]mm."""
    if n_knots == 10:
        return np.array([0.0, 0.040, 0.080, 0.105, 0.115, 0.128, 0.136, 0.141, 0.146, r_wafer])
    n_in = max(n_knots // 2, 3)
    n_out = n_knots - n_in
    return np.concatenate([
        np.linspace(0.0, 0.125, n_in, endpoint=False),
        np.linspace(0.125, r_wafer, n_out),
    ])


def eval_hat_1d(r: np.ndarray, knots: np.ndarray, idx: int) -> np.ndarray:
    """Piecewise-linear hat function centered at knots[idx] (partition of unity on [0, R_w])."""
    val = np.zeros_like(r, dtype=float)
    if idx > 0:
        left = knots[idx - 1]
        mid = knots[idx]
        m = (r >= left) & (r <= mid)
        val[m] = (r[m] - left) / (mid - left)
    if idx < len(knots) - 1:
        mid = knots[idx]
        right = knots[idx + 1]
        m = (r >= mid) & (r <= right)
        val[m] = (right - r[m]) / (right - mid)
    return val


def build_basis_descriptors(knots: np.ndarray, k_max: int = 4) -> list[tuple[int, int, str]]:
    """List of (radial_knot_idx, harmonic_k, trig_type) for Polar Fourier-Radial basis.

    Includes:
      - k = 0 (constant angular): all radial knots 0..n_knots-1 (spans constant 1 exactly!)
      - k = 1..k_max (cos, sin): radial knots 1..n_knots-1 (vanishes at r=0 for smoothness)
    """
    desc: list[tuple[int, int, str]] = []
    for m in range(len(knots)):
        desc.append((m, 0, "const"))
    for k in range(1, k_max + 1):
        for m in range(1, len(knots)):
            desc.append((m, k, "cos"))
            desc.append((m, k, "sin"))
    return desc


def eval_basis_fn(x: np.ndarray, y: np.ndarray, knots: np.ndarray, desc_item: tuple[int, int, str]) -> np.ndarray:
    m, k, trig = desc_item
    r = np.hypot(x, y)
    phi = eval_hat_1d(r, knots, m)
    if k == 0:
        return phi
    theta = np.arctan2(y, x)
    return phi * (np.cos(k * theta) if trig == "cos" else np.sin(k * theta))


def extract_esc_3d_unit_load_rom(
    refinement: str = "coarse",
    ceramic_type: str = "Al2O3",
    n_r: int = 24,
    n_theta: int = 36,
    n_knots: int = 10,
    k_max: int = 4,
    h_chiller: float = 2000.0,
    force_recompute: bool = False,
) -> Path:
    """Extract and cache the 3D linear ESC unit-load ROM (Mode A) to .npz."""
    cache_dir = PROJECT_ROOT / "results" / "rom"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"esc3d_rom_{ceramic_type}_{refinement}_nr{n_r}_nt{n_theta}_nk{n_knots}_km{k_max}.npz"
    if cache_path.exists() and not force_recompute:
        return cache_path

    t0 = time.perf_counter()
    geom = ESC3DGeometrySpec()
    msh_path = PROJECT_ROOT / "results" / "mesh" / f"esc_3d_{refinement}.msh"
    if not msh_path.exists():
        build_esc_3d_gmsh(refinement=refinement, msh_path=msh_path, geom=geom)

    mesh_data = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, 0, gdim=3)
    full_domain = mesh_data.mesh
    full_ct = mesh_data.cell_tags
    full_ft = mesh_data.facet_tags

    heater_tags = (TAG_HEATER_Z1, TAG_HEATER_Z2) + tuple(
        TAG_HEATER_EDGE_START + s for s in range(N_EDGE_SECTORS)
    )
    esc_vol_tags = (TAG_AL_BASE, TAG_CERAMIC_BULK) + heater_tags

    # Extract ESC-only 3D submesh (Al Base + Ceramic + 10 Heaters, NO wafer/gap)
    esc_cells = np.sort(np.concatenate([full_ct.find(t) for t in esc_vol_tags]))
    sub_domain, entity_map, _, _ = dmesh.create_submesh(full_domain, 3, esc_cells)

    # Transfer cell tags onto submesh (dolfinx 0.11.0 EntityMap API)
    sub_num_cells = sub_domain.topology.index_map(3).size_local
    sub_idx = np.arange(sub_num_cells, dtype=np.int32)
    parent_idx = (
        entity_map.sub_topology_to_topology(sub_idx, inverse=False)
        if hasattr(entity_map, "sub_topology_to_topology")
        else np.asarray(entity_map, dtype=np.int32)
    )
    parent_cell_vals = np.zeros(full_domain.topology.index_map(3).size_local, dtype=np.int32)
    parent_cell_vals[full_ct.indices] = full_ct.values
    sub_cell_vals = parent_cell_vals[parent_idx]
    sub_ct = dmesh.meshtags(sub_domain, 3, sub_idx, sub_cell_vals)

    # Identify boundary facets directly on submesh by geometry (z=0 chiller bottom, z=z_esc_top interface)
    sub_domain.topology.create_connectivity(2, 3)
    sub_domain.topology.create_connectivity(2, 0)
    z0, _, _, _, z_esc_top, _, _ = geom.z_levels
    f2c = sub_domain.topology.connectivity(2, 3)
    f2v = sub_domain.topology.connectivity(2, 0)
    coords = sub_domain.geometry.x

    chiller_facets = []
    esc_top_facets = []
    num_facets = sub_domain.topology.index_map(2).size_local
    for f in range(num_facets):
        if len(f2c.links(f)) != 1:
            continue  # interior facet
        verts = f2v.links(f)
        pts = coords[verts]
        zm, zM = pts[:, 2].min(), pts[:, 2].max()
        if abs(zm - z0) < 1e-5 and abs(zM - z0) < 1e-5:
            chiller_facets.append(f)
        elif abs(zm - z_esc_top) < 1e-5 and abs(zM - z_esc_top) < 1e-5:
            # Check centroid radius <= r_wafer (interface under wafer)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            if np.hypot(cx, cy) <= geom.r_wafer + 1e-4:
                esc_top_facets.append(f)

    facet_idx = np.array(chiller_facets + esc_top_facets, dtype=np.int32)
    facet_val = np.array(
        [BND_CHILLER_BOTTOM] * len(chiller_facets) + [BND_ESC_TOP_INTERFACE] * len(esc_top_facets),
        dtype=np.int32,
    )
    order = np.argsort(facet_idx)
    sub_ft = dmesh.meshtags(sub_domain, 2, facet_idx[order], facet_val[order])

    V = fem.functionspace(sub_domain, ("Lagrange", 1))
    W_dg = fem.functionspace(sub_domain, ("DG", 0))
    dx = ufl.Measure("dx", domain=sub_domain, subdomain_data=sub_ct)
    ds = ufl.Measure("ds", domain=sub_domain, subdomain_data=sub_ft)

    # Assign linear conductivities
    k_dg = fem.Function(W_dg)
    k_dg.x.array[sub_ct.find(TAG_AL_BASE)] = MATERIALS["Al"].k_ref
    k_cer = MATERIALS[ceramic_type].k_ref
    for t in (TAG_CERAMIC_BULK,) + heater_tags:
        k_dg.x.array[sub_ct.find(t)] = k_cer

    # Assemble linear ESC stiffness matrix A ONCE (T_chiller = 0 K shift; add T_chiller back by linearity)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = fem.form(k_dg * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + h_chiller * u * v * ds(BND_CHILLER_BOTTOM))
    A = assemble_matrix(a_form)
    A.assemble()

    # Use direct MUMPS LU factorization ONCE -> back-substitution per RHS takes milliseconds!
    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.setUp()

    # Build Polar FVM grid & evaluation points on ESC top surface (r <= R_w, z = z_esc_top)
    grid = make_polar_fvm_grid(n_r=n_r, n_theta=n_theta, r_wafer=geom.r_wafer)
    n_cells_2d = n_r * n_theta
    eval_pts = np.zeros((n_cells_2d, 3), dtype=float)
    eval_pts[:, 0] = grid["xx"].ravel()
    eval_pts[:, 1] = grid["yy"].ravel()
    eval_pts[:, 2] = z_esc_top - 1e-6  # slightly inside ceramic top for robust cell lookup

    # Precompute point-evaluation cell tree on submesh
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(sub_domain, 3)
    cell_candidates = geometry.compute_collisions_points(bb_tree, eval_pts)
    colliding_cells = geometry.compute_colliding_cells(sub_domain, cell_candidates, eval_pts)
    eval_cells = np.array([colliding_cells.links(i)[0] for i in range(n_cells_2d)], dtype=np.int32)

    # Also build ESC top full protection-domain evaluation points (r <= 155 mm)
    n_r_esc = n_r + 2
    r_esc_centers = np.concatenate([grid["r_centers"], [0.1515, 0.1540]])
    rr_e, tt_e = np.meshgrid(r_esc_centers, grid["theta_centers"], indexing="ij")
    esc_pts = np.zeros((n_r_esc * n_theta, 3), dtype=float)
    esc_pts[:, 0] = (rr_e * np.cos(tt_e)).ravel()
    esc_pts[:, 1] = (rr_e * np.sin(tt_e)).ravel()
    esc_pts[:, 2] = z_esc_top - 1e-6
    cand_e = geometry.compute_collisions_points(bb_tree, esc_pts)
    coll_e = geometry.compute_colliding_cells(sub_domain, cand_e, esc_pts)
    esc_eval_cells = np.array([coll_e.links(i)[0] for i in range(len(esc_pts))], dtype=np.int32)

    # Independent bottom chiller Robin cooling functional: Q_cool = int h_chiller * u_shift ds_bottom
    uh = fem.Function(V)
    cool_form = fem.form(h_chiller * uh * ds(BND_CHILLER_BOTTOM))

    # 1. Solve 10 unit heater loads (Q_i = 1 W)
    one_c = fem.Constant(sub_domain, 1.0)
    G_heaters = np.zeros((n_cells_2d, N_HEATER_ZONES), dtype=np.float64)
    G_heaters_esc_full = np.zeros((len(esc_pts), N_HEATER_ZONES), dtype=np.float64)
    c_heaters_cool = np.zeros(N_HEATER_ZONES, dtype=np.float64)

    for idx_h, h_tag in enumerate(heater_tags):
        V_h = float(fem.assemble_scalar(fem.form(one_c * dx(h_tag))))
        L_form = fem.form((1.0 / V_h) * v * dx(h_tag))
        b = assemble_vector(L_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        G_heaters[:, idx_h] = uh.eval(eval_pts, eval_cells).ravel()
        G_heaters_esc_full[:, idx_h] = uh.eval(esc_pts, esc_eval_cells).ravel()
        c_heaters_cool[idx_h] = float(fem.assemble_scalar(cool_form))
        b.destroy()

    # 2. Solve M Polar Fourier-Radial interface flux basis functions psi_alpha(r, theta)
    knots = make_radial_basis_knots(geom.r_wafer, n_knots=n_knots)
    desc = build_basis_descriptors(knots, k_max=k_max)
    M = len(desc)

    x_flat = grid["xx"].ravel()
    y_flat = grid["yy"].ravel()
    w_flat = grid["cell_areas"].ravel()
    Psi = np.column_stack([eval_basis_fn(x_flat, y_flat, knots, item) for item in desc])  # (N, M)
    I_fvm = w_flat @ Psi  # (M,) exact FVM area integral of each basis function

    T_basis = np.zeros((n_cells_2d, M), dtype=np.float64)
    T_basis_esc_full = np.zeros((len(esc_pts), M), dtype=np.float64)
    c_basis_cool = np.zeros(M, dtype=np.float64)
    psi_fn = fem.Function(V)
    psi_rad_cache: dict[int, tuple[np.ndarray, float]] = {}

    for alpha, d_item in enumerate(desc):
        m_idx, k_harm, _ = d_item
        psi_fn.interpolate(lambda x, item=d_item: eval_basis_fn(x[0], x[1], knots, item))
        I_fem = float(fem.assemble_scalar(fem.form(psi_fn * ds(BND_ESC_TOP_INTERFACE))))
        if k_harm == 0:
            # Normalize radial basis load so discrete FEM Watts == exact FVM Watts (eliminates polygonal area mismatch)
            psi_fn.x.array[:] *= I_fvm[alpha] / I_fem
            psi_rad_cache[m_idx] = (psi_fn.x.array.copy(), float(I_fvm[alpha]))
        else:
            # Remove tiny unstructured-mesh zero-mean drift along the same radial ring m_idx
            rad_arr, rad_int = psi_rad_cache[m_idx]
            psi_fn.x.array[:] -= (I_fem / rad_int) * rad_arr
        L_form = fem.form(psi_fn * v * ds(BND_ESC_TOP_INTERFACE))
        b = assemble_vector(L_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ksp.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        T_basis[:, alpha] = uh.eval(eval_pts, eval_cells).ravel()
        T_basis_esc_full[:, alpha] = uh.eval(esc_pts, esc_eval_cells).ravel()
        c_basis_cool[alpha] = float(fem.assemble_scalar(cool_form))
        b.destroy()

    A.destroy()
    ksp.destroy()

    # 3. Area-weighted L^2 mass-matrix projection from Polar FVM cells -> M basis functions
    #    Psi[p, alpha] = psi_alpha(x_p, y_p), W_A = diag(cell_areas)
    #    c = (Psi^T W_A Psi)^{-1} Psi^T W_A q_gap  -> preserves total Watts identically!
    Mass = Psi.T @ (w_flat[:, None] * Psi)  # (M, M)
    Proj = np.linalg.solve(Mass, Psi.T * w_flat[None, :])  # (M, N)

    G_q = T_basis @ Proj                        # (N, N) [K / (W/m^2)]
    G_q_esc_full = T_basis_esc_full @ Proj      # (N_esc_full, N)
    c_q_cool = c_basis_cool @ Proj              # (N,) [W / (W/m^2)] == m^2 (should match cell_areas!)

    elapsed = time.perf_counter() - t0
    # Verify independent energy balance on unit loads: c_heaters_cool should be ~1.0 W/W
    max_htr_imbal = float(np.max(np.abs(c_heaters_cool - 1.0)))
    max_q_cool_rel = float(np.max(np.abs(c_q_cool.sum() - w_flat.sum())) / w_flat.sum())

    np.savez_compressed(
        cache_path,
        ceramic_type=ceramic_type,
        refinement=refinement,
        h_chiller=h_chiller,
        n_r=n_r,
        n_theta=n_theta,
        n_knots=n_knots,
        k_max=k_max,
        M_basis=M,
        r_faces=grid["r_faces"],
        r_centers=grid["r_centers"],
        dr=grid["dr"],
        dtheta=grid["dtheta"],
        theta_centers=grid["theta_centers"],
        xx=grid["xx"],
        yy=grid["yy"],
        cell_areas=grid["cell_areas"],
        r_esc_centers=r_esc_centers,
        G_heaters=G_heaters,
        G_heaters_esc_full=G_heaters_esc_full,
        c_heaters_cool=c_heaters_cool,
        G_q=G_q,
        G_q_esc_full=G_q_esc_full,
        c_q_cool=c_q_cool,
        extract_time_s=elapsed,
        max_htr_cool_imbal=max_htr_imbal,
        max_q_cool_area_rel=max_q_cool_rel,
    )
    print(
        f" [Mode A 3D ESC ROM] {ceramic_type}/{refinement} | M={M} basis + 10 heaters | "
        f"Time={elapsed:.2f}s | max|c_htr-1|={max_htr_imbal:.2e} | "
        f"rel|sum(c_q)-A_w|={max_q_cool_rel:.2e} -> {cache_path.name}"
    )
    return cache_path


if __name__ == "__main__":
    extract_esc_3d_unit_load_rom(refinement="coarse", force_recompute=True)
    extract_esc_3d_unit_load_rom(refinement="medium", force_recompute=True)
