"""3D Conformal Multi-Layer Cylindrical ESC & Wafer Mesh Generator (Gmsh 4.x).

Builds a 3D multi-domain mesh for a 300 mm plasma etch Electrostatic Chuck (ESC):
  1. Al Cooling Base      : R_base = 160 mm, t_base = 15.0 mm (z in [0, 15.0] mm)
  2. ESC Ceramic Lower    : R_esc  = 155 mm, z in [15.0, 15.9] mm
  3. Embedded Heaters     : R_w    = 150 mm, t_htr = 0.2 mm (z in [15.9, 16.1] mm)
       - Zone 1 (Center)  : r in [0, 55] mm
       - Zone 2 (Middle)  : r in [55, 110] mm
       - Zones 3..10      : 8 azimuthal 45° edge sectors on r in [110, 150] mm
                            (AMAT US10332772B2-inspired multi-zone edge control)
     + Ceramic ring       : r in [150, 155] mm at heater z-level
  4. ESC Ceramic Upper    : R_esc  = 155 mm, z in [16.1, 17.0] mm
  5. Rarefied He Gap      : R_w    = 150 mm, t_gap_mesh = 0.2 mm (z in [17.0, 17.2] mm)
       (Virtual orthotropic interface layer: k_z = h_eff(r,theta)*t_gap_mesh, k_x=k_y=1e-6)
  6. Silicon Wafer        : R_w    = 150 mm, t_w = 0.775 mm (z in [17.2, 17.975] mm)

Uses OpenCASCADE 2D disk fragmentation + layered extrusion so all vertical
interfaces are 100% conformal with clean prismatic elements and low DOF count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import gmsh

# Physical Volume Tags
TAG_AL_BASE = 1
TAG_CERAMIC_BULK = 2
TAG_HE_GAP = 3
TAG_WAFER = 4
# 10 Heater Zone Volume Tags: 11 (Z1 center), 12 (Z2 mid), 13..20 (Z3..Z10 8 edge sectors)
TAG_HEATER_Z1 = 11
TAG_HEATER_Z2 = 12
TAG_HEATER_EDGE_START = 13  # 13..20 for sectors 0..7 (angle [k*45, (k+1)*45] deg)
N_EDGE_SECTORS = 8
N_HEATER_ZONES = 2 + N_EDGE_SECTORS  # 10 zones total

# Physical Surface Tags
BND_CHILLER_BOTTOM = 101
BND_WAFER_TOP = 102
BND_WAFER_EDGE = 103
BND_ESC_TOP_INTERFACE = 104  # Top of ceramic under wafer (z = z_esc_top, r <= R_w)


@dataclass(frozen=True)
class ESC3DGeometrySpec:
    """Geometric specification [meters] for 3D multi-zone ESC + 300 mm wafer."""
    r_z1: float = 0.055          # Inner center heater radius [m]
    r_z2: float = 0.110          # Middle ring heater outer radius [m]
    r_wafer: float = 0.150       # 300 mm wafer radius [m]
    r_esc: float = 0.155         # ESC ceramic outer radius [m]
    r_base: float = 0.160        # Al cooling base outer radius [m]

    t_base: float = 0.0150       # Al base thickness [m]
    t_cer_bot: float = 0.0009    # Ceramic below heater [m]
    t_heater: float = 0.0002     # Embedded heater layer thickness [m]
    t_cer_top: float = 0.0009    # Ceramic above heater [m] (total ceramic = 2.0 mm)
    t_gap_mesh: float = 0.0002   # Virtual He gap mesh thickness [m]
    t_wafer: float = 0.000775    # Silicon wafer thickness [m]

    @property
    def z_levels(self) -> tuple[float, ...]:
        z0 = 0.0
        z1 = z0 + self.t_base
        z2 = z1 + self.t_cer_bot
        z3 = z2 + self.t_heater
        z4 = z3 + self.t_cer_top      # z_esc_top
        z5 = z4 + self.t_gap_mesh     # z_wafer_bot
        z6 = z5 + self.t_wafer        # z_wafer_top
        return (z0, z1, z2, z3, z4, z5, z6)

    @property
    def heater_zone_areas(self) -> tuple[float, ...]:
        """Planar area [m^2] of each of the 10 heater zones."""
        a1 = math.pi * self.r_z1 ** 2
        a2 = math.pi * (self.r_z2 ** 2 - self.r_z1 ** 2)
        a_edge_total = math.pi * (self.r_wafer ** 2 - self.r_z2 ** 2)
        a_sec = a_edge_total / N_EDGE_SECTORS
        return (a1, a2) + (a_sec,) * N_EDGE_SECTORS

    @property
    def heater_zone_volumes(self) -> tuple[float, ...]:
        """Volume [m^3] of each of the 10 heater zones."""
        return tuple(a * self.t_heater for a in self.heater_zone_areas)


def _classify_entity_xy(dim: int, tag: int, geom: ESC3DGeometrySpec) -> tuple[str, int]:
    """Return (radial_region, edge_sector_idx) robustly for disks, annuli, and pie sectors."""
    xm, ym, _, xM, yM, _ = gmsh.model.getBoundingBox(dim, tag)
    cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
    r_c = math.hypot(cx, cy)
    # For concentric full disks/annuli, center of mass is at (0,0), so use bounding-box outer radius
    if r_c < 1e-3:
        r_out = max(abs(xm), abs(xM), abs(ym), abs(yM))
        if r_out <= geom.r_z1 + 1e-3:
            return "z1", -1
        if r_out <= geom.r_z2 + 1e-3:
            return "z2", -1
        if r_out <= geom.r_esc + 1e-3:
            return "esc_rim", -1
        return "base_rim", -1
    # Non-symmetric piece -> one of the 8 azimuthal edge sectors on [r_z2, r_wafer]
    ang = math.atan2(cy, cx) % (2.0 * math.pi)
    sec = min(int(ang / (2.0 * math.pi / N_EDGE_SECTORS)), N_EDGE_SECTORS - 1)
    return "edge", sec


def build_esc_3d_gmsh(
    refinement: str = "coarse",
    msh_path: Path | str | None = None,
    geom: ESC3DGeometrySpec = ESC3DGeometrySpec(),
    verbose: bool = False,
) -> Path:
    """Generate a conformal 3D multi-layer ESC + wafer mesh using Gmsh OCC."""
    # Mesh size & vertical layer subdivisions per refinement level
    presets = {
        "coarse": dict(lc_inner=0.016, lc_edge=0.011, nz_base=3, nz_cer_b=1, nz_htr=1, nz_cer_t=1, nz_gap=1, nz_waf=1),
        "medium": dict(lc_inner=0.011, lc_edge=0.0075, nz_base=4, nz_cer_b=2, nz_htr=1, nz_cer_t=2, nz_gap=1, nz_waf=2),
    }
    if refinement not in presets:
        raise ValueError(f"Unknown refinement '{refinement}'. Choose from {tuple(presets)}.")
    cfg = presets[refinement]

    if msh_path is None:
        out_dir = Path(__file__).resolve().parents[2] / "results" / "mesh"
        out_dir.mkdir(parents=True, exist_ok=True)
        msh_path = out_dir / f"esc_3d_{refinement}.msh"
    else:
        msh_path = Path(msh_path)
        msh_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add(f"esc_3d_{refinement}")
    occ = gmsh.model.occ

    # 1. Create 2D base disks + 8 radial divider lines in [r_z2, r_wafer]
    d_base = occ.addDisk(0, 0, 0, geom.r_base, geom.r_base)
    d_esc = occ.addDisk(0, 0, 0, geom.r_esc, geom.r_esc)
    d_waf = occ.addDisk(0, 0, 0, geom.r_wafer, geom.r_wafer)
    d_z2 = occ.addDisk(0, 0, 0, geom.r_z2, geom.r_z2)
    d_z1 = occ.addDisk(0, 0, 0, geom.r_z1, geom.r_z1)

    sector_lines = []
    for k in range(N_EDGE_SECTORS):
        theta = k * (2.0 * math.pi / N_EDGE_SECTORS)
        p_in = occ.addPoint(geom.r_z2 * math.cos(theta), geom.r_z2 * math.sin(theta), 0.0)
        p_out = occ.addPoint(geom.r_wafer * math.cos(theta), geom.r_wafer * math.sin(theta), 0.0)
        sector_lines.append((1, occ.addLine(p_in, p_out)))

    # Fragment all 2D disks & sector lines into non-overlapping conformal 2D faces
    occ.fragment(
        [(2, d_base)],
        [(2, d_esc), (2, d_waf), (2, d_z2), (2, d_z1)] + sector_lines,
    )
    occ.synchronize()

    # Classify z=0 2D faces by radial region
    base_faces = gmsh.model.getEntities(2)
    z_levels = geom.z_levels

    # Helper to extrude selected 2D faces upward by dz with n_layers structured layers
    def extrude_layer(
        faces: list[tuple[int, int]],
        dz: float,
        n_layers: int,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Extrude 2D faces by dz; returns (volume_tags, top_face_dimtags)."""
        out = occ.extrude(faces, 0.0, 0.0, dz, numElements=[n_layers], recombine=False)
        vols = []
        top_faces = []
        # For each input face, occ.extrude emits [top_face, volume, lateral_faces...]
        # Identify volumes (dim=3) and top faces at z_top
        occ.synchronize()
        z_target = None
        for dim, tag in out:
            if dim == 3:
                vols.append(tag)
        # Find top faces by checking bounding box z_max
        if faces:
            _, _, z_min_f, _, _, _ = gmsh.model.getBoundingBox(faces[0][0], faces[0][1])
            z_target = z_min_f + dz
        for dim, tag in out:
            if dim == 2:
                _, _, zm, _, _, zM = gmsh.model.getBoundingBox(dim, tag)
                if z_target is not None and abs(zm - z_target) < 1e-5 and abs(zM - z_target) < 1e-5:
                    top_faces.append((dim, tag))
        return vols, top_faces

    def filter_faces_by_rmax(faces: list[tuple[int, int]], r_max: float) -> list[tuple[int, int]]:
        kept = []
        for dim, tag in faces:
            xm, ym, _, xM, yM, _ = gmsh.model.getBoundingBox(dim, tag)
            r_out = max(abs(xm), abs(xM), abs(ym), abs(yM))
            if r_out <= r_max + 1e-3:
                kept.append((dim, tag))
        return kept

    # Layer 1: Al Cooling Base (all faces, r <= 160 mm)
    al_vols, top_z1 = extrude_layer(base_faces, geom.t_base, cfg["nz_base"])

    # Layer 2: ESC Ceramic Lower (r <= 155 mm)
    faces_esc_z1 = filter_faces_by_rmax(top_z1, geom.r_esc)
    cer_bot_vols, top_z2 = extrude_layer(faces_esc_z1, geom.t_cer_bot, cfg["nz_cer_b"])

    # Layer 3: Embedded Heater Layer + outer ceramic ring (r <= 155 mm)
    htr_layer_vols, top_z3 = extrude_layer(top_z2, geom.t_heater, cfg["nz_htr"])

    # Layer 4: ESC Ceramic Upper (r <= 155 mm)
    cer_top_vols, top_z4 = extrude_layer(top_z3, geom.t_cer_top, cfg["nz_cer_t"])

    # Layer 5: Rarefied He Virtual Gap (r <= 150 mm)
    faces_waf_z4 = filter_faces_by_rmax(top_z4, geom.r_wafer)
    gap_vols, top_z5 = extrude_layer(faces_waf_z4, geom.t_gap_mesh, cfg["nz_gap"])

    # Layer 6: Silicon Wafer (r <= 150 mm)
    wafer_vols, top_z6 = extrude_layer(top_z5, geom.t_wafer, cfg["nz_waf"])

    # Remove any duplicate entities at shared interfaces via coherence
    occ.removeAllDuplicates()
    occ.synchronize()

    # Re-classify ALL 3D volumes robustly by centroid (x, y, z)
    z0, z1, z2, z3, z4, z5, z6 = z_levels
    vol_groups: dict[int, list[int]] = {
        TAG_AL_BASE: [],
        TAG_CERAMIC_BULK: [],
        TAG_HE_GAP: [],
        TAG_WAFER: [],
        TAG_HEATER_Z1: [],
        TAG_HEATER_Z2: [],
    }
    for s in range(N_EDGE_SECTORS):
        vol_groups[TAG_HEATER_EDGE_START + s] = []

    for dim, tag in gmsh.model.getEntities(3):
        _, _, cz = occ.getCenterOfMass(dim, tag)
        region, sec = _classify_entity_xy(dim, tag, geom)
        if cz < z1:
            vol_groups[TAG_AL_BASE].append(tag)
        elif cz < z2:
            vol_groups[TAG_CERAMIC_BULK].append(tag)
        elif cz < z3:
            if region == "z1":
                vol_groups[TAG_HEATER_Z1].append(tag)
            elif region == "z2":
                vol_groups[TAG_HEATER_Z2].append(tag)
            elif region == "edge":
                vol_groups[TAG_HEATER_EDGE_START + sec].append(tag)
            else:
                vol_groups[TAG_CERAMIC_BULK].append(tag)
        elif cz < z4:
            vol_groups[TAG_CERAMIC_BULK].append(tag)
        elif cz < z5:
            vol_groups[TAG_HE_GAP].append(tag)
        else:
            vol_groups[TAG_WAFER].append(tag)

    for ptag, vlist in vol_groups.items():
        if not vlist:
            raise RuntimeError(f"Empty volume physical group {ptag}!")
        gmsh.model.addPhysicalGroup(3, vlist, tag=ptag)

    # Re-classify boundary surfaces robustly by bounding box & centroid
    chiller_surfs = []
    wafer_top_surfs = []
    wafer_edge_surfs = []
    esc_top_if_surfs = []

    for dim, tag in gmsh.model.getEntities(2):
        xm, ym, zm, xM, yM, zM = gmsh.model.getBoundingBox(dim, tag)
        cx, cy, cz = occ.getCenterOfMass(dim, tag)
        r_c = math.hypot(cx, cy)
        if abs(zm - z0) < 1e-5 and abs(zM - z0) < 1e-5:
            chiller_surfs.append(tag)
        elif abs(zm - z6) < 1e-5 and abs(zM - z6) < 1e-5:
            wafer_top_surfs.append(tag)
        elif abs(zm - z4) < 1e-5 and abs(zM - z4) < 1e-5 and max(abs(xm), abs(xM), abs(ym), abs(yM)) <= geom.r_wafer + 1e-3:
            esc_top_if_surfs.append(tag)
        elif zm >= z5 - 1e-5 and zM <= z6 + 1e-5 and abs(zM - zm) > 1e-5:
            # Vertical surface in wafer layer: check if outer cylindrical rim (r ~ R_w)
            r_max_bb = max(math.hypot(xm, ym), math.hypot(xM, yM), math.hypot(xm, yM), math.hypot(xM, ym))
            # Check center of mass radius on circular arc (chord radius > 0.140 m)
            if r_c > 0.135 and r_max_bb >= geom.r_wafer - 1e-3:
                wafer_edge_surfs.append(tag)

    gmsh.model.addPhysicalGroup(2, chiller_surfs, tag=BND_CHILLER_BOTTOM)
    gmsh.model.addPhysicalGroup(2, wafer_top_surfs, tag=BND_WAFER_TOP)
    gmsh.model.addPhysicalGroup(2, wafer_edge_surfs, tag=BND_WAFER_EDGE)
    gmsh.model.addPhysicalGroup(2, esc_top_if_surfs, tag=BND_ESC_TOP_INTERFACE)

    # Set spatially varying mesh size (finer near edge r >= 110 mm)
    for dim, ptag in gmsh.model.getEntities(0):
        px, py, _ = gmsh.model.getValue(0, ptag, [])
        r_p = math.hypot(px, py)
        lc = cfg["lc_edge"] if r_p >= geom.r_z2 - 1e-3 else cfg["lc_inner"]
        gmsh.model.mesh.setSize([(0, ptag)], lc)

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))
    gmsh.finalize()
    return msh_path


if __name__ == "__main__":
    p = build_esc_3d_gmsh(refinement="coarse", verbose=True)
    print(f"Generated 3D mesh: {p}")
