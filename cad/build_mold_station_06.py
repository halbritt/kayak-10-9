"""Build 100% watertight, support-free printable female mold STLs (mm) for Kayak 10.9 Station 06."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HULL_JSON = REPO_ROOT / "hull" / "kayak-10-9.hull.json"
BUILD_ROOT = REPO_ROOT / "build"
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = B / "s13" / "mold_station_06"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
import vtk
from deck_geometry import Z_KEEL_SHIFT, keyhole_inner_curve
from kayakgen.model.hull import Hull


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    su = np.linspace(0.0, float(seg[-1]), n)
    return np.column_stack((np.interp(su, seg, pts[:, 0]), np.interp(su, seg, pts[:, 1])))


def _cap_polygon_with_holes(outer_uv: np.ndarray, holes_uv: list[tuple[float, float, float]], z_val: float, normal_z: float):
    """Exact constrained 2D polygon-with-holes triangulation via vtkContourTriangulator (outer CCW, holes CW)."""
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE)
    lines = vtk.vtkCellArray()
    hole_meta: list[tuple[int, int, float, float, float]] = []
    area_out = float(np.sum(outer_uv[:, 0] * np.roll(outer_uv[:, 1], -1) - np.roll(outer_uv[:, 0], -1) * outer_uv[:, 1]))
    outer_ccw = outer_uv if area_out > 0.0 else outer_uv[::-1]
    loops = [outer_ccw]
    for uc, vc, rad in holes_uv:
        n_h = max(28, int(np.ceil(2.0 * np.pi * rad / 2.2)))
        th = np.linspace(0.0, 2.0 * np.pi, n_h, endpoint=False)
        h_pts = np.column_stack((uc + rad * np.cos(th), vc - rad * np.sin(th)))  # CW hole loop relative to +Z!
        hole_meta.append((sum(len(lp) for lp in loops), n_h, uc, vc, rad))
        loops.append(h_pts)

    idx = 0
    for lp in loops:
        n = len(lp)
        for ux, uy in lp:
            pts.InsertNextPoint(float(ux), float(uy), float(z_val))
        lines.InsertNextCell(n + 1)
        for k in range(n):
            lines.InsertCellPoint(idx + k)
        lines.InsertCellPoint(idx)
        idx += n

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    ct = vtk.vtkContourTriangulator()
    ct.SetInputData(pd)
    ct.Update()
    out = ct.GetOutput()

    xyz = np.column_stack((np.vstack(loops), np.full(idx, float(z_val))))
    polys = out.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    faces_l: list[list[int]] = []
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            faces_l.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])
    faces = np.asarray(faces_l, dtype=int)
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    flip = (np.cross(v1 - v0, v2 - v0)[:, 2] * normal_z) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False), hole_meta

def _loft_mold_block(outer_rings: np.ndarray, holes_uv: list[tuple[float, float, float]], zs_mm: np.ndarray) -> trimesh.Trimesh:
    """Build 100% watertight solid from (Nz, M, 2) outer_rings (dense <=2mm spacing) and straight Z holes."""
    cap_bot, h_meta = _cap_polygon_with_holes(outer_rings[0], holes_uv, float(zs_mm[0]), normal_z=-1.0)
    cap_top, _ = _cap_polygon_with_holes(outer_rings[-1], holes_uv, float(zs_mm[-1]), normal_z=+1.0)
    parts = [cap_bot, cap_top]

    # 1. Outer perimeter wall loft (orient normals outward via signed polygon area)
    nz, m_out, _ = outer_rings.shape
    area2 = float(np.sum(outer_rings[0, :, 0] * np.roll(outer_rings[0, :, 1], -1) - np.roll(outer_rings[0, :, 0], -1) * outer_rings[0, :, 1]))
    ccw = area2 > 0.0
    v_wall = np.column_stack((outer_rings.reshape(-1, 2), np.repeat(zs_mm, m_out)))
    f_wall: list[list[int]] = []
    for k in range(nz - 1):
        s, t = k * m_out, (k + 1) * m_out
        for j in range(m_out):
            j1 = (j + 1) % m_out
            c1, c2, n1, n2 = s + j, s + j1, t + j, t + j1
            if ccw:
                f_wall.extend([[c1, c2, n1], [c2, n2, n1]])
            else:
                f_wall.extend([[c1, n1, c2], [c2, n1, n2]])
    parts.append(trimesh.Trimesh(vertices=v_wall, faces=np.asarray(f_wall, dtype=int), process=False))

    # 2. Straight cylindrical bore walls (CW circle: normals point INWARD toward hole axis)
    for _, n_h, uc, vc, rad in h_meta:
        th = np.linspace(0.0, 2.0 * np.pi, n_h, endpoint=False)
        circle_xy = np.column_stack((uc + rad * np.cos(th), vc - rad * np.sin(th)))
        vh = np.column_stack((np.tile(circle_xy, (nz, 1)), np.repeat(zs_mm, n_h)))
        fh: list[list[int]] = []
        for k in range(nz - 1):
            s, t = k * n_h, (k + 1) * n_h
            for j in range(n_h):
                j1 = (j + 1) % n_h
                c1, c2, n1, n2 = s + j, s + j1, t + j, t + j1
                fh.extend([[c1, c2, n1], [c2, n2, n1]])
        parts.append(trimesh.Trimesh(vertices=vh, faces=np.asarray(fh, dtype=int), process=False))

    solid = trimesh.util.concatenate(parts)
    solid.merge_vertices(digits_vertex=4)
    if solid.volume < 0.0:
        solid.invert()
    assert solid.is_watertight, "Mold block must be 100% watertight!"
    assert solid.is_winding_consistent, "Mold block must have consistent manifold winding!"
    return solid


def _rotate_45_onto_bed(mesh_uvz: trimesh.Trimesh, v_center_mm: float) -> trimesh.Trimesh:
    """Shift v by -v_center_mm and rotate 45 deg around Z so 616 mm flanged trough nests inside 600x600 mm bed."""
    m = mesh_uvz.copy()
    v = np.asarray(m.vertices).copy()
    u_s, v_s, z_s = v[:, 0], v[:, 1] - v_center_mm, v[:, 2] - v[:, 2].min()
    inv_rt2 = 1.0 / np.sqrt(2.0)
    m.vertices = np.column_stack(((u_s - v_s) * inv_rt2, (u_s + v_s) * inv_rt2, z_s))
    return m

def _station_curves_mm(geom, x_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (hull_stbd_mm [keel -> widest], deck_stbd_mm [widest -> deck_cl]) with 0.000 mm undercut."""
    low, up = geom.half_section_outline(float(x_m))
    full = np.vstack((low, up[1:])) * 1000.0
    full[:, 1] += Z_KEEL_SHIFT * 1000.0
    full[0, 0] = 0.0
    full[-1, 0] = 0.0
    iw = int(np.argmax(full[:, 0]))
    hull_s = full[: iw + 1].copy()
    hull_s[:, 0] = np.maximum.accumulate(hull_s[:, 0])
    deck_s = full[iw:].copy()
    deck_s[:, 0] = np.minimum.accumulate(deck_s[:, 0])
    return hull_s, deck_s


def build_voron600_hull_mold(geom, xs_m: np.ndarray) -> tuple[trimesh.Trimesh, trimesh.Trimesh, list[tuple[float, float, float]]]:
    """1-piece Female Hull Mold for Station 06 (497.27 mm Z, 4x 1\" tube bores + 4x M8 tie-bolt bores)."""
    rings: list[np.ndarray] = []
    for xm in xs_m:
        hs, _ = _station_curves_mm(geom, float(xm))
        hs_r = _resample_polyline(hs, 110)  # ~2.8 mm spacing along starboard bilge
        ym, zm = float(hs_r[-1, 0]), float(hs_r[-1, 1])

        # Inner cavity: port flange outer -> port sheer -> keel -> stbd sheer -> stbd flange outer
        fl_port = _resample_polyline(np.array([[-308.0, zm], [-ym, zm]]), 24)[:-1]
        cav_port = np.column_stack((-hs_r[::-1, 0], hs_r[::-1, 1]))[:-1]
        cav_stbd = hs_r[:-1]
        fl_stbd = _resample_polyline(np.array([[ym, zm], [308.0, zm]]), 24)

        # Outer structural backing: stbd flange drop -> upper 1" tube boss -> 12 mm offset bilge -> flat bench base -> mirror port
        mid_bilge = hs_r[(hs_r[:, 0] >= 186.0) & (hs_r[:, 1] <= 105.0)][::-1]
        off_b = np.column_stack((mid_bilge[:, 0] + 10.5, mid_bilge[:, 1] - 11.5)) if len(mid_bilge) >= 2 else np.array([[220.0, 52.0]])
        stbd_back = _resample_polyline(
            np.vstack(([[308.0, zm], [308.0, 92.0], [270.0, 86.0]], off_b, [[186.0, 28.0], [186.0, -16.0]])),
            68,
        )[1:]
        base = _resample_polyline(np.array([[186.0, -16.0], [-186.0, -16.0]]), 40)[1:-1]
        port_back = np.column_stack((-stbd_back[::-1, 0], stbd_back[::-1, 1]))
        rings.append(np.vstack((fl_port, cav_port, cav_stbd, fl_stbd, stbd_back, base, port_back)))

    holes = [
        (-289.0, 124.0, 12.90), (+289.0, 124.0, 12.90),  # Upper 1.000" (25.4 mm) steel alignment tubes (+0.4 mm slip)
        (-154.0, +8.0, 12.90), (+154.0, +8.0, 12.90),    # Lower 1.000" bench-cradle steel alignment tubes
        (-289.0, 100.0, 4.25), (+289.0, 100.0, 4.25),    # Upper M8 section clamping tie-rod bores
        (-112.0, -2.0, 4.25), (+112.0, -2.0, 4.25),      # Lower M8 bench-base clamping tie-rod bores
    ]
    zs_mm = (xs_m - xs_m[0]) * 1000.0
    uvz = _loft_mold_block(np.asarray(rings), holes, zs_mm)
    return uvz, _rotate_45_onto_bed(uvz, v_center_mm=77.0), holes

def build_voron600_deck_mold(geom, xs_m: np.ndarray) -> tuple[trimesh.Trimesh, trimesh.Trimesh, list[tuple[float, float, float]]]:
    """Matching 1-piece Female Deck + Tumblehome Mold for Station 06 (497.27 mm Z, 0.000 mm undercut, keyhole scribe)."""
    th_kh = np.linspace(0.0, np.pi, 400)
    kh_stbd = keyhole_inner_curve(th_kh) * 1000.0  # mm
    kh_ord = np.argsort(kh_stbd[:, 0])
    kh_x, kh_y = kh_stbd[kh_ord, 0], kh_stbd[kh_ord, 1]

    rings: list[np.ndarray] = []
    for xm in xs_m:
        _, ds = _station_curves_mm(geom, float(xm))  # widest (ym, zm) -> centerline (0, z_cl)
        ds_r = _resample_polyline(ds, 110)
        # Add crisp 1.2 mm molded trim-scribe groove at keyhole inner rim so continuous deck bag trims cleanly!
        y_kh = float(np.interp(xm * 1000.0, kh_x, kh_y))
        groove = 1.2 * np.exp(-(((ds_r[:, 0] - y_kh) / 2.2) ** 2))
        ds_r[:, 1] -= groove
        ds_r[-1, 0] = 0.0
        ym, zm = float(ds_r[0, 0]), float(ds_r[0, 1])

        # Inner female deck cavity: stbd flange outer -> stbd sheer -> crown -> port sheer -> port flange outer
        fl_stbd = _resample_polyline(np.array([[308.0, zm], [ym, zm]]), 24)[:-1]
        cav_stbd = ds_r[:-1]
        cav_port = np.column_stack((-ds_r[::-1, 0], ds_r[::-1, 1]))[:-1]
        fl_port = _resample_polyline(np.array([[-ym, zm], [-308.0, zm]]), 24)

        # Inverted bench-cradle backing (flat top @ v = +318 mm clears 287 mm fwd arch & sits flat when flipped!)
        port_up = _resample_polyline(np.array([[-308.0, zm], [-308.0, 240.0], [-235.0, 318.0]]), 28)[1:]
        top_flat = _resample_polyline(np.array([[-235.0, 318.0], [+235.0, 318.0]]), 48)[1:-1]
        stbd_dn = np.column_stack((-port_up[::-1, 0], port_up[::-1, 1]))
        rings.append(np.vstack((fl_stbd, cav_stbd, cav_port, fl_port, port_up, top_flat, stbd_dn)))

    holes = [
        (-289.0, 204.0, 12.90), (+289.0, 204.0, 12.90),  # Outboard 1.000" steel alignment-tube bores
        (-178.0, 298.0, 12.90), (+178.0, 298.0, 12.90),  # Inverted-bench 1.000" steel alignment-tube bores
        (-289.0, 228.0, 4.25), (+289.0, 228.0, 4.25),    # Upper M8 section clamping bores
        (-95.0, 305.0, 4.25), (+95.0, 305.0, 4.25),      # Crown-base M8 section clamping bores
    ]
    zs_mm = (xs_m - xs_m[0]) * 1000.0
    uvz = _loft_mold_block(np.asarray(rings), holes, zs_mm)
    return uvz, _rotate_45_onto_bed(uvz, v_center_mm=236.0), holes


def build_h2s_stbd_half_mold(geom, xs_m: np.ndarray) -> tuple[trimesh.Trimesh, list[tuple[float, float, float]]]:
    """Bambu Lab H2S-Ready (340x320x340 mm) 300 mm Starboard Centerline-Split Half-Station Mold."""
    rings: list[np.ndarray] = []
    for xm in xs_m:
        hs, _ = _station_curves_mm(geom, float(xm))
        hs_r = _resample_polyline(hs, 110)
        ym, zm, zk = float(hs_r[-1, 0]), float(hs_r[-1, 1]), float(hs_r[0, 1])

        keel_wall = _resample_polyline(np.array([[0.0, -18.0], [0.0, zk]]), 12)[:-1]
        cav_stbd = hs_r[:-1]
        fl_stbd = _resample_polyline(np.array([[ym, zm], [304.0, zm]]), 24)
        mid_bilge = hs_r[(hs_r[:, 0] >= 184.0) & (hs_r[:, 1] <= 105.0)][::-1]
        off_b = np.column_stack((mid_bilge[:, 0] + 10.5, mid_bilge[:, 1] - 11.5)) if len(mid_bilge) >= 2 else np.array([[218.0, 52.0]])
        stbd_back = _resample_polyline(
            np.vstack(([[304.0, zm], [304.0, 92.0], [266.0, 86.0]], off_b, [[184.0, 28.0], [184.0, -18.0]])),
            68,
        )[1:]
        base = _resample_polyline(np.array([[184.0, -18.0], [0.0, -18.0]]), 76)[1:-1]
        rings.append(np.vstack((keel_wall, cav_stbd, fl_stbd, stbd_back, base)))

    holes = [(+286.0, 124.0, 12.90), (+148.0, +6.0, 12.90), (+286.0, 100.0, 4.25), (+42.0, -6.0, 4.25)]
    uvz = _loft_mold_block(np.asarray(rings), holes, (xs_m - xs_m[0]) * 1000.0)
    v = np.asarray(uvz.vertices).copy()
    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())
    v[:, 1] -= 0.5 * (v[:, 1].min() + v[:, 1].max())
    v[:, 2] -= v[:, 2].min()
    uvz.vertices = v
    return uvz, holes

def _style(m: trimesh.Trimesh, name: str, rgba: list[int], alpha_mode: str = "OPAQUE", rough: float = 0.36, metal: float = 0.08) -> trimesh.Trimesh:
    mc = m.copy()
    mc.visual = TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=rgba, roughnessFactor=rough, metallicFactor=metal, alphaMode=alpha_mode, doubleSided=True))
    mc.metadata["name"] = name
    return mc


def main() -> None:
    hull_spec = Hull.model_validate_json((HULL_JSON).read_text())
    geom = hull_spec.to_geometry()
    L_sec = hull_spec.length_m / 11.0  # 0.4972727 m = 497.27 mm (Station 06 of 11: x in [-0.24864, +0.24864] m)
    xs_v600 = np.linspace(-0.5 * L_sec, +0.5 * L_sec, 65)
    xs_h2s = np.linspace(-0.150, +0.150, 49)

    hull_uvz, hull_bed, h_holes = build_voron600_hull_mold(geom, xs_v600)
    deck_uvz, deck_bed, d_holes = build_voron600_deck_mold(geom, xs_v600)
    h2s_bed, h2s_holes = build_h2s_stbd_half_mold(geom, xs_h2s)

    # Export print-ready STLs in MILLIMETERS (slicer native unit: 1.0 = 1.0 mm, flat on Z = 0.000 mm)
    p_hull = OUT / "kayak-10-9-station-06-hull-mold-voron600.stl"
    p_deck = OUT / "kayak-10-9-station-06-deck-mold-voron600.stl"
    p_h2s = OUT / "kayak-10-9-station-06-stbd-half-mold-h2s.stl"
    hull_bed.export(p_hull)
    deck_bed.export(p_deck)
    h2s_bed.export(p_h2s)

    # Strict slicer-envelope & manifold assertions
    for lbl, m, lim in [
        ("Voron600_Hull_Mold", hull_bed, (300.0, 300.0, 600.0)),
        ("Voron600_Deck_Mold", deck_bed, (300.0, 300.0, 600.0)),
        ("BambuH2S_Stbd_Half_Mold", h2s_bed, (170.0, 160.0, 340.0)),
    ]:
        bmin, bmax = m.bounds
        assert abs(bmin[2]) < 1e-4 and bmax[2] <= lim[2], f"{lbl} Z out of bounds: {bmin}..{bmax}"
        assert abs(bmin[0]) <= lim[0] and abs(bmax[0]) <= lim[0] and abs(bmin[1]) <= lim[1] and abs(bmax[1]) <= lim[1], f"{lbl} XY out of bed: {bmin}..{bmax}"

    # Build interactive 3dviewer assembly in boat coords (mm): Hull Mold + Exploded Deck Mold (+60mm Z) + 1" Steel Tubes + Smoked Kayak Shell
    def to_boat_xyz(m_uvz: trimesh.Trimesh, x0_mm: float, dz_mm: float = 0.0) -> trimesh.Trimesh:
        mb = m_uvz.copy()
        vv = np.asarray(mb.vertices)
        mb.vertices = np.column_stack((x0_mm + vv[:, 2], vv[:, 0], vv[:, 1] + dz_mm))
        return mb

    x0_mm = float(xs_v600[0] * 1000.0)
    m_h_boat = to_boat_xyz(hull_uvz, x0_mm, 0.0)
    m_d_boat = to_boat_xyz(deck_uvz, x0_mm, +65.0)  # +65 mm exploded lift so cavity & keyhole scribe are clearly visible!

    tubes = []
    for uc, vc, rad in h_holes[:4]:
        cyl = trimesh.creation.cylinder(radius=12.70, height=620.0, sections=28)
        cyl.apply_transform(trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 1, 0]))
        cyl.apply_translation([0.0, uc, vc])
        tubes.append(cyl)
    m_tubes = trimesh.util.concatenate(tubes)

    # Smoked translucent Kayak 10.9 hull+deck shell segment nested inside the mold cavity
    kayak_rings = []
    for xm in xs_v600:
        hs_raw, ds_raw = _station_curves_mm(geom, float(xm))
        hs, ds = _resample_polyline(hs_raw, 60), _resample_polyline(ds_raw, 60)
        ring_yz = np.vstack((hs, ds[1:], np.column_stack((-ds[-2::-1, 0], ds[-2::-1, 1])), np.column_stack((-hs[-2::-1, 0], hs[-2::-1, 1]))))
        kayak_rings.append(np.column_stack((np.full(len(ring_yz), xm * 1000.0), ring_yz)))
    kr = np.asarray(kayak_rings)
    fk = [[k * kr.shape[1] + j, k * kr.shape[1] + (j + 1) % kr.shape[1], (k + 1) * kr.shape[1] + j] for k in range(kr.shape[0] - 1) for j in range(kr.shape[1])] + \
         [[k * kr.shape[1] + (j + 1) % kr.shape[1], (k + 1) * kr.shape[1] + (j + 1) % kr.shape[1], (k + 1) * kr.shape[1] + j] for k in range(kr.shape[0] - 1) for j in range(kr.shape[1])]
    m_kayak_seg = trimesh.Trimesh(vertices=kr.reshape(-1, 3), faces=np.asarray(fk, dtype=int), process=False)

    sc = trimesh.Scene()
    sc.add_geometry(_style(m_h_boat, "01_station06_female_hull_mold", [64, 74, 86, 255], rough=0.45, metal=0.12), node_name="01_station06_female_hull_mold")
    sc.add_geometry(_style(m_d_boat, "02_station06_female_deck_mold_exploded_65mm", [78, 92, 108, 255], rough=0.45, metal=0.12), node_name="02_station06_female_deck_mold_exploded_65mm")
    sc.add_geometry(_style(m_tubes, "03_1in_steel_alignment_tubes", [195, 202, 210, 255], rough=0.22, metal=0.82), node_name="03_1in_steel_alignment_tubes")
    sc.add_geometry(_style(m_kayak_seg, "04_kayak_10_9_laminate_part_preview", [56, 148, 208, 96], alpha_mode="BLEND", rough=0.20, metal=0.05), node_name="04_kayak_10_9_laminate_part_preview")
    sc.export(OUT / "kayak-10-9-station-06-mold-assembly.glb")

    # Estimate ABS-GF print mass (~1.08 g/cm^3, ~34% effective solid fraction w/ 4 perimeters + 22% gyroid infill)
    def est_kg(vol_mm3: float) -> float:
        return round((vol_mm3 / 1000.0) * 1.08 * 0.34 / 1000.0, 2)

    manifest = {
        "station": "Station 06 of 11 (Midship Cockpit Bay)",
        "boat_x_range_mm": [round(float(xs_v600[0] * 1000), 2), round(float(xs_v600[-1] * 1000), 2)],
        "stl_units": "millimeters (1.0 = 1.0 mm, slicer-ready on Z=0, zero supports needed)",
        "max_undercut_mm": 0.0,
        "voron_600_hull_mold": {"file": p_hull.name, "watertight": hull_bed.is_watertight, "bounds_mm": np.round(hull_bed.bounds, 2).tolist(), "solid_vol_cm3": round(hull_bed.volume / 1000.0, 1), "est_abs_gf_mass_kg_22pct_gyroid": est_kg(hull_bed.volume)},
        "voron_600_deck_mold": {"file": p_deck.name, "watertight": deck_bed.is_watertight, "bounds_mm": np.round(deck_bed.bounds, 2).tolist(), "solid_vol_cm3": round(deck_bed.volume / 1000.0, 1), "est_abs_gf_mass_kg_22pct_gyroid": est_kg(deck_bed.volume)},
        "bambu_h2s_stbd_half_mold": {"file": p_h2s.name, "watertight": h2s_bed.is_watertight, "bounds_mm": np.round(h2s_bed.bounds, 2).tolist(), "solid_vol_cm3": round(h2s_bed.volume / 1000.0, 1), "est_abs_gf_mass_kg_22pct_gyroid": est_kg(h2s_bed.volume)},
    }
    (OUT / "station_06_mold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    # Render 4-panel slicer build-plate & mated cross-section verification sheet
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2), dpi=190, facecolor="white")
    ax1, ax2, ax3, ax4 = axes.ravel()

    def plot_bed_mesh(ax, m_bed: trimesh.Trimesh, wx: float, wy: float, title: str, col: str) -> None:
        sl = trimesh.intersections.mesh_plane(m_bed, [0, 0, 1], [0, 0, 10.0])
        sl_top = trimesh.intersections.mesh_plane(m_bed, [0, 0, 1], [0, 0, float(m_bed.bounds[1, 2]) - 10.0])
        ax.plot([-wx / 2, wx / 2, wx / 2, -wx / 2, -wx / 2], [-wy / 2, -wy / 2, wy / 2, wy / 2, -wy / 2],
                "--", color="#0f172a", lw=2.0, label=f"Build Plate ({wx:.0f} × {wy:.0f} mm)")
        first_a, first_b = True, True
        for seg in sl:
            ax.plot(seg[:, 0], seg[:, 1], "-", color=col, lw=1.7, label="Z = 10 mm slice (bed base)" if first_a else None)
            first_a = False
        for seg in sl_top:
            ax.plot(seg[:, 0], seg[:, 1], ":", color="#64748b", lw=1.2, label=f"Z = {m_bed.bounds[1,2]:.0f} mm slice (top)" if first_b else None)
            first_b = False
        b0, b1 = m_bed.bounds
        ax.text(0, wy / 2 - 22, f"Bounds: X [{b0[0]:+.1f}, {b1[0]:+.1f}] | Y [{b0[1]:+.1f}, {b1[1]:+.1f}] | Z [0.0, {b1[2]:.1f}] mm\nWatertight Manifold: {m_bed.is_watertight} | 0.00 mm Undercut | Zero Supports",
                ha="center", va="top", fontsize=7.8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#94a3b8"))
        ax.set_xlim(-wx * 0.57, wx * 0.57); ax.set_ylim(-wy * 0.57, wy * 0.57); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.8, fontweight="bold"); ax.set_xlabel("Bed X (mm)"); ax.set_ylabel("Bed Y (mm)")
        ax.legend(loc="lower left", fontsize=7.5); ax.grid(True, alpha=0.25)

    plot_bed_mesh(ax1, hull_bed, 600.0, 600.0, "1. Voron Phoenix 600³ — Station 06 Female Hull Mold (45° Diagonal)", "#1d6fa5")
    plot_bed_mesh(ax2, deck_bed, 600.0, 600.0, "2. Voron Phoenix 600³ — Station 06 Female Deck Mold (45° Diagonal)", "#0b8a6a")
    plot_bed_mesh(ax3, h2s_bed, 340.0, 320.0, "3. Bambu Lab H2S (340×320×340) — 300 mm Starboard Half-Mold Coupon", "#d97724")

    # Panel 4: Mated Hull + Deck Female Molds around Kayak 10.9 @ Midship (x = 0.00 m)
    for m_uvz, col, lbl in [(hull_uvz, "#1d6fa5", "Female Hull Mold (0.00 mm undercut)"), (deck_uvz, "#0b8a6a", "Female Deck Mold + Keyhole Scribe")]:
        segs = trimesh.intersections.mesh_plane(m_uvz, [0, 0, 1], [0, 0, 248.636])
        f_l = True
        for seg in segs:
            ax4.plot(seg[:, 0], seg[:, 1], color=col, lw=1.9, label=lbl if f_l else None)
            f_l = False
    hs0, ds0 = _station_curves_mm(geom, 0.0)
    ax4.plot(np.concatenate((-hs0[::-1, 0], hs0[:, 0])), np.concatenate((hs0[::-1, 1], hs0[:, 1])), "--", color="#b83227", lw=1.4, label="Kayak 10.9 OML + Parting Plane")
    ax4.axhline(float(hs0[-1, 1]), color="#b83227", ls=":", lw=1.1)
    ax4.set_xlim(-335, 335); ax4.set_ylim(-35, 325); ax4.set_aspect("equal"); ax4.grid(True, alpha=0.25)
    ax4.set_title("4. Mated Hull + Deck Female Molds @ Midship (x = 0.00 m)", fontsize=9.8, fontweight="bold")
    ax4.set_xlabel("Boat y (mm)"); ax4.set_ylabel("Boat z above keel (mm)"); ax4.legend(loc="lower right", fontsize=7.5)

    fig.suptitle("Kayak 10.9 — Station 06 3D-Printable Female Mold Tooling (Watertight Slicer-Ready STLs in mm)", fontsize=12.0, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "station_06_mold_verification.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
