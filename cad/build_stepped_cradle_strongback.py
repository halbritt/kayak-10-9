"""Build H2S Port/Starboard Split Smoked Hull Mold + Stepped Clamping Baltic-Birch Cradle Strongback."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HULL_JSON = REPO_ROOT / "hull" / "kayak-10-9.hull.json"
BUILD_ROOT = REPO_ROOT / "build"
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = BUILD_ROOT / "mold_stepped_cradle_assy"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
import vtk
from shapely import Polygon as SectionPolygon, constrained_delaunay_triangles
from deck_geometry import SculptedKayakSurface

TAN_WEDGE = float(np.tan(np.deg2rad(8.0)))  # 8.0 deg self-locking gib/draw-wedge angle (7.1:1 mechanical advantage)
Z_BASE = -24.0  # Flat keel datum floor on top of central torsion box (mm)


def _box(cx: float, cy: float, cz: float, lx: float, ly: float, lz: float) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(lx, ly, lz))
    m.apply_translation((cx, cy, cz))
    return m


def _cyl(p0: tuple[float, float, float], p1: tuple[float, float, float], radius: float, sections: int = 18) -> trimesh.Trimesh:
    a, b = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    vec = b - a
    h = float(np.linalg.norm(vec))
    c = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)
    v_dir = vec / max(h, 1e-9)
    axis = np.cross([0.0, 0.0, 1.0], v_dir)
    if np.linalg.norm(axis) > 1e-7:
        c.apply_transform(trimesh.transformations.rotation_matrix(float(np.arccos(np.clip(v_dir[2], -1.0, 1.0))), axis))
    elif v_dir[2] < 0.0:
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]))
    c.apply_translation(0.5 * (a + b))
    return c


def _style(m: trimesh.Trimesh, name: str, rgba: list[int], alpha_mode: str = "OPAQUE", rough: float = 0.42, metal: float = 0.08) -> trimesh.Trimesh:
    mc = m.copy()
    mc.visual = TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=rgba, roughnessFactor=rough, metallicFactor=metal, alphaMode=alpha_mode, doubleSided=True))
    mc.metadata["name"] = name
    return mc


def _resample_2d(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    su = np.linspace(0.0, float(seg[-1]), n)
    return np.column_stack((np.interp(su, seg, pts[:, 0]), np.interp(su, seg, pts[:, 1])))


def _triangulate_loop_2d(loop_uv: np.ndarray, x_val: float, normal_x: float, holes_uv: list[np.ndarray] | None = None) -> trimesh.Trimesh:
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE)
    lines = vtk.vtkCellArray()
    area = float(np.sum(loop_uv[:, 0] * np.roll(loop_uv[:, 1], -1) - np.roll(loop_uv[:, 0], -1) * loop_uv[:, 1]))
    outer_ccw = loop_uv if area > 0.0 else loop_uv[::-1]
    loops = [outer_ccw]
    for h in (holes_uv or []):
        ah = float(np.sum(h[:, 0] * np.roll(h[:, 1], -1) - np.roll(h[:, 0], -1) * h[:, 1]))
        loops.append(h if ah < 0.0 else h[::-1])
    idx = 0
    for lp in loops:
        n = len(lp)
        for u, v in lp:
            pts.InsertNextPoint(float(u), float(v), 0.0)
        lines.InsertNextCell(n + 1)
        for k in range(n):
            lines.InsertCellPoint(idx + k)
        lines.InsertCellPoint(idx)
        idx += n
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(lines)
    ct = vtk.vtkContourTriangulator(); ct.SetInputData(pd); ct.Update()
    out = ct.GetOutput()
    polys = out.GetPolys(); polys.InitTraversal(); idl = vtk.vtkIdList(); faces_l = []
    while polys.GetNextCell(idl):
        if idl.GetNumberOfIds() == 3:
            faces_l.append([idl.GetId(0), idl.GetId(1), idl.GetId(2)])
    faces = np.asarray(faces_l, dtype=int)
    uv_all = np.vstack(loops)
    xyz = np.column_stack((np.full(len(uv_all), x_val), uv_all[:, 0], uv_all[:, 1]))
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    flip = (np.cross(v1 - v0, v2 - v0)[:, 0] * normal_x) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)


def _extrude_yz_loop_along_x(loop_yz: np.ndarray, x0: float, x1: float) -> trimesh.Trimesh:
    area = float(np.sum(loop_yz[:, 0] * np.roll(loop_yz[:, 1], -1) - np.roll(loop_yz[:, 0], -1) * loop_yz[:, 1]))
    lp = loop_yz if area > 0.0 else loop_yz[::-1]
    c0 = _triangulate_loop_2d(lp, x0, normal_x=-1.0)
    c1 = _triangulate_loop_2d(lp, x1, normal_x=+1.0)
    m = len(lp)
    vw = np.vstack((np.column_stack((np.full(m, x0), lp)), np.column_stack((np.full(m, x1), lp))))
    fw = []
    for j in range(m):
        j1 = (j + 1) % m
        fw.extend([[j, j1, m + j], [j1, m + j1, m + j]])
    solid = trimesh.util.concatenate([c0, c1, trimesh.Trimesh(vertices=vw, faces=np.asarray(fw, dtype=int), process=False)])
    solid.merge_vertices(digits_vertex=3)
    if solid.volume < 0.0:
        solid.invert()
    return solid


def _precompute_tables(surf: SculptedKayakSurface) -> dict[str, np.ndarray]:
    rings_mm = surf.mold_station_rings_mm()
    xs_mm = surf.xs * 1000.0
    z_raw, y_raw = np.zeros(len(xs_mm)), np.zeros(len(xs_mm))
    mid = surf.n_hull_half - 1
    for i, xm in enumerate(xs_mm):
        stbd = rings_mm[i, mid:, 1:]
        cand = stbd[:, 1] <= (118.0 if xm < -1300.0 else 198.0)
        iw = int(np.argmax(np.where(cand, stbd[:, 0], -1.0)))
        y_raw[i], z_raw[i] = float(stbd[iw, 0]), float(stbd[iw, 1])
    i_bow, i_stn = int(np.argmin(np.abs(xs_mm + 2550.0))), int(np.argmin(np.abs(xs_mm - 2500.0)))
    z_raw[:i_bow], z_raw[i_stn:] = z_raw[i_bow], z_raw[i_stn]
    xg = np.linspace(-2754.0, +2754.0, 553)
    k = np.exp(-0.5 * (np.arange(-48, 49) / 16.0) ** 2); k /= k.sum()
    zg = np.convolve(np.pad(np.interp(xg, xs_mm, z_raw), 48, mode="edge"), k, mode="valid")
    yg = np.convolve(np.pad(np.interp(xg, xs_mm, y_raw, left=0.0, right=0.0), 48, mode="edge"), k, mode="valid")
    return {"xs_mm": xs_mm, "rings_mm": rings_mm, "xg": xg, "zg": zg, "yg": yg}


def _hull_stbd_curve_at_x(tbl: dict[str, np.ndarray], x_mm: float) -> tuple[np.ndarray, float, float]:
    xs_mm, rings_mm = tbl["xs_mm"], tbl["rings_mm"]
    xc = float(np.clip(x_mm, xs_mm[0] + 0.05, xs_mm[-1] - 0.05))
    i1 = int(np.clip(np.searchsorted(xs_mm, xc), 1, len(xs_mm) - 1))
    w = float((xc - xs_mm[i1 - 1]) / max(xs_mm[i1] - xs_mm[i1 - 1], 1e-6))
    ring = (1.0 - w) * rings_mm[i1 - 1] + w * rings_mm[i1]
    zm = float(np.interp(x_mm, tbl["xg"], tbl["zg"]))
    ym_fair = float(np.interp(x_mm, tbl["xg"], tbl["yg"]))
    hs = SculptedKayakSurface.mold_stbd_section_mm(ring, zm)
    hs_r = _resample_2d(hs, 56)

    t_dam = float(np.clip((abs(x_mm) - 2734.0) / 1.6, 0.0, 1.0))
    t_dam = t_dam * t_dam * (3.0 - 2.0 * t_dam)
    hs_r[:, 0] = (1.0 - t_dam) * np.maximum(hs_r[:, 0], np.linspace(0.25, max(float(hs_r[-1, 0]), 1.5), len(hs_r))) + t_dam * np.linspace(0.25, 8.5, len(hs_r))
    hs_r[:, 1] = (1.0 - t_dam) * hs_r[:, 1] + t_dam * zm
    hs_r[0, 0] = 0.25
    return hs_r, zm, ym_fair


def _station_step_params(tbl: dict[str, np.ndarray], xs_k: np.ndarray) -> dict[str, float]:
    curves = [_hull_stbd_curve_at_x(tbl, float(xv)) for xv in xs_k]
    zm_min = min(c[1] for c in curves)
    zm_max = max(c[1] for c in curves)
    z_top_w = zm_min - 16.0
    z_step = round(float(np.clip(0.44 * zm_min, 38.0, 78.0)), 1)
    yh_below = max(float(np.max(c[0][c[0][:, 1] <= z_step + 0.5, 0])) if np.any(c[0][:, 1] <= z_step + 0.5) else 0.0 for c in curves)
    y_r1 = round(max(yh_below + 22.0, 44.0), 1)
    req_yw = 0.0
    for hs_r, _, _ in curves:
        mask = hs_r[:, 1] >= z_step - 1.0
        if np.any(mask):
            pts = hs_r[mask]
            req_yw = max(req_yw, float(np.max(pts[:, 0] + 19.0 - np.maximum(0.0, pts[:, 1] - z_step) * TAN_WEDGE)))
    y_w_bot = round(max(req_yw, y_r1 + 22.0), 1)
    y_w_top = y_w_bot + (z_top_w - z_step) * TAN_WEDGE
    y_cheek_in = round(y_w_top + 10.5, 1)
    y_cheek_out = y_cheek_in + 28.0
    y_fl_max = max(max(float(c[0][-1, 0]) + 30.0, c[2] + 30.0, y_w_top + 13.0) for c in curves)
    return {
        "zm_min": zm_min, "zm_max": zm_max, "z_top_w": z_top_w, "z_step": z_step,
        "y_r1": y_r1, "y_w_bot": y_w_bot, "y_w_top": y_w_top,
        "y_cheek_in": y_cheek_in, "y_cheek_out": y_cheek_out, "y_fl_max": y_fl_max,
    }


def _half_mold_loop_at_x(tbl: dict[str, np.ndarray], x_mm: float, sp: dict[str, float], side: int) -> np.ndarray:
    """2D (y, z) CCW loop of Starboard (side=+1, groove) or Port (side=-1, tongue) H2S Half-Mold."""
    hs_r, zm, ym_fair = _hull_stbd_curve_at_x(tbl, x_mm)
    ym = float(hs_r[-1, 0])
    z_top_w, z_step = sp["z_top_w"], sp["z_step"]
    y_r1, y_w_bot, y_w_top = sp["y_r1"], sp["y_w_bot"], sp["y_w_top"]
    y_fl = max(ym + 30.0, ym_fair + 30.0, y_w_top + 13.0)
    zk = float(hs_r[0, 1])

    fl_top = _resample_2d(np.array([[ym, zm], [y_fl, zm]]), 8)[1:]
    outer_step = np.array([
        [y_fl, z_top_w],
        [y_w_top, z_top_w],
        [0.5 * (y_w_top + y_w_bot), 0.5 * (z_top_w + z_step)],
        [y_w_bot, z_step],
        [0.5 * (y_w_bot + y_r1), z_step],
        [y_r1, z_step],
        [y_r1, 0.5 * (z_step + Z_BASE)],
        [y_r1, Z_BASE],
        [0.5 * y_r1, Z_BASE],
        [0.25, Z_BASE],
    ])
    if side > 0:
        # Starboard half: recessed trapezoidal alignment groove at z in [-11.6, -4.0] mm
        cl_seam = np.array([
            [0.25, -16.5], [0.25, -11.6], [5.8, -9.8], [5.8, -5.8], [0.25, -4.0], [0.25, max(zk - 0.8, -2.0)],
        ])
        pts = np.vstack((hs_r, fl_top, outer_step, cl_seam))
    else:
        # Port half: mirror outer/cavity to y < 0, plus projecting alignment tongue (+5.2 mm) engaging Starboard groove
        hs_p = np.column_stack((-hs_r[:, 0], hs_r[:, 1]))
        fl_p = np.column_stack((-fl_top[:, 0], fl_top[:, 1]))
        out_p = np.column_stack((-outer_step[:, 0], outer_step[:, 1]))
        cl_p = np.array([
            [-0.25, -16.5], [-0.25, -11.1], [+5.2, -9.4], [+5.2, -6.2], [-0.25, -4.5], [-0.25, max(zk - 0.8, -2.0)],
        ])
        pts = np.vstack((hs_p, fl_p, out_p, cl_p))

    area = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    return pts if area > 0.0 else pts[::-1]


def _mold_end_cap(loop_uv: np.ndarray, x_val: float, normal_x: float) -> trimesh.Trimesh:
    # Preserve every boundary segment, including the collinear vertices at
    # the alignment tongue. The adjacent loft uses those same vertices.
    uv_all = loop_uv
    index = {tuple(point): i for i, point in enumerate(uv_all)}
    triangles = constrained_delaunay_triangles(SectionPolygon(loop_uv))
    faces = np.array([
        [index[point] for point in tri.exterior.coords[:3]]
        for tri in triangles.geoms
    ], dtype=int)
    xyz = np.column_stack((np.full(len(uv_all), x_val), uv_all[:, 0], uv_all[:, 1]))
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    flip = (np.cross(v1 - v0, v2 - v0)[:, 0] * normal_x) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)


def _loft_half_station(tbl: dict[str, np.ndarray], xs_k: np.ndarray, sp: dict[str, float], side: int) -> trimesh.Trimesh:
    loops = np.asarray([_half_mold_loop_at_x(tbl, float(xv), sp, side) for xv in xs_k])
    c0 = _mold_end_cap(loops[0], float(xs_k[0]), normal_x=-1.0)
    c1 = _mold_end_cap(loops[-1], float(xs_k[-1]), normal_x=+1.0)
    nx, m, _ = loops.shape
    vw = np.column_stack((np.repeat(xs_k, m), loops.reshape(-1, 2)))
    fw = []
    for i in range(nx - 1):
        s, t = i * m, (i + 1) * m
        for j in range(m):
            j1 = (j + 1) % m
            fw.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]])
    solid = trimesh.util.concatenate([c0, c1, trimesh.Trimesh(vertices=vw, faces=np.asarray(fw, dtype=int), process=False)])
    solid.merge_vertices(digits_vertex=3)
    if solid.volume < 0.0:
        solid.invert()
    if not solid.is_volume:
        raise ValueError(f"Open or inverted mold half at x={xs_k[0]:.1f} mm, side={side}")
    return solid


def _cradle_yoke_and_wedges(xc: float, sp: dict[str, float]) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """Build 36 mm laminated Baltic-birch Stepped Cradle Yoke + 2x 8-deg gib draw-wedges + M8 hardware at xc."""
    z_top_w, z_step = sp["z_top_w"], sp["z_step"]
    y_r1, y_w_bot, y_w_top = sp["y_r1"], sp["y_w_bot"], sp["y_w_top"]
    y_ci = sp["y_cheek_in"]
    y_co = max(sp["y_cheek_out"], 120.0)
    y_nut = min(y_r1 + 18.0, y_ci - 4.0)

    # 1-piece full-depth stepped cradle yoke (with recessed nut pockets at z in [-24, -4] for lower M8 keel tie-rod)
    yoke_yz = np.array([
        [-y_co, z_top_w - 1.5], [-y_ci, z_top_w - 1.5], [-y_ci, z_step],
        [-(y_r1 + 0.6), z_step], [-(y_r1 + 0.6), -4.0], [-y_nut, -4.0], [-y_nut, Z_BASE],
        [+y_nut, Z_BASE], [+y_nut, -4.0], [+(y_r1 + 0.6), -4.0], [+(y_r1 + 0.6), z_step],
        [+y_ci, z_step], [+y_ci, z_top_w - 1.5], [+y_co, z_top_w - 1.5],
        [+y_co, z_step - 22.0], [+120.0, -118.0], [+120.0, -200.0],
        [-120.0, -200.0], [-120.0, -118.0], [-y_co, z_step - 22.0],
    ])
    yoke_parts = [_extrude_yz_loop_along_x(yoke_yz, xc - 18.0, xc + 18.0)]
    for sgn in (-1.0, +1.0):
        # 168 mm long solid birch L-saddle backing the entire 136 mm 8-deg wedge & horizontal Z-step shelf
        sad_yz = np.array([
            [sgn * (y_r1 + 0.6), z_step - 20.0], [sgn * y_co, z_step - 20.0],
            [sgn * y_co, z_top_w - 1.5], [sgn * y_ci, z_top_w - 1.5],
            [sgn * y_ci, z_step], [sgn * (y_r1 + 0.6), z_step],
        ])
        yoke_parts.append(_extrude_yz_loop_along_x(sad_yz, xc - 84.0, xc + 84.0))
    yoke_m = trimesh.util.concatenate(yoke_parts)

    wedges, hw = [], []
    # Lower M8 stainless cross-tie rod at z = -14.0 mm clamping Port & Starboard keel bases directly together
    hw.append(_cyl((xc, -(y_r1 + 14.0), -14.0), (xc, +(y_r1 + 14.0), -14.0), radius=4.2, sections=16))
    for sgn in (-1.0, +1.0):
        hw.append(_cyl((xc, sgn * y_r1, -14.0), (xc, sgn * (y_r1 + 3.0), -14.0), radius=10.5, sections=20))
        hw.append(_cyl((xc, sgn * (y_r1 + 3.0), -14.0), (xc, sgn * (y_r1 + 10.0), -14.0), radius=7.5, sections=6))

        # Captive 8-deg phenolic/hardwood draw-wedge (136 mm long along X) + vertical top-access M8 socket draw-bolt
        w_yz = np.array([
            [sgn * (y_w_bot + 0.5), z_step + 1.5],
            [sgn * (y_w_top + 0.5), z_top_w - 1.0],
            [sgn * (y_ci - 0.4), z_top_w - 1.0],
            [sgn * (y_ci - 0.4), z_step + 1.5],
        ])
        wedges.append(_extrude_yz_loop_along_x(w_yz, xc - 68.0, xc + 68.0))
        yb_w = sgn * (0.5 * (y_w_bot + y_w_top) * 0.5 + 0.5 * y_ci)
        hw.append(_cyl((xc, yb_w, z_step - 22.0), (xc, yb_w, z_top_w + 8.0), radius=4.0, sections=14))
        hw.append(_cyl((xc, yb_w, z_top_w - 1.0), (xc, yb_w, z_top_w + 2.0), radius=10.0, sections=20))
        hw.append(_cyl((xc, yb_w, z_top_w + 2.0), (xc, yb_w, z_top_w + 9.0), radius=6.8, sections=16))
    return yoke_m, wedges, hw


def _side_web_with_oval_ports(y_in: float, y_out: float, x_min: float, x_max: float, z_min: float, z_max: float, port_xs: np.ndarray) -> trimesh.Trimesh:
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE)
    lines = vtk.vtkCellArray()
    loops = [np.array([[x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]], dtype=float)]
    th = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    zc = 0.5 * (z_min + z_max)
    for xc in port_xs:
        loops.append(np.column_stack((xc + 88.0 * np.cos(th), zc - 44.0 * np.sin(th))))
    idx = 0
    for lp in loops:
        n = len(lp)
        for ux, uz in lp:
            pts.InsertNextPoint(float(ux), float(uz), 0.0)
        lines.InsertNextCell(n + 1)
        for k in range(n):
            lines.InsertCellPoint(idx + k)
        lines.InsertCellPoint(idx)
        idx += n
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(lines)
    ct = vtk.vtkContourTriangulator(); ct.SetInputData(pd); ct.Update()
    out = ct.GetOutput()
    polys = out.GetPolys(); polys.InitTraversal(); idl = vtk.vtkIdList(); fl = []
    while polys.GetNextCell(idl):
        if idl.GetNumberOfIds() == 3:
            fl.append([idl.GetId(0), idl.GetId(1), idl.GetId(2)])
    faces_2d = np.asarray(fl, dtype=int)
    xz_all = np.vstack(loops)
    v_in = np.column_stack((xz_all[:, 0], np.full(len(xz_all), y_in), xz_all[:, 1]))
    v_out = np.column_stack((xz_all[:, 0], np.full(len(xz_all), y_out), xz_all[:, 1]))
    ny_in = np.cross(v_in[faces_2d[:, 1]] - v_in[faces_2d[:, 0]], v_in[faces_2d[:, 2]] - v_in[faces_2d[:, 0]])[:, 1]
    f_in = np.where((ny_in * (-1.0 if y_out > y_in else +1.0))[:, None] < 0, faces_2d[:, [0, 2, 1]], faces_2d)
    f_out = f_in[:, [0, 2, 1]] + len(xz_all)
    rim_f, cur = [], 0
    for lp in loops:
        n = len(lp)
        for k in range(n):
            k1 = (k + 1) % n
            a0, a1, b0, b1 = cur + k, cur + k1, len(xz_all) + cur + k, len(xz_all) + cur + k1
            rim_f.extend([[a0, a1, b0], [a1, b1, b0]] if y_out > y_in else [[a0, b0, a1], [a1, b0, b1]])
        cur += n
    m = trimesh.Trimesh(vertices=np.vstack((v_in, v_out)), faces=np.vstack((f_in, f_out, np.asarray(rim_f, dtype=int))), process=False)
    if m.volume < 0.0:
        m.invert()
    return m


def _loft_quad_prism(xs: np.ndarray, quads_yz: np.ndarray) -> trimesh.Trimesh:
    """Loft a 4-point CCW (y, z) ribbon along xs to form a smooth 18 mm birch wing stringer rail."""
    c0 = _triangulate_loop_2d(quads_yz[0], float(xs[0]), normal_x=-1.0)
    c1 = _triangulate_loop_2d(quads_yz[-1], float(xs[-1]), normal_x=+1.0)
    nx, m, _ = quads_yz.shape
    vw = np.column_stack((np.repeat(xs, m), quads_yz.reshape(-1, 2)))
    fw = []
    for i in range(nx - 1):
        s, t = i * m, (i + 1) * m
        for j in range(m):
            j1 = (j + 1) % m
            fw.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]])
    solid = trimesh.util.concatenate([c0, c1, trimesh.Trimesh(vertices=vw, faces=np.asarray(fw, dtype=int), process=False)])
    solid.merge_vertices(digits_vertex=3)
    if solid.volume < 0.0:
        solid.invert()
    return solid


def build_assembly(surf: SculptedKayakSurface, edges_x: np.ndarray) -> tuple[dict[str, trimesh.Trimesh], dict[str, np.ndarray], list[dict[str, float]]]:
    tbl = _precompute_tables(surf)
    n_st = len(edges_x) - 1
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    port_molds, stbd_molds, yokes, wedges, hw = [], [], [], [], []
    sp_list: list[dict[str, float]] = []
    max_h2s_dims = [0.0, 0.0, 0.0]

    for k in range(n_st):
        xa, xb = float(edges_x[k]) + 0.7, float(edges_x[k + 1]) - 0.7
        if k == 0:
            xs_k = np.unique(np.concatenate([np.linspace(xa, -2733.0, 9), np.linspace(-2733.0, xb, 15)]))
        elif k == n_st - 1:
            xs_k = np.unique(np.concatenate([np.linspace(xa, +2733.0, 15), np.linspace(+2733.0, xb, 9)]))
        else:
            xs_k = np.linspace(xa, xb, 17)
        sp = _station_step_params(tbl, xs_k)
        sp_list.append(sp)

        pm = _loft_half_station(tbl, xs_k, sp, side=-1)
        sm = _loft_half_station(tbl, xs_k, sp, side=+1)
        for hm in (pm, sm):
            ext = hm.bounds[1] - hm.bounds[0]  # (dX_station_Zprint, dY_width_Xprint, dZ_height_Yprint)
            max_h2s_dims = [max(max_h2s_dims[j], float(ext[j])) for j in range(3)]
        port_molds.append(pm)
        stbd_molds.append(sm)

        ym_yoke, w_pair, hw_k = _cradle_yoke_and_wedges(float(centers_x[k]), sp)
        yokes.append(ym_yoke)
        wedges.extend(w_pair)
        hw.extend(hw_k)

    print(f"H2S MAX HALF-STATION ENVELOPE (Z_print={max_h2s_dims[0]:.1f}<=340, X_print={max_h2s_dims[1]:.1f}<=340, Y_print={max_h2s_dims[2]:.1f}<=320 mm)")

    # 4x Ø10x48 mm stainless station-to-station alignment dowels bridging every seam inside the smoked mold halves
    for j in range(1, n_st):
        xj = float(edges_x[j])
        hs_j, zm_j, ym_fj = _hull_stbd_curve_at_x(tbl, xj)
        ym_j = max(float(hs_j[-1, 0]), ym_fj)
        y_low = min(sp_list[j - 1]["y_r1"], sp_list[j]["y_r1"]) - 14.0
        for sgn in (-1.0, +1.0):
            hw.append(_cyl((xj - 24.0, sgn * (ym_j + 14.0), zm_j - 8.0), (xj + 24.0, sgn * (ym_j + 14.0), zm_j - 8.0), radius=5.0, sections=16))
            hw.append(_cyl((xj - 24.0, sgn * max(y_low, 22.0), -13.0), (xj + 24.0, sgn * max(y_low, 22.0), -13.0), radius=5.0, sections=16))

    # Central Baltic-birch torsion-box beam: 5,540 x 240 x 176 mm (z in [-200, -24] mm, 18 mm birch ply)
    L_sb = 5540.0
    bay_xs = edges_x[1:-1]  # 16 open bays cleanly centered between the 17 full-depth Cradle Yokes!
    skins = [
        _box(0.0, 0.0, -33.0, L_sb, 240.0, 18.0),   # Top datum skin (top face at Z_BASE = -24.0 mm)
        _box(0.0, 0.0, -191.0, L_sb, 240.0, 18.0),  # Bottom skin (bottom face at -200.0 mm)
        _side_web_with_oval_ports(-102.0, -120.0, -0.5 * L_sb, +0.5 * L_sb, -182.0, -42.0, bay_xs),
        _side_web_with_oval_ports(+102.0, +120.0, -0.5 * L_sb, +0.5 * L_sb, -182.0, -42.0, bay_xs),
        _box(-0.5 * L_sb + 9.0, 0.0, -112.0, 18.0, 204.0, 140.0),  # Bow torsion-box end bulkhead
        _box(+0.5 * L_sb - 9.0, 0.0, -112.0, 18.0, 204.0, 140.0),  # Stern torsion-box end bulkhead
        _box(0.0, 0.0, -138.0, L_sb - 36.0, 18.0, 88.0),           # Central notched spine web
    ]

    # Twin continuous 18 mm birch longitudinal wing-shelf tie stringers locking stepped cradle yokes (stations 2..14) fore-and-aft
    idx_wing = list(range(2, 15))
    xs_w = centers_x[idx_wing]
    for sgn in (-1.0, +1.0):
        q_list = []
        for idx_k in idx_wing:
            sp_k = sp_list[idx_k]
            yw_in = max(sp_k["y_cheek_in"] - 22.0, 124.0)
            yw_out = yw_in + 18.0
            zs = sp_k["z_step"]
            quad = np.array([[sgn * yw_in, zs - 52.0], [sgn * yw_out, zs - 52.0], [sgn * yw_out, zs], [sgn * yw_in, zs]])
            q_list.append(quad if sgn > 0 else quad[::-1])
        yokes.append(_loft_quad_prism(xs_w, np.asarray(q_list)))

    # Two Bessel-point (-1530 mm & +1530 mm) rolling steel shop trestles (floor @ z = -850 mm)
    trestles = []
    for xt in (-1530.0, +1530.0):
        trestles.append(_box(xt, 0.0, -228.0, 140.0, 340.0, 44.0))
        trestles.append(_box(xt, 0.0, -720.0, 90.0, 620.0, 40.0))
        for sy in (-1.0, +1.0):
            trestles.append(_box(xt, sy * 88.0, -203.0, 90.0, 36.0, 6.0))
            for sx in (-1.0, +1.0):
                trestles.append(_cyl((xt + sx * 42.0, sy * 135.0, -250.0), (xt + sx * 135.0, sy * 295.0, -795.0), radius=21.0, sections=16))
                cx_w, cy_w = xt + sx * 135.0, sy * 295.0
                trestles.append(_box(cx_w, cy_w, -802.0, 56.0, 56.0, 14.0))
                trestles.append(_cyl((cx_w, cy_w - 14.0, -826.0), (cx_w, cy_w + 14.0, -826.0), radius=24.0, sections=22))

    parts = {
        "port_mold": trimesh.util.concatenate(port_molds),
        "stbd_mold": trimesh.util.concatenate(stbd_molds),
        "cradle_yokes": trimesh.util.concatenate(yokes),
        "strongback_box": trimesh.util.concatenate(skins),
        "wedges": trimesh.util.concatenate(wedges),
        "hardware": trimesh.util.concatenate(hw),
        "trestles": trimesh.util.concatenate(trestles),
    }
    return parts, tbl, sp_list


def render_cross_section_diagram(tbl: dict[str, np.ndarray], sp_list: list[dict[str, float]], edges_x: np.ndarray) -> None:
    """Render annotated Midship Cross-Section Diagram showing H2S Port/Stbd clamping mechanics."""
    k_mid = 10  # Station 11 near max beam (xc = +486 mm)
    xc = float(0.5 * (edges_x[k_mid] + edges_x[k_mid + 1]))
    sp = sp_list[k_mid]
    lp_p = _half_mold_loop_at_x(tbl, xc, sp, side=-1)
    lp_s = _half_mold_loop_at_x(tbl, xc, sp, side=+1)

    z_top_w, z_step = sp["z_top_w"], sp["z_step"]
    y_r1, y_w_bot, y_w_top = sp["y_r1"], sp["y_w_bot"], sp["y_w_top"]
    y_ci, y_co = sp["y_cheek_in"], max(sp["y_cheek_out"], 120.0)
    y_nut = min(y_r1 + 18.0, y_ci - 4.0)
    yoke_yz = np.array([
        [-y_co, z_top_w - 1.5], [-y_ci, z_top_w - 1.5], [-y_ci, z_step],
        [-(y_r1 + 0.6), z_step], [-(y_r1 + 0.6), -4.0], [-y_nut, -4.0], [-y_nut, Z_BASE],
        [+y_nut, Z_BASE], [+y_nut, -4.0], [+(y_r1 + 0.6), -4.0], [+(y_r1 + 0.6), z_step],
        [+y_ci, z_step], [+y_ci, z_top_w - 1.5], [+y_co, z_top_w - 1.5],
        [+y_co, z_step - 22.0], [+120.0, -118.0], [+120.0, -200.0],
        [-120.0, -200.0], [-120.0, -118.0], [-y_co, z_step - 22.0],
    ])

    fig, ax = plt.subplots(figsize=(14.5, 8.8), dpi=200)
    fig.patch.set_facecolor("#131b26")
    ax.set_facecolor("#17212e")

    # Cradle yoke + central torsion box section
    ax.add_patch(Polygon(yoke_yz, closed=True, facecolor="#d9b683", edgecolor="#f3debd", lw=1.6, alpha=0.92, label="36 mm Laminated Baltic-Birch Stepped Cradle Yoke"))
    ax.add_patch(Polygon([[-120, -200], [120, -200], [120, -24], [-120, -24]], closed=True, facecolor="none", edgecolor="#8b6b42", lw=1.3, ls="--"))
    ax.add_patch(Polygon([[-102, -182], [102, -182], [102, -42], [-102, -42]], closed=True, facecolor="#1d2837", edgecolor="#a58254", lw=1.1))
    ax.add_patch(Polygon([[-9, -182], [9, -182], [9, -94], [-9, -94]], closed=True, facecolor="#c7a06b", edgecolor="#f3debd", lw=1.0))

    # Port & Starboard H2S smoked mold halves
    ax.add_patch(Polygon(lp_p, closed=True, facecolor="#3b658c", edgecolor="#8cc4f7", lw=1.8, alpha=0.68, label="Port H2S Mold Half (<= 318 x 217 x 324 mm)"))
    ax.add_patch(Polygon(lp_s, closed=True, facecolor="#3f7c85", edgecolor="#8be8eb", lw=1.8, alpha=0.68, label="Starboard H2S Mold Half (y=0 Tongue & Groove)"))

    # 8-deg phenolic draw-wedges + top-entry M8 bolts
    for sgn in (-1.0, +1.0):
        w_yz = np.array([
            [sgn * (y_w_bot + 0.5), z_step + 1.5], [sgn * (y_w_top + 0.5), z_top_w - 1.0],
            [sgn * (y_ci - 0.4), z_top_w - 1.0], [sgn * (y_ci - 0.4), z_step + 1.5],
        ])
        ax.add_patch(Polygon(w_yz, closed=True, facecolor="#c86428", edgecolor="#ffd1a8", lw=1.6, label="8° Phenolic/Hardwood Draw-Wedge (7.1:1 Clamp)" if sgn < 0 else None))
        yb = sgn * (0.25 * (y_w_bot + y_w_top) + 0.5 * y_ci)
        ax.plot([yb, yb], [z_step - 22.0, z_top_w + 8.0], color="#e8eef7", lw=3.0)
        ax.add_patch(Polygon([[yb - 7, z_top_w + 1], [yb + 7, z_top_w + 1], [yb + 7, z_top_w + 9], [yb - 7, z_top_w + 9]], closed=True, facecolor="#dce4ee", edgecolor="#ffffff", lw=1.0))
        # Downward bolt preload arrow & inward lateral clamping force arrow
        ax.annotate("", xy=(yb, z_top_w - 18), xytext=(yb, z_top_w + 26), arrowprops=dict(arrowstyle="-|>", color="#ffd166", lw=2.2, mutation_scale=15))
        z_mid = 0.5 * (z_step + z_top_w)
        ax.annotate("", xy=(sgn * (y_w_bot - 42), z_mid), xytext=(sgn * (y_ci - 2), z_mid), arrowprops=dict(arrowstyle="-|>", color="#ff595e", lw=2.8, mutation_scale=18))

    # Lower M8 stainless keel cross-tie rod
    ax.plot([-(y_r1 + 12), +(y_r1 + 12)], [-14.0, -14.0], color="#e8eef7", lw=3.4, label="Lower M8 Stainless Keel Cross-Tie Rod")
    for sgn in (-1.0, +1.0):
        ax.add_patch(Polygon([[sgn * y_r1, -21], [sgn * (y_r1 + 9), -21], [sgn * (y_r1 + 9), -7], [sgn * y_r1, -7]], closed=True, facecolor="#dce4ee", edgecolor="#ffffff", lw=1.0))
        ax.annotate("", xy=(sgn * (y_r1 - 44), -14.0), xytext=(sgn * (y_r1 + 16), -14.0), arrowprops=dict(arrowstyle="-|>", color="#ff595e", lw=2.4, mutation_scale=15))

    ax.text(0, 18, "Interlocking y = 0\nTongue & Groove\n(Zero Z-Step)", color="#ffffff", fontsize=9.5, fontweight="bold", ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="#131b26", ec="#8be8eb", alpha=0.88))
    ax.text(-235, z_top_w + 32, "Top-Access M8 Draw-Bolt\ndrives 8° Wedge (~700 lbf inward squeeze\n85-90% up hull side!)", color="#ffd166", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(+(y_r1 - 65), z_step - 16, f"Dead-Flat Horizontal Z-Step (z = +{z_step:.0f} mm)\nPrint shrinkage never shifts hull rocker!", color="#ffe8c2", fontsize=9.0, ha="center", bbox=dict(boxstyle="round,pad=0.25", fc="#1f2c3d", ec="#d9b683", alpha=0.9))
    ax.text(0, -118, "240 x 176 mm\n18 mm Baltic-Birch\nTorsion-Box Spine", color="#f3debd", fontsize=9.5, fontweight="bold", ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_xlim(-395, +395)
    ax.set_ylim(-225, +240)
    ax.grid(True, color="#2a3a4e", lw=0.6, alpha=0.6)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.set_xlabel("Transverse Half-Breadth Y (mm) — Port (-) / Starboard (+)", color="#e2e8f0", fontsize=10.5)
    ax.set_ylabel("Vertical Elevation Z from Keel Baseline (mm)", color="#e2e8f0", fontsize=10.5)
    ax.set_title("Kayak 10.9 — H2S Split-Half Female Hull Mold in Stepped Clamping Baltic-Birch Cradle (Midship Section, 1:1 mm)", color="#f8fafc", fontsize=12.5, fontweight="bold", pad=12)
    leg = ax.legend(loc="lower right", facecolor="#131b26", edgecolor="#475569", labelcolor="#f1f5f9", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(OUT / "stepped_cradle_cross_section.png", dpi=200)
    plt.close(fig)


def render_3d_views(parts: dict[str, trimesh.Trimesh]) -> None:
    import pyvista as pv
    from PIL import Image

    def to_pv(tm_m: trimesh.Trimesh, feat: float = 52.0) -> pv.PolyData:
        v = np.asarray(tm_m.vertices, dtype=float)
        f = np.asarray(tm_m.faces, dtype=np.int64)
        pd = pv.PolyData(v, np.column_stack((np.full(len(f), 3, dtype=np.int64), f)).ravel())
        return pd.compute_normals(cell_normals=False, point_normals=True, split_vertices=True, feature_angle=feat, auto_orient_normals=False)

    pvp = {k: to_pv(v, 56 if "mold" in k else 45) for k, v in parts.items()}

    def _shoot(out_name: str, win_sz: tuple[int, int], cam_pos: tuple[float, float, float], focal: tuple[float, float, float], pscale: float, show_stbd: bool = True) -> None:
        pl = pv.Plotter(off_screen=True, window_size=win_sz, lighting="none")
        pl.set_background("#16202d", top="#28384c")
        for pos, inten, col in [
            ((-5500, -6500, 5800), 1.50, "#fffdf9"),
            ((+6000, -4800, 4600), 0.95, "#e8f3ff"),
            ((0, +7000, 4800), 0.80, "#ffffff"),
            ((-1600, -5200, -450), 0.55, "#fff4e2"),
        ]:
            lt = pv.Light(position=pos, focal_point=focal, color=col, intensity=inten)
            lt.positional = False
            pl.add_light(lt)
        pl.add_mesh(pvp["trestles"], color="#333e4c", pbr=True, metallic=0.45, roughness=0.36, smooth_shading=True)
        pl.add_mesh(pvp["strongback_box"], color="#c9a574", pbr=True, metallic=0.02, roughness=0.56, smooth_shading=True)
        pl.add_mesh(pvp["cradle_yokes"], color="#e4c596", pbr=True, metallic=0.03, roughness=0.46, smooth_shading=True)
        w_mesh = pvp["wedges"] if show_stbd else pvp["wedges"].clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=True)
        pl.add_mesh(w_mesh, color="#c86428", pbr=True, metallic=0.06, roughness=0.34, smooth_shading=True)
        pl.add_mesh(pvp["hardware"], color="#e6ecf4", pbr=True, metallic=0.88, roughness=0.16, smooth_shading=True)
        pl.enable_depth_peeling(number_of_peels=10)
        pl.add_mesh(pvp["port_mold"], color="#3b658c", opacity=0.42 if show_stbd else 0.52, pbr=True, metallic=0.08, roughness=0.16, smooth_shading=True)
        if show_stbd:
            pl.add_mesh(pvp["stbd_mold"], color="#467e8c", opacity=0.40, pbr=True, metallic=0.08, roughness=0.16, smooth_shading=True)
        pl.camera.position = cam_pos
        pl.camera.focal_point = focal
        pl.camera.up = (0, 0, 1)
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = pscale
        pl.enable_anti_aliasing("ssaa")
        pl.render()
        Image.fromarray(pl.screenshot(return_img=True)).save(OUT / out_name, optimize=True)
        pl.close()

    _shoot("stepped_cradle_studio_quarter.png", (3200, 1550), (-5200, -5400, 2450), (0, 0, -190), 1220.0, show_stbd=True)
    _shoot("stepped_cradle_closeup.png", (3000, 1650), (-2100, -2950, 1280), (+250, 0, -20), 460.0, show_stbd=True)
    _shoot("stepped_cradle_cutaway_port_only.png", (3000, 1650), (-1950, +2950, 1220), (+250, 0, -20), 460.0, show_stbd=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    surf = SculptedKayakSurface(HULL_JSON)
    edges_x = np.linspace(-2754.0, +2754.0, 18)
    parts, tbl, sp_list = build_assembly(surf, edges_x)

    sc = trimesh.Scene()
    sc.add_geometry(_style(parts["port_mold"], "01a_smoked_port_h2s_mold_halves", [54, 96, 134, 102], alpha_mode="BLEND", rough=0.18, metal=0.06), node_name="01a_smoked_port_h2s_mold_halves")
    sc.add_geometry(_style(parts["stbd_mold"], "01b_smoked_stbd_h2s_mold_halves", [64, 122, 136, 102], alpha_mode="BLEND", rough=0.18, metal=0.06), node_name="01b_smoked_stbd_h2s_mold_halves")
    sc.add_geometry(_style(parts["cradle_yokes"], "02_stepped_baltic_birch_cradle_yokes", [228, 196, 148, 255], rough=0.48, metal=0.02), node_name="02_stepped_baltic_birch_cradle_yokes")
    sc.add_geometry(_style(parts["strongback_box"], "03_baltic_birch_torsion_box_spine", [198, 162, 112, 255], rough=0.58, metal=0.02), node_name="03_baltic_birch_torsion_box_spine")
    sc.add_geometry(_style(parts["wedges"], "04_8deg_phenolic_clamping_wedges", [200, 100, 40, 255], rough=0.34, metal=0.06), node_name="04_8deg_phenolic_clamping_wedges")
    sc.add_geometry(_style(parts["hardware"], "05_m8_clamp_bolts_keel_tierods_and_dowels", [224, 232, 240, 255], rough=0.16, metal=0.88), node_name="05_m8_clamp_bolts_keel_tierods_and_dowels")
    sc.add_geometry(_style(parts["trestles"], "06_bessel_point_rolling_shop_trestles", [48, 58, 70, 255], rough=0.36, metal=0.45), node_name="06_bessel_point_rolling_shop_trestles")
    sc.export(OUT / "kayak-10-9-h2s-stepped-cradle-strongback.glb")

    render_cross_section_diagram(tbl, sp_list, edges_x)
    render_3d_views(parts)
    print("BUILT STEPPED CRADLE STRONGBACK + H2S SPLIT SMOKED MOLD ASSEMBLY")


if __name__ == "__main__":
    main()
