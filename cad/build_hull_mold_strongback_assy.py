"""Build full-length 17-station Smoked Female Hull Mold + Plywood Torsion-Box Strongback + Shop Trestles 3D Assembly."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HULL_JSON = REPO_ROOT / "hull" / "kayak-10-9.hull.json"
BUILD_ROOT = REPO_ROOT / "build"
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = BUILD_ROOT / "mold_strongback_assy"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
import vtk
from deck_geometry import SculptedKayakSurface


def _box(cx: float, cy: float, cz: float, lx: float, ly: float, lz: float) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(lx, ly, lz))
    m.apply_translation((cx, cy, cz))
    return m


def _cyl(p0: tuple[float, float, float], p1: tuple[float, float, float], radius: float, sections: int = 20) -> trimesh.Trimesh:
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


def _triangulate_loop_2d(loop_uv: np.ndarray, x_val: float, normal_x: float, holes_uv: list[np.ndarray] | None = None) -> trimesh.Trimesh:
    """Triangulate a 2D (y, z) polygon (with optional CW holes) on plane X = x_val via vtkContourTriangulator."""
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

def _resample_2d(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    su = np.linspace(0.0, float(seg[-1]), n)
    return np.column_stack((np.interp(su, seg, pts[:, 0]), np.interp(su, seg, pts[:, 1])))


def _precompute_mold_tables(surf: SculptedKayakSurface) -> dict[str, np.ndarray]:
    rings_mm = surf.mold_station_rings_mm()
    xs_mm = surf.xs * 1000.0
    z_raw, y_raw = np.zeros(len(xs_mm)), np.zeros(len(xs_mm))
    for i, xm in enumerate(xs_mm):
        stbd = rings_mm[i][rings_mm[i][:, 1] >= -1e-4][:, 1:]
        stbd = stbd[np.argsort(stbd[:, 1])]
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


def _mold_loop_at_x_mm(tbl: dict[str, np.ndarray], x_mm: float) -> np.ndarray:
    """Closed CCW (y, z) cross-section of strongback-mounted Female Hull Mold (+11.8x8 mm y=0 key tongue)."""
    xs_mm, rings_mm = tbl["xs_mm"], tbl["rings_mm"]
    xc = float(np.clip(x_mm, xs_mm[0] + 0.05, xs_mm[-1] - 0.05))
    i1 = int(np.clip(np.searchsorted(xs_mm, xc), 1, len(xs_mm) - 1))
    w = float((xc - xs_mm[i1 - 1]) / max(xs_mm[i1] - xs_mm[i1 - 1], 1e-6))
    ring = (1.0 - w) * rings_mm[i1 - 1] + w * rings_mm[i1]
    stbd = ring[ring[:, 1] >= -1e-4][:, 1:]
    stbd = stbd[np.argsort(stbd[:, 1])]
    stbd[0, 0] = 0.0

    zm = float(np.interp(x_mm, tbl["xg"], tbl["zg"]))
    ym_fair = float(np.interp(x_mm, tbl["xg"], tbl["yg"]))
    y_at_zm = float(np.interp(zm, stbd[:, 1], stbd[:, 0]))
    sub = stbd[stbd[:, 1] < zm]
    hs = np.vstack((sub, [[y_at_zm, zm]])) if len(sub) >= 2 else np.array([[0.0, zm - 1.0], [max(y_at_zm, 0.5), zm]])
    hs[:, 0] = np.maximum.accumulate(np.maximum(hs[:, 0], 0.0))
    hs_r = _resample_2d(hs, 56)

    # Smoothly close inner cavity into solid end dam across |x| in [2734.0, 2735.6] mm (zero internal ghost wall!)
    t_dam = float(np.clip((abs(x_mm) - 2734.0) / 1.6, 0.0, 1.0))
    t_dam = t_dam * t_dam * (3.0 - 2.0 * t_dam)
    y_dam = np.linspace(0.15, 8.0, len(hs_r))
    hs_r[:, 0] = (1.0 - t_dam) * np.maximum(hs_r[:, 0], np.linspace(0.05, max(float(hs_r[-1, 0]), 1.2), len(hs_r))) + t_dam * y_dam
    hs_r[:, 1] = (1.0 - t_dam) * hs_r[:, 1] + t_dam * zm

    ym = float(hs_r[-1, 0])
    y_fl = max(ym + 34.0, ym_fair + 34.0, 48.0)
    y_foot = min(120.0, max(y_fl, 56.0))
    fl_p = _resample_2d(np.array([[-y_fl, zm], [-ym, zm]]), 10)[:-1]
    cav_p = np.column_stack((-hs_r[::-1, 0], hs_r[::-1, 1]))[:-1]
    cav_s = hs_r[:-1]
    fl_s = _resample_2d(np.array([[+ym, zm], [+y_fl, zm]]), 10)
    top_seg = np.vstack((fl_p, cav_p, cav_s, fl_s))

    z_sh = max(zm - 16.0, 20.0)
    y_neck = max(y_foot, y_fl - 22.0)
    z_neck = max(min(float(hs_r[0, 1]), zm - 24.0) + 20.0, 16.0)
    s_side = _resample_2d(np.array([[y_fl, zm], [y_fl, z_sh], [y_neck, z_neck], [y_foot, 8.0], [y_foot, -20.0]]), 24)[1:]
    foot_key = np.array([[5.9, -20.0], [5.9, -28.0], [-5.9, -28.0], [-5.9, -20.0]])
    p_side = np.column_stack((-s_side[::-1, 0], s_side[::-1, 1]))
    loop = np.vstack((top_seg, s_side, foot_key, p_side))
    area = float(np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1]))
    return loop if area > 0.0 else loop[::-1]


def _loft_station_prism(tbl: dict[str, np.ndarray], x0_mm: float, x1_mm: float, xs_override: np.ndarray | None = None) -> trimesh.Trimesh:
    xs = xs_override if xs_override is not None else np.linspace(x0_mm, x1_mm, 17)
    loops = np.asarray([_mold_loop_at_x_mm(tbl, float(xv)) for xv in xs])
    c0 = _triangulate_loop_2d(loops[0], float(xs[0]), normal_x=-1.0)
    c1 = _triangulate_loop_2d(loops[-1], float(xs[-1]), normal_x=+1.0)
    nx, m, _ = loops.shape
    vw = np.column_stack((np.repeat(xs, m), loops.reshape(-1, 2)))
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

def _side_web_with_oval_ports(y_in: float, y_out: float, x_min: float, x_max: float, z_min: float, z_max: float, port_xs: np.ndarray) -> trimesh.Trimesh:
    """18 mm Baltic-birch side web (x in [x_min, x_max], z in [z_min, z_max]) with 16 stadium wrench-access ports."""
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE)
    lines = vtk.vtkCellArray()
    outer = np.array([[x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]], dtype=float)
    loops = [outer]
    th = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    zc = 0.5 * (z_min + z_max)
    for xc in port_xs:
        # CW stadium/oval cutout (196 mm x 96 mm) for hand/wrench access to M8 hold-down nuts & weight reduction
        oval = np.column_stack((xc + 98.0 * np.cos(th), zc - 48.0 * np.sin(th)))
        loops.append(oval)
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
    rim_f = []
    cur = 0
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

def build_strongback_and_trestles(surf: SculptedKayakSurface, edges_x: np.ndarray) -> dict[str, trimesh.Trimesh]:
    """Assemble 17 smoked mold stations, Baltic-birch torsion box, internal egg-crate, hardware & Bessel trestles."""
    tbl = _precompute_mold_tables(surf)
    odd_st, even_st, dowels, bolts = [], [], [], []
    n_st = len(edges_x) - 1  # 17 stations across [-2754, +2754] mm (324.0 mm each <= 340 mm H2S Z-height!)
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])

    for k in range(n_st):
        xa, xb = float(edges_x[k]) + 0.7, float(edges_x[k + 1]) - 0.7  # 1.4 mm crisp visual seam gap
        if k == 0:
            xs_ov = np.unique(np.concatenate([np.linspace(xa, -2733.0, 9), np.linspace(-2733.0, xb, 15)]))
        elif k == n_st - 1:
            xs_ov = np.unique(np.concatenate([np.linspace(xa, +2733.0, 15), np.linspace(+2733.0, xb, 9)]))
        else:
            xs_ov = None
        st_mesh = _loft_station_prism(tbl, xa, xb, xs_override=xs_ov)
        (odd_st if k % 2 == 0 else even_st).append(st_mesh)

        # M8 hold-down hex bolts + wide Ø26 mm fender washers + internal locknuts (at y = ±72 mm where foot >= 88 mm)
        xc = float(centers_x[k])
        lp_c = _mold_loop_at_x_mm(tbl, xc)
        max_y_c = float(np.max(np.abs(lp_c[:, 0])))
        yb = min(74.0, max(32.0, max_y_c - 22.0))
        for sgn in (-1.0, +1.0):
            bolts.append(_cyl((xc, sgn * yb, +4.0), (xc, sgn * yb, -52.0), radius=4.0, sections=14))
            bolts.append(_cyl((xc, sgn * yb, -2.0), (xc, sgn * yb, +4.0), radius=7.5, sections=6))    # M8 hex head
            bolts.append(_cyl((xc, sgn * yb, -5.0), (xc, sgn * yb, -2.0), radius=13.0, sections=20))  # Ø26 fender washer
            bolts.append(_cyl((xc, sgn * yb, -44.0), (xc, sgn * yb, -38.0), radius=12.0, sections=20))
            bolts.append(_cyl((xc, sgn * yb, -50.0), (xc, sgn * yb, -44.0), radius=7.5, sections=6))  # internal locknut

    # 4x Ø10x48 mm stainless station-to-station alignment dowels bridging every seam inside the smoked mold!
    for xs_j in edges_x[1:-1]:
        xj = float(xs_j)
        lp_j = _mold_loop_at_x_mm(tbl, xj)
        ym_j = float(np.max(lp_j[:, 0])) - 34.0
        zm_j = float(np.max(lp_j[:, 1]))
        for sgn in (-1.0, +1.0):
            dowels.append(_cyl((xj - 24.0, sgn * (ym_j + 18.0), zm_j - 8.0), (xj + 24.0, sgn * (ym_j + 18.0), zm_j - 8.0), radius=5.0, sections=16))
            dowels.append(_cyl((xj - 24.0, sgn * min(92.0, max(ym_j, 34.0)), -11.0), (xj + 24.0, sgn * min(92.0, max(ym_j, 34.0)), -11.0), radius=5.0, sections=16))

    # Baltic-birch torsion-box strongback: 5,540 x 240 x 190 mm (z in [-210, -20] mm, 18 mm birch ply)
    L_sb = 5540.0
    skins = [
        _box(0.0, -64.0, -29.0, L_sb, 112.0, 18.0),  # Port top skin half (leaves 16 mm y=0 keyway!)
        _box(0.0, +64.0, -29.0, L_sb, 112.0, 18.0),  # Starboard top skin half
        _box(0.0, 0.0, -34.0, L_sb, 16.0, 8.0),      # Bottom floor of routed centerline keyway
        _box(0.0, 0.0, -201.0, L_sb, 240.0, 18.0),   # Full 240 mm bottom skin
        _side_web_with_oval_ports(-102.0, -120.0, -0.5 * L_sb, +0.5 * L_sb, -192.0, -38.0, centers_x),
        _side_web_with_oval_ports(+102.0, +120.0, -0.5 * L_sb, +0.5 * L_sb, -192.0, -38.0, centers_x),
    ]
    # Internal 18 mm birch egg-crate bulkheads at all 18 station seam stations + central spine web
    eggcrate = [_box(float(xb_pos), 0.0, -115.0, 18.0, 204.0, 154.0) for xb_pos in edges_x]
    eggcrate.append(_box(0.0, 0.0, -148.0, L_sb - 36.0, 18.0, 88.0))  # Central notched spine web

    # Anodized aluminum y=0 alignment keyway channel
    keyway = [_box(0.0, -7.0, -25.0, L_sb, 2.0, 10.0), _box(0.0, +7.0, -25.0, L_sb, 2.0, 10.0), _box(0.0, 0.0, -29.5, L_sb, 16.0, 1.5)]

    # Two Bessel-point (-1530 mm & +1530 mm) rolling steel shop trestles (floor @ z = -860 mm -> 40" ergonomic height!)
    trestles = []
    for xt in (-1530.0, +1530.0):
        trestles.append(_box(xt, 0.0, -238.0, 140.0, 340.0, 44.0))   # Top saddle crossbeam
        trestles.append(_box(xt, 0.0, -730.0, 90.0, 620.0, 40.0))    # Lower tie shelf
        for sy in (-1.0, +1.0):
            trestles.append(_box(xt, sy * 88.0, -213.0, 90.0, 36.0, 6.0))  # Leveling shim pads under strongback
            for sx in (-1.0, +1.0):
                trestles.append(_cyl((xt + sx * 42.0, sy * 135.0, -260.0), (xt + sx * 135.0, sy * 295.0, -805.0), radius=21.0, sections=16))
                # Locking shop caster wheel assembly on z = -860 mm floor
                cx_w, cy_w = xt + sx * 135.0, sy * 295.0
                trestles.append(_box(cx_w, cy_w, -812.0, 56.0, 56.0, 14.0))
                p0_w = (cx_w, cy_w - 14.0, -836.0)
                p1_w = (cx_w, cy_w + 14.0, -836.0)
                trestles.append(_cyl(p0_w, p1_w, radius=24.0, sections=22))

    return {
        "odd_mold": trimesh.util.concatenate(odd_st),
        "even_mold": trimesh.util.concatenate(even_st),
        "strongback_skins": trimesh.util.concatenate(skins),
        "strongback_eggcrate": trimesh.util.concatenate(eggcrate),
        "hardware": trimesh.util.concatenate(keyway + bolts + dowels),
        "trestles": trimesh.util.concatenate(trestles),
    }

def render_studio_and_ortho(parts: dict[str, trimesh.Trimesh], edges_x: np.ndarray) -> None:
    """Render high-res PyVista smoked-mold studio quarter, X-ray midship/trestle close-up, and side elevation."""
    import pyvista as pv
    from PIL import Image

    def to_pv(tm_m: trimesh.Trimesh, feat: float = 52.0) -> pv.PolyData:
        v = np.asarray(tm_m.vertices, dtype=float)
        f = np.asarray(tm_m.faces, dtype=np.int64)
        pd = pv.PolyData(v, np.column_stack((np.full(len(f), 3, dtype=np.int64), f)).ravel())
        return pd.compute_normals(cell_normals=False, point_normals=True, split_vertices=True, feature_angle=feat, auto_orient_normals=False)

    pv_parts = {
        "trestles": to_pv(parts["trestles"], 45),
        "eggcrate": to_pv(parts["strongback_eggcrate"], 45),
        "skins": to_pv(parts["strongback_skins"], 45),
        "hardware": to_pv(parts["hardware"], 45),
        "odd_mold": to_pv(parts["odd_mold"], 58),
        "even_mold": to_pv(parts["even_mold"], 58),
    }

    def _shoot(out_name: str, win_sz: tuple[int, int], cam_pos: tuple[float, float, float], focal: tuple[float, float, float], pscale: float) -> None:
        pl = pv.Plotter(off_screen=True, window_size=win_sz, lighting="none")
        pl.set_background("#17212e", top="#29394d")
        lights = [
            ((-5500, -6500, 5800), 1.50, "#fffdf9"),
            ((+6000, -4800, 4600), 0.92, "#e8f3ff"),
            ((0, +7000, 4800), 0.78, "#ffffff"),
            ((-1600, -5200, -450), 0.55, "#fff4e2"),  # Warm fill into strongback side-web cutouts
        ]
        for pos, inten, col in lights:
            lt = pv.Light(position=pos, focal_point=focal, color=col, intensity=inten)
            lt.positional = False
            pl.add_light(lt)
        pl.add_mesh(pv_parts["trestles"], color="#333e4c", pbr=True, metallic=0.45, roughness=0.36, smooth_shading=True)
        pl.add_mesh(pv_parts["eggcrate"], color="#c49d68", pbr=True, metallic=0.02, roughness=0.58, smooth_shading=True)
        pl.add_mesh(pv_parts["skins"], color="#e1c193", pbr=True, metallic=0.03, roughness=0.48, smooth_shading=True)
        pl.add_mesh(pv_parts["hardware"], color="#e4eaf2", pbr=True, metallic=0.88, roughness=0.16, smooth_shading=True)
        pl.enable_depth_peeling(number_of_peels=10)
        pl.add_mesh(pv_parts["odd_mold"], color="#3d5875", opacity=0.40, pbr=True, metallic=0.08, roughness=0.16, smooth_shading=True)
        pl.add_mesh(pv_parts["even_mold"], color="#587b9e", opacity=0.40, pbr=True, metallic=0.08, roughness=0.16, smooth_shading=True)
        pl.camera.position = cam_pos
        pl.camera.focal_point = focal
        pl.camera.up = (0, 0, 1)
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = pscale
        pl.enable_anti_aliasing("ssaa")
        pl.render()
        Image.fromarray(pl.screenshot(return_img=True)).save(OUT / out_name, optimize=True)
        pl.close()

    _shoot("mold_strongback_studio_quarter.png", (3200, 1550), (-5200, -5400, 2350), (0, 0, -210), 1200.0)
    _shoot("mold_strongback_closeup.png", (3000, 1600), (-2550, -3150, 1180), (-820, 0, -110), 470.0)
    _shoot("mold_strongback_side_elevation.png", (3400, 1150), (0, -6500, -290), (0, 0, -290), 1020.0)


def main() -> None:
    surf = SculptedKayakSurface(HULL_JSON)
    edges_x = np.linspace(-2754.0, +2754.0, 18)  # 17 stations x 324.0 mm = 5,508 mm (each <= 340 mm H2S Z-height!)
    parts = build_strongback_and_trestles(surf, edges_x)

    sc = trimesh.Scene()
    sc.add_geometry(_style(parts["odd_mold"], "01a_smoked_hull_mold_odd_stations", [52, 72, 94, 98], alpha_mode="BLEND", rough=0.18, metal=0.06), node_name="01a_smoked_hull_mold_odd_stations")
    sc.add_geometry(_style(parts["even_mold"], "01b_smoked_hull_mold_even_stations", [76, 100, 126, 98], alpha_mode="BLEND", rough=0.18, metal=0.06), node_name="01b_smoked_hull_mold_even_stations")
    sc.add_geometry(_style(parts["strongback_skins"], "02_baltic_birch_strongback_skins", [218, 186, 138, 255], rough=0.55, metal=0.02), node_name="02_baltic_birch_strongback_skins")
    sc.add_geometry(_style(parts["strongback_eggcrate"], "03_internal_birch_eggcrate_bulkheads", [186, 148, 98, 255], rough=0.62, metal=0.02), node_name="03_internal_birch_eggcrate_bulkheads")
    sc.add_geometry(_style(parts["hardware"], "04_keyway_holddown_bolts_and_dowels", [214, 220, 228, 255], rough=0.18, metal=0.86), node_name="04_keyway_holddown_bolts_and_dowels")
    sc.add_geometry(_style(parts["trestles"], "05_bessel_point_rolling_shop_trestles", [44, 52, 62, 255], rough=0.38, metal=0.45), node_name="05_bessel_point_rolling_shop_trestles")
    sc.export(OUT / "kayak-10-9-smoked-hull-mold-and-strongback.glb")

    render_studio_and_ortho(parts, edges_x)
    print("BUILT SMOKED HULL MOLD + PLYWOOD STRONGBACK 3D ASSEMBLY & RENDERS")


if __name__ == "__main__":
    main()
