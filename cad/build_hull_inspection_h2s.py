"""Build one-shot Bambu Lab H2S (340x320x340 mm) bare-hull inspection .3mf + .stl (3:50 / 1:16.67, 328 mm)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HULL_JSON = REPO_ROOT / "hull" / "kayak-10-9.hull.json"
OUT = REPO_ROOT / "build" / "hull_inspection_h2s"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
import vtk

from build_model_1to8_h2s import write_bambu_3mf
from deck_geometry import SculptedKayakSurface, smoothstep

SCALE_MM = 60.0        # Exact 3:50 (1:16.67) metric scale: 5.467 m -> 328.03 mm
Z_BASE_MM = 2.4        # Flat bed base plate thickness -> Z_max = 327.5 mm (< 340 mm H2S ceiling)
NECK_HALF_W = 0.28     # 0.56 mm single-extrusion snap-off V-throat on flat sheer back-face


def _triangulate_planar_cap(loop_xy: np.ndarray, z_val: float, normal_z: float) -> trimesh.Trimesh:
    area = float(np.sum(loop_xy[:, 0] * np.roll(loop_xy[:, 1], -1) - np.roll(loop_xy[:, 0], -1) * loop_xy[:, 1]))
    lp = loop_xy if area > 0.0 else loop_xy[::-1]
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE)
    lines = vtk.vtkCellArray()
    n = len(lp)
    lines.InsertNextCell(n + 1)
    for k, (ux, uy) in enumerate(lp):
        pts.InsertNextPoint(float(ux), float(uy), 0.0)
        lines.InsertCellPoint(k)
    lines.InsertCellPoint(0)
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(lines)
    ct = vtk.vtkContourTriangulator(); ct.SetInputData(pd); ct.Update()
    polys = ct.GetOutput().GetPolys(); polys.InitTraversal(); idl = vtk.vtkIdList(); fl = []
    while polys.GetNextCell(idl):
        if idl.GetNumberOfIds() == 3:
            fl.append([idl.GetId(0), idl.GetId(1), idl.GetId(2)])
    faces = np.asarray(fl, dtype=int)
    xyz = np.column_stack((lp[:, 0], lp[:, 1], np.full(n, z_val)))
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    flip = (np.cross(v1 - v0, v2 - v0)[:, 2] * normal_z) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)


def build_clean_bare_hull_mm(surf: SculptedKayakSurface) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build pure 1:16.67 watertight bare hull (below sheer seam, flat sheer chord) in print orientation."""
    xs = np.linspace(float(surf.xs[0]), float(surf.xs[-1]), 661)
    u_chord = np.linspace(1.0, -1.0, 23)[1:-1]  # interior sheer chord stbd -> port
    rings_3d: list[np.ndarray] = []
    for xv in xs:
        pos = int(np.clip(np.searchsorted(surf.xs, xv), 1, len(surf.xs) - 1))
        w = float((xv - surf.xs[pos - 1]) / max(surf.xs[pos] - surf.xs[pos - 1], 1e-9))
        hr = (1.0 - w) * surf.hull_rings[pos - 1] + w * surf.hull_rings[pos]
        sy = float((1.0 - w) * surf.seam_y[pos - 1] + w * surf.seam_y[pos])
        sz = float((1.0 - w) * surf.seam_z[pos - 1] + w * surf.seam_z[pos])
        chord = np.column_stack((u_chord * sy, np.full_like(u_chord, sz)))
        loop_yz = np.vstack((hr, chord))
        p = surf.end_shape_xyz(np.column_stack((np.full(len(loop_yz), xv), loop_yz))) * SCALE_MM
        rings_3d.append(p)
    arr = np.asarray(rings_3d)  # (nx, m_ring, 3) in boat mm; bow at i=0, stern at i=-1
    x_max = float(arr[:, :, 0].max())
    d_end = np.minimum(arr[:, :, 0] - float(arr[:, :, 0].min()), x_max - arr[:, :, 0])
    arr[:, :, 1] *= 1.0 + 0.85 * np.exp(-np.maximum(d_end, 0.0) / 2.0)

    # Convert to print frame (stern at Z=0, bow up at +Z, keel at -Y front door, sheer at +Y back)
    vp = np.empty_like(arr)
    vp[:, :, 0] = arr[:, :, 1]
    vp[:, :, 1] = arr[:, :, 2]
    vp[:, :, 2] = x_max - arr[:, :, 0]
    vp = vp[::-1].copy()  # now i=0 is stern (Z~0), i=-1 is bow (Z=328.03 mm)

    interior = vp[1:-1]
    nz, m, _ = interior.shape
    v_list = [vp[0].mean(axis=0)[None, :], interior.reshape(-1, 3), vp[-1].mean(axis=0)[None, :]]
    v_all = np.vstack(v_list)
    i_bow = len(v_all) - 1
    faces: list[list[int]] = []
    for j in range(m):
        j1 = (j + 1) % m
        faces.append([0, 1 + j1, 1 + j])
    for k in range(nz - 1):
        s, t = 1 + k * m, 1 + (k + 1) * m
        for j in range(m):
            j1 = (j + 1) % m
            faces.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]])
    top_s = 1 + (nz - 1) * m
    for j in range(m):
        j1 = (j + 1) % m
        faces.append([top_s + j, top_s + j1, i_bow])
    solid = trimesh.Trimesh(vertices=v_all, faces=np.asarray(faces, dtype=int), process=False)
    solid.merge_vertices(digits_vertex=5)
    if solid.volume < 0.0:
        solid.invert()
    return solid, vp


def _cap_annulus_z(outer_xy: np.ndarray, inner_xy: np.ndarray, z_val: float, normal_z: float) -> trimesh.Trimesh:
    a_out = float(np.sum(outer_xy[:, 0] * np.roll(outer_xy[:, 1], -1) - np.roll(outer_xy[:, 0], -1) * outer_xy[:, 1]))
    a_in = float(np.sum(inner_xy[:, 0] * np.roll(inner_xy[:, 1], -1) - np.roll(inner_xy[:, 0], -1) * inner_xy[:, 1]))
    lp_out = outer_xy if a_out > 0.0 else outer_xy[::-1]
    lp_in = inner_xy if a_in < 0.0 else inner_xy[::-1]  # CW hole
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE)
    lines = vtk.vtkCellArray()
    idx = 0
    for lp in (lp_out, lp_in):
        n = len(lp)
        lines.InsertNextCell(n + 1)
        for k, (ux, uy) in enumerate(lp):
            pts.InsertNextPoint(float(ux), float(uy), 0.0)
            lines.InsertCellPoint(idx + k)
        lines.InsertCellPoint(idx)
        idx += n
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(lines)
    ct = vtk.vtkContourTriangulator(); ct.SetInputData(pd); ct.Update()
    polys = ct.GetOutput().GetPolys(); polys.InitTraversal(); idl = vtk.vtkIdList(); fl = []
    while polys.GetNextCell(idl):
        if idl.GetNumberOfIds() == 3:
            fl.append([idl.GetId(0), idl.GetId(1), idl.GetId(2)])
    faces = np.asarray(fl, dtype=int)
    uv = np.vstack((lp_out, lp_in))
    xyz = np.column_stack((uv[:, 0], uv[:, 1], np.full(len(uv), z_val)))
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    flip = (np.cross(v1 - v0, v2 - v0)[:, 2] * normal_z) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)


def build_printable_hull_with_snap_spine(surf: SculptedKayakSurface) -> trimesh.Trimesh:
    """Build 1-piece watertight vertical hull + sheer-back 0.48 mm snap-off T-spine + flat Z=0 base."""
    xs = np.linspace(float(surf.xs[0]), float(surf.xs[-1]), 661)
    # 12-point back-chord / T-fin profile (stbd sheer -> +neck -> T-fin -> -neck -> port sheer)
    rings_boat: list[np.ndarray] = []
    sy_arr = np.zeros(len(xs))
    sz_arr = np.zeros(len(xs))
    for i, xv in enumerate(xs):
        pos = int(np.clip(np.searchsorted(surf.xs, xv), 1, len(surf.xs) - 1))
        w = float((xv - surf.xs[pos - 1]) / max(surf.xs[pos] - surf.xs[pos - 1], 1e-9))
        hr = (1.0 - w) * surf.hull_rings[pos - 1] + w * surf.hull_rings[pos]
        sy_arr[i] = float((1.0 - w) * surf.seam_y[pos - 1] + w * surf.seam_y[pos])
        sz_arr[i] = float((1.0 - w) * surf.seam_z[pos - 1] + w * surf.seam_z[pos])
        rings_boat.append(np.column_stack((np.full(len(hr), xv), hr)))

    hull_3d = np.asarray([surf.end_shape_xyz(r) * SCALE_MM for r in rings_boat])
    sheer_3d = surf.end_shape_xyz(np.column_stack((xs, sy_arr, sz_arr))) * SCALE_MM
    x_max = float(hull_3d[:, :, 0].max())
    d_end = np.minimum(hull_3d[:, :, 0] - float(hull_3d[:, :, 0].min()), x_max - hull_3d[:, :, 0])
    boost = 1.0 + 0.85 * np.exp(-np.maximum(d_end, 0.0) / 2.0)
    hull_3d[:, :, 1] *= boost
    sheer_3d[:, 1] *= boost[:, -1]

    # Flip stern-first (i=0 stern near Z=2.4 mm, i=-1 bow at Z=330.43 mm)
    hull_3d = hull_3d[::-1].copy()
    sheer_3d = sheer_3d[::-1].copy()

    # Start tower 2.4 mm above stern apex (sits inside the Z=0..2.4 mm snap-off base; total Z = 328.0 mm <= 340 mm)
    i_start = 6
    x_stern_cut = float(sheer_3d[i_start, 0])
    loops_3d: list[np.ndarray] = []
    for i in range(i_start, len(xs) - 1):
        hp = hull_3d[i]
        xh = hp[:, 1]
        sy_p = float(sheer_3d[i, 2])
        yh = np.minimum(hp[:, 2], sy_p)
        zh = (x_stern_cut - hp[:, 0]) + Z_BASE_MM
        sx = max(float(sheer_3d[i, 1]), 0.60)
        sz_p = float((x_stern_cut - sheer_3d[i, 0]) + Z_BASE_MM)

        # Dorsal T-spine height & depth: 36 mm deep at Z=2.4 mm, tapering at <=12 deg to 0 at Z=236 mm
        t_fin = float(1.0 - smoothstep(np.array((sz_p - 25.0) / 211.0)))
        r_mid = 0.65 * sx
        flat_x = np.linspace(r_mid, -r_mid, 12)
        fin_d = 34.0 * t_fin
        neck_w = min(NECK_HALF_W, 0.40 * sx)
        web_w = neck_w + 0.85
        flg_w = web_w + 4.8
        fin_x = np.array([r_mid, neck_w, web_w, web_w, flg_w, flg_w, -flg_w, -flg_w, -web_w, -web_w, -neck_w, -r_mid])
        fin_dy = np.array([0.0, 0.0, 0.85, fin_d - 2.2, fin_d - 1.0, fin_d, fin_d, fin_d - 1.0, fin_d - 2.2, 0.85, 0.0, 0.0])
        use_fin = float(smoothstep(np.array(t_fin / 0.08)))
        bx = (1.0 - use_fin) * flat_x + use_fin * fin_x
        by = sy_p + t_fin * np.maximum(fin_dy, 0.0)
        back_3d = np.column_stack((bx, by, np.full(12, sz_p)))
        loops_3d.append(np.vstack((np.column_stack((xh, yh, zh)), back_3d)))

    # Pin bottom ring (loops_3d[0]) planar at Z = Z_BASE_MM so it mates flat onto the base plate top!
    loops_3d[0][:, 2] = Z_BASE_MM
    arr = np.asarray(loops_3d)
    nz, m, _ = arr.shape

    # Rounded rectangular flat bed base plate (68 x 60 x 2.4 mm) encompassing bottom ring + T-fin
    cx, cy, rx, ry, r_c = 0.0, 22.0, 34.0, 30.0, 7.0
    base_pts: list[list[float]] = []
    for ang0, ox, oy in [(0.0, rx - r_c, ry - r_c), (0.5 * np.pi, -(rx - r_c), ry - r_c),
                         (np.pi, -(rx - r_c), -(ry - r_c)), (1.5 * np.pi, rx - r_c, -(ry - r_c))]:
        for th in np.linspace(ang0, ang0 + 0.5 * np.pi, 12, endpoint=False):
            base_pts.append([cx + ox + r_c * float(np.cos(th)), cy + oy + r_c * float(np.sin(th))])
    base_xy = np.asarray(base_pts, dtype=float)
    nb = len(base_xy)
    cap_bot = _triangulate_planar_cap(base_xy, 0.0, normal_z=-1.0)
    cap_step = _cap_annulus_z(base_xy, arr[0, :, :2], Z_BASE_MM, normal_z=+1.0)
    vb = np.vstack((np.column_stack((base_xy, np.zeros(nb))), np.column_stack((base_xy, np.full(nb, Z_BASE_MM)))))
    fb = [[j, (j + 1) % nb, nb + j] for j in range(nb)] + [[(j + 1) % nb, nb + (j + 1) % nb, nb + j] for j in range(nb)]
    wall_base = trimesh.Trimesh(vertices=vb, faces=np.asarray(fb, dtype=int), process=False)

    bow_apex = np.array([[0.0, float(hull_3d[-1, :, 2].mean()), float((x_stern_cut - hull_3d[-1, :, 0].min()) + Z_BASE_MM)]])
    v_tower = np.vstack((arr.reshape(-1, 3), bow_apex))
    i_apex = len(v_tower) - 1
    ft: list[list[int]] = []
    for k in range(nz - 1):
        s, t = k * m, (k + 1) * m
        for j in range(m):
            j1 = (j + 1) % m
            ft.extend([[s + j, t + j, s + j1], [s + j1, t + j, t + j1]])
    top_s = (nz - 1) * m
    for j in range(m):
        ft.append([top_s + j, i_apex, top_s + (j + 1) % m])
    tower = trimesh.Trimesh(vertices=v_tower, faces=np.asarray(ft, dtype=int), process=False)

    full = trimesh.util.concatenate([cap_bot, wall_base, cap_step, tower])
    full.merge_vertices(digits_vertex=5)
    trimesh.repair.fix_normals(full)
    if full.volume < 0.0:
        full.invert()
    return full


def build_desk_stand(surf: SculptedKayakSurface, x_boat_m: float) -> trimesh.Trimesh:
    """Contoured desk cradle at x_boat_m (1:16.67 scale); flat base also lets hull sit keel-up on its sheer flat."""
    pos = int(np.clip(np.searchsorted(surf.xs, x_boat_m), 1, len(surf.xs) - 1))
    hr = surf.hull_rings[pos] * SCALE_MM
    yh, zh = hr[:, 0], hr[:, 1]
    keep = np.abs(yh) <= max(float(yh.max()) * 0.78, 6.0)
    yc, zc = yh[keep], zh[keep] + 0.25  # +0.25 mm slip clearance
    zc_shift = zc - float(zc.min()) + 4.2
    w_half = max(float(np.abs(yc).max()) + 4.5, 13.0)
    top_z = float(zc_shift.max()) + 1.2
    prof = np.vstack((
        [[-w_half, 0.0], [w_half, 0.0], [w_half, top_z], [float(yc[-1]), top_z]],
        np.column_stack((yc[::-1], zc_shift[::-1])),
        [[float(yc[0]), top_z], [-w_half, top_z]],
    ))
    zs = np.linspace(0.0, 9.0, 10)
    np_p = len(prof)
    c0 = _triangulate_planar_cap(prof, 0.0, -1.0)
    c1 = _triangulate_planar_cap(prof, 9.0, +1.0)
    vw = np.column_stack((np.tile(prof, (len(zs), 1)), np.repeat(zs, np_p)))
    fw = []
    area = float(np.sum(prof[:, 0] * np.roll(prof[:, 1], -1) - np.roll(prof[:, 0], -1) * prof[:, 1]))
    for k in range(len(zs) - 1):
        s, t = k * np_p, (k + 1) * np_p
        for j in range(np_p):
            j1 = (j + 1) % np_p
            if area > 0:
                fw.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]])
            else:
                fw.extend([[s + j, t + j, s + j1], [s + j1, t + j, t + j1]])
    st = trimesh.util.concatenate([c0, c1, trimesh.Trimesh(vertices=vw, faces=np.asarray(fw, dtype=int), process=False)])
    st.merge_vertices(digits_vertex=5)
    if st.volume < 0.0:
        st.invert()
    # Lay flat on Z = 0 (extruded thickness 9 mm in Z, cradle profile in X-Y -> 0 deg overhangs!)
    st.apply_translation((-0.5 * (st.bounds[0, 0] + st.bounds[1, 0]), -0.5 * (st.bounds[0, 1] + st.bounds[1, 1]), 0.0))
    return st


def _style(m: trimesh.Trimesh, name: str, rgba: list[int], rough: float = 0.26, metal: float = 0.06) -> trimesh.Trimesh:
    mc = m.copy()
    mc.visual = TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=rgba, roughnessFactor=rough, metallicFactor=metal, doubleSided=True))
    mc.metadata["name"] = name
    return mc


def render_preview(printable: trimesh.Trimesh, clean_hull: trimesh.Trimesh, stands: list[tuple[str, trimesh.Trimesh, tuple[float, float, float]]]) -> None:
    import pyvista as pv

    def to_pv(tm_m: trimesh.Trimesh, feat: float = 46.0) -> pv.PolyData:
        v = np.asarray(tm_m.vertices, dtype=float)
        f = np.asarray(tm_m.faces, dtype=np.int64)
        pd = pv.PolyData(v, np.column_stack((np.full(len(f), 3, dtype=np.int64), f)).ravel())
        return pd.compute_normals(cell_normals=False, point_normals=True, split_vertices=True, feature_angle=feat, auto_orient_normals=False)

    pl = pv.Plotter(off_screen=True, window_size=(1550, 1350), lighting="none")
    pl.set_background("#121a24", top="#223244")
    for pos, inten in [((-380, -480, 520), 1.35), ((+580, -380, 460), 0.90), ((170, +620, 380), 0.75)]:
        lt = pv.Light(position=pos, focal_point=(170, 155, 155), color="white", intensity=inten)
        lt.positional = False; pl.add_light(lt)
    bed = pv.Cube(center=(170.0, 160.0, -1.5), x_length=340.0, y_length=320.0, z_length=3.0)
    pl.add_mesh(bed, color="#2b333e", pbr=True, metallic=0.35, roughness=0.58)
    for gx in range(20, 340, 40):
        pl.add_mesh(pv.Line((gx, 0, 0.15), (gx, 320, 0.15)), color="#425063", line_width=1.1)
    for gy in range(40, 320, 40):
        pl.add_mesh(pv.Line((0, gy, 0.15), (340, gy, 0.15)), color="#425063", line_width=1.1)
    vbox = pv.Box(bounds=(0, 340, 0, 320, 0, 340))
    pl.add_mesh(vbox.extract_feature_edges(boundary_edges=True, feature_edges=True, manifold_edges=False), color="#6889ab", line_width=1.6, opacity=0.55)
    mp = printable.copy(); mp.apply_translation((170.0, 148.0, 0.0))
    pl.add_mesh(to_pv(mp, 44), color="#eef3f8", pbr=True, metallic=0.06, roughness=0.24, smooth_shading=True)
    for _, sm, (tx, ty, tz) in stands:
        sp = sm.copy(); sp.apply_translation((tx, ty, tz))
        pl.add_mesh(to_pv(sp, 44), color="#d8b480", pbr=True, metallic=0.08, roughness=0.38, smooth_shading=True)
    pl.camera.position = (-210, -295, 380); pl.camera.focal_point = (170, 158, 155); pl.camera.up = (0, 0, 1)
    pl.enable_parallel_projection(); pl.camera.parallel_scale = 228.0; pl.enable_anti_aliasing("ssaa")
    img_bed = pl.screenshot(return_img=True); pl.close()

    # Panel 2: Snapped-free clean bare hull (keel-up inverted inspection pose!)
    pl2 = pv.Plotter(off_screen=True, window_size=(1550, 820), lighting="none")
    pl2.set_background("#121a24", top="#223244")
    ch = clean_hull.copy()
    v = np.asarray(ch.vertices)
    # Place horizontally keel-up (Z_up = -Y_print + sheer_offset -> flat sheer sits on Z=0, keel & forefoot face up!)
    ch.vertices = np.column_stack((v[:, 2] - 0.5 * 328.03, v[:, 0], float(v[:, 1].max()) - v[:, 1]))
    for pos, inten in [((-280, -260, 210), 1.30), ((+280, -220, 190), 0.85), ((0, +280, 160), 0.65)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 8), color="white", intensity=inten)
        lt.positional = False; pl2.add_light(lt)
    pl2.add_mesh(to_pv(ch, 50), color="#eef3f8", pbr=True, metallic=0.06, roughness=0.22, smooth_shading=True)
    pl2.camera.position = (+235, -205, 105); pl2.camera.focal_point = (+10, 0, 7); pl2.camera.up = (0, 0, 1)
    pl2.enable_parallel_projection(); pl2.camera.parallel_scale = 98.0; pl2.enable_anti_aliasing("ssaa")
    img_hull = pl2.screenshot(return_img=True); pl2.close()

    # Cross-section at Z = 120 mm showing 0.48 mm snap-off neck on flat sheer back-face
    sec = printable.section([0, 0, 1], [0, 0, 120.0])
    loop_xy = max(sec.discrete, key=len)[:, :2]
    y_keel = float(loop_xy[:, 1].min())
    y_sheer_neck = float(loop_xy[np.abs(loop_xy[:, 0]) > 8.0, 1].max())

    fig = plt.figure(figsize=(17.5, 9.2), dpi=180, facecolor="#0e151f", layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.15], height_ratios=[1.15, 0.85])
    ax_bed = fig.add_subplot(gs[:, 0])
    ax_bed.imshow(img_bed); ax_bed.set_axis_off()
    ax_bed.set_title("1. One-Shot Bambu H2S Plate (327.5 mm Z ≤ 340 mm, Keel Faces Front Door, 0 g Supports)", color="#f1f5f9", fontsize=10.8, fontweight="bold")
    ax_h = fig.add_subplot(gs[0, 1])
    ax_h.imshow(img_hull); ax_h.set_axis_off()
    ax_h.set_title("2. Snapped-Free 328 mm (1:16.67) Bare Hull — Sits Flat Keel-Up for Grazing-Light Inspection", color="#f1f5f9", fontsize=10.5, fontweight="bold")
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_facecolor("#15202e")
    ax_c.fill(loop_xy[:, 0], loop_xy[:, 1], color="#60a5fa", alpha=0.28)
    ax_c.plot(loop_xy[:, 0], loop_xy[:, 1], color="#e2e8f0", lw=2.0)
    ax_c.annotate("0.56 mm V-notched snap throat\n(on flat sheer back-face ONLY;\n0.00 mm contact on hull/keel/bilge)",
                  xy=(0.30, y_sheer_neck), xytext=(4.8, y_sheer_neck + 3.2),
                  color="#fbbf24", fontsize=8.5, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.6))
    ax_c.annotate("Pure faired keel & bilges\n(each XY slice = true station!)",
                  xy=(0.0, y_keel), xytext=(-16.8, y_keel + 0.6),
                  color="#34d399", fontsize=8.5, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#34d399", lw=1.6))
    ax_c.set_xlim(-19.5, 19.5); ax_c.set_ylim(-1.8, 27.0)
    ax_c.set_aspect("equal"); ax_c.grid(True, alpha=0.22, color="#64748b"); ax_c.tick_params(colors="#cbd5e1")
    ax_c.set_xlabel("X on build plate (mm)", color="#e2e8f0"); ax_c.set_ylabel("Y on build plate (mm)", color="#e2e8f0")
    ax_c.set_title("3. Horizontal Slice @ Z = 120 mm — Hull D-Section + Sheer-Back T-Beam Stabilizer", color="#f1f5f9", fontsize=10.3, fontweight="bold")
    fig.savefig(OUT / "h2s_hull_inspection_preview.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    surf = SculptedKayakSurface(HULL_JSON)
    clean_hull, _ = build_clean_bare_hull_mm(surf)
    printable = build_printable_hull_with_snap_spine(surf)
    st_fwd = build_desk_stand(surf, -1.53)
    st_aft = build_desk_stand(surf, +1.53)

    for nm, m in [("clean_hull", clean_hull), ("printable_hull_with_spine", printable), ("stand_fwd", st_fwd), ("stand_aft", st_aft)]:
        assert m.is_watertight and m.is_volume, f"{nm} must be watertight positive volume!"

    stands = [
        ("02a_desk_cradle_fwd", st_fwd, (86.0, 152.0, 0.0)),
        ("02b_desk_cradle_aft", st_aft, (254.0, 152.0, 0.0)),
    ]
    plate_all = [("01_kayak_10_9_bare_hull_1to16p7_snap_spine", printable, (170.0, 148.0, 0.0)), *stands]
    plate_hull_only = [("01_kayak_10_9_bare_hull_1to16p7_snap_spine", printable, (170.0, 148.0, 0.0))]

    for nm, m, (tx, ty, tz) in plate_all:
        b = m.bounds + np.array([tx, ty, tz])
        assert np.all(b[0] >= -1e-4) and np.all(b[1] <= [340.0, 320.0, 340.0]), f"{nm} out of H2S bounds: {b}"

    res_all = write_bambu_3mf(
        plate_all,
        OUT / "kayak-10-9-hull-inspection-h2s.3mf",
        title="Kayak 10.9 v0.2.1 — One-Shot Bare Hull Inspection Model (1:16.67, Bambu Lab H2S)",
        plate_name="Kayak 10.9 Bare Hull + Desk Cradles (H2S)",
    )
    res_hull = write_bambu_3mf(
        plate_hull_only,
        OUT / "kayak-10-9-hull-only-h2s.3mf",
        title="Kayak 10.9 v0.2.1 — Bare Hull Only (1:16.67, Bambu Lab H2S)",
        plate_name="Kayak 10.9 Bare Hull Only (H2S)",
    )
    printable.export(OUT / "kayak-10-9-hull-inspection-h2s-one-shot.stl")
    clean_hull.export(OUT / "kayak-10-9-bare-hull-1to16p7-clean.stl")

    sc = trimesh.Scene()
    mp = printable.copy(); mp.apply_translation((170.0, 148.0, 0.0))
    sc.add_geometry(_style(mp, "01_bare_hull_with_snap_spine", [238, 243, 248, 255]), node_name="01_bare_hull_with_snap_spine")
    for nm, sm, (tx, ty, tz) in stands:
        sp = sm.copy(); sp.apply_translation((tx, ty, tz))
        sc.add_geometry(_style(sp, nm, [216, 180, 128, 255], rough=0.38), node_name=nm)
    bed_g = trimesh.creation.box(extents=(340.0, 320.0, 2.0)); bed_g.apply_translation((170.0, 160.0, -1.0))
    sc.add_geometry(_style(bed_g, "00_bambu_h2s_340x320_build_plate", [44, 52, 62, 255], rough=0.6, metal=0.3), node_name="00_bambu_h2s_340x320_build_plate")
    sc.export(OUT / "kayak-10-9-hull-inspection-h2s-preview.glb", include_normals=True)

    render_preview(printable, clean_hull, stands)
    report = {
        "scale": "3:50 (1:16.67, 60.0 mm/m)",
        "clean_hull_extents_mm": [round(float(x), 2) for x in clean_hull.extents],
        "printable_with_spine_extents_mm": [round(float(x), 2) for x in printable.extents],
        "clean_hull_volume_cm3": round(float(clean_hull.volume) / 1000.0, 2),
        "printable_total_volume_cm3": round(float(printable.volume) / 1000.0, 2),
        "snap_neck_width_mm": round(2.0 * NECK_HALF_W, 2),
        "plate_with_stands_3mf": res_all,
        "hull_only_3mf": res_hull,
    }
    (OUT / "hull_inspection_h2s_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
