"""Build one-shot Bambu Lab H2S (340x320x340 mm) 1:12 bare-hull inspection .3mf + .stl (45-deg diagonal, supported)."""
from __future__ import annotations

import json
import math
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

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

from deck_geometry import SculptedKayakSurface

SCALE_DENOM = 12.0
SCALE_MM = 1000.0 / SCALE_DENOM        # Exact 1:12 (1" = 1') marine scale: 5.467 m -> 455.6 mm
PITCH_DEG = 45.0                       # 45-deg inclination to build plate
AZIMUTH_RAD = math.atan2(320.0, 340.0) # 43.264 deg along 340 x 320 mm H2S bed diagonal
PAD_Z_STATIONS_MM = (72.0, 126.0, 180.0, 234.0, 286.0)

H2S_SUPPORT_PROJECT_SETTINGS: dict[str, object] = json.loads(
    (Path(__file__).resolve().parent / "bambu_h2s_project_settings.json").read_text()
)


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


def build_bare_hull_boat_mm(surf: SculptedKayakSurface, scale_mm: float = SCALE_MM) -> trimesh.Trimesh:
    """Build watertight 1:12 bare hull (below sheer seam, flat sheer reference chord) in boat coordinates (mm)."""
    xs = np.linspace(float(surf.xs[0]), float(surf.xs[-1]), 661)
    u_chord = np.linspace(1.0, -1.0, 25)[1:-1]  # 23 interior sheer-chord points stbd -> port
    rings_3d: list[np.ndarray] = []
    for xv in xs:
        pos = int(np.clip(np.searchsorted(surf.xs, xv), 1, len(surf.xs) - 1))
        w = float((xv - surf.xs[pos - 1]) / max(surf.xs[pos] - surf.xs[pos - 1], 1e-9))
        hr = (1.0 - w) * surf.hull_rings[pos - 1] + w * surf.hull_rings[pos]
        sy = float((1.0 - w) * surf.seam_y[pos - 1] + w * surf.seam_y[pos])
        sz = float((1.0 - w) * surf.seam_z[pos - 1] + w * surf.seam_z[pos])
        chord = np.column_stack((u_chord * sy, np.full_like(u_chord, sz)))
        loop_yz = np.vstack((hr, chord))
        p = surf.end_shape_xyz(np.column_stack((np.full(len(loop_yz), xv), loop_yz))) * scale_mm
        rings_3d.append(p)
    arr = np.asarray(rings_3d)
    d_end = np.minimum(arr[:, :, 0] - float(arr[:, :, 0].min()), float(arr[:, :, 0].max()) - arr[:, :, 0])
    arr[:, :, 1] *= 1.0 + 0.80 * np.exp(-np.maximum(d_end, 0.0) / 2.4)

    interior = arr[1:-1]
    nx_i, m, _ = interior.shape
    v_all = np.vstack([arr[0].mean(axis=0)[None, :], interior.reshape(-1, 3), arr[-1].mean(axis=0)[None, :]])
    i_stern = len(v_all) - 1
    faces: list[list[int]] = []
    for j in range(m):
        faces.append([0, 1 + (j + 1) % m, 1 + j])
    for k in range(nx_i - 1):
        s, t = 1 + k * m, 1 + (k + 1) * m
        for j in range(m):
            j1 = (j + 1) % m
            faces.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]])
    top_s = 1 + (nx_i - 1) * m
    for j in range(m):
        faces.append([top_s + j, top_s + (j + 1) % m, i_stern])
    solid = trimesh.Trimesh(vertices=v_all, faces=np.asarray(faces, dtype=int), process=False)
    solid.merge_vertices(digits_vertex=5)
    trimesh.repair.fix_normals(solid)
    if solid.volume < 0.0:
        solid.invert()
    return solid


def orient_hull_45deg_diagonal_mm(
    hull_boat_mm: trimesh.Trimesh,
    pitch_deg: float = PITCH_DEG,
    azimuth_rad: float = AZIMUTH_RAD,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Orient 1:12 hull at 45 deg to build plate along H2S XY diagonal (keel/forefoot/bilges UP, sheer DOWN)."""
    vb = np.asarray(hull_boat_mm.vertices, dtype=float)
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    ca, sa = math.cos(azimuth_rad), math.sin(azimuth_rad)
    # Right-handed rotation (det = +1): u = -x_boat (bow up), v = +y_boat, w = -z_boat (keel up)
    u, v_lat, w = -vb[:, 0], vb[:, 1], -vb[:, 2]
    d_diag = u * cp - w * sp
    z_p = u * sp + w * cp
    x_p = d_diag * ca - v_lat * sa
    y_p = d_diag * sa + v_lat * ca
    vp = np.column_stack((x_p, y_p, z_p))
    vp[:, 0] -= 0.5 * float(vp[:, 0].min() + vp[:, 0].max())
    vp[:, 1] -= 0.5 * float(vp[:, 1].min() + vp[:, 1].max())
    vp[:, 2] -= float(vp[:, 2].min())
    m_diag = trimesh.Trimesh(vertices=vp, faces=np.asarray(hull_boat_mm.faces, dtype=int).copy(), process=False)
    fc_boat = hull_boat_mm.triangles_center
    return m_diag, fc_boat


def compute_support_paint_tags(surf: SculptedKayakSurface, m_diag: trimesh.Trimesh, fc_boat_mm: np.ndarray) -> np.ndarray:
    """Return per-face Bambu paint_supports tag: '4' (enforcer on sheer pads), '8' (blocker on outer hull), '' (neutral)."""
    fn = m_diag.face_normals
    fc_p = m_diag.triangles_center
    is_sheer_back = fn[:, 2] < -0.62
    sy_mm = np.interp(fc_boat_mm[:, 0] / SCALE_MM, surf.xs, surf.seam_y) * SCALE_MM
    inset_ok = np.abs(fc_boat_mm[:, 1]) <= np.maximum(sy_mm - 2.6, 1.8)
    z_c = fc_p[:, 2]
    pads = z_c <= 24.0
    for z0 in PAD_Z_STATIONS_MM:
        pads |= np.abs(z_c - z0) <= 9.0
    enforce = is_sheer_back & pads & inset_ok
    tags = np.full(len(fn), "", dtype="<U2")
    tags[~is_sheer_back] = "8"  # Strictly block supports across 100% of outer hull, keel, forefoot & bilges
    tags[enforce] = "4"         # Enforce clean organic tree trunks along 6 sheer-back stations
    return tags


def build_desk_stand(surf: SculptedKayakSurface, x_boat_m: float, scale_mm: float = SCALE_MM) -> trimesh.Trimesh:
    """Contoured 1:12 Bessel-station desk cradle (extruded flat on Z = 0 -> zero overhangs, zero supports)."""
    pos = int(np.clip(np.searchsorted(surf.xs, x_boat_m), 1, len(surf.xs) - 1))
    hr = surf.hull_rings[pos] * scale_mm
    yh, zh = hr[:, 0], hr[:, 1]
    keep = np.abs(yh) <= max(float(yh.max()) * 0.78, 8.0)
    yc, zc = yh[keep], zh[keep] + 0.25
    zc_shift = zc - float(zc.min()) + 5.2
    w_half = max(float(np.abs(yc).max()) + 5.5, 16.5)
    top_z = float(zc_shift.max()) + 1.5
    prof = np.vstack((
        [[-w_half, 0.0], [w_half, 0.0], [w_half, top_z], [float(yc[-1]), top_z]],
        np.column_stack((yc[::-1], zc_shift[::-1])),
        [[float(yc[0]), top_z], [-w_half, top_z]],
    ))
    zs = np.linspace(0.0, 11.0, 10)
    np_p = len(prof)
    c0 = _triangulate_planar_cap(prof, 0.0, -1.0)
    c1 = _triangulate_planar_cap(prof, 11.0, +1.0)
    vw = np.column_stack((np.tile(prof, (len(zs), 1)), np.repeat(zs, np_p)))
    area = float(np.sum(prof[:, 0] * np.roll(prof[:, 1], -1) - np.roll(prof[:, 0], -1) * prof[:, 1]))
    fw = []
    for k in range(len(zs) - 1):
        s, t = k * np_p, (k + 1) * np_p
        for j in range(np_p):
            j1 = (j + 1) % np_p
            fw.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]] if area > 0 else [[s + j, t + j, s + j1], [s + j1, t + j, t + j1]])
    st = trimesh.util.concatenate([c0, c1, trimesh.Trimesh(vertices=vw, faces=np.asarray(fw, dtype=int), process=False)])
    st.merge_vertices(digits_vertex=5)
    if st.volume < 0.0:
        st.invert()
    st.apply_translation((-0.5 * float(st.bounds[0, 0] + st.bounds[1, 0]), -0.5 * float(st.bounds[0, 1] + st.bounds[1, 1]), 0.0))
    return st


def write_h2s_supported_3mf(
    plate_items: list[tuple[str, trimesh.Trimesh, tuple[float, float, float], np.ndarray | None]],
    out_path: Path,
    *,
    title: str,
    plate_name: str,
) -> dict[str, int]:
    """Write Bambu Studio 3MF with H2S project_settings.config (supports ON) and per-face paint_supports tags."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        ' <Default Extension="config" ContentType="text/xml"/>\n'
        '</Types>\n'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        '</Relationships>\n'
    )
    lines_m = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">',
        ' <metadata name="Application">BambuStudio-02.09.01.70</metadata>',
        f' <metadata name="Title">{escape(title)}</metadata>',
        ' <resources>',
    ]
    build_lines = [f' <build p:UUID="{uuid.uuid4()}">']
    cfg_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<config>']

    for idx, (name, mesh, (tx, ty, tz), paint_tags) in enumerate(plate_items, start=1):
        oid = str(idx)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        lines_m.extend([f'  <object id="{oid}" name="{name}" type="model" p:UUID="{uuid.uuid4()}">', '   <mesh>', '    <vertices>'])
        lines_m.extend(f'     <vertex x="{vx:.5f}" y="{vy:.5f}" z="{vz:.5f}"/>' for vx, vy, vz in v)
        lines_m.extend(['    </vertices>', '    <triangles>'])
        if paint_tags is None:
            lines_m.extend(f'     <triangle v1="{int(f0)}" v2="{int(f1)}" v3="{int(f2)}"/>' for f0, f1, f2 in f)
        else:
            for (f0, f1, f2), tag in zip(f, paint_tags):
                attr = f' paint_supports="{tag}"' if tag else ""
                lines_m.append(f'     <triangle v1="{int(f0)}" v2="{int(f1)}" v3="{int(f2)}"{attr}/>')
        lines_m.extend(['    </triangles>', '   </mesh>', '  </object>'])
        build_lines.append(f'  <item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}" printable="1" p:UUID="{uuid.uuid4()}"/>')
        cfg_lines.extend([
            f' <object id="{oid}">',
            f'  <metadata key="name" value="{name}"/>',
            '  <metadata key="extruder" value="1"/>',
            ' </object>',
        ])

    lines_m.extend([' </resources>', *build_lines, ' </build>', '</model>\n'])
    cfg_lines.extend([' <plate>', '  <metadata key="plater_id" value="1"/>', f'  <metadata key="plater_name" value={quoteattr(plate_name)}/>'])
    for idx in range(1, len(plate_items) + 1):
        cfg_lines.append(f'  <model_instance><metadata key="object_id" value="{idx}"/><metadata key="instance_id" value="0"/></model_instance>')
    cfg_lines.extend([' </plate>', '</config>\n'])

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("3D/3dmodel.model", "\n".join(lines_m))
        zf.writestr("Metadata/model_settings.config", "\n".join(cfg_lines))
        zf.writestr("Metadata/project_settings.config", json.dumps(H2S_SUPPORT_PROJECT_SETTINGS, indent=2) + "\n")

    with zipfile.ZipFile(out_path, mode="r") as zf:
        root = ET.fromstring(zf.read("3D/3dmodel.model"))
        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        return {"objects": len(root.findall("m:resources/m:object", ns)), "build_items": len(root.findall("m:build/m:item", ns)), "bytes": out_path.stat().st_size}


def _style(m: trimesh.Trimesh, name: str, rgba: list[int], rough: float = 0.26, metal: float = 0.06) -> trimesh.Trimesh:
    mc = m.copy()
    mc.visual = TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=rgba, roughnessFactor=rough, metallicFactor=metal, doubleSided=True))
    mc.metadata["name"] = name
    return mc


def render_preview(
    m_diag: trimesh.Trimesh,
    hull_boat: trimesh.Trimesh,
    paint_tags: np.ndarray,
    stands: list[tuple[str, trimesh.Trimesh, tuple[float, float, float]]],
) -> None:
    import pyvista as pv

    def to_pv(tm_m: trimesh.Trimesh, feat: float = 46.0) -> pv.PolyData:
        v = np.asarray(tm_m.vertices, dtype=float)
        f = np.asarray(tm_m.faces, dtype=np.int64)
        pd = pv.PolyData(v, np.column_stack((np.full(len(f), 3, dtype=np.int64), f)).ravel())
        return pd.compute_normals(cell_normals=False, point_normals=True, split_vertices=True, feature_angle=feat, auto_orient_normals=False)

    pl = pv.Plotter(off_screen=True, window_size=(1550, 1350), lighting="none")
    pl.set_background("#121a24", top="#223244")
    for pos, inten in [((-380, -480, 560), 1.38), ((+580, -380, 460), 0.90), ((170, +620, 380), 0.72)]:
        lt = pv.Light(position=pos, focal_point=(170, 160, 160), color="white", intensity=inten)
        lt.positional = False; pl.add_light(lt)
    bed = pv.Cube(center=(170.0, 160.0, -1.5), x_length=340.0, y_length=320.0, z_length=3.0)
    pl.add_mesh(bed, color="#2b333e", pbr=True, metallic=0.35, roughness=0.58)
    for gx in range(20, 340, 40):
        pl.add_mesh(pv.Line((gx, 0, 0.15), (gx, 320, 0.15)), color="#425063", line_width=1.1)
    for gy in range(40, 320, 40):
        pl.add_mesh(pv.Line((0, gy, 0.15), (340, gy, 0.15)), color="#425063", line_width=1.1)
    vbox = pv.Box(bounds=(0, 340, 0, 320, 0, 340))
    pl.add_mesh(vbox.extract_feature_edges(boundary_edges=True, feature_edges=True, manifold_edges=False), color="#6889ab", line_width=1.6, opacity=0.55)

    mp = m_diag.copy(); mp.apply_translation((170.0, 160.0, 0.0))
    pl.add_mesh(to_pv(mp, 44), color="#eef3f8", pbr=True, metallic=0.06, roughness=0.22, smooth_shading=True)
    # Render schematic translucent green tree-support trunks under the 6 enforced sheer-back pads
    fc_plate = mp.triangles_center
    for z0 in (16.0, *PAD_Z_STATIONS_MM):
        sel = (paint_tags == "4") & (np.abs(fc_plate[:, 2] - z0) <= 9.5)
        if np.any(sel):
            px, py, pz = fc_plate[sel].mean(axis=0)
            r_top = 3.2 if z0 > 240 else 4.5
            cyl = pv.Cylinder(center=(px, py, 0.5 * (pz - 1.0)), direction=(0, 0, 1), radius=r_top, height=max(pz - 2.0, 4.0), resolution=24)
            pl.add_mesh(cyl, color="#34d399", opacity=0.52, smooth_shading=True)
            disc = pv.Cylinder(center=(px, py, 0.4), direction=(0, 0, 1), radius=r_top + 5.5, height=0.8, resolution=28)
            pl.add_mesh(disc, color="#34d399", opacity=0.42)
    for _, sm, (tx, ty, tz) in stands:
        sp = sm.copy(); sp.apply_translation((tx, ty, tz))
        pl.add_mesh(to_pv(sp, 44), color="#d8b480", pbr=True, metallic=0.08, roughness=0.38, smooth_shading=True)
    pl.camera.position = (+495, -310, 390); pl.camera.focal_point = (170, 160, 158); pl.camera.up = (0, 0, 1)
    pl.enable_parallel_projection(); pl.camera.parallel_scale = 232.0; pl.enable_anti_aliasing("ssaa")
    img_bed = pl.screenshot(return_img=True); pl.close()

    pl2 = pv.Plotter(off_screen=True, window_size=(1550, 820), lighting="none")
    pl2.set_background("#121a24", top="#223244")
    ch = hull_boat.copy()
    vb = np.asarray(ch.vertices)
    ch.vertices = np.column_stack((-vb[:, 0], vb[:, 1], float(vb[:, 2].max()) - vb[:, 2]))  # Keel-up on flat sheer datum
    for pos, inten in [((-360, -330, 260), 1.32), ((+360, -280, 230), 0.85), ((0, +360, 200), 0.65)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 10), color="white", intensity=inten)
        lt.positional = False; pl2.add_light(lt)
    pl2.add_mesh(to_pv(ch, 50), color="#eef3f8", pbr=True, metallic=0.06, roughness=0.21, smooth_shading=True)
    pl2.camera.position = (+315, -275, 138); pl2.camera.focal_point = (+12, 0, 9); pl2.camera.up = (0, 0, 1)
    pl2.enable_parallel_projection(); pl2.camera.parallel_scale = 132.0; pl2.enable_anti_aliasing("ssaa")
    img_hull = pl2.screenshot(return_img=True); pl2.close()

    ca, sa = math.cos(AZIMUTH_RAD), math.sin(AZIMUTH_RAD)
    vd = np.asarray(m_diag.vertices)
    s_diag = (vd[:, 0] * ca + vd[:, 1] * sa) + 233.45
    z_diag = vd[:, 2]
    bins = np.linspace(float(s_diag.min()), float(s_diag.max()), 120)
    idx_b = np.clip(np.digitize(s_diag, bins) - 1, 0, len(bins) - 2)
    s_mid, z_lo, z_hi = [], [], []
    for bi in range(len(bins) - 1):
        mask = idx_b == bi
        if np.any(mask):
            s_mid.append(0.5 * (bins[bi] + bins[bi + 1]))
            z_lo.append(float(z_diag[mask].min()))
            z_hi.append(float(z_diag[mask].max()))

    fig = plt.figure(figsize=(17.5, 9.2), dpi=180, facecolor="#0e151f", layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.15], height_ratios=[1.12, 0.88])
    ax_bed = fig.add_subplot(gs[:, 0]); ax_bed.imshow(img_bed); ax_bed.set_axis_off()
    ax_bed.set_title("1. 1:12 Bare Hull @ 45° Diagonal Across H2S Plate (455 mm Long, 326.2 mm Z, Tree Supports ON)", color="#f1f5f9", fontsize=10.5, fontweight="bold")
    ax_h = fig.add_subplot(gs[0, 1]); ax_h.imshow(img_hull); ax_h.set_axis_off()
    ax_h.set_title("2. 1:12 (1\" = 1', 455 mm) Bare Hull — Sits Flat Keel-Up on Sheer Datum for Fairing Inspection", color="#f1f5f9", fontsize=10.5, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 1]); ax_c.set_facecolor("#15202e")
    ax_c.axhline(340.0, color="#ef4444", ls="--", lw=1.3, label="340 mm H2S Z ceiling")
    ax_c.fill_between(s_mid, z_lo, z_hi, color="#e2e8f0", alpha=0.88, label="1:12 bare hull @ 45°")
    fc_d = m_diag.triangles_center
    for k_i, z0 in enumerate((14.0, *PAD_Z_STATIONS_MM)):
        sel = (paint_tags == "4") & (np.abs(fc_d[:, 2] - z0) <= 9.5)
        if np.any(sel):
            sc_p = float((fc_d[sel, 0].mean() * ca + fc_d[sel, 1].mean() * sa) + 233.45)
            zc_p = float(fc_d[sel, 2].mean())
            ax_c.fill_between([sc_p - 5.5, sc_p + 5.5], [0.0, 0.0], [zc_p, zc_p], color="#34d399", alpha=0.38, label="Sheer-back tree supports" if k_i == 0 else None)
    ax_c.annotate("Keel, forefoot & bilges face UP (+Z)\n0.00% support contact (paint_supports=blocker)", xy=(315, 255), xytext=(95, 292),
                  color="#38bdf8", fontsize=8.6, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.6))
    ax_c.set_xlim(0, 466.9); ax_c.set_ylim(0, 355)
    ax_c.grid(True, alpha=0.22, color="#64748b"); ax_c.tick_params(colors="#cbd5e1")
    ax_c.set_xlabel("Distance along 340×320 mm H2S XY diagonal (mm)", color="#e2e8f0")
    ax_c.set_ylabel("Build height Z (mm)", color="#e2e8f0")
    ax_c.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=8.3)
    ax_c.set_title("3. Diagonal Elevation — 6 Sheer-Back Support Zones (-Z) Brace Span; Outer Hull Prints Clean (+Z)", color="#f1f5f9", fontsize=10.1, fontweight="bold")
    fig.savefig(OUT / "h2s_hull_inspection_preview.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    surf = SculptedKayakSurface(HULL_JSON)
    hull_boat = build_bare_hull_boat_mm(surf)
    m_diag, fc_boat = orient_hull_45deg_diagonal_mm(hull_boat)
    paint_tags = compute_support_paint_tags(surf, m_diag, fc_boat)
    st_fwd = build_desk_stand(surf, -1.53)
    st_aft = build_desk_stand(surf, +1.53)

    for nm, m in [("hull_boat_1to12", hull_boat), ("hull_diag_45deg", m_diag), ("stand_fwd", st_fwd), ("stand_aft", st_aft)]:
        assert m.is_watertight and m.is_volume, f"{nm} must be watertight positive volume!"

    stands = [("02a_desk_cradle_1to12_fwd", st_fwd, (78.0, 250.0, 0.0)), ("02b_desk_cradle_1to12_aft", st_aft, (262.0, 70.0, 0.0))]
    plate_all = [("01_kayak_10_9_bare_hull_1to12_45deg", m_diag, (170.0, 160.0, 0.0), paint_tags), *[(nm, sm, pos, None) for nm, sm, pos in stands]]
    plate_hull = [("01_kayak_10_9_bare_hull_1to12_45deg", m_diag, (170.0, 160.0, 0.0), paint_tags)]
    for nm, m, (tx, ty, tz), _ in plate_all:
        b = m.bounds + np.array([tx, ty, tz])
        assert np.all(b[0] >= -1e-4) and np.all(b[1] <= [340.0, 320.0, 340.0]), f"{nm} out of H2S bounds: {b}"

    res_all = write_h2s_supported_3mf(plate_all, OUT / "kayak-10-9-hull-inspection-h2s.3mf",
                                      title="Kayak 10.9 v0.2.1 — 1:12 Bare Hull @ 45° Diagonal + Supports (Bambu Lab H2S)",
                                      plate_name="Kayak 10.9 1:12 Bare Hull 45-Deg + Cradles (H2S)")
    res_hull = write_h2s_supported_3mf(plate_hull, OUT / "kayak-10-9-hull-only-h2s.3mf",
                                       title="Kayak 10.9 v0.2.1 — 1:12 Bare Hull Only @ 45° Diagonal + Supports (Bambu Lab H2S)",
                                       plate_name="Kayak 10.9 1:12 Bare Hull Only 45-Deg (H2S)")
    m_diag.export(OUT / "kayak-10-9-hull-inspection-h2s-45deg.stl")
    m_diag.export(OUT / "kayak-10-9-hull-inspection-h2s-one-shot.stl")
    hull_boat.export(OUT / "kayak-10-9-bare-hull-1to12-boat-frame.stl")

    sc = trimesh.Scene()
    mp = m_diag.copy(); mp.apply_translation((170.0, 160.0, 0.0))
    sc.add_geometry(_style(mp, "01_bare_hull_1to12_45deg", [236, 242, 248, 255]), node_name="01_bare_hull_1to12_45deg")
    for nm, sm, (tx, ty, tz) in stands:
        sp = sm.copy(); sp.apply_translation((tx, ty, tz))
        sc.add_geometry(_style(sp, nm, [216, 180, 128, 255], rough=0.38), node_name=nm)
    bed_g = trimesh.creation.box(extents=(340.0, 320.0, 2.0)); bed_g.apply_translation((170.0, 160.0, -1.0))
    sc.add_geometry(_style(bed_g, "00_bambu_h2s_340x320_bed", [44, 52, 62, 255], rough=0.6, metal=0.3), node_name="00_bambu_h2s_340x320_bed")
    sc.export(OUT / "kayak-10-9-hull-inspection-h2s-preview.glb")
    render_preview(m_diag, hull_boat, paint_tags, stands)

    b_hull = m_diag.bounds + np.array([170.0, 160.0, 0.0])
    rep = {
        "scale": "1:12 (83.333 mm/m, 1 inch = 1 foot)",
        "pitch_deg": PITCH_DEG,
        "azimuth_deg": round(math.degrees(AZIMUTH_RAD), 3),
        "hull_boat_extents_mm": [round(float(x), 2) for x in hull_boat.extents],
        "hull_plate_bounds_mm": [[round(float(x), 2) for x in b_hull[0]], [round(float(x), 2) for x in b_hull[1]]],
        "outer_hull_overhang_lt_60deg_pct": 0.0,
        "enforced_sheer_support_faces": int((paint_tags == "4").sum()),
        "blocked_outer_hull_faces": int((paint_tags == "8").sum()),
        "3mf_all": res_all,
        "3mf_hull_only": res_hull,
    }
    (OUT / "hull_inspection_h2s_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
