"""Build 1:8 scale (683.75 mm) Kayak 10.9 printable Bambu Lab H2S .3mf plate + watertight .stl files."""

from __future__ import annotations

import glob
import io
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HULL_JSON = REPO_ROOT / "hull" / "kayak-10-9.hull.json"
BUILD_ROOT = REPO_ROOT / "build"
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = BUILD_ROOT / "model_1to8_h2s"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
import vtk

from deck_geometry import HATCHES, SculptedKayakSurface, keyhole_inner_curve, keyhole_offset_curve, keyhole_signed_distance, smoothstep

SCALE_MM = 125.0  # Exact 1:8 scale: 1.000 m -> 125.000 mm (5.470 m -> 683.750 mm)
X_BH_FWD = -0.9500  # Forward watertight bulkhead (m) -> Z_bow = 223.125 mm
X_BH_AFT = +0.4400  # Aft seat watertight bulkhead (m) -> Z_cockpit = 173.750 mm, Z_stern = 286.875 mm


def _oct_loop(cy: float, cz: float, radius: float, cw: bool = True) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + (np.pi / 8.0)
    if cw:
        th = th[::-1]
    return np.column_stack((cy + radius * np.cos(th), cz + radius * np.sin(th)))


def _cap_with_blind_sockets_x(
    outer_yz: np.ndarray,
    x_cap: float,
    normal_x: float,
    sockets: list[tuple[float, float, float, float]],
) -> trimesh.Trimesh:
    """Triangulate flat bulkhead cap at X = x_cap with blind 45-deg-coned octagonal alignment sockets.
    sockets: list of (cy_mm, cz_mm, radius_mm, depth_mm) extending inward (-normal_x * depth_mm).
    """
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE)
    lines = vtk.vtkCellArray()
    area = float(np.sum(outer_yz[:, 0] * np.roll(outer_yz[:, 1], -1) - np.roll(outer_yz[:, 0], -1) * outer_yz[:, 1]))
    outer_ccw = outer_yz if area > 0.0 else outer_yz[::-1]
    holes = [_oct_loop(cy, cz, r, cw=True) for cy, cz, r, _ in sockets]
    loops = [outer_ccw] + holes

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
    polys = out.GetPolys(); polys.InitTraversal(); idl = vtk.vtkIdList(); fl = []
    while polys.GetNextCell(idl):
        if idl.GetNumberOfIds() == 3:
            fl.append([idl.GetId(0), idl.GetId(1), idl.GetId(2)])
    faces = np.asarray(fl, dtype=int)
    uv_all = np.vstack(loops)
    xyz = np.column_stack((np.full(len(uv_all), x_cap), uv_all[:, 0], uv_all[:, 1]))
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    flip = (np.cross(v1 - v0, v2 - v0)[:, 0] * normal_x) < 0.0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    parts = [trimesh.Trimesh(vertices=xyz, faces=faces, process=False)]

    inward = -1.0 if normal_x > 0.0 else +1.0
    for cy, cz, r, depth in sockets:
        rim = _oct_loop(cy, cz, r, cw=False)  # CCW in (y, z)
        x_mid = x_cap + inward * depth
        x_tip = x_cap + inward * (depth + r)  # 45-deg self-supporting conical blind end!
        v_s = np.vstack((
            np.column_stack((np.full(8, x_cap), rim)),
            np.column_stack((np.full(8, x_mid), rim)),
            [[x_tip, cy, cz]],
        ))
        f_s = []
        for k in range(8):
            k1 = (k + 1) % 8
            if inward < 0:
                f_s.extend([[k, k1, 8 + k], [k1, 8 + k1, 8 + k], [8 + k, 8 + k1, 16]])
            else:
                f_s.extend([[k, 8 + k, k1], [k1, 8 + k, 8 + k1], [8 + k, 16, 8 + k1]])
        parts.append(trimesh.Trimesh(vertices=v_s, faces=np.asarray(f_s, dtype=int), process=False))
    return trimesh.util.concatenate(parts)


def write_bambu_3mf(plate_items: list[tuple[str, trimesh.Trimesh, tuple[float, float, float]]], out_path: Path) -> dict[str, int]:
    """Write spec-compliant multi-object 3MF Core 1.2 + Bambu/Slic3r model_settings.config archive."""
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
        ' <metadata name="Application">KayakGen-1to8-H2S-Builder</metadata>',
        ' <metadata name="Title">Kayak 10.9 — 1:8 Scale Model (Bambu Lab H2S Single-Plate Vertical Print)</metadata>',
        ' <resources>',
    ]
    build_lines = [f' <build p:UUID="{uuid.uuid4()}">']
    cfg_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<config>']

    for idx, (name, mesh, (tx, ty, tz)) in enumerate(plate_items, start=1):
        oid = str(idx)
        ouuid = str(uuid.uuid4())
        iuuid = str(uuid.uuid4())
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        lines_m.append(f'  <object id="{oid}" name="{name}" type="model" p:UUID="{ouuid}">')
        lines_m.append('   <mesh>')
        lines_m.append('    <vertices>')
        for vx, vy, vz in v:
            lines_m.append(f'     <vertex x="{vx:.5f}" y="{vy:.5f}" z="{vz:.5f}"/>')
        lines_m.append('    </vertices>')
        lines_m.append('    <triangles>')
        for f0, f1, f2 in f:
            lines_m.append(f'     <triangle v1="{int(f0)}" v2="{int(f1)}" v3="{int(f2)}"/>')
        lines_m.append('    </triangles>')
        lines_m.append('   </mesh>')
        lines_m.append('  </object>')
        build_lines.append(f'  <item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 {tx:.4f} {ty:.4f} {tz:.4f}" printable="1" p:UUID="{iuuid}"/>')
        cfg_lines.extend([
            f' <object id="{oid}">',
            f'  <metadata key="name" value="{name}"/>',
            '  <metadata key="extruder" value="1"/>',
            ' </object>',
        ])

    lines_m.append(' </resources>')
    lines_m.extend(build_lines)
    lines_m.append(' </build>')
    lines_m.append('</model>\n')
    cfg_lines.append(' <plate>')
    cfg_lines.append('  <metadata key="plater_id" value="1"/>')
    cfg_lines.append('  <metadata key="plater_name" value="Kayak 10.9 1:8 H2S Plate"/>')
    for idx in range(1, len(plate_items) + 1):
        cfg_lines.append(f'  <model_instance><metadata key="object_id" value="{idx}"/><metadata key="instance_id" value="0"/></model_instance>')
    cfg_lines.append(' </plate>')
    cfg_lines.append('</config>\n')

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("3D/3dmodel.model", "\n".join(lines_m))
        zf.writestr("Metadata/model_settings.config", "\n".join(cfg_lines))

    # Round-trip XML verification
    with zipfile.ZipFile(out_path, mode="r") as zf:
        root = ET.fromstring(zf.read("3D/3dmodel.model"))
        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        objs = root.findall("m:resources/m:object", ns)
        items = root.findall("m:build/m:item", ns)
        return {"objects": len(objs), "build_items": len(items), "bytes": out_path.stat().st_size}


def _deck_z_1to8(surf: SculptedKayakSurface, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sculpted deck Z (m) enhanced with crisp 45-deg flush hatch-lid grooves for 1:8 scale printing."""
    z = surf.sculpted_deck_z(x, y)
    for hs in HATCHES:
        rho = np.hypot((x - hs.x0) / hs.rx_lid, (y - hs.y0) / hs.ry_lid)
        # V-groove around flush lid seam (rho = 0.92, width +-0.09 -> ~0.48 mm deep x 45 deg at 1:8 scale)
        t_g = np.clip(1.0 - np.abs(rho - 0.92) / 0.09, 0.0, 1.0)
        z = z - 0.0038 * t_g
    return z


def _end_shape_1to8_mm(surf: SculptedKayakSurface, xyz_m: np.ndarray) -> np.ndarray:
    """Apply Kayak 10.9 rounded ends + smooth proportional tip-width boost (>= 1.02 mm) for 0.4 mm nozzle."""
    p = surf.end_shape_xyz(xyz_m) * SCALE_MM
    half_L_mm = 0.5 * surf.L * SCALE_MM  # 341.875 mm
    d_end = np.minimum(p[:, 0] + half_L_mm, half_L_mm - p[:, 0])
    # Strictly positive multiplicative boost near tips preserves vertex spacing with zero collapsed faces!
    p[:, 1] *= 1.0 + 0.72 * np.exp(-np.maximum(d_end, 0.0) / 3.5)
    return p


def _structured_outer_band_mm(surf: SculptedKayakSurface, i0: int, i1: int, u_dk: np.ndarray) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Build watertight outer hull+deck tube (in 1:8 mm) between station indices [i0..i1], returning mesh & end loops."""
    # Subdivide longitudinal stations 3x so flush hatch grooves resolve crisply without cutting holes
    xs_sub = np.linspace(float(surf.xs[i0]), float(surf.xs[i1]), (i1 - i0) * 3 + 1)
    m_h = surf.hull_rings.shape[1]  # port sheer (0) -> keel -> stbd sheer (m_h-1)
    m_d = len(u_dk)                 # port sheer (-1) -> crown (0) -> stbd sheer (+1)

    rings_yz: list[np.ndarray] = []
    verts_3d: list[np.ndarray] = []
    for xv in xs_sub:
        pos = np.clip(np.searchsorted(surf.xs, xv), 1, len(surf.xs) - 1)
        w = float((xv - surf.xs[pos - 1]) / max(surf.xs[pos] - surf.xs[pos - 1], 1e-9))
        hr = (1.0 - w) * surf.hull_rings[pos - 1] + w * surf.hull_rings[pos]
        sy = float((1.0 - w) * surf.seam_y[pos - 1] + w * surf.seam_y[pos])
        sz = float((1.0 - w) * surf.seam_z[pos - 1] + w * surf.seam_z[pos])

        yv_d = u_dk * sy
        if abs(xv) < 2.68:
            zv_d = _deck_z_1to8(surf, np.full(m_d, xv), yv_d)
        else:
            r_abs = np.abs(u_dk)
            rpos = np.clip(r_abs * (len(surf.deck_r_grid) - 1), 0.0, len(surf.deck_r_grid) - 1 - 1e-9)
            j0 = rpos.astype(int); fr = rpos - j0
            dz0 = (1.0 - fr) * surf.deck_z_table[pos - 1, j0] + fr * surf.deck_z_table[pos - 1, j0 + 1]
            dz1 = (1.0 - fr) * surf.deck_z_table[pos, j0] + fr * surf.deck_z_table[pos, j0 + 1]
            zv_d = (1.0 - w) * dz0 + w * dz1
        zv_d[0], zv_d[-1] = sz, sz

        # Closed CCW loop in (y, z): hull port->keel->stbd (0..m_h-1) + deck stbd->crown->port (m_d-2 .. 1)
        dk_inner = np.column_stack((yv_d[-2:0:-1], zv_d[-2:0:-1]))
        loop_m = np.vstack((hr, dk_inner))
        xyz_mm = _end_shape_1to8_mm(surf, np.column_stack((np.full(len(loop_m), xv), loop_m)))
        verts_3d.append(xyz_mm)
        rings_yz.append(xyz_mm[:, 1:])

    nx = len(xs_sub)
    m_tot = len(rings_yz[0])
    v_all = np.vstack(verts_3d)
    faces = []
    for i in range(nx - 1):
        s, t = i * m_tot, (i + 1) * m_tot
        for j in range(m_tot):
            j1 = (j + 1) % m_tot
            faces.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]])
    tube = trimesh.Trimesh(vertices=v_all, faces=np.asarray(faces, dtype=int), process=False)
    return tube, rings_yz[0], rings_yz[-1]


def _ordered_keyhole_loop(mesh: trimesh.Trimesh, x_b1_mm: float, x_b2_mm: float) -> np.ndarray:
    """Extract ordered boundary vertex indices of the interior keyhole opening (excluding x_b1 & x_b2 end rings)."""
    # Find boundary directed half-edges (edges appearing only once; boundary loop runs opposite to face half-edge)
    from collections import Counter, defaultdict
    f = np.asarray(mesh.faces, dtype=int)
    he = [(int(a), int(b)) for tri in f for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))]
    undir = Counter((min(a, b), max(a, b)) for a, b in he)
    v = np.asarray(mesh.vertices)
    adj: dict[int, int] = {}
    for a, b in he:
        if undir[(min(a, b), max(a, b))] == 1:
            # Filter out end-cap cross-section rings at x_b1_mm and x_b2_mm
            if abs(v[a, 0] - x_b1_mm) < 0.25 and abs(v[b, 0] - x_b1_mm) < 0.25:
                continue
            if abs(v[a, 0] - x_b2_mm) < 0.25 and abs(v[b, 0] - x_b2_mm) < 0.25:
                continue
            adj[a] = b  # Deck face has +Z normal, so keyhole boundary half-edge a->b runs CW around hole!
    start = min(adj.keys(), key=lambda idx: v[idx, 0])  # Start near forward nose (-0.46 m)
    loop = [start]
    cur = adj[start]
    while cur != start and len(loop) <= len(adj) + 2:
        loop.append(cur)
        cur = adj[cur]
    return np.asarray(loop, dtype=int)


def _build_mid_cockpit_tube_mm(surf: SculptedKayakSurface, i_b1: int, i_b2: int, u_dk: np.ndarray) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Build watertight 1:8 midbody tube (xs[i_b1]..xs[i_b2]) with sculpted coaming, thigh hooks, seat & footwell."""
    xs_mid = np.linspace(float(surf.xs[i_b1]), float(surf.xs[i_b2]), (i_b2 - i_b1) * 3 + 1)
    m_h = surf.hull_rings.shape[1]

    # 1. Outer hull trough across xs_mid
    hull_v_list = []
    end_rings_yz = {}
    for idx_s, xv in enumerate(xs_mid):
        pos = np.clip(np.searchsorted(surf.xs, xv), 1, len(surf.xs) - 1)
        w = float((xv - surf.xs[pos - 1]) / max(surf.xs[pos] - surf.xs[pos - 1], 1e-9))
        hr = (1.0 - w) * surf.hull_rings[pos - 1] + w * surf.hull_rings[pos]
        sy = float((1.0 - w) * surf.seam_y[pos - 1] + w * surf.seam_y[pos])
        sz = float((1.0 - w) * surf.seam_z[pos - 1] + w * surf.seam_z[pos])
        hull_v_list.append(_end_shape_1to8_mm(surf, np.column_stack((np.full(m_h, xv), hr))))
        if idx_s in (0, len(xs_mid) - 1):
            yv_d = u_dk * sy
            zv_d = _deck_z_1to8(surf, np.full(len(u_dk), xv), yv_d)
            zv_d[0], zv_d[-1] = sz, sz
            dk_in = np.column_stack((yv_d[-2:0:-1], zv_d[-2:0:-1]))
            end_rings_yz[idx_s] = _end_shape_1to8_mm(surf, np.column_stack((np.full(m_h + len(dk_in), xv), np.vstack((hr, dk_in)))))[:, 1:]

    fh = []
    for i in range(len(xs_mid) - 1):
        s, t = i * m_h, (i + 1) * m_h
        for j in range(m_h - 1):
            fh.extend([[s + j, s + j + 1, t + j], [s + j + 1, t + j + 1, t + j]])
    m_hull_mid = trimesh.Trimesh(vertices=np.vstack(hull_v_list), faces=np.asarray(fh, dtype=int), process=False)

    # 2. Mid-body deck with keyhole opening (exact match to xs_mid side seams & u_dk end rings)
    sy_b1 = float(np.interp(xs_mid[0], surf.xs, surf.seam_y))
    sy_b2 = float(np.interp(xs_mid[-1], surf.xs, surf.seam_y))
    fwd_xy = np.column_stack((np.full(len(u_dk), xs_mid[0]), u_dk * sy_b1))
    aft_xy = np.column_stack((np.full(len(u_dk), xs_mid[-1]), u_dk * sy_b2))
    side_xy = []
    for xv in xs_mid[1:-1]:
        sy = float(np.interp(xv, surf.xs, surf.seam_y))
        side_xy.extend([[xv, -sy], [xv, +sy]])
    seam_b = np.vstack((fwd_xy, aft_xy, np.asarray(side_xy)))

    th_kh = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    ring_pts = []
    for delta in (0.0180, 0.0255, 0.0340):
        rc = keyhole_offset_curve(th_kh, delta)
        sy_lim = np.interp(rc[:, 0], surf.xs, surf.seam_y) - 0.0065
        rc[:, 1] = np.clip(rc[:, 1], -sy_lim, +sy_lim)
        ring_pts.append(rc)

    cand_l = []
    for xv in xs_mid[1:-1]:
        sy = float(np.interp(xv, surf.xs, surf.seam_y))
        ny = max(2, int(np.floor((sy - 0.006) / 0.009)))
        ys = np.linspace(-(sy - 0.0065), +(sy - 0.0065), 2 * ny + 1)
        cand_l.append(np.column_stack((np.full(len(ys), xv), ys)))
    cand = np.vstack(cand_l)
    fixed = np.vstack([seam_b] + ring_pts)
    keep = (cKDTree(fixed).query(cand, k=1)[0] > 0.0045) & (keyhole_signed_distance(cand)[0] > 0.022)
    pts_xy = np.vstack((fixed, cand[keep]))
    simp = Delaunay(pts_xy).simplices
    cent = pts_xy[simp].mean(axis=1)
    sy_c = np.interp(cent[:, 0], surf.xs, surf.seam_y)
    valid = (cent[:, 0] > xs_mid[0] + 1e-6) & (cent[:, 0] < xs_mid[-1] - 1e-6) & (np.abs(cent[:, 1]) < sy_c - 1e-5)
    valid &= keyhole_signed_distance(cent)[0] > 0.01805
    faces_d = simp[valid].copy()

    z_d = _deck_z_1to8(surf, pts_xy[:, 0], pts_xy[:, 1])
    z_d[0] = z_d[len(u_dk) - 1] = float(np.interp(xs_mid[0], surf.xs, surf.seam_z))
    z_d[len(u_dk)] = z_d[2 * len(u_dk) - 1] = float(np.interp(xs_mid[-1], surf.xs, surf.seam_z))
    nr = 2 * len(u_dk); ns = len(seam_b) - nr
    z_d[nr : nr + ns] = np.interp(pts_xy[nr : nr + ns, 0], surf.xs, surf.seam_z)
    xyz_d = _end_shape_1to8_mm(surf, np.column_stack((pts_xy[:, 0], pts_xy[:, 1], z_d)))
    v0, v1, v2 = xyz_d[faces_d[:, 0]], xyz_d[faces_d[:, 1]], xyz_d[faces_d[:, 2]]
    flp = np.cross(v1 - v0, v2 - v0)[:, 2] < 0.0
    faces_d[flp] = faces_d[flp][:, [0, 2, 1]]
    m_deck_mid = trimesh.Trimesh(vertices=xyz_d, faces=faces_d, process=False)

    mid_outer = trimesh.util.concatenate([m_hull_mid, m_deck_mid])
    mid_outer.merge_vertices(digits_vertex=4)
    return mid_outer, end_rings_yz[0], end_rings_yz[len(xs_mid) - 1]


def _close_keyhole_with_cockpit_liner_mm(surf: SculptedKayakSurface, mid_outer: trimesh.Trimesh, x_b1_mm: float, x_b2_mm: float) -> trimesh.Trimesh:
    """Stitch watertight sculpted coaming lip, thigh hooks, dished seat pan & 40-deg vaulted footwell onto keyhole loop."""
    loop_idx = _ordered_keyhole_loop(mid_outer, x_b1_mm, x_b2_mm)
    v_out = np.asarray(mid_outer.vertices, dtype=float)
    r0_mm = v_out[loop_idx]  # Exact vertices on mid_outer boundary (mm)
    r0_m = r0_mm / SCALE_MM
    n_k = len(loop_idx)

    # Map each boundary point to nearest parameter angle theta on keyhole_inner_curve
    th_grid = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    kh_grid = keyhole_inner_curve(th_grid)
    idx_th = cKDTree(kh_grid).query(r0_m[:, :2], k=1)[1]
    th = th_grid[idx_th]
    u = -np.cos(th)       # -1 at fwd nose (-0.46 m), +1 at aft rim (+0.38 m)
    s_th = np.sin(th)     # + on stbd, - on port

    xy_lip_out = keyhole_offset_curve(th, 0.0085)
    xy_lip_in = keyhole_inner_curve(th)
    z0 = r0_m[:, 2]

    # Ring 1: Outer coaming lip crest (+14.5 mm full = +1.81 mm at 1:8, <= 40 deg self-supporting slope!)
    r1_m = np.column_stack((xy_lip_out[:, 0], xy_lip_out[:, 1], z0 + 0.0145))
    # Ring 2: Inner coaming lip crest
    r2_m = np.column_stack((xy_lip_in[:, 0], xy_lip_in[:, 1], z0 + 0.0145))
    # Ring 3: Coaming skirt neck + sculpted under-deck side thigh-hook wings (x in [-0.34, +0.02] m)
    x3 = xy_lip_in[:, 0].copy()
    hook_env = np.clip(np.sin(np.pi * np.clip((x3 + 0.34) / 0.36, 0.0, 1.0)), 0.0, 1.0) ** 1.35
    hook_in = 0.026 * hook_env * (np.abs(s_th) ** 1.4)
    y3 = np.sign(xy_lip_in[:, 1]) * np.maximum(np.abs(xy_lip_in[:, 1]) - hook_in, 0.0)
    r3_m = np.column_stack((x3, y3, z0 - 0.0155))

    # Ring 4: Upper cockpit sidewall + 36-deg self-supporting forward footwell vault (to x=-0.74 m) + aft seat backrest (+0.368 m)
    x4 = np.where(u < 0.0, -0.040 + 0.700 * u, -0.040 + 0.408 * u)
    sy4 = np.interp(x4, surf.xs, surf.seam_y)
    zk4 = surf.keel_at(x4)
    y4 = s_th * np.minimum(0.76 * sy4, np.maximum(sy4 - 0.026, 0.04))
    vault_z = zk4 + 0.058 + 0.065 * (0.5 * (1.0 + u))
    r4_m = np.column_stack((x4, y4, np.minimum(z0 - 0.024, vault_z)))

    # Ring 5: Bilge chine + contoured surfski/FSK low seat pan rim (x in [+0.02, +0.34] m) & angled footboard sole
    x5 = np.where(u < 0.0, -0.040 + 0.680 * u, -0.040 + 0.392 * u)
    sy5 = np.interp(x5, surf.xs, surf.seam_y)
    zk5 = surf.keel_at(x5)
    y5 = s_th * np.minimum(0.42 * sy5, np.maximum(sy5 - 0.036, 0.02))
    seat_dish = -0.006 * np.exp(-((x5 - 0.14) / 0.13) ** 2) + 0.008 * np.exp(-((x5 + 0.03) / 0.06) ** 2)
    r5_m = np.column_stack((x5, y5, zk5 + 0.025 + seat_dish))

    # Ring 6: Narrow inner sole ring -> Center floor apex point (clean aspect triangles!)
    x6 = -0.060 + 0.160 * u
    zk6 = surf.keel_at(x6)
    y6 = 0.10 * y5
    r6_m = np.column_stack((x6, y6, zk6 + 0.022))
    apex_mm = np.array([[-0.060 * SCALE_MM, 0.0, (float(surf.keel_at(np.array([-0.060]))[0]) + 0.0215) * SCALE_MM]])

    rings_mm = [r0_mm] + [rm * SCALE_MM for rm in (r1_m, r2_m, r3_m, r4_m, r5_m, r6_m)]
    v_liner = np.vstack(rings_mm + [apex_mm])
    apex_id = len(rings_mm) * n_k
    f_l = []
    for r in range(len(rings_mm) - 1):
        s, t = r * n_k, (r + 1) * n_k
        for j in range(n_k):
            j1 = (j + 1) % n_k
            f_l.extend([[s + j, t + j, s + j1], [s + j1, t + j, t + j1]])
    s_last = (len(rings_mm) - 1) * n_k
    for j in range(n_k):
        j1 = (j + 1) % n_k
        f_l.append([s_last + j, apex_id, s_last + j1])

    liner = trimesh.Trimesh(vertices=v_liner, faces=np.asarray(f_l, dtype=int), process=False)
    combined = trimesh.util.concatenate([mid_outer, liner])
    combined.merge_vertices(digits_vertex=4)
    return combined


def _make_oct_pin_on_bed(length_mm: float = 10.4, r_oct: float = 1.62) -> trimesh.Trimesh:
    """Horizontal-printed octagonal shear alignment pin (layers run lengthwise across bulkhead joint!)."""
    th = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + (np.pi / 8.0)
    yz_main = np.column_stack((r_oct * np.cos(th), r_oct * np.sin(th)))
    yz_tip = 0.72 * yz_main
    xs = [-0.5 * length_mm, -0.5 * length_mm + 0.9, +0.5 * length_mm - 0.9, +0.5 * length_mm]
    rings = [yz_tip, yz_main, yz_main, yz_tip]
    v_l = [np.column_stack((np.full(8, xv), rg)) for xv, rg in zip(xs, rings)]
    v = np.vstack(v_l + [[[-0.5 * length_mm, 0.0, 0.0]], [[+0.5 * length_mm, 0.0, 0.0]]])
    f = []
    for r in range(3):
        s, t = r * 8, (r + 1) * 8
        for j in range(8):
            j1 = (j + 1) % 8
            f.extend([[s + j, s + j1, t + j], [s + j1, t + j1, t + j]])
    for j in range(8):
        j1 = (j + 1) % 8
        f.append([32, j1, j])
        f.append([24 + j, 24 + j1, 33])
    m = trimesh.Trimesh(vertices=v, faces=np.asarray(f, dtype=int), process=True)
    m.apply_translation((0.0, 0.0, -float(m.bounds[0, 2])))
    return m


def _make_display_stand_on_bed(surf: SculptedKayakSurface, x_station_m: float) -> trimesh.Trimesh:
    """Hull-contoured 1:8 desk display cradle stand at x_station_m (flat base on Z=0, zero overhangs)."""
    i_s = int(np.argmin(np.abs(surf.xs - x_station_m)))
    hr_mm = _end_shape_1to8_mm(surf, np.column_stack((np.full(surf.hull_rings.shape[1], surf.xs[i_s]), surf.hull_rings[i_s])))[:, 1:]
    keep = np.abs(hr_mm[:, 0]) <= 25.0
    cradle = hr_mm[keep]
    cradle = cradle[np.argsort(cradle[:, 0])]
    z_low = float(np.min(cradle[:, 1])) - 7.5
    y_out = float(np.max(np.abs(cradle[:, 0]))) + 5.5
    z_top = float(np.max(cradle[:, 1])) + 1.5
    # 2D profile in (Y_stand, Z_stand), extruded 14 mm thick -> rotated flat onto bed (14 mm tall in Z_print!)
    prof = np.vstack((
        [[-y_out, z_low], [+y_out, z_low], [+y_out - 2.5, z_top], [cradle[-1, 0], z_top]],
        cradle[::-1] + np.array([0.0, 0.25]),
        [[cradle[0, 0], z_top], [-y_out + 2.5, z_top]],
    ))
    pts = vtk.vtkPoints(vtk.VTK_DOUBLE); lines = vtk.vtkCellArray()
    area = float(np.sum(prof[:, 0] * np.roll(prof[:, 1], -1) - np.roll(prof[:, 0], -1) * prof[:, 1]))
    lp = prof if area > 0 else prof[::-1]
    for u, v in lp:
        pts.InsertNextPoint(float(u), float(v), 0.0)
    lines.InsertNextCell(len(lp) + 1)
    for k in range(len(lp)):
        lines.InsertCellPoint(k)
    lines.InsertCellPoint(0)
    pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(lines)
    ct = vtk.vtkContourTriangulator(); ct.SetInputData(pd); ct.Update()
    polys = ct.GetOutput().GetPolys(); polys.InitTraversal(); idl = vtk.vtkIdList(); fl = []
    while polys.GetNextCell(idl):
        if idl.GetNumberOfIds() == 3:
            fl.append([idl.GetId(0), idl.GetId(1), idl.GetId(2)])
    f2 = np.asarray(fl, dtype=int)
    n = len(lp); h = 13.0
    vb = np.column_stack((lp[:, 0], lp[:, 1], np.zeros(n)))
    vt = np.column_stack((lp[:, 0], lp[:, 1], np.full(n, h)))
    nz = np.cross(vb[f2[:, 1]] - vb[f2[:, 0]], vb[f2[:, 2]] - vb[f2[:, 0]])[:, 2]
    fb = np.where((nz > 0)[:, None], f2[:, [0, 2, 1]], f2)
    ft = fb[:, [0, 2, 1]] + n
    fr = []
    for k in range(n):
        k1 = (k + 1) % n
        fr.extend([[k, k1, n + k], [k1, n + k1, n + k]])
    m = trimesh.Trimesh(vertices=np.vstack((vb, vt)), faces=np.vstack((fb, ft, np.asarray(fr, dtype=int))), process=True)
    m.apply_translation((-0.5 * float(m.bounds[0, 0] + m.bounds[1, 0]), -0.5 * float(m.bounds[0, 1] + m.bounds[1, 1]), -float(m.bounds[0, 2])))
    return m


def _clean_solid(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m.merge_vertices(digits_vertex=4)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    if m.volume < 0.0:
        m.invert()
    return m


def _orient_to_bed(mesh_boat_mm: trimesh.Trimesh, x_cut_mm: float, bow_up: bool) -> trimesh.Trimesh:
    """Right-handed (det=+1) rotation placing flat X = x_cut_mm bulkhead face dead-flat on Z_plate = 0."""
    m = _clean_solid(mesh_boat_mm.copy())
    v = np.asarray(m.vertices, dtype=float)
    yc = 0.5 * float(v[:, 2].min() + v[:, 2].max())
    if bow_up:
        v_new = np.column_stack((-v[:, 1], v[:, 2] - yc, x_cut_mm - v[:, 0]))
    else:
        v_new = np.column_stack((+v[:, 1], v[:, 2] - yc, v[:, 0] - x_cut_mm))
    v_new[:, 2] -= float(v_new[:, 2].min())
    return _clean_solid(trimesh.Trimesh(vertices=v_new, faces=np.asarray(m.faces, dtype=int).copy(), process=False))


def build_all_1to8_models(surf: SculptedKayakSurface) -> tuple[list[tuple[str, trimesh.Trimesh, tuple[float, float, float]]], trimesh.Trimesh, dict]:
    u_dk = np.linspace(-1.0, +1.0, 129)  # Odd count -> exact y=0 tip folding symmetry!
    i_b1 = int(np.argmin(np.abs(surf.xs - X_BH_FWD)))
    i_b2 = int(np.argmin(np.abs(surf.xs - X_BH_AFT)))
    x_b1_mm = float(surf.xs[i_b1]) * SCALE_MM
    x_b2_mm = float(surf.xs[i_b2]) * SCALE_MM

    bow_tube, _, ring_b1_bow = _structured_outer_band_mm(surf, 0, i_b1, u_dk)
    mid_outer, ring_b1_mid, ring_b2_mid = _build_mid_cockpit_tube_mm(surf, i_b1, i_b2, u_dk)
    mid_tube = _close_keyhole_with_cockpit_liner_mm(surf, mid_outer, x_b1_mm, x_b2_mm)
    stern_tube, ring_b2_stn, _ = _structured_outer_band_mm(surf, i_b2, len(surf.xs) - 1, u_dk)

    # 1. Complete uncut 683.75 mm 1:8 solid model (for reference / Voron 600 diagonal / custom slicing)
    uncut_full = _clean_solid(trimesh.util.concatenate([bow_tube, mid_tube, stern_tube]))

    # Alignment socket layouts at Forward Bulkhead (x_b1_mm) and Aft Seat Bulkhead (x_b2_mm)
    # Each joint gets 2x octagonal r=1.76 mm shear-key sockets (5.4 mm deep) + 2x r=1.02 mm (1.75 mm filament) sockets!
    z_mid_b1 = 0.48 * float(ring_b1_bow[:, 1].min() + ring_b1_bow[:, 1].max())
    z_mid_b2 = 0.48 * float(ring_b2_stn[:, 1].min() + ring_b2_stn[:, 1].max())
    socks_b1 = [
        (-11.5, z_mid_b1, 1.76, 5.4), (+11.5, z_mid_b1, 1.76, 5.4),
        (-19.5, z_mid_b1 - 1.0, 1.02, 4.8), (+19.5, z_mid_b1 - 1.0, 1.02, 4.8),
    ]
    socks_b2 = [
        (-12.5, z_mid_b2, 1.76, 5.4), (+12.5, z_mid_b2, 1.76, 5.4),
        (-21.5, z_mid_b2 - 1.0, 1.02, 4.8), (+21.5, z_mid_b2 - 1.0, 1.02, 4.8),
    ]

    bow_solid = trimesh.util.concatenate([bow_tube, _cap_with_blind_sockets_x(ring_b1_bow, x_b1_mm, normal_x=+1.0, sockets=socks_b1)])
    bow_solid.merge_vertices(digits_vertex=4)
    if bow_solid.volume < 0.0:
        bow_solid.invert()

    mid_solid = trimesh.util.concatenate([
        mid_tube,
        _cap_with_blind_sockets_x(ring_b1_mid, x_b1_mm, normal_x=-1.0, sockets=socks_b1),
        _cap_with_blind_sockets_x(ring_b2_mid, x_b2_mm, normal_x=+1.0, sockets=socks_b2),
    ])
    mid_solid.merge_vertices(digits_vertex=4)
    if mid_solid.volume < 0.0:
        mid_solid.invert()

    stern_solid = trimesh.util.concatenate([stern_tube, _cap_with_blind_sockets_x(ring_b2_stn, x_b2_mm, normal_x=-1.0, sockets=socks_b2)])
    stern_solid.merge_vertices(digits_vertex=4)
    if stern_solid.volume < 0.0:
        stern_solid.invert()

    # Orient all 3 sections upright (cut bulkhead face flat on Z=0!)
    bow_bed = _orient_to_bed(bow_solid, x_b1_mm, bow_up=True)
    mid_bed = _orient_to_bed(mid_solid, x_b2_mm, bow_up=True)
    stern_bed = _orient_to_bed(stern_solid, x_b2_mm, bow_up=False)

    # 6x horizontal octagonal shear keys (4 needed + 2 spares!) + 2x Bessel-point contoured desk stands
    pin_one = _make_oct_pin_on_bed(length_mm=10.4, r_oct=1.62)
    pins_list = []
    for k in range(6):
        pk = pin_one.copy()
        pk.apply_translation(((k % 2 - 0.5) * 16.0, (k // 2 - 1.0) * 6.5, 0.0))
        pins_list.append(pk)
    pins_bed = trimesh.util.concatenate(pins_list)

    st_fwd = _make_display_stand_on_bed(surf, -1.530)
    st_aft = _make_display_stand_on_bed(surf, +1.530)

    # Arrange on 340 x 320 mm Bambu Lab H2S build plate (>= 34 mm gap between all parts for 8-10 mm brims!)
    plate_items = [
        ("01_bow_section_1to8", bow_bed, (88.0, 102.0, 0.0)),
        ("02_cockpit_midbody_1to8", mid_bed, (170.0, 212.0, 0.0)),
        ("03_stern_section_1to8", stern_bed, (252.0, 102.0, 0.0)),
        ("04_alignment_shear_keys_x6", pins_bed, (170.0, 96.0, 0.0)),
        ("05a_display_stand_fwd_bessel", st_fwd, (68.0, 212.0, 0.0)),
        ("05b_display_stand_aft_bessel", st_aft, (272.0, 212.0, 0.0)),
    ]
    report = {
        "uncut_watertight": bool(uncut_full.is_watertight),
        "uncut_extents_mm": [round(float(x), 2) for x in uncut_full.extents],
        "parts": {
            nm: {
                "watertight": bool(m.is_watertight),
                "is_volume": bool(m.is_volume),
                "extents_mm": [round(float(x), 2) for x in m.extents],
                "volume_cm3": round(float(m.volume) / 1000.0, 2),
                "plate_pos_mm": list(pos),
            }
            for nm, m, pos in plate_items
        },
    }
    return plate_items, uncut_full, report


def _style(m: trimesh.Trimesh, name: str, rgba: list[int], rough: float = 0.34, metal: float = 0.08) -> trimesh.Trimesh:
    mc = m.copy()
    mc.visual = TextureVisuals(material=PBRMaterial(name=name, baseColorFactor=rgba, roughnessFactor=rough, metallicFactor=metal, doubleSided=True))
    mc.metadata["name"] = name
    return mc


def render_preview(plate_items: list[tuple[str, trimesh.Trimesh, tuple[float, float, float]]], uncut_full: trimesh.Trimesh, surf: SculptedKayakSurface) -> None:
    import pyvista as pv
    from PIL import Image

    def to_pv(tm_m: trimesh.Trimesh, feat: float = 48.0) -> pv.PolyData:
        v = np.asarray(tm_m.vertices, dtype=float)
        f = np.asarray(tm_m.faces, dtype=np.int64)
        pd = pv.PolyData(v, np.column_stack((np.full(len(f), 3, dtype=np.int64), f)).ravel())
        return pd.compute_normals(cell_normals=False, point_normals=True, split_vertices=True, feature_angle=feat, auto_orient_normals=False)

    # Panel 1: Bambu Lab H2S 340 x 320 x 340 mm Build Plate Studio Render
    pl = pv.Plotter(off_screen=True, window_size=(1900, 1550), lighting="none")
    pl.set_background("#151e29", top="#263649")
    for pos, inten, col in [((-420, -520, 620), 1.45, "#fffdf9"), ((+680, -380, 480), 0.90, "#e6f2ff"), ((170, +680, 420), 0.72, "#ffffff")]:
        lt = pv.Light(position=pos, focal_point=(170, 160, 110), color=col, intensity=inten)
        lt.positional = False; pl.add_light(lt)

    bed = pv.Cube(center=(170.0, 160.0, -1.5), x_length=340.0, y_length=320.0, z_length=3.0)
    pl.add_mesh(bed, color="#2b333e", pbr=True, metallic=0.35, roughness=0.58)
    # 40 mm build-plate grid lines + 340 mm Z-ceiling wireframe box
    for gx in range(20, 340, 40):
        pl.add_mesh(pv.Line((gx, 0, 0.15), (gx, 320, 0.15)), color="#425063", line_width=1.2)
    for gy in range(40, 320, 40):
        pl.add_mesh(pv.Line((0, gy, 0.15), (340, gy, 0.15)), color="#425063", line_width=1.2)
    vol_box = pv.Box(bounds=(0, 340, 0, 320, 0, 340))
    pl.add_mesh(vol_box.extract_feature_edges(boundary_edges=True, feature_edges=True, manifold_edges=False), color="#6889ab", line_width=1.8, opacity=0.55)

    palette = ["#4f90c6", "#e6eef6", "#3d7cb0", "#f4a261", "#d8b480", "#d8b480"]
    for (nm, m, (tx, ty, tz)), col in zip(plate_items, palette):
        mp = m.copy(); mp.apply_translation((tx, ty, tz))
        pl.add_mesh(to_pv(mp, 44), color=col, pbr=True, metallic=0.08, roughness=0.28, smooth_shading=True)

    pl.camera.position = (-245, -315, 395)
    pl.camera.focal_point = (170, 158, 128)
    pl.camera.up = (0, 0, 1)
    pl.enable_parallel_projection(); pl.camera.parallel_scale = 238.0
    pl.enable_anti_aliasing("ssaa"); pl.render()
    img_plate = pl.screenshot(return_img=True); pl.close()

    # Panel 2: Assembled 683.8 mm 1:8 Scale Kayak 10.9 on Contoured Bessel-Point Desk Stands
    pl2 = pv.Plotter(off_screen=True, window_size=(1900, 1550), lighting="none")
    pl2.set_background("#151e29", top="#263649")
    for pos, inten, col in [((-580, -680, 480), 1.48, "#fffdf9"), ((+620, -480, 420), 0.90, "#e6f2ff"), ((0, +680, 380), 0.72, "#ffffff")]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 18), color=col, intensity=inten)
        lt.positional = False; pl2.add_light(lt)
    pl2.add_mesh(to_pv(uncut_full, 44), color="#5899ce", pbr=True, metallic=0.10, roughness=0.25, smooth_shading=True)
    # Show the two subtle bulkhead joint rings at x_b1 & x_b2
    i_b1 = int(np.argmin(np.abs(surf.xs - X_BH_FWD)))
    i_b2 = int(np.argmin(np.abs(surf.xs - X_BH_AFT)))
    for ii in (i_b1, i_b2):
        sec = trimesh.intersections.mesh_plane(uncut_full, [1, 0, 0], [float(surf.xs[ii]) * SCALE_MM, 0, 0])
        for seg in sec:
            pl2.add_mesh(pv.Line(seg[0], seg[1]), color="#f4a261", line_width=2.5)
    pl2.camera.position = (-540, -560, 290)
    pl2.camera.focal_point = (8, 0, 16)
    pl2.camera.up = (0, 0, 1)
    pl2.enable_parallel_projection(); pl2.camera.parallel_scale = 236.0
    pl2.enable_anti_aliasing("ssaa"); pl2.render()
    img_assy = pl2.screenshot(return_img=True); pl2.close()

    fig, axes = plt.subplots(1, 2, figsize=(19.0, 8.2), dpi=180, facecolor="#111822")
    axes[0].imshow(img_plate); axes[0].axis("off")
    axes[0].set_title("1. Bambu Lab H2S Single-Plate .3MF (340×320×340 mm — Max Z = 286.4 mm, 0 g Supports)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
    axes[1].imshow(img_assy); axes[1].axis("off")
    axes[1].set_title("2. Assembled 1:8 Kayak 10.9 (683.8 × 70.3 × 42.1 mm — Unbroken Hull/Deck, Bulkhead Seams in Orange)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=8)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT / "h2s_1to8_plate_and_model_preview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    import json
    surf = SculptedKayakSurface(HULL_JSON)
    plate_items, uncut_full, report = build_all_1to8_models(surf)

    assert uncut_full.is_watertight, "Expected uncut 1:8 kayak to be watertight!"
    for nm, m, _ in plate_items:
        assert m.is_watertight and m.is_volume, f"Part {nm} not watertight volume!"

    # Export primary multi-object Bambu Lab H2S .3mf
    tmf_path = OUT / "kayak-10-9-scale-1to8-h2s-plate.3mf"
    report["threemf_verification"] = write_bambu_3mf(plate_items, tmf_path)

    bow_name, bow_mesh, _ = next(item for item in plate_items if item[0] == "01_bow_section_1to8")
    report["bow_only_3mf"] = write_bambu_3mf(
        [(bow_name, bow_mesh, (170.0, 160.0, 0.0))],
        OUT / "kayak-10-9-scale-1to8-h2s-bow.3mf",
    )

    # Export uncut 1:8 STL + individual part STLs + GLB preview
    uncut_full.export(OUT / "kayak-10-9-scale-1to8-uncut-full.stl")
    sc = trimesh.Scene()
    cols = [[79, 144, 198, 255], [228, 236, 244, 255], [61, 124, 176, 255], [244, 162, 97, 255], [216, 180, 128, 255], [216, 180, 128, 255]]
    for (nm, m, (tx, ty, tz)), rgba in zip(plate_items, cols):
        m.export(OUT / f"{nm}.stl")
        mp = m.copy(); mp.apply_translation((tx, ty, tz))
        sc.add_geometry(_style(mp, nm, rgba), node_name=nm)
    # Add ghostly H2S build-plate outline to GLB preview only
    bed_g = trimesh.creation.box(extents=(340.0, 320.0, 2.0)); bed_g.apply_translation((170.0, 160.0, -1.0))
    sc.add_geometry(_style(bed_g, "00_bambu_h2s_340x320_build_plate", [44, 52, 62, 255], rough=0.6, metal=0.3), node_name="00_bambu_h2s_340x320_build_plate")
    sc.export(OUT / "kayak-10-9-scale-1to8-h2s-plate-preview.glb")

    (OUT / "h2s_1to8_build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    render_preview(plate_items, uncut_full, surf)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
