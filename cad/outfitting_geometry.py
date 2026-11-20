"""Stage 10 composite outfitting: keyhole coaming + thigh hooks, hatches, bulkheads, seat, footboard, rudder."""

from __future__ import annotations

import numpy as np
import trimesh
from deck_geometry import HATCHES, SculptedKayakSurface, keyhole_inner_curve, keyhole_offset_curve, smoothstep


def _grid_mesh(grid: np.ndarray, wrap_u: bool = False, wrap_v: bool = False, outward_ref: np.ndarray | None = None) -> trimesh.Trimesh:
    """Build a quad-triangulated mesh from (Nu, Nv, 3) grid; optionally orient normals vs outward_ref."""
    nu, nv, _ = grid.shape
    verts = grid.reshape(-1, 3)
    mu = nu if wrap_u else nu - 1
    mv = nv if wrap_v else nv - 1
    faces: list[list[int]] = []
    for i in range(mu):
        i1 = (i + 1) % nu
        for j in range(mv):
            j1 = (j + 1) % nv
            c1, c2 = i * nv + j, i * nv + j1
            n1, n2 = i1 * nv + j, i1 * nv + j1
            faces.append([c1, c2, n1])
            faces.append([c2, n2, n1])
    f_arr = np.asarray(faces, dtype=int)
    if outward_ref is not None:
        v0, v1, v2 = verts[f_arr[:, 0]], verts[f_arr[:, 1]], verts[f_arr[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fc = (v0 + v1 + v2) / 3.0
        ref = outward_ref.reshape(-1, 3)
        if ref.shape[0] == verts.shape[0]:
            rc = (ref[f_arr[:, 0]] + ref[f_arr[:, 1]] + ref[f_arr[:, 2]]) / 3.0
        else:
            rc = ref[0]
        if float(np.sum(fn * (fc - rc))) < 0.0:
            f_arr = f_arr[:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=verts, faces=f_arr, process=True)


def _cap_ring(ring_xyz: np.ndarray, normal_dir: np.ndarray) -> trimesh.Trimesh:
    """Fan-cap a convex/star-shaped 3D ring around its centroid."""
    c = ring_xyz.mean(axis=0, keepdims=True)
    verts = np.vstack((c, ring_xyz))
    n = len(ring_xyz)
    faces = np.asarray([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)], dtype=int)
    fn = np.cross(verts[faces[:, 1]] - verts[0], verts[faces[:, 2]] - verts[0]).sum(axis=0)
    if float(np.dot(fn, normal_dir)) < 0.0:
        faces = faces[:, [0, 2, 1]]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def build_coaming_and_thigh_hooks(surf: SculptedKayakSurface) -> trimesh.Trimesh:
    """84x42 cm keyhole composite coaming rim + sculpted under-deck side thigh-hook wings."""
    n_th = 220
    th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    neck_xy = keyhole_offset_curve(th, 0.0040)
    z_base = surf.sculpted_deck_z(neck_xy[:, 0], neck_xy[:, 1])

    # Cross-section profile (delta_m, dz_m relative to deck neck base)
    prof = np.array([
        [0.0000, -0.0110],
        [0.0000, +0.0155],
        [0.0030, +0.0185],
        [0.0130, +0.0190],
        [0.0225, +0.0176],
        [0.0240, +0.0154],
        [0.0220, +0.0134],
        [0.0048, +0.0134],
        [0.0042, -0.0040],
        [0.0020, -0.0115],
    ])
    grid = np.zeros((n_th, len(prof), 3))
    ref = np.zeros_like(grid)
    c_xy = keyhole_offset_curve(th, 0.0060)
    c_z = z_base + 0.0080
    for j, (dlt, dz) in enumerate(prof):
        xy = keyhole_offset_curve(th, float(dlt))
        grid[:, j, 0] = xy[:, 0]
        grid[:, j, 1] = xy[:, 1]
        grid[:, j, 2] = z_base + float(dz)
        ref[:, j, 0] = c_xy[:, 0]
        ref[:, j, 1] = c_xy[:, 1]
        ref[:, j, 2] = c_z
    rim_mesh = _grid_mesh(grid, wrap_u=True, wrap_v=True, outward_ref=ref)
    parts = [rim_mesh]

    # Hybrid under-deck side thigh-hook wings (open 31 cm center channel for wing knees; solid hooks when splayed)
    th_stbd = np.linspace(0.20 * np.pi, 0.56 * np.pi, 40)
    kh_stbd = keyhole_inner_curve(th_stbd)
    order = np.argsort(kh_stbd[:, 0])
    xs_ref, ys_ref = kh_stbd[order, 0], kh_stbd[order, 1]

    xs_hook = np.linspace(-0.34, +0.02, 34)
    u_env = np.sin(np.pi * np.clip((xs_hook - (-0.34)) / 0.36, 0.0, 1.0)) ** 1.35
    s_loop = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False)

    for sgn in (+1.0, -1.0):
        y_outer = np.interp(xs_hook, xs_ref, ys_ref)
        z_outer = surf.sculpted_deck_z(xs_hook, sgn * y_outer)
        reach_y = 0.002 + 0.041 * u_env
        drop_z = 0.002 + 0.015 * u_env
        hgrid = np.zeros((len(xs_hook), len(s_loop), 3))
        href = np.zeros_like(hgrid)
        for i in range(len(xs_hook)):
            t_in = 0.5 * (1.0 - np.cos(s_loop))            # 0 at coaming wall -> 1 at inner thigh bead
            thick = 0.0032 * np.sin(s_loop) * (0.4 + 0.6 * u_env[i])
            dy = -reach_y[i] * t_in
            dz = -0.005 - drop_z[i] * (t_in ** 1.25) + thick
            hgrid[i, :, 0] = xs_hook[i]
            hgrid[i, :, 1] = sgn * (y_outer[i] + dy)
            hgrid[i, :, 2] = z_outer[i] + dz
            href[i, :, 0] = xs_hook[i]
            href[i, :, 1] = sgn * (y_outer[i] - 0.5 * reach_y[i])
            href[i, :, 2] = z_outer[i] - 0.005 - 0.5 * drop_z[i]
        parts.append(_grid_mesh(hgrid, wrap_u=False, wrap_v=True, outward_ref=href))

    return trimesh.util.concatenate(parts)


def build_hatch_covers(surf: SculptedKayakSurface) -> trimesh.Trimesh:
    """Flush domed composite/rubber lids + inner hatch sealing rims for bow, day, and stern hatches."""
    parts: list[trimesh.Trimesh] = []
    n_th = 96
    th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    rhos = np.array([0.02, 0.35, 0.70, 0.92, 1.00, 1.015, 0.97, 0.82])
    dz_top = np.array([+0.0035, +0.0030, +0.0018, +0.0002, -0.0020, -0.0110, -0.0115, -0.0065])

    for hs in HATCHES:
        grid = np.zeros((n_th, len(rhos), 3))
        ref = np.zeros_like(grid)
        zc_base = float(surf.base_deck_z(np.array([hs.x0]), np.array([hs.y0]))[0])
        for j, (r, dz) in enumerate(zip(rhos, dz_top)):
            xv = hs.x0 + r * hs.rx_lid * np.cos(th)
            yv = hs.y0 + r * hs.ry_lid * np.sin(th)
            zb = surf.base_deck_z(xv, yv)
            grid[:, j, 0] = xv
            grid[:, j, 1] = yv
            grid[:, j, 2] = zb + dz
            ref[:, j, 0] = hs.x0
            ref[:, j, 1] = hs.y0
            ref[:, j, 2] = zc_base - 0.005
        grid = surf.end_shape_xyz(grid.reshape(-1, 3)).reshape(grid.shape)
        parts.append(_grid_mesh(grid, wrap_u=True, wrap_v=True, outward_ref=ref))
        parts.append(_cap_ring(grid[:, 0, :], np.array([0.0, 0.0, +1.0])))
        parts.append(_cap_ring(grid[:, -1, :], np.array([0.0, 0.0, -1.0])))
    return trimesh.util.concatenate(parts)


def _section_ring_3d(surf: SculptedKayakSurface, x_keel: float, x_deck: float, inset: float = 0.0085) -> np.ndarray:
    """Closed 3D inner perimeter loop at a transverse or slanted bulkhead station."""
    x_mid = 0.5 * (x_keel + x_deck)
    idx = int(np.argmin(np.abs(surf.xs - x_mid)))
    hr = surf.hull_rings[idx].copy()  # port seam -> keel -> stbd seam
    zs_min, zs_max = float(hr[:, 1].min()), float(surf.deck_cl_z[idx])
    ys_dk = np.linspace(float(hr[-1, 0]), float(hr[0, 0]), 33)[1:-1]
    zs_dk = surf.sculpted_deck_z(np.full_like(ys_dk, x_deck), ys_dk) - 0.0030
    yz = np.vstack((hr, np.column_stack((ys_dk, zs_dk))))
    c_yz = np.array([0.0, 0.5 * (zs_min + zs_max)])
    vec = yz - c_yz
    nrm = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-4)
    yz_in = yz - inset * (vec / nrm)
    tau = np.clip((yz_in[:, 1] - zs_min) / max(zs_max - zs_min, 1e-4), 0.0, 1.0)
    xs = x_keel + (x_deck - x_keel) * tau
    return surf.end_shape_xyz(np.column_stack((xs, yz_in[:, 0], yz_in[:, 1])))


def _slab_from_ring(ring: np.ndarray, half_thick: float = 0.0035) -> trimesh.Trimesh:
    r_fwd = ring.copy()
    r_aft = ring.copy()
    r_fwd[:, 0] -= half_thick
    r_aft[:, 0] += half_thick
    band = _grid_mesh(np.stack((r_fwd, r_aft), axis=0), wrap_u=False, wrap_v=True, outward_ref=ring.mean(axis=0, keepdims=True))
    return trimesh.util.concatenate([
        _cap_ring(r_fwd, np.array([-1.0, 0.0, 0.0])),
        _cap_ring(r_aft, np.array([+1.0, 0.0, 0.0])),
        band,
    ])


def build_interior_and_bulkheads(surf: SculptedKayakSurface) -> trimesh.Trimesh:
    """Composite Kevlar/glass interior liner + 3 watertight bulkheads (-1.08 m, +0.44 m slanted, +0.94 m)."""
    i0, i1 = 8, len(surf.xs) - 8
    xs_sub = surf.xs[i0:i1]
    hr = surf.hull_rings[i0:i1].copy()
    grid = np.zeros((len(xs_sub), hr.shape[1], 3))
    center_ref = np.zeros_like(grid)
    for k, xv in enumerate(xs_sub):
        c_yz = np.array([0.0, float(surf.seam_z[i0 + k]) + 0.02])
        vec = c_yz - hr[k]
        nrm = np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-4)
        yz_in = hr[k] + 0.0045 * (vec / nrm)
        grid[k, :, 0] = xv
        grid[k, :, 1:] = yz_in
        center_ref[k, :, 0] = xv
        center_ref[k, :, 1:] = hr[k]  # outward_ref = outer hull -> normals point INWARD into cockpit!
    grid = surf.end_shape_xyz(grid.reshape(-1, 3)).reshape(grid.shape)
    center_ref = surf.end_shape_xyz(center_ref.reshape(-1, 3)).reshape(center_ref.shape)
    liner = _grid_mesh(grid, wrap_u=False, wrap_v=False, outward_ref=center_ref)

    bh_fwd = _slab_from_ring(_section_ring_3d(surf, -0.95, -0.95))
    bh_seat = _slab_from_ring(_section_ring_3d(surf, +0.42, +0.46))
    return trimesh.util.concatenate([liner, bh_fwd, bh_seat])


def build_day_pod_and_structural_grid(surf: SculptedKayakSurface) -> trimesh.Trimesh:
    """Contoured starboard carbon day pod (~5.5 L) + cruciform (+) rear-deck UD grid + flush 60 mm UD keel strap."""
    parts: list[trimesh.Trimesh] = []

    # 1. Self-contained contoured starboard carbon day pod beneath x=+0.66m, y=+0.118m day hatch
    n_th = 64
    th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    # 7 vertical stations from hatch neck (s=0) through expanded belly to rounded floor (s=1)
    s_lvls = np.array([0.0, 0.14, 0.35, 0.62, 0.84, 0.96, 1.0])
    rx_s = np.array([0.072, 0.095, 0.128, 0.132, 0.118, 0.082, 0.015])
    ry_s = np.array([0.072, 0.078, 0.084, 0.082, 0.072, 0.050, 0.010])
    pgrid = np.zeros((len(s_lvls), n_th, 3))
    pref = np.zeros_like(pgrid)
    z_c_pod = float(surf.base_deck_z(np.array([0.66]), np.array([0.118]))[0]) - 0.060
    for i, sv in enumerate(s_lvls):
        xv = 0.66 + rx_s[i] * np.cos(th)
        yv = 0.118 + ry_s[i] * np.sin(th)
        z_dk_loc = surf.base_deck_z(xv, yv) - 0.0095
        # Conform top flange to cambered deck; slope bottom upward outboard (+y) to clear bilge >= 2.5 cm
        bilge_tilt = 0.020 * np.clip((yv - 0.11) / 0.10, 0.0, 1.0) * (sv ** 1.15)
        zv = z_dk_loc - 0.112 * sv + bilge_tilt
        pgrid[i, :, 0] = xv
        pgrid[i, :, 1] = yv
        pgrid[i, :, 2] = zv
        pref[i, :, 0] = 0.66
        pref[i, :, 1] = 0.118
        pref[i, :, 2] = z_c_pod
    pgrid = surf.end_shape_xyz(pgrid.reshape(-1, 3)).reshape(pgrid.shape)
    parts.append(_grid_mesh(pgrid, wrap_u=False, wrap_v=True, outward_ref=pref))
    parts.append(_cap_ring(pgrid[-1], np.array([0.0, 0.0, -1.0])))

    # 2. Cruciform (+) Rear-Deck Grid: unbroken centerline 18x15 mm UD rib + 20x20 mm transverse arch beam @ +0.94 m
    hat_yz = np.array([[-0.011, -0.0065], [-0.0085, -0.021], [+0.0085, -0.021], [+0.011, -0.0065]])
    for xa, xb in ((+0.46, +1.23), (+1.69, +2.38)):
        xr = np.linspace(xa, xb, 24)
        z_dk = surf.sculpted_deck_z(xr, np.zeros_like(xr))
        rg = np.zeros((len(xr), 4, 3))
        rref = np.zeros_like(rg)
        for j, (hy, hz) in enumerate(hat_yz):
            rg[:, j, 0] = xr
            rg[:, j, 1] = hy
            rg[:, j, 2] = z_dk + hz
            rref[:, j, 0] = xr
            rref[:, j, 2] = z_dk - 0.012
        parts.append(_grid_mesh(surf.end_shape_xyz(rg.reshape(-1, 3)).reshape(rg.shape), wrap_u=False, wrap_v=True,
                                outward_ref=surf.end_shape_xyz(rref.reshape(-1, 3)).reshape(rref.shape)))

    # Curved transverse arch deck beam at x = +0.94 m (beveled to 4 mm at sheer ends, wide open below!)
    sy_94 = float(np.interp(0.94, surf.xs, surf.seam_y)) - 0.022
    ys_bm = np.linspace(-sy_94, +sy_94, 35)
    taper_bm = 0.25 + 0.75 * np.sin(np.pi * np.clip((ys_bm + sy_94) / (2.0 * sy_94), 0.0, 1.0)) ** 0.6
    z_bm = surf.sculpted_deck_z(np.full_like(ys_bm, 0.94), ys_bm)
    hat_xz = np.array([[-0.012, -0.0065], [-0.0095, -0.025], [+0.0095, -0.025], [+0.012, -0.0065]])
    bg = np.zeros((len(ys_bm), 4, 3))
    bref = np.zeros_like(bg)
    for j, (hx, hz) in enumerate(hat_xz):
        bg[:, j, 0] = 0.94 + hx
        bg[:, j, 1] = ys_bm
        bg[:, j, 2] = z_bm + (-0.0065 + (hz + 0.0065) * taper_bm)
        bref[:, j, 0] = 0.94
        bref[:, j, 1] = ys_bm
        bref[:, j, 2] = z_bm - 0.013
    parts.append(_grid_mesh(bg, wrap_u=False, wrap_v=True, outward_ref=bref))

    # 3. Flush 60 mm-wide 0-deg UD-carbon internal keel strap (bow + heel floor + stern; clear under seat!)
    ys_k = np.linspace(-0.030, +0.030, 11)
    for xa, xb in ((-2.38, -0.97), (-0.92, -0.11), (+0.47, +2.38)):
        xk = np.linspace(xa, xb, 30)
        kg = np.zeros((len(xk), len(ys_k), 3))
        for i, xv in enumerate(xk):
            idx = int(np.argmin(np.abs(surf.xs - xv)))
            hr_ring = surf.hull_rings[idx]
            ord_y = np.argsort(hr_ring[:, 0])
            z_loc = np.interp(ys_k, hr_ring[ord_y, 0], hr_ring[ord_y, 1])
            kg[i, :, 0] = xv
            kg[i, :, 1] = ys_k
            kg[i, :, 2] = z_loc + 0.0056
        kg = surf.end_shape_xyz(kg.reshape(-1, 3)).reshape(kg.shape)
        parts.append(_grid_mesh(kg, outward_ref=kg - np.array([0.0, 0.0, 0.04])))
    return trimesh.util.concatenate(parts)


def _box_mesh(cx: float, cy: float, cz: float, lx: float, ly: float, lz: float) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(lx, ly, lz))
    m.apply_translation((cx, cy, cz))
    return m


def build_seat_and_footboard(surf: SculptedKayakSurface) -> trimesh.Trimesh:
    """10 mm low fixed twist-friendly surfski/FSK bucket pan + lumbar backband + angled wing footboard & pedals."""
    parts: list[trimesh.Trimesh] = []

    # 1. Low fixed dished seat pan (x in [-0.07, +0.30] m; sit-bone pocket at x = +0.14 m, ~15 mm above outer keel)
    xs_s = np.linspace(-0.07, +0.30, 36)
    us_s = np.linspace(-1.0, +1.0, 41)
    top_g = np.zeros((len(xs_s), len(us_s), 3))
    for i, xv in enumerate(xs_s):
        tx = (xv - (-0.07)) / 0.37  # 0 at front edge -> 1 at rear lip
        hw = 0.168 + 0.024 * np.sin(np.pi * np.clip(tx * 1.15, 0.0, 1.0)) - 0.032 * (tx ** 4)
        yv = us_s * hw
        zk = float(surf.keel_at(np.array([xv]))[0])
        # Low 10 mm sit-bone floor + smooth cut-down hip cradle + subtle center coccyx groove + front thigh pommel
        side_cup = (0.040 + 0.022 * smoothstep((tx - 0.35) / 0.45)) * (np.abs(us_s) ** 2.3)
        rear_lip = 0.036 * smoothstep((tx - 0.72) / 0.28)
        front_pommel = 0.014 * (1.0 - smoothstep(tx / 0.32)) * np.exp(-((us_s / 0.28) ** 2))
        sit_pocket = -0.004 * np.exp(-(((xv - 0.14) / 0.09) ** 2)) * np.exp(-(((np.abs(yv) - 0.058) / 0.042) ** 2))
        zv = zk + 0.015 + side_cup + rear_lip + front_pommel + sit_pocket
        top_g[i, :, 0] = xv
        top_g[i, :, 1] = yv
        top_g[i, :, 2] = zv

    bot_g = top_g.copy()
    bot_g[:, :, 2] -= 0.0055
    # Closed slab: top + bottom + 4 perimeter strips
    parts.append(_grid_mesh(top_g, outward_ref=top_g - np.array([0.0, 0.0, 0.05])))
    parts.append(_grid_mesh(bot_g, outward_ref=bot_g + np.array([0.0, 0.0, 0.05])))
    for r_a, r_b, ref_off in (
        (top_g[0], bot_g[0], np.array([+0.05, 0.0, 0.0])),
        (top_g[-1], bot_g[-1], np.array([-0.05, 0.0, 0.0])),
        (top_g[:, 0], bot_g[:, 0], np.array([0.0, +0.05, 0.0])),
        (top_g[:, -1], bot_g[:, -1], np.array([0.0, -0.05, 0.0])),
    ):
        parts.append(_grid_mesh(np.stack((r_a, r_b), axis=0), outward_ref=0.5 * (r_a + r_b) + ref_off))

    # 2. Low contoured lumbar backband (x ~ +0.32 m, top at z = 0.178 m — 6 cm below aft coaming lip for layback!)
    th_bb = np.linspace(-0.44 * np.pi, +0.44 * np.pi, 29)
    zs_bb = np.linspace(0.118, 0.174, 10)
    bb_f = np.zeros((len(th_bb), len(zs_bb), 3))
    for i, a in enumerate(th_bb):
        bb_f[i, :, 0] = 0.255 + 0.068 * np.cos(a)
        bb_f[i, :, 1] = 0.168 * np.sin(a)
        bb_f[i, :, 2] = zs_bb
    bb_b = bb_f.copy()
    bb_b[:, :, 0] += 0.007
    parts.append(_grid_mesh(bb_f, outward_ref=bb_f + np.array([+0.05, 0.0, 0.0])))
    parts.append(_grid_mesh(bb_b, outward_ref=bb_b - np.array([+0.05, 0.0, 0.0])))

    # 3. Full-width 12-deg wing footboard + pull-bar + gas-pedal rudder toe controls at x = -0.82 m (inside -0.95 m fwd BH)
    us_f = np.linspace(-1.0, +1.0, 27)
    vs_f = np.linspace(0.0, 1.0, 15)
    fb_f = np.zeros((len(us_f), len(vs_f), 3))
    for j, vf in enumerate(vs_f):
        zv = 0.042 + 0.158 * vf
        hw_f = 0.132 + 0.038 * vf
        xv = -0.835 + 0.034 * vf  # 12 deg comfortable aft tilt
        fb_f[:, j, 0] = xv
        fb_f[:, j, 1] = us_f * hw_f * np.cos(0.15 * (1.0 - vf))
        fb_f[:, j, 2] = zv
    fb_b = fb_f.copy()
    fb_b[:, :, 0] -= 0.008
    parts.append(_grid_mesh(fb_f, outward_ref=fb_f - np.array([+0.05, 0.0, 0.0])))
    parts.append(_grid_mesh(fb_b, outward_ref=fb_b + np.array([+0.05, 0.0, 0.0])))

    # Pivoted upper rudder toe pedals + port/starboard side adjustment tracks (curved inside hull wall, aft of -0.95 m BH)
    for sgn in (-1.0, +1.0):
        pedal = _box_mesh(0.0, 0.0, 0.0, 0.008, 0.074, 0.066)
        pedal.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-18.0), [0.0, 1.0, 0.0]))
        pedal.apply_translation((-0.790, sgn * 0.092, 0.188))
        parts.append(pedal)
        parts.append(
            _cylinder_between(
                np.array([-0.92, sgn * 0.148, 0.132]),
                np.array([-0.68, sgn * 0.164, 0.132]),
                radius=0.0075,
                sections=12,
            )
        )
        # Twin outboard seat tracks at y = +-12.5 cm + front thumb-screw clamp knobs (cleanly inside bilge & below seat)
        parts.append(_box_mesh(+0.090, sgn * 0.125, 0.025, 0.320, 0.016, 0.006))
        parts.append(_cylinder_between(np.array([-0.068, sgn * 0.125, 0.028]), np.array([-0.068, sgn * 0.125, 0.041]), radius=0.009))
    # Central foot pull-strap bar across footboard
    parts.append(_box_mesh(-0.798, 0.0, 0.145, 0.012, 0.236, 0.022))
    return trimesh.util.concatenate(parts)



def _cylinder_between(p0: np.ndarray, p1: np.ndarray, radius: float, sections: int = 14) -> trimesh.Trimesh:
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    z_axis = np.array([0.0, 0.0, 1.0])
    v_dir = vec / max(length, 1e-9)
    axis = np.cross(z_axis, v_dir)
    if np.linalg.norm(axis) > 1e-7:
        ang = float(np.arccos(np.clip(np.dot(z_axis, v_dir), -1.0, 1.0)))
        cyl.apply_transform(trimesh.transformations.rotation_matrix(ang, axis))
    elif np.dot(z_axis, v_dir) < 0.0:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0]))
    cyl.apply_translation(0.5 * (p0 + p1))
    return cyl


def build_overstern_rudder(surf: SculptedKayakSurface) -> trimesh.Trimesh:
    """Over-stern flip-up NACA-0012 foil rudder, gudgeon bracket, stainless pin, tiller yoke & deck cable tubes."""
    parts: list[trimesh.Trimesh] = []

    # 1. Stern gudgeon mounting bracket + upper/lower pin lugs + stainless hinge pin
    parts.append(_box_mesh(2.724, 0.0, 0.155, 0.036, 0.013, 0.120))
    parts.append(_box_mesh(2.740, 0.0, 0.205, 0.016, 0.014, 0.018))
    parts.append(_box_mesh(2.740, 0.0, 0.106, 0.016, 0.014, 0.018))
    parts.append(_cylinder_between(np.array([2.745, 0.0, 0.092]), np.array([2.745, 0.0, 0.232]), radius=0.0028))

    # 2. Pivot stock head + 14.4 cm steering tiller yoke
    parts.append(_box_mesh(2.766, 0.0, 0.206, 0.040, 0.016, 0.038))
    parts.append(_box_mesh(2.746, 0.0, 0.226, 0.018, 0.144, 0.006))

    # 3. Swept NACA-0012 kick-up foil blade (z = +0.214 m down to -0.078 m, 16.6 cm submerged below 225 lb WL)
    n_span = 36
    zs = np.linspace(0.214, -0.078, n_span)
    s_span = np.linspace(0.0, 1.0, n_span)
    x_le = 2.750 + 0.022 * (s_span ** 1.45)
    chord = 0.052 * np.sqrt(np.clip(1.0 - 0.68 * (s_span ** 2.2), 0.12, 1.0))

    th_f = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    xc = 0.5 * (1.0 - np.cos(th_f))  # cosine-clustered chordwise 0 -> 1 -> 0
    sgn_y = np.where(th_f < np.pi, 1.0, -1.0)
    yt_unit = 5.0 * 0.12 * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1036 * xc**4)

    fgrid = np.zeros((n_span, len(th_f), 3))
    fref = np.zeros_like(fgrid)
    for i in range(n_span):
        fgrid[i, :, 0] = x_le[i] + chord[i] * xc
        fgrid[i, :, 1] = sgn_y * chord[i] * yt_unit
        fgrid[i, :, 2] = zs[i]
        fref[i, :, 0] = x_le[i] + 0.35 * chord[i]
        fref[i, :, 2] = zs[i]
    parts.append(_grid_mesh(fgrid, wrap_u=False, wrap_v=True, outward_ref=fref))
    parts.append(_cap_ring(fgrid[0], np.array([0.0, 0.0, +1.0])))
    parts.append(_cap_ring(fgrid[-1], np.array([0.0, 0.0, -1.0])))

    # 4. Aft-deck recessed cable exit fairleads + Spectra steering runs to yoke tips
    for sgn in (-1.0, +1.0):
        z_exit = float(surf.sculpted_deck_z(np.array([2.36]), np.array([sgn * 0.048]))[0])
        p_exit = surf.end_shape_xyz(np.array([[2.36, sgn * 0.048, z_exit + 0.004]]))[0]
        p_yoke = np.array([2.746, sgn * 0.068, 0.226])
        parts.append(_cylinder_between(p_exit - np.array([0.035, 0.0, 0.005]), p_exit + np.array([0.015, 0.0, 0.002]), radius=0.0045))
        parts.append(_cylinder_between(p_exit, p_yoke, radius=0.0016, sections=10))
    return trimesh.util.concatenate(parts)
