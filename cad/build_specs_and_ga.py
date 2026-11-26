"""Compute Kayak 10.9 3-load hydrostatics/speeds (kayak-10-9-specs.json) and render Plates 2 & 7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import trimesh

from deck_geometry import HATCHES, Z_WL_225, Z_WL_265, SculptedKayakSurface, keyhole_inner_curve, keyhole_offset_curve
from hydro import KN, LB, NU_SEAWATER, RHO_SEAWATER, float_at, half_breadth_grid, ittc57, michell
from kayakgen.eval.form_metrics import _plan_hollow, evaluate_form_metrics
from kayakgen.model.hull import Hull
from outfitting_geometry import (
    build_coaming_and_thigh_hooks,
    build_day_pod_and_structural_grid,
    build_hatch_covers,
    build_interior_and_bulkheads,
    build_overstern_rudder,
    build_seat_and_footboard,
)

KG = 0.25
SP = np.arange(3.5, 7.76, 0.25)
PW = (40.0, 60.0, 90.0, 120.0)
ANG = (10, 20, 30, 40)


def _station_beams_ge_10cm(V: np.ndarray, F: np.ndarray, lcg: float, T: float, bowsign: int, offs: list[float]) -> dict[str, float]:
    tm = trimesh.Trimesh(V, F, process=False)
    out = {}
    for off in offs:
        sl = trimesh.intersections.mesh_plane(tm, [1, 0, 0], [lcg + bowsign * off, 0, 0]).reshape(-1, 3)
        u = sl[sl[:, 2] - T >= 0.10, 1]
        out[f"{off:+.1f}"] = round(float(u.max() - u.min()) * 100, 1) if len(u) >= 2 else 0.0
    return out


def _mesh_plan_hollow_deg(V: np.ndarray, F: np.ndarray) -> float:
    tm = trimesh.Trimesh(V, F, process=False)
    xs = np.linspace(float(V[:, 0].min()) + 0.02, float(V[:, 0].max()) - 0.02, 150)
    hbs = []
    for xv in xs:
        sl = trimesh.intersections.mesh_plane(tm, [1, 0, 0], [float(xv), 0, 0]).reshape(-1, 3)
        hbs.append(float(np.abs(sl[:, 1]).max()) if len(sl) else 0.0)
    hol, _ = _plan_hollow(xs - 0.5 * (xs[0] + xs[-1]), np.array(hbs), 0.90)
    return round(hol, 2)


def _crown_drop_10cm_mm(hull: Hull, x: float = -1.0) -> float:
    sec = hull.to_geometry().section(x, "deck")
    mask = sec[:, 0] >= 0
    ys, zs = sec[mask, 0], sec[mask, 1]
    order = np.argsort(ys)
    return round(float(zs[order][0] - np.interp(0.10, ys[order], zs[order])) * 1000, 1)


def full_eval_3load(label: str, V: np.ndarray, F: np.ndarray, bowsign: int, hull: Hull) -> dict:
    tm = trimesh.Trimesh(V, F, process=False)
    r4 = trimesh.intersections.mesh_plane(tm, [0, 0, 1], [0, 0, 0.1016]).reshape(-1, 3)
    l4, b4 = float(np.ptp(r4[:, 0])), float(np.ptp(r4[:, 1]))
    loa = float(np.ptp(V[:, 0]))
    fm = evaluate_form_metrics(hull, catch_zone_forward_m=(0.8, 1.2), foot_box_width_m=0.28, foot_box_height_m=0.25)
    crown = hull.distribution_v2.topsides.deck_crown_exponent
    res: dict = {
        "label": label,
        "L_oa_m": round(loa, 3),
        "B_oa_cm": round(float(np.ptp(V[:, 1])) * 100, 1),
        "D_max_cm": round(float(np.ptp(V[:, 2])) * 100, 1),
        "fsk_4in_ratio": round(l4 / b4, 2),
        "b_4in_cm": round(b4 * 100, 2),
        "usca_4in_beam_pct": round(100.0 * b4 / loa, 2),
        "mesh_plan_hollow_deg": _mesh_plan_hollow_deg(V, F),
        "plan_hollow_deg": round(float(fm.plan_hollow_max_deg or 0.0), 2),
        "foot_clear_height_cm": round(float(fm.foot_clear_height_m) * 100, 2),
        "foot_clear_width_at_25cm": round(float(fm.foot_clear_width_m) * 100, 2),
        "upper_knuckle_deg": round(float(fm.upper_knuckle_max_deg), 2),
        "waterline_knuckle_deg": round(float(fm.waterline_knuckle_max_deg), 2),
        "upper_hollow_deg": round(float(fm.upper_hollow_max_deg), 2),
        "foredeck_drop_at_10cm_mm": _crown_drop_10cm_mm(hull, -1.0),
        "crown_fwd_aft": [round(crown.forward, 2), round(crown.aft, 2)] if hasattr(crown, "forward") else [round(float(crown), 2), round(float(crown), 2)],
    }
    for lb in (225, 265, 300):
        M = lb * LB
        r = float_at(V, F, M, RHO_SEAWATER, np.array([0.0, 0.0, KG]), free_trim=False)
        lcg, T = float(r["B"][0]), float(-r["zmin"])
        Lwl, Bwl = float(np.ptp(r["wl_x"])), float(np.ptp(r["wl_y"]))
        GM = float(r["B"][2] - r["zmin"] + r["IT"] / r["vol"] - KG)
        gz = [round(float(-float_at(V, F, M, RHO_SEAWATER, np.array([lcg, 0.0, KG]), heel=np.radians(a))["GZ"] * 1000), 1) for a in ANG]
        if lb == 225:
            xs, zs, Y = half_breadth_grid(r["tris"], 401, 81)
            R = np.array([ittc57(k * KN, Lwl, r["wet"], RHO_SEAWATER, NU_SEAWATER)[0] + michell(xs, zs, Y, k * KN, RHO_SEAWATER, 4000) for k in SP])
            res["kn_225"] = {int(p): round(float(np.interp(p, R * SP * KN, SP)), 2) for p in PW}
            # Unrounded values: the reference for exact speed comparisons (the 2-decimal table alone
            # hides up to 0.005 kn per power level).
            res["kn_225_exact"] = {int(p): float(np.interp(p, R * SP * KN, SP)) for p in PW}
            res["catch_beams_225_cm"] = _station_beams_ge_10cm(V, F, lcg, T, bowsign, [0.0, 0.8, 0.9, 1.0, 1.1, 1.2])
            grid_offs = [1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4]
            grid_b = _station_beams_ge_10cm(V, F, lcg, T, bowsign, grid_offs)
            vals = [grid_b[f"{o:+.1f}"] for o in grid_offs]
            res["transition_beams_cm"] = grid_b
            res["transition_dB_per_20cm"] = [round(vals[i + 1] - vals[i], 1) for i in range(len(vals) - 1)]
            res["heel_bow_pitch_deg"] = {f"{a}deg": round(float(-bowsign * np.degrees(float_at(V, F, M, RHO_SEAWATER, np.array([lcg, 0.0, KG]), heel=np.radians(a), free_trim=True)["trim"])), 3) for a in (15, 25)}
        res[f"load_{lb}"] = {"T_cm": round(T * 100, 2), "Bwl_cm": round(Bwl * 100, 2), "wet_m2": round(float(r["wet"]), 3), "GM_cm": round(GM * 100, 2), "GM_cm_exact": float(GM * 100), "gz_10_20_30_40_mm": gz}
    return res


def compute_compartment_volumes_liters(outer_closed: trimesh.Trimesh) -> dict[str, float]:
    xs = np.linspace(-2.72, 2.72, 240)
    areas = np.zeros_like(xs)
    for i, xv in enumerate(xs):
        sec = trimesh.intersections.mesh_plane(outer_closed, [1.0, 0.0, 0.0], [float(xv), 0.0, 0.0])
        if len(sec) >= 3:
            p0, p1 = sec[:, 0, 1:], sec[:, 1, 1:]
            c = 0.5 * (p0.mean(axis=0) + p1.mean(axis=0))
            d0, d1 = p0 - c, p1 - c
            areas[i] = 0.5 * float(np.sum(np.abs(d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0])))

    def vol_between(xa: float, xb: float) -> float:
        m = (xs >= xa) & (xs <= xb)
        return round(float(np.trapezoid(areas[m], xs[m])) * 1000.0 * 0.94, 1)

    day_pod_l = 5.5
    return {
        "bow_compartment_L": vol_between(-2.735, -0.95),
        "cockpit_foot_and_seat_well_L": vol_between(-0.95, +0.44),
        "day_pod_L": day_pod_l,
        "stern_compartment_L": round(vol_between(+0.44, +2.735) - day_pod_l, 1),
    }


def _camera_axes(azim_deg: float, elev_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    eye_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross([0.0, 0.0, 1.0], eye_dir)
    right /= np.linalg.norm(right)
    up = np.cross(eye_dir, right)
    up /= np.linalg.norm(up)
    return right, up, eye_dir


def render_shaded_panel(ax: plt.Axes, part_list: list[tuple[trimesh.Trimesh, tuple[float, float, float, float], float]], azim_deg: float, elev_deg: float, x_clip: tuple[float, float] | None, title: str) -> None:
    right, up, eye_dir = _camera_axes(azim_deg, elev_deg)
    l1 = np.array([-0.35, -0.45, 0.82]); l1 /= np.linalg.norm(l1)
    l2 = np.array([0.55, 0.35, 0.45]); l2 /= np.linalg.norm(l2)
    half_v = (l1 + eye_dir) / np.linalg.norm(l1 + eye_dir)
    tris_2d, depths, colors = [], [], []
    for mesh, base_rgba, z_bias in part_list:
        tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        cent = tri.mean(axis=1)
        if x_clip is not None:
            keep = (cent[:, 0] >= x_clip[0]) & (cent[:, 0] <= x_clip[1])
            if not np.any(keep):
                continue
            tri, cent = tri[keep], cent[keep]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        fn[np.sum(fn * eye_dir, axis=1) < 0.0] *= -1.0
        diff = 0.28 + 0.56 * np.clip(fn @ l1, 0.0, 1.0) + 0.18 * np.clip(fn @ l2, 0.0, 1.0)
        spec = 0.16 * (np.clip(fn @ half_v, 0.0, 1.0) ** 22)
        rgb = np.clip(np.asarray(base_rgba[:3])[None, :] * diff[:, None] + spec[:, None], 0.0, 1.0)
        tris_2d.append(np.stack((tri @ right, tri @ up), axis=2))
        depths.append(cent @ eye_dir + z_bias)
        colors.append(np.column_stack((rgb, np.full(len(rgb), base_rgba[3]))))
    all_2d, all_d, all_c = np.concatenate(tris_2d, 0), np.concatenate(depths, 0), np.concatenate(colors, 0)
    order = np.argsort(all_d)
    ec = "none" if np.any(all_c[:, 3] < 0.99) else all_c[order]
    ax.add_collection(PolyCollection(all_2d[order], facecolors=all_c[order], edgecolors=ec, linewidths=0.10, antialiased=True))
    pts = all_2d.reshape(-1, 2)
    xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
    px, py = 0.04 * (xmax - xmin), 0.08 * (ymax - ymin)
    ax.set_xlim(xmin - px, xmax + px); ax.set_ylim(ymin - py, ymax + py); ax.set_aspect("equal")
    ax.set_facecolor("#f3f6f9"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6, color="#182430")


def render_engineering_layout(surf: SculptedKayakSurface, scene_parts: dict[str, trimesh.Trimesh], vols: dict[str, float], specs: dict, out_png: Path) -> None:
    fig = plt.figure(figsize=(16.5, 11.5), dpi=200, facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.15, 1.30], hspace=0.34, wspace=0.22)
    ax_plan, ax_prof = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    sec_axes = [fig.add_subplot(gs[2, k]) for k in range(4)]
    xs, outer = surf.xs, scene_parts["outer_closed"]
    wl_stbd = trimesh.intersections.mesh_plane(outer, [0, 0, 1], [0, 0, Z_WL_225]).reshape(-1, 3)
    wl_stbd = wl_stbd[wl_stbd[:, 1] >= 0]
    wl_ord = np.argsort(wl_stbd[:, 0])
    max_y = np.max(np.abs(surf.hull_rings[:, :, 0]), axis=1)

    ax_plan.fill_between(xs, -max_y, +max_y, color="#dde9f3", alpha=0.55, label=f"Overall beam ({specs['B_oa_cm']:.1f} cm)")
    ax_plan.plot(xs, +max_y, color="#1b3650", lw=1.6); ax_plan.plot(xs, -max_y, color="#1b3650", lw=1.6)
    ax_plan.plot(xs, +surf.seam_y, "--", color="#3a729a", lw=1.0, label="Hull/deck mold seam"); ax_plan.plot(xs, -surf.seam_y, "--", color="#3a729a", lw=1.0)
    ax_plan.plot(wl_stbd[wl_ord, 0], +wl_stbd[wl_ord, 1], ":", color="#0c7b6a", lw=1.2, label="225 lb WL"); ax_plan.plot(wl_stbd[wl_ord, 0], -wl_stbd[wl_ord, 1], ":", color="#0c7b6a", lw=1.2)
    th = np.linspace(0, 2 * np.pi, 300)
    kh_in, kh_lip = keyhole_inner_curve(th), keyhole_offset_curve(th, 0.023)
    ax_plan.fill(kh_in[:, 0], kh_in[:, 1], color="#ede4cf", zorder=3)
    ax_plan.plot(kh_lip[:, 0], kh_lip[:, 1], color="#222831", lw=1.8, zorder=5, label="Keyhole coaming (84×42 cm)")
    ax_plan.plot(kh_in[:, 0], kh_in[:, 1], color="#222831", lw=1.1, zorder=5)
    xh = np.linspace(-0.34, +0.02, 40)
    th_s = keyhole_inner_curve(np.linspace(0.20 * np.pi, 0.56 * np.pi, 60))
    yo = np.interp(xh, th_s[np.argsort(th_s[:, 0]), 0], th_s[np.argsort(th_s[:, 0]), 1])
    yi = yo - (0.002 + 0.041 * (np.clip(np.sin(np.pi * (xh + 0.34) / 0.36), 0.0, 1.0) ** 1.35))
    for sgn in (+1, -1):
        ax_plan.fill_between(xh, sgn * yi, sgn * yo, color="#d97724", alpha=0.75, zorder=4, label="Under-deck side thigh hooks (31.2 cm open center)" if sgn == 1 else None)
    for hs in HATCHES:
        ax_plan.fill(hs.x0 + hs.rx_lid * np.cos(th), hs.y0 + hs.ry_lid * np.sin(th), color="#3f4956", alpha=0.85, zorder=4)
        ax_plan.plot(hs.x0 + 1.10 * hs.rx_lid * np.cos(th), hs.y0 + 1.10 * hs.ry_lid * np.sin(th), color="#222831", lw=0.9, zorder=4)
    ax_plan.plot(0.66 + 0.132 * np.cos(th), 0.118 + 0.084 * np.sin(th), "--", color="#0b8a6a", lw=1.2, zorder=4)
    for xa_k, xb_k, lbl_k in ((-2.38, -0.97, "60 mm flush UD keel strap"), (-0.92, -0.11, None), (+0.47, +2.38, None)):
        ax_plan.fill_between([xa_k, xb_k], [-0.030, -0.030], [+0.030, +0.030], color="#334155", alpha=0.30, zorder=2, label=lbl_k)
    ax_plan.plot([+0.46, +1.23], [0.0, 0.0], "-", color="#0b8a6a", lw=2.3, zorder=5, label="Cruciform (+) UD deck grid (+64 g)")
    ax_plan.plot([+1.69, +2.38], [0.0, 0.0], "-", color="#0b8a6a", lw=2.3, zorder=5)
    sy_bm = float(np.interp(0.94, xs, surf.seam_y)) - 0.022
    ax_plan.plot([0.94, 0.94], [-sy_bm, +sy_bm], "-", color="#0b8a6a", lw=2.3, zorder=5)
    for x_bh, lbl in ((-0.95, "Fwd BH\n-0.95m"), (+0.44, "Seat BH\n+0.44m")):
        sy_b = float(np.interp(x_bh, xs, max_y))
        ax_plan.plot([x_bh, x_bh], [-sy_b, +sy_b], color="#b83227", lw=1.7, zorder=6)
        ax_plan.text(x_bh, sy_b + 0.03, lbl, ha="center", va="bottom", fontsize=7.5, color="#982219", fontweight="bold")
    ax_plan.plot([2.746, 2.746], [-0.072, +0.072], color="#333", lw=2.2, zorder=6)
    ax_plan.set_xlim(-2.82, 2.92); ax_plan.set_ylim(-0.42, 0.46); ax_plan.set_aspect("equal"); ax_plan.set_ylabel("y (m)", fontsize=8.5)
    ax_plan.set_title(f"PLATE 2 — Kayak 10.9 General Arrangement: L_oa 5.470 m | B_oa {specs['B_oa_cm']:.1f} cm | SR 4\" Ratio {specs['SR_4in']:.2f} <= 10.99 | {specs['total_sealed_buoyancy_liters']:.1f} L Dry Storage", fontsize=11, fontweight="bold", color="#14202c")
    ax_plan.legend(loc="lower left", ncol=6, fontsize=7.2, framealpha=0.92); ax_plan.grid(True, alpha=0.22)

    keel_pts = surf.end_shape_xyz(np.column_stack((xs, np.zeros_like(xs), surf.keel_z)))
    seam_pts = surf.end_shape_xyz(np.column_stack((xs, surf.seam_y, surf.seam_z)))
    deck_cl = surf.end_shape_xyz(np.column_stack((xs, np.zeros_like(xs), surf.sculpted_deck_z(xs, np.zeros_like(xs)))))
    ax_prof.plot(xs[xs > 0.0], np.full(np.sum(xs > 0.0), 0.273) - 0.015 * np.clip((xs[xs > 0.0] - 2.0) / 0.735, 0, 1), ":", color="#9aa4ae", lw=1.2, label="Stage 9 unfinished rear slab (~18.5 cm above WL)")
    mask_fwd, mask_aft = (deck_cl[:, 0] <= -0.460) & (deck_cl[:, 0] >= -2.709), (deck_cl[:, 0] >= +0.380) & (deck_cl[:, 0] <= +2.724)
    first_k = True
    for seg in trimesh.intersections.mesh_plane(outer, [0, 1, 0], [0, 0, 0]):
        if seg[:, 2].mean() < 0.20 or seg[:, 0].min() < -2.68 or seg[:, 0].max() > +2.68:
            ax_prof.plot(seg[:, 0], seg[:, 2], color="#1b3650", lw=1.9, label="Rounded bow stem, forefoot & swept stern" if first_k else None)
            first_k = False
    ax_prof.plot(deck_cl[mask_fwd, 0], deck_cl[mask_fwd, 2], color="#2a6f9e", lw=2.0, label="Sculpted deck (~14.8 cm low rear deck)")
    ax_prof.plot(deck_cl[mask_aft, 0], deck_cl[mask_aft, 2], color="#2a6f9e", lw=2.0)
    ax_prof.plot(seam_pts[:, 0], seam_pts[:, 2], "--", color="#588bae", lw=1.0, label="Sheer mold seam")
    kh_s = keyhole_offset_curve(np.linspace(0.0, np.pi, 100), 0.004)
    ax_prof.plot(kh_s[:, 0], surf.sculpted_deck_z(kh_s[:, 0], kh_s[:, 1]) + 0.0185, color="#222831", lw=2.2, label="Coaming lip (recessed aft well)")
    ax_prof.axhline(Z_WL_225, color="#0c7b6a", ls="-.", lw=1.2, label=f"225 lb WL ({Z_WL_225*100:.1f} cm)")
    ax_prof.axhline(Z_WL_265, color="#2b9348", ls=":", lw=1.1, label=f"265 lb Baja WL ({Z_WL_265*100:.1f} cm)")
    for xk, xd in ((-0.95, -0.95), (+0.42, +0.46)):
        ax_prof.plot([xk, xd], [float(np.interp(xk, keel_pts[:, 0], keel_pts[:, 2])), float(np.interp(xd, deck_cl[:, 0], deck_cl[:, 2]))], color="#b83227", lw=2.1)
    ax_prof.fill([0.53, 0.79, 0.76, 0.56], [0.224, 0.221, 0.114, 0.116], color="#0b8a6a", alpha=0.22, edgecolor="#0b8a6a", ls="--", lw=1.2)
    z94 = float(np.interp(0.94, deck_cl[:, 0], deck_cl[:, 2]))
    ax_prof.fill([0.928, 0.952, 0.950, 0.930], [z94 - 0.005, z94 - 0.005, z94 - 0.025, z94 - 0.025], color="#0b8a6a", alpha=0.9)
    ax_prof.plot([-0.07, 0.14, 0.30], [0.028, 0.015, 0.052], color="#222831", lw=2.6, label="10 mm fixed seat + wing footboard")
    ax_prof.plot([0.31, 0.32], [0.118, 0.174], color="#222831", lw=2.6); ax_prof.plot([-0.835, -0.801], [0.042, 0.200], color="#222831", lw=2.6)
    ax_prof.fill([2.750, 2.802, 2.792, 2.772], [0.214, 0.214, -0.078, -0.078], color="#6f7c8a", alpha=0.85, label="Over-stern NACA-0012 rudder")
    for cx, cz, txt in [(-1.82, 0.16, f"Bow Hatch (34×14 cm)\n{vols['bow_compartment_L']:.1f} L dry"), (-0.26, 0.14, f"Cockpit Well\n{vols['cockpit_foot_and_seat_well_L']:.1f} L"), (+0.66, 0.17, f"Carbon Pod\n{vols['day_pod_L']:.1f} L"), (+1.65, 0.15, f"Unified Stern (42×24 cm)\n{vols['stern_compartment_L']:.1f} L dry")]:
        ax_prof.text(cx, cz, txt, ha="center", va="center", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cbd5e1", alpha=0.88))
    ax_prof.set_xlim(-2.82, 2.92); ax_prof.set_ylim(-0.10, 0.37); ax_prof.set_aspect("equal")
    ax_prof.set_xlabel("Longitudinal station x from midship (m)   [Bow at -2.735 m, Stern at +2.735 m]", fontsize=8.8); ax_prof.set_ylabel("z above keel (m)", fontsize=8.5)
    ax_prof.legend(bbox_to_anchor=(0.5, 1.24), loc="center", ncol=4, fontsize=7.5, framealpha=0.95); ax_prof.grid(True, alpha=0.22)

    slice_meshes = [(scene_parts["hull"], "#1b3650", 2.0), (scene_parts["deck"], "#2a6f9e", 2.0), (scene_parts["coaming"], "#1f242c", 2.2), (scene_parts["hatches"], "#45505e", 1.8), (scene_parts["seat_fb"], "#b83227", 1.9), (scene_parts["pod_grid"], "#0b8a6a", 2.2)]
    for ax_s, (xv, stitle) in zip(sec_axes, [(-0.82, "x = -0.82 m: Wing Footboard,\nPedals & 60 mm UD Keel Strap"), (-0.18, "x = -0.18 m: Hybrid Thigh Hooks\n(31.2 cm Open Wing-Knee Arch)"), (+0.14, "x = +0.14 m: 10 mm Seat & Twin\n±12.5 cm Foil Tracks"), (+0.66, "x = +0.66 m: Starboard Carbon Pod,\nCenter UD Rib & Keel Strap")]):
        for m_part, col, lw in slice_meshes:
            for seg in trimesh.intersections.mesh_plane(m_part, [1.0, 0.0, 0.0], [float(xv), 0.0, 0.0]):
                ax_s.plot(seg[:, 1] * 100.0, seg[:, 2] * 100.0, color=col, lw=lw)
        if xv == -0.82:
            ax_s.fill([-13.2, 13.2, 16.8, -16.8], [4.2, 4.2, 20.0, 20.0], color="#b83227", alpha=0.22)
            ax_s.plot([-13.2, 13.2, 16.8, -16.8, -13.2], [4.2, 4.2, 20.0, 20.0, 4.2], color="#b83227", lw=1.7)
            for sgn in (-1, +1):
                ax_s.fill([sgn * 5.5, sgn * 12.9, sgn * 12.9, sgn * 5.5], [15.5, 15.5, 22.1, 22.1], color="#38414e", alpha=0.75)
        ax_s.axhline(Z_WL_225 * 100.0, color="#0c7b6a", ls="-.", lw=1.0)
        ax_s.set_xlim(-31.0, 31.0); ax_s.set_ylim(-2.0, 36.0); ax_s.set_aspect("equal"); ax_s.set_title(stitle, fontsize=8.6, fontweight="bold"); ax_s.set_xlabel("y (cm)", fontsize=8.0)
        if xv == -0.82:
            ax_s.set_ylabel("z above keel (cm)", fontsize=8.0)
        ax_s.grid(True, alpha=0.25)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs-out", type=Path, default=REPO_ROOT / "data" / "kayak-10-9-specs.json")
    ap.add_argument("--plates-dir", type=Path, default=REPO_ROOT / "build" / "plates")
    args = ap.parse_args()
    args.plates_dir.mkdir(parents=True, exist_ok=True)

    surf = SculptedKayakSurface(REPO_ROOT / "hull" / "kayak-10-9.hull.json")
    m_hull, m_deck = surf.build_hull_shell(), surf.build_deck_shell(cut_openings=True)
    outer_closed = trimesh.util.concatenate([m_hull, surf.build_deck_shell(cut_openings=False)])
    outer_closed.merge_vertices(merge_tex=True, merge_norm=True)
    assert outer_closed.is_watertight
    m_coam, m_hatch, m_int = build_coaming_and_thigh_hooks(surf), build_hatch_covers(surf), build_interior_and_bulkheads(surf)
    m_pod_grid, m_seat, m_rud = build_day_pod_and_structural_grid(surf), build_seat_and_footboard(surf), build_overstern_rudder(surf)

    V_out, F_out = np.asarray(outer_closed.vertices), np.asarray(outer_closed.faces)
    ev3 = full_eval_3load("fsk-d10-finished", V_out, F_out, -1, surf.hull)
    r4 = trimesh.intersections.mesh_plane(outer_closed, [0, 0, 1], [0, 0, 0.1016]).reshape(-1, 3)
    vols = compute_compartment_volumes_liters(outer_closed)
    specs = {
        "model": "sporty-tourer-fsk-d10-finished",
        "L_oa_m": round(float(np.ptp(V_out[:, 0])), 4),
        "B_oa_cm": round(float(np.ptp(V_out[:, 1])) * 100.0, 2),
        "SR_4in": round(float(np.ptp(r4[:, 0]) / np.ptp(r4[:, 1])), 2),
        "USCA_4in_pct": round(float(np.ptp(r4[:, 1]) / np.ptp(V_out[:, 0])) * 100.0, 2),
        "keyhole_inner_cm": [84.0, 42.0],
        "open_wing_knee_channel_cm": 31.2,
        "aft_coaming_lip_above_225wl_cm": round((float(surf.sculpted_deck_z(np.array([0.38]), np.array([0.0]))[0]) + 0.0185 - Z_WL_225) * 100.0, 1),
        "rear_deck_at_day_hatch_above_225wl_cm": round((float(surf.base_deck_z(np.array([0.68]), np.array([0.0]))[0]) - Z_WL_225) * 100.0, 1),
        "compartments_liters": vols,
        "total_sealed_buoyancy_liters": round(vols["bow_compartment_L"] + vols["day_pod_L"] + vols["stern_compartment_L"], 1),
        "baja_40lb_trim_balanced_split_lb": {"bow_hatch": 16.0, "day_pod": 4.0, "stern_hatch": 20.0, "cg_x_m": 0.025},
        "hydro_3load": ev3,
    }
    args.specs_out.parent.mkdir(parents=True, exist_ok=True)
    args.specs_out.write_text(json.dumps(specs, indent=2) + "\n")

    styled = [
        ("01_hull_shell", m_hull, [238, 242, 246]), ("02_deck_shell", m_deck, [88, 148, 192]),
        ("03_cockpit_coaming_and_thigh_hooks", m_coam, [38, 43, 50]), ("04_recessed_hatch_covers", m_hatch, [58, 66, 76]),
        ("05_cockpit_interior_and_bulkheads", m_int, [208, 196, 168]), ("06_seat_pan_and_footboard", m_seat, [50, 56, 66]),
        ("07_overstern_rudder", m_rud, [112, 122, 134]), ("08_day_pod_and_deck_ribs_keel_strap", m_pod_grid, [42, 48, 56]),
    ]
    render_parts = [(m_obj, (*(c / 255.0 for c in rgb), 1.0), 0.0) for name, m_obj, rgb in styled if name != "08_day_pod_and_deck_ribs_keel_strap"]
    cutaway_parts = [(m_obj, (0.07, 0.48, 0.38, 1.0) if name == "08_day_pod_and_deck_ribs_keel_strap" else (*(c / 255.0 for c in rgb), 0.24 if name == "02_deck_shell" else (0.50 if name == "04_recessed_hatch_covers" else 1.0)), +0.025 if name == "08_day_pod_and_deck_ribs_keel_strap" else (+0.12 if name in ("02_deck_shell", "03_cockpit_coaming_and_thigh_hooks", "04_recessed_hatch_covers") else 0.0)) for name, m_obj, rgb in styled]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180, facecolor="white")
    render_shaded_panel(axes[0, 0], render_parts, -138, 23, None, "1. Bow-Quarter 3/4 View — Raked Wave-Piercing Bow, Narrow Foredeck & Keyhole")
    render_shaded_panel(axes[0, 1], render_parts, -42, 24, None, "2. Stern-Quarter 3/4 View — 14.8 cm Low Rear Deck, Recessed Hatches & Over-Stern Rudder")
    render_shaded_panel(axes[1, 0], cutaway_parts, -130, 38, (-1.04, +0.50), "3. Cockpit X-Ray — -0.95 m Fwd BH, Wing Footboard, 60 mm UD Keel Strap & 10 mm Seat")
    render_shaded_panel(axes[1, 1], cutaway_parts, -54, 32, (+0.36, +2.86), "4. Aft-Deck X-Ray — 5.5 L Carbon Day Pod, Cruciform (+) UD Grid & Unified Stern")
    fig.tight_layout(pad=1.2)
    fig.savefig(args.plates_dir / "plate7_3d_views.png", dpi=180)
    plt.close(fig)
    render_engineering_layout(surf, {"outer_closed": outer_closed, "hull": m_hull, "deck": m_deck, "coaming": m_coam, "hatches": m_hatch, "seat_fb": m_seat, "interior": m_int, "pod_grid": m_pod_grid}, vols, specs, args.plates_dir / "plate2_general_arrangement.png")
    print("WROTE", args.specs_out, "AND PLATES 2 & 7")


if __name__ == "__main__":
    main()
