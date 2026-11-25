"""Generate Naval Architecture Plates 1, 3, 4, 8 and whole-hull fairing sheet for Kayak 10.9."""

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
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pyvista as pv
import trimesh

from deck_geometry import Z_KEEL_SHIFT, Z_WL_225, SculptedKayakSurface
from hydro import LB, RHO_SEAWATER, float_at, load_epic_18x_arrays

INK, BLUE, TEAL, ORANGE, SLATE, RED = "#132334", "#1d6fa5", "#0b8a6a", "#d97724", "#64748b", "#b83227"


def render_plate1_lines_plan(surf: SculptedKayakSurface, outer10: trimesh.Trimesh, epic: trimesh.Trimesh, specs: dict, out_dir: Path) -> None:
    fig = plt.figure(figsize=(16.5, 11.8), dpi=200, facecolor="white")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.05, 1.25], width_ratios=[1.0, 1.0, 1.15], hspace=0.35, wspace=0.25)
    ax_prof, ax_half, ax_body, ax_sac = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :2]), fig.add_subplot(gs[2, 2])
    xs = surf.xs
    seam = surf.end_shape_xyz(np.column_stack((xs, surf.seam_y, surf.seam_z)))

    first_cl = True
    for seg in trimesh.intersections.mesh_plane(outer10, [0, 1, 0], [0, 0, 0]):
        col_s = BLUE if seg[:, 2].mean() > 0.205 and (-2.70 < seg[:, 0].mean() < 2.71) else INK
        ax_prof.plot(seg[:, 0], seg[:, 2] * 100, color=col_s, lw=1.95, label="10.9 Rounded Stem, Forefoot & Deck Profile" if first_cl else None)
        first_cl = False
    ax_prof.plot(seam[:, 0], seam[:, 2] * 100, "--", color=BLUE, lw=1.0, label="10.9 Mold Seam")
    for yb, ls_b in ((0.06, "-."), (0.12, ":"), (0.18, "--")):
        for seg in trimesh.intersections.mesh_plane(outer10, [0, 1, 0], [0, yb, 0]):
            ax_prof.plot(seg[:, 0], seg[:, 2] * 100, ls_b, color="#7a8b99", lw=0.75)
    for zw, lbl, col in ((Z_WL_225, "225 lb WL (8.82 cm)", TEAL), (0.1016, "4.0\" SR/USCA Plane (10.16 cm)", ORANGE)):
        ax_prof.axhline(zw * 100, color=col, ls="-.", lw=1.1, label=lbl)
    ax_prof.set_xlim(-2.82, 2.82); ax_prof.set_ylim(-2, 37); ax_prof.set_ylabel("z (cm)", fontsize=8.5)
    ax_prof.set_title("PLATE 1 — Kayak 10.9 Naval Architecture Lines Plan (Buttocks @ 6/12/18 cm, Split Body Plan & Sectional Area Curve)", fontsize=11.2, fontweight="bold", color=INK)
    ax_prof.legend(loc="upper right", ncol=5, fontsize=7.6); ax_prof.grid(True, alpha=0.22)

    max_hb10 = np.max(np.abs(surf.hull_rings[:, :, 0]), axis=1) * 100
    xs_ep = np.linspace(-2.72, 2.72, 180)
    hb_ep = np.array([float(np.abs(trimesh.intersections.mesh_plane(epic, [1, 0, 0], [float(x), 0, 0]).reshape(-1, 3)[:, 1]).max()) * 100 if len(trimesh.intersections.mesh_plane(epic, [1, 0, 0], [float(x), 0, 0])) else 0.0 for x in xs_ep])
    ax_half.plot(xs, max_hb10, color=INK, lw=2.0, label=f"10.9 Max Half-Breadth ({specs['B_oa_cm']/2:.1f} cm / {specs['B_oa_cm']:.1f} cm OA)")
    ax_half.plot(xs, surf.seam_y * 100, "--", color=BLUE, lw=1.1, label="10.9 Deck Mold Seam")
    ax_half.plot(xs_ep, hb_ep, "--", color=SLATE, lw=1.5, label="Epic 18X Scan Max Half-Breadth (27.9 cm)")
    ax_half.axhline(25.6, color="#9333ea", ls=":", lw=1.3, label="Stellar S18R G2 Max Half-Beam Limit (25.6 cm / 51.2 cm OA)")
    for zw, lbl, col in ((0.1016, f"10.9 4.0\" SR Plane (SR {specs['SR_4in']:.2f})", ORANGE), (Z_WL_225, "10.9 225 lb WL (48.1 cm)", TEAL)):
        sl = trimesh.intersections.mesh_plane(outer10, [0, 0, 1], [0, 0, zw]).reshape(-1, 3)
        st = sl[sl[:, 1] >= 0]
        o = np.argsort(st[:, 0])
        ax_half.plot(st[o, 0], st[o, 1] * 100, color=col, lw=1.4, label=lbl)
    ax_half.axvspan(-1.25, -0.75, color="#fef3c7", alpha=0.55, label="Wing-paddle catch zone (+0.8..+1.3 m fwd of seat)")
    ax_half.set_xlim(-2.82, 2.82); ax_half.set_ylim(0, 31.5); ax_half.set_ylabel("Half-breadth y (cm)", fontsize=8.5)
    ax_half.set_xlabel("Station x from midship (m)   [Bow at -2.735 m, Stern at +2.735 m]", fontsize=8.5)
    ax_half.legend(loc="upper right", ncol=3, fontsize=7.3); ax_half.grid(True, alpha=0.22)

    for xv in np.linspace(-2.50, 0.0, 10):
        col_c = RED if abs(xv + 1.00) < 0.14 else BLUE
        for seg in trimesh.intersections.mesh_plane(outer10, [1, 0, 0], [float(xv), 0, 0]):
            if (seg[0, 1] >= -1e-4) and (seg[1, 1] >= -1e-4):
                ax_body.plot(seg[:, 1] * 100, seg[:, 2] * 100, color=col_c, lw=1.3 if col_c == BLUE else 2.0)
    for xv in np.linspace(0.0, +2.50, 10):
        for seg in trimesh.intersections.mesh_plane(outer10, [1, 0, 0], [float(xv), 0, 0]):
            if (seg[0, 1] <= 1e-4) and (seg[1, 1] <= 1e-4):
                ax_body.plot(seg[:, 1] * 100, seg[:, 2] * 100, color=INK, lw=1.25)
    for xv, side_sgn, lbl_e in ((-1.00, +1, "Epic 18X @ Catch (-1.0m)"), (0.00, -1, "Epic 18X @ Midship")):
        first = True
        for seg in trimesh.intersections.mesh_plane(epic, [1, 0, 0], [float(xv), 0, 0]):
            if side_sgn * seg[:, 1].mean() >= 0:
                ax_body.plot(seg[:, 1] * 100, seg[:, 2] * 100, "--", color=SLATE, lw=1.5, label=lbl_e if first else None)
                first = False
    ax_body.axvline(0, color="#111", lw=1.2); ax_body.axhline(Z_WL_225 * 100, color=TEAL, ls="-.", lw=1.1, label="225 lb WL"); ax_body.axhline(10.16, color=ORANGE, ls=":", lw=1.1, label="4.0\" SR Plane")
    ax_body.text(-28, 32.5, "AFTERBODY\n(x = 0 to +2.5 m)", fontsize=8.5, fontweight="bold", color=INK)
    ax_body.text(+12, 32.5, "FOREBODY (Red = Catch)\n(x = -2.5 to 0 m)", fontsize=8.5, fontweight="bold", color=BLUE)
    ax_body.set_xlim(-31, 31); ax_body.set_ylim(-1, 36.5); ax_body.set_aspect("equal")
    ax_body.set_xlabel("Half-breadth y (cm)   [Left: Afterbody | Right: Forebody]", fontsize=8.5); ax_body.set_ylabel("z above keel (cm)", fontsize=8.5)
    ax_body.legend(loc="lower left", fontsize=7.2); ax_body.grid(True, alpha=0.24)

    def sub_area_curve(tm_obj: trimesh.Trimesh, wl_z: float, x_arr: np.ndarray) -> np.ndarray:
        areas = np.zeros_like(x_arr)
        for i, xv in enumerate(x_arr):
            sl = trimesh.intersections.mesh_plane(tm_obj, [1, 0, 0], [float(xv), 0, 0]).reshape(-1, 3)
            sub = sl[(sl[:, 2] <= wl_z + 1e-3) & (sl[:, 1] >= 0)]
            if len(sub) >= 3:
                o = np.argsort(sub[:, 2])
                areas[i] = 2.0 * float(np.trapezoid(np.append(sub[o, 1], sub[o[-1], 1]), np.append(sub[o, 2], wl_z))) * 1e4
        return areas

    xs_sac = np.linspace(-2.70, 2.70, 140)
    re = float_at(np.asarray(epic.vertices), np.asarray(epic.faces), 225 * LB, RHO_SEAWATER, np.array([0.0, 0.0, 0.25]), free_trim=False)
    a10, aep = sub_area_curve(outer10, Z_WL_225, xs_sac), sub_area_curve(epic, float(-re["zmin"]), xs_sac)
    cp10 = float(np.trapezoid(a10 / 1e4, xs_sac) / (a10.max() / 1e4 * 5.42))
    cpep = float(np.trapezoid(aep / 1e4, xs_sac) / (aep.max() / 1e4 * 5.44))
    ax_sac.fill_between(xs_sac, 0, a10, color=BLUE, alpha=0.22)
    ax_sac.plot(xs_sac, a10, color=BLUE, lw=2.0, label=f"10.9 (Cp={cp10:.3f}, max={a10.max():.0f} cm²)")
    ax_sac.plot(xs_sac, aep, "--", color=SLATE, lw=1.7, label=f"Epic 18X (Cp={cpep:.3f}, max={aep.max():.0f} cm²)")
    ax_sac.set_title("Submerged Sectional Area Curve (225 lb)", fontsize=9.8, fontweight="bold")
    ax_sac.set_xlabel("Station x (m)", fontsize=8.5); ax_sac.set_ylabel("Submerged area A(x) (cm²)", fontsize=8.5)
    ax_sac.legend(fontsize=7.6, loc="upper right"); ax_sac.grid(True, alpha=0.25)
    fig.savefig(out_dir / "plate1_lines_plan.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def render_plate3_hydro_and_gz(outer10: trimesh.Trimesh, epic: trimesh.Trimesh, out_dir: Path) -> None:
    fig, (ax_c, ax_gz) = plt.subplots(1, 2, figsize=(16.2, 6.8), dpi=200, facecolor="white")
    V10, F10 = np.asarray(outer10.vertices), np.asarray(outer10.faces)
    Ve, Fe = np.asarray(epic.vertices), np.asarray(epic.faces)
    masses_lb = np.linspace(115, 340, 22)
    T_cm, disp_lb, awp_m2, swet_m2, bwl_cm, kmt_cm = [], [], [], [], [], []
    for mlb in masses_lb:
        r = float_at(V10, F10, mlb * LB, RHO_SEAWATER, np.array([0.025, 0.0, 0.25]), free_trim=False)
        t_val = float(-r["zmin"])
        T_cm.append(t_val * 100); disp_lb.append(mlb); awp_m2.append(float(r["awp"])); swet_m2.append(float(r["wet"]))
        sl_wl = trimesh.intersections.mesh_plane(outer10, [0, 0, 1], [0, 0, t_val]).reshape(-1, 3)
        bwl_cm.append(float(np.ptp(sl_wl[:, 1])) * 100); kmt_cm.append(float(r["B"][2] + t_val + r["IT"] / r["vol"]) * 100)
    ax_c.plot(T_cm, np.array(disp_lb) / 10.0, color=INK, lw=2.2, label="Displacement / 10 (10× lb)")
    ax_c.plot(T_cm, np.array(bwl_cm), color=BLUE, lw=2.0, label="Waterline Beam B_wl (cm)")
    ax_c.plot(T_cm, np.array(swet_m2) * 10.0, color=TEAL, lw=1.9, label="Wetted Surface × 10 (0.1 m²)")
    ax_c.plot(T_cm, np.array(awp_m2) * 10.0, "--", color="#6d28d9", lw=1.7, label="Waterplane Area × 10 (0.1 m²)")
    ax_c.plot(T_cm, np.array(kmt_cm), "-.", color=RED, lw=1.7, label="Transverse KM_T above keel (cm)")
    for idx_m, (t_mark, lbl_m, col_m) in enumerate([(8.53, "213 lb\nUltralight", TEAL), (8.86, "225 lb\nDay", BLUE), (9.85, "265 lb\nBaja", "#1e3a8a"), (10.16, "4.0\"\nSR Rule", ORANGE), (10.69, "300 lb\nHeavy", RED)]):
        ax_c.axvline(t_mark, color=col_m, ls=":", lw=1.2, alpha=0.85)
        ax_c.text(t_mark, 54.8 if idx_m % 2 == 0 else 50.2, lbl_m, ha="center", va="top", fontsize=7.2, fontweight="bold", color=col_m, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax_c.set_xlim(5.8, 11.6); ax_c.set_ylim(10, 56); ax_c.set_xlabel("Calm-water draft T above keel baseline (cm)", fontsize=9.0)
    ax_c.set_ylabel("Scaled hydrostatic value (see legend units)", fontsize=9.0); ax_c.set_title("Curves of Form vs. Draft — Kayak 10.9 (Seawater 1025 kg/m³)", fontsize=10.5, fontweight="bold")
    ax_c.legend(loc="lower right", fontsize=7.8); ax_c.grid(True, alpha=0.25)

    angs = np.arange(0, 45.1, 2.5)
    for lbl, Vm, Fm, lcg_g, m_kg, kg_v, col, ls, lw in [
        ("10.9 Ultralight (213 lb, KG=24.2 cm)", V10, F10, +0.025, 213 * LB, 0.242, TEAL, "-", 2.3),
        ("10.9 Day Matched (225 lb, KG=25.0 cm)", V10, F10, +0.025, 225 * LB, 0.250, BLUE, "-", 2.1),
        ("10.9 Baja +40 lb (265 lb, KG=25.0 cm)", V10, F10, +0.025, 265 * LB, 0.250, "#1e3a8a", "-", 1.9),
        ("10.9 Heavy +75 lb (300 lb, KG=25.0 cm)", V10, F10, +0.025, 300 * LB, 0.250, ORANGE, "-", 1.8),
        ("Epic 18X Scan (225 lb, KG=25.0 cm)", Ve, Fe, +0.15, 225 * LB, 0.250, SLATE, "--", 1.8),
        ("Epic 18X Scan (265 lb, KG=25.0 cm)", Ve, Fe, +0.15, 265 * LB, 0.250, "#94a3b8", ":", 1.7),
    ]:
        lcg_true = float(float_at(Vm, Fm, m_kg, RHO_SEAWATER, np.array([lcg_g, 0.0, kg_v]), free_trim=False)["B"][0])
        gz_mm = [0.0] + [float(-float_at(Vm, Fm, m_kg, RHO_SEAWATER, np.array([lcg_true, 0.0, kg_v]), heel=np.radians(a))["GZ"] * 1000.0) for a in angs[1:]]
        ax_gz.plot(angs, gz_mm, ls=ls, color=col, lw=lw, label=lbl)
    ax_gz.axhline(0, color="#111", lw=1.1); ax_gz.set_xlim(0, 45); ax_gz.set_ylim(-12, 16.5)
    ax_gz.set_xlabel("Heel angle φ (deg)", fontsize=9.0); ax_gz.set_ylabel("Righting arm GZ (mm)   [Unvalidated Hydrostatic Comparison]", fontsize=8.8)
    ax_gz.set_title("High-Angle Righting-Arm (GZ) Envelope: 10.9 vs. Epic 18X Scan", fontsize=10.5, fontweight="bold")
    ax_gz.legend(loc="lower left", fontsize=7.6); ax_gz.grid(True, alpha=0.25)
    fig.suptitle("PLATE 3 — Hydrostatics Curve Sheet & Multi-Load Stability Envelope", fontsize=11.8, fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(out_dir / "plate3_hydrostatics_and_gz.png", bbox_inches="tight", dpi=200); plt.close(fig)


def render_plate4_ergo_and_3boat(outer10: trimesh.Trimesh, epic: trimesh.Trimesh, specs: dict, out_dir: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.8), dpi=200, facecolor="white")
    offs = np.linspace(-0.40, 1.30, 35)

    def beam_z10(tm_obj: trimesh.Trimesh, lcg: float, twl: float) -> np.ndarray:
        vals = []
        for off in offs:
            sl = trimesh.intersections.mesh_plane(tm_obj, [1, 0, 0], [float(lcg - off), 0, 0]).reshape(-1, 3)
            u = sl[sl[:, 2] >= twl + 0.10, 1]
            vals.append(float(u.max() - u.min()) * 100 if len(u) >= 2 else 0.0)
        return np.array(vals)

    c10 = float(specs["hydro_3load"]["catch_beams_225_cm"]["+1.0"])
    ax1.plot(offs, beam_z10(outer10, +0.025, Z_WL_225), color=BLUE, lw=2.3, label=f"10.9 ({c10:.1f} cm @ +1.0m -> 55.2 cm aft)")
    ax1.plot(offs, beam_z10(epic, +0.150, 0.0934), "--", color=SLATE, lw=2.0, label="Epic 18X Scan (42.4 cm @ +1.0m)")
    ax1.axhline(51.2, color="#9333ea", ls=":", lw=1.6, label="Stellar S18R G2 Max Beam (51.2 cm OA)")
    ax1.axvspan(0.80, 1.25, color="#fef3c7", alpha=0.6, label="Wing-paddle catch entry (+0.8..+1.25 m)")
    ax1.set_xlabel("Distance forward of LCG / seat (m)", fontsize=8.8); ax1.set_ylabel("Beam above z_WL + 10 cm (cm)", fontsize=8.8)
    ax1.set_title("1. Catch Clearance & Smooth Aft Flare", fontsize=10.2, fontweight="bold"); ax1.legend(fontsize=7.4, loc="lower left"); ax1.grid(True, alpha=0.25)

    for seg in trimesh.intersections.mesh_plane(epic, [1, 0, 0], [-0.850, 0, 0]):
        ax2.plot(seg[:, 1] * 100, seg[:, 2] * 100, "--", color=SLATE, lw=1.6)
    for seg in trimesh.intersections.mesh_plane(outer10, [1, 0, 0], [-0.975, 0, 0]):
        ax2.plot(seg[:, 1] * 100, seg[:, 2] * 100, color=BLUE, lw=2.2)
    ax2.plot([8.0, 23.2], [62.0, 3.0], color=TEAL, lw=2.8, label="10.9 Wing Shaft (75° steep catch)")
    ax2.plot([8.0, 25.5], [62.0, 3.0], ":", color=RED, lw=2.0, label="Epic 18X Shaft (-2.3 cm wider entry)")
    ax2.axhline(Z_WL_225 * 100, color=TEAL, ls="-.", lw=1.0)
    ax2.fill_between([-14, 14], 8.8, 30.0, color="#dbeafe", alpha=0.28, label="30.0 cm × 26.2 cm Footbox")
    ax2.set_xlim(-28, 30); ax2.set_ylim(-2, 65); ax2.set_aspect("equal")
    ax2.set_xlabel("y from centerline (cm)", fontsize=8.8); ax2.set_ylabel("z above keel (cm)", fontsize=8.8)
    ax2.set_title("2. Wing Catch Geometry @ +1.0 m Fwd", fontsize=10.2, fontweight="bold"); ax2.legend(fontsize=7.3, loc="upper left"); ax2.grid(True, alpha=0.25)

    metrics = ["Lightest Weight\n(lb, lower=better)", "SR 4\" Ratio × 3\n(higher=faster)", "Dry Storage / 6\n(×6 L, higher=more)", "Overall Beam\n(cm, secondary reserve)"]
    y_pos, bh = np.arange(len(metrics)), 0.25
    sr_v, tot_l, boa_v = float(specs["SR_4in"]), float(specs["total_sealed_buoyancy_liters"]), float(specs["B_oa_cm"])
    b_a = ax3.barh(y_pos + bh, [27.0, sr_v * 3, tot_l / 6, boa_v], height=bh, color=BLUE, label="Kayak 10.9 (TeXtreme/Rohacell)")
    b_b = ax3.barh(y_pos, [34.2, 10.74 * 3, 206.0 / 6, 51.20], height=bh, color="#9333ea", alpha=0.85, label="Stellar S18R G2 Alpha ($7,275)")
    b_c = ax3.barh(y_pos - bh, [39.0, 10.58 * 3, 206.0 / 6, 55.88], height=bh, color=SLATE, alpha=0.85, label="Epic 18X Sport Ultra ($5,095)")
    ax3.set_yticks(y_pos, metrics, fontsize=8.3)
    ax3.bar_label(b_a, labels=["27.0 lb", f"SR {sr_v:.2f}", f"{tot_l:.0f} L (2 BH + pod)", f"{boa_v:.1f} cm"], fontsize=7.6, padding=3, fontweight="bold")
    ax3.bar_label(b_b, labels=["34.2 lb", "SR 10.74", "206 L (no day)", "51.2 cm"], fontsize=7.5, padding=3)
    ax3.bar_label(b_c, labels=["39.0 lb", "SR 10.58*", "206 L (3 hatches)", "55.9 cm"], fontsize=7.5, padding=3)
    ax3.set_xlim(0, 86); ax3.set_xlabel("Comparative scale (*Epic 10.58 measured on 3D scan; 10.16 SR list)", fontsize=8.0)
    ax3.set_title("3. Benchmark Spec Comparison: 10.9 vs. S18R vs. 18X", fontsize=10.2, fontweight="bold"); ax3.legend(fontsize=7.3, loc="lower right"); ax3.grid(axis="x", alpha=0.25)
    fig.suptitle("PLATE 4 — Wing-Paddle Ergonomics, Catch-Blade Clearance & 3-Boat Benchmark Comparison", fontsize=11.8, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "plate4_ergonomics_and_3boat.png", bbox_inches="tight", dpi=200); plt.close(fig)


def render_plate8_construction_and_tooling(out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.2), dpi=200, facecolor="white")
    ax_lam, ax_bom, ax_mold, ax_jig = axes.ravel()
    for y0, h, col, txt in [
        (0.0, 0.7, "#0ea5e9", "1. Thin 2K Clear/Tinted UV Polyurethane (135–165 g/m², NO heavy gelcoat -> saves 6.5 lb)"),
        (0.9, 1.3, "#1e293b", "2. Outer Skin: 160–200 gsm TeXtreme Spread-Tow Carbon/Innegra (impact + stiffness)"),
        (2.4, 3.4, "#fde68a", "3. Closed-Cell PMI Core: 3–4 mm Rohacell IG-F 51 (52 kg/m³, zero water absorption)"),
        (6.0, 1.1, "#334155", "4. Inner Skin: 100–130 gsm TeXtreme Spread-Tow Carbon (+160 gsm heel/seat patch)"),
    ]:
        ax_lam.add_patch(Rectangle((0.5, -y0 - h), 9.0, h, facecolor=col, edgecolor="#111", lw=1.0))
        ax_lam.text(5.0, -y0 - 0.5 * h, txt, ha="center", va="center", fontsize=8.0, color="white" if col in ("#1e293b", "#334155") else "#111", fontweight="bold")
    ax_lam.add_patch(Polygon([[2.3, -7.1], [2.7, -10.0], [7.3, -10.0], [7.7, -7.1]], closed=True, facecolor="#fef08a", edgecolor=TEAL, lw=2.2))
    ax_lam.add_patch(Rectangle((2.6, -10.35), 4.8, 0.35, facecolor=TEAL, edgecolor="#111"))
    ax_lam.text(5.0, -8.55, "Cruciform (+) Rear-Deck Grid:\n18×15 mm Rib + 20×20 mm Arch Beam\n(+64 g -> 1.79 mm Flex, 0 Rear Day BH!)", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#0f172a")
    ax_lam.text(5.0, -10.85, "Hull Keel: Flush 60 mm 200 gsm 0° UD-Carbon Internal Strap (Zero Proud Floor Rib, Clear Under 10 mm Seat)", ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
    ax_lam.set_xlim(0, 10); ax_lam.set_ylim(-11.4, 0.6); ax_lam.axis("off"); ax_lam.set_title("1. Aerospace Sandwich, Cruciform (+) Deck Grid & Flush UD Keel Strap", fontsize=9.8, fontweight="bold")

    cats = ["Hull Skins\n& Core", "Deck Skins,\nCore & Rib", "Keel, Seam\n& Patch", "Keyhole, BHs\n& Hatches", "Seat, Foot-\nboard & Rudder", "2K UV Finish\n& Deck Lines"]
    xi = np.arange(len(cats))
    b1 = ax_bom.bar(xi - 0.18, np.array([3.12, 1.16, 2.25, 3.17, 1.34, 1.26]) * 2.2046, width=0.35, color=BLUE, label="10.9 Prepreg Spec (Total: 27.0 lb / 12.2 kg)")
    b2 = ax_bom.bar(xi + 0.18, np.array([3.49, 1.41, 1.62, 3.39, 1.38, 2.05]) * 2.2046, width=0.35, color=TEAL, label="10.9 Vac-Bag One-Off (Total: 29.4 lb / 13.3 kg)")
    ax_bom.bar_label(b1, fmt="%.1f", fontsize=7.5, padding=2); ax_bom.bar_label(b2, fmt="%.1f", fontsize=7.5, padding=2)
    ax_bom.set_xticks(xi, cats, fontsize=8.0); ax_bom.set_ylabel("Subsystem mass (lb)", fontsize=8.8); ax_bom.set_ylim(0, 9.2)
    ax_bom.set_title("2. Subsystem Weight Breakdown (vs. S18R Alpha 34.2 lb / 18X Ultra 39.0 lb)", fontsize=10.0, fontweight="bold"); ax_bom.legend(fontsize=7.8, loc="upper right"); ax_bom.grid(axis="y", alpha=0.25)

    ax_mold.add_patch(Rectangle((-300, -300), 600, 600, fill=False, edgecolor=INK, lw=2.0, ls="--", label="600 × 600 mm Build Plate (848 mm Diagonal)"))
    ang = np.radians(45); R45 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    us = np.linspace(-1, 1, 80)
    outer_m = np.vstack(([[-322, 18], [-281, 18]], np.column_stack((us * 281.2, -135 + 135 * (us**2))), [[281, 18], [322, 18], [322, -22], [295, -22], [200, -162], [-200, -162], [-295, -22], [-322, -22]])) @ R45.T
    ax_mold.add_patch(Polygon(outer_m, closed=True, facecolor="#cbd5e1", edgecolor=INK, lw=1.6, label="ABS-GF / ASA-GF Female Mold Trough (644 mm flanged)"))
    for rx, ry in ((-245, -105), (+245, -105), (-305, -2), (+305, -2)):
        ax_mold.add_patch(plt.Circle(np.array([rx, ry]) @ R45.T, 13.0, facecolor=ORANGE, edgecolor="#111", lw=1.2))
    ax_mold.text(0, 255, "11 × 497.3 mm Vertical Z Sections = 5,470 mm Full Mold\n4× Ø25.4 mm (1\") Steel Alignment Tubes Lock Registration < 0.2 mm", ha="center", va="center", fontsize=8.0, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#94a3b8"))
    ax_mold.set_xlim(-345, 345); ax_mold.set_ylim(-345, 345); ax_mold.set_aspect("equal"); ax_mold.set_xlabel("Printer X (mm)", fontsize=8.3); ax_mold.set_ylabel("Printer Y (mm)", fontsize=8.3)
    ax_mold.set_title("3. Voron Phoenix 600³ One-Piece Diagonal Female Mold Layout", fontsize=10.0, fontweight="bold"); ax_mold.legend(loc="lower left", fontsize=7.4); ax_mold.grid(True, alpha=0.22)

    uh = np.linspace(-22, 22, 90)
    ax_jig.plot(uh, 0.042 * (uh**2), color=INK, lw=2.6, label="Inner Carbon Hull Floor")
    for sgn in (-1, 1):
        ax_jig.add_patch(Rectangle((sgn * 12.5 - 1.1, 6.6), 2.2, 2.2, facecolor="#1e293b", edgecolor="#111", label="12.6\" Carbon Foil Track (@ ±12.5 cm)" if sgn == 1 else None))
        ax_jig.plot([sgn * 12.5, sgn * 12.5], [8.8, 15.0], color=RED, lw=2.2)
    ax_jig.add_patch(Polygon([[-14.0, 15.0], [14.0, 15.0], [14.0, 17.4], [1.6, 17.4], [1.2, 0.3], [-1.2, 0.3], [-1.6, 17.4], [-14.0, 17.4]], closed=True, facecolor="#fed7aa", edgecolor=ORANGE, lw=1.8, label="H2S Printed H-Bridge Jig + Keel Touch-Off Foot"))
    ax_jig.text(0, 22.2, "Rear: Delrin Slider + SS Shoulder Bushing (Anti-Uplift Glide)\nFront: 316/Ti Foil T-Nut + EPDM Shim (520+ lbf Caliper Lock)\nFootboard Sidewalls: Tapered Fairing Wedges Keep Tracks Parallel", ha="center", va="center", fontsize=7.8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1"))
    ax_jig.set_xlim(-25, 25); ax_jig.set_ylim(-2, 27.5); ax_jig.set_aspect("equal"); ax_jig.set_xlabel("y from centerline (cm)", fontsize=8.3); ax_jig.set_ylabel("z above keel (cm)", fontsize=8.3)
    ax_jig.set_title("4. H2S Track-Bonding Alignment Jig & Twin Foil-Track Mount", fontsize=10.0, fontweight="bold"); ax_jig.legend(loc="lower right", fontsize=7.3); ax_jig.grid(True, alpha=0.22)
    fig.suptitle("PLATE 8 — Composite Laminate Schedule, Subsystem Weight Budget & 3D-Printed Tooling Architecture", fontsize=11.8, fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(out_dir / "plate8_construction_and_tooling.png", bbox_inches="tight", dpi=200); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", type=Path, default=REPO_ROOT / "data" / "kayak-10-9-specs.json")
    ap.add_argument("--plates-dir", type=Path, default=REPO_ROOT / "build" / "plates")
    args = ap.parse_args()
    args.plates_dir.mkdir(parents=True, exist_ok=True)

    surf = SculptedKayakSurface(REPO_ROOT / "hull" / "kayak-10-9.hull.json")
    outer10 = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
    outer10.merge_vertices(merge_tex=True, merge_norm=True)
    Ve, Fe = load_epic_18x_arrays(bow_at_pos_x=False)
    epic = trimesh.Trimesh(Ve, Fe, process=False)
    specs = json.loads(args.specs.read_text())

    render_plate1_lines_plan(surf, outer10, epic, specs, args.plates_dir)
    render_plate3_hydro_and_gz(outer10, epic, args.plates_dir)
    render_plate4_ergo_and_3boat(outer10, epic, specs, args.plates_dir)
    render_plate8_construction_and_tooling(args.plates_dir)
    print("GENERATED PLATES 1, 3, 4, 8 CLEANLY ->", args.plates_dir)


if __name__ == "__main__":
    main()
