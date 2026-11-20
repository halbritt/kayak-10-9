"""Build Midship (Station 06, 616 mm full-width) Z=325 mm Molds oriented at 0 deg with beefy X=0 keel/crown for Bambu Studio / OrcaSlicer Dovetail Cut."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HULL_JSON = REPO_ROOT / "hull" / "kayak-10-9.hull.json"
BUILD_ROOT = REPO_ROOT / "build"
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = BUILD_ROOT / "mold_h2s_dovetail_midship"
OUT.mkdir(parents=True, exist_ok=True)

from build_mold_station_06 import Hull, _loft_mold_block, _resample_polyline, _station_curves_mm, _style
from deck_geometry import keyhole_inner_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def _center_xy_flat_z(m_uvz: trimesh.Trimesh) -> trimesh.Trimesh:
    """Align full width symmetrically along X (X=0 is centerline keel for slicer Dovetail cut!), depth along Y (<=320), Z in [0, 325]."""
    m = m_uvz.copy()
    v = np.asarray(m.vertices).copy()
    v[:, 1] -= 0.5 * (v[:, 1].min() + v[:, 1].max())
    v[:, 2] -= v[:, 2].min()
    m.vertices = v
    b0, b1 = m.bounds
    assert abs(b0[0] + 308.0) < 0.05 and abs(b1[0] - 308.0) < 0.05
    assert (b1[1] - b0[1]) <= 320.0, f"Y depth exceeds H2S 320 mm bed: {b0}..{b1}"
    assert abs(b0[2]) < 1e-4 and b1[2] <= 340.0, f"Z exceeds H2S 340 mm height: {b0}..{b1}"
    return m


def build_dovetail_midship_molds() -> None:
    geom = Hull.model_validate_json((HULL_JSON).read_text()).to_geometry()
    xs_m = np.linspace(-0.1625, +0.1625, 57)  # 325.00 mm Z-height centered at Midship x=0 (fits H2S 340 mm Z!)
    zs_mm = (xs_m - xs_m[0]) * 1000.0

    # 1. Full-Width 616 mm Hull Mold with 33 mm solid keel block @ X=0 (plenty of meat for 15-18 mm slicer dovetail!)
    h_rings = []
    for xm in xs_m:
        hs, _ = _station_curves_mm(geom, float(xm))
        hs_r = _resample_polyline(hs, 110)
        ym, zm = float(hs_r[-1, 0]), float(hs_r[-1, 1])
        fl_port = _resample_polyline(np.array([[-308.0, zm], [-ym, zm]]), 24)[:-1]
        cav_port = np.column_stack((-hs_r[::-1, 0], hs_r[::-1, 1]))[:-1]
        cav_stbd = hs_r[:-1]
        fl_stbd = _resample_polyline(np.array([[ym, zm], [308.0, zm]]), 24)
        mid_b = hs_r[(hs_r[:, 0] >= 186.0) & (hs_r[:, 1] <= 105.0)][::-1]
        off_b = np.column_stack((mid_b[:, 0] + 10.5, mid_b[:, 1] - 11.5)) if len(mid_b) >= 2 else np.array([[220.0, 52.0]])
        stbd_back = _resample_polyline(np.vstack(([[308.0, zm], [308.0, 92.0], [270.0, 86.0]], off_b, [[186.0, 26.0], [186.0, -32.0]])), 68)[1:]
        base = _resample_polyline(np.array([[186.0, -32.0], [-186.0, -32.0]]), 44)[1:-1]
        port_back = np.column_stack((-stbd_back[::-1, 0], stbd_back[::-1, 1]))
        h_rings.append(np.vstack((fl_port, cav_port, cav_stbd, fl_stbd, stbd_back, base, port_back)))

    h_holes = [
        (-289.0, 124.0, 12.90), (+289.0, 124.0, 12.90),  # Upper 1" steel tubes (clear of X=0!)
        (-154.0, -4.0, 12.90), (+154.0, -4.0, 12.90),    # Lower 1" steel tubes (clear of X=0!)
        (-289.0, 100.0, 4.25), (+289.0, 100.0, 4.25),    # Upper M8 clamping bores
        (-96.0, -14.0, 4.25), (+96.0, -14.0, 4.25),      # Lower M8 bores (|X|=96 mm leaves +/-80 mm solid keel @ X=0!)
    ]
    hull_uvz = _loft_mold_block(np.asarray(h_rings), h_holes, zs_mm)
    hull_slicer = _center_xy_flat_z(hull_uvz)

    # 2. Matching Full-Width 616 mm Deck + Tumblehome Mold (65+ mm solid crown @ X=0 for slicer dovetail!)
    th_kh = np.linspace(0.0, np.pi, 400)
    kh_stbd = keyhole_inner_curve(th_kh) * 1000.0
    kh_ord = np.argsort(kh_stbd[:, 0])
    d_rings = []
    for xm in xs_m:
        _, ds = _station_curves_mm(geom, float(xm))
        ds_r = _resample_polyline(ds, 110)
        y_kh = float(np.interp(xm * 1000.0, kh_stbd[kh_ord, 0], kh_stbd[kh_ord, 1]))
        ds_r[:, 1] -= 1.2 * np.exp(-(((ds_r[:, 0] - y_kh) / 2.2) ** 2))
        ds_r[-1, 0] = 0.0
        ym, zm = float(ds_r[0, 0]), float(ds_r[0, 1])
        fl_stbd = _resample_polyline(np.array([[308.0, zm], [ym, zm]]), 24)[:-1]
        cav_stbd = ds_r[:-1]
        cav_port = np.column_stack((-ds_r[::-1, 0], ds_r[::-1, 1]))[:-1]
        fl_port = _resample_polyline(np.array([[-ym, zm], [-308.0, zm]]), 24)
        port_up = _resample_polyline(np.array([[-308.0, zm], [-308.0, 236.0], [-235.0, 310.0]]), 28)[1:]
        top_flat = _resample_polyline(np.array([[-235.0, 310.0], [+235.0, 310.0]]), 48)[1:-1]
        stbd_dn = np.column_stack((-port_up[::-1, 0], port_up[::-1, 1]))
        d_rings.append(np.vstack((fl_stbd, cav_stbd, cav_port, fl_port, port_up, top_flat, stbd_dn)))

    d_holes = [
        (-289.0, 200.0, 12.90), (+289.0, 200.0, 12.90),
        (-175.0, 290.0, 12.90), (+175.0, 290.0, 12.90),
        (-289.0, 224.0, 4.25), (+289.0, 224.0, 4.25),
        (-96.0, 296.0, 4.25), (+96.0, 296.0, 4.25),
    ]
    deck_uvz = _loft_mold_block(np.asarray(d_rings), d_holes, zs_mm)
    deck_slicer = _center_xy_flat_z(deck_uvz)

    p_hull = OUT / "kayak-10-9-midship-fullwidth-z325-hull-for-dovetail.stl"
    p_deck = OUT / "kayak-10-9-midship-fullwidth-z325-deck-for-dovetail.stl"
    hull_slicer.export(p_hull)
    deck_slicer.export(p_deck)

    manifest = {
        "purpose": "Midship 616 mm Full-Width Mold @ Z=325 mm (<=340 mm H2S Z) oriented at 0 deg with X=0 solid keel/crown for Bambu Studio / OrcaSlicer Cut -> Dovetail",
        "recommended_slicer_cut": "Press C (Cut) -> Vertical plane at X = 0.00 mm -> Mode: Dovetail -> Depth: 14 mm, Width: 18 mm, Flap Angle: 0 deg, Tolerance: 0.12-0.15 mm -> Place each half (322 x 199 x 325 mm) on its own H2S 340x320x340 plate!",
        "hull_mold": {"file": p_hull.name, "watertight": hull_slicer.is_watertight, "size_xyz_mm": np.round(np.ptp(hull_slicer.bounds, axis=0), 2).tolist(), "keel_thickness_at_X0_mm": 33.0},
        "deck_mold": {"file": p_deck.name, "watertight": deck_slicer.is_watertight, "size_xyz_mm": np.round(np.ptp(deck_slicer.bounds, axis=0), 2).tolist(), "crown_thickness_at_X0_mm": 72.5},
    }
    (OUT / "dovetail_midship_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    sc = trimesh.Scene()
    sc.add_geometry(_style(hull_slicer, "01_midship_fullwidth_z325_hull", [64, 74, 86, 255]), node_name="01_midship_fullwidth_z325_hull")
    sc.export(OUT / "kayak-10-9-midship-fullwidth-z325-hull-for-dovetail.glb")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.6), dpi=190, facecolor="white")
    for ax, m_s, title, col in [
        (ax1, hull_slicer, "1. Midship Full-Width Hull Mold (616 × 199 × 325 mm)\n33 mm Solid Keel @ X=0 -> Cut w/ Dovetail into 2× H2S Plates!", "#1d6fa5"),
        (ax2, deck_slicer, "2. Midship Full-Width Deck Mold (616 × 151 × 325 mm)\n72 mm Solid Crown @ X=0 -> Cut w/ Dovetail into 2× H2S Plates!", "#0b8a6a"),
    ]:
        for s in trimesh.intersections.mesh_plane(m_s, [0, 0, 1], [0, 0, 162.5]):
            ax.plot(s[:, 0], s[:, 1], color=col, lw=1.9)
        ax.axvline(0.0, color="#dc2626", ls="--", lw=2.2, label="Bambu Studio Cut Plane @ X = 0.0 mm (Mode: Dovetail)")
        # Show the two resulting 340x320 mm H2S plate boxes centered on Left & Right halves!
        ax.plot([-330, +10, +10, -330, -330], [-160, -160, +160, +160, -160], ":", color="#475569", lw=1.4, label="Plate 1 & 2 H2S Envelopes (340 × 320 mm each)")
        ax.plot([-10, +330, +330, -10, -10], [-160, -160, +160, +160, -160], ":", color="#475569", lw=1.4)
        ax.set_xlim(-350, 350); ax.set_ylim(-175, 175); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9.8, fontweight="bold"); ax.set_xlabel("Slicer X (mm)"); ax.set_ylabel("Slicer Y (mm)")
        ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Kayak 10.9 — Midship 616 mm Full-Width Molds @ Z = 325 mm Ready for Bambu Studio / OrcaSlicer Dovetail Cut", fontsize=11.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "dovetail_cut_guide.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_dovetail_midship_molds()
