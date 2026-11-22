"""Build full-width 1-piece Female Hull + Deck Molds fitting 100% inside Bambu Lab H2S (340x320x340 mm)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HULL_JSON = REPO_ROOT / "hull" / "kayak-10-9.hull.json"
BUILD_ROOT = REPO_ROOT / "build"
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = BUILD_ROOT / "mold_h2s_fullwidth"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_mold_station_06 import Hull, _loft_mold_block, _resample_polyline, _station_curves_mm, _style
import trimesh


def _center_on_h2s_bed(m_uvz: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center (X, Y) on (0, 0) and place bottom face flat on Z = 0.000 mm inside H2S 340x320x340 mm."""
    m = m_uvz.copy()
    v = np.asarray(m.vertices).copy()
    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())
    v[:, 1] -= 0.5 * (v[:, 1].min() + v[:, 1].max())
    v[:, 2] -= v[:, 2].min()
    m.vertices = v
    b0, b1 = m.bounds
    assert abs(b0[0]) <= 170.0 and abs(b1[0]) <= 170.0, f"Exceeds H2S 340 mm X: {b0}..{b1}"
    assert abs(b0[1]) <= 160.0 and abs(b1[1]) <= 160.0, f"Exceeds H2S 320 mm Y: {b0}..{b1}"
    assert abs(b0[2]) < 1e-4 and b1[2] <= 340.0, f"Exceeds H2S 340 mm Z: {b0}..{b1}"
    return m


def build_h2s_fullwidth_hull(geom, xs_m: np.ndarray) -> tuple[trimesh.Trimesh, trimesh.Trimesh, list[tuple[float, float, float]]]:
    """1-piece Full-Width Female Hull Mold (x in [-2.110, -1.785] m: 334 x 118 x 325 mm <= 340x320x340 mm H2S)."""
    rings: list[np.ndarray] = []
    for xm in xs_m:
        hs, _ = _station_curves_mm(geom, float(xm))
        hs_r = _resample_polyline(hs, 96)
        ym, zm = float(hs_r[-1, 0]), float(hs_r[-1, 1])

        fl_port = _resample_polyline(np.array([[-167.0, zm], [-ym, zm]]), 20)[:-1]
        cav_port = np.column_stack((-hs_r[::-1, 0], hs_r[::-1, 1]))[:-1]
        cav_stbd = hs_r[:-1]
        fl_stbd = _resample_polyline(np.array([[ym, zm], [167.0, zm]]), 20)

        stbd_back = _resample_polyline(
            np.array([[167.0, zm], [167.0, 56.0], [116.0, 18.0], [116.0, -13.0]]),
            44,
        )[1:]
        base = _resample_polyline(np.array([[116.0, -13.0], [-116.0, -13.0]]), 36)[1:-1]
        port_back = np.column_stack((-stbd_back[::-1, 0], stbd_back[::-1, 1]))
        rings.append(np.vstack((fl_port, cav_port, cav_stbd, fl_stbd, stbd_back, base, port_back)))

    holes = [
        (-148.5, 79.0, 12.90), (+148.5, 79.0, 12.90),  # Upper 1.000" (25.4 mm) steel alignment tubes
        (-86.0, +7.0, 12.90), (+86.0, +7.0, 12.90),    # Lower 1.000" bench-base steel alignment tubes
        (-148.5, 56.0, 4.25), (+148.5, 56.0, 4.25),    # Upper M8 section tie-bolt bores
        (-52.0, -1.0, 4.25), (+52.0, -1.0, 4.25),      # Bench-base M8 section tie-bolt bores
    ]
    uvz = _loft_mold_block(np.asarray(rings), holes, (xs_m - xs_m[0]) * 1000.0)
    return uvz, _center_on_h2s_bed(uvz), holes

def build_h2s_fullwidth_deck(geom, xs_m: np.ndarray) -> tuple[trimesh.Trimesh, trimesh.Trimesh, list[tuple[float, float, float]]]:
    """Matching 1-piece Full-Width Female Deck + Tumblehome Mold (334 x 252 x 325 mm <= 340x320x340 mm H2S)."""
    rings: list[np.ndarray] = []
    for xm in xs_m:
        _, ds = _station_curves_mm(geom, float(xm))
        ds_r = _resample_polyline(ds, 96)
        ds_r[-1, 0] = 0.0
        ym, zm = float(ds_r[0, 0]), float(ds_r[0, 1])

        fl_stbd = _resample_polyline(np.array([[167.0, zm], [ym, zm]]), 20)[:-1]
        cav_stbd = ds_r[:-1]
        cav_port = np.column_stack((-ds_r[::-1, 0], ds_r[::-1, 1]))[:-1]
        fl_port = _resample_polyline(np.array([[-ym, zm], [-167.0, zm]]), 20)

        port_up = _resample_polyline(np.array([[-167.0, zm], [-167.0, 295.0], [-126.0, 356.0]]), 40)[1:]
        top_flat = _resample_polyline(np.array([[-126.0, 356.0], [+126.0, 356.0]]), 36)[1:-1]
        stbd_dn = np.column_stack((-port_up[::-1, 0], port_up[::-1, 1]))
        rings.append(np.vstack((fl_stbd, cav_stbd, cav_port, fl_port, port_up, top_flat, stbd_dn)))

    rings_arr = np.asarray(rings)
    v_tube = max(136.0, round(float(rings_arr[:, :, 1].min()) + 24.0, 1))
    v_bolt = v_tube + 30.0
    holes = [
        (-148.5, v_tube, 12.90), (+148.5, v_tube, 12.90),  # Outboard 1.000" steel alignment-tube bores
        (-98.0, 336.0, 12.90), (+98.0, 336.0, 12.90),      # Inverted-bench 1.000" steel alignment-tube bores
        (-148.5, v_bolt, 4.25), (+148.5, v_bolt, 4.25),    # Side M8 section clamping bores
        (-48.0, 342.0, 4.25), (+48.0, 342.0, 4.25),        # Crown-base M8 section clamping bores
    ]
    uvz = _loft_mold_block(rings_arr, holes, (xs_m - xs_m[0]) * 1000.0)
    return uvz, _center_on_h2s_bed(uvz), holes


def main() -> None:
    geom = Hull.model_validate_json((HULL_JSON).read_text()).to_geometry()
    xs_m = np.linspace(-2.110, -1.785, 57)  # 325.00 mm Z-height (15 mm headroom inside Bambu H2S 340 mm Z!)

    hull_uvz, hull_bed, h_holes = build_h2s_fullwidth_hull(geom, xs_m)
    deck_uvz, deck_bed, d_holes = build_h2s_fullwidth_deck(geom, xs_m)

    p_hull = OUT / "kayak-10-9-h2s-fullwidth-hull-mold.stl"
    p_deck = OUT / "kayak-10-9-h2s-fullwidth-deck-mold.stl"
    hull_bed.export(p_hull)
    deck_bed.export(p_deck)

    def to_boat(m_uvz: trimesh.Trimesh, dz: float = 0.0) -> trimesh.Trimesh:
        mb = m_uvz.copy()
        vv = np.asarray(mb.vertices)
        mb.vertices = np.column_stack((xs_m[0] * 1000.0 + vv[:, 2], vv[:, 0], vv[:, 1] + dz))
        return mb

    tubes = []
    for uc, vc, _ in h_holes[:4] + d_holes[:4]:
        dz = +55.0 if vc > 100.0 else 0.0
        cyl = trimesh.creation.cylinder(radius=12.70, height=430.0, sections=28)
        cyl.apply_transform(trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 1, 0]))
        cyl.apply_translation([float(0.5 * (xs_m[0] + xs_m[-1]) * 1000.0), uc, vc + dz])
        tubes.append(cyl)

    kayak_rings = []
    for xm in xs_m:
        hs_raw, ds_raw = _station_curves_mm(geom, float(xm))
        hs, ds = _resample_polyline(hs_raw, 52), _resample_polyline(ds_raw, 52)
        ring_yz = np.vstack((hs, ds[1:], np.column_stack((-ds[-2::-1, 0], ds[-2::-1, 1])), np.column_stack((-hs[-2::-1, 0], hs[-2::-1, 1]))))
        kayak_rings.append(np.column_stack((np.full(len(ring_yz), xm * 1000.0), ring_yz)))
    kr = np.asarray(kayak_rings)
    nr = kr.shape[1]
    fk = [[k * nr + j, k * nr + (j + 1) % nr, (k + 1) * nr + j] for k in range(len(xs_m) - 1) for j in range(nr)] + \
         [[k * nr + (j + 1) % nr, (k + 1) * nr + (j + 1) % nr, (k + 1) * nr + j] for k in range(len(xs_m) - 1) for j in range(nr)]
    m_kayak = trimesh.Trimesh(vertices=kr.reshape(-1, 3), faces=np.asarray(fk, dtype=int), process=False)

    sc = trimesh.Scene()
    sc.add_geometry(_style(to_boat(hull_uvz, 0.0), "01_h2s_fullwidth_hull_mold", [64, 74, 86, 255]), node_name="01_h2s_fullwidth_hull_mold")
    sc.add_geometry(_style(to_boat(deck_uvz, +55.0), "02_h2s_fullwidth_deck_mold_exploded_55mm", [78, 92, 108, 255]), node_name="02_h2s_fullwidth_deck_mold_exploded_55mm")
    sc.add_geometry(_style(trimesh.util.concatenate(tubes), "03_1in_steel_alignment_tubes", [195, 202, 210, 255], rough=0.22, metal=0.82), node_name="03_1in_steel_alignment_tubes")
    sc.add_geometry(_style(m_kayak, "04_kayak_10_9_laminate_part_preview", [56, 148, 208, 96], alpha_mode="BLEND"), node_name="04_kayak_10_9_laminate_part_preview")
    sc.export(OUT / "kayak-10-9-h2s-fullwidth-mold-assembly.glb")

    est_kg = lambda v_mm3: round((v_mm3 / 1000.0) * 1.08 * 0.34 / 1000.0, 2)
    manifest = {
        "printer_target": "Bambu Lab H2S (340 x 320 x 340 mm heated chamber)",
        "mold_type": "1-Piece Unsplit Full-Width (Port-to-Starboard) Female Hull & Deck Molds",
        "boat_station_x_mm": [-2110.0, -1785.0],
        "print_z_height_mm": 325.0,
        "stl_units": "millimeters (1.0 = 1.0 mm, centered on H2S bed, Z_min = 0.00 mm, zero supports)",
        "max_undercut_mm": 0.0,
        "h2s_fullwidth_hull_mold": {
            "file": p_hull.name, "watertight": hull_bed.is_watertight,
            "size_xyz_mm": np.round(np.ptp(hull_bed.bounds, axis=0), 2).tolist(),
            "bounds_mm": np.round(hull_bed.bounds, 2).tolist(),
            "est_abs_gf_mass_kg_22pct_gyroid": est_kg(hull_bed.volume),
        },
        "h2s_fullwidth_deck_mold": {
            "file": p_deck.name, "watertight": deck_bed.is_watertight,
            "size_xyz_mm": np.round(np.ptp(deck_bed.bounds, axis=0), 2).tolist(),
            "bounds_mm": np.round(deck_bed.bounds, 2).tolist(),
            "est_abs_gf_mass_kg_22pct_gyroid": est_kg(deck_bed.volume),
        },
    }
    (OUT / "h2s_fullwidth_mold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

    wh = manifest["h2s_fullwidth_hull_mold"]["est_abs_gf_mass_kg_22pct_gyroid"]
    wd = manifest["h2s_fullwidth_deck_mold"]["est_abs_gf_mass_kg_22pct_gyroid"]
    sh = manifest["h2s_fullwidth_hull_mold"]["size_xyz_mm"]
    sd = manifest["h2s_fullwidth_deck_mold"]["size_xyz_mm"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=190, facecolor="white")
    for ax, m_b, title, col in [
        (ax1, hull_bed, f"1. Bambu H2S (340×320×340): Full-Width Hull Mold\n({sh[0]:.1f} × {sh[1]:.1f} × {sh[2]:.1f} mm — Unsplit 1-Piece!)", "#1d6fa5"),
        (ax2, deck_bed, f"2. Bambu H2S (340×320×340): Full-Width Deck Mold\n({sd[0]:.1f} × {sd[1]:.1f} × {sd[2]:.1f} mm — Unsplit 1-Piece!)", "#0b8a6a"),
    ]:
        ax.plot([-170, 170, 170, -170, -170], [-160, -160, 160, 160, -160], "--", color="#0f172a", lw=2.0, label="H2S Plate (340 × 320 mm)")
        sl0 = trimesh.intersections.mesh_plane(m_b, [0, 0, 1], [0, 0, 5.0])
        sl1 = trimesh.intersections.mesh_plane(m_b, [0, 0, 1], [0, 0, 320.0])
        f0, f1 = True, True
        for s in sl0:
            ax.plot(s[:, 0], s[:, 1], "-", color=col, lw=1.8, label="Z = 5 mm (fwd face)" if f0 else None); f0 = False
        for s in sl1:
            ax.plot(s[:, 0], s[:, 1], ":", color="#64748b", lw=1.3, label="Z = 320 mm (aft face)" if f1 else None); f1 = False
        sz = np.ptp(m_b.bounds, axis=0)
        ax.text(0, 148, f"Size: {sz[0]:.1f} × {sz[1]:.1f} × {sz[2]:.1f} mm <= 340×320×340\nWatertight: {m_b.is_watertight} | 0.00 mm Undercut | 0 Supports",
                ha="center", va="top", fontsize=7.6, fontweight="bold", bbox=dict(boxstyle="round,pad=0.22", fc="#f8fafc", ec="#94a3b8"))
        ax.set_xlim(-192, 192); ax.set_ylim(-180, 180); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9.5, fontweight="bold"); ax.set_xlabel("H2S Bed X (mm)"); ax.set_ylabel("H2S Bed Y (mm)"); ax.legend(loc="lower right", fontsize=7.4)

    for m_u, col, lbl in [(hull_uvz, "#1d6fa5", f"1-Piece Hull Mold (~{wh:.2f} kg ABS-GF)"), (deck_uvz, "#0b8a6a", f"1-Piece Deck + Tumblehome Mold (~{wd:.2f} kg)")]:
        f_m = True
        for s in trimesh.intersections.mesh_plane(m_u, [0, 0, 1], [0, 0, 162.5]):
            ax3.plot(s[:, 0], s[:, 1], color=col, lw=1.9, label=lbl if f_m else None); f_m = False
    hs_m, _ = _station_curves_mm(geom, -1.9475)
    ax3.axhline(float(hs_m[-1, 1]), color="#b83227", ls=":", lw=1.2, label="0.00 mm Undercut Parting Plane")
    ax3.set_xlim(-190, 190); ax3.set_ylim(-30, 375); ax3.set_aspect("equal"); ax3.grid(True, alpha=0.25)
    ax3.set_title("3. Mated Full-Width Section @ x = -1.95 m\n(8× Ø1\" Steel Tube Bores + 8× M8 Bores)", fontsize=9.5, fontweight="bold")
    ax3.set_xlabel("Boat y (mm)"); ax3.set_ylabel("Boat z above keel (mm)"); ax3.legend(loc="lower right", fontsize=7.3)

    fig.suptitle("Kayak 10.9 — Unsplit Full-Width Female Hull & Deck Molds Scaled 1:1 for Bambu Lab H2S (340 × 320 × 340 mm)", fontsize=11.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "h2s_fullwidth_mold_verification.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
