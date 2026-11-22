"""Inspect full-size, metre-unit, closed outer shells in boat coordinates (+X aft)."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import trimesh

# Offsets in metres: z, y or y + z. Use the outer starboard / lower branch.
CUTS = {
    "Waterlines": ((0, 1), (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1016, 0.13, 0.16)),
    "Buttocks": ((1, 0), (0.025, 0.075, 0.125, 0.175, 0.225)),
    "Diagonals": ((1, 1), (0.075, 0.125, 0.175, 0.225, 0.275)),
}


def cut_line(mesh, x, normal, offset):
    n = np.asarray(normal)
    t = np.array([n[1], -n[0]]) / np.linalg.norm(n)
    segments = trimesh.intersections.mesh_plane(mesh, np.r_[0, n], np.r_[0, n * offset / (n @ n)])
    result = np.full(len(x), -np.inf)
    for seg in segments:
        a, b = seg[:, 0]
        i, j = np.searchsorted(x, sorted((a, b)))
        if i < j:
            c, d = seg[:, 1:] @ t
            result[i:j] = np.maximum(result[i:j], c + (x[i:j] - a) * (d - c) / (b - a))
    result[~np.isfinite(result)] = np.nan
    return result


def turns(y, spacing, span=0.05):
    h = round(span / spacing)
    angles = np.arctan2(y[h:] - y[:-h], h * spacing)
    result = np.full_like(y, np.nan)
    result[h:-h] = np.degrees(angles[h:] - angles[:-h])
    return result


def section_curve(mesh, x):
    ring = max(mesh.section([1, 0, 0], [x, 0, 0]).discrete, key=len)[:, 1:]
    # Follow the closed section at equal arc intervals, starting at the keel.
    points = []
    for a, b in zip(ring[:-1], ring[1:]):
        points.append(a)
        if a[0] * b[0] < 0:
            points.append(a + (b - a) * (-a[0] / (b[0] - a[0])))
    ring = np.asarray(points)
    centreline = np.flatnonzero(abs(ring[:, 0]) < 1e-12)
    start = centreline[np.argmin(ring[centreline, 1])]
    ring = np.roll(ring, -start, axis=0)
    if ring[np.argmax(abs(ring[:, 0]) > 0.0001), 0] < 0:
        ring = np.r_[ring[:1], ring[:0:-1]]
    ring = np.vstack((ring, ring[0]))
    s = np.r_[0, np.linalg.norm(np.diff(ring, axis=0), axis=1).cumsum()]
    arc = np.linspace(0, s[-1], max(60, round(s[-1] / 0.005)), endpoint=False)
    q = np.column_stack([np.interp(arc, s, c) for c in ring.T])
    # Signed curvature of the circle through three points about 15 mm apart.
    h = max(1, round(0.015 / arc[1]))
    a, b = q - np.roll(q, h, axis=0), np.roll(q, -h, axis=0) - q
    k = (
        2
        * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a + b, axis=1))
    )
    return arc, q, k


def reflections(ax, mesh):
    # Orthographic reflection of parallel light bands in triangle normals.
    view = np.array([-0.24, 0.82, -0.52])
    view /= np.linalg.norm(view)
    across = np.cross([0, 0, 1], view)
    across *= -1 / np.linalg.norm(across)
    up = np.cross(across, view)
    visible = (mesh.face_normals @ view) > 0
    triangles = mesh.triangles[visible]
    normals = mesh.face_normals[visible]
    reflection = 2 * (normals @ view)[:, None] * normals - view
    phase = np.arctan2(reflection[:, 2], reflection[:, 1])
    intensity = 0.5 + 0.4 * np.tanh(2 * np.sin(10 * phase))
    order = np.argsort(triangles.mean(axis=1) @ view)
    projected = triangles[order] @ np.stack([across, up], axis=1)
    ax.add_collection(
        PolyCollection(
            projected, facecolors=plt.cm.Greys(intensity[order]), edgecolors="face", linewidths=0.14
        )
    )
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")


def inspect(path, axes, column, entry_ax, label):
    mesh = trimesh.load_mesh(path)
    if not mesh.is_volume or not 5.4 < mesh.extents[0] < 5.6:
        raise ValueError(f"{path}: expected a closed 5.47 m outer shell in metres")
    bow, stern = mesh.bounds[:, 0]
    x = np.arange(bow + 0.001, stern, 0.002)
    report = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cuts": {}}
    for row, (name, (normal, offsets)) in enumerate(CUTS.items()):
        for offset in offsets:
            y = cut_line(mesh, x, normal, offset)
            angle = turns(y, 0.002)
            axes[2 * row, column].plot(x - bow, y * 1000, lw=0.8)
            axes[2 * row + 1, column].plot(x - bow, angle, lw=0.8)
            if name == "Waterlines" and offset == 0.07:
                entry_ax.plot(x - bow, angle, label=label)
            window = (x - bow >= 0.70) & (x - bow <= 1.02)
            interior = np.flatnonzero(np.isfinite(angle))
            # Report the highest change of turning angle over a 50 mm step.
            # The tight tips have separate plots; allow 100 mm at cut ends.
            inside = interior[(interior >= interior[0] + 50) & (interior <= interior[-1] - 50)]
            variation = np.abs(angle[inside + 25] - angle[inside])
            peak = np.nanargmax(variation)
            report["cuts"][f"{name} {offset:g}"] = {
                "peak_turn_change_deg": float(variation[peak]),
                "distance_aft_m": float(x[inside[peak]] - bow + 0.025),
                "rms_turn_change_deg": float(np.sqrt(np.nanmean(variation**2))),
                "max_signed_turn_forward_deg": float(np.nanmax(angle[window]))
                if np.isfinite(angle[window]).any()
                else None,
            }
    report["sections"] = []
    for frac in np.unique(
        np.r_[np.linspace(0.06, 0.94, 19), 0.635 / 5.47, 1.085 / 5.47, 1.385 / 5.47, 1.785 / 5.47]
    ):
        arc, q, k = section_curve(mesh, bow + frac * (stern - bow))
        half = arc < arc[-1] / 2
        axes[6, column].plot(q[half, 0] * 1000, q[half, 1] * 1000, lw=0.7)
        submerged = half & (q[:, 1] <= 0.1016)
        axes[7, column].plot(arc[submerged] * 1000, k[submerged], lw=0.7)
        # Sample the comb on both sides of the waterline. Use the same
        # physical gauge length for every mesh, irrespective of facet density.
        lower = half & (q[:, 1] < 0.120)
        dk = (np.roll(k, -2) - np.roll(k, 2)) / (4 * arc[1])
        report["sections"].append(
            {
                "fraction": float(frac),
                "keel_groove_mm": float((q[0, 1] - min(q[:, 1])) * 1000),
                "peak_abs_dk_ds_per_m2": float(np.max(np.abs(dk[lower]))) if lower.any() else None,
            }
        )
    reflections(axes[8, column], mesh)
    return report


def compare_surfaces(first, second, out):
    """Point-to-triangle signed distances in both directions, sampled at mesh vertices."""
    import pyvista as pv

    paths = [Path(arg.split("=", 1)[1]) for arg in (first, second)]
    labels = [arg.split("=", 1)[0] for arg in (first, second)]
    meshes = [pv.read(path) for path in paths]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), layout="constrained")
    result = []
    for i, ax in enumerate(axes):
        mesh = meshes[i]
        dist = 1000 * mesh.compute_implicit_distance(meshes[1 - i])["implicit_distance"]
        pts = mesh.points
        idx = np.argmax(abs(dist))
        result.append(
            {
                "from": labels[i],
                "to": labels[1 - i],
                "p95_absolute_mm": float(np.percentile(abs(dist), 95)),
                "maximum_absolute_mm": float(abs(dist[idx])),
                "maximum_at_xyz_m": pts[idx].tolist(),
            }
        )
        side = pts[:, 1] >= 0
        plot = ax.scatter(
            pts[side, 0] - pts[:, 0].min(),
            pts[side, 2] * 1000,
            c=dist[side],
            s=2,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        ax.set_ylabel("Height (mm)")
        ax.set_title(f"{labels[i]} relative to {labels[1 - i]}", loc="left")
    fig.colorbar(plot, ax=axes, label="Signed distance (mm)")
    axes[-1].set_xlabel("Distance aft of bow (m)")
    fig.savefig(out / "displacement.png", dpi=160)
    plt.close(fig)
    (out / "deviation.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shell", nargs="+", help="label=outer-shell.stl")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(
        9,
        len(args.shell),
        squeeze=False,
        figsize=(5 * len(args.shell), 17),
        sharey="row",
        layout="constrained",
    )
    entry, entry_ax = plt.subplots(figsize=(8, 4), layout="constrained")
    report = {}
    for col, arg in enumerate(args.shell):
        label, filename = arg.split("=", 1)
        report[label] = inspect(Path(filename), axes, col, entry_ax, label)
        axes[0, col].set_title(label)
    for row, label in enumerate(
        (
            "Waterline breadth (mm)",
            "Turn (deg)",
            "Buttock depth (mm)",
            "Turn (deg)",
            "Diagonal offset (mm)",
            "Turn (deg)",
            "Section height (mm)",
            "Section curvature (1/m)",
            "Reflection bands",
        )
    ):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.grid(alpha=0.18)
            if row < 6:
                ax.set_xlabel("Distance aft of bow (m)")
            elif row == 6:
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("Half-breadth (mm)")
            elif row == 7:
                ax.set_xlabel("Girth from keel (mm)")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "hull-lines.png", dpi=145)
    plt.close(fig)
    entry_ax.set(
        xlim=(0.65, 1.05),
        ylim=(-0.65, 0.65),
        xlabel="Distance aft of bow (m)",
        ylabel="Turn across adjoining 50 mm spans (deg)",
        title="Forward waterline at 70 mm",
    )
    entry_ax.axhline(0, lw=0.8, color="gray")
    entry_ax.legend()
    entry.savefig(args.out / "entry-waterline.png", dpi=160)
    plt.close(entry)
    if len(args.shell) >= 2:
        compare_surfaces(*args.shell[-2:], args.out)
    (args.out / "fairness.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
