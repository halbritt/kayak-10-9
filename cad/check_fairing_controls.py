"""Compare two finished metre-unit shells at equal payload, KG and water density."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import brentq, root
from scipy.spatial.transform import Rotation


def immersed(mesh, rotation, height):
    # The open waterplane cap contributes zero to tetrahedra with their
    # origin on that plane. The signed volumes give the buoyancy centroid.
    vertices = rotation.apply(mesh.vertices)
    vertices[:, 2] -= height
    v, f, _ = trimesh.intersections.slice_faces_plane(vertices, mesh.faces, [0, 0, -1], [0, 0, 0])
    a, b, c = v[f].transpose(1, 0, 2)
    six_vol = np.einsum("ij,ij->i", a, np.cross(b, c))
    total = six_vol.sum()
    centre = six_vol @ (a + b + c) / (4 * total)
    return total / 6, centre


def catch_widths(mesh, lcg, draft):
    widths = []
    for offset in [0, 0.8, 0.9, 1.0, 1.1, 1.2]:
        seg = trimesh.intersections.mesh_plane(mesh, [1, 0, 0], [lcg - offset, 0, 0])
        height = draft + 0.10
        # Include the exact intersections with the lower edge of the catch
        # envelope. Vertex-only measurements change with mesh resolution.
        a, b = seg[:, 0], seg[:, 1]
        cross = (a[:, 2] < height) != (b[:, 2] < height)
        a, b = a[cross], b[cross]
        hit = a[:, 1] + (b[:, 1] - a[:, 1]) * (height - a[:, 2]) / (b[:, 2] - a[:, 2])
        widths.append(float(np.ptp(np.r_[seg[:, :, 1][seg[:, :, 2] >= height], hit]) * 1000))
    return widths


def measure(path):
    mesh = trimesh.load_mesh(path)
    if not mesh.is_volume:
        raise ValueError(f"Expected a closed oriented shell: {path}")
    wl = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, 0.1016]).reshape(-1, 3)
    dims = np.ptp(wl, axis=0)[:2]
    result = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "waterline_mm": (dims * 1000).tolist(),
        "waterline_ratio": float(dims[0] / dims[1]),
        "loads": {},
    }
    upright = Rotation.identity()
    result["mass_at_4in_kg"] = 1025 * immersed(mesh, upright, 0.1016)[0]
    z0 = mesh.bounds[0, 2]
    for lb in (225, 265, 300):
        volume = lb * 0.45359237 / 1025
        height = brentq(lambda h: immersed(mesh, upright, h)[0] - volume, z0 + 0.01, z0 + 0.20)
        _, centre = immersed(mesh, upright, height)
        cg = np.array([centre[0], 0, z0 + 0.25])
        loads = {"draft_mm": (height - z0) * 1000, "gz_mm": []}
        for deg in (10, 20, 30):
            heel = Rotation.from_euler("x", deg, degrees=True)

            def balance(pose, return_arm=False):
                h, trim = pose
                rot = heel * Rotation.from_euler("y", trim)
                vol, buoyancy = immersed(mesh, rot, h)
                lever = rot.apply(cg) - buoyancy
                if return_arm:
                    return float(1000 * lever[1])
                return [vol / volume - 1, lever[0]]

            sol = root(balance, [height, 0])
            if not sol.success or np.max(np.abs(balance(sol.x))) > 1e-7:
                raise ValueError(f"Unbalanced {lb} lb at {deg} degrees: {sol.message}")
            loads["gz_mm"].append(balance(sol.x, True))
        result["loads"][str(lb)] = loads
        if lb == 225:
            result["catch_widths_mm"] = catch_widths(mesh, centre[0], height)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    a, b = measure(args.reference), measure(args.candidate)

    def delta(first, second):
        return float(np.max(np.abs(np.subtract(first, second))))

    changes = {
        "waterline_mm": delta(a["waterline_mm"], b["waterline_mm"]),
        "catch_width_mm": delta(a["catch_widths_mm"], b["catch_widths_mm"]),
    }
    for key in ["draft_mm", "gz_mm"]:
        changes[key] = max(delta(a["loads"][lb][key], b["loads"][lb][key]) for lb in a["loads"])
    limits = {"waterline_mm": 1.0, "catch_width_mm": 2.0, "draft_mm": 1.0, "gz_mm": 0.5}
    passed = all(changes[key] <= limit for key, limit in limits.items())
    print(
        json.dumps(
            dict(reference=a, candidate=b, maximum_changes=changes, limits=limits, passed=passed),
            indent=2,
            allow_nan=False,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
