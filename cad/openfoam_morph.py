"""Move an OpenFOAM mesh built on one hull onto another hull of the same surface topology.

Two hulls from the same parametric surface give STL files whose triangle ``i`` is the same patch of
skin, so the displacement of every point of the skin is known. This module carries that displacement
into the volume mesh as a function of position only::

    d(x) = D(closest point of the first hull's skin to x) * w(distance from x to that skin)

``w`` is 1 up to ``r0`` (the cells at the wall move rigidly with the wall, so wall spacing is kept)
and falls smoothly to 0 at ``radius``. Because ``d`` depends on position and not on mesh numbering,
the same call moves an undecomposed mesh and every processor mesh of a decomposed case consistently.
Points on the symmetry plane ``y = 0`` stay on it.

``D`` is not the raw parametric displacement. A parametric surface slides along itself when one end
of a section moves: on 2026-11-26 a keel 9.5 mm deeper at the bow slid the skin at the waterline
down by 6 mm where its shape had changed by 1.4 mm. Cells carry their water fraction with them, so
that slide dented the free surface around the bow and started a 0.5 N slosh in a warm-started run.
``skin_displacement`` therefore moves every smooth point along its normal onto the second hull, and
every point of a sharp edge (keel, stem, sheer) across the edge onto the second hull's edge.

Why: two hulls meshed separately differ in drag by the mesh as well as by the hull. One mesh moved
onto each hull keeps cell count, connectivity and wall spacing identical, so a drag difference
between them belongs to the hull.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

N_CANDIDATE_TRIANGLES = 48
_CHUNK = 8000
_STL_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def read_binary_stl(path: Path) -> np.ndarray:
    """Triangles of a binary STL as an ``(n, 3, 3)`` float64 array, in file order."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL")
    n = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL ({len(raw)} bytes for {n} triangles)")
    return np.frombuffer(raw, dtype=_STL_RECORD, count=n, offset=84)["vertices"].astype(float)


def read_points(path: Path) -> tuple[np.ndarray, bytes, bytes]:
    """Read an OpenFOAM ``points`` file (ASCII or binary).

    Returns the points, the header bytes before the point count, and the bytes after the list.
    """
    raw = Path(path).read_bytes()
    m = re.search(rb"^(\d+)\s*$", raw, re.M)
    if m is None:
        raise ValueError(f"no point count in {path}")
    n = int(m.group(1))
    header = raw[: m.start()]
    open_paren = raw.index(b"(", m.end())
    if re.search(rb"format\s+binary", header) is not None:
        end = open_paren + 1 + 24 * n
        if len(raw) < end + 1:
            raise ValueError(f"{path}: binary point list is truncated")
        pts = np.frombuffer(raw[open_paren + 1 : end], dtype="<f8").reshape(n, 3).copy()
        rest = raw[end:]
        trailer = rest[rest.index(b")") + 1 :]
    else:
        close = raw.rindex(b")")
        body = raw[open_paren + 1 : close].replace(b"(", b" ").replace(b")", b" ")
        pts = np.array(body.split(), dtype=float)
        if pts.size != 3 * n:
            raise ValueError(f"{path}: expected {3 * n} numbers, found {pts.size}")
        pts = pts.reshape(n, 3)
        trailer = raw[close + 1 :]
    return pts, header, trailer


def write_points(path: Path, pts: np.ndarray, header: bytes, trailer: bytes) -> None:
    """Write an OpenFOAM ``points`` file in ASCII at full double precision."""
    header = re.sub(rb"format\s+binary", b"format      ascii", header)
    body = "\n".join("(%.17g %.17g %.17g)" % (x, y, z) for x, y, z in np.asarray(pts, dtype=float))
    Path(path).write_bytes(header + f"{len(pts)}\n(\n".encode() + body.encode() + b"\n)" + trailer)


def closest_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of the point of triangle ``(a, b, c)`` closest to ``p``.

    All arguments broadcast over leading axes. Degenerate triangles give a vertex or edge point.
    (Ericson, Real-Time Collision Detection, 5.1.5.)
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    zero, one = np.zeros_like(d1), np.ones_like(d1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v_in, w_in = vb / denom, vc / denom
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    conds = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
    ]
    u = np.select(conds, [one, zero, 1 - v_ab, zero, 1 - w_ac, zero], default=1 - v_in - w_in)
    v = np.select(conds, [zero, one, v_ab, zero, zero, 1 - w_bc], default=v_in)
    w = np.select(conds, [zero, zero, zero, one, w_ac, w_bc], default=w_in)
    bary = np.stack([u, v, w], axis=-1)
    bary[~np.isfinite(bary).all(-1)] = (1.0, 0.0, 0.0)
    return bary


@dataclass(frozen=True)
class MorphSettings:
    """How far from the hull the mesh moves. Part of every cache key that depends on a moved mesh."""

    r0_m: float = 0.02
    radius_m: float = 0.2
    # A vertex whose faces turn by more than this from their mean direction is on a sharp edge.
    feature_angle_deg: float = 30.0
    # Next to a sharp edge the skin follows the edge's own move, fading out over this distance.
    feature_width_m: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 <= self.r0_m < self.radius_m:
            raise ValueError(f"need 0 <= r0 < radius, got r0={self.r0_m}, radius={self.radius_m}")
        if not 0.0 < self.feature_angle_deg < 90.0 or self.feature_width_m <= 0.0:
            raise ValueError("feature_angle_deg must be in (0, 90) and feature_width_m positive")


def _smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _closest_point_on_surface(tri: np.ndarray, tree: cKDTree, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point of a triangle soup to each of ``pts`` and its distance (nearest centroids only)."""
    out = np.empty_like(pts)
    dist = np.empty(len(pts))
    k = min(N_CANDIDATE_TRIANGLES, len(tri))
    for start in range(0, len(pts), _CHUNK):
        p = pts[start : start + _CHUNK]
        _, cand = tree.query(p, k=k)
        if k == 1:
            cand = cand[:, None]
        t = tri[cand]
        pk = np.broadcast_to(p[:, None, :], t.shape[:2] + (3,))
        bary = closest_on_triangles(pk, t[:, :, 0], t[:, :, 1], t[:, :, 2])
        cp = (bary[..., None] * t).sum(axis=2)
        d_all = np.linalg.norm(cp - pk, axis=-1)
        best = d_all.argmin(axis=1)
        rows = np.arange(len(p))
        out[start : start + _CHUNK] = cp[rows, best]
        dist[start : start + _CHUNK] = d_all[rows, best]
    return out, dist


def _closest_point_on_segments(a: np.ndarray, b: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Closest point of a set of segments ``a[i]..b[i]`` to each of ``pts`` (nearest midpoints only)."""
    k = min(12, len(a))
    _, cand = cKDTree(0.5 * (a + b)).query(pts, k=k)
    if k == 1:
        cand = cand[:, None]
    sa, sb = a[cand], b[cand]
    ab = sb - sa
    ap = pts[:, None, :] - sa
    denom = (ab * ab).sum(axis=-1)
    t = np.clip(np.divide((ap * ab).sum(axis=-1), denom, out=np.zeros_like(denom), where=denom > 0), 0.0, 1.0)
    cp = sa + t[..., None] * ab
    best = np.linalg.norm(cp - pts[:, None, :], axis=-1).argmin(axis=1)
    return cp[np.arange(len(pts)), best]


def skin_displacement(
    base_tri: np.ndarray, cand_tri: np.ndarray, settings: MorphSettings | None = None
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Where each corner of each triangle of the first hull goes on the second hull.

    Smooth points move along their normal and are then dropped onto the second hull's skin. Points
    on a sharp edge move across the edge (their move along the edge is removed) and are dropped
    onto the second hull's edge; where three or more faces meet, or an edge vertex has no edge
    neighbour, they keep the parametric move. Within ``feature_width_m`` of a sharp edge the
    skin also takes the edge's move across the surface, fading out, so the skin next to an edge is
    stretched over a few cells and not torn.

    Returns the ``(n, 3, 3)`` displacements and a report.
    """
    settings = settings or MorphSettings()
    n_tri = len(base_tri)
    verts, inv = np.unique(base_tri.reshape(-1, 3), axis=0, return_inverse=True)
    inv = inv.reshape(n_tri, 3)
    flat = inv.ravel()
    count = np.bincount(flat, minlength=len(verts)).astype(float)
    cand_verts = np.zeros_like(verts)
    np.add.at(cand_verts, flat, cand_tri.reshape(-1, 3))
    cand_verts /= count[:, None]
    if np.abs(cand_tri.reshape(-1, 3) - cand_verts[flat]).max() > 1e-6:
        raise ValueError("the two hulls do not have the same surface topology: shared corners differ")
    move = cand_verts - verts

    face_n = np.cross(base_tri[:, 1] - base_tri[:, 0], base_tri[:, 2] - base_tri[:, 0])
    area2 = np.linalg.norm(face_n, axis=1)
    live = area2 > 1e-14
    unit_face = np.zeros_like(face_n)
    unit_face[live] = face_n[live] / area2[live, None]
    vert_n = np.zeros_like(verts)
    cov = np.zeros((len(verts), 3, 3))
    outer = unit_face[:, :, None] * unit_face[:, None, :]
    for k in range(3):
        np.add.at(vert_n, inv[:, k], face_n)
        np.add.at(cov, inv[:, k], outer)
    length = np.linalg.norm(vert_n, axis=1)
    has_n = length > 1e-14
    unit_n = np.zeros_like(vert_n)
    unit_n[has_n] = vert_n[has_n] / length[has_n, None]

    spread = np.ones(len(verts))
    for k in range(3):
        cosine = (unit_face * unit_n[inv[:, k]]).sum(axis=1)
        np.minimum.at(spread, inv[:, k][live], cosine[live])
    feature = (spread < np.cos(np.radians(settings.feature_angle_deg))) | ~has_n

    out = move.copy()
    smooth = ~feature
    normal_part = (move * unit_n).sum(axis=1)[:, None] * unit_n
    lam = np.zeros(len(verts))
    edge_move = np.zeros_like(move)
    if feature.any():
        f_idx = np.flatnonzero(feature)
        evals, evecs = np.linalg.eigh(cov[f_idx])
        tangent = evecs[:, :, 0]
        corner = evals[:, 0] > 0.15 * evals[:, 2]
        along = (move[f_idx] * tangent).sum(axis=1)[:, None] * tangent
        f_move = np.where(corner[:, None], move[f_idx], move[f_idx] - along)
        # The second hull's sharp edges are the same triangle edges, at its own coordinates.
        edges = np.concatenate([inv[:, [0, 1]], inv[:, [1, 2]], inv[:, [2, 0]]])
        edges = edges[feature[edges].all(axis=1) & (edges[:, 0] != edges[:, 1])]
        on_edge = np.zeros(len(verts), dtype=bool)
        on_edge[edges.ravel()] = True
        slide = ~corner & on_edge[f_idx]
        if slide.any():
            landed = _closest_point_on_segments(
                cand_verts[edges[:, 0]], cand_verts[edges[:, 1]], verts[f_idx[slide]] + f_move[slide]
            )
            f_move[slide] = landed - verts[f_idx[slide]]
        f_move[~slide] = move[f_idx[~slide]]
        out[f_idx] = f_move
        d_feat, nearest = cKDTree(verts[f_idx]).query(verts)
        lam = 1.0 - _smoothstep(d_feat / settings.feature_width_m)
        edge_move = f_move[nearest]
    across = edge_move - (edge_move * unit_n).sum(axis=1)[:, None] * unit_n
    out[smooth] = normal_part[smooth] + lam[smooth, None] * across[smooth]

    cand_tree = cKDTree(cand_tri.mean(axis=1))
    target, miss = _closest_point_on_surface(cand_tri, cand_tree, verts[smooth] + out[smooth])
    out[smooth] = target - verts[smooth]
    on_plane = np.abs(verts[:, 1]) < 1e-9
    out[on_plane, 1] = 0.0
    _, off_skin = _closest_point_on_surface(cand_tri, cand_tree, verts + out)
    report = {
        "vertices": int(len(verts)),
        "sharp_edge_vertices": int(feature.sum()),
        "max_first_order_miss_mm": float(1e3 * miss.max()) if smooth.any() else 0.0,
        "max_distance_from_candidate_skin_mm": float(1e3 * off_skin.max()),
        "max_move_mm": float(1e3 * np.linalg.norm(out, axis=1).max()),
        "max_parametric_move_mm": float(1e3 * np.linalg.norm(move, axis=1).max()),
    }
    return out[inv], report


class SurfaceMorph:
    """Displacement field that takes the mesh around ``base_tri`` to the hull ``cand_tri``."""

    def __init__(
        self,
        base_tri: np.ndarray,
        cand_tri: np.ndarray,
        settings: MorphSettings | None = None,
        *,
        scale: float = 1.0,
    ) -> None:
        base_tri = np.asarray(base_tri, dtype=float)
        cand_tri = np.asarray(cand_tri, dtype=float)
        if base_tri.ndim != 3 or base_tri.shape[1:] != (3, 3) or len(base_tri) == 0:
            raise ValueError(f"triangles must be a non-empty (n, 3, 3) array, got {base_tri.shape}")
        if base_tri.shape != cand_tri.shape:
            raise ValueError(
                "the two hulls do not have the same surface topology: "
                f"{len(base_tri)} and {len(cand_tri)} triangles"
            )
        self.settings = settings or MorphSettings()
        self._tri = base_tri
        disp, self.skin_report = skin_displacement(base_tri, cand_tri, self.settings)
        self._disp = disp * float(scale)
        self._tree = cKDTree(base_tri.mean(axis=1))
        flat = base_tri.reshape(-1, 3)
        self._lo = flat.min(axis=0) - self.settings.radius_m
        self._hi = flat.max(axis=0) + self.settings.radius_m
        self.max_surface_move_m = float(np.linalg.norm(self._disp, axis=-1).max())

    def weight(self, dist: np.ndarray) -> np.ndarray:
        r0, radius = self.settings.r0_m, self.settings.radius_m
        return 1.0 - _smoothstep((np.asarray(dist, dtype=float) - r0) / (radius - r0))

    def displacement(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Displacement of each point, and its distance to the first hull (inf where it is far)."""
        pts = np.asarray(pts, dtype=float)
        disp = np.zeros_like(pts)
        dist = np.full(len(pts), np.inf)
        k = min(N_CANDIDATE_TRIANGLES, len(self._tri))
        near = np.flatnonzero(((pts >= self._lo) & (pts <= self._hi)).all(axis=1))
        for start in range(0, len(near), _CHUNK):
            idx = near[start : start + _CHUNK]
            p = pts[idx]
            d_centroid, cand = self._tree.query(p, k=k)
            if k == 1:
                d_centroid, cand = d_centroid[:, None], cand[:, None]
            tri = self._tri[cand]
            pk = np.broadcast_to(p[:, None, :], tri.shape[:2] + (3,))
            bary = closest_on_triangles(pk, tri[:, :, 0], tri[:, :, 1], tri[:, :, 2])
            d_all = np.linalg.norm((bary[..., None] * tri).sum(axis=2) - pk, axis=-1)
            best = d_all.argmin(axis=1)
            rows = np.arange(len(idx))
            d_best = d_all[rows, best]
            moved = (bary[rows, best][..., None] * self._disp[cand[rows, best]]).sum(axis=1)
            disp[idx] = moved * self.weight(d_best)[:, None]
            dist[idx] = d_best
        disp[np.abs(pts[:, 1]) < 1e-9, 1] = 0.0
        return disp, dist


def surface_change_report(base_tri: np.ndarray, cand_tri: np.ndarray) -> dict[str, float | int]:
    """How far the second hull's skin is from the first, split into normal and tangential parts."""
    normal = np.cross(base_tri[:, 1] - base_tri[:, 0], base_tri[:, 2] - base_tri[:, 0])
    area2 = np.linalg.norm(normal, axis=1)
    ok = area2 > 1e-14
    unit = normal[ok] / area2[ok, None]
    dv = (cand_tri - base_tri)[ok]
    dn = (dv * unit[:, None, :]).sum(axis=-1)
    dt = np.sqrt(np.maximum((dv**2).sum(axis=-1) - dn**2, 0.0))
    wet = base_tri[ok][..., 2] < 0.0
    return {
        "triangles": int(len(base_tri)),
        "degenerate_triangles": int((~ok).sum()),
        "max_move_mm": float(1e3 * np.linalg.norm(cand_tri - base_tri, axis=-1).max()),
        "max_normal_move_mm": float(1e3 * np.abs(dn).max()),
        "max_tangential_move_mm": float(1e3 * dt.max()),
        "wetted_max_normal_move_mm": float(1e3 * np.abs(dn[wet]).max()) if wet.any() else 0.0,
        "wetted_rms_normal_move_mm": float(1e3 * np.sqrt((dn[wet] ** 2).mean())) if wet.any() else 0.0,
    }


def morph_case_points(case_dir: Path, morph: SurfaceMorph) -> dict[str, float | int]:
    """Move every ``points`` file of a case (undecomposed and per processor) with ``morph``.

    The unmoved file is kept once as ``points.orig`` and is always the input, so calling this again
    with another morph starts from the unmoved mesh.
    """
    case_dir = Path(case_dir)
    files = sorted(case_dir.glob("processor*/constant/polyMesh/points"))
    files += sorted(case_dir.glob("constant/polyMesh/points"))
    if not files:
        raise FileNotFoundError(f"no polyMesh/points under {case_dir}")
    n_points = n_moved = n_wall = 0
    max_move = 0.0
    for f in files:
        orig = f.with_name("points.orig")
        pts, header, trailer = read_points(orig if orig.exists() else f)
        disp, dist = morph.displacement(pts)
        mag = np.linalg.norm(disp, axis=1)
        n_points += len(pts)
        n_moved += int((mag > 0.0).sum())
        n_wall += int((dist < 1e-3).sum())
        max_move = max(max_move, float(mag.max()) if len(mag) else 0.0)
        if not orig.exists():
            f.rename(orig)
        write_points(f, pts + disp, header, trailer)
    if n_wall == 0:
        raise ValueError(f"no mesh point of {case_dir} lies on the hull the mesh was built for")
    return {
        "points_files": len(files),
        "points": n_points,
        "points_moved": n_moved,
        "points_within_1mm_of_hull": n_wall,
        "max_point_move_mm": 1e3 * max_move,
    }
