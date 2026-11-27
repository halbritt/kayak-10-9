"""Moving a mesh from one hull's skin onto another's (cad/openfoam_morph.py)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))

from openfoam_morph import (  # noqa: E402
    MorphSettings,
    SurfaceMorph,
    closest_on_triangles,
    morph_case_points,
    read_binary_stl,
    read_points,
    skin_displacement,
    write_points,
)

HEADER = b"""FoamFile
{
    version     2.0;
    format      %s;
    class       vectorField;
    location    "constant/polyMesh";
    object      points;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


"""
TRAILER = b"\n\n\n// ************************************************************************* //\n"


def _ascii_points(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"({x:.17g} {y:.17g} {z:.17g})" for x, y, z in pts)
    path.write_bytes(HEADER % b"ascii" + f"{len(pts)}\n(\n{body}\n)".encode() + TRAILER)


def _binary_points(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(HEADER % b"binary" + f"{len(pts)}\n(".encode() + pts.astype("<f8").tobytes() + b")" + TRAILER)


def _plate(n: int = 12) -> np.ndarray:
    """A triangulated plate in the plane z = 0 over x in [0, 2], y in [-1, 1]: both sides of y = 0."""
    xs, ys = np.linspace(0.0, 2.0, n + 1), np.linspace(-1.0, 1.0, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a = (xs[i], ys[j], 0.0)
            b = (xs[i + 1], ys[j], 0.0)
            c = (xs[i + 1], ys[j + 1], 0.0)
            d = (xs[i], ys[j + 1], 0.0)
            tris += [(a, b, c), (a, c, d)]
    return np.array(tris, dtype=float)


def _linear_move(tri: np.ndarray) -> np.ndarray:
    """The plate lifted onto the plane z = 0.004 + 0.003 x, and stretched 0.2 % in y within it.

    The stretch moves every point along the skin without changing the surface: a slide.
    """
    moved = tri.copy()
    moved[..., 2] += 0.004 + 0.003 * tri[..., 0]
    moved[..., 1] += 0.002 * tri[..., 1]
    return moved


def _tube(radius: float, n_x: int = 16, n_th: int = 240, *, turn: float = 0.0, y0: float = 0.0) -> np.ndarray:
    """A round tube about the x axis from x = 0 to 2, as triangles. ``turn`` rotates its vertices
    about the axis without changing the surface; ``y0`` shifts the tube sideways."""
    xs = np.linspace(0.0, 2.0, n_x + 1)
    th = np.linspace(-np.pi, np.pi, n_th + 1)[:-1] + np.pi / n_th + turn
    def pt(i: int, j: int) -> tuple[float, float, float]:
        a = th[j % n_th]
        return (xs[i], y0 + radius * np.sin(a), radius * np.cos(a))
    tris = []
    for i in range(n_x):
        for j in range(n_th):
            a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    return np.array(tris, dtype=float)


def _wedge(depth: float, half_width: float, x_slide: float = 0.0, n: int = 40) -> np.ndarray:
    """A V-shaped keel: two flat faces from the keel line (y = 0, z = -depth) up to
    (y = +-half_width, z = 0). ``x_slide`` moves the vertices along the keel without changing
    the surface. Triangles are 12.5 mm long and 1.5 mm wide, like a hull's near its ends."""
    xs = np.linspace(0.0, 2.0, 4 * n + 1) + x_slide
    ts = np.linspace(0.0, 1.0, n + 1)
    tris = []
    for side in (1.0, -1.0):

        def pt(ii: int, jj: int, side: float = side) -> tuple[float, float, float]:
            return (xs[ii], side * half_width * ts[jj], -depth * (1.0 - ts[jj]))

        for i in range(4 * n):
            for j in range(n):
                a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
                tris += [(a, b, c), (a, c, d)] if side > 0 else [(a, c, b), (a, d, c)]
    return np.array(tris, dtype=float)


def test_points_files_round_trip_in_both_formats(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(257, 3))
    for writer, name in ((_ascii_points, "a"), (_binary_points, "b")):
        f = tmp_path / name / "points"
        writer(f, pts)
        got, header, trailer = read_points(f)
        assert np.array_equal(got, pts)
        assert trailer == TRAILER
        out = tmp_path / name / "points.out"
        write_points(out, got, header, trailer)
        again, header2, _ = read_points(out)
        assert np.array_equal(again, got), "a rewritten file must hold the same doubles"
        assert b"format      ascii" in header2 and b"binary" not in header2


def test_closest_point_lands_in_the_right_part_of_the_triangle() -> None:
    a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    cases = {
        (0.25, 0.25, 0.7): (0.5, 0.25, 0.25),  # above the interior
        (-1.0, -1.0, 0.0): (1.0, 0.0, 0.0),  # beyond vertex a
        (2.0, -0.5, 0.3): (0.0, 1.0, 0.0),  # beyond vertex b
        (0.5, -2.0, 0.0): (0.5, 0.5, 0.0),  # off edge ab
        (1.0, 1.0, 0.0): (0.0, 0.5, 0.5),  # off edge bc
    }
    for p, want in cases.items():
        got = closest_on_triangles(np.array(p), a, b, c)
        assert np.allclose(got, want), (p, got)
    # a triangle squeezed to a line still gives a finite answer on it
    got = closest_on_triangles(np.array([0.5, 1.0, 0.0]), a, b, b)
    assert np.isfinite(got).all() and np.isclose(got.sum(), 1.0)


def test_wall_points_go_to_the_candidate_skin_and_far_points_stay() -> None:
    base = _plate()
    cand = _linear_move(base)
    morph = SurfaceMorph(base, cand, MorphSettings(r0_m=0.05, radius_m=0.4))
    rng = np.random.default_rng(3)
    xy = np.column_stack([rng.uniform(0.2, 1.8, 400), rng.uniform(-0.8, 0.8, 400)])

    on_wall = np.column_stack([xy, np.zeros(len(xy))])
    disp, dist = morph.displacement(on_wall)
    want = np.column_stack([np.zeros(len(xy)), np.zeros(len(xy)), 0.004 + 0.003 * xy[:, 0]])
    assert np.allclose(dist, 0.0, atol=1e-12)
    assert np.allclose(disp, want, atol=1e-12), "the skin must rise onto the new plane and not slide in it"

    near = on_wall + [0.0, 0.0, 0.03]  # inside r0: the wall cells move with the wall
    assert np.allclose(morph.displacement(near)[0], want, atol=1e-12)

    mid = on_wall + [0.0, 0.0, 0.225]  # halfway between r0 and the radius
    assert np.allclose(morph.displacement(mid)[0], 0.5 * want, atol=1e-12)

    far = on_wall + [0.0, 0.0, 0.4]
    assert np.array_equal(morph.displacement(far)[0], np.zeros_like(far))
    very_far = on_wall + [0.0, 0.0, 50.0]
    disp, dist = morph.displacement(very_far)
    assert np.array_equal(disp, np.zeros_like(very_far)) and np.isinf(dist).all()


def test_displacement_falls_monotonically_with_distance() -> None:
    base = _plate()
    morph = SurfaceMorph(base, _linear_move(base), MorphSettings(r0_m=0.02, radius_m=0.3))
    z = np.linspace(0.0, 0.35, 71)
    pts = np.column_stack([np.full_like(z, 1.0), np.full_like(z, 0.3), z])
    mag = np.linalg.norm(morph.displacement(pts)[0], axis=1)
    assert np.all(np.diff(mag) <= 1e-15)
    assert mag[0] > 0.0 and mag[-1] == 0.0


def test_points_on_the_symmetry_plane_stay_on_it() -> None:
    base = _tube(0.30)
    cand = _tube(0.30, y0=0.003)  # shifted sideways, which a symmetric hull never is
    morph = SurfaceMorph(base, cand, MorphSettings(r0_m=0.05, radius_m=0.4))
    pts = np.array([[1.0, 0.0, 0.31], [1.0, 1e-12, 0.45], [1.0, 0.31, 0.0]])
    disp, _ = morph.displacement(pts)
    assert disp[0, 1] == 0.0 and disp[1, 1] == 0.0
    assert disp[2, 1] == pytest.approx(0.003, abs=2e-5), "off the plane the sideways move is kept"


def test_skin_moves_along_its_normal_and_does_not_slide() -> None:
    base = _tube(0.30)
    cand = _tube(0.302, turn=0.02)  # 2 mm fatter; its vertices also turned 6 mm around the girth
    assert np.linalg.norm(cand - base, axis=-1).max() > 0.006
    disp, report = skin_displacement(base, cand)
    flat, moved = base.reshape(-1, 3), disp.reshape(-1, 3)
    radial = flat * [0.0, 1.0, 1.0] / 0.30
    along = (moved * radial).sum(axis=1)
    slide = np.linalg.norm(moved - along[:, None] * radial, axis=1)
    assert np.allclose(along, 0.002, atol=3e-5), "the skin must move out by the change in radius"
    assert slide.max() < 2e-4, f"the skin slid {1e3 * slide.max():.2f} mm along itself"
    assert report["max_distance_from_candidate_skin_mm"] < 1e-6
    assert report["sharp_edge_vertices"] == 0
    assert report["max_parametric_move_mm"] > 6.0 > report["max_move_mm"]


def test_a_sharp_keel_goes_to_the_new_keel_and_the_sides_do_not_slide_down() -> None:
    # Both faces move out along their normal by 1.5 mm, as a fuller forefoot does. With faces 10
    # degrees from vertical that puts the keel 8.6 mm deeper.
    alpha, out = np.radians(10.0), 0.0015
    base = _wedge(0.060, 0.060 * np.tan(alpha))
    drop = out / np.sin(alpha)
    cand = _wedge(0.060 + drop, 0.060 * np.tan(alpha) + out / np.cos(alpha), x_slide=0.004)
    disp, report = skin_displacement(base, cand, MorphSettings(feature_width_m=0.02))
    flat, moved = base.reshape(-1, 3), disp.reshape(-1, 3)
    inside = (flat[:, 0] > 0.1) & (flat[:, 0] < 1.9)  # away from the open ends

    keel = inside & (np.abs(flat[:, 1]) < 1e-12)
    assert keel.sum() > 200
    assert np.allclose(moved[keel], [0.0, 0.0, -drop], atol=1e-9), "keel points go straight down to the new keel"

    new = flat + moved
    on_new_face = -(0.060 + drop) + np.abs(new[:, 1]) / np.tan(alpha)
    assert np.allclose(new[inside, 2], on_new_face[inside], atol=1e-9), "every point lands on the new skin"

    # A parametric map slides the side down with the keel. Beyond the edge zone the skin must move
    # out along the face normal only: 1.48 mm sideways and 0.26 mm down.
    girth = np.hypot(flat[:, 1], flat[:, 2] + 0.060)
    side = inside & (girth > 0.025) & (flat[:, 2] < -0.005)
    assert (-(cand - base).reshape(-1, 3)[side, 2]).max() > 0.004
    assert np.allclose(np.abs(moved[side, 1]), out * np.cos(alpha), atol=1e-9)
    assert np.allclose(moved[side, 2], -out * np.sin(alpha), atol=1e-9)
    assert np.abs(moved[inside, 0]).max() < 1e-9, "no slide along the keel"
    assert report["sharp_edge_vertices"] == 161


def test_scale_gives_a_fraction_of_the_move() -> None:
    base = _plate()
    cand = _linear_move(base)
    pts = np.array([[0.7, 0.2, 0.0], [1.3, -0.4, 0.1]])
    full = SurfaceMorph(base, cand).displacement(pts)[0]
    half = SurfaceMorph(base, cand, scale=0.5).displacement(pts)[0]
    assert np.allclose(half, 0.5 * full, atol=1e-15)


def test_hulls_of_different_topology_are_refused() -> None:
    base = _plate(12)
    with pytest.raises(ValueError, match="same surface topology"):
        SurfaceMorph(base, _plate(10))
    with pytest.raises(ValueError, match="r0 < radius"):
        MorphSettings(r0_m=0.5, radius_m=0.5)


def test_binary_stl_reader_keeps_triangle_order(tmp_path: Path) -> None:
    tri = _plate(3).astype("<f4")
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = tri
    f = tmp_path / "s.stl"
    f.write_bytes(b"\0" * 80 + np.uint32(len(tri)).tobytes() + rec.tobytes())
    assert np.array_equal(read_binary_stl(f), tri.astype(float))
    f.write_text("solid x\nendsolid x\n")
    with pytest.raises(ValueError):
        read_binary_stl(f)


def test_a_case_is_moved_consistently_and_always_from_the_unmoved_mesh(tmp_path: Path) -> None:
    base = _plate()
    cand = _linear_move(base)
    rng = np.random.default_rng(11)
    shared = np.column_stack([rng.uniform(0.2, 1.8, 40), rng.uniform(-0.8, 0.8, 40), rng.uniform(0.0, 0.3, 40)])
    wall = np.column_stack([rng.uniform(0.2, 1.8, 30), rng.uniform(-0.8, 0.8, 30), np.zeros(30)])
    p0 = np.vstack([shared, wall, [[9.0, 9.0, 9.0]]])
    p1 = np.vstack([[[-9.0, 0.0, 0.0]], shared[::-1]])
    case = tmp_path / "case"
    _ascii_points(case / "processor0/constant/polyMesh/points", p0)
    _binary_points(case / "processor1/constant/polyMesh/points", p1)

    info = morph_case_points(case, SurfaceMorph(base, cand))
    assert info["points_files"] == 2 and info["points"] == len(p0) + len(p1)
    assert info["points_within_1mm_of_hull"] == 30
    m0 = read_points(case / "processor0/constant/polyMesh/points")[0]
    m1 = read_points(case / "processor1/constant/polyMesh/points")[0]
    assert np.allclose(m0[:40], m1[1:][::-1], atol=1e-9), "a point shared by two processors must move the same"
    assert np.array_equal(m0[-1], p0[-1]) and np.array_equal(m1[0], p1[0])
    assert np.allclose(m0[40:70, 2], 0.004 + 0.003 * wall[:, 0], atol=1e-9)
    assert np.allclose(m0[40:70, :2], wall[:, :2], atol=1e-12), "wall points rise and do not slide"

    # a second call starts again from the unmoved mesh, it does not move the moved mesh further
    morph_case_points(case, SurfaceMorph(base, cand, scale=0.5))
    h0 = read_points(case / "processor0/constant/polyMesh/points")[0]
    assert np.allclose(h0 - p0, 0.5 * (m0 - p0), rtol=0, atol=1e-15)
    assert np.array_equal(read_points(case / "processor0/constant/polyMesh/points.orig")[0], p0)


def test_a_mesh_that_does_not_touch_the_hull_is_refused(tmp_path: Path) -> None:
    base = _plate()
    case = tmp_path / "case"
    _ascii_points(case / "constant/polyMesh/points", np.array([[1.0, 0.2, 0.2], [5.0, 5.0, 5.0]]))
    with pytest.raises(ValueError, match="lies on the hull"):
        morph_case_points(case, SurfaceMorph(base, _linear_move(base)))
    with pytest.raises(FileNotFoundError):
        morph_case_points(tmp_path / "empty", SurfaceMorph(base, _linear_move(base)))
