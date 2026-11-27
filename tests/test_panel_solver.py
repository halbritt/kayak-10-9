"""Free-surface panel solver (rsi.panel_solver): kernels, geometry, limits with known answers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from rsi import panel_solver as ps
from rsi.cli import app
from rsi.panel_hull import evaluate_hull

BASELINE_HULL = Path(__file__).resolve().parents[1] / "hull" / "kayak-10-9.hull.json"

# Small grids: these tests check that the method is right, not how accurate a fine grid is. The
# accuracy and convergence figures are in docs/panel-solver/README.md.
COARSE = dict(nz_hull=6)


def _quadrature_velocity(corners: np.ndarray, p: np.ndarray, n: int = 160) -> np.ndarray:
    """Velocity at p from unit source density on a flat quadrilateral, by Gauss quadrature."""
    g, w = np.polynomial.legendre.leggauss(n)
    g, w = 0.5 * (g + 1.0), 0.5 * w
    u, v = np.meshgrid(g, g, indexing="ij")
    u, v = u[..., None], v[..., None]
    q = (1 - u) * (1 - v) * corners[0] + u * (1 - v) * corners[1] + u * v * corners[2] + (1 - u) * v * corners[3]
    qu = (1 - v) * (corners[1] - corners[0]) + v * (corners[2] - corners[3])
    qv = (1 - u) * (corners[3] - corners[0]) + u * (corners[2] - corners[1])
    jac = np.linalg.norm(np.cross(qu, qv), axis=-1) * np.outer(w, w)
    r = p - q
    rn = np.linalg.norm(r, axis=-1)
    return (r / rn[..., None] ** 3 * jac[..., None]).sum((0, 1)) / (4.0 * math.pi)


def _sphere_mesh(radius: float = 1.0, n_long: int = 96, n_round: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """A sphere with its poles on the x axis, centred on the origin."""
    alpha = np.linspace(0.0, math.pi, n_long + 1)
    beta = np.linspace(0.0, 2.0 * math.pi, n_round, endpoint=False)
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    v = np.stack(
        [-radius * np.cos(aa), radius * np.sin(aa) * np.cos(bb), radius * np.sin(aa) * np.sin(bb)], -1
    ).reshape(-1, 3)
    idx = np.arange((n_long + 1) * n_round).reshape(n_long + 1, n_round)
    a, b = idx[:-1], idx[1:]
    a2, b2 = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
    f = np.concatenate([np.stack([a, b, b2], -1).reshape(-1, 3), np.stack([a, b2, a2], -1).reshape(-1, 3)])
    return v, f


def test_panel_velocity_matches_quadrature_on_both_sides() -> None:
    corners = np.array([[[0.0, 0.0, 0.0], [1.0, 0.1, 0.02], [1.2, 0.9, 0.0], [-0.1, 1.1, 0.03]]])
    pan = ps.make_panels(corners)
    pts = np.array([[0.5, 0.5, 0.7], [0.5, 0.5, -0.4], [3.0, 1.0, 2.0], [-1.0, 0.2, 0.0], [0.3, 2.5, -0.1]])
    got = ps._influence_block(np, pts, pan)[:, 0, :]
    for p, v in zip(pts, got):
        assert v == pytest.approx(_quadrature_velocity(pan.corners[0], p), rel=1e-6, abs=1e-9)


def test_a_triangle_is_a_quadrilateral_with_a_repeated_corner() -> None:
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.8, 0.0], [0.3, 0.8, 0.0]]])
    pan = ps.make_panels(tri)
    assert pan.area[0] == pytest.approx(0.4)
    p = np.array([[0.4, 0.2, 0.5]])
    assert ps._influence_block(np, p, pan)[0, 0] == pytest.approx(
        _quadrature_velocity(pan.corners[0], p[0]), rel=1e-6
    )


def test_a_panel_blows_half_its_density_out_of_its_own_fluid_side() -> None:
    corners = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    pan = ps.make_panels(corners)  # this corner order puts the normal along -z
    assert pan.normal[0] == pytest.approx([0.0, 0.0, -1.0])
    just_below = pan.centre + 1e-7 * pan.normal
    assert ps._influence_block(np, just_below, pan)[0, 0, 2] == pytest.approx(-0.5, abs=1e-5)
    # the y image is far away here; the self term is exactly one half along the normal
    far = ps.make_panels(corners + np.array([0.0, 50.0, 0.0]))
    v = ps.velocity_influence(np, far.centre, far, self_term=True)[0, 0]
    assert float(v @ far.normal[0]) == pytest.approx(0.5, abs=1e-4)


def test_the_images_make_the_centreplane_and_the_waterplane_walls() -> None:
    corners = np.array([[[0.0, 0.3, -0.2], [0.5, 0.4, -0.25], [0.6, 0.6, -0.1], [0.1, 0.5, -0.05]]])
    pan = ps.make_panels(corners)
    on_centreplane = np.array([[0.2, 0.0, -0.3], [-1.0, 0.0, -0.1]])
    assert ps.velocity_influence(np, on_centreplane, pan)[:, 0, 1] == pytest.approx(0.0, abs=1e-12)
    on_waterplane = np.array([[0.2, 0.7, 0.0], [-1.0, 0.2, 0.0]])
    v, w = ps.velocity_influence(np, on_waterplane, pan, z_image=True)
    assert (v + w)[:, 0, 2] == pytest.approx(0.0, abs=1e-12)


def test_the_upstream_difference_is_exact_for_a_quadratic_and_never_looks_downstream() -> None:
    i = np.arange(12.0)[:, None] * np.ones((1, 3))
    f = 0.3 * i**2 - 2.0 * i + 1.0
    d = ps._d_xi(np, f, upstream_zero=True)
    assert d[3:] == pytest.approx(0.6 * i[3:] - 2.0)
    assert np.all(d[0] == 0.0)
    g = f.copy()
    g[8:] += 5.0  # a change downstream of column 7 ...
    assert ps._d_xi(np, g, upstream_zero=True)[:8] == pytest.approx(d[:8])  # ... is not felt upstream
    cross = np.arange(5.0)[None, :] ** 2 * np.ones((4, 1))
    assert ps._d_eta(np, cross) == pytest.approx(2.0 * np.arange(5.0)[None, :] * np.ones((4, 1)))


def test_wigley_volume_and_panelled_area() -> None:
    v, f = ps.wigley_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    hs = ps.hydrostatics(tris)
    assert hs["vol"] == pytest.approx(4.0 / 9.0 * 0.1 * 0.0625, rel=2e-3)
    assert hs["xb"] == pytest.approx(0.0, abs=1e-4)
    grid = ps._build_grid(tris, ps.PanelConfig(nx_hull=40, nz_hull=10, ny_fs=8))
    assert grid.wetted_area == pytest.approx(hs["wet"], rel=5e-3)
    assert grid.lwl == pytest.approx(1.0)
    assert np.all(grid.hull.normal[:, 1] >= 0.0)  # out of the hull, on the starboard side
    assert np.all(grid.fs.normal[:, 2] == -1.0)


def test_station_clustering_keeps_the_ends_and_gathers_stations_there() -> None:
    even = ps._station_fractions(20, 0.0)
    packed = ps._station_fractions(20, 0.8)
    assert even == pytest.approx(np.linspace(0.0, 1.0, 21))
    assert packed[[0, -1]] == pytest.approx([0.0, 1.0])
    assert np.all(np.diff(packed) > 0.0)
    assert np.diff(packed)[0] < 0.4 * np.diff(even)[0] < np.diff(packed)[10]


def test_double_body_flow_about_a_sphere_gives_the_known_suction() -> None:
    """At vanishing speed the surface is a rigid lid. A half-immersed sphere then has the surface
    speed 1.5 U sin(theta) of a sphere in unbounded flow, and its lower half is drawn down with
    0.6875 pi R^2 times the dynamic pressure."""
    v, f = _sphere_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    cfg = ps.PanelConfig(nx_hull=24, ny_fs=8, **{**COARSE, "nz_hull": 10, "half_width": 2.0})
    speed = 0.02
    r = ps.solve_fixed(tris, speed, cfg, rho=1000.0)[0]
    q = 0.5 * 1000.0 * speed**2
    assert r.fz_dyn_N / q == pytest.approx(-0.6875 * math.pi, rel=0.03)
    assert abs(r.rw_N) / q < 0.01  # no waves, no resistance


def test_thin_ship_limit_agrees_with_michell() -> None:
    """With the uniform stream as base flow and a hull thin enough for Michell's assumption, the
    panel solver and Michell's integral solve the same problem. A coarse grid falls a little
    short of Michell; docs/panel-solver/README.md shows the gap closing with refinement."""
    from rsi.evaluator import _michell_fast_curve

    beam = 0.005
    v, f = ps.wigley_mesh(beam_ratio=beam)
    tris = ps.pose_triangles(v, f, 0.0)
    speeds = np.array([0.35, 0.45]) * math.sqrt(ps.G)
    cfg = ps.PanelConfig(nx_hull=40, ny_fs=12, linearisation="kelvin", **COARSE)
    res = ps.solve_fixed(tris, speeds, cfg, rho=1000.0)
    xs = np.linspace(-0.5, 0.5, 401)
    zs = np.linspace(-0.0625, 0.0, 81)
    yy = 0.5 * beam * (1.0 - (2.0 * xs[:, None]) ** 2) * (1.0 - (zs[None, :] / 0.0625) ** 2)
    michell = _michell_fast_curve(xs, zs, yy, speeds, 1000.0, g=ps.G)
    for r, m in zip(res, michell):
        assert 0.80 < r.rw_N / m < 1.05, (r.froude, r.rw_N / m)


def test_resistance_does_not_depend_on_how_many_columns_lie_upstream() -> None:
    """Guards the collocation shift. With collocation under the panel centres a point-to-point
    oscillation reached the upstream edge, and the resistance jumped by up to a tenth between an
    odd and an even number of upstream columns."""
    v, f = ps.wigley_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    speed = 0.40 * math.sqrt(ps.G)
    rw = [
        ps.solve_fixed(tris, speed, ps.PanelConfig(nx_hull=24, ny_fs=10, upstream=n / 24, **COARSE))[0].rw_N
        for n in (12, 13, 14)
    ]
    assert max(rw) - min(rw) < 0.01 * rw[0]


def test_wigley_wave_resistance_is_in_the_published_range() -> None:
    """A coarse-grid sanity bound at Fn 0.40: tabulated fixed-model residuary coefficient 2.12e-3
    (Shahshahan and Landweber 1986, Table 6) and Michell 2.74e-3; see the validation document."""
    v, f = ps.wigley_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    r = ps.solve_fixed(tris, 0.40 * math.sqrt(ps.G), ps.PanelConfig(nx_hull=30, ny_fs=16, **COARSE))[0]
    assert 1.8e-3 < r.cw < 2.8e-3
    assert r.fz_dyn_N < 0.0  # the hull is drawn down


def test_same_inputs_give_the_same_answer() -> None:
    v, f = ps.wigley_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    cfg = ps.PanelConfig(nx_hull=20, ny_fs=8, **COARSE)
    a = ps.solve_fixed(tris, 1.2, cfg)[0]
    b = ps.solve_fixed(tris, 1.2, cfg)[0]
    assert a.rw_N == pytest.approx(b.rw_N, rel=1e-10)
    assert a.fz_dyn_N == pytest.approx(b.fz_dyn_N, rel=1e-10)


def test_free_hull_balances_weight_buoyancy_and_dynamic_force() -> None:
    v, f = ps.wigley_mesh()
    volume = ps.hydrostatics(ps.pose_triangles(v, f, 0.0))["vol"]
    cfg = ps.PanelConfig(nx_hull=20, ny_fs=8, **COARSE)
    rho = 1000.0
    r = ps.solve_free(v, f, 0.35 * math.sqrt(ps.G), cfg, volume_m3=volume, vcg_m=0.0, rho=rho)
    assert r.pose_converged
    assert r.sinkage_m is not None and 0.0 < r.sinkage_m < 0.01
    hs = ps.hydrostatics(ps.pose_triangles(v, f, r.h_m, r.trim_rad))
    weight = rho * ps.G * volume
    assert rho * ps.G * hs["vol"] + r.fz_dyn_N == pytest.approx(weight, rel=1e-5)
    fixed = ps.solve_fixed(ps.pose_triangles(v, f, 0.0), r.speed_mps, cfg, rho=rho)[0]
    assert r.rw_N != pytest.approx(fixed.rw_N, rel=1e-4)  # the pose moved, and the flow with it


def test_unconverged_free_pose_keeps_the_last_solved_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    from rsi.panel_hull import evaluate_mesh

    v, f = ps.wigley_mesh(nx=80, nz=22)
    volume = ps.hydrostatics(ps.pose_triangles(v, f, 0.0))["vol"]
    cfg = ps.PanelConfig(nx_hull=12, ny_fs=5, nz_hull=3)
    speed = 0.35 * math.sqrt(ps.G)
    r = ps.solve_free(v, f, speed, cfg, volume_m3=volume, vcg_m=0.0, max_iterations=1)

    assert r.pose_converged is False
    assert r.sinkage_m == pytest.approx(r.h_m - ps.static_pose(v, f, volume))
    assert r.trim_change_rad == r.trim_rad == 0.0
    monkeypatch.setattr(ps, "solve_free", lambda *args, **kwargs: r)
    with pytest.raises(RuntimeError, match="Free pose did not converge"):
        evaluate_mesh(v, f, 225.0, np.array([speed]), cfg, free=True)


def test_a_hull_with_an_immersed_transom_is_rejected() -> None:
    v, f = ps.wigley_mesh()
    cut = v.copy()
    cut[:, 0] = np.minimum(cut[:, 0], 0.3)  # squash the after body onto a flat transom
    with pytest.raises(ValueError):
        ps._build_grid(ps.pose_triangles(cut, f, 0.0), ps.PanelConfig(nx_hull=20, ny_fs=8))


def test_unknown_settings_are_refused() -> None:
    v, f = ps.wigley_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    with pytest.raises(ValueError):
        ps.solve_fixed(tris, 1.0, ps.PanelConfig(linearisation="nonlinear"))
    with pytest.raises(ValueError):
        ps.get_backend("fortran")
    with pytest.raises(ValueError):
        ps._build_grid(tris, ps.PanelConfig(fs_raise=0.0, fs_shift=0.5))


def test_hull_json_report_has_panel_and_tier1_parts_and_a_zero_self_difference() -> None:
    cfg = ps.PanelConfig(nx_hull=16, ny_fs=8, nz_hull=4)
    rep = evaluate_hull(
        BASELINE_HULL, [5.5], cfg, baseline_json=BASELINE_HULL, free=False, mesh_resolution=(161, 49)
    )
    panel, tier1 = rep["candidate"]["panel"][0], rep["candidate"]["tier1"]
    assert 5.3 < panel["lwl_m"] < 5.5
    assert 1.0 < panel["rw_N"] < 8.0 and 1.0 < tier1["michell_N"][0] < 8.0
    # a hull that is its own baseline pays the plain ITTC-57 friction
    assert tier1["friction_by_position_N"][0] == pytest.approx(tier1["friction_ittc57_N"][0], rel=1e-12)
    diff = rep["difference_N"][0]
    assert diff["panel_wave"] == pytest.approx(0.0, abs=1e-9)
    assert diff["michell_wave"] == 0.0 and diff["friction_by_position"] == 0.0


def test_panel_wave_command(tmp_path: Path) -> None:
    out = tmp_path / "panel.json"
    args = ["panel-wave", "--hull", str(BASELINE_HULL), "--fixed", "--speed-kn", "5.5"]
    args += ["--nx", "16", "--ny", "8", "--nz", "4", "--out", str(out)]
    done = CliRunner().invoke(app, args)
    assert done.exit_code == 0, done.output
    assert "panel wave" in done.output
    payload = json.loads(out.read_text())
    assert payload["command"] == "panel-wave" and payload["free_to_sink_and_trim"] is False
    assert payload["candidate"]["panel"][0]["rw_N"] > 0.0
    bad = CliRunner().invoke(app, ["panel-wave", "--hull", str(BASELINE_HULL), "--backend", "fortran"])
    assert bad.exit_code == 2
