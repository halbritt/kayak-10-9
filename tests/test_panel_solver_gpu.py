"""GPU backend of the panel solver (CuPy) against the numpy backend.

Skipped where CuPy or a CUDA device is missing: the repository's default environment does not
install the ``gpu`` extra. A skip is not a pass; run these in an environment that has CuPy.
The grids are small so the run fits beside other users of the card (about 0.2 GiB).
"""

from __future__ import annotations

import math

import pytest

from rsi import panel_solver as ps

cupy = pytest.importorskip("cupy")
try:
    cupy.cuda.Device(0).compute_capability
except Exception as exc:  # noqa: BLE001 - any driver failure means there is no usable device
    pytest.skip(f"no usable CUDA device: {exc}", allow_module_level=True)

# Stated tolerances, relative, on wave resistance and on the dynamic vertical force.
DOUBLE_MATCH = 1e-9  # same arithmetic, different order of summation and LU pivoting
SINGLE_MATCH = 2e-3  # single precision on either backend against double precision


@pytest.fixture(scope="module")
def wigley() -> tuple:
    v, f = ps.wigley_mesh()
    return ps.pose_triangles(v, f, 0.0), [0.35 * math.sqrt(ps.G), 0.45 * math.sqrt(ps.G)]


def _solve(wigley: tuple, backend: str, precision: str = "double", **kw: object) -> list[ps.PanelResult]:
    cfg = ps.PanelConfig(nx_hull=24, ny_fs=14, nz_hull=6, precision=precision, pair_budget=60_000, **kw)
    return ps.solve_fixed(wigley[0], wigley[1], cfg, rho=1000.0, backend=backend)


def test_gpu_double_precision_matches_numpy(wigley: tuple) -> None:
    cpu = _solve(wigley, "numpy")
    gpu = _solve(wigley, "cupy")
    for a, b in zip(cpu, gpu):
        assert b.backend == "cupy"
        assert b.rw_N == pytest.approx(a.rw_N, rel=DOUBLE_MATCH)
        assert b.fz_dyn_N == pytest.approx(a.fz_dyn_N, rel=DOUBLE_MATCH)
        assert b.my_dyn_Nm == pytest.approx(a.my_dyn_Nm, rel=DOUBLE_MATCH, abs=1e-12)
        assert b.rw_waterline_N == pytest.approx(a.rw_waterline_N, rel=1e-7)


def test_gpu_control_surface_estimate_matches_numpy(wigley: tuple) -> None:
    cpu = _solve(wigley, "numpy", farfield=True, downstream=1.0)
    gpu = _solve(wigley, "cupy", farfield=True, downstream=1.0)
    for a, b in zip(cpu, gpu):
        assert b.rw_farfield_N == pytest.approx(a.rw_farfield_N, rel=1e-7)


def test_gpu_single_precision_is_close_to_double(wigley: tuple) -> None:
    ref = _solve(wigley, "numpy")
    for backend in ("numpy", "cupy"):
        for a, b in zip(ref, _solve(wigley, backend, "single")):
            assert b.rw_N == pytest.approx(a.rw_N, rel=SINGLE_MATCH)
            assert b.fz_dyn_N == pytest.approx(a.fz_dyn_N, rel=SINGLE_MATCH)


def test_gpu_free_run_reaches_the_same_pose() -> None:
    v, f = ps.wigley_mesh()
    volume = ps.hydrostatics(ps.pose_triangles(v, f, 0.0))["vol"]
    cfg = ps.PanelConfig(nx_hull=20, ny_fs=8, nz_hull=6, pair_budget=60_000)
    kw = dict(volume_m3=volume, vcg_m=0.0, rho=1000.0)
    a = ps.solve_free(v, f, 0.35 * math.sqrt(ps.G), cfg, backend="numpy", **kw)
    b = ps.solve_free(v, f, 0.35 * math.sqrt(ps.G), cfg, backend="cupy", **kw)
    assert b.pose_converged and b.pose_iterations == a.pose_iterations
    assert b.sinkage_m == pytest.approx(a.sinkage_m, rel=1e-7)
    assert b.rw_N == pytest.approx(a.rw_N, rel=1e-7)
