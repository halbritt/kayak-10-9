"""Unit and integration tests for direct 3-grid panel-solver Tier-1 wave drag."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from rsi.baseline import DEFAULT_REPO_ROOT, PANEL_WAVE_REFERENCE_FILE, load_baseline_reference
from rsi.evaluator import (
    PANEL_WAVE_GRID_FAMILY,
    RESISTANCE_SPEEDS_KN,
    evaluate_tier0_tier1,
    solve_panel_wave_curve,
)
from rsi.gates import GATES_VERSION, evaluate_gates
from rsi.loop import CampaignConfig, ResumeStateError, run_campaign, write_json_atomic
from rsi.panel_hull import hull_mesh
from rsi.panel_solver import KN


def test_panel_wave_reference_and_baseline_self_consistency(tmp_path: Path) -> None:
    ref = load_baseline_reference()
    assert ref.panel_wave is not None
    assert ref.panel_wave.grid_family == PANEL_WAVE_GRID_FAMILY
    stored = json.loads((DEFAULT_REPO_ROOT / "rsi" / PANEL_WAVE_REFERENCE_FILE).read_text())
    assert stored["design_hash"] == ref.design_hash
    assert stored["speeds_kn"] == [float(v) for v in RESISTANCE_SPEEDS_KN]

    base = evaluate_tier0_tier1(ref.hull, tmp_path / "base_pw", ref)
    assert base.valid, base.error
    assert base.wave_model == "panel_dawson_3grid"
    assert base.wave_backend in ("cupy", "numpy")
    assert base.kn_225 == ref.kn_225
    assert base.michell_kn_225 == {40: 4.47, 60: 5.08, 90: 5.67, 120: 6.17}
    assert base.kn_225_exact is not None and ref.kn_225_exact is not None
    for pw in (40, 60, 90, 120):
        assert base.kn_225_exact[pw] == pytest.approx(ref.kn_225_exact[pw], abs=5e-5)
    report = evaluate_gates(base, ref)
    assert report.passed is True, report.violations
    assert abs(report.objectives["speed_gain_kn"]) < 5e-5


def test_cpu_numpy_panel_wave_matches_reference(tmp_path: Path) -> None:
    del tmp_path
    ref = load_baseline_reference()
    assert ref.panel_wave is not None
    V, F = hull_mesh(DEFAULT_REPO_ROOT / "hull" / "kayak-10-9.hull.json")
    rw_cpu, bk = solve_panel_wave_curve(V, F, 225.0, RESISTANCE_SPEEDS_KN * KN, backend="numpy")
    assert bk == {"numpy"}
    assert rw_cpu == pytest.approx(np.asarray(ref.panel_wave.baseline_panel_rw_N[225]), abs=5e-4)


def test_refused_geometry_raises_panel_wave_refused() -> None:
    ref = load_baseline_reference()
    V, F = hull_mesh(DEFAULT_REPO_ROOT / "hull" / "kayak-10-9.hull.json")
    # Chop stern tips below water so the waterline is an immersed transom.
    V_cut = V.copy()
    z_clip = float(V_cut[:, 2].min()) + 0.04
    V_cut[:, 0] = np.minimum(V_cut[:, 0], 1.5)
    assert np.any(V_cut[:, 2] <= z_clip)
    with pytest.raises(ValueError, match="panel_wave_refused"):
        solve_panel_wave_curve(V_cut, F, 225.0, RESISTANCE_SPEEDS_KN * KN, backend="numpy")

    bad_pw = replace(
        ref,
        panel_wave=replace(
            ref.panel_wave,
            speeds_kn=tuple(v + 0.25 for v in ref.panel_wave.speeds_kn),
        ),
    )
    out = evaluate_tier0_tier1(ref.hull, Path.home() / ".cache" / "kayak-10-9" / "test-work" / "pw_bad", bad_pw)
    assert out.valid is False and "panel wave" in (out.error or "")
    assert evaluate_gates(out, ref).passed is False


def test_gates_v10_provenance_and_multiworker_panel_campaign(tmp_path: Path) -> None:
    assert GATES_VERSION == "10"
    w2 = tmp_path / "w2"
    res = run_campaign(CampaignConfig(iterations=1, population=2, workers=2, seed=17, work_dir=w2, pre_screen=False))
    assert res.status == "completed"
    prov = res.provenance
    assert prov["gates_version"] == "10"
    assert prov["panel_solver_sha"]
    assert prov["panel_grid_family"] == [list(g) for g in PANEL_WAVE_GRID_FAMILY]
    assert prov["panel_precision"] == "single"
    assert prov["panel_linearisation"] == "dawson"
    assert "cupy_version" in prov and "cuda_runtime_version" in prov

    w_stale = tmp_path / "stale_v9"
    shutil.copytree(w2, w_stale)
    st = json.loads((w_stale / "state.json").read_text())
    st["provenance"]["gates_version"] = "9"
    write_json_atomic(w_stale / "state.json", st)
    with pytest.raises(ResumeStateError, match="gates version"):
        run_campaign(CampaignConfig(iterations=2, population=2, seed=17, work_dir=w_stale, resume=True))
