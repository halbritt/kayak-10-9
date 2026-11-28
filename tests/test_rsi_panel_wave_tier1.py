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


def test_refused_geometry_raises_panel_wave_refused(tmp_path: Path) -> None:
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
    out = evaluate_tier0_tier1(ref.hull, tmp_path / "pw_bad", bad_pw)
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


def test_panel_wave_guards_stale_ref_drift_resume_and_sanity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    import rsi.evaluator as ev_mod
    import rsi.panel_solver as ps_mod
    from rsi.loop import _summarize_wave_runs, load_campaign_status

    # 1. Stale panel_wave_reference.json is rejected by load_baseline_reference.
    mock_repo = tmp_path / "repo"
    for rel in (
        "hull/kayak-10-9.hull.json",
        "data/kayak-10-9-specs.json",
        "data/rsi-friction-reference.json",
        "data/rsi-seakeeping-reference.json",
        "data/openfoam-cfd-metrics.json",
        "docs/surface-fairing/controls.json",
        "docs/surface-fairing/fairness.json",
        "rsi/panel_wave_reference.json",
    ):
        (mock_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEFAULT_REPO_ROOT / rel, mock_repo / rel)
    pw_path = mock_repo / "rsi" / PANEL_WAVE_REFERENCE_FILE
    pw_raw = json.loads(pw_path.read_text())
    pw_raw["panel_solver_sha256"] = "0" * 64
    pw_path.write_text(json.dumps(pw_raw))
    with pytest.raises(ValueError, match="panel_solver_sha256"):
        load_baseline_reference(mock_repo)

    # 2. Sanity checks reject non-positive rw_N and excessive 3-grid spread.
    monkeypatch.setattr(ps_mod, "static_pose", lambda V, F, vol, trim=0.0: 0.05)
    monkeypatch.setattr(ps_mod, "pose_triangles", lambda V, F, h, trim=0.0: np.zeros((1, 3, 3)))
    V0, F0 = np.zeros((3, 3)), np.zeros((1, 3), dtype=int)
    monkeypatch.setattr(
        ps_mod,
        "solve_fixed",
        lambda tris, u, cfg, **kw: [SimpleNamespace(rw_N=-0.1) for _ in u],
    )
    with pytest.raises(ValueError, match="non-positive rw_N"):
        solve_panel_wave_curve(V0, F0, 225.0, np.array([2.5, 3.0]), backend="numpy")

    monkeypatch.setattr(
        ps_mod,
        "solve_fixed",
        lambda tris, u, cfg, **kw: [SimpleNamespace(rw_N=1.0 if cfg.nx_hull == 45 else 2.2) for _ in u],
    )
    with pytest.raises(ValueError, match="grid family disagreement"):
        solve_panel_wave_curve(V0, F0, 225.0, np.array([2.5, 3.0]), backend="numpy")
    monkeypatch.undo()

    # 3. wave_summary counts GPU fallbacks and panel_wave_refused failures.
    summary = _summarize_wave_runs([
        {"design_hash": "h1", "metrics": {"valid": True, "wave_backend": "cupy"}},
        {"design_hash": "h2", "metrics": {"valid": True, "wave_backend": "cupy+numpy"}},
        {"design_hash": "h3", "metrics": {"valid": False, "wave_backend": None, "error": "failed: panel_wave_refused (lb=225)"}},
    ])
    assert summary == {
        "wave_backend_counts": {"cupy": 1, "cupy+numpy": 1, "none": 1},
        "gpu_fallback_count": 1,
        "panel_wave_refused_count": 1,
    }

    # 4. Resume rejects solver_sha/grid/precision mismatch, and fresh baseline drift raises RuntimeError.
    bad_dir = tmp_path / "drift"
    monkeypatch.setattr(ev_mod, "_cupy_usable", lambda: False)
    with pytest.raises(RuntimeError, match="drifted"):
        with monkeypatch.context() as mp:
            mp.setattr(ps_mod, "static_pose", lambda V, F, vol, trim=0.0: 0.05)
            mp.setattr(ps_mod, "pose_triangles", lambda V, F, h, trim=0.0: np.zeros((1, 3, 3)))
            mp.setattr(ps_mod, "solve_fixed", lambda tris, u, cfg, **kw: [SimpleNamespace(rw_N=1.0 + 0.3 * i) for i, _ in enumerate(u)])
            run_campaign(CampaignConfig(iterations=1, population=1, workers=1, work_dir=bad_dir, pre_screen=False))

    # Synthetic checkpoint verifying resume solver/grid/precision refusal.
    res_dir = tmp_path / "res_chk"
    base_res = run_campaign(
        CampaignConfig(iterations=1, population=1, workers=1, seed=9, work_dir=res_dir, pre_screen=False, skip_resistance=True)
    )
    assert base_res.wave_summary["gpu_fallback_count"] == 0
    assert load_campaign_status(res_dir)["wave_summary"]["gpu_fallback_count"] == 0
    st = json.loads((res_dir / "state.json").read_text())
    st["provenance"]["panel_grid_family"] = [[60, 36, 8]]
    write_json_atomic(res_dir / "state.json", st)
    with pytest.raises(ResumeStateError, match="panel_grid_family"):
        run_campaign(CampaignConfig(iterations=2, population=1, seed=9, work_dir=res_dir, resume=True))
