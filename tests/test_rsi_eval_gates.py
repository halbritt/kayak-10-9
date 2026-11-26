"""Unit and integration tests for rsi.evaluator, rsi.gates, and rsi.fidelity."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from kayakgen.model.hull import Hull
import numpy as np
import pytest

from rsi.baseline import load_baseline_reference
from rsi.evaluator import evaluate_tier0_tier1
from rsi.fidelity import (
    CfdMetrics,
    SeakeepingMetrics,
    evaluate_tier2_cfd,
    evaluate_tier2_seakeeping,
    parse_openfoam_case_metrics,
    resolve_openfoam_bashrc,
)
from rsi.gates import evaluate_gates


@pytest.fixture(scope="module")
def baseline_eval() -> tuple:
    ref = load_baseline_reference()
    work_dir = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "tier01_baseline"
    metrics = evaluate_tier0_tier1(ref.hull, work_dir, ref)
    return ref, metrics


def test_tier0_tier1_baseline_exact_match(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    assert metrics.valid is True, metrics.error
    assert metrics.is_watertight is True
    assert metrics.is_volume is True
    assert metrics.z_min_m is not None and abs(metrics.z_min_m) <= 1e-5

    assert metrics.sr_4in == 10.88
    assert metrics.sr_4in_exact is not None
    assert abs(metrics.sr_4in_exact - ref.sr_4in_exact) < 1e-4
    assert metrics.usca_4in_pct == 9.15
    assert metrics.l4_m is not None and metrics.l4_m >= 5.442

    assert metrics.kn_225 == {40: 4.47, 60: 5.08, 90: 5.67, 120: 6.17}
    assert metrics.transition_dB_per_20cm == ref.transition_dB_per_20cm
    assert metrics.catch_widths_mm is not None and len(metrics.catch_widths_mm) == 6
    assert abs(metrics.catch_widths_mm[0] - ref.catch_widths_mm[0]) < 1e-3

    assert metrics.loads is not None
    for lb, base_load in ref.loads.items():
        cand_load = metrics.loads[lb]
        assert abs(cand_load.gm_cm - base_load.gm_cm) < 0.01
        for actual_gz, exp_gz in zip(cand_load.gz_mm, base_load.gz_mm):
            assert abs(actual_gz - exp_gz) < 1e-3

    assert metrics.has_surface_fairing is True
    assert metrics.fwd_wl_max_turn_deg is not None and metrics.fwd_wl_max_turn_deg < 0.0
    assert metrics.entry_zone_max_turn_deg is not None and metrics.entry_zone_max_turn_deg < 0.03
    assert metrics.buttock_0075_peak_deg is not None and metrics.buttock_0075_peak_deg < 3.50
    assert (
        metrics.forebody_max_keel_groove_mm is not None
        and metrics.forebody_max_keel_groove_mm < 1e-4
    )
    assert metrics.forebody_min_k is not None and metrics.forebody_min_k > 0.0
    assert metrics.forebody_max_abs_dk_ds is not None and metrics.forebody_max_abs_dk_ds < 210.0

    encoded = json.dumps(metrics.to_dict(), allow_nan=False)
    assert json.loads(encoded)["sr_4in"] == 10.88


def test_tier0_tier1_degenerate_candidate_returns_invalid() -> None:
    ref = load_baseline_reference()
    data = ref.hull.model_dump()
    # Proportionally shrink waterline, sheer_plan, and draft so Hull validates
    # but 225 lb displacement cannot converge cleanly
    wl = data["distribution_v2"]["waterline_half_breadth"]["coefficients"]
    data["distribution_v2"]["waterline_half_breadth"]["coefficients"] = [c * 0.25 for c in wl]
    dp = data["distribution_v2"]["draft_profile"]["coefficients"]
    data["distribution_v2"]["draft_profile"]["coefficients"] = [c * 0.25 for c in dp]
    data["distribution_v2"]["topsides"]["sheer_plan"]["max_half_breadth_m"] *= 0.25
    data["distribution_v2"]["topsides"]["shoulder_round_m"] *= 0.25
    tiny_hull = Hull.model_validate(data)
    work_dir = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "tier01_invalid"
    bad_metrics = evaluate_tier0_tier1(tiny_hull, work_dir, ref, skip_resistance=True)
    assert bad_metrics.valid is False
    assert bad_metrics.error is not None
    report = evaluate_gates(bad_metrics, ref)
    assert report.passed is False
    assert any(v.gate == "valid" for v in report.violations)


def test_gates_pass_on_baseline_and_fail_on_regressions(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    ok_report = evaluate_gates(metrics, ref)
    assert ok_report.passed is True, ok_report.violations
    assert ok_report.violations == ()
    assert json.loads(json.dumps(ok_report.to_dict(), allow_nan=False))["passed"] is True

    # 1. SR_4in > 10.99
    bad_sr = replace(metrics, sr_4in=11.02, sr_4in_exact=11.02)
    r_sr = evaluate_gates(bad_sr, ref)
    assert r_sr.passed is False
    assert any(v.gate == "sr_4in_max" for v in r_sr.violations)

    # 2. USCA < 8.50%
    bad_usca = replace(metrics, usca_4in_pct=8.40, usca_4in_pct_exact=8.40)
    r_usca = evaluate_gates(bad_usca, ref)
    assert r_usca.passed is False
    assert any(v.gate == "usca_4in_pct_min" for v in r_usca.violations)

    # 3. Negative fairness (forward WL hollow, buttock kink, negative curvature)
    bad_fair = replace(
        metrics,
        fwd_wl_max_turn_deg=0.05,
        buttock_0075_peak_deg=4.10,
        forebody_min_k=-0.10,
    )
    r_fair = evaluate_gates(bad_fair, ref)
    assert r_fair.passed is False
    fair_gates = {v.gate for v in r_fair.violations}
    assert {"fwd_wl_turn_deg", "buttock_0075_peak_deg", "forebody_k_min"} <= fair_gates

    # 4. Lost stability
    assert metrics.loads is not None
    degraded_loads = dict(metrics.loads)
    l225 = degraded_loads[225]
    degraded_loads[225] = replace(l225, gz_mm=(l225.gz_mm[0] - 0.5, l225.gz_mm[1], l225.gz_mm[2]))
    bad_stab = replace(metrics, loads=degraded_loads)
    r_stab = evaluate_gates(bad_stab, ref)
    assert r_stab.passed is False
    assert any(v.gate == "gz_225_10_mm" for v in r_stab.violations)

    # 5. Non-monotonic transition
    bad_trans = replace(metrics, transition_dB_per_20cm=(3.4, 3.1, -0.2, 2.7, 2.1, 1.9, 1.5, 1.0))
    r_trans = evaluate_gates(bad_trans, ref)
    assert r_trans.passed is False
    assert any(v.gate == "transition_monotonic" for v in r_trans.violations)


def test_tier2_capytaine_quick_smoke_on_baseline(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    work_dir = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "capytaine_smoke"
    seak = evaluate_tier2_seakeeping(ref.hull, work_dir, ref, quick=True, n_jobs=1)

    assert seak.valid is True, seak.error
    assert seak.roll_period_s is not None and abs(seak.roll_period_s - 2.07) <= 0.05
    assert seak.gm_t_cm is not None and abs(seak.gm_t_cm - 4.09) <= 0.05
    assert Path(seak.output_json_path).is_file()
    assert len(seak.rows) > 0

    report = evaluate_gates(metrics, ref, seakeeping=seak)
    assert report.passed is True, report.violations


def test_openfoam_staging_dry_run_and_synthetic_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = load_baseline_reference()
    custom_bashrc = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "fake_bashrc"
    monkeypatch.setenv("OPENFOAM_BASHRC", str(custom_bashrc))
    assert resolve_openfoam_bashrc() == custom_bashrc
    monkeypatch.delenv("OPENFOAM_BASHRC", raising=False)
    assert resolve_openfoam_bashrc().name == "bashrc"

    work_dir = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "openfoam_smoke"
    cfd_dry = evaluate_tier2_cfd(ref.hull, work_dir, ref, dry_run=True)
    assert cfd_dry.valid is True, cfd_dry.error
    assert cfd_dry.dry_run is True
    assert cfd_dry.posed is not None
    assert (work_dir / "candidate.stl").is_file()
    assert (work_dir / "mesh_candidate" / "system" / "snappyHexMeshDict").is_file()
    assert (work_dir / "run_candidate_5p5kn" / "system" / "controlDict").is_file()
    assert (work_dir / "run_candidate_6p2kn" / "system" / "controlDict").is_file()
    assert (work_dir / "cfd_manifest.json").is_file()

    # Synthetic force.dat and moment.dat verification (half-hull doubled by _parse_forces)
    case_55 = work_dir / "run_candidate_5p5kn"
    forces_dir = case_55 / "postProcessing" / "forces" / "0"
    forces_dir.mkdir(parents=True, exist_ok=True)
    (forces_dir / "force.dat").write_text(
        "# Time (total_x total_y total_z) (p_x p_y p_z) (v_x v_y v_z)\n"
        "400 (10.0 0.0 440.0) (2.0 0.0 440.0) (8.0 0.0 0.0)\n"
        "500 (10.2 0.0 440.0) (2.1 0.0 440.0) (8.1 0.0 0.0)\n"
    )
    (forces_dir / "moment.dat").write_text(
        "# Time (total_x total_y total_z) (p_x p_y p_z) (v_x v_y v_z)\n"
        "400 (0.0 38.0 0.0) (0.0 38.0 0.0) (0.0 0.0 0.0)\n"
        "500 (0.0 39.0 0.0) (0.0 39.0 0.0) (0.0 0.0 0.0)\n"
    )
    parsed = parse_openfoam_case_metrics(case_55, cfd_dry.posed, 5.5)
    assert abs(parsed["drag_pressure_N"] - 4.1) < 1e-3
    assert abs(parsed["drag_viscous_N"] - 16.1) < 1e-3
    assert abs(parsed["drag_total_N"] - 20.2) < 1e-3

    # Verify CFD gate rejects dry_run=True, passes on improved drag across all speeds, and fails on regressed drag
    metrics_dummy = evaluate_tier0_tier1(
        ref.hull,
        Path.home() / ".cache" / "kayak-10-9" / "test-work" / "tier01_for_cfd",
        ref,
        skip_resistance=True,
    )
    r_dry = evaluate_gates(metrics_dummy, ref, cfd=cfd_dry)
    assert r_dry.passed is False
    assert any(v.gate == "cfd_dry_run" for v in r_dry.violations)

    parsed_62 = {
        **parsed,
        "speed_kn": 6.2,
        "drag_total_N": ref.openfoam.speed_6_2kn.drag_total_N - 0.1,
    }
    cfd_ok = CfdMetrics(
        valid=True,
        error=None,
        dry_run=False,
        elapsed_s=1.0,
        bashrc=str(resolve_openfoam_bashrc()),
        bashrc_exists=True,
        staging_dir=str(work_dir),
        posed=cfd_dry.posed,
        speeds={"5.5kn": parsed, "6.2kn": parsed_62},
    )
    assert evaluate_gates(metrics_dummy, ref, cfd=cfd_ok).passed is True

    cfd_bad = replace(
        cfd_ok,
        speeds={
            "5.5kn": {**parsed, "drag_total_N": ref.openfoam.speed_5_5kn.drag_total_N + 0.5},
            "6.2kn": parsed_62,
        },
    )
    r_cfd_bad = evaluate_gates(metrics_dummy, ref, cfd=cfd_bad)
    assert r_cfd_bad.passed is False
    assert any(v.gate == "cfd_drag_5.5kn" for v in r_cfd_bad.violations)


def test_gates_fail_closed_on_missing_or_incomplete_metrics(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval

    # 1. skip_resistance=True still allows kn_225=None
    skip_ok = evaluate_gates(replace(metrics, skip_resistance=True, kn_225=None, kn_225_exact=None), ref)
    assert skip_ok.passed is True

    # 2. skip_resistance=False with kn_225=None fails
    r_none_kn = evaluate_gates(replace(metrics, skip_resistance=False, kn_225=None, kn_225_exact=None), ref)
    assert r_none_kn.passed is False
    assert any(v.gate == "kn_225_missing" for v in r_none_kn.violations)

    # 3. skip_resistance=False with partial kn_225 fails cleanly (no KeyError)
    r_part_kn = evaluate_gates(replace(metrics, skip_resistance=False, kn_225={40: 4.50}), ref)
    assert r_part_kn.passed is False
    assert {f"kn_225_{w}w_missing" for w in (60, 90, 120)} <= {v.gate for v in r_part_kn.violations}

    # 4. Tier 2 Seakeeping quick=False and quick=True with empty rows fail
    seak_full_empty = SeakeepingMetrics(
        valid=True,
        error=None,
        quick=False,
        elapsed_s=1.0,
        gm_t_cm=ref.capytaine.gm_t_cm,
        rows=(),
    )
    r_seak_full = evaluate_gates(metrics, ref, seakeeping=seak_full_empty)
    assert r_seak_full.passed is False
    assert len(r_seak_full.violations) == len(ref.capytaine.target_conditions_225)

    seak_quick_empty = SeakeepingMetrics(
        valid=True,
        error=None,
        quick=True,
        elapsed_s=1.0,
        gm_t_cm=ref.capytaine.gm_t_cm,
        rows=(),
    )
    r_seak_quick = evaluate_gates(metrics, ref, seakeeping=seak_quick_empty)
    assert r_seak_quick.passed is False
    assert {v.gate for v in r_seak_quick.violations} == {
        "seak_missing_4.5kn_head_bay",
        "seak_missing_4.5kn_head_steep",
    }

    # 5. Tier 2 CFD empty speeds, invalid+dry_run, and valid+dry_run all fail
    cfd_empty = CfdMetrics(
        valid=True,
        error=None,
        dry_run=False,
        elapsed_s=1.0,
        bashrc="bashrc",
        bashrc_exists=True,
        staging_dir="staging",
        speeds={},
    )
    r_cfd_empty = evaluate_gates(metrics, ref, cfd=cfd_empty)
    assert r_cfd_empty.passed is False
    assert {"cfd_missing_5.5kn", "cfd_missing_6.2kn"} <= {v.gate for v in r_cfd_empty.violations}

    cfd_invalid_dry = CfdMetrics(
        valid=False,
        error="staging failed",
        dry_run=True,
        elapsed_s=1.0,
        bashrc="bashrc",
        bashrc_exists=True,
        staging_dir="staging",
    )
    r_cfd_inv_dry = evaluate_gates(metrics, ref, cfd=cfd_invalid_dry)
    assert r_cfd_inv_dry.passed is False
    assert {"cfd_valid", "cfd_dry_run"} <= {v.gate for v in r_cfd_inv_dry.violations}

    cfd_valid_dry = CfdMetrics(
        valid=True,
        error=None,
        dry_run=True,
        elapsed_s=1.0,
        bashrc="bashrc",
        bashrc_exists=True,
        staging_dir="staging",
        speeds={},
    )
    r_cfd_val_dry = evaluate_gates(metrics, ref, cfd=cfd_valid_dry)
    assert r_cfd_val_dry.passed is False
    assert any(v.gate == "cfd_dry_run" for v in r_cfd_val_dry.violations)


def test_parse_openfoam_case_metrics_mismatched_force_and_moment_rows() -> None:
    work_dir = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "openfoam_mismatch"
    forces_dir = work_dir / "postProcessing" / "forces" / "0"
    forces_dir.mkdir(parents=True, exist_ok=True)
    (forces_dir / "force.dat").write_text(
        "# Time (total_x total_y total_z) (p_x p_y p_z) (v_x v_y v_z)\n"
        "300 (9.5 0.0 440.0) (2.0 0.0 440.0) (7.5 0.0 0.0)\n"
        "400 (10.0 0.0 440.0) (2.0 0.0 440.0) (8.0 0.0 0.0)\n"
        "500 (10.2 0.0 440.0) (2.1 0.0 440.0) (8.1 0.0 0.0)\n"
    )
    (forces_dir / "moment.dat").write_text(
        "# Time (total_x total_y total_z) (p_x p_y p_z) (v_x v_y v_z)\n"
        "100 (0.0 30.0 0.0) (0.0 30.0 0.0) (0.0 0.0 0.0)\n"
        "380 (0.0 37.0 0.0) (0.0 37.0 0.0) (0.0 0.0 0.0)\n"
        "400 (0.0 38.0 0.0) (0.0 38.0 0.0) (0.0 0.0 0.0)\n"
        "500 (0.0 39.0 0.0) (0.0 39.0 0.0) (0.0 0.0 0.0)\n"
    )
    parsed = parse_openfoam_case_metrics(work_dir, {"gm_l_m": 25.0}, 5.5)
    assert abs(parsed["drag_total_N"] - 20.2) < 1e-3
    assert abs(parsed["pitch_moment_My_Nm"] - 76.0) < 1e-3


def test_gates_fail_closed_on_nonfinite_tier1_and_none_rows(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    assert metrics.loads is not None
    nan = float("nan")
    pos_inf = float("inf")
    neg_inf = float("-inf")

    cases: list[tuple[str, object]] = [
        ("sr_4in_max", replace(metrics, sr_4in=nan)),
        ("sr_4in_max", replace(metrics, sr_4in_exact=nan)),
        ("sr_4in_max", replace(metrics, sr_4in=neg_inf, sr_4in_exact=neg_inf)),
        ("sr_4in_max", replace(metrics, sr_4in=pos_inf)),
        ("usca_4in_pct_min", replace(metrics, usca_4in_pct=nan)),
        ("usca_4in_pct_min", replace(metrics, usca_4in_pct_exact=nan)),
        ("usca_4in_pct_min", replace(metrics, usca_4in_pct=pos_inf, usca_4in_pct_exact=pos_inf)),
        ("usca_4in_pct_min", replace(metrics, usca_4in_pct=neg_inf)),
        ("transition_monotonic", replace(metrics, transition_dB_per_20cm=(0.5, nan, 0.4))),
        ("transition_monotonic", replace(metrics, transition_dB_per_20cm=(0.5, pos_inf, 0.4))),
        ("transition_monotonic", replace(metrics, transition_dB_per_20cm=(0.5, neg_inf, 0.4))),
        ("z_min_m", replace(metrics, z_min_m=nan)),
        ("z_min_m", replace(metrics, z_min_m=pos_inf)),
        ("l4_m_min", replace(metrics, l4_m=nan)),
        ("l4_m_min", replace(metrics, l4_m=pos_inf)),
        ("kayakgen_waterline_hollow", replace(metrics, waterline_hollow_max_deg=nan)),
        ("kayakgen_waterline_hollow", replace(metrics, waterline_hollow_max_deg=neg_inf)),
        ("fwd_wl_turn_deg", replace(metrics, fwd_wl_max_turn_deg=nan)),
        ("fwd_wl_turn_deg", replace(metrics, fwd_wl_max_turn_deg=neg_inf)),
        ("entry_zone_turn_deg", replace(metrics, entry_zone_max_turn_deg=nan)),
        ("entry_zone_turn_deg", replace(metrics, entry_zone_max_turn_deg=neg_inf)),
        ("buttock_0075_peak_deg", replace(metrics, buttock_0075_peak_deg=nan)),
        ("buttock_0075_peak_deg", replace(metrics, buttock_0075_peak_deg=neg_inf)),
        ("forebody_keel_groove_mm", replace(metrics, forebody_max_keel_groove_mm=nan)),
        ("forebody_keel_groove_mm", replace(metrics, forebody_max_keel_groove_mm=neg_inf)),
        ("forebody_k_min", replace(metrics, forebody_min_k=nan)),
        ("forebody_k_min", replace(metrics, forebody_min_k=pos_inf)),
        ("forebody_dk_ds_max", replace(metrics, forebody_max_abs_dk_ds=nan)),
        ("forebody_dk_ds_max", replace(metrics, forebody_max_abs_dk_ds=neg_inf)),
    ]
    for lb in (225, 265, 300):
        base_load = metrics.loads[lb]
        for bad_gm in (nan, pos_inf, neg_inf):
            cases.append(
                (f"gm_{lb}_cm", replace(metrics, loads={**metrics.loads, lb: replace(base_load, gm_cm=bad_gm)}))
            )
        for idx, deg in enumerate((10, 20, 30)):
            for bad_gz in (nan, pos_inf, neg_inf):
                gz = list(base_load.gz_mm)
                gz[idx] = bad_gz
                cases.append(
                    (
                        f"gz_{lb}_{deg}_mm",
                        replace(metrics, loads={**metrics.loads, lb: replace(base_load, gz_mm=tuple(gz))}),
                    )
                )

    for expected_gate, bad_metrics in cases:
        rep = evaluate_gates(bad_metrics, ref)
        assert rep.passed is False, f"{expected_gate} failed open"
        assert any(v.gate == expected_gate for v in rep.violations), (
            f"Expected violation {expected_gate}, got {[v.gate for v in rep.violations]}"
        )
        encoded = json.dumps(rep.to_dict(), allow_nan=False)
        assert json.loads(encoded)["passed"] is False

    for quick_flag in (True, False):
        seak_none = SeakeepingMetrics(
            valid=True,
            error=None,
            quick=quick_flag,
            elapsed_s=1.0,
            gm_t_cm=ref.capytaine.gm_t_cm,
            rows=None,  # type: ignore[arg-type]
        )
        assert seak_none.to_dict()["rows"] == []
        rep_none = evaluate_gates(metrics, ref, seakeeping=seak_none)
        assert rep_none.passed is False
        assert any(v.gate.startswith("seak_missing_") for v in rep_none.violations)
        json.dumps(rep_none.to_dict(), allow_nan=False)


def test_gates_fail_closed_on_malformed_tier2_and_numpy_bool(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    assert metrics.loads is not None

    # 1. Malformed SeakeepingMetrics.rows entries fail closed with strict JSON
    malformed_rows_cases: list[tuple[object, ...]] = [
        ({},),
        (None,),
        ({"speed_kn": None, "heading": "head", "sea": "Bay chop (Hs 0.35m, Tp 2.5s)"},),
        ({"speed_kn": "bad", "heading": "head", "sea": "Bay chop (Hs 0.35m, Tp 2.5s)"},),
        ({"speed_kn": 4.5, "heading": "", "sea": "Bay chop (Hs 0.35m, Tp 2.5s)"},),
        ({"speed_kn": np.bool_(True), "heading": "head", "sea": "Bay chop (Hs 0.35m, Tp 2.5s)"},),
    ]
    for bad_rows in malformed_rows_cases:
        seak_bad = SeakeepingMetrics(
            valid=True,
            error=None,
            quick=True,
            elapsed_s=1.0,
            gm_t_cm=ref.capytaine.gm_t_cm,
            rows=bad_rows,  # type: ignore[arg-type]
        )
        rep = evaluate_gates(metrics, ref, seakeeping=seak_bad)
        assert rep.passed is False
        assert any(v.gate == "seak_malformed_row" for v in rep.violations)
        assert json.loads(json.dumps(rep.to_dict(), allow_nan=False))["passed"] is False

    # Extra malformed row alongside valid required rows still fails closed
    valid_quick_rows = tuple(
        {
            "speed_kn": r.speed_kn,
            "heading": r.heading,
            "sea": r.sea,
            "bow_dips_per_min": r.bow_dips_per_min,
            "seat_acc_rms_ms2": r.seat_acc_rms_ms2,
        }
        for r in ref.capytaine.target_conditions_225
        if r.speed_kn == 4.5 and r.heading == "head"
    )
    seak_extra_bad = SeakeepingMetrics(
        valid=True,
        error=None,
        quick=True,
        elapsed_s=1.0,
        gm_t_cm=ref.capytaine.gm_t_cm,
        rows=valid_quick_rows + ({},),  # type: ignore[arg-type]
    )
    rep_extra_bad = evaluate_gates(metrics, ref, seakeeping=seak_extra_bad)
    assert rep_extra_bad.passed is False
    assert any(v.gate == "seak_malformed_row" for v in rep_extra_bad.violations)
    json.dumps(rep_extra_bad.to_dict(), allow_nan=False)

    # 2. Non-mapping CfdMetrics.speeds or speed entries fail closed with strict JSON
    for bad_speeds in (
        {"5.5kn": 20.0, "6.2kn": 25.0},
        {"5.5kn": ["drag_total_N"]},
        ["5.5kn", "6.2kn"],
    ):
        cfd_bad = CfdMetrics(
            valid=True,
            error=None,
            dry_run=False,
            elapsed_s=1.0,
            bashrc="bashrc",
            bashrc_exists=True,
            staging_dir="staging",
            speeds=bad_speeds,  # type: ignore[arg-type]
        )
        rep_cfd = evaluate_gates(metrics, ref, cfd=cfd_bad)
        assert rep_cfd.passed is False
        assert any(v.gate.startswith("cfd_missing_") for v in rep_cfd.violations)
        assert json.loads(json.dumps(rep_cfd.to_dict(), allow_nan=False))["passed"] is False

    # 3. np.bool_(True) rejected in Tier 1 and Tier 2 numeric metric fields
    nb = np.bool_(True)
    base_load = metrics.loads[225]
    np_bool_cases: list[tuple[str, object]] = [
        ("z_min_m", replace(metrics, z_min_m=nb)),  # type: ignore[arg-type]
        ("sr_4in_max", replace(metrics, sr_4in=nb, sr_4in_exact=nb)),  # type: ignore[arg-type]
        ("usca_4in_pct_min", replace(metrics, usca_4in_pct=nb, usca_4in_pct_exact=nb)),  # type: ignore[arg-type]
        ("l4_m_min", replace(metrics, l4_m=nb)),  # type: ignore[arg-type]
        ("transition_monotonic", replace(metrics, transition_dB_per_20cm=(0.5, nb, 0.4))),  # type: ignore[arg-type]
        ("kayakgen_waterline_hollow", replace(metrics, waterline_hollow_max_deg=nb)),  # type: ignore[arg-type]
        ("fwd_wl_turn_deg", replace(metrics, fwd_wl_max_turn_deg=nb)),  # type: ignore[arg-type]
        ("entry_zone_turn_deg", replace(metrics, entry_zone_max_turn_deg=nb)),  # type: ignore[arg-type]
        ("buttock_0075_peak_deg", replace(metrics, buttock_0075_peak_deg=nb)),  # type: ignore[arg-type]
        ("forebody_keel_groove_mm", replace(metrics, forebody_max_keel_groove_mm=nb)),  # type: ignore[arg-type]
        ("forebody_k_min", replace(metrics, forebody_min_k=nb)),  # type: ignore[arg-type]
        ("forebody_dk_ds_max", replace(metrics, forebody_max_abs_dk_ds=nb)),  # type: ignore[arg-type]
        (
            "gm_225_cm",
            replace(metrics, loads={**metrics.loads, 225: replace(base_load, gm_cm=nb)}),  # type: ignore[arg-type]
        ),
        (
            "gz_225_10_mm",
            replace(
                metrics,
                loads={
                    **metrics.loads,
                    225: replace(base_load, gz_mm=(nb, base_load.gz_mm[1], base_load.gz_mm[2])),  # type: ignore[arg-type]
                },
            ),
        ),
        (
            "kn_225_40w_missing",
            replace(metrics, skip_resistance=False, kn_225={**metrics.kn_225, 40: nb}),  # type: ignore[dict-item]
        ),
    ]
    for expected_gate, bad_metrics in np_bool_cases:
        rep_nb = evaluate_gates(bad_metrics, ref)  # type: ignore[arg-type]
        assert rep_nb.passed is False, f"{expected_gate} accepted np.bool_(True)"
        assert any(v.gate == expected_gate for v in rep_nb.violations)
        json.dumps(rep_nb.to_dict(), allow_nan=False)

    seak_nb_gm = SeakeepingMetrics(
        valid=True,
        error=None,
        quick=True,
        elapsed_s=1.0,
        gm_t_cm=nb,  # type: ignore[arg-type]
        rows=valid_quick_rows,
    )
    rep_seak_nb = evaluate_gates(metrics, ref, seakeeping=seak_nb_gm)
    assert rep_seak_nb.passed is False
    assert any(v.gate == "seakeeping_gm_t_cm" for v in rep_seak_nb.violations)
    json.dumps(rep_seak_nb.to_dict(), allow_nan=False)

    cfd_nb = CfdMetrics(
        valid=True,
        error=None,
        dry_run=False,
        elapsed_s=1.0,
        bashrc="bashrc",
        bashrc_exists=True,
        staging_dir="staging",
        speeds={"5.5kn": {"drag_total_N": nb}, "6.2kn": {"drag_total_N": 25.0}},
    )
    rep_cfd_nb = evaluate_gates(metrics, ref, cfd=cfd_nb)
    assert rep_cfd_nb.passed is False
    assert any(v.gate == "cfd_drag_5.5kn" for v in rep_cfd_nb.violations)
    json.dumps(rep_cfd_nb.to_dict(), allow_nan=False)


def test_baseline_tolerated_gate_margins_nonnegative_and_foot_clearances_match_specs(
    baseline_eval: tuple,
) -> None:
    ref, metrics = baseline_eval
    assert metrics.foot_clear_height_m is not None
    assert metrics.foot_clear_width_m is not None
    assert abs(metrics.foot_clear_height_m * 100.0 - 26.24) < 0.05
    assert abs(metrics.foot_clear_width_m * 100.0 - 29.95) < 0.05

    rep = evaluate_gates(metrics, ref)
    assert rep.passed is True, rep.violations
    assert all(v >= 0.0 for v in rep.margins.values()), {
        k: v for k, v in rep.margins.items() if v < 0.0
    }

    seak_borderline = SeakeepingMetrics(
        valid=True,
        error=None,
        quick=True,
        elapsed_s=1.0,
        gm_t_cm=ref.capytaine.gm_t_cm - 0.02,
        rows=tuple(
            {
                "speed_kn": r.speed_kn,
                "heading": r.heading,
                "sea": r.sea,
                "bow_dips_per_min": r.bow_dips_per_min,
                "seat_acc_rms_ms2": r.seat_acc_rms_ms2,
            }
            for r in ref.capytaine.target_conditions_225
            if r.speed_kn == 4.5 and r.heading == "head"
        ),
    )
    rep_seak = evaluate_gates(metrics, ref, seakeeping=seak_borderline)
    assert rep_seak.passed is True
    assert rep_seak.margins["seakeeping_gm_t_cm"] >= 0.0


def test_tier2_cache_root_opt_out_and_corrupt_entry_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rsi.fidelity import _load_cfd_cache, _load_seakeeping_cache, resolve_tier2_cache_root

    monkeypatch.setenv("KAYAK_RSI_NO_TIER2_CACHE", "1")
    assert resolve_tier2_cache_root() is None

    monkeypatch.delenv("KAYAK_RSI_NO_TIER2_CACHE", raising=False)
    fake_xdg = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "xdg_cache_unit"
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_xdg))
    root = resolve_tier2_cache_root()
    assert root == fake_xdg / "kayak-10-9" / "tier2-cache"

    bad_entry = root / "seak_corrupt"
    bad_entry.mkdir(parents=True, exist_ok=True)
    (bad_entry / "meta.json").write_text("{corrupt")
    (bad_entry / "capytaine_seakeeping.json").write_text("{}")
    assert _load_seakeeping_cache(root, "seak_corrupt", {"solver": "x"}, quick=False) is None

    bad_cfd = root / "cfd_corrupt"
    bad_cfd.mkdir(parents=True, exist_ok=True)
    (bad_cfd / "meta.json").write_text(json.dumps({"source": "wrong", "key": {"solver": "y"}}))
    (bad_cfd / "openfoam_cfd_metrics.json").write_text(json.dumps({"speeds": {}}))
    assert _load_cfd_cache(root, "cfd_corrupt", {"solver": "y"}, (5.5, 6.2)) is None


def test_michell_fast_curve_matches_cad() -> None:
    from rsi.evaluator import RESISTANCE_SPEEDS_KN, _ensure_cad_on_path, _michell_fast_curve

    _ensure_cad_on_path()
    from hydro import G, KN, RHO_SEAWATER, michell

    rng = np.random.default_rng(7)
    xs = np.linspace(-2.7, 2.7, 401)
    zs = np.linspace(-0.09, 0.0, 81)
    Y = np.abs(rng.normal(0.10, 0.04, size=(401, 81)))
    u_arr = RESISTANCE_SPEEDS_KN * KN
    ref_vals = np.array([michell(xs, zs, Y, float(u), RHO_SEAWATER, 4000) for u in u_arr])
    fast_vals = _michell_fast_curve(xs, zs, Y, u_arr, RHO_SEAWATER, g=G, ntheta=4000)
    assert float(np.max(np.abs(ref_vals - fast_vals))) < 1e-12


def test_solve_bem_full_fast_matches_solve_bem_case_across_headings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib
    import sys
    import trimesh
    from rsi.evaluator import _ensure_cad_on_path
    from rsi.fidelity import _single_thread_blas_env, _solve_bem_full_fast

    _ensure_cad_on_path()
    import run_capytaine_seakeeping as cap_mod
    from deck_geometry import SculptedKayakSurface
    from hydro import LB

    # Include periods on both sides of the 1.90s stern cutoff so heading-dependent filtering is exercised
    monkeypatch.setattr(cap_mod, "PERIODS", np.array([1.20, 2.10, 4.00], dtype=float))

    ref = load_baseline_reference()
    tmp_hull = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "bem_fast_eq.hull.json"
    tmp_hull.parent.mkdir(parents=True, exist_ok=True)
    tmp_hull.write_text(ref.hull.model_dump_json(indent=2) + "\n")
    surf = SculptedKayakSurface(tmp_hull)
    closed = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
    closed.merge_vertices(merge_tex=True, merge_norm=True)
    V = np.asarray(closed.vertices, dtype=float).copy()
    F = np.asarray(closed.faces, dtype=int).copy()
    V[:, 0] = -V[:, 0]
    F = F[:, ::-1]
    if trimesh.Trimesh(V, F, process=False).volume < 0:
        F = F[:, ::-1]
    V[:, 0] -= V[:, 0].min()
    V[:, 1] -= np.median(V[:, 1])
    V[:, 2] -= V[:, 2].min()

    with contextlib.redirect_stdout(sys.stderr), _single_thread_blas_env(True):
        body, I_mat, info = cap_mod.build_bem_body("eq_225", V, F, mass_kg=225.0 * LB, kg_m=0.25)
        ref_out = cap_mod.solve_bem_case(body, I_mat, info, speeds_kn=(0.0, 4.5), n_jobs=4)
        fast_out = _solve_bem_full_fast(body, I_mat, info, speeds_kn=(0.0, 4.5), n_jobs=4)

    ref_sp = cap_mod.evaluate_spectra(ref_out)
    fast_sp = cap_mod.evaluate_spectra(fast_out)
    assert ref_sp["gm_t_cm"] == fast_sp["gm_t_cm"]
    ref_map = {(r["speed_kn"], r["heading"], r["sea"]): r for r in ref_sp["rows"]}
    fast_map = {(r["speed_kn"], r["heading"], r["sea"]): r for r in fast_sp["rows"]}
    assert set(ref_map) == set(fast_map)
    for key, ref_row in ref_map.items():
        for fld in (
            "heave_sig_cm",
            "pitch_sig_deg",
            "roll_sig_deg",
            "bow_dips_per_min",
            "seat_acc_rms_ms2",
            "yaw_mom_rms_Nm",
        ):
            assert ref_row[fld] == fast_row_val(fast_map[key], fld), (key, fld)


def fast_row_val(row: dict[str, object], key: str) -> object:
    return row[key]
