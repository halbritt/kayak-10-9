"""Unit tests for rsi.diagnose and rsi.promote."""

from __future__ import annotations

import atexit
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

from kayakgen.model.hull import Hull
from pydantic import ValidationError
import pytest

from rsi.baseline import load_baseline_reference
from rsi.diagnose import (
    aggregate_diagnostics,
    diagnose_candidate,
    diagnose_validation_error,
)
from rsi.evaluator import evaluate_tier0_tier1
from rsi.fidelity import CfdMetrics, SeakeepingMetrics, resolve_openfoam_bashrc
from rsi.gates import evaluate_gates
from rsi.genome import DesignSpace, vector_to_hull
from rsi.promote import compare_candidate, promote_candidate

_TEST_WORK_PARENT = Path.home() / ".cache" / "kayak-10-9" / "test-work"
_TEST_WORK_PARENT.mkdir(parents=True, exist_ok=True)
TEST_WORK_ROOT = Path(tempfile.mkdtemp(prefix="diagnose_promote_", dir=_TEST_WORK_PARENT))
atexit.register(shutil.rmtree, TEST_WORK_ROOT, ignore_errors=True)


@pytest.fixture(scope="module")
def baseline_eval() -> tuple:
    ref = load_baseline_reference()
    work_dir = TEST_WORK_ROOT / "baseline_eval"
    metrics = evaluate_tier0_tier1(ref.hull, work_dir, ref)
    assert metrics.valid is True, metrics.error
    return ref, metrics


def _synthetic_full_tier2(ref) -> tuple[SeakeepingMetrics, CfdMetrics]:
    rows = tuple(
        {
            "speed_kn": r.speed_kn,
            "heading": r.heading,
            "sea": r.sea,
            "bow_dips_per_min": r.bow_dips_per_min,
            "roll_sig_deg": r.roll_sig_deg,
            "seat_acc_rms_ms2": r.seat_acc_rms_ms2,
            "heave_sig_cm": r.heave_sig_cm,
            "pitch_sig_deg": r.pitch_sig_deg,
        }
        for r in ref.capytaine.rows_225
    )
    seak = SeakeepingMetrics(
        valid=True,
        error=None,
        quick=False,
        elapsed_s=1.0,
        gm_t_cm=ref.capytaine.gm_t_cm,
        roll_period_s=ref.capytaine.roll_period_s,
        rows=rows,
    )
    cfd = CfdMetrics(
        valid=True,
        error=None,
        dry_run=False,
        elapsed_s=1.0,
        bashrc=str(resolve_openfoam_bashrc()),
        bashrc_exists=True,
        staging_dir=str(TEST_WORK_ROOT / "synthetic_cfd"),
        speeds={
            "5.5kn": ref.openfoam.speed_5_5kn.to_dict(),
            "6.2kn": ref.openfoam.speed_6_2kn.to_dict(),
        },
    )
    return seak, cfd


def test_diagnose_validation_error_sheer_plan_and_ramp_order() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()

    vec_sheer = space.baseline_vector()
    vec_sheer[idx["sheer_max_hb_m"]] = 0.45
    with pytest.raises(ValidationError) as exc_sheer:
        vector_to_hull(vec_sheer, ref.hull, space, clamp=False)

    rep_sheer = diagnose_validation_error(exc_sheer.value, vector=vec_sheer, space=space)
    assert rep_sheer.feasible is False
    assert "validation_sheer_plan_coupled" in rep_sheer.failure_modes
    assert "topsides" in rep_sheer.primary_groups
    assert "waterline" in rep_sheer.primary_groups
    assert "sheer_max_hb_m" in rep_sheer.hints.avoid_bounds
    assert rep_sheer.hints.group_scales["topsides"] < 1.0
    json.dumps(rep_sheer.to_dict(), allow_nan=False)

    vec_ramp = space.baseline_vector()
    vec_ramp[idx["sec_side_fwd_xi"]] = 0.5
    vec_ramp[idx["sec_side_aft_xi"]] = 0.1
    with pytest.raises(ValidationError) as exc_ramp:
        vector_to_hull(vec_ramp, ref.hull, space, clamp=False)

    rep_ramp = diagnose_validation_error(exc_ramp.value, vector=vec_ramp, space=space)
    assert rep_ramp.feasible is False
    assert "validation_ramp_order" in rep_ramp.failure_modes
    assert rep_ramp.primary_groups[0] == "section"
    assert "sec_side_fwd_xi" in rep_ramp.hints.avoid_bounds
    assert "sec_side_aft_xi" in rep_ramp.hints.avoid_bounds
    json.dumps(rep_ramp.to_dict(), allow_nan=False)


def test_diagnose_draft_a1_z_min_and_fairness_violations(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    space = DesignSpace.default()
    idx = space.index_by_name()
    vec = space.baseline_vector()
    vec[idx["draft_a1"]] = 0.004

    bad = replace(
        metrics,
        z_min_m=-0.002,
        buttock_0075_peak_deg=4.2,
        entry_zone_max_turn_deg=0.08,
        waterline_hollow_max_deg=0.05,
    )
    gate_rep = evaluate_gates(bad, ref)
    assert gate_rep.passed is False

    diag = diagnose_candidate(bad, gate_rep, vector=vec, space=space)
    assert diag.feasible is False
    assert {"keel_datum_shift", "buttock_0075_kink", "waterline_entry_hollow"} <= set(diag.failure_modes)
    assert diag.hints.preferred_directions["draft_a1"] < 0.0
    assert diag.hints.group_scales["draft"] < 1.0
    assert diag.hints.group_scales["rocker"] < 1.0
    assert diag.hints.group_scales["waterline"] < 1.0
    assert "draft_a1" in diag.hints.avoid_bounds
    json.dumps(diag.to_dict(), allow_nan=False)


def test_diagnose_stability_losses_at_the_gated_load_and_aggregate(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    assert metrics.loads is not None
    space = DesignSpace.default()

    # 265 and 300 lb are reported loads since gates version 9: a loss there is not a failure.
    for lb in (265, 300):
        ld = metrics.loads[lb]
        m_rep = replace(metrics, loads={**metrics.loads, lb: replace(ld, gm_cm=ld.gm_cm - 0.35)})
        assert evaluate_gates(m_rep, ref).passed is True

    reports = []
    for lb in (225,):
        loads_bad = dict(metrics.loads)
        ld = loads_bad[lb]
        loads_bad[lb] = replace(
            ld,
            gm_cm=ld.gm_cm - 0.35,
            gz_mm=(ld.gz_mm[0] - 0.4, ld.gz_mm[1] - 0.8, ld.gz_mm[2] - 1.2),
        )
        m_bad = replace(metrics, loads=loads_bad)
        g_bad = evaluate_gates(m_bad, ref)
        rep = diagnose_candidate(m_bad, g_bad, vector=space.baseline_vector(), space=space)
        assert rep.feasible is False
        assert f"stability_loss_{lb}" in rep.failure_modes
        assert rep.hints.preferred_directions["wl_a0"] > 0.0
        assert rep.hints.preferred_directions["target_beam_m"] > 0.0
        assert rep.hints.preferred_directions["sheer_max_hb_m"] > 0.0
        reports.append(rep)

    merged = aggregate_diagnostics(reports, space)
    assert merged.preferred_directions["wl_a0"] > 0.0
    assert merged.preferred_directions["target_beam_m"] > 0.0
    assert merged.preferred_directions["sheer_max_hb_m"] > 0.0
    json.dumps(merged.to_dict(), allow_nan=False)


def test_diagnose_tier2_seakeeping_and_cfd_regressions(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    seak, cfd = _synthetic_full_tier2(ref)

    bad_rows = tuple(
        {
            **r,
            "bow_dips_per_min": r["bow_dips_per_min"] * 1.5 + 2.0,
            "seat_acc_rms_ms2": r["seat_acc_rms_ms2"] * 1.4 + 0.3,
        }
        for r in seak.rows
    )
    seak_bad = replace(seak, rows=bad_rows)
    cfd_bad = replace(
        cfd,
        speeds={
            **cfd.speeds,
            "5.5kn": {**cfd.speeds["5.5kn"], "drag_total_N": ref.openfoam.speed_5_5kn.drag_total_N + 1.2},
        },
    )
    gate_rep = evaluate_gates(metrics, ref, seakeeping=seak_bad, cfd=cfd_bad)
    assert gate_rep.passed is False

    diag = diagnose_candidate(metrics, gate_rep, seakeeping=seak_bad, cfd=cfd_bad)
    assert {"seakeeping_regression", "speed_or_drag_regression"} <= set(diag.failure_modes)
    assert "topsides" in diag.primary_groups
    assert diag.hints.preferred_directions["deck_fb_fwd_scale"] > 0.0
    json.dumps(diag.to_dict(), allow_nan=False)


def test_compare_candidate_baseline_and_deltas(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    seak, cfd = _synthetic_full_tier2(ref)

    comp_base = compare_candidate(
        ref.hull,
        ref,
        TEST_WORK_ROOT / "compare_base",
        candidate_path=Path("hull/kayak-10-9.hull.json"),
        tier2=True,
        metrics=metrics,
        seakeeping=seak,
        cfd=cfd,
    )
    assert comp_base.passed is True
    assert comp_base.tier01_passed is True
    assert comp_base.tier2_complete is True
    assert comp_base.tier2_passed is True
    assert comp_base.promotion_eligible is False
    assert comp_base.deltas["class_form"]["sr_4in"] == 0.0
    assert abs(comp_base.deltas["kn_225"]["mean_speed_gain_kn"]) < 1e-3
    fair_deltas = comp_base.deltas["fairness"]
    assert isinstance(fair_deltas["fwd_wl_max_turn_deg"], float)
    wl_keys = (
        "Waterlines 0.05",
        "Waterlines 0.06",
        "Waterlines 0.07",
        "Waterlines 0.08",
        "Waterlines 0.09",
        "Waterlines 0.1016",
    )
    base_fwd_wl_turn = max(
        ref.fairness.cuts[k].max_signed_turn_forward_deg
        for k in wl_keys
        if ref.fairness.cuts[k].max_signed_turn_forward_deg is not None
    )
    assert abs(fair_deltas["fwd_wl_max_turn_deg"]) < 1e-3
    assert fair_deltas["fwd_wl_max_turn_deg"] == pytest.approx(
        metrics.fwd_wl_max_turn_deg - base_fwd_wl_turn,
        abs=1e-6,
    )
    fore_secs = [s for s in ref.fairness.sections if 0.11 <= s.fraction <= 0.35]
    base_fore_groove = max(s.keel_groove_mm for s in fore_secs)
    base_fore_dk = max(
        s.peak_abs_dk_ds_per_m2 for s in fore_secs if s.peak_abs_dk_ds_per_m2 is not None
    )
    assert abs(fair_deltas["forebody_max_keel_groove_mm"]) < 1e-6
    assert fair_deltas["forebody_max_keel_groove_mm"] == pytest.approx(
        metrics.forebody_max_keel_groove_mm - base_fore_groove,
        abs=1e-6,
    )
    assert abs(fair_deltas["forebody_max_abs_dk_ds"]) < 0.05
    assert fair_deltas["forebody_max_abs_dk_ds"] == pytest.approx(
        metrics.forebody_max_abs_dk_ds - base_fore_dk,
        abs=1e-4,
    )

    payload = comp_base.to_dict()
    assert "hydro_3load" in payload
    assert {"load_225", "load_265", "load_300", "kn_225"} <= set(payload["hydro_3load"])
    json.dumps(payload, allow_nan=False)

    # Perturbed failing metrics produce exact deltas and diagnostics
    bad_metrics = replace(
        metrics,
        sr_4in=11.05,
        sr_4in_exact=11.05,
        usca_4in_pct=8.40,
        usca_4in_pct_exact=8.40,
        buttock_0075_peak_deg=4.1,
    )
    comp_bad = compare_candidate(
        ref.hull,
        ref,
        TEST_WORK_ROOT / "compare_bad",
        metrics=bad_metrics,
    )
    assert comp_bad.passed is False
    assert comp_bad.promotion_eligible is False
    assert comp_bad.deltas["class_form"]["sr_4in"] == pytest.approx(11.05 - ref.sr_4in, abs=1e-4)
    # The Sound Rowers ratio is reported, not gated; the USCA beam and the fairness limit fail.
    assert {v.gate for v in comp_bad.gate_report.violations} == {"usca_4in_pct_min", "buttock_0075_peak_deg"}
    json.dumps(comp_bad.to_dict(), allow_nan=False)


def test_promote_candidate_dry_run_and_apply_strict_gating(baseline_eval: tuple) -> None:
    ref, metrics = baseline_eval
    seak, cfd = _synthetic_full_tier2(ref)

    # Stage an isolated repository copy on SSD3 so the real repo stays untouched
    staged_repo = TEST_WORK_ROOT / "staged_repo"
    if staged_repo.exists():
        shutil.rmtree(staged_repo)
    for sub in ("hull", "data", "docs"):
        shutil.copytree(Path(__file__).resolve().parents[1] / sub, staged_repo / sub)

    target_hull = staged_repo / "hull" / "kayak-10-9.hull.json"
    original_bytes = target_hull.read_bytes()

    space = DesignSpace.default()
    idx = space.index_by_name()
    cand_vec = space.baseline_vector()
    cand_vec[idx["sheer_aft_exp"]] = 2.005
    cand_hull = vector_to_hull(cand_vec, ref.hull, space)
    cand_path = TEST_WORK_ROOT / "cand_promote.hull.json"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.write_text(cand_hull.model_dump_json(indent=2) + "\n")

    # 1. Invalid combination dry_run=True & apply=True
    with pytest.raises(ValueError, match="dry_run=True and apply=True"):
        promote_candidate(
            cand_path,
            TEST_WORK_ROOT / "prom_invalid_flags",
            repo_root=staged_repo,
            dry_run=True,
            apply=True,
            metrics=metrics,
        )

    # 2. Dry-run failing Tier 0/1 gate
    failing_metrics = replace(metrics, usca_4in_pct=8.40, usca_4in_pct_exact=8.40)
    rep_fail_dry = promote_candidate(
        cand_path,
        TEST_WORK_ROOT / "prom_fail_dry",
        repo_root=staged_repo,
        dry_run=True,
        apply=False,
        metrics=failing_metrics,
    )
    assert rep_fail_dry.passed is False
    assert rep_fail_dry.applied is False
    assert target_hull.read_bytes() == original_bytes

    # 3. Dry-run passing Tier 0/1 without Tier 2
    rep_ok_dry = promote_candidate(
        cand_path,
        TEST_WORK_ROOT / "prom_ok_dry",
        repo_root=staged_repo,
        dry_run=True,
        apply=False,
        metrics=metrics,
    )
    assert rep_ok_dry.passed is True
    assert rep_ok_dry.applied is False
    assert rep_ok_dry.promotion_eligible is False
    assert target_hull.read_bytes() == original_bytes

    # 4. --apply refused without Tier 2, with quick=True Capytaine, and with dry_run=True CFD
    for tag, sk, cf in (
        ("no_tier2", None, None),
        ("quick_seak", replace(seak, quick=True), cfd),
        ("dry_cfd", seak, replace(cfd, dry_run=True)),
        ("passing_cold_screen", seak, cfd),
        (
            "bad_cfd",
            seak,
            replace(
                cfd,
                speeds={
                    **cfd.speeds,
                    "5.5kn": {**cfd.speeds["5.5kn"], "drag_total_N": ref.openfoam.speed_5_5kn.drag_total_N + 1.0},
                },
            ),
        ),
    ):
        rep_refused = promote_candidate(
            cand_path,
            TEST_WORK_ROOT / f"prom_refused_{tag}",
            repo_root=staged_repo,
            dry_run=False,
            apply=True,
            metrics=metrics,
            seakeeping=sk,
            cfd=cf,
        )
        assert rep_refused.applied is False, tag
        assert rep_refused.passed is False, tag
        assert target_hull.read_bytes() == original_bytes, tag

    # 5. Only executed paired CFD can qualify a passing candidate for promotion.
    prom_work = TEST_WORK_ROOT / "prom_apply_ok"
    rep_applied = promote_candidate(
        cand_path,
        prom_work,
        repo_root=staged_repo,
        dry_run=False,
        apply=True,
        metrics=metrics,
        seakeeping=seak,
        cfd=_paired_cfd(0.0, 0.0),
    )
    assert rep_applied.applied is True
    assert rep_applied.passed is True
    assert rep_applied.promotion_eligible is True
    assert rep_applied.verified_design_hash == cand_hull.design_hash()
    assert rep_applied.backup_path is not None and Path(rep_applied.backup_path).is_file()
    reloaded = Hull.model_validate_json(target_hull.read_text())
    assert reloaded.design_hash() == cand_hull.design_hash()
    json.dumps(rep_applied.to_dict(), allow_nan=False)


def test_augment_metrics_dict_and_evaluate_subdir_tier2_cache_lookup(
    baseline_eval: tuple,
) -> None:
    from rsi.promote import _lookup_verified_tier2_cache, augment_metrics_dict

    ref, metrics = baseline_eval
    aug = augment_metrics_dict(metrics)
    for key in (
        "max_signed_turn_forward_deg",
        "buttock_075_max_turn_change_deg",
        "keel_groove_mm",
        "bilge_k_min",
        "bilge_dk_ds_max",
        "foot_clear_height_cm",
        "foot_clear_width_at_25cm",
        "b_4in_cm",
        "watertight",
    ):
        assert key in aug and aug[key] is not None
    assert aug["foot_clear_height_cm"] >= 25.0
    assert aug["foot_clear_width_at_25cm"] >= 28.0

    # Verify _lookup_verified_tier2_cache recognizes evaluate/<hash16>/ layout
    seak, cfd = _synthetic_full_tier2(ref)
    d_hash = ref.hull.design_hash()
    hash16 = d_hash.split(":")[-1][:16]
    eval_work = TEST_WORK_ROOT / "lookup_eval_cache"
    cand_dir = eval_work / "evaluate" / hash16
    (cand_dir / "seakeeping").mkdir(parents=True, exist_ok=True)
    (cand_dir / "cfd").mkdir(parents=True, exist_ok=True)
    (cand_dir / "candidate.hull.json").write_text(ref.hull.model_dump_json(indent=2) + "\n")
    seak_dict = seak.to_dict()
    seak_dict["baja_265"] = {
        "gm_t_cm": ref.capytaine.gm_t_cm_265_baja,
        "roll_period_s": ref.capytaine.roll_period_s_265_baja,
        "rows": [],
    }
    (cand_dir / "seakeeping" / "capytaine_seakeeping.json").write_text(json.dumps(seak_dict))
    (cand_dir / "cfd" / "openfoam_cfd_metrics.json").write_text(
        json.dumps({"posed": {}, "mesh_info": {}, "speeds": cfd.speeds})
    )

    found_seak, found_cfd = _lookup_verified_tier2_cache(
        eval_work,
        d_hash,
        require_full_seakeeping=True,
        require_executed_cfd=True,
    )
    assert found_seak is not None and found_seak.valid is True
    assert found_seak.baja_265 is not None
    assert found_seak.baja_265["roll_period_s"] == pytest.approx(ref.capytaine.roll_period_s_265_baja)
    assert found_cfd is not None and found_cfd.valid is True


def _paired_cfd(delta_55: float, delta_62: float) -> CfdMetrics:
    def sp(v: float, d: float) -> dict[str, float]:
        return {
            "speed_kn": v,
            "drag_delta_N": d,
            "drag_delta_pressure_N": d - 0.05,
            "drag_delta_viscous_N": 0.05,
            "drag_delta_drift_N": 0.0,
            "baseline_drag_N": 18.0,
            "drag_total_N": 18.0 + d,
        }

    return CfdMetrics(
        valid=True,
        error=None,
        dry_run=False,
        elapsed_s=1.0,
        bashrc=str(resolve_openfoam_bashrc()),
        bashrc_exists=True,
        staging_dir="",
        mesh_info={"baseline_drift_N": {"5.5kn": 0.0, "6.2kn": 0.0}},
        speeds={"5.5kn": sp(5.5, delta_55), "6.2kn": sp(6.2, delta_62)},
        mode="paired_warm_start",
    )


def test_cached_cfd_of_one_kind_never_stands_in_for_the_other(baseline_eval: tuple, tmp_path: Path) -> None:
    from rsi.promote import _lookup_verified_tier2_cache

    ref, _ = baseline_eval
    d_hash = ref.hull.design_hash()
    cand_dir = tmp_path / "evaluate" / d_hash.split(":")[-1][:16]
    (cand_dir / "cfd").mkdir(parents=True)
    (cand_dir / "candidate.hull.json").write_text(ref.hull.model_dump_json(indent=2) + "\n")
    paired = _paired_cfd(+0.20, +0.20)
    cfd_file = cand_dir / "cfd" / "openfoam_cfd_metrics.json"
    cfd_file.write_text(
        json.dumps({"mode": paired.mode, "posed": {}, "mesh_info": {"reference": "cfdref_old"}, "speeds": paired.speeds})
    )

    def lookup(mode: str | None, reference: str | None = None) -> CfdMetrics | None:
        return _lookup_verified_tier2_cache(
            tmp_path,
            d_hash,
            require_full_seakeeping=True,
            require_executed_cfd=True,
            cfd_result_mode=mode,
            paired_reference=reference,
        )[1]

    # a paired difference counts only against the reference it was taken on
    assert lookup("paired_warm_start", "cfdref_old") is not None
    assert lookup("paired_warm_start", "cfdref_new") is None

    found = lookup("paired_warm_start")
    assert found is not None and found.mode == "paired_warm_start", "a paired file must come back as paired"
    assert lookup("cold_start") is None, "a paired result must not answer a request for a cold-start one"
    assert lookup(None).mode == "paired_warm_start"

    # the same through pareto.json, and a file written before modes existed counts as cold start
    cfd_file.unlink()
    (tmp_path / "pareto.json").write_text(json.dumps({"tier2_verified": {d_hash: {"cfd": paired.to_dict()}}}))
    assert lookup("paired_warm_start") is not None and lookup("cold_start") is None
    (tmp_path / "pareto.json").unlink()
    cfd_file.write_text(json.dumps({"posed": {}, "mesh_info": {}, "speeds": paired.speeds}))
    assert lookup("cold_start") is not None and lookup("paired_warm_start") is None


def test_a_paired_result_is_reported_by_its_own_differences(baseline_eval: tuple) -> None:
    from rsi.promote import _compute_deltas

    ref, metrics = baseline_eval
    # The candidate is 0.30 N worse than the baseline on the same mesh. Its absolute drag, 18.3 N at
    # step 4900, is 2.2 N below the published 500-step baseline; that must not be reported as a gain.
    deltas = _compute_deltas(metrics, ref, None, _paired_cfd(+0.30, -0.10))["cfd"]
    assert deltas["5.5kn"]["drag_total_N"] == pytest.approx(0.30)
    assert deltas["5.5kn"]["drag_pressure_N"] == pytest.approx(0.25)
    assert deltas["5.5kn"]["drag_viscous_N"] == pytest.approx(0.05)
    assert deltas["5.5kn"]["effective_power_W"] == pytest.approx(0.30 * 5.5 * 0.514444, abs=1e-3)
    assert deltas["6.2kn"]["drag_total_N"] == pytest.approx(-0.10)

    report = evaluate_gates(metrics, ref, cfd=_paired_cfd(+0.30, -0.10))
    assert [v.gate for v in report.violations if v.gate.startswith("cfd_")] == ["cfd_drag_delta_5.5kn"]
