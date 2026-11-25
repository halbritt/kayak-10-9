"""Unit tests for rsi.loop recursive optimization, durability, and resume."""

from __future__ import annotations

import atexit
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

from kayakgen.model.hull import Hull
import pytest

from rsi.baseline import load_baseline_reference
from rsi.evaluator import CandidateMetrics, LoadEvalMetrics, evaluate_tier0_tier1
from rsi.fidelity import CfdMetrics, SeakeepingMetrics, resolve_openfoam_bashrc
from rsi.genome import DesignSpace, hull_to_vector, vector_to_hull
from rsi.loop import (
    CampaignConfig,
    ResumeStateError,
    append_jsonl_atomic,
    load_campaign_status,
    run_campaign,
    run_smoke,
    write_json_atomic,
)

_TEST_WORK_PARENT = Path.home() / ".cache" / "kayak-10-9" / "test-work"
_TEST_WORK_PARENT.mkdir(parents=True, exist_ok=True)
TEST_WORK_ROOT = Path(tempfile.mkdtemp(prefix="loop_tests_", dir=_TEST_WORK_PARENT))
atexit.register(shutil.rmtree, TEST_WORK_ROOT, ignore_errors=True)


def _clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _surrogate_evaluator(hull: Hull, cand_dir: Path, baseline) -> CandidateMetrics:
    """Fast deterministic physics-like surrogate for unit-testing loop mechanics."""
    cand_dir.mkdir(parents=True, exist_ok=True)
    hull_file = cand_dir / "candidate.hull.json"
    hull_file.write_text(hull.model_dump_json(indent=2) + "\n")

    space = DesignSpace.default()
    vec = hull_to_vector(hull, space)
    base_vec = space.baseline_vector()
    idx = space.index_by_name()

    d_beam = vec[idx["target_beam_m"]] - base_vec[idx["target_beam_m"]]
    d_wl0 = vec[idx["wl_a0"]] - base_vec[idx["wl_a0"]]
    d_sec_fwd = vec[idx["sec_side_fwd"]] - base_vec[idx["sec_side_fwd"]]
    d_rock_stern = vec[idx["rocker_stern_end_m"]] - base_vec[idx["rocker_stern_end_m"]]
    d_draft_a1 = vec[idx["draft_a1"]]

    sr_exact = baseline.sr_4in_exact - 22.0 * d_beam - 35.0 * d_wl0
    usca_exact = baseline.usca_4in_pct_exact + 18.0 * d_beam + 30.0 * d_wl0
    z_min = -abs(d_draft_a1) * 0.005
    b75 = 3.46 - 0.1 * d_rock_stern + 2.0 * abs(d_wl0)

    loads = {}
    for lb, base_ld in baseline.loads.items():
        gm = base_ld.gm_cm + 30.0 * d_beam + 40.0 * d_wl0 + 5.0 * d_rock_stern - 0.5 * d_sec_fwd
        gz = tuple(
            base_ld.gz_mm[i] + 150.0 * d_beam + 180.0 * d_wl0 + 20.0 * d_rock_stern - 1.5 * d_sec_fwd
            for i in range(3)
        )
        loads[lb] = LoadEvalMetrics(
            load_lb=lb,
            t_cm=base_ld.t_cm,
            draft_mm=base_ld.draft_mm,
            bwl_cm=base_ld.bwl_cm,
            wet_m2=base_ld.wet_m2,
            gm_cm=gm,
            gz_mm=(gz[0], gz[1], gz[2]),
            lcg_m=0.0,
        )

    kn_exact = {
        w: v - 1.2 * d_beam - 1.5 * d_wl0 - 0.02 * d_sec_fwd
        for w, v in baseline.kn_225.items()
    }
    kn_round = {w: round(v, 2) for w, v in kn_exact.items()}

    return CandidateMetrics(
        valid=True,
        error=None,
        design_hash=hull.design_hash(),
        hull_json_path=str(hull_file),
        mesh_resolution=(441, 129),
        elapsed_s=0.01,
        skip_resistance=False,
        is_watertight=True,
        is_volume=True,
        n_faces=1000,
        z_min_m=z_min,
        l_oa_m=baseline.l_oa_m,
        b_oa_cm=baseline.b_oa_cm,
        l4_m=baseline.l4_m,
        b4_m=baseline.b4_m,
        sr_4in_exact=sr_exact,
        sr_4in=round(sr_exact, 2),
        usca_4in_pct_exact=usca_exact,
        usca_4in_pct=round(usca_exact, 2),
        transition_dB_per_20cm=baseline.transition_dB_per_20cm,
        has_surface_fairing=True,
        waterline_hollow_max_deg=0.005,
        fwd_wl_max_turn_deg=-0.01,
        entry_zone_max_turn_deg=0.02,
        buttock_0075_peak_deg=b75,
        forebody_max_keel_groove_mm=0.0,
        forebody_min_k=0.5,
        forebody_max_abs_dk_ds=120.0,
        loads=loads,
        kn_225=kn_round,
        kn_225_exact=kn_exact,
    )


def test_atomic_persistence_and_strict_json_no_nan() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "atomic_json")
    json_path = work / "payload.json"
    jsonl_path = work / "events.jsonl"

    write_json_atomic(json_path, {"ok": True, "val": 1.25})
    append_jsonl_atomic(jsonl_path, {"candidate_id": "c1", "val": 2.5})
    assert json.loads(json_path.read_text())["val"] == 1.25
    assert json.loads(jsonl_path.read_text().splitlines()[0])["candidate_id"] == "c1"
    assert not any(p.name.endswith(".tmp") for p in work.iterdir())

    with pytest.raises(ValueError):
        write_json_atomic(work / "nan.json", {"bad": float("nan")})
    with pytest.raises(ValueError):
        append_jsonl_atomic(jsonl_path, {"bad": float("inf")})
    assert not any(p.name.endswith(".tmp") for p in work.iterdir())


def test_validation_error_and_duplicate_hash_handling() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "validation_and_dup")
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()

    # Start from a near-boundary sheer_max_hb_m so some proposals trigger ValidationError
    bad_seed_vec = space.baseline_vector()
    bad_seed_vec[idx["sheer_max_hb_m"]] = 0.2915
    bad_seed_hull = vector_to_hull(bad_seed_vec, ref.hull, space)
    seed_path = work / "near_edge_seed.hull.json"
    seed_path.write_text(bad_seed_hull.model_dump_json(indent=2) + "\n")

    cfg = CampaignConfig(
        iterations=2,
        population=3,
        seed=5,
        work_dir=work / "run",
        seed_hull_path=seed_path,
        pre_screen=False,
    )
    res = run_campaign(cfg, evaluator_fn=_surrogate_evaluator)
    assert res.status == "completed"
    assert res.completed_iterations == 2
    assert res.total_evaluations >= 4
    lines = [json.loads(ln) for ln in (work / "run" / "ledger.jsonl").read_text().splitlines()]
    assert all(isinstance(rec["candidate_id"], str) for rec in lines)
    assert res.provenance["kayak_10_9_sha"]
    assert res.provenance["kayak_gen_sha"]


def test_deterministic_seed_replay_and_distinct_seeds() -> None:
    w_a = _clean_dir(TEST_WORK_ROOT / "seed7_a")
    w_b = _clean_dir(TEST_WORK_ROOT / "seed7_b")
    w_c = _clean_dir(TEST_WORK_ROOT / "seed19")

    res_a = run_campaign(
        CampaignConfig(iterations=2, population=3, seed=7, work_dir=w_a, pre_screen=False),
        evaluator_fn=_surrogate_evaluator,
    )
    res_b = run_campaign(
        CampaignConfig(iterations=2, population=3, seed=7, work_dir=w_b, pre_screen=False),
        evaluator_fn=_surrogate_evaluator,
    )
    run_campaign(
        CampaignConfig(iterations=2, population=3, seed=19, work_dir=w_c, pre_screen=False),
        evaluator_fn=_surrogate_evaluator,
    )

    hashes_a = [
        json.loads(ln)["design_hash"]
        for ln in (w_a / "ledger.jsonl").read_text().splitlines()
    ]
    hashes_b = [
        json.loads(ln)["design_hash"]
        for ln in (w_b / "ledger.jsonl").read_text().splitlines()
    ]
    hashes_c = [
        json.loads(ln)["design_hash"]
        for ln in (w_c / "ledger.jsonl").read_text().splitlines()
    ]
    assert hashes_a == hashes_b
    assert hashes_a != hashes_c
    assert res_a.best_design_hash == res_b.best_design_hash


def test_checkpoint_resume_continuity_and_corrupt_resume_rejection() -> None:
    w_full = _clean_dir(TEST_WORK_ROOT / "resume_full")
    w_split = _clean_dir(TEST_WORK_ROOT / "resume_split")

    res_full = run_campaign(
        CampaignConfig(iterations=4, population=3, seed=13, work_dir=w_full, pre_screen=False),
        evaluator_fn=_surrogate_evaluator,
    )
    run_campaign(
        CampaignConfig(iterations=2, population=3, seed=13, work_dir=w_split, pre_screen=False),
        evaluator_fn=_surrogate_evaluator,
    )
    res_resumed = run_campaign(
        CampaignConfig(
            iterations=4, population=3, seed=42, work_dir=w_split, resume=True, pre_screen=False
        ),
        evaluator_fn=_surrogate_evaluator,
    )

    assert res_resumed.seed == 13
    assert res_resumed.provenance["seed"] == 13
    assert res_resumed.completed_iterations == 4
    assert res_resumed.best_design_hash == res_full.best_design_hash
    st_resumed = json.loads((w_split / "state.json").read_text())
    assert st_resumed["seed"] == 13
    assert st_resumed["completed_iterations"] == 4

    # Resuming with smaller iterations must not downgrade completed_iterations or overwrite seed
    res_smaller = run_campaign(
        CampaignConfig(
            iterations=1, population=3, seed=99, work_dir=w_split, resume=True, pre_screen=False
        ),
        evaluator_fn=_surrogate_evaluator,
    )
    assert res_smaller.seed == 13
    assert res_smaller.completed_iterations == 4
    assert res_smaller.target_iterations == 4
    st_smaller = json.loads((w_split / "state.json").read_text())
    assert st_smaller["seed"] == 13
    assert st_smaller["completed_iterations"] == 4
    full_hashes = [
        json.loads(ln)["design_hash"]
        for ln in (w_full / "ledger.jsonl").read_text().splitlines()
    ]
    split_hashes = [
        json.loads(ln)["design_hash"]
        for ln in (w_split / "ledger.jsonl").read_text().splitlines()
    ]
    assert split_hashes == full_hashes

    # Corrupt resume tests
    w_bad_json = _clean_dir(TEST_WORK_ROOT / "corrupt_state")
    shutil.copytree(w_split, w_bad_json, dirs_exist_ok=True)
    (w_bad_json / "state.json").write_text("{corrupt")
    with pytest.raises(ResumeStateError):
        run_campaign(
            CampaignConfig(iterations=5, population=3, seed=13, work_dir=w_bad_json, resume=True),
            evaluator_fn=_surrogate_evaluator,
        )

    w_nan = _clean_dir(TEST_WORK_ROOT / "corrupt_nan")
    shutil.copytree(w_split, w_nan, dirs_exist_ok=True)
    (w_nan / "state.json").write_text((w_nan / "state.json").read_text().replace('"seed": 13', '"seed": NaN'))
    with pytest.raises(ResumeStateError):
        load_campaign_status(w_nan)

    w_trunc_ledger = _clean_dir(TEST_WORK_ROOT / "corrupt_ledger")
    shutil.copytree(w_split, w_trunc_ledger, dirs_exist_ok=True)
    (w_trunc_ledger / "ledger.jsonl").write_text('{"candidate_id": "x"\n')
    with pytest.raises(ResumeStateError):
        load_campaign_status(w_trunc_ledger)

    w_mismatch = _clean_dir(TEST_WORK_ROOT / "mismatch_base")
    shutil.copytree(w_split, w_mismatch, dirs_exist_ok=True)
    st_data = json.loads((w_mismatch / "state.json").read_text())
    st_data["provenance"]["baseline_design_hash"] = "wrong_hash"
    write_json_atomic(w_mismatch / "state.json", st_data)
    with pytest.raises(ResumeStateError):
        run_campaign(
            CampaignConfig(iterations=5, population=3, seed=13, work_dir=w_mismatch, resume=True),
            evaluator_fn=_surrogate_evaluator,
        )


def test_adaptation_and_diagnostic_steering_on_infeasible_seed() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "infeasible_seed")
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()

    bad_vec = space.baseline_vector()
    bad_vec[idx["target_beam_m"]] = 0.491
    bad_vec[idx["draft_a1"]] = 0.005
    bad_hull = vector_to_hull(bad_vec, ref.hull, space)
    seed_path = work / "infeasible.hull.json"
    seed_path.write_text(bad_hull.model_dump_json(indent=2) + "\n")

    res = run_campaign(
        CampaignConfig(
            iterations=4,
            population=5,
            seed=21,
            work_dir=work / "run",
            seed_hull_path=seed_path,
            pre_screen=False,
        ),
        evaluator_fn=_surrogate_evaluator,
    )
    records = [json.loads(ln) for ln in (work / "run" / "ledger.jsonl").read_text().splitlines()]
    seed_rec = next(r for r in records if r["candidate_id"] == "seed_hull")
    assert seed_rec["gate_report"]["passed"] is False
    assert seed_rec["violation_deficit"] > 0.0

    later_evals = [r for r in records if r["iteration"] >= 1 and r["status"] == "evaluated"]
    min_later_deficit = min(r["violation_deficit"] for r in later_evals)
    assert min_later_deficit < seed_rec["violation_deficit"]
    assert res.feasible_count >= 1


def test_tier2_top_k_promotion_and_cached_reuse() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "tier2_top_cache")
    calls = {"seak": 0, "cfd": 0}

    def fake_seak(hull: Hull, out_dir: Path, ref) -> SeakeepingMetrics:
        calls["seak"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = tuple(
            {
                "speed_kn": r.speed_kn,
                "heading": r.heading,
                "sea": r.sea,
                "bow_dips_per_min": r.bow_dips_per_min,
                "seat_acc_rms_ms2": r.seat_acc_rms_ms2,
            }
            for r in ref.capytaine.rows_225
        )
        return SeakeepingMetrics(
            valid=True,
            error=None,
            quick=False,
            elapsed_s=0.1,
            gm_t_cm=ref.capytaine.gm_t_cm,
            roll_period_s=ref.capytaine.roll_period_s,
            rows=rows,
        )

    def fake_cfd(hull: Hull, out_dir: Path, ref) -> CfdMetrics:
        calls["cfd"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        return CfdMetrics(
            valid=True,
            error=None,
            dry_run=False,
            elapsed_s=0.1,
            bashrc=str(resolve_openfoam_bashrc()),
            bashrc_exists=True,
            staging_dir=str(out_dir),
            speeds={
                "5.5kn": ref.openfoam.speed_5_5kn.to_dict(),
                "6.2kn": ref.openfoam.speed_6_2kn.to_dict(),
            },
        )

    res = run_campaign(
        CampaignConfig(iterations=2, population=3, seed=42, work_dir=work, tier2_top=1, pre_screen=False),
        evaluator_fn=_surrogate_evaluator,
        seakeeping_fn=fake_seak,
        cfd_fn=fake_cfd,
    )
    assert calls["seak"] == 1
    assert calls["cfd"] == 1
    assert len(res.tier2_verified) == 1

    # Resume with same target iterations reuses cached Tier 2 without re-running
    run_campaign(
        CampaignConfig(
            iterations=2, population=3, seed=42, work_dir=work, tier2_top=1, resume=True, pre_screen=False
        ),
        evaluator_fn=_surrogate_evaluator,
        seakeeping_fn=fake_seak,
        cfd_fn=fake_cfd,
    )
    assert calls["seak"] == 1
    assert calls["cfd"] == 1

    # Resume with tier2_top=0 and no new iterations preserves tier2_completed=True
    run_campaign(
        CampaignConfig(
            iterations=1, population=3, seed=99, work_dir=work, tier2_top=0, resume=True, pre_screen=False
        ),
        evaluator_fn=_surrogate_evaluator,
    )
    st_t2 = json.loads((work / "state.json").read_text())
    assert st_t2["tier2_completed"] is True
    assert st_t2["completed_iterations"] == 2
    assert st_t2["seed"] == 42


def test_tier2_diagnostic_mode_preserves_pareto_candidates() -> None:
    base_work = _clean_dir(TEST_WORK_ROOT / "tier2_diag_base")
    dry_work = _clean_dir(TEST_WORK_ROOT / "tier2_diag_dry")
    quick_work = _clean_dir(TEST_WORK_ROOT / "tier2_diag_quick")

    def fake_seak_quick(hull: Hull, out_dir: Path, ref) -> SeakeepingMetrics:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = tuple(
            {
                "speed_kn": r.speed_kn,
                "heading": r.heading,
                "sea": r.sea,
                "bow_dips_per_min": r.bow_dips_per_min,
                "seat_acc_rms_ms2": r.seat_acc_rms_ms2,
            }
            for r in ref.capytaine.rows_225
            if r.speed_kn == 4.5 and r.heading == "head"
        )
        return SeakeepingMetrics(
            valid=True,
            error=None,
            quick=True,
            elapsed_s=0.1,
            gm_t_cm=ref.capytaine.gm_t_cm,
            roll_period_s=ref.capytaine.roll_period_s,
            rows=rows,
        )

    def fake_cfd_dry(hull: Hull, out_dir: Path, ref) -> CfdMetrics:
        out_dir.mkdir(parents=True, exist_ok=True)
        return CfdMetrics(
            valid=True,
            error=None,
            dry_run=True,
            elapsed_s=0.1,
            bashrc=str(resolve_openfoam_bashrc()),
            bashrc_exists=True,
            staging_dir=str(out_dir),
            speeds=None,
        )

    def fake_cfd_ok(hull: Hull, out_dir: Path, ref) -> CfdMetrics:
        out_dir.mkdir(parents=True, exist_ok=True)
        return CfdMetrics(
            valid=True,
            error=None,
            dry_run=False,
            elapsed_s=0.1,
            bashrc=str(resolve_openfoam_bashrc()),
            bashrc_exists=True,
            staging_dir=str(out_dir),
            speeds={
                "5.5kn": ref.openfoam.speed_5_5kn.to_dict(),
                "6.2kn": ref.openfoam.speed_6_2kn.to_dict(),
            },
        )

    res_base = run_campaign(
        CampaignConfig(iterations=2, population=3, seed=42, work_dir=base_work, tier2_top=0, pre_screen=False),
        evaluator_fn=_surrogate_evaluator,
    )
    res_dry = run_campaign(
        CampaignConfig(
            iterations=2,
            population=3,
            seed=42,
            work_dir=dry_work,
            tier2_top=1,
            cfd_dry_run=True,
            quick_tier2=True,
            pre_screen=False,
        ),
        evaluator_fn=_surrogate_evaluator,
        seakeeping_fn=fake_seak_quick,
        cfd_fn=fake_cfd_dry,
    )
    assert res_dry.pareto_count == res_base.pareto_count
    assert res_dry.feasible_count == res_base.feasible_count
    assert {e["design_hash"] for e in res_dry.pareto_front} == {
        e["design_hash"] for e in res_base.pareto_front
    }
    top_dry = res_dry.pareto_front[0]
    assert top_dry["gate_report"]["passed"] is True
    assert top_dry["tier2_passed"] is False
    assert top_dry["tier2_gate_report"]["passed"] is False
    assert len(res_dry.tier2_verified) == 1

    res_quick = run_campaign(
        CampaignConfig(
            iterations=2,
            population=3,
            seed=42,
            work_dir=quick_work,
            tier2_top=1,
            cfd_dry_run=False,
            quick_tier2=True,
            pre_screen=False,
        ),
        evaluator_fn=_surrogate_evaluator,
        seakeeping_fn=fake_seak_quick,
        cfd_fn=fake_cfd_ok,
    )
    assert res_quick.pareto_count == res_base.pareto_count
    assert res_quick.pareto_front[0]["gate_report"]["passed"] is True
    assert res_quick.pareto_front[0]["tier2_passed"] is False
    assert "tier2_gate_report" in res_quick.pareto_front[0]


def test_real_smoke_on_ssd3() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "real_smoke")
    res = run_smoke(work, seed=42, iterations=1, population=2, skip_resistance=True)
    assert res.status == "completed"
    assert res.feasible_count >= 1
    assert (work / "state.json").is_file()
    assert (work / "pareto.json").is_file()
    assert (work / "ledger.jsonl").is_file()
    assert (work / "best_candidate.hull.json").is_file()
    # Verify real baseline metrics inside ledger
    first = json.loads((work / "ledger.jsonl").read_text().splitlines()[0])
    assert first["candidate_id"] == "baseline"
    assert first["metrics"]["sr_4in"] == 10.88
    assert first["gate_report"]["passed"] is True
    # Also verify real evaluator works directly on best_candidate
    ref = load_baseline_reference()
    best_hull = Hull.model_validate_json((work / "best_candidate.hull.json").read_text())
    m = evaluate_tier0_tier1(best_hull, work / "verify_best", ref, skip_resistance=True)
    assert m.valid is True
    assert replace(m).valid is True
