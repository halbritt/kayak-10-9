"""Unit tests for rsi.cli, rsi.__main__, and rsi package exports."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import typer
from typer.testing import CliRunner

import rsi
from rsi.baseline import load_baseline_reference
from rsi.cli import app
from rsi.genome import DesignSpace, vector_to_hull

_TEST_WORK_PARENT = Path.home() / ".cache" / "kayak-10-9" / "test-work"
_TEST_WORK_PARENT.mkdir(parents=True, exist_ok=True)
TEST_WORK_ROOT = Path(tempfile.mkdtemp(prefix="cli_tests_", dir=_TEST_WORK_PARENT))
atexit.register(shutil.rmtree, TEST_WORK_ROOT, ignore_errors=True)
RUNNER = CliRunner()


def _clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_cli_app_entrypoint_and_exports() -> None:
    assert isinstance(app, typer.Typer)
    m2_expected = {
        "BaselineReference",
        "CandidateMetrics",
        "CapytaineBaseline",
        "CfdMetrics",
        "DesignSpace",
        "FairnessBaseline",
        "FairnessCutBaseline",
        "FairnessSectionBaseline",
        "GateReport",
        "GateViolation",
        "LoadBaseline",
        "LoadEvalMetrics",
        "OpenFoamBaseline",
        "OpenFoamSpeedBaseline",
        "ParameterSpec",
        "SeakeepingConditionBaseline",
        "SeakeepingMetrics",
        "evaluate_gates",
        "evaluate_tier0_tier1",
        "evaluate_tier2_cfd",
        "evaluate_tier2_seakeeping",
        "hull_to_vector",
        "load_baseline_reference",
        "parse_openfoam_case_metrics",
        "resolve_openfoam_bashrc",
        "stage_openfoam_case",
        "vector_to_hull",
    }
    m3_expected = {
        "AdaptationHints",
        "CampaignConfig",
        "CampaignResult",
        "CampaignState",
        "ComparisonReport",
        "DiagnosticItem",
        "DiagnosticReport",
        "LedgerEntry",
        "PromotionReport",
        "aggregate_diagnostics",
        "compare_candidate",
        "diagnose_candidate",
        "diagnose_validation_error",
        "load_campaign_status",
        "promote_candidate",
        "run_campaign",
        "run_smoke",
    }
    assert m2_expected <= set(rsi.__all__)
    assert m3_expected <= set(rsi.__all__)

    proc = subprocess.run(
        [sys.executable, "-m", "rsi", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    for cmd in ("evaluate", "smoke", "run", "status", "compare", "promote"):
        assert cmd in proc.stdout


def test_cli_evaluate_baseline_and_invalid_inputs() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "evaluate")
    out1 = work / "eval1.json"
    out2 = work / "eval2.json"

    res = RUNNER.invoke(
        app,
        [
            "evaluate",
            "--hull",
            "hull/kayak-10-9.hull.json",
            "--out",
            str(out1),
            "--json-out",
            str(out2),
            "--work-dir",
            str(work / "work"),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(out1.read_text())
    assert json.loads(out2.read_text()) == data
    assert data["valid"] is True
    assert data["passed"] is True
    assert data["metrics"]["sr_4in"] == 10.88
    assert data["metrics"]["usca_4in_pct"] == 9.15
    assert "hydro_3load" in data
    assert data["gates"]["passed"] is True
    json.dumps(data, allow_nan=False)

    # Invalid tier
    bad_tier = RUNNER.invoke(
        app,
        ["evaluate", "--hull", "hull/kayak-10-9.hull.json", "--tier", "9"],
    )
    assert bad_tier.exit_code != 0

    # Missing hull
    missing = RUNNER.invoke(
        app,
        ["evaluate", "--hull", str(work / "missing.hull.json")],
    )
    assert missing.exit_code != 0

    # Malformed JSON
    malformed = work / "bad.hull.json"
    malformed.write_text("{bad json")
    res_mal = RUNNER.invoke(app, ["evaluate", "--hull", str(malformed)])
    assert res_mal.exit_code != 0

    # Schema ValidationError hull JSON
    ref = load_baseline_reference()
    raw = ref.hull.model_dump(mode="json")
    raw["distribution_v2"]["topsides"]["sheer_plan"]["max_half_breadth_m"] = 0.45
    schema_bad = work / "schema_bad.hull.json"
    schema_bad.write_text(json.dumps(raw))
    out_err = work / "eval_schema_err.json"
    res_val = RUNNER.invoke(
        app,
        ["evaluate", "--hull", str(schema_bad), "--out", str(out_err), "--work-dir", str(work / "w2")],
    )
    assert res_val.exit_code != 0
    assert json.loads(out_err.read_text())["diagnostic"]["feasible"] is False


def test_cli_smoke_and_status() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "smoke_status")
    smoke_out = work / "smoke.json"
    res = RUNNER.invoke(
        app,
        [
            "smoke",
            "--work-dir",
            str(work / "w"),
            "--out",
            str(smoke_out),
            "--iterations",
            "1",
            "--population",
            "2",
            "--seed",
            "7",
            "--skip-resistance",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (work / "w" / "state.json").is_file()
    assert (work / "w" / "pareto.json").is_file()
    assert (work / "w" / "ledger.jsonl").is_file()

    st_res = RUNNER.invoke(app, ["status", "--work-dir", str(work / "w"), "--json"])
    assert st_res.exit_code == 0, st_res.output
    st_json = json.loads(st_res.stdout)
    assert st_json["completed_iterations"] == 1
    assert st_json["feasible_count"] >= 1

    empty_res = RUNNER.invoke(app, ["status", "--work-dir", str(work / "empty"), "--json"])
    assert empty_res.exit_code != 0

    (work / "w" / "state.json").write_text("{broken")
    corrupt_res = RUNNER.invoke(app, ["status", "--work-dir", str(work / "w"), "--json"])
    assert corrupt_res.exit_code != 0


def test_cli_run_resume_and_seed_determinism() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "run_resume")
    w1 = work / "r1"
    w2 = work / "r2"

    r_part1 = RUNNER.invoke(
        app,
        [
            "run",
            "--iterations",
            "1",
            "--population",
            "2",
            "--seed",
            "11",
            "--work-dir",
            str(w1),
            "--skip-resistance",
        ],
    )
    assert r_part1.exit_code == 0, r_part1.output

    r_part2 = RUNNER.invoke(
        app,
        [
            "run",
            "--iterations",
            "2",
            "--population",
            "2",
            "--work-dir",
            str(w1),
            "--resume",
            "--skip-resistance",
        ],
    )
    assert r_part2.exit_code == 0, r_part2.output
    st_after_p2 = json.loads((w1 / "state.json").read_text())
    assert st_after_p2["seed"] == 11
    assert st_after_p2["provenance"]["seed"] == 11
    assert st_after_p2["completed_iterations"] == 2

    # Resuming with iterations < completed_iterations and omitted --seed must not downgrade state
    r_noop = RUNNER.invoke(
        app,
        [
            "run",
            "--iterations",
            "1",
            "--population",
            "2",
            "--work-dir",
            str(w1),
            "--resume",
            "--skip-resistance",
        ],
    )
    assert r_noop.exit_code == 0, r_noop.output
    st_after_noop = json.loads((w1 / "state.json").read_text())
    assert st_after_noop["seed"] == 11
    assert st_after_noop["completed_iterations"] == 2
    assert st_after_noop["target_iterations"] == 2

    r_full = RUNNER.invoke(
        app,
        [
            "run",
            "--iterations",
            "2",
            "--population",
            "2",
            "--seed",
            "11",
            "--work-dir",
            str(w2),
            "--skip-resistance",
        ],
    )
    assert r_full.exit_code == 0, r_full.output

    h1 = [json.loads(ln)["design_hash"] for ln in (w1 / "ledger.jsonl").read_text().splitlines()]
    h2 = [json.loads(ln)["design_hash"] for ln in (w2 / "ledger.jsonl").read_text().splitlines()]
    assert h1 == h2

    (w1 / "state.json").write_text("{corrupt")
    r_bad_resume = RUNNER.invoke(
        app,
        ["run", "--iterations", "3", "--work-dir", str(w1), "--resume"],
    )
    assert r_bad_resume.exit_code != 0


def test_cli_compare_and_promote_gating() -> None:
    work = _clean_dir(TEST_WORK_ROOT / "cmp_prom")
    base_bytes = Path("hull/kayak-10-9.hull.json").read_bytes()

    cmp_out = work / "cmp.json"
    res_cmp = RUNNER.invoke(
        app,
        [
            "compare",
            "--candidate",
            "hull/kayak-10-9.hull.json",
            "--work-dir",
            str(work / "w"),
            "--out",
            str(cmp_out),
            "--skip-resistance",
        ],
    )
    assert res_cmp.exit_code == 0, res_cmp.output
    cmp_data = json.loads(cmp_out.read_text())
    assert cmp_data["deltas"]["class_form"]["sr_4in"] == 0.0
    assert cmp_data["passed"] is True

    prom_dry_out = work / "prom_dry.json"
    res_dry = RUNNER.invoke(
        app,
        [
            "promote",
            "--candidate",
            "hull/kayak-10-9.hull.json",
            "--work-dir",
            str(work / "w"),
            "--dry-run",
            "--out",
            str(prom_dry_out),
        ],
    )
    assert res_dry.exit_code == 0, res_dry.output
    assert json.loads(prom_dry_out.read_text())["applied"] is False

    prom_apply_out = work / "prom_apply.json"
    res_apply = RUNNER.invoke(
        app,
        [
            "promote",
            "--candidate",
            "hull/kayak-10-9.hull.json",
            "--work-dir",
            str(work / "w"),
            "--apply",
            "--out",
            str(prom_apply_out),
        ],
    )
    assert res_apply.exit_code != 0
    assert json.loads(prom_apply_out.read_text())["applied"] is False
    assert Path("hull/kayak-10-9.hull.json").read_bytes() == base_bytes

    # Failing hull should exit non-zero on promote --dry-run
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()
    bad_vec = space.baseline_vector()
    bad_vec[idx["target_beam_m"]] = 0.491
    bad_hull = vector_to_hull(bad_vec, ref.hull, space)
    bad_path = work / "bad_cand.hull.json"
    bad_path.write_text(bad_hull.model_dump_json(indent=2) + "\n")

    res_bad_dry = RUNNER.invoke(
        app,
        [
            "promote",
            "--candidate",
            str(bad_path),
            "--work-dir",
            str(work / "w_bad"),
            "--dry-run",
        ],
    )
    assert res_bad_dry.exit_code != 0
    assert Path("hull/kayak-10-9.hull.json").read_bytes() == base_bytes
