"""Typer CLI for Kayak 10.9 Recursive Self-Improvement Environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import ValidationError
import typer

from rsi.baseline import load_baseline_reference
from rsi.diagnose import diagnose_candidate, diagnose_validation_error
from rsi.evaluator import evaluate_tier0_tier1
from rsi.fidelity import evaluate_tier2_cfd, evaluate_tier2_seakeeping
from rsi.gates import evaluate_gates
from rsi.genome import DesignSpace, hull_to_vector
from rsi.loop import (
    CampaignConfig,
    ResumeStateError,
    load_campaign_status,
    run_campaign,
    run_smoke,
    write_json_atomic,
)
from rsi.promote import (
    augment_metrics_dict,
    build_hydro_3load_payload,
    compare_candidate,
    load_candidate_hull,
    promote_candidate,
)

app = typer.Typer(
    name="kayak109-rsi",
    help="Kayak 10.9 recursive self-improvement CLI.",
    no_args_is_help=True,
    add_completion=False,
)


def default_work_dir() -> Path:
    env_dir = os.environ.get("KAYAK109_WORK_DIR")
    if env_dir:
        return Path(env_dir)
    xdg = os.environ.get("XDG_CACHE_HOME")
    cache_base = Path(xdg) if xdg else (Path.home() / ".cache")
    return cache_base / "kayak-10-9" / "work"


def _resolve_work_dir(work_dir: Path | None) -> Path:
    return default_work_dir() if work_dir is None else Path(work_dir)


def _emit_output(
    payload: dict[str, Any],
    *,
    out: Path | None = None,
    json_out: Path | None = None,
    emit_json: bool = False,
    human_summary: str | None = None,
) -> None:
    targets: list[Path] = []
    if out is not None:
        targets.append(Path(out))
    if json_out is not None and Path(json_out) not in targets:
        targets.append(Path(json_out))
    for path in targets:
        write_json_atomic(path, payload)

    if emit_json or not targets:
        typer.echo(json.dumps(payload, indent=2, allow_nan=False))
    elif human_summary:
        typer.echo(human_summary)


def _err(msg: str, code: int = 1) -> None:
    typer.echo(f"Error: {msg}", err=True)
    raise typer.Exit(code=code)


@app.command()
def evaluate(
    hull: Annotated[Path, typer.Option("--hull", help="Path to candidate *.hull.json")],
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Output JSON path alias")] = None,
    tier: Annotated[int, typer.Option("--tier", help="Evaluation tier: 1 or 2")] = 1,
    work_dir: Annotated[Path | None, typer.Option("--work-dir", help="SSD3 scratch work directory")] = None,
    emit_json: Annotated[bool, typer.Option("--json", help="Emit JSON to stdout")] = False,
    skip_resistance: Annotated[bool, typer.Option("--skip-resistance", help="Skip resistance curve")] = False,
    quick_tier2: Annotated[bool, typer.Option("--quick-tier2", help="Use quick subset for Capytaine Tier 2")] = False,
    cfd_dry_run: Annotated[bool, typer.Option("--cfd-dry-run", help="Stage OpenFOAM without solving")] = False,
    tier2_n_jobs: Annotated[int, typer.Option("--tier2-n-jobs", help="Capytaine worker threads")] = 12,
) -> None:
    """Evaluate a hull JSON against Tier 0/1 (and optionally Tier 2) gates."""
    if tier not in (1, 2):
        _err(f"--tier must be 1 or 2, got {tier}", code=2)
    if not hull.is_file():
        _err(f"Hull file not found: {hull}", code=2)

    work = _resolve_work_dir(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline_reference()

    try:
        cand_hull = load_candidate_hull(hull)
    except ValidationError as exc:
        diag = diagnose_validation_error(exc)
        payload = {
            "command": "evaluate",
            "valid": False,
            "passed": False,
            "tier": tier,
            "hull_path": str(hull),
            "error": str(exc),
            "diagnostic": diag.to_dict(),
        }
        _emit_output(payload, out=out, json_out=json_out, emit_json=emit_json)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        _err(f"Failed to load hull JSON {hull}: {exc}", code=1)

    d_hash = cand_hull.design_hash()
    eval_dir = work / "evaluate" / d_hash.split(":")[-1][:16]
    metrics = evaluate_tier0_tier1(
        cand_hull,
        eval_dir,
        baseline,
        skip_resistance=skip_resistance,
    )

    seakeeping = None
    cfd = None
    if tier == 2 and metrics.valid:
        seakeeping = evaluate_tier2_seakeeping(
            cand_hull,
            eval_dir / "seakeeping",
            baseline,
            quick=quick_tier2,
            n_jobs=tier2_n_jobs,
        )
        cfd = evaluate_tier2_cfd(
            cand_hull,
            eval_dir / "cfd",
            baseline,
            dry_run=cfd_dry_run,
        )

    gate_report = evaluate_gates(metrics, baseline, seakeeping=seakeeping, cfd=cfd)
    space = DesignSpace.default()
    vec: list[float] | None = None
    try:
        vec = hull_to_vector(cand_hull, space)
    except Exception:  # noqa: BLE001
        vec = None
    diagnostic = diagnose_candidate(
        metrics,
        gate_report,
        seakeeping=seakeeping,
        cfd=cfd,
        vector=vec,
        space=space,
    )

    tier2_exec_ok = True
    if tier == 2:
        tier2_exec_ok = bool(
            seakeeping is not None
            and seakeeping.valid
            and cfd is not None
            and cfd.valid
        )
    overall_valid = bool(metrics.valid and tier2_exec_ok)

    gates_dict = gate_report.to_dict()
    payload = {
        "command": "evaluate",
        "valid": overall_valid,
        "passed": gate_report.passed,
        "tier": tier,
        "hull_path": str(hull),
        "design_hash": d_hash,
        "metrics": augment_metrics_dict(metrics),
        "hydro_3load": build_hydro_3load_payload(metrics),
        "gates": gates_dict,
        "gate_report": gates_dict,
        "seakeeping": seakeeping.to_dict() if seakeeping is not None else None,
        "cfd": cfd.to_dict() if cfd is not None else None,
        "diagnostic": diagnostic.to_dict(),
    }
    _emit_output(
        payload,
        out=out,
        json_out=json_out,
        emit_json=emit_json,
        human_summary=(
            f"evaluate valid={overall_valid} passed={gate_report.passed} "
            f"SR_4in={metrics.sr_4in} USCA={metrics.usca_4in_pct}%"
        ),
    )
    if not overall_valid:
        raise typer.Exit(code=1)


@app.command()
def smoke(
    work_dir: Annotated[Path | None, typer.Option("--work-dir", help="SSD3 scratch work directory")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Output JSON path alias")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic RNG seed")] = 42,
    iterations: Annotated[int, typer.Option("--iterations", help="Smoke iterations (>= 1)")] = 2,
    population: Annotated[int, typer.Option("--population", help="Smoke population (>= 1)")] = 2,
    workers: Annotated[int, typer.Option("--workers", help="Parallel evaluation workers")] = 1,
    skip_resistance: Annotated[bool, typer.Option("--skip-resistance", help="Skip resistance curve")] = False,
    emit_json: Annotated[bool, typer.Option("--json", help="Emit JSON to stdout")] = False,
) -> None:
    """Run deterministic baseline + perturbation smoke verification."""
    if iterations < 1 or population < 1 or workers < 1:
        _err("iterations, population, and workers must be >= 1", code=2)
    work = _resolve_work_dir(work_dir)
    result = run_smoke(
        work,
        seed=seed,
        iterations=iterations,
        population=population,
        workers=workers,
        skip_resistance=skip_resistance,
    )
    payload = {"command": "smoke", **result.to_dict()}
    _emit_output(
        payload,
        out=out,
        json_out=json_out,
        emit_json=emit_json,
        human_summary=(
            f"smoke completed: evals={result.total_evaluations} "
            f"feasible={result.feasible_count} pareto={result.pareto_count}"
        ),
    )
    if result.feasible_count < 1:
        raise typer.Exit(code=1)


@app.command()
def run(
    iterations: Annotated[int, typer.Option("--iterations", help="Target total iterations (>= 1)")] = 5,
    population: Annotated[int, typer.Option("--population", help="Candidates per iteration (>= 1)")] = 6,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic RNG seed")] = 42,
    work_dir: Annotated[Path | None, typer.Option("--work-dir", help="SSD3 scratch work directory")] = None,
    tier2_top: Annotated[int, typer.Option("--tier2-top", help="Top Pareto hulls to verify with Tier 2")] = 0,
    workers: Annotated[int, typer.Option("--workers", help="Parallel Tier 0/1 workers")] = 1,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from existing checkpoint in work_dir")] = False,
    seed_hull: Annotated[Path | None, typer.Option("--seed-hull", help="Optional starting hull JSON")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Output JSON path alias")] = None,
    emit_json: Annotated[bool, typer.Option("--json", help="Emit JSON to stdout")] = False,
    quick_tier2: Annotated[bool, typer.Option("--quick-tier2", help="Use quick subset for Capytaine Tier 2")] = False,
    cfd_dry_run: Annotated[bool, typer.Option("--cfd-dry-run", help="Stage OpenFOAM without solving")] = False,
    skip_resistance: Annotated[bool, typer.Option("--skip-resistance", help="Skip resistance curve")] = False,
    tier2_n_jobs: Annotated[int, typer.Option("--tier2-n-jobs", help="Capytaine worker threads")] = 12,
    freeze_group: Annotated[
        list[str] | None,
        typer.Option("--freeze-group", help="Parameter group(s) to hold fixed at initial/seed values"),
    ] = None,
) -> None:
    """Execute or resume a multi-iteration recursive self-improvement campaign."""
    if iterations < 1 or population < 1 or tier2_top < 0 or workers < 1:
        _err("Invalid numeric campaign bounds", code=2)
    if seed_hull is not None and not seed_hull.is_file():
        _err(f"Seed hull file not found: {seed_hull}", code=2)

    work = _resolve_work_dir(work_dir)
    cfg = CampaignConfig(
        iterations=iterations,
        population=population,
        seed=seed,
        work_dir=work,
        tier2_top=tier2_top,
        workers=workers,
        resume=resume,
        seed_hull_path=seed_hull,
        skip_resistance=skip_resistance,
        quick_tier2=quick_tier2,
        cfd_dry_run=cfd_dry_run,
        tier2_n_jobs=tier2_n_jobs,
        frozen_groups=tuple(freeze_group or ()),
    )
    try:
        result = run_campaign(cfg)
    except (ResumeStateError, FileNotFoundError, ValidationError, ValueError) as exc:
        _err(str(exc), code=1)

    payload = {"command": "run", **result.to_dict()}
    _emit_output(
        payload,
        out=out,
        json_out=json_out,
        emit_json=emit_json,
        human_summary=(
            f"run completed: iters={result.completed_iterations}/{result.target_iterations} "
            f"evals={result.total_evaluations} feasible={result.feasible_count} "
            f"pareto={result.pareto_count} best={result.best_candidate_id}"
        ),
    )


@app.command()
def status(
    work_dir: Annotated[Path | None, typer.Option("--work-dir", help="SSD3 campaign work directory")] = None,
    emit_json: Annotated[bool, typer.Option("--json", help="Emit JSON to stdout")] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Output JSON path alias")] = None,
) -> None:
    """Inspect durable campaign state, Pareto frontier, and best candidate."""
    work = _resolve_work_dir(work_dir)
    try:
        payload = load_campaign_status(work)
    except ResumeStateError as exc:
        _err(str(exc), code=1)

    _emit_output(
        payload,
        out=out,
        json_out=json_out,
        emit_json=emit_json,
        human_summary=(
            f"status={payload['status']} iter={payload['completed_iterations']}/"
            f"{payload['target_iterations']} evals={payload['total_evaluations']} "
            f"feasible={payload['feasible_count']} pareto={payload['pareto_count']} "
            f"best={payload['best_candidate_id']}"
        ),
    )


@app.command()
def compare(
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate *.hull.json path")],
    work_dir: Annotated[Path | None, typer.Option("--work-dir", help="SSD3 scratch work directory")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Output JSON path alias")] = None,
    tier2: Annotated[bool, typer.Option("--tier2", help="Include Tier 2 Capytaine & OpenFOAM")] = False,
    quick_tier2: Annotated[bool, typer.Option("--quick-tier2", help="Use quick Capytaine subset")] = False,
    cfd_dry_run: Annotated[bool, typer.Option("--cfd-dry-run", help="Stage OpenFOAM without solving")] = False,
    skip_resistance: Annotated[bool, typer.Option("--skip-resistance", help="Skip resistance curve")] = False,
    tier2_n_jobs: Annotated[int, typer.Option("--tier2-n-jobs", help="Capytaine worker threads")] = 12,
    emit_json: Annotated[bool, typer.Option("--json", help="Emit JSON to stdout")] = False,
) -> None:
    """Compare a candidate hull against baseline v0.2.3 metrics and gates."""
    if not candidate.is_file():
        _err(f"Candidate file not found: {candidate}", code=2)

    work = _resolve_work_dir(work_dir)
    baseline = load_baseline_reference()
    try:
        cand_hull = load_candidate_hull(candidate)
    except Exception as exc:  # noqa: BLE001
        _err(f"Invalid candidate hull JSON {candidate}: {exc}", code=1)

    report = compare_candidate(
        cand_hull,
        baseline,
        work,
        candidate_path=candidate,
        tier2=tier2,
        quick_seakeeping=quick_tier2,
        cfd_dry_run=cfd_dry_run,
        skip_resistance=skip_resistance,
        tier2_n_jobs=tier2_n_jobs,
    )
    payload = {"command": "compare", **report.to_dict()}
    _emit_output(
        payload,
        out=out,
        json_out=json_out,
        emit_json=emit_json,
        human_summary=(
            f"compare passed={report.passed} promotion_eligible={report.promotion_eligible} "
            f"delta_SR={report.deltas['class_form']['sr_4in_exact']} "
            f"delta_speed_kn={report.deltas['kn_225']['mean_speed_gain_kn']}"
        ),
    )

    tier2_exec_ok = True
    if tier2:
        tier2_exec_ok = bool(
            report.seakeeping is not None
            and report.seakeeping.valid
            and report.cfd is not None
            and report.cfd.valid
        )
    if not report.metrics.valid or not tier2_exec_ok:
        raise typer.Exit(code=1)


@app.command()
def promote(
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate *.hull.json path")],
    work_dir: Annotated[Path | None, typer.Option("--work-dir", help="SSD3 scratch work directory")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Check gates and preview without modifying repo")] = False,
    apply: Annotated[bool, typer.Option("--apply", help="Atomically update hull/kayak-10-9.hull.json if all Tier 0/1+2 gates pass")] = False,
    tier2: Annotated[bool, typer.Option("--tier2", help="Run/verify Tier 2 Capytaine & OpenFOAM")] = False,
    quick_tier2: Annotated[bool, typer.Option("--quick-tier2", help="Use quick Capytaine subset (disallowed for --apply)")] = False,
    cfd_dry_run: Annotated[bool, typer.Option("--cfd-dry-run", help="Stage OpenFOAM only (disallowed for --apply)")] = False,
    tier2_n_jobs: Annotated[int, typer.Option("--tier2-n-jobs", help="Capytaine worker threads")] = 12,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json-out", help="Output JSON path alias")] = None,
    emit_json: Annotated[bool, typer.Option("--json", help="Emit JSON to stdout")] = False,
) -> None:
    """Safely verify and optionally promote a candidate hull to hull/kayak-10-9.hull.json."""
    if dry_run and apply:
        _err("Cannot pass both --dry-run and --apply", code=2)
    if not candidate.is_file():
        _err(f"Candidate file not found: {candidate}", code=2)

    eff_dry_run = not apply
    work = _resolve_work_dir(work_dir)
    try:
        report = promote_candidate(
            candidate,
            work,
            dry_run=eff_dry_run,
            apply=apply,
            tier2=tier2,
            quick_seakeeping=quick_tier2,
            cfd_dry_run=cfd_dry_run,
            tier2_n_jobs=tier2_n_jobs,
        )
    except Exception as exc:  # noqa: BLE001
        _err(f"Promotion check failed: {exc}", code=1)

    payload = {"command": "promote", **report.to_dict()}
    _emit_output(
        payload,
        out=out,
        json_out=json_out,
        emit_json=emit_json,
        human_summary=(
            f"promote dry_run={report.dry_run} applied={report.applied} "
            f"passed={report.passed} eligible={report.promotion_eligible}: {report.reason}"
        ),
    )

    if apply:
        if not report.applied:
            raise typer.Exit(code=1)
    else:
        if not report.comparison.metrics.valid or not report.passed:
            raise typer.Exit(code=1)


__all__ = ["app", "default_work_dir"]
