"""Recursive self-improvement optimization loop with SSD3 durability and resume."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
import contextlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from pydantic import ValidationError

from kayakgen.model.hull import Hull

from rsi.baseline import DEFAULT_REPO_ROOT, BaselineReference, load_baseline_reference
from rsi.diagnose import (
    PARAMETER_GROUPS,
    AdaptationHints,
    DiagnosticReport,
    aggregate_diagnostics,
    diagnose_candidate,
    diagnose_validation_error,
)
from rsi.evaluator import (
    CandidateMetrics,
    LoadEvalMetrics,
    evaluate_tier0_tier1,
)
from rsi.fidelity import (
    CfdMetrics,
    SeakeepingMetrics,
    evaluate_tier2_cfd,
    evaluate_tier2_seakeeping,
)
from rsi.gates import GATES_VERSION, GateReport, GateViolation, evaluate_gates
from rsi.genome import DesignSpace, hull_to_vector, vector_to_hull

SCHEMA_VERSION = "1"
PARETO_OBJECTIVES: tuple[str, ...] = (
    "sr_closeness_to_10_9",
    "speed_gain_kn",
    "stability_min_margin",
    "fairness_margin",
    "seakeeping_gain",
)


class ResumeStateError(ValueError):
    """Raised when campaign checkpoint artifacts are missing, corrupt, or incompatible."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reject_nonfinite(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant rejected: {token!r}")


def _load_json_strict(path: Path) -> Any:
    if not path.is_file():
        raise ResumeStateError(f"Required campaign file missing: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw, parse_constant=_reject_nonfinite)
    except Exception as exc:  # noqa: BLE001
        raise ResumeStateError(f"Corrupt JSON in {path}: {exc}") from exc


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write strict finite JSON (allow_nan=False) with fsync + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def append_jsonl_atomic(path: Path, record: dict[str, Any]) -> None:
    """Append a single strict JSON line (allow_nan=False) and fsync."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, allow_nan=False) + "\n"
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _git_head_sha(repo_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _git_is_dirty(repo_dir: Path) -> bool | None:
    """True when tracked files differ from HEAD, so a recorded SHA does not describe the code that ran."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            return bool(proc.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def _collect_provenance(repo_root: Path, space: DesignSpace, baseline: BaselineReference, seed: int) -> dict[str, Any]:
    import kayakgen

    kayak_gen_root = Path(kayakgen.__file__).resolve().parents[1]
    space_json = json.dumps(space.to_dict(), sort_keys=True, allow_nan=False)
    space_hash = hashlib.sha256(space_json.encode("utf-8")).hexdigest()[:16]
    return {
        "seed": int(seed),
        "kayak_10_9_sha": _git_head_sha(repo_root),
        "kayak_10_9_dirty": _git_is_dirty(repo_root),
        "kayak_gen_sha": _git_head_sha(kayak_gen_root),
        "kayak_gen_dirty": _git_is_dirty(kayak_gen_root),
        "gates_version": GATES_VERSION,
        "design_space_hash": space_hash,
        "baseline_design_hash": baseline.design_hash,
    }


def _composite_utility(objectives: dict[str, float]) -> float:
    sr_c = float(objectives.get("sr_closeness_to_10_9", -10.0))
    spd = float(objectives.get("speed_gain_kn", -10.0))
    stab = float(objectives.get("stability_min_margin", -10.0))
    fair = float(objectives.get("fairness_margin", -10.0))
    seak = float(objectives.get("seakeeping_gain", 0.0))
    val = 120.0 * sr_c + 2500.0 * spd + 15.0 * min(stab, 0.05) + 10.0 * min(fair, 0.10) + 50.0 * seak
    return round(val if math.isfinite(val) else -1e6, 6)


def _violation_deficit(gate_report: GateReport) -> float:
    if gate_report.passed:
        return 0.0
    total = 0.0
    for v in gate_report.violations:
        if v.margin is not None and math.isfinite(float(v.margin)):
            total += max(0.0, -float(v.margin))
        else:
            total += 1.0
        total += 0.25
    return round(total, 6)


def _dominates(obj_a: dict[str, float], obj_b: dict[str, float]) -> bool:
    ge_all = True
    gt_one = False
    for key in PARETO_OBJECTIVES:
        va = float(obj_a.get(key, 0.0))
        vb = float(obj_b.get(key, 0.0))
        if va < vb - 1e-12:
            ge_all = False
            break
        if va > vb + 1e-12:
            gt_one = True
    return ge_all and gt_one


def _update_pareto_frontier(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute non-dominated feasible frontier, deduped by design_hash."""
    by_hash: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("status") != "evaluated":
            continue
        d_hash = str(entry.get("design_hash", ""))
        if not d_hash:
            continue
        prev = by_hash.get(d_hash)
        has_tier2 = entry.get("tier2_seakeeping") is not None or entry.get("tier2_cfd") is not None
        if prev is None or has_tier2:
            by_hash[d_hash] = entry

    unique = [
        e for e in by_hash.values() if bool((e.get("gate_report") or {}).get("passed"))
    ]
    frontier: list[dict[str, Any]] = []
    for cand in unique:
        c_obj = cand["gate_report"]["objectives"]
        if any(_dominates(other["gate_report"]["objectives"], c_obj) for other in unique if other is not cand):
            continue
        frontier.append(cand)

    frontier.sort(
        key=lambda e: (
            1 if e.get("tier2_passed") else 0,
            _composite_utility(e["gate_report"]["objectives"]),
            str(e.get("candidate_id", "")),
        ),
        reverse=True,
    )
    return frontier


def _metrics_summary_for_ledger(metrics: CandidateMetrics) -> dict[str, Any]:
    return metrics.to_dict()


def _metrics_from_dict(raw: dict[str, Any]) -> CandidateMetrics:
    loads_raw = raw.get("loads")
    loads_obj: dict[int, LoadEvalMetrics] | None = None
    if isinstance(loads_raw, dict):
        loads_obj = {}
        for k, v in loads_raw.items():
            lb = int(k)
            gz = tuple(float(x) for x in v["gz_mm"])
            loads_obj[lb] = LoadEvalMetrics(
                load_lb=lb,
                t_cm=float(v["t_cm"]),
                draft_mm=float(v["draft_mm"]),
                bwl_cm=float(v["bwl_cm"]),
                wet_m2=float(v["wet_m2"]),
                gm_cm=float(v["gm_cm"]),
                gz_mm=(gz[0], gz[1], gz[2]),
                lcg_m=float(v["lcg_m"]),
            )
    kn_raw = raw.get("kn_225")
    kn_obj = {int(k): float(v) for k, v in kn_raw.items()} if isinstance(kn_raw, dict) else None
    kn_ex_raw = raw.get("kn_225_exact")
    kn_ex_obj = {int(k): float(v) for k, v in kn_ex_raw.items()} if isinstance(kn_ex_raw, dict) else None
    mesh_res = tuple(int(x) for x in raw.get("mesh_resolution", (441, 129)))
    return CandidateMetrics(
        valid=bool(raw.get("valid", False)),
        error=raw.get("error"),
        design_hash=str(raw.get("design_hash", "")),
        hull_json_path=str(raw.get("hull_json_path", "")),
        mesh_resolution=(mesh_res[0], mesh_res[1]),
        elapsed_s=float(raw.get("elapsed_s", 0.0)),
        skip_resistance=bool(raw.get("skip_resistance", False)),
        kg_equilibrium_draft_m=raw.get("kg_equilibrium_draft_m"),
        kg_sinkage_m=raw.get("kg_sinkage_m"),
        kg_wetted_surface_m2=raw.get("kg_wetted_surface_m2"),
        kg_gm0_m=raw.get("kg_gm0_m"),
        kg_lcb_frac=raw.get("kg_lcb_frac"),
        kg_rt_N=raw.get("kg_rt_N"),
        waterline_hollow_max_deg=raw.get("waterline_hollow_max_deg"),
        plan_hollow_max_deg=raw.get("plan_hollow_max_deg"),
        upper_hollow_max_deg=raw.get("upper_hollow_max_deg"),
        upper_knuckle_max_deg=raw.get("upper_knuckle_max_deg"),
        waterline_knuckle_max_deg=raw.get("waterline_knuckle_max_deg"),
        foot_clear_height_m=raw.get("foot_clear_height_m"),
        foot_clear_width_m=raw.get("foot_clear_width_m"),
        catch_zone_max_beam_m=raw.get("catch_zone_max_beam_m"),
        catch_zone_max_upper_beam_m=raw.get("catch_zone_max_upper_beam_m"),
        kg_high_angle_gz_mm=raw.get("kg_high_angle_gz_mm"),
        is_watertight=bool(raw.get("is_watertight", False)),
        is_volume=bool(raw.get("is_volume", False)),
        n_faces=int(raw.get("n_faces", 0)),
        outer_stl_sha256=raw.get("outer_stl_sha256"),
        z_min_m=raw.get("z_min_m"),
        l_oa_m=raw.get("l_oa_m"),
        b_oa_cm=raw.get("b_oa_cm"),
        l4_m=raw.get("l4_m"),
        b4_m=raw.get("b4_m"),
        sr_4in_exact=raw.get("sr_4in_exact"),
        sr_4in=raw.get("sr_4in"),
        usca_4in_pct_exact=raw.get("usca_4in_pct_exact"),
        usca_4in_pct=raw.get("usca_4in_pct"),
        catch_widths_mm=tuple(raw["catch_widths_mm"]) if raw.get("catch_widths_mm") is not None else None,
        catch_beams_225_cm=raw.get("catch_beams_225_cm"),
        transition_dB_per_20cm=(
            tuple(raw["transition_dB_per_20cm"]) if raw.get("transition_dB_per_20cm") is not None else None
        ),
        has_surface_fairing=bool(raw.get("has_surface_fairing", False)),
        fwd_wl_max_turn_deg=raw.get("fwd_wl_max_turn_deg"),
        entry_zone_max_turn_deg=raw.get("entry_zone_max_turn_deg"),
        buttock_0075_peak_deg=raw.get("buttock_0075_peak_deg"),
        forebody_max_keel_groove_mm=raw.get("forebody_max_keel_groove_mm"),
        forebody_min_k=raw.get("forebody_min_k"),
        forebody_max_abs_dk_ds=raw.get("forebody_max_abs_dk_ds"),
        loads=loads_obj,
        kn_225=kn_obj,
        kn_225_exact=kn_ex_obj,
    )


def _gate_report_from_dict(raw: dict[str, Any]) -> GateReport:
    violations = tuple(
        GateViolation(
            gate=str(v["gate"]),
            message=str(v["message"]),
            actual=v.get("actual"),
            limit=v.get("limit"),
            margin=v.get("margin"),
        )
        for v in raw.get("violations", ())
    )
    return GateReport(
        passed=bool(raw.get("passed", False)),
        violations=violations,
        margins={str(k): float(v) for k, v in (raw.get("margins") or {}).items()},
        objectives={str(k): float(v) for k, v in (raw.get("objectives") or {}).items()},
    )


_WORKER_BASELINES: dict[str, BaselineReference] = {}


def _init_eval_worker() -> None:
    """Pin BLAS/OpenMP threads to 1 per worker process to prevent oversubscription."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def _prescreen_and_evaluate(
    hull_json: str,
    cand_dir_str: str,
    repo_root_str: str,
    skip_resistance: bool,
    pre_screen: bool,
) -> CandidateMetrics:
    """Worker-safe evaluation: fast single-pass Tier 0 gate + Tier 1."""
    hull = Hull.model_validate_json(hull_json)
    cand_dir = Path(cand_dir_str)
    cand_dir.mkdir(parents=True, exist_ok=True)
    baseline = _WORKER_BASELINES.get(repo_root_str)
    if baseline is None:
        baseline = load_baseline_reference(Path(repo_root_str))
        _WORKER_BASELINES[repo_root_str] = baseline

    return evaluate_tier0_tier1(
        hull,
        cand_dir,
        baseline,
        skip_resistance=skip_resistance,
        fast_tier0=pre_screen,
    )


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration for a recursive self-improvement campaign."""

    iterations: int = 5
    population: int = 6
    seed: int = 42
    work_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("KAYAK109_WORK_DIR")
            or (Path.home() / ".cache" / "kayak-10-9" / "work")
        )
    )
    tier2_top: int = 0
    workers: int = 1
    resume: bool = False
    seed_hull_path: Path | None = None
    skip_resistance: bool = False
    quick_tier2: bool = False
    cfd_dry_run: bool = False
    tier2_n_jobs: int = 6
    pre_screen: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "population": self.population,
            "seed": self.seed,
            "work_dir": str(self.work_dir),
            "tier2_top": self.tier2_top,
            "workers": self.workers,
            "resume": self.resume,
            "seed_hull_path": str(self.seed_hull_path) if self.seed_hull_path is not None else None,
            "skip_resistance": self.skip_resistance,
            "quick_tier2": self.quick_tier2,
            "cfd_dry_run": self.cfd_dry_run,
            "tier2_n_jobs": self.tier2_n_jobs,
            "pre_screen": self.pre_screen,
        }


@dataclass(frozen=True)
class LedgerEntry:
    """Single durable entry in ledger.jsonl."""

    candidate_id: str
    iteration: int
    index: int
    timestamp_utc: str
    status: str
    design_hash: str
    hull_path: str | None
    vector: list[float]
    metrics: dict[str, Any] | None
    gate_report: dict[str, Any] | None
    diagnosis: dict[str, Any]
    provenance: dict[str, Any]
    utility_score: float = -1e6
    violation_deficit: float = 0.0
    tier2_seakeeping: dict[str, Any] | None = None
    tier2_cfd: dict[str, Any] | None = None
    tier2_passed: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignState:
    """Checkpoint state stored in state.json."""

    status: str
    seed: int
    rng_state: dict[str, Any]
    target_iterations: int
    completed_iterations: int
    population: int
    tier2_top: int
    tier2_completed: bool
    baseline_evaluated: bool
    mu: list[float]
    sigma: list[float]
    evaluated_hashes: list[str]
    provenance: dict[str, Any]
    adaptation_hints: dict[str, Any] = field(default_factory=lambda: AdaptationHints().to_dict())
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignResult:
    """Summary returned by run_campaign and run_smoke."""

    work_dir: str
    status: str
    seed: int
    completed_iterations: int
    target_iterations: int
    total_evaluations: int
    feasible_count: int
    pareto_count: int
    best_candidate_id: str | None
    best_design_hash: str | None
    best_hull_path: str | None
    pareto_front: list[dict[str, Any]]
    tier2_verified: dict[str, Any]
    provenance: dict[str, Any]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _propose_candidate_vectors(
    rng: np.random.Generator,
    population: int,
    space: DesignSpace,
    mu: list[float],
    sigma: list[float],
    hints: AdaptationHints,
    pareto_vectors: list[list[float]],
    best_near_vec: list[float] | None,
) -> list[tuple[list[float], str]]:
    """Deterministically propose candidate vectors using multi-strategy RSI search."""
    d_dim = len(space.params)
    by_group: dict[str, list[int]] = {g: [] for g in PARAMETER_GROUPS}
    for idx, spec in enumerate(space.params):
        by_group.setdefault(spec.group, []).append(idx)

    group_scales = hints.group_scales
    eff_sigma = [
        float(sigma[i]) * float(group_scales.get(spec.group, 1.0))
        for i, spec in enumerate(space.params)
    ]

    proposals: list[tuple[list[float], str]] = []
    max_attempts = max(population * 3, population + 4)

    for attempt in range(max_attempts):
        if len(proposals) >= max_attempts:
            break

        # Choose rank-weighted elite parent vector
        if pareto_vectors:
            if rng.random() < 0.70:
                p_weights = np.array(
                    [1.0 / (k + 2.0) for k in range(len(pareto_vectors))], dtype=float
                )
                p_weights /= p_weights.sum()
                p_idx = int(rng.choice(len(pareto_vectors), p=p_weights))
                parent = list(pareto_vectors[p_idx])
            else:
                parent = list(mu)
        elif best_near_vec is not None and rng.random() < 0.65:
            parent = list(best_near_vec)
        else:
            parent = list(mu)

        u = float(rng.random())
        cand = list(parent)

        if u < 0.42:
            # Sparse group-local / coordinate perturbation (preserves tight coupled margins)
            strategy = "sparse_group"
            if rng.random() < 0.60:
                n_grp = int(rng.integers(1, 3))
                chosen_groups = rng.choice(PARAMETER_GROUPS, size=n_grp, replace=False)
                active_idx = [i for g in chosen_groups for i in by_group.get(str(g), [])]
                if len(active_idx) > 5:
                    k_sub = int(rng.integers(2, 6))
                    active_idx = [int(x) for x in rng.choice(active_idx, size=k_sub, replace=False)]
            else:
                k_sub = int(rng.integers(2, 6))
                active_idx = [int(x) for x in rng.choice(d_dim, size=k_sub, replace=False)]

            for idx in active_idx:
                cand[idx] += float(rng.normal(0.0, 1.0)) * eff_sigma[idx]

        elif u < 0.78:
            # Diagnostic-guided directional step + small orthogonal jitter
            strategy = "diagnostic_steered"
            dirs = hints.preferred_directions
            for idx, spec in enumerate(space.params):
                d_val = float(dirs.get(spec.name, 0.0))
                if abs(d_val) > 0.05:
                    step_mag = float(rng.uniform(0.30, 1.15))
                    cand[idx] += d_val * eff_sigma[idx] * step_mag
                elif spec.name == "draft_a1" and "draft_a1" in hints.avoid_bounds:
                    cand[idx] *= 0.25
            k_jit = int(rng.integers(1, 5))
            jit_idx = [int(x) for x in rng.choice(d_dim, size=k_jit, replace=False)]
            for idx in jit_idx:
                cand[idx] += float(rng.normal(0.0, 0.45)) * eff_sigma[idx]

        else:
            # Subspace elite crossover + light local jitter
            strategy = "elite_crossover"
            if len(pareto_vectors) >= 2:
                p2_weights = np.array(
                    [1.0 / (k + 2.0) for k in range(len(pareto_vectors))], dtype=float
                )
                p2_weights /= p2_weights.sum()
                p2_idx = int(rng.choice(len(pareto_vectors), p=p2_weights))
                other = pareto_vectors[p2_idx]
                k_mask = int(rng.integers(4, min(10, d_dim + 1)))
                mask_idx = [int(x) for x in rng.choice(d_dim, size=k_mask, replace=False)]
                alpha = float(rng.uniform(0.35, 0.85))
                for idx in mask_idx:
                    cand[idx] = parent[idx] + alpha * (other[idx] - parent[idx])
            k_jit = int(rng.integers(2, min(5, d_dim + 1)))
            jit_idx = [int(x) for x in rng.choice(d_dim, size=k_jit, replace=False)]
            for idx in jit_idx:
                cand[idx] += float(rng.normal(0.0, 0.30)) * eff_sigma[idx]

        # Respect diagnostic avoid_bounds hints softly before clamping
        for idx, spec in enumerate(space.params):
            rule = hints.avoid_bounds.get(spec.name)
            if not rule:
                continue
            side = rule.get("side")
            lim = rule.get("limit")
            if side == "away_from_nonzero" and isinstance(lim, (int, float)):
                if abs(cand[idx]) > float(lim):
                    cand[idx] = math.copysign(float(lim) * 0.25, cand[idx])
            elif side == "lower" and isinstance(lim, (int, float)) and cand[idx] < float(lim):
                cand[idx] = float(lim) + 0.1 * spec.scale
            elif side == "upper" and isinstance(lim, (int, float)) and cand[idx] > float(lim):
                cand[idx] = float(lim) - 0.1 * spec.scale

        clamped = space.clamp(cand)
        # Ensure we didn't produce an exact zero-step copy of parent on attempt
        if attempt < max_attempts - 1 and all(abs(a - b) <= 1e-12 for a, b in zip(clamped, parent)):
            idx = int(rng.integers(0, d_dim))
            clamped[idx] = space.clamp(
                [
                    clamped[j] + (eff_sigma[j] * (1.0 if rng.random() >= 0.5 else -1.0) if j == idx else 0.0)
                    for j in range(d_dim)
                ]
            )[idx]

        proposals.append((clamped, strategy))

    return proposals


def _validate_and_load_resume_state(
    work_dir: Path,
    space: DesignSpace,
    baseline: BaselineReference,
) -> tuple[CampaignState, list[dict[str, Any]], dict[str, Any]]:
    state_path = work_dir / "state.json"
    pareto_path = work_dir / "pareto.json"
    ledger_path = work_dir / "ledger.jsonl"

    state_raw = _load_json_strict(state_path)
    pareto_raw = _load_json_strict(pareto_path)
    if not ledger_path.is_file():
        raise ResumeStateError(f"Required campaign file missing: {ledger_path}")

    if not isinstance(state_raw, dict) or state_raw.get("schema_version") != SCHEMA_VERSION:
        raise ResumeStateError("Invalid or unsupported state.json schema_version")
    if not isinstance(pareto_raw, dict) or pareto_raw.get("schema_version") != SCHEMA_VERSION:
        raise ResumeStateError("Invalid or unsupported pareto.json schema_version")

    required_state = (
        "seed",
        "rng_state",
        "completed_iterations",
        "mu",
        "sigma",
        "evaluated_hashes",
        "provenance",
    )
    for k in required_state:
        if k not in state_raw:
            raise ResumeStateError(f"Missing required key {k!r} in state.json")

    prov = state_raw["provenance"]
    if not isinstance(prov, dict) or prov.get("baseline_design_hash") != baseline.design_hash:
        raise ResumeStateError(
            "Checkpoint baseline_design_hash mismatch or missing provenance in state.json"
        )

    mu = state_raw["mu"]
    sigma = state_raw["sigma"]
    if (
        not isinstance(mu, list)
        or not isinstance(sigma, list)
        or len(mu) != len(space.params)
        or len(sigma) != len(space.params)
    ):
        raise ResumeStateError("Checkpoint mu/sigma vector length mismatch in state.json")

    try:
        mu_f = [float(x) for x in mu]
        sigma_f = [float(x) for x in sigma]
    except (TypeError, ValueError) as exc:
        raise ResumeStateError(f"Non-numeric mu/sigma in state.json: {exc}") from exc
    if not all(math.isfinite(x) for x in mu_f + sigma_f):
        raise ResumeStateError("Non-finite mu/sigma in state.json")

    # Validate PCG64 RNG state
    try:
        bit_gen = np.random.PCG64()
        bit_gen.state = state_raw["rng_state"]
    except Exception as exc:  # noqa: BLE001
        raise ResumeStateError(f"Corrupt rng_state in state.json: {exc}") from exc

    ledger_entries: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped, parse_constant=_reject_nonfinite)
            if not isinstance(rec, dict) or "candidate_id" not in rec:
                raise ValueError(f"Malformed ledger record at line {line_no}")
            ledger_entries.append(rec)
    except Exception as exc:  # noqa: BLE001
        raise ResumeStateError(f"Corrupt ledger.jsonl: {exc}") from exc

    state = CampaignState(
        status=str(state_raw.get("status", "running")),
        seed=int(state_raw["seed"]),
        rng_state=dict(state_raw["rng_state"]),
        target_iterations=int(state_raw.get("target_iterations", 0)),
        completed_iterations=int(state_raw["completed_iterations"]),
        population=int(state_raw.get("population", 1)),
        tier2_top=int(state_raw.get("tier2_top", 0)),
        tier2_completed=bool(state_raw.get("tier2_completed", False)),
        baseline_evaluated=bool(state_raw.get("baseline_evaluated", False)),
        mu=mu_f,
        sigma=sigma_f,
        evaluated_hashes=[str(h) for h in state_raw["evaluated_hashes"]],
        provenance=dict(prov),
        adaptation_hints=dict(state_raw.get("adaptation_hints") or AdaptationHints().to_dict()),
    )
    return state, ledger_entries, pareto_raw


def _write_pareto_and_best(
    work_dir: Path,
    ledger_entries: Sequence[dict[str, Any]],
    tier2_verified: dict[str, Any],
    provenance: dict[str, Any],
    baseline: BaselineReference,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    frontier = _update_pareto_frontier(ledger_entries)
    best_entry: dict[str, Any] | None = None
    non_base = [e for e in frontier if e.get("design_hash") != baseline.design_hash]
    if non_base:
        best_entry = non_base[0]
    elif frontier:
        best_entry = frontier[0]
    else:
        # Fallback to baseline evaluated entry if present
        for e in ledger_entries:
            if e.get("status") == "evaluated" and e.get("design_hash") == baseline.design_hash:
                best_entry = e
                break

    best_hull = baseline.hull
    if best_entry is not None and best_entry.get("hull_path"):
        hp = Path(str(best_entry["hull_path"]))
        if hp.is_file():
            best_hull = Hull.model_validate_json(hp.read_text(encoding="utf-8"))

    best_hull_path = work_dir / "best_candidate.hull.json"
    tmp_hull = best_hull_path.with_name(f".best_candidate.hull.json.{os.getpid()}.tmp")
    with open(tmp_hull, "w", encoding="utf-8") as fh:
        fh.write(best_hull.model_dump_json(indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_hull, best_hull_path)

    pareto_payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": _utc_now(),
        "provenance": provenance,
        "objectives": list(PARETO_OBJECTIVES),
        "frontier": frontier,
        "best_candidate_id": best_entry.get("candidate_id") if best_entry else None,
        "best_design_hash": best_entry.get("design_hash") if best_entry else baseline.design_hash,
        "tier2_verified": tier2_verified,
    }
    write_json_atomic(work_dir / "pareto.json", pareto_payload)
    return frontier, best_entry


def run_campaign(
    config: CampaignConfig,
    *,
    repo_root: Path | None = None,
    evaluator_fn: Callable[[Hull, Path, BaselineReference], CandidateMetrics] | None = None,
    seakeeping_fn: Callable[[Hull, Path, BaselineReference], SeakeepingMetrics] | None = None,
    cfd_fn: Callable[[Hull, Path, BaselineReference], CfdMetrics] | None = None,
) -> CampaignResult:
    """Execute or resume a deterministic multi-iteration RSI campaign on SSD3."""
    if config.iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {config.iterations}")
    if config.population < 1:
        raise ValueError(f"population must be >= 1, got {config.population}")
    if config.tier2_top < 0:
        raise ValueError(f"tier2_top must be >= 0, got {config.tier2_top}")
    if config.workers < 1:
        raise ValueError(f"workers must be >= 1, got {config.workers}")

    root = DEFAULT_REPO_ROOT if repo_root is None else Path(repo_root).resolve()
    work = Path(config.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cand_root = work / "candidates"
    cand_root.mkdir(parents=True, exist_ok=True)
    ledger_path = work / "ledger.jsonl"

    baseline = load_baseline_reference(root)
    space = DesignSpace.default()
    provenance = _collect_provenance(root, space, baseline, config.seed)

    def _eval_one(hull: Hull, c_dir: Path) -> CandidateMetrics:
        if evaluator_fn is not None:
            return evaluator_fn(hull, c_dir, baseline)
        return _prescreen_and_evaluate(
            hull.model_dump_json(),
            str(c_dir),
            str(root),
            config.skip_resistance,
            config.pre_screen,
        )

    ledger_entries: list[dict[str, Any]] = []
    tier2_verified: dict[str, Any] = {}
    effective_seed = config.seed
    bit_gen = np.random.PCG64(effective_seed)
    rng = np.random.Generator(bit_gen)

    if config.resume:
        state, ledger_entries, pareto_raw = _validate_and_load_resume_state(work, space, baseline)
        stored_gates = str(state.provenance.get("gates_version", "1"))
        if stored_gates != GATES_VERSION:
            raise ResumeStateError(
                f"Checkpoint verdicts were produced under gates version {stored_gates}; this harness "
                f"is at gates version {GATES_VERSION}. The stored pass/fail flags are stale and would "
                "seed the search with hulls the current gates reject. Start a new campaign in a "
                "fresh work directory."
            )
        effective_seed = state.seed
        bit_gen.state = state.rng_state
        rng = np.random.Generator(bit_gen)
        mu = list(state.mu)
        sigma = list(state.sigma)
        evaluated_hashes = set(state.evaluated_hashes)
        start_iter = state.completed_iterations + 1
        tier2_verified = dict(pareto_raw.get("tier2_verified") or {})
        baseline_evaluated = state.baseline_evaluated
        created_utc = str(state.provenance.get("created_utc", _utc_now()))
        current_hints = AdaptationHints(
            group_scales=dict(state.adaptation_hints.get("group_scales") or {g: 1.0 for g in PARAMETER_GROUPS}),
            preferred_directions=dict(state.adaptation_hints.get("preferred_directions") or {}),
            avoid_bounds=dict(state.adaptation_hints.get("avoid_bounds") or {}),
            tight_margins=tuple(state.adaptation_hints.get("tight_margins") or ()),
        )
        provenance = {
            **state.provenance,
            **provenance,
            "seed": effective_seed,
            "created_utc": created_utc,
        }
        final_completed = max(state.completed_iterations, config.iterations)
        prior_tier2_completed = state.tier2_completed
    else:
        final_completed = config.iterations
        prior_tier2_completed = False
        if ledger_path.exists():
            ledger_path.unlink()
        created_utc = _utc_now()
        provenance["created_utc"] = created_utc
        mu = space.baseline_vector()
        sigma = [0.16 * p.scale for p in space.params]
        evaluated_hashes = set()
        start_iter = 1
        baseline_evaluated = False
        current_hints = AdaptationHints()

        # Optional custom seed hull (e.g., infeasible seed recovery)
        seed_hull: Hull | None = None
        if config.seed_hull_path is not None:
            sp_path = Path(config.seed_hull_path)
            if not sp_path.is_file():
                raise FileNotFoundError(f"Seed hull file not found: {sp_path}")
            seed_hull = Hull.model_validate_json(sp_path.read_text(encoding="utf-8"))
            mu = hull_to_vector(seed_hull, space)

        # Evaluate baseline anchor first
        base_dir = cand_root / "baseline"
        base_metrics = _eval_one(baseline.hull, base_dir)
        base_gate = evaluate_gates(base_metrics, baseline)
        base_vec = space.baseline_vector()
        base_diag = diagnose_candidate(base_metrics, base_gate, vector=base_vec, space=space)
        base_hull_file = base_dir / "candidate.hull.json"
        if not base_hull_file.is_file():
            base_dir.mkdir(parents=True, exist_ok=True)
            base_hull_file.write_text(baseline.hull.model_dump_json(indent=2) + "\n")
        base_entry = LedgerEntry(
            candidate_id="baseline",
            iteration=0,
            index=0,
            timestamp_utc=_utc_now(),
            status="evaluated",
            design_hash=baseline.design_hash,
            hull_path=str(base_hull_file),
            vector=base_vec,
            metrics=_metrics_summary_for_ledger(base_metrics),
            gate_report=base_gate.to_dict(),
            diagnosis=base_diag.to_dict(),
            provenance=provenance,
            utility_score=_composite_utility(base_gate.objectives),
            violation_deficit=_violation_deficit(base_gate),
        ).to_dict()
        append_jsonl_atomic(ledger_path, base_entry)
        ledger_entries.append(base_entry)
        evaluated_hashes.add(baseline.design_hash)
        baseline_evaluated = True
        if seed_hull is None:
            current_hints = base_diag.hints

        # Evaluate seed_hull if distinct from baseline
        if seed_hull is not None and seed_hull.design_hash() != baseline.design_hash:
            seed_dir = cand_root / "seed_hull"
            seed_metrics = _eval_one(seed_hull, seed_dir)
            seed_gate = evaluate_gates(seed_metrics, baseline)
            seed_vec = hull_to_vector(seed_hull, space)
            seed_diag = diagnose_candidate(seed_metrics, seed_gate, vector=seed_vec, space=space)
            seed_file = seed_dir / "candidate.hull.json"
            if not seed_file.is_file():
                seed_dir.mkdir(parents=True, exist_ok=True)
                seed_file.write_text(seed_hull.model_dump_json(indent=2) + "\n")
            seed_entry = LedgerEntry(
                candidate_id="seed_hull",
                iteration=0,
                index=1,
                timestamp_utc=_utc_now(),
                status="evaluated",
                design_hash=seed_hull.design_hash(),
                hull_path=str(seed_file),
                vector=seed_vec,
                metrics=_metrics_summary_for_ledger(seed_metrics),
                gate_report=seed_gate.to_dict(),
                diagnosis=seed_diag.to_dict(),
                provenance=provenance,
                utility_score=_composite_utility(seed_gate.objectives),
                violation_deficit=_violation_deficit(seed_gate),
            ).to_dict()
            append_jsonl_atomic(ledger_path, seed_entry)
            ledger_entries.append(seed_entry)
            evaluated_hashes.add(seed_hull.design_hash())
            current_hints = seed_diag.hints

    # Save initial state before iteration loop
    def _save_checkpoint(completed_iter: int, status_str: str, tier2_done: bool) -> None:
        prov_now = {**provenance, "updated_utc": _utc_now()}
        st_obj = CampaignState(
            status=status_str,
            seed=effective_seed,
            rng_state=dict(bit_gen.state),
            target_iterations=final_completed,
            completed_iterations=completed_iter,
            population=config.population,
            tier2_top=config.tier2_top,
            tier2_completed=tier2_done,
            baseline_evaluated=baseline_evaluated,
            mu=[float(x) for x in mu],
            sigma=[float(x) for x in sigma],
            evaluated_hashes=sorted(evaluated_hashes),
            provenance=prov_now,
            adaptation_hints=current_hints.to_dict(),
        )
        write_json_atomic(work / "state.json", st_obj.to_dict())

    if not config.resume:
        _write_pareto_and_best(work, ledger_entries, tier2_verified, provenance, baseline)
        _save_checkpoint(0, "running", False)

    # Track best utility and lowest deficit across history
    best_utility = max(
        (
            float(e.get("utility_score", -1e6))
            for e in ledger_entries
            if e.get("status") == "evaluated" and (e.get("gate_report") or {}).get("passed")
        ),
        default=-1e6,
    )
    stagnation_iters = 0

    with contextlib.ExitStack() as pool_stack:
        eval_pool: ProcessPoolExecutor | None = None
        if config.workers > 1 and evaluator_fn is None and start_iter <= config.iterations:
            eval_pool = pool_stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=min(config.workers, config.population),
                    mp_context=multiprocessing.get_context("forkserver"),
                    initializer=_init_eval_worker,
                )
            )
        for it in range(start_iter, config.iterations + 1):
            # Gather current Pareto vectors (preferring non-baseline feasible) and best near-feasible
            pareto_now = _update_pareto_frontier(ledger_entries)
            non_base_pareto = [e for e in pareto_now if e.get("design_hash") != baseline.design_hash]
            if non_base_pareto:
                pareto_source = non_base_pareto
            elif config.seed_hull_path is not None and it <= 2:
                # Allow diagnostic repair from custom seed_hull before falling back to baseline anchor
                pareto_source = []
            else:
                pareto_source = pareto_now
            pareto_vecs = [
                list(e["vector"])
                for e in pareto_source
                if isinstance(e.get("vector"), list) and len(e["vector"]) == len(space.params)
            ]

            near_entries = [
                e
                for e in ledger_entries
                if e.get("status") == "evaluated"
                and not (e.get("gate_report") or {}).get("passed")
                and isinstance(e.get("vector"), list)
                and len(e["vector"]) == len(space.params)
            ]
            best_near_vec: list[float] | None = None
            if near_entries:
                best_near = min(near_entries, key=lambda e: float(e.get("violation_deficit", 1e6)))
                best_near_vec = list(best_near["vector"])

            # Step 1: Sequentially propose & validate up to P distinct valid Hull candidates
            raw_proposals = _propose_candidate_vectors(
                rng,
                config.population,
                space,
                mu,
                sigma,
                current_hints,
                pareto_vecs,
                best_near_vec,
            )

            valid_batch: list[tuple[int, str, list[float], Hull, str]] = []
            iter_diag_reports: list[DiagnosticReport] = []
            err_counter = 0

            for vec, strategy in raw_proposals:
                if len(valid_batch) >= config.population:
                    break
                try:
                    cand_hull = vector_to_hull(vec, baseline.hull, space, clamp=True)
                except (ValidationError, ValueError) as exc:
                    err_counter += 1
                    diag_rep = diagnose_validation_error(exc, vector=vec, space=space)
                    iter_diag_reports.append(diag_rep)
                    err_entry = LedgerEntry(
                        candidate_id=f"iter{it:03d}_err{err_counter:03d}",
                        iteration=it,
                        index=-err_counter,
                        timestamp_utc=_utc_now(),
                        status="validation_error",
                        design_hash="",
                        hull_path=None,
                        vector=[float(x) for x in vec],
                        metrics=None,
                        gate_report=None,
                        diagnosis=diag_rep.to_dict(),
                        provenance={**provenance, "strategy": strategy},
                        utility_score=-1e6,
                        violation_deficit=10.0,
                    ).to_dict()
                    append_jsonl_atomic(ledger_path, err_entry)
                    ledger_entries.append(err_entry)
                    continue

                d_hash = cand_hull.design_hash()
                if d_hash in evaluated_hashes:
                    err_counter += 1
                    dup_entry = LedgerEntry(
                        candidate_id=f"iter{it:03d}_dup{err_counter:03d}",
                        iteration=it,
                        index=-err_counter,
                        timestamp_utc=_utc_now(),
                        status="duplicate_skipped",
                        design_hash=d_hash,
                        hull_path=None,
                        vector=[float(x) for x in vec],
                        metrics=None,
                        gate_report=None,
                        diagnosis=DiagnosticReport(
                            feasible=False,
                            failure_modes=(),
                            primary_groups=(),
                            items=(),
                            hints=AdaptationHints(),
                            summary="Duplicate design_hash skipped before geometry evaluation.",
                        ).to_dict(),
                        provenance={**provenance, "strategy": strategy},
                    ).to_dict()
                    append_jsonl_atomic(ledger_path, dup_entry)
                    ledger_entries.append(dup_entry)
                    continue

                evaluated_hashes.add(d_hash)
                idx_in_pop = len(valid_batch)
                cand_id = f"iter{it:03d}_cand{idx_in_pop:03d}"
                valid_batch.append((idx_in_pop, cand_id, vec, cand_hull, strategy))

            # Step 2: Evaluate batch (using persistent worker pool when workers > 1)
            batch_metrics: list[CandidateMetrics] = []
            if valid_batch:
                if eval_pool is not None:
                    futures = [
                        eval_pool.submit(
                            _prescreen_and_evaluate,
                            cand_hull.model_dump_json(),
                            str(cand_root / cand_id),
                            str(root),
                            config.skip_resistance,
                            config.pre_screen,
                        )
                        for _, cand_id, _, cand_hull, _ in valid_batch
                    ]
                    batch_metrics = [f.result() for f in futures]
                else:
                    for _, cand_id, _, cand_hull, _ in valid_batch:
                        c_dir = cand_root / cand_id
                        batch_metrics.append(_eval_one(cand_hull, c_dir))

            # Step 3: Gate, diagnose, and archive in deterministic order
            iter_feasible: list[dict[str, Any]] = []
            iter_evaluated: list[dict[str, Any]] = []
            for (idx_in_pop, cand_id, vec, cand_hull, strategy), m in zip(valid_batch, batch_metrics):
                c_dir = cand_root / cand_id
                c_hull_file = c_dir / "candidate.hull.json"
                if not c_hull_file.is_file():
                    c_dir.mkdir(parents=True, exist_ok=True)
                    c_hull_file.write_text(cand_hull.model_dump_json(indent=2) + "\n")

                gate_rep = evaluate_gates(m, baseline)
                diag_rep = diagnose_candidate(m, gate_rep, vector=vec, space=space)
                iter_diag_reports.append(diag_rep)

                util = _composite_utility(gate_rep.objectives)
                deficit = _violation_deficit(gate_rep)
                entry = LedgerEntry(
                    candidate_id=cand_id,
                    iteration=it,
                    index=idx_in_pop,
                    timestamp_utc=_utc_now(),
                    status="evaluated",
                    design_hash=cand_hull.design_hash(),
                    hull_path=str(c_hull_file),
                    vector=[round(float(x), 10) for x in vec],
                    metrics=_metrics_summary_for_ledger(m),
                    gate_report=gate_rep.to_dict(),
                    diagnosis=diag_rep.to_dict(),
                    provenance={**provenance, "strategy": strategy},
                    utility_score=util,
                    violation_deficit=deficit,
                ).to_dict()
                append_jsonl_atomic(ledger_path, entry)
                ledger_entries.append(entry)
                iter_evaluated.append(entry)
                if gate_rep.passed:
                    iter_feasible.append(entry)

            # Step 4: Adapt search distribution (mu, sigma, hints)
            if iter_diag_reports:
                current_hints = aggregate_diagnostics(iter_diag_reports, space)

            iter_best_util = max((float(e["utility_score"]) for e in iter_feasible), default=-1e6)
            feasible_rate = len(iter_feasible) / max(len(valid_batch), 1)
            prev_best_utility = best_utility
            improved = bool(iter_feasible and iter_best_util > prev_best_utility + 1e-9)

            if iter_feasible and iter_best_util >= best_utility:
                best_utility = iter_best_util
                sorted_feas = sorted(iter_feasible, key=lambda e: float(e["utility_score"]), reverse=True)
                top_elites = sorted_feas[: max(1, min(3, len(sorted_feas)))]
                weights = [1.0 / (k + 1) for k in range(len(top_elites))]
                w_sum = sum(weights)
                elite_mu = [
                    sum(w * float(e["vector"][j]) for w, e in zip(weights, top_elites)) / w_sum
                    for j in range(len(space.params))
                ]
                mu = space.clamp([0.65 * elite_mu[j] + 0.35 * mu[j] for j in range(len(space.params))])
            elif not non_base_pareto and iter_evaluated:
                # Move mu toward lowest violation deficit when repairing infeasible region
                best_near_iter = min(iter_evaluated, key=lambda e: float(e["violation_deficit"]))
                mu = space.clamp(
                    [
                        0.50 * float(best_near_iter["vector"][j]) + 0.50 * mu[j]
                        for j in range(len(space.params))
                    ]
                )

            stagnation_iters = 0 if improved else (stagnation_iters + 1)
            step_mult = 1.12 if improved else (0.82 if feasible_rate < 0.15 else 0.94)
            for j, spec in enumerate(space.params):
                g_scale = float(current_hints.group_scales.get(spec.group, 1.0))
                new_sig = sigma[j] * step_mult * (0.75 + 0.25 * g_scale)
                sigma[j] = min(0.40 * spec.scale, max(0.03 * spec.scale, new_sig))
            if stagnation_iters >= 10:
                sigma = [max(sigma[j], 0.16 * spec.scale) for j, spec in enumerate(space.params)]
                stagnation_iters = 0

            _write_pareto_and_best(work, ledger_entries, tier2_verified, provenance, baseline)
            _save_checkpoint(it, "running", False)

    # Step 5: Optional Tier 2 top-K verification
    ran_new_iterations = start_iter <= config.iterations
    diagnostic_tier2 = bool(config.cfd_dry_run or config.quick_tier2)
    tier2_done = False
    if config.tier2_top > 0:
        frontier_now = _update_pareto_frontier(ledger_entries)
        non_base_front = [e for e in frontier_now if e.get("design_hash") != baseline.design_hash]
        candidates_to_verify = list(non_base_front if non_base_front else frontier_now)
        candidates_to_verify.sort(
            key=lambda e: (
                _composite_utility((e.get("gate_report") or {}).get("objectives") or {}),
                str(e.get("candidate_id", "")),
            ),
            reverse=True,
        )
        selected = candidates_to_verify[: config.tier2_top]

        for entry in selected:
            d_hash = str(entry["design_hash"])
            cached = tier2_verified.get(d_hash)
            if isinstance(cached, dict):
                c_seak = cached.get("seakeeping") or {}
                c_cfd = cached.get("cfd") or {}
                c_gate = cached.get("gate_report") or {}
                if (
                    not diagnostic_tier2
                    and cached.get("seakeeping_rejected")
                    and c_seak.get("valid")
                    and not c_seak.get("quick")
                ):
                    entry["tier2_seakeeping"] = c_seak
                    entry["tier2_cfd"] = None
                    entry["tier2_passed"] = False
                    if c_gate:
                        entry["gate_report"] = c_gate
                        entry["utility_score"] = _composite_utility(c_gate.get("objectives") or {})
                    if cached.get("diagnosis"):
                        entry["diagnosis"] = cached["diagnosis"]
                    continue
                if (
                    c_seak.get("valid")
                    and bool(c_seak.get("quick")) == bool(config.quick_tier2)
                    and c_cfd.get("valid")
                    and bool(c_cfd.get("dry_run")) == bool(config.cfd_dry_run)
                ):
                    entry["tier2_seakeeping"] = c_seak
                    entry["tier2_cfd"] = c_cfd
                    entry["tier2_passed"] = bool(
                        c_gate.get("passed")
                        and not c_seak.get("quick")
                        and not c_cfd.get("dry_run")
                    )
                    if c_gate:
                        objs = dict(c_gate.get("objectives") or {})
                        if diagnostic_tier2:
                            entry["tier2_gate_report"] = c_gate
                            if cached.get("diagnosis"):
                                entry["tier2_diagnosis"] = cached["diagnosis"]
                        else:
                            entry["gate_report"] = c_gate
                            if cached.get("diagnosis"):
                                entry["diagnosis"] = cached["diagnosis"]
                            entry["utility_score"] = _composite_utility(objs)
                    continue

            hull_path = Path(str(entry["hull_path"]))
            cand_hull = Hull.model_validate_json(hull_path.read_text(encoding="utf-8"))
            t2_dir = work / "tier2" / str(entry["candidate_id"])
            t2_dir.mkdir(parents=True, exist_ok=True)

            if seakeeping_fn is not None:
                seak = seakeeping_fn(cand_hull, t2_dir / "seakeeping", baseline)
            else:
                seak = evaluate_tier2_seakeeping(
                    cand_hull,
                    t2_dir / "seakeeping",
                    baseline,
                    quick=config.quick_tier2,
                    n_jobs=config.tier2_n_jobs,
                )

            cand_metrics = _metrics_from_dict(entry["metrics"])
            seak_gate = evaluate_gates(cand_metrics, baseline, seakeeping=seak)
            if not diagnostic_tier2 and (not seak.valid or not seak_gate.passed):
                seak_diag = diagnose_candidate(
                    cand_metrics,
                    seak_gate,
                    seakeeping=seak,
                    vector=entry.get("vector"),
                    space=space,
                )
                t2_seak_d = seak.to_dict()
                t2_gate_d = seak_gate.to_dict()
                t2_diag_d = seak_diag.to_dict()
                entry["tier2_seakeeping"] = t2_seak_d
                entry["tier2_cfd"] = None
                entry["tier2_passed"] = False
                entry["gate_report"] = t2_gate_d
                entry["diagnosis"] = t2_diag_d
                entry["utility_score"] = _composite_utility(seak_gate.objectives)
                tier2_verified[d_hash] = {
                    "candidate_id": entry["candidate_id"],
                    "design_hash": d_hash,
                    "hull_path": str(hull_path),
                    "seakeeping": t2_seak_d,
                    "cfd": None,
                    "gate_report": t2_gate_d,
                    "diagnosis": t2_diag_d,
                    "tier2_passed": False,
                    "seakeeping_rejected": True,
                }
                t2_ledger = dict(entry)
                t2_ledger["timestamp_utc"] = _utc_now()
                t2_ledger["provenance"] = {**provenance, "stage": "tier2_verification"}
                append_jsonl_atomic(ledger_path, t2_ledger)
                continue

            if cfd_fn is not None:
                cfd = cfd_fn(cand_hull, t2_dir / "cfd", baseline)
            else:
                cfd = evaluate_tier2_cfd(
                    cand_hull,
                    t2_dir / "cfd",
                    baseline,
                    dry_run=config.cfd_dry_run,
                )

            t2_gate = evaluate_gates(cand_metrics, baseline, seakeeping=seak, cfd=cfd)
            t2_diag = diagnose_candidate(
                cand_metrics,
                t2_gate,
                seakeeping=seak,
                cfd=cfd,
                vector=entry.get("vector"),
                space=space,
            )
            t2_seak_d = seak.to_dict()
            t2_cfd_d = cfd.to_dict()
            t2_gate_d = t2_gate.to_dict()
            t2_diag_d = t2_diag.to_dict()

            entry["tier2_seakeeping"] = t2_seak_d
            entry["tier2_cfd"] = t2_cfd_d
            entry["tier2_passed"] = bool(
                t2_gate.passed
                and seak.valid
                and not seak.quick
                and cfd.valid
                and not cfd.dry_run
            )
            if diagnostic_tier2:
                entry["tier2_gate_report"] = t2_gate_d
                entry["tier2_diagnosis"] = t2_diag_d
            else:
                entry["gate_report"] = t2_gate_d
                entry["diagnosis"] = t2_diag_d
                entry["utility_score"] = _composite_utility(t2_gate.objectives)

            tier2_verified[d_hash] = {
                "candidate_id": entry["candidate_id"],
                "design_hash": d_hash,
                "hull_path": str(hull_path),
                "seakeeping": t2_seak_d,
                "cfd": t2_cfd_d,
                "gate_report": t2_gate_d,
                "diagnosis": t2_diag_d,
                "tier2_passed": entry["tier2_passed"],
            }
            # Record Tier 2 verification event in ledger for durability across resume
            t2_ledger = dict(entry)
            t2_ledger["timestamp_utc"] = _utc_now()
            t2_ledger["provenance"] = {**provenance, "stage": "tier2_verification"}
            append_jsonl_atomic(ledger_path, t2_ledger)

        tier2_done = True

    final_tier2_done = (
        (tier2_done or prior_tier2_completed)
        if (config.resume and config.tier2_top == 0 and not ran_new_iterations)
        else tier2_done
    )
    frontier, best_entry = _write_pareto_and_best(
        work, ledger_entries, tier2_verified, provenance, baseline
    )
    _save_checkpoint(final_completed, "completed", final_tier2_done)

    eval_entries = [e for e in ledger_entries if e.get("status") == "evaluated"]
    feasible_entries = [e for e in eval_entries if (e.get("gate_report") or {}).get("passed")]
    unique_eval_hashes = {e.get("design_hash") for e in eval_entries if e.get("design_hash")}
    unique_feas_hashes = {e.get("design_hash") for e in feasible_entries if e.get("design_hash")}

    return CampaignResult(
        work_dir=str(work),
        status="completed",
        seed=effective_seed,
        completed_iterations=final_completed,
        target_iterations=final_completed,
        total_evaluations=len(unique_eval_hashes),
        feasible_count=len(unique_feas_hashes),
        pareto_count=len(frontier),
        best_candidate_id=best_entry.get("candidate_id") if best_entry else None,
        best_design_hash=best_entry.get("design_hash") if best_entry else baseline.design_hash,
        best_hull_path=str(work / "best_candidate.hull.json"),
        pareto_front=frontier,
        tier2_verified=tier2_verified,
        provenance=provenance,
        artifacts={
            "state_json": str(work / "state.json"),
            "pareto_json": str(work / "pareto.json"),
            "ledger_jsonl": str(work / "ledger.jsonl"),
            "best_candidate_hull_json": str(work / "best_candidate.hull.json"),
        },
    )


def run_smoke(
    work_dir: Path,
    *,
    seed: int = 42,
    iterations: int = 2,
    population: int = 2,
    workers: int = 1,
    skip_resistance: bool = False,
    repo_root: Path | None = None,
    evaluator_fn: Callable[[Hull, Path, BaselineReference], CandidateMetrics] | None = None,
) -> CampaignResult:
    """Run a fast deterministic smoke campaign (baseline + small perturbations)."""
    cfg = CampaignConfig(
        iterations=iterations,
        population=population,
        seed=seed,
        work_dir=Path(work_dir),
        tier2_top=0,
        workers=workers,
        resume=False,
        skip_resistance=skip_resistance,
        pre_screen=False,
    )
    return run_campaign(cfg, repo_root=repo_root, evaluator_fn=evaluator_fn)


def load_campaign_status(
    work_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load and validate durable campaign artifacts (state, pareto, ledger)."""
    work = Path(work_dir)
    root = DEFAULT_REPO_ROOT if repo_root is None else Path(repo_root).resolve()
    baseline = load_baseline_reference(root)
    space = DesignSpace.default()
    state, ledger_entries, pareto_raw = _validate_and_load_resume_state(work, space, baseline)

    eval_entries = [e for e in ledger_entries if e.get("status") == "evaluated"]
    unique_eval_hashes = {e.get("design_hash") for e in eval_entries if e.get("design_hash")}
    unique_feas_hashes = {
        e.get("design_hash")
        for e in eval_entries
        if (e.get("gate_report") or {}).get("passed") and e.get("design_hash")
    }
    frontier = pareto_raw.get("frontier") or []
    best_id = pareto_raw.get("best_candidate_id")
    best_hash = pareto_raw.get("best_design_hash")
    best_entry = next(
        (e for e in reversed(eval_entries) if e.get("candidate_id") == best_id),
        frontier[0] if frontier else None,
    )

    return {
        "command": "status",
        "work_dir": str(work),
        "status": state.status,
        "iteration": state.completed_iterations,
        "completed_iterations": state.completed_iterations,
        "target_iterations": state.target_iterations,
        "seed": state.seed,
        "total_evaluations": len(unique_eval_hashes),
        "ledger_records": len(ledger_entries),
        "feasible_count": len(unique_feas_hashes),
        "pareto_count": len(frontier),
        "pareto_front": frontier,
        "best_candidate": best_entry,
        "best_candidate_id": best_id,
        "best_design_hash": best_hash,
        "best_hull_path": str(work / "best_candidate.hull.json"),
        "tier2_verified": pareto_raw.get("tier2_verified") or {},
        "state": state.to_dict(),
        "adaptation_hints": state.adaptation_hints,
        "artifacts": {
            "state_json": str(work / "state.json"),
            "pareto_json": str(work / "pareto.json"),
            "ledger_jsonl": str(work / "ledger.jsonl"),
            "best_candidate_hull_json": str(work / "best_candidate.hull.json"),
        },
    }


__all__ = [
    "CampaignConfig",
    "CampaignResult",
    "CampaignState",
    "LedgerEntry",
    "ResumeStateError",
    "append_jsonl_atomic",
    "load_campaign_status",
    "run_campaign",
    "run_smoke",
    "write_json_atomic",
]
