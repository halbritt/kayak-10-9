"""Failure-mode classification, parameter-group attribution, and adaptation hints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any

from pydantic import ValidationError

from rsi.evaluator import CandidateMetrics
from rsi.fidelity import CfdMetrics, SeakeepingMetrics
from rsi.gates import (
    CATCH_WIDTH_TOL_MM,
    LCB_225_TOL_MM,
    WATERLINE_HOLLOW_MAX_DEG,
    Z_MIN_ABS_TOL_M,
    GateReport,
)
from rsi.genome import DesignSpace

PARAMETER_GROUPS: tuple[str, ...] = (
    "waterline",
    "draft",
    "rocker",
    "section",
    "topsides",
    "fairing",
)

_GROUP_SCALE_MIN = 0.25
_GROUP_SCALE_MAX = 1.50

_XI_RE = re.compile(r"xi\s*=\s*([-+]?\d+(?:\.\d+)?)")
_LOAD_GATE_RE = re.compile(r"^(?:load_(\d+)_present|gm_(\d+)_cm|gz_(\d+)_(\d+)_mm)$")


def _finite_float_or_none(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clamp(val: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, float(val)))


def _default_group_scales() -> dict[str, float]:
    return {group: 1.0 for group in PARAMETER_GROUPS}


@dataclass(frozen=True)
class DiagnosticItem:
    """Single structured diagnostic finding for a candidate or validation failure."""

    code: str
    source: str
    gate: str | None
    severity: str
    groups: tuple[str, ...]
    parameters: tuple[str, ...]
    message: str
    margin: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source": self.source,
            "gate": self.gate,
            "severity": self.severity,
            "groups": list(self.groups),
            "parameters": list(self.parameters),
            "message": self.message,
            "margin": _finite_float_or_none(self.margin),
        }


@dataclass(frozen=True)
class AdaptationHints:
    """Search adaptation signals derived from candidate diagnostics."""

    group_scales: dict[str, float] = field(default_factory=_default_group_scales)
    preferred_directions: dict[str, float] = field(default_factory=dict)
    avoid_bounds: dict[str, dict[str, Any]] = field(default_factory=dict)
    tight_margins: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        clean_scales = {
            g: round(
                _clamp(
                    _finite_float_or_none(self.group_scales.get(g, 1.0)) or 1.0,
                    _GROUP_SCALE_MIN,
                    _GROUP_SCALE_MAX,
                ),
                6,
            )
            for g in PARAMETER_GROUPS
        }
        clean_dirs: dict[str, float] = {}
        for k, v in self.preferred_directions.items():
            fv = _finite_float_or_none(v)
            if fv is not None:
                clean_dirs[k] = round(_clamp(fv, -1.0, 1.0), 6)

        clean_avoid: dict[str, dict[str, Any]] = {}
        for param, rule in self.avoid_bounds.items():
            clean_rule: dict[str, Any] = {}
            for rk, rv in rule.items():
                if isinstance(rv, float):
                    clean_rule[rk] = _finite_float_or_none(rv)
                else:
                    clean_rule[rk] = rv
            clean_avoid[param] = clean_rule

        return {
            "group_scales": clean_scales,
            "preferred_directions": clean_dirs,
            "avoid_bounds": clean_avoid,
            "tight_margins": list(self.tight_margins),
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Complete diagnostic summary and adaptation hints for one candidate."""

    feasible: bool
    failure_modes: tuple[str, ...]
    primary_groups: tuple[str, ...]
    items: tuple[DiagnosticItem, ...]
    hints: AdaptationHints
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "failure_modes": list(self.failure_modes),
            "primary_groups": list(self.primary_groups),
            "items": [item.to_dict() for item in self.items],
            "hints": self.hints.to_dict(),
            "summary": self.summary,
        }


class _HintAccumulator:
    """Mutable builder that merges diagnostic attributions into AdaptationHints."""

    def __init__(self, space: DesignSpace, vector: Sequence[float] | None) -> None:
        self.space = space
        self.idx_by_name = space.index_by_name()
        self.spec_by_name = {p.name: p for p in space.params}
        self.vec = list(vector) if vector is not None and len(vector) == len(space.params) else None
        self.group_weights: dict[str, float] = {g: 0.0 for g in PARAMETER_GROUPS}
        self.group_scales: dict[str, float] = _default_group_scales()
        self.dir_sums: dict[str, float] = {}
        self.dir_weights: dict[str, float] = {}
        self.avoid_bounds: dict[str, dict[str, Any]] = {}
        self.tight_margins: list[str] = []

    def val(self, name: str) -> float | None:
        if self.vec is None:
            return None
        idx = self.idx_by_name.get(name)
        if idx is None:
            return None
        return _finite_float_or_none(self.vec[idx])

    def add_group_blame(self, groups: Sequence[str], weight: float = 1.0) -> None:
        for pos, g in enumerate(groups):
            if g in self.group_weights:
                self.group_weights[g] += weight / (1.0 + 0.5 * pos)

    def shrink_group(self, group: str, factor: float) -> None:
        if group in self.group_scales:
            self.group_scales[group] = min(
                self.group_scales[group],
                _clamp(factor, _GROUP_SCALE_MIN, _GROUP_SCALE_MAX),
            )

    def add_direction(self, param: str, direction: float, weight: float = 1.0) -> None:
        if param not in self.spec_by_name:
            return
        self.dir_sums[param] = self.dir_sums.get(param, 0.0) + float(direction) * weight
        self.dir_weights[param] = self.dir_weights.get(param, 0.0) + weight

    def bias_toward_baseline(self, param: str, strength: float = 0.8, weight: float = 1.0) -> None:
        spec = self.spec_by_name.get(param)
        if spec is None:
            return
        v = self.val(param)
        if v is None:
            return
        diff = spec.baseline - v
        if abs(diff) > 1e-12:
            self.add_direction(param, math.copysign(strength, diff), weight=weight)

    def set_avoid(self, param: str, side: str, limit: float, reason: str) -> None:
        if param not in self.spec_by_name:
            return
        lim = _finite_float_or_none(limit)
        if lim is None:
            return
        self.avoid_bounds[param] = {"side": side, "limit": lim, "reason": reason}

    def ranked_groups(self) -> tuple[str, ...]:
        ranked = sorted(
            ((g, w) for g, w in self.group_weights.items() if w > 0.0),
            key=lambda kv: (-kv[1], PARAMETER_GROUPS.index(kv[0])),
        )
        return tuple(g for g, _ in ranked)

    def build_hints(self) -> AdaptationHints:
        dirs: dict[str, float] = {}
        for param, w_sum in self.dir_weights.items():
            if w_sum > 0.0:
                val = _clamp(self.dir_sums[param] / w_sum, -1.0, 1.0)
                if abs(val) > 1e-6:
                    dirs[param] = val
        return AdaptationHints(
            group_scales={
                g: _clamp(v, _GROUP_SCALE_MIN, _GROUP_SCALE_MAX)
                for g, v in self.group_scales.items()
            },
            preferred_directions=dirs,
            avoid_bounds=self.avoid_bounds,
            tight_margins=tuple(dict.fromkeys(self.tight_margins)),
        )


def diagnose_validation_error(
    exc: Exception,
    vector: Sequence[float] | None = None,
    space: DesignSpace | None = None,
) -> DiagnosticReport:
    """Classify a Hull/DistributionV2 pydantic ValidationError into groups & repair hints."""
    sp = DesignSpace.default() if space is None else space
    acc = _HintAccumulator(sp, vector)
    items: list[DiagnosticItem] = []

    raw_errors: list[dict[str, Any]]
    if isinstance(exc, ValidationError):
        raw_errors = [dict(e) for e in exc.errors()]
    else:
        raw_errors = [{"loc": (), "msg": str(exc), "type": type(exc).__name__}]

    for err in raw_errors:
        msg = str(err.get("msg", str(exc)))
        loc_parts = tuple(str(x) for x in err.get("loc", ()))
        loc_str = ".".join(loc_parts)

        if "sheer_plan must stay within" in msg:
            groups = ("topsides", "waterline")
            params = (
                "sheer_max_hb_m",
                "sheer_max_xi",
                "sheer_fwd_exp",
                "sheer_aft_exp",
                "wl_a0",
                "wl_a1",
                "wl_a2",
                "wl_a3",
            )
            items.append(
                DiagnosticItem(
                    code="validation_sheer_plan_coupled",
                    source="validation",
                    gate=None,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=msg,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("topsides", 0.55)
            acc.shrink_group("waterline", 0.75)

            xi_match = _XI_RE.search(msg)
            xi_val = float(xi_match.group(1)) if xi_match else 0.0
            acc.bias_toward_baseline("sheer_max_hb_m", strength=0.9, weight=1.5)
            acc.bias_toward_baseline("wl_a0", strength=0.6, weight=1.0)
            if xi_val < 0.0:
                acc.bias_toward_baseline("sheer_fwd_exp", strength=0.9, weight=1.5)
                acc.bias_toward_baseline("wl_a1", strength=0.6, weight=1.0)
            else:
                acc.bias_toward_baseline("sheer_aft_exp", strength=0.9, weight=1.5)
                acc.bias_toward_baseline("sheer_max_xi", strength=0.6, weight=1.0)

            hb_val = acc.val("sheer_max_hb_m")
            hb_spec = acc.spec_by_name["sheer_max_hb_m"]
            if hb_val is not None:
                if hb_val > hb_spec.baseline:
                    acc.set_avoid(
                        "sheer_max_hb_m",
                        "upper",
                        max(hb_spec.baseline + 0.005, hb_val - 0.002),
                        "sheer_plan exceeded 1.5x waterline half-breadth",
                    )
                else:
                    acc.set_avoid(
                        "sheer_max_hb_m",
                        "lower",
                        min(hb_spec.baseline - 0.005, hb_val + 0.002),
                        "sheer_plan fell below 0.6x waterline half-breadth",
                    )
            else:
                acc.set_avoid(
                    "sheer_max_hb_m",
                    "clamp_toward_baseline",
                    hb_spec.baseline,
                    "sheer_plan coupled [0.6, 1.5] waterline envelope violated",
                )

        elif "forward_xi < aft_xi" in msg:
            if "section_side_exponent" in loc_str:
                groups = ("section",)
                params = ("sec_side_fwd_xi", "sec_side_aft_xi")
                fwd_name, aft_name = "sec_side_fwd_xi", "sec_side_aft_xi"
            elif "bilge_round_m" in loc_str:
                groups = ("section",)
                params = ("bilge_fwd_xi", "bilge_aft_xi")
                fwd_name, aft_name = "bilge_fwd_xi", "bilge_aft_xi"
            elif "deck_crown_exponent" in loc_str:
                groups = ("topsides",)
                params = ("crown_fwd_xi", "crown_aft_xi")
                fwd_name, aft_name = "crown_fwd_xi", "crown_aft_xi"
            else:
                groups = ("section", "topsides")
                params = (
                    "sec_side_fwd_xi",
                    "sec_side_aft_xi",
                    "bilge_fwd_xi",
                    "bilge_aft_xi",
                    "crown_fwd_xi",
                    "crown_aft_xi",
                )
                fwd_name, aft_name = "sec_side_fwd_xi", "sec_side_aft_xi"

            items.append(
                DiagnosticItem(
                    code="validation_ramp_order",
                    source="validation",
                    gate=None,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=msg,
                )
            )
            acc.add_group_blame(groups, 2.0)
            for g in groups:
                acc.shrink_group(g, 0.55)
            acc.add_direction(fwd_name, -1.0, weight=1.5)
            acc.add_direction(aft_name, +1.0, weight=1.5)
            aft_val = acc.val(aft_name)
            fwd_val = acc.val(fwd_name)
            limit_mid = (
                0.5 * (fwd_val + aft_val)
                if (fwd_val is not None and aft_val is not None)
                else 0.0
            )
            acc.set_avoid(fwd_name, "upper", limit_mid - 0.02, "RampDistribution requires forward_xi < aft_xi")
            acc.set_avoid(aft_name, "lower", limit_mid + 0.02, "RampDistribution requires forward_xi < aft_xi")

        elif any(
            key in loc_str or key in msg
            for key in ("sheer_height_frac", "deck_crown_exponent", "shoulder_round_m", "sheer_plan", "sheer_beam_ratio")
        ):
            groups = ("topsides",)
            params = (
                "sheer_h_fwd_frac",
                "sheer_h_aft_frac",
                "crown_fwd",
                "crown_aft",
                "shoulder_round_m",
                "sheer_max_hb_m",
            )
            items.append(
                DiagnosticItem(
                    code="validation_topsides_range",
                    source="validation",
                    gate=None,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=msg,
                )
            )
            acc.add_group_blame(groups, 1.5)
            acc.shrink_group("topsides", 0.55)
            for p in params:
                acc.bias_toward_baseline(p, strength=0.8, weight=1.0)

        elif "bilge_round_m" in loc_str or "bilge_round_m" in msg or "section_side_exponent" in loc_str:
            groups = ("section",)
            params = ("bilge_fwd_m", "bilge_aft_m", "sec_side_fwd", "sec_side_aft")
            items.append(
                DiagnosticItem(
                    code="validation_section_range",
                    source="validation",
                    gate=None,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=msg,
                )
            )
            acc.add_group_blame(groups, 1.5)
            acc.shrink_group("section", 0.55)
            for p in params:
                acc.bias_toward_baseline(p, strength=0.8, weight=1.0)

        elif "surface_fairing" in loc_str or "surface_fairing" in msg or "target_beam_m" in msg:
            groups = ("fairing",)
            params = ("target_beam_m",)
            items.append(
                DiagnosticItem(
                    code="validation_fairing_spec",
                    source="validation",
                    gate=None,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=msg,
                )
            )
            acc.add_group_blame(groups, 1.5)
            acc.shrink_group("fairing", 0.55)
            acc.bias_toward_baseline("target_beam_m", strength=0.8, weight=1.0)

        else:
            groups = ("waterline", "topsides", "section")
            params = ("wl_a0", "sheer_max_hb_m", "draft_a1", "draft_a2")
            items.append(
                DiagnosticItem(
                    code="validation_schema_error",
                    source="validation",
                    gate=None,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=msg,
                )
            )
            acc.add_group_blame(groups, 1.0)
            for g in groups:
                acc.shrink_group(g, 0.70)
            for p in params:
                acc.bias_toward_baseline(p, strength=0.6, weight=0.8)

    failure_modes = tuple(dict.fromkeys(it.code for it in items))
    primary_groups = acc.ranked_groups()
    summary = (
        f"ValidationError ({len(items)} issue(s)): "
        + "; ".join(f"{it.code}: {it.message}" for it in items[:3])
    )
    return DiagnosticReport(
        feasible=False,
        failure_modes=failure_modes,
        primary_groups=primary_groups,
        items=tuple(items),
        hints=acc.build_hints(),
        summary=summary,
    )


def _diagnose_invalid_metrics(
    metrics: CandidateMetrics,
    acc: _HintAccumulator,
) -> list[DiagnosticItem]:
    err = metrics.error or "Candidate evaluation failed"
    items: list[DiagnosticItem] = []

    if any(tok in err for tok in ("solve_target_draft", "operating_pose_dry", "operating_pose_fully_submerged", "evaluate_hydrostatics")):
        groups = ("draft", "waterline", "section")
        params = ("draft_a1", "draft_a2", "wl_a0", "sec_side_fwd", "sec_side_aft")
        items.append(
            DiagnosticItem(
                code="hydrostatic_pose_failure",
                source="metrics",
                gate="valid",
                severity="error",
                groups=groups,
                parameters=params,
                message=err,
            )
        )
        acc.add_group_blame(groups, 2.0)
        acc.shrink_group("draft", 0.45)
        acc.shrink_group("waterline", 0.55)
        acc.bias_toward_baseline("draft_a1", strength=1.0, weight=1.5)
        acc.bias_toward_baseline("draft_a2", strength=0.8, weight=1.2)
        acc.bias_toward_baseline("wl_a0", strength=0.8, weight=1.2)
    elif any(tok in err for tok in ("Closed shell invalid", "4-inch waterline", "SculptedKayakSurface")):
        groups = ("topsides", "section", "fairing")
        params = ("sheer_max_hb_m", "shoulder_round_m", "bilge_fwd_m", "target_beam_m")
        items.append(
            DiagnosticItem(
                code="shell_geometry_failure",
                source="metrics",
                gate="valid",
                severity="error",
                groups=groups,
                parameters=params,
                message=err,
            )
        )
        acc.add_group_blame(groups, 2.0)
        for g in groups:
            acc.shrink_group(g, 0.50)
        for p in params:
            acc.bias_toward_baseline(p, strength=0.8, weight=1.0)
    elif "Mesh fairness evaluation failed" in err:
        groups = ("rocker", "section", "waterline")
        params = ("rocker_bow_end_m", "rocker_stern_end_m", "bilge_fwd_m", "wl_a1", "wl_a2")
        items.append(
            DiagnosticItem(
                code="fairness_eval_failure",
                source="metrics",
                gate="valid",
                severity="error",
                groups=groups,
                parameters=params,
                message=err,
            )
        )
        acc.add_group_blame(groups, 1.8)
        for g in groups:
            acc.shrink_group(g, 0.55)
        for p in params:
            acc.bias_toward_baseline(p, strength=0.7, weight=1.0)
    elif "Free-trim balance failed" in err or "Hydrostatics / stability / resistance failed" in err:
        groups = ("waterline", "section", "draft", "topsides")
        params = ("wl_a0", "sec_side_fwd", "sec_side_aft", "draft_a2", "sheer_max_hb_m")
        items.append(
            DiagnosticItem(
                code="equilibrium_or_gz_failure",
                source="metrics",
                gate="valid",
                severity="error",
                groups=groups,
                parameters=params,
                message=err,
            )
        )
        acc.add_group_blame(groups, 1.8)
        for g in groups:
            acc.shrink_group(g, 0.55)
        for p in params:
            acc.bias_toward_baseline(p, strength=0.8, weight=1.0)
    else:
        groups = ("waterline", "draft", "section")
        params = ("wl_a0", "draft_a1", "draft_a2", "target_beam_m")
        items.append(
            DiagnosticItem(
                code="evaluation_invalid",
                source="metrics",
                gate="valid",
                severity="error",
                groups=groups,
                parameters=params,
                message=err,
            )
        )
        acc.add_group_blame(groups, 1.5)
        for g in groups:
            acc.shrink_group(g, 0.60)
        for p in params:
            acc.bias_toward_baseline(p, strength=0.7, weight=1.0)

    return items


def diagnose_candidate(
    metrics: CandidateMetrics | None,
    gate_report: GateReport | None,
    *,
    seakeeping: SeakeepingMetrics | None = None,
    cfd: CfdMetrics | None = None,
    vector: Sequence[float] | None = None,
    space: DesignSpace | None = None,
    error: Exception | None = None,
) -> DiagnosticReport:
    """Attribute gate failures, execution errors, and tight margins to parameter groups."""
    if error is not None:
        return diagnose_validation_error(error, vector=vector, space=space)

    sp = DesignSpace.default() if space is None else space
    acc = _HintAccumulator(sp, vector)
    items: list[DiagnosticItem] = []

    if metrics is None:
        items.append(
            DiagnosticItem(
                code="missing_metrics",
                source="metrics",
                gate="valid",
                severity="error",
                groups=("waterline", "draft"),
                parameters=("wl_a0", "draft_a1"),
                message="CandidateMetrics was not produced",
            )
        )
        acc.add_group_blame(("waterline", "draft"), 1.0)
        return DiagnosticReport(
            feasible=False,
            failure_modes=("missing_metrics",),
            primary_groups=acc.ranked_groups(),
            items=tuple(items),
            hints=acc.build_hints(),
            summary="Candidate evaluation missing metrics.",
        )

    if not metrics.valid:
        items.extend(_diagnose_invalid_metrics(metrics, acc))

    violations = gate_report.violations if gate_report is not None else ()
    violation_gates = {v.gate for v in violations}
    has_stability_failure = any(
        g.startswith("gm_") or g.startswith("gz_") or g == "seakeeping_gm_t_cm" or g.startswith("load_")
        for g in violation_gates
    )
    has_usca_or_sr_failure = ("sr_4in_max" in violation_gates or "usca_4in_pct_min" in violation_gates)
    # Slacker forward sections (lower sec_side_fwd) buy heeled stability with flare, and that flare
    # is the dominant lever on catch width. Stop steering toward it once less than half the catch
    # tolerance is left; otherwise every stability or speed hint walks the search into the catch gate.
    catch_margin = _finite_float_or_none(
        (gate_report.margins if gate_report is not None else {}).get("catch_width_mm")
    )
    fwd_flare_room = catch_margin is None or catch_margin >= 0.5 * CATCH_WIDTH_TOL_MM

    for v in violations:
        g_id = v.gate
        margin = _finite_float_or_none(v.margin)

        if g_id == "valid":
            if metrics.valid:
                items.append(
                    DiagnosticItem(
                        code="evaluation_invalid",
                        source="gate",
                        gate=g_id,
                        severity="error",
                        groups=("waterline", "draft"),
                        parameters=("wl_a0", "draft_a1"),
                        message=v.message,
                        margin=margin,
                    )
                )
                acc.add_group_blame(("waterline", "draft"), 1.5)
            continue

        if g_id == "watertight_and_volume":
            groups = ("topsides", "section", "fairing")
            params = ("sheer_max_hb_m", "shoulder_round_m", "bilge_fwd_m", "target_beam_m")
            items.append(
                DiagnosticItem(
                    code="shell_not_watertight",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            for grp in groups:
                acc.shrink_group(grp, 0.50)

        elif g_id == "z_min_m":
            groups = ("draft", "rocker")
            params = ("draft_a1", "draft_a2", "rocker_bow_end_m", "rocker_stern_end_m")
            items.append(
                DiagnosticItem(
                    code="keel_datum_shift",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.5)
            acc.shrink_group("draft", 0.35)
            acc.set_avoid(
                "draft_a1",
                "away_from_nonzero",
                0.002,
                "draft_a1!=0 drops keel below Z_KEEL_SHIFT=0.1030",
            )
            d1_val = acc.val("draft_a1")
            if d1_val is not None and abs(d1_val) > 1e-12:
                acc.add_direction("draft_a1", -math.copysign(1.0, d1_val), weight=2.5)
            else:
                acc.add_direction("draft_a1", 0.0, weight=2.0)
            d2_val = acc.val("draft_a2")
            d2_base = acc.spec_by_name["draft_a2"].baseline
            if d2_val is not None and d2_val > d2_base:
                acc.add_direction("draft_a2", -0.8, weight=1.2)
            acc.add_direction("rocker_bow_end_m", +0.4, weight=0.6)
            acc.add_direction("rocker_stern_end_m", +0.4, weight=0.6)

        elif g_id == "sr_4in_min":
            groups = ("fairing", "waterline", "section")
            params = ("target_beam_m", "wl_a0", "wl_a1", "wl_a2", "sec_side_fwd", "sec_side_aft")
            items.append(
                DiagnosticItem(
                    code="sr_4in_below_min",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("fairing", 0.70)
            acc.add_direction("target_beam_m", -1.0, weight=2.0)
            acc.add_direction("wl_a0", -0.8, weight=1.2)
            tb_val = acc.val("target_beam_m")
            if tb_val is not None:
                acc.set_avoid(
                    "target_beam_m",
                    "upper",
                    tb_val - 0.0005,
                    f"sr_4in_min violated at target_beam_m={tb_val:.5f}",
                )

        elif g_id in ("foot_clear_height_m", "foot_clear_width_m"):
            groups = ("topsides", "waterline", "section")
            params = (
                "deck_fb_fwd_scale",
                "crown_fwd",
                "sheer_h_fwd_frac",
                "wl_a0",
                "wl_a1",
                "bilge_fwd_m",
            )
            items.append(
                DiagnosticItem(
                    code="foot_box_clearance",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            if g_id == "foot_clear_height_m":
                acc.shrink_group("topsides", 0.65)
                acc.add_direction("deck_fb_fwd_scale", +0.8, weight=1.5)
                acc.add_direction("crown_fwd", -0.6, weight=1.2)
                acc.add_direction("sheer_h_fwd_frac", +0.6, weight=1.0)
            else:
                acc.shrink_group("waterline", 0.65)
                acc.add_direction("wl_a0", +0.7, weight=1.2)
                acc.add_direction("wl_a1", +0.7, weight=1.2)
                acc.add_direction("bilge_fwd_m", -0.5, weight=1.0)

        elif g_id == "lcb_225_mm":
            groups = ("waterline", "rocker", "draft")
            params = ("wl_a1", "rocker_bow_end_m", "rocker_stern_end_m", "draft_a1", "draft_a2")
            items.append(
                DiagnosticItem(
                    code="lcb_shift_exceeded",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("waterline", 0.70)
            acc.shrink_group("rocker", 0.70)
            for p in params:
                acc.bias_toward_baseline(p, strength=0.8, weight=1.2)

        elif g_id in ("sr_4in_max", "usca_4in_pct_min"):
            code = "sr_4in_exceeded" if g_id == "sr_4in_max" else "usca_beam_below_min"
            groups = ("fairing", "waterline", "section")
            params = ("target_beam_m", "wl_a0", "wl_a1", "wl_a2", "sec_side_fwd", "sec_side_aft")
            items.append(
                DiagnosticItem(
                    code=code,
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("fairing", 0.70)
            acc.add_direction("target_beam_m", +1.0, weight=2.0)
            acc.add_direction("wl_a0", +1.0, weight=1.5)
            tb_val = acc.val("target_beam_m")
            if tb_val is not None:
                acc.set_avoid(
                    "target_beam_m",
                    "lower",
                    tb_val + 0.0005,
                    f"{g_id} violated at target_beam_m={tb_val:.5f}",
                )

        elif g_id == "l4_m_min":
            groups = ("rocker", "draft")
            params = ("rocker_bow_end_m", "rocker_stern_end_m", "draft_a2", "draft_a1")
            items.append(
                DiagnosticItem(
                    code="waterline_length_short",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 1.8)
            acc.shrink_group("rocker", 0.65)
            acc.add_direction("rocker_bow_end_m", -0.8, weight=1.5)
            acc.add_direction("draft_a2", +0.6, weight=1.0)

        elif g_id in ("kayakgen_waterline_hollow", "fwd_wl_turn_deg", "entry_zone_turn_deg"):
            groups = ("waterline", "fairing", "section")
            params = ("wl_a1", "wl_a2", "wl_a3", "wl_a0", "target_beam_m", "bilge_fwd_m")
            items.append(
                DiagnosticItem(
                    code="waterline_entry_hollow",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("waterline", 0.55)
            acc.add_direction("wl_a1", -0.7, weight=1.2)
            acc.bias_toward_baseline("wl_a2", strength=0.7, weight=1.0)
            acc.bias_toward_baseline("wl_a3", strength=0.7, weight=1.0)
            acc.add_direction("bilge_fwd_m", +0.5, weight=0.8)

        elif g_id == "buttock_0075_peak_deg":
            groups = ("rocker", "draft", "section", "waterline")
            params = (
                "rocker_bow_end_m",
                "rocker_stern_end_m",
                "draft_a2",
                "draft_a1",
                "bilge_fwd_m",
                "bilge_fwd_xi",
                "sec_side_aft",
                "wl_a1",
                "sheer_aft_exp",
            )
            items.append(
                DiagnosticItem(
                    code="buttock_0075_kink",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("rocker", 0.60)
            acc.shrink_group("draft", 0.65)
            acc.add_direction("sec_side_aft", -0.7, weight=1.2)
            acc.add_direction("wl_a1", -0.6, weight=1.0)
            acc.add_direction("sheer_aft_exp", +0.6, weight=1.0)
            acc.bias_toward_baseline("rocker_bow_end_m", strength=0.6, weight=0.9)
            acc.bias_toward_baseline("draft_a2", strength=0.6, weight=0.9)

        elif g_id in ("forebody_keel_groove_mm", "forebody_k_min", "forebody_dk_ds_max"):
            groups = ("section", "rocker")
            params = ("bilge_fwd_m", "bilge_fwd_xi", "sec_side_fwd", "sec_side_fwd_xi", "rocker_bow_end_m")
            items.append(
                DiagnosticItem(
                    code="forebody_section_fairness",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("section", 0.60)
            if g_id in ("forebody_dk_ds_max", "forebody_keel_groove_mm"):
                acc.add_direction("bilge_fwd_m", +0.7, weight=1.2)
            acc.bias_toward_baseline("sec_side_fwd", strength=0.6, weight=0.8)

        elif g_id == "catch_width_mm":
            groups = ("section", "topsides")
            params = (
                "sec_side_fwd",
                "sec_side_fwd_xi",
                "sheer_max_xi",
                "deck_fb_fwd_scale",
                "draft_a1",
            )
            items.append(
                DiagnosticItem(
                    code="catch_width_widened",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)
            acc.shrink_group("section", 0.60)
            sf_val = acc.val("sec_side_fwd")
            sf_spec = acc.spec_by_name["sec_side_fwd"]
            if sf_val is not None and sf_val < sf_spec.baseline:
                # This much forward flare is known to exceed the catch limit; keep proposals above it.
                acc.add_direction("sec_side_fwd", +1.0, weight=2.5)
                acc.set_avoid(
                    "sec_side_fwd",
                    "lower",
                    sf_val,
                    f"catch width exceeded at sec_side_fwd={sf_val:.4f}",
                )
            else:
                acc.add_direction("sec_side_fwd", +0.5, weight=1.5)
            acc.add_direction("sec_side_fwd_xi", +0.4, weight=0.8)

        elif g_id == "transition_monotonic":
            groups = ("topsides", "waterline")
            params = ("sheer_max_hb_m", "sheer_max_xi", "sheer_fwd_exp", "shoulder_round_m", "wl_a0", "wl_a1")
            items.append(
                DiagnosticItem(
                    code="catch_transition_non_monotonic",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 1.8)
            acc.shrink_group("topsides", 0.65)
            acc.bias_toward_baseline("sheer_max_hb_m", strength=0.7, weight=1.0)
            acc.bias_toward_baseline("sheer_max_xi", strength=0.7, weight=1.0)

        elif g_id == "has_surface_fairing":
            groups = ("fairing",)
            params = ("target_beam_m",)
            items.append(
                DiagnosticItem(
                    code="missing_surface_fairing",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 2.0)

        elif (m := _LOAD_GATE_RE.match(g_id)) or g_id == "seakeeping_gm_t_cm":
            if g_id == "seakeeping_gm_t_cm":
                lb_tag = "bem"
                deg_val = 0
                source_tag = "seakeeping"
            else:
                assert m is not None
                lb_tag = m.group(1) or m.group(2) or m.group(3)
                deg_val = int(m.group(4)) if m.group(4) is not None else 0
                source_tag = "gate"

            if deg_val in (20, 30):
                code = f"stability_loss_{lb_tag}"
                groups = ("topsides", "waterline", "section")
                params = (
                    "sheer_max_hb_m",
                    "shoulder_round_m",
                    "sheer_h_fwd_frac",
                    "wl_a0",
                    "target_beam_m",
                    "sec_side_fwd",
                    "sec_side_aft",
                )
                acc.add_direction("sheer_max_hb_m", +0.8, weight=1.3)
                acc.add_direction("sheer_h_fwd_frac", +0.7, weight=1.1)
            else:
                code = f"stability_loss_{lb_tag}"
                groups = ("waterline", "section", "fairing")
                params = (
                    "wl_a0",
                    "target_beam_m",
                    "sec_side_fwd",
                    "sec_side_aft",
                    "bilge_fwd_m",
                    "bilge_aft_m",
                    "rocker_stern_end_m",
                    "draft_a2",
                )

            items.append(
                DiagnosticItem(
                    code=code,
                    source=source_tag,
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 1.8)
            acc.add_direction("wl_a0", +0.9, weight=1.5)
            acc.add_direction("target_beam_m", +0.9, weight=1.5)
            acc.add_direction("rocker_stern_end_m", +0.6, weight=0.9)
            acc.add_direction("wl_a2", +0.5, weight=0.8)
            if fwd_flare_room:
                acc.add_direction("sec_side_fwd", -0.5, weight=0.8)

        elif g_id.startswith("kn_225_") or g_id.startswith("kn_265_") or g_id.startswith("cfd_"):
            is_cfd = g_id.startswith("cfd_")
            groups = ("waterline", "fairing", "draft", "rocker", "section")
            params = (
                "wl_a0",
                "wl_a1",
                "wl_a2",
                "target_beam_m",
                "draft_a2",
                "rocker_bow_end_m",
                "rocker_stern_end_m",
                "sec_side_fwd",
            )
            code = "speed_or_drag_regression"
            if g_id in ("cfd_valid", "cfd_dry_run", "cfd_speeds_malformed") or g_id.startswith("cfd_missing_"):
                code = g_id if g_id in ("cfd_valid", "cfd_dry_run") else "cfd_metrics_incomplete"
            items.append(
                DiagnosticItem(
                    code=code,
                    source="cfd" if is_cfd else "gate",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 1.5)
            if not has_stability_failure and not has_usca_or_sr_failure:
                acc.add_direction("target_beam_m", -0.5, weight=1.0)
                acc.add_direction("wl_a0", -0.4, weight=0.8)
            if fwd_flare_room:
                acc.add_direction("sec_side_fwd", -0.6, weight=1.0)
            acc.add_direction("wl_a2", -0.4, weight=0.7)

        elif g_id.startswith("seak_") or g_id == "seakeeping_valid":
            groups = ("topsides", "rocker", "draft", "waterline")
            params = (
                "deck_fb_fwd_scale",
                "sheer_h_fwd_frac",
                "sheer_fwd_exp",
                "sheer_max_hb_m",
                "rocker_bow_end_m",
                "draft_a1",
            )
            items.append(
                DiagnosticItem(
                    code="seakeeping_regression",
                    source="seakeeping",
                    gate=g_id,
                    severity="error",
                    groups=groups,
                    parameters=params,
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(groups, 1.6)
            acc.add_direction("deck_fb_fwd_scale", +0.8, weight=1.2)
            acc.add_direction("sheer_h_fwd_frac", +0.6, weight=0.9)
            acc.add_direction("sheer_fwd_exp", +0.5, weight=0.8)
            acc.add_direction("sheer_max_hb_m", +0.5, weight=0.8)

        else:
            items.append(
                DiagnosticItem(
                    code=f"gate_violation_{g_id}",
                    source="gate",
                    gate=g_id,
                    severity="error",
                    groups=("waterline", "section"),
                    parameters=("wl_a0", "target_beam_m"),
                    message=v.message,
                    margin=margin,
                )
            )
            acc.add_group_blame(("waterline", "section"), 1.0)

    # Check tight margins when feasible (early warning & target-SR nudge)
    if gate_report is not None and gate_report.passed and metrics.valid:
        margin_scales: list[tuple[str, float, tuple[str, ...], tuple[str, ...]]] = [
            ("z_min_m", Z_MIN_ABS_TOL_M, ("draft", "rocker"), ("draft_a1", "draft_a2")),
            ("sr_4in_min", 0.05, ("fairing", "waterline"), ("target_beam_m", "wl_a0")),
            ("sr_4in_max", 0.10, ("fairing", "waterline"), ("target_beam_m", "wl_a0")),
            ("usca_4in_pct_min", 0.50, ("fairing", "waterline"), ("target_beam_m", "wl_a0")),
            ("l4_m_min", 0.01, ("rocker", "draft"), ("rocker_bow_end_m", "draft_a2")),
            ("foot_clear_height_m", 0.02, ("topsides",), ("deck_fb_fwd_scale", "crown_fwd")),
            ("foot_clear_width_m", 0.02, ("waterline", "section"), ("wl_a0", "wl_a1", "bilge_fwd_m")),
            ("lcb_225_mm", LCB_225_TOL_MM, ("waterline", "rocker"), ("wl_a1", "rocker_bow_end_m", "rocker_stern_end_m")),
            ("catch_width_mm", CATCH_WIDTH_TOL_MM, ("section",), ("sec_side_fwd", "sec_side_fwd_xi")),
            ("kayakgen_waterline_hollow", WATERLINE_HOLLOW_MAX_DEG, ("waterline",), ("wl_a1", "wl_a2")),
            ("fwd_wl_turn_deg", 0.05, ("waterline",), ("wl_a1", "wl_a2")),
            ("entry_zone_turn_deg", 0.01, ("waterline", "section"), ("wl_a1", "bilge_fwd_m")),
            ("buttock_0075_peak_deg", 0.50, ("rocker", "section"), ("sec_side_aft", "wl_a1", "sheer_aft_exp")),
            ("forebody_k_min", 0.50, ("section",), ("bilge_fwd_m", "sec_side_fwd")),
            ("forebody_dk_ds_max", 25.0, ("section",), ("bilge_fwd_m", "bilge_fwd_xi")),
        ]
        for lb in (225, 265, 300):
            margin_scales.append(
                (f"gm_{lb}_cm", 0.10, ("waterline", "section"), ("wl_a0", "target_beam_m", "rocker_stern_end_m"))
            )
            for deg in (10, 20, 30):
                margin_scales.append(
                    (f"gz_{lb}_{deg}_mm", 0.50, ("waterline", "section", "topsides"), ("wl_a0", "target_beam_m", "sheer_max_hb_m"))
                )
        for prefix in ("kn_225", "kn_265"):
            for w in (40, 60, 90, 120):
                margin_scales.append(
                    (f"{prefix}_{w}w", 0.05, ("waterline", "fairing", "section"), ("target_beam_m", "sec_side_fwd", "wl_a2"))
                )

        for gate_name, scale, groups, params in margin_scales:
            raw_m = gate_report.margins.get(gate_name)
            m_val = _finite_float_or_none(raw_m)
            if m_val is not None and (m_val / scale) < 0.10:
                acc.tight_margins.append(gate_name)
                items.append(
                    DiagnosticItem(
                        code=f"tight_margin_{gate_name}",
                        source="gate",
                        gate=f"tight_margin:{gate_name}",
                        severity="warning",
                        groups=groups,
                        parameters=params,
                        message=f"Tight passing margin on {gate_name}: {m_val:.5g} (scale {scale:.4g})",
                        margin=m_val,
                    )
                )

        # Gentle race-pace / fairness nudges when passing (SR band & stability are floors, not targets)
        stab_min = _finite_float_or_none(gate_report.objectives.get("stability_min_margin"))
        if stab_min is not None and stab_min <= 0.005:
            acc.add_direction("rocker_stern_end_m", +0.5, weight=0.6)
            if fwd_flare_room:
                acc.add_direction("sec_side_fwd", -0.4, weight=0.6)
            acc.add_direction("sec_side_aft", -0.4, weight=0.5)
            acc.add_direction("wl_a1", -0.4, weight=0.5)
            acc.add_direction("sheer_aft_exp", +0.4, weight=0.5)

    # Extra info if unused seakeeping/cfd passed without gate_report
    if gate_report is None:
        if seakeeping is not None and not seakeeping.valid:
            items.append(
                DiagnosticItem(
                    code="seakeeping_regression",
                    source="seakeeping",
                    gate="seakeeping_valid",
                    severity="error",
                    groups=("topsides", "rocker", "draft"),
                    parameters=("deck_fb_fwd_scale", "sheer_max_hb_m", "rocker_bow_end_m"),
                    message=seakeeping.error or "Seakeeping evaluation failed",
                )
            )
            acc.add_group_blame(("topsides", "rocker", "draft"), 1.5)
        if cfd is not None and not cfd.valid:
            items.append(
                DiagnosticItem(
                    code="cfd_valid",
                    source="cfd",
                    gate="cfd_valid",
                    severity="error",
                    groups=("waterline", "fairing", "draft"),
                    parameters=("wl_a0", "target_beam_m", "draft_a2"),
                    message=cfd.error or "CFD evaluation failed",
                )
            )
            acc.add_group_blame(("waterline", "fairing", "draft"), 1.5)

    feasible = bool(metrics.valid and (gate_report.passed if gate_report is not None else True))
    failure_modes = tuple(dict.fromkeys(it.code for it in items if it.severity == "error"))
    primary_groups = acc.ranked_groups()
    if feasible:
        tight_note = (
            f" ({len(acc.tight_margins)} tight margin(s))" if acc.tight_margins else ""
        )
        summary = f"Candidate feasible{tight_note}."
    else:
        summary = (
            f"Candidate failed ({len(failure_modes)} mode(s): "
            + ", ".join(failure_modes[:4])
            + ")"
        )

    return DiagnosticReport(
        feasible=feasible,
        failure_modes=failure_modes,
        primary_groups=primary_groups,
        items=tuple(items),
        hints=acc.build_hints(),
        summary=summary,
    )


def aggregate_diagnostics(
    reports: Sequence[DiagnosticReport],
    space: DesignSpace | None = None,
) -> AdaptationHints:
    """Aggregate population diagnostic reports into iteration-level adaptation hints."""
    del space
    if not reports:
        return AdaptationHints()

    scale_acc: dict[str, list[float]] = {g: [] for g in PARAMETER_GROUPS}
    dir_sum: dict[str, float] = {}
    dir_cnt: dict[str, int] = {}
    avoid_merged: dict[str, dict[str, Any]] = {}
    tight_all: list[str] = []

    for rep in reports:
        h = rep.hints
        weight = 1.0 if not rep.feasible else 0.4
        for g in PARAMETER_GROUPS:
            val = _finite_float_or_none(h.group_scales.get(g, 1.0))
            if val is not None:
                scale_acc[g].append(val)
        for param, d_val in h.preferred_directions.items():
            fd = _finite_float_or_none(d_val)
            if fd is not None and abs(fd) > 1e-9:
                dir_sum[param] = dir_sum.get(param, 0.0) + fd * weight
                dir_cnt[param] = dir_cnt.get(param, 0) + 1
        for param, rule in h.avoid_bounds.items():
            prev = avoid_merged.get(param)
            if prev is None:
                avoid_merged[param] = dict(rule)
                continue
            # Same-side numeric limits: keep the most conservative one, so the merged hint does
            # not depend on the order candidates were evaluated in.
            side = rule.get("side")
            new_lim = _finite_float_or_none(rule.get("limit"))
            old_lim = _finite_float_or_none(prev.get("limit"))
            if side != prev.get("side") or new_lim is None or old_lim is None:
                continue
            if (side == "lower" and new_lim > old_lim) or (side == "upper" and new_lim < old_lim):
                avoid_merged[param] = dict(rule)
        for tm in h.tight_margins:
            if tm not in tight_all:
                tight_all.append(tm)

    merged_scales = {
        g: _clamp(
            sum(scale_acc[g]) / len(scale_acc[g]) if scale_acc[g] else 1.0,
            _GROUP_SCALE_MIN,
            _GROUP_SCALE_MAX,
        )
        for g in PARAMETER_GROUPS
    }
    merged_dirs = {
        param: _clamp(dir_sum[param] / max(dir_cnt[param], 1), -1.0, 1.0)
        for param in dir_sum
        if abs(dir_sum[param]) > 1e-6
    }
    return AdaptationHints(
        group_scales=merged_scales,
        preferred_directions=merged_dirs,
        avoid_bounds=avoid_merged,
        tight_margins=tuple(tight_all),
    )


def diagnostic_report_from_dict(data: dict[str, Any]) -> DiagnosticReport:
    """Reconstruct DiagnosticReport cleanly from serialized JSON dict."""
    items = tuple(
        DiagnosticItem(
            code=str(it["code"]),
            source=str(it["source"]),
            gate=it.get("gate"),
            severity=str(it["severity"]),
            groups=tuple(str(g) for g in it.get("groups", ())),
            parameters=tuple(str(p) for p in it.get("parameters", ())),
            message=str(it["message"]),
            margin=_finite_float_or_none(it.get("margin")),
        )
        for it in data.get("items", ())
    )
    h_raw = data.get("hints", {})
    hints = AdaptationHints(
        group_scales={
            g: float(h_raw.get("group_scales", {}).get(g, 1.0))
            for g in PARAMETER_GROUPS
        },
        preferred_directions={
            str(k): float(v)
            for k, v in h_raw.get("preferred_directions", {}).items()
            if _finite_float_or_none(v) is not None
        },
        avoid_bounds={str(k): dict(v) for k, v in h_raw.get("avoid_bounds", {}).items()},
        tight_margins=tuple(str(x) for x in h_raw.get("tight_margins", ())),
    )
    return DiagnosticReport(
        feasible=bool(data.get("feasible", False)),
        failure_modes=tuple(str(x) for x in data.get("failure_modes", ())),
        primary_groups=tuple(str(x) for x in data.get("primary_groups", ())),
        items=items,
        hints=hints,
        summary=str(data.get("summary", "")),
    )


__all__ = [
    "AdaptationHints",
    "DiagnosticItem",
    "DiagnosticReport",
    "PARAMETER_GROUPS",
    "aggregate_diagnostics",
    "diagnose_candidate",
    "diagnose_validation_error",
    "diagnostic_report_from_dict",
]
