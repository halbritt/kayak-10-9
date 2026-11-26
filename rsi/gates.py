"""Hard non-regression gates and multi-objective scoring for Kayak 10.9 RSI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from rsi.baseline import BaselineReference
from rsi.evaluator import CATCH_OFFSETS_M, CandidateMetrics
from rsi.fidelity import CfdMetrics, SeakeepingMetrics

# Bump whenever a gate is added or removed, or a tolerance or comparison baseline changes. A campaign
# checkpoint stores pass/fail verdicts next to the measurements; verdicts from another version are
# stale, so a resumed search would breed from hulls the current gates reject.
#   1: original Tier 0/1/2 gates
#   2: catch-width gate; speed and GM compared against the unrounded baseline
#   3: CFD_DRAG_TOL_N = 0.25 N matching 500-step interFoam/snappyHexMesh noise floor
#   4: race-pace 90/120W objective, SR [10.875, 10.99] band, 265 lb speed gate,
#      28x25 cm foot box, +-10 mm 225 lb LCB cap, fixed-frame catch
#   5: seakeeping floor is the Epic 18X (owner, 2026-11-26), not v0.2.3 + 1.5 %; seakeeping rows
#      come from a body with irregular-frequency removal
GATES_VERSION = "5"

SR_4IN_MIN = 10.875
SR_4IN_MAX = 10.99
SR_4IN_TARGET = 10.90
USCA_4IN_MIN_PCT = 8.50
L4_MIN_M = 5.442
FOOT_BOX_MIN_WIDTH_M = 0.28
FOOT_BOX_MIN_HEIGHT_M = 0.25
LCB_225_TOL_MM = 10.0
RACE_PACE_POWERS_W = (90, 120)
Z_MIN_ABS_TOL_M = 1e-5
WATERLINE_HOLLOW_MAX_DEG = 0.02
# Upper-hull width over the paddle stroke (at the LCG and 0.8-1.2 m forward, from 100 mm above the
# 225 lb waterline) may not grow by more than the project's catch-width comparison limit
# (docs/surface-fairing/README.md, cad/check_fairing_controls.py). One-sided: narrower is allowed.
CATCH_WIDTH_TOL_MM = 2.0
GM_TOL_CM = 0.01
GZ_TOL_MM = 1e-3
KN_TOL_KN = 0.005
# Seakeeping floor (owner, 2026-11-26): "I don't want it to regress to be worse than the Epic 18x".
# In each gated condition seat acceleration and bow dips may not exceed the Epic 18X's value from
# the same model. Where v0.2.3 is itself worse than the 18X the limit is v0.2.3's value, so the
# baseline always passes and nothing may fall below both boats. Quick mode solves every second
# period, so it gets an allowance against rows computed on the full grid.
CAPYTAINE_QUICK_REL_TOL = 0.05
CAPYTAINE_GM_T_TOL_CM = 0.05
CFD_DRAG_TOL_N = 0.25


def _finite_float(val: Any) -> float | None:
    if val is None or isinstance(val, (bool, np.bool_)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _json_safe_scalar(val: float | str | None) -> float | str | None:
    if isinstance(val, float) and not math.isfinite(val):
        return str(val)
    return val


@dataclass(frozen=True)
class GateViolation:
    gate: str
    message: str
    actual: float | str | None = None
    limit: float | str | None = None
    margin: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["actual"] = _json_safe_scalar(d["actual"])
        d["limit"] = _json_safe_scalar(d["limit"])
        if isinstance(d["margin"], float) and not math.isfinite(d["margin"]):
            d["margin"] = -1.0
        return d


@dataclass(frozen=True)
class GateReport:
    passed: bool
    violations: tuple[GateViolation, ...]
    margins: dict[str, float]
    objectives: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "margins": {
                k: (float(v) if math.isfinite(float(v)) else -1.0)
                for k, v in self.margins.items()
            },
            "objectives": {
                k: (float(v) if math.isfinite(float(v)) else -1.0)
                for k, v in self.objectives.items()
            },
        }


def evaluate_gates(
    metrics: CandidateMetrics,
    baseline: BaselineReference,
    *,
    seakeeping: SeakeepingMetrics | None = None,
    cfd: CfdMetrics | None = None,
) -> GateReport:
    """Evaluate hard non-regression constraints and compute multi-objective scores."""
    violations: list[GateViolation] = []
    margins: dict[str, float] = {}

    if not metrics.valid:
        violations.append(
            GateViolation(
                gate="valid",
                message=metrics.error or "Candidate evaluation failed",
                actual="invalid",
                limit="valid",
            )
        )
        return GateReport(
            passed=False,
            violations=tuple(violations),
            margins=margins,
            objectives={
                "sr_closeness_to_10_9": -10.0,
                "speed_gain_kn": -10.0,
                "stability_min_margin": -10.0,
                "seakeeping_gain": 0.0,
                "fairness_margin": -10.0,
            },
        )

    def _require(
        gate: str,
        cond: bool,
        margin: float,
        msg: str,
        actual: float | str | None,
        limit: float | str | None,
    ) -> None:
        safe_margin = float(margin)
        if not math.isfinite(safe_margin):
            safe_margin = -1.0
            cond = False
        safe_actual = _json_safe_scalar(actual)
        safe_limit = _json_safe_scalar(limit)
        margins[gate] = safe_margin
        if not cond:
            violations.append(
                GateViolation(
                    gate=gate,
                    message=msg,
                    actual=safe_actual,
                    limit=safe_limit,
                    margin=safe_margin,
                )
            )

    # 1. Shell integrity & keel datum
    shell_ok = bool(metrics.is_watertight and metrics.is_volume)
    _require(
        "watertight_and_volume",
        shell_ok,
        1.0 if shell_ok else -1.0,
        "Outer shell must be watertight and a valid volume",
        f"watertight={metrics.is_watertight},volume={metrics.is_volume}",
        "True,True",
    )

    z_min = _finite_float(metrics.z_min_m)
    if z_min is None:
        _require(
            "z_min_m",
            False,
            -1.0,
            f"Missing or non-finite z_min_m: {metrics.z_min_m}",
            "missing" if metrics.z_min_m is None else str(metrics.z_min_m),
            Z_MIN_ABS_TOL_M,
        )
    else:
        _require(
            "z_min_m",
            abs(z_min) <= Z_MIN_ABS_TOL_M,
            Z_MIN_ABS_TOL_M - abs(z_min),
            f"|z_min_m|={abs(z_min):.6g} exceeds {Z_MIN_ABS_TOL_M}",
            z_min,
            Z_MIN_ABS_TOL_M,
        )

    # 2. Class dimensions: SR_4in, USCA_4in_pct, l4_m
    raw_sr_pairs = (
        ("sr_4in_exact", metrics.sr_4in_exact),
        ("sr_4in", metrics.sr_4in),
    )
    sr_nonfinite = [
        (name, val) for name, val in raw_sr_pairs if val is not None and _finite_float(val) is None
    ]
    sr_candidates = [
        f for _, val in raw_sr_pairs if (f := _finite_float(val)) is not None
    ]
    if sr_nonfinite or not sr_candidates:
        bad_desc = (
            ", ".join(f"{name}={val}" for name, val in sr_nonfinite)
            if sr_nonfinite
            else "missing"
        )
        _require(
            "sr_4in_min",
            False,
            -1.0,
            f"Missing or non-finite SR_4in: {bad_desc}",
            bad_desc,
            SR_4IN_MIN,
        )
        _require(
            "sr_4in_max",
            False,
            -1.0,
            f"Missing or non-finite SR_4in: {bad_desc}",
            bad_desc,
            SR_4IN_MAX,
        )
        sr_exact_obj: float | None = None
    else:
        sr_min_check = min(sr_candidates)
        sr_check = max(sr_candidates)
        sr_exact_f = _finite_float(metrics.sr_4in_exact)
        sr_exact_obj = sr_exact_f if sr_exact_f is not None else sr_check
        _require(
            "sr_4in_min",
            sr_min_check >= SR_4IN_MIN,
            sr_min_check - SR_4IN_MIN,
            f"SR_4in {sr_min_check:.4f} below minimum {SR_4IN_MIN:.3f}",
            sr_min_check,
            SR_4IN_MIN,
        )
        _require(
            "sr_4in_max",
            sr_check <= SR_4IN_MAX,
            SR_4IN_MAX - sr_check,
            f"SR_4in {sr_check:.4f} exceeds USCA FSK limit {SR_4IN_MAX}",
            sr_check,
            SR_4IN_MAX,
        )

    raw_usca_pairs = (
        ("usca_4in_pct_exact", metrics.usca_4in_pct_exact),
        ("usca_4in_pct", metrics.usca_4in_pct),
    )
    usca_nonfinite = [
        (name, val) for name, val in raw_usca_pairs if val is not None and _finite_float(val) is None
    ]
    usca_candidates = [
        f for _, val in raw_usca_pairs if (f := _finite_float(val)) is not None
    ]
    if usca_nonfinite or not usca_candidates:
        bad_desc = (
            ", ".join(f"{name}={val}" for name, val in usca_nonfinite)
            if usca_nonfinite
            else "missing"
        )
        _require(
            "usca_4in_pct_min",
            False,
            -1.0,
            f"Missing or non-finite USCA_4in_pct: {bad_desc}",
            bad_desc,
            USCA_4IN_MIN_PCT,
        )
    else:
        usca_check = min(usca_candidates)
        _require(
            "usca_4in_pct_min",
            usca_check >= USCA_4IN_MIN_PCT,
            usca_check - USCA_4IN_MIN_PCT,
            f"USCA_4in_pct {usca_check:.4f}% below minimum {USCA_4IN_MIN_PCT:.2f}%",
            usca_check,
            USCA_4IN_MIN_PCT,
        )

    l4_val = _finite_float(metrics.l4_m)
    if l4_val is None:
        _require(
            "l4_m_min",
            False,
            -1.0,
            f"Missing or non-finite 4-inch waterline length: {metrics.l4_m}",
            "missing" if metrics.l4_m is None else str(metrics.l4_m),
            L4_MIN_M,
        )
    else:
        _require(
            "l4_m_min",
            l4_val >= L4_MIN_M,
            l4_val - L4_MIN_M,
            f"4-inch waterline length {l4_val:.4f} m below {L4_MIN_M} m",
            l4_val,
            L4_MIN_M,
        )

    # 2b. Cockpit foot-box clearance (28 cm wide x 25 cm high)
    for gate_name, raw_val, min_lim, desc in (
        ("foot_clear_height_m", metrics.foot_clear_height_m, FOOT_BOX_MIN_HEIGHT_M, "Foot box clear height"),
        ("foot_clear_width_m", metrics.foot_clear_width_m, FOOT_BOX_MIN_WIDTH_M, "Foot box clear width"),
    ):
        fv = _finite_float(raw_val)
        if fv is None:
            _require(
                gate_name,
                False,
                -1.0,
                f"Missing or non-finite {desc}: {raw_val}",
                "missing" if raw_val is None else str(raw_val),
                min_lim,
            )
        else:
            _require(
                gate_name,
                fv >= min_lim,
                fv - min_lim,
                f"{desc} {fv:.4f} m below minimum {min_lim:.2f} m",
                fv,
                min_lim,
            )

    # 3. Monotonic catch-to-cockpit flare transition
    trans_raw = metrics.transition_dB_per_20cm
    if not trans_raw:
        _require(
            "transition_monotonic",
            False,
            -1.0,
            "Missing catch-to-cockpit beam transition values",
            "missing",
            0.0,
        )
    else:
        trans_vals = [_finite_float(v) for v in trans_raw]
        if any(v is None for v in trans_vals):
            _require(
                "transition_monotonic",
                False,
                -1.0,
                f"Non-finite catch-to-cockpit beam transition values: {trans_raw}",
                str(trans_raw),
                0.0,
            )
        else:
            finite_trans = [v for v in trans_vals if v is not None]
            min_trans = min(finite_trans)
            _require(
                "transition_monotonic",
                min_trans > 0.0,
                min_trans,
                f"Non-monotonic catch-to-cockpit beam transition (min dB/20cm={min_trans})",
                min_trans,
                0.0,
            )

    # 3b. Wing-paddle catch width non-regression
    base_catch = tuple(baseline.catch_widths_mm or ())
    raw_catch = metrics.catch_widths_mm
    catch_limit_desc = f"baseline + {CATCH_WIDTH_TOL_MM} mm"
    if not base_catch:
        _require(
            "catch_width_mm",
            False,
            -1.0,
            "Baseline catch widths missing; cannot check catch-width non-regression",
            "missing",
            catch_limit_desc,
        )
    elif raw_catch is None or len(raw_catch) != len(base_catch):
        _require(
            "catch_width_mm",
            False,
            -1.0,
            f"Missing or incomplete catch widths: {raw_catch}",
            "missing" if raw_catch is None else str(tuple(raw_catch)),
            catch_limit_desc,
        )
    else:
        cand_catch = [_finite_float(v) for v in raw_catch]
        if any(v is None for v in cand_catch):
            _require(
                "catch_width_mm",
                False,
                -1.0,
                f"Non-finite catch widths: {tuple(raw_catch)}",
                str(tuple(raw_catch)),
                catch_limit_desc,
            )
        else:
            widening = [c - float(b) for c, b in zip(cand_catch, base_catch) if c is not None]
            worst = max(range(len(widening)), key=widening.__getitem__)
            station = (
                f"+{CATCH_OFFSETS_M[worst]:.1f} m"
                if len(base_catch) == len(CATCH_OFFSETS_M)
                else f"station {worst}"
            )
            _require(
                "catch_width_mm",
                widening[worst] <= CATCH_WIDTH_TOL_MM,
                CATCH_WIDTH_TOL_MM - widening[worst],
                (
                    f"Catch width at {station} widened by {widening[worst]:.2f} mm "
                    f"(limit {CATCH_WIDTH_TOL_MM} mm over baseline {float(base_catch[worst]):.1f} mm)"
                ),
                cand_catch[worst],
                float(base_catch[worst]) + CATCH_WIDTH_TOL_MM,
            )

    # 4. Kayakgen waterline hollow & surface fairing invariants
    def _check_finite_lt(
        gate: str,
        raw_val: Any,
        limit: float,
        msg_prefix: str,
        fmt: str,
    ) -> None:
        val = _finite_float(raw_val)
        if val is None:
            _require(
                gate,
                False,
                -1.0,
                f"Missing or non-finite {msg_prefix}: {raw_val}",
                "missing" if raw_val is None else str(raw_val),
                limit,
            )
        else:
            _require(
                gate,
                val < limit,
                limit - val,
                f"{msg_prefix} {val:{fmt}} >= {limit}",
                val,
                limit,
            )

    _check_finite_lt(
        "kayakgen_waterline_hollow",
        metrics.waterline_hollow_max_deg,
        WATERLINE_HOLLOW_MAX_DEG,
        "waterline_hollow_max_deg",
        ".4f",
    )

    _require(
        "has_surface_fairing",
        bool(metrics.has_surface_fairing),
        1.0 if metrics.has_surface_fairing else -1.0,
        "surface_fairing must be enabled",
        str(metrics.has_surface_fairing),
        "True",
    )

    fair = baseline.fairness
    _check_finite_lt(
        "fwd_wl_turn_deg",
        metrics.fwd_wl_max_turn_deg,
        fair.fwd_wl_turn_max_deg,
        "Forward waterline turn",
        ".5f",
    )
    _check_finite_lt(
        "entry_zone_turn_deg",
        metrics.entry_zone_max_turn_deg,
        fair.entry_zone_turn_max_deg,
        "Entry zone turn",
        ".5f",
    )
    _check_finite_lt(
        "buttock_0075_peak_deg",
        metrics.buttock_0075_peak_deg,
        fair.buttock_0075_peak_max_deg,
        "Buttocks 0.075 peak turn change",
        ".4f",
    )
    _check_finite_lt(
        "forebody_keel_groove_mm",
        metrics.forebody_max_keel_groove_mm,
        fair.forebody_keel_groove_max_mm,
        "Forebody keel groove",
        ".6f",
    )

    k_min = _finite_float(metrics.forebody_min_k)
    if k_min is None:
        _require(
            "forebody_k_min",
            False,
            -1.0,
            f"Missing or non-finite Forebody min curvature: {metrics.forebody_min_k}",
            "missing" if metrics.forebody_min_k is None else str(metrics.forebody_min_k),
            fair.forebody_k_min_per_m,
        )
    else:
        _require(
            "forebody_k_min",
            k_min > fair.forebody_k_min_per_m,
            k_min - fair.forebody_k_min_per_m,
            f"Forebody min curvature {k_min:.4f} <= {fair.forebody_k_min_per_m}",
            k_min,
            fair.forebody_k_min_per_m,
        )

    _check_finite_lt(
        "forebody_dk_ds_max",
        metrics.forebody_max_abs_dk_ds,
        fair.forebody_dk_ds_max_per_m2,
        "Forebody |dk/ds|",
        ".2f",
    )

    fairness_norm_margins = [
        margins["kayakgen_waterline_hollow"] / WATERLINE_HOLLOW_MAX_DEG,
        margins["fwd_wl_turn_deg"] / 0.05,
        margins["entry_zone_turn_deg"] / fair.entry_zone_turn_max_deg,
        margins["buttock_0075_peak_deg"] / fair.buttock_0075_peak_max_deg,
        margins["forebody_keel_groove_mm"] / fair.forebody_keel_groove_max_mm,
        margins["forebody_k_min"] / 1.0,
        margins["forebody_dk_ds_max"] / fair.forebody_dk_ds_max_per_m2,
    ]
    fairness_margin_raw = float(min(fairness_norm_margins))
    fairness_margin = fairness_margin_raw if math.isfinite(fairness_margin_raw) else -1.0

    # 5. Stability non-regression across 225 / 265 / 300 lb
    stability_margins_raw: list[float] = []
    loads = metrics.loads or {}
    for lb, base_load in baseline.loads.items():
        cand_load = loads.get(lb)
        if cand_load is None:
            _require(
                f"load_{lb}_present",
                False,
                -1.0,
                f"Missing load case {lb} lb",
                "missing",
                str(lb),
            )
            stability_margins_raw.append(-1.0)
            continue

        gm_val = _finite_float(cand_load.gm_cm)
        if gm_val is None:
            _require(
                f"gm_{lb}_cm",
                False,
                -1.0,
                f"Missing or non-finite GM at {lb} lb: {cand_load.gm_cm}",
                "missing" if cand_load.gm_cm is None else str(cand_load.gm_cm),
                base_load.gm_cm,
            )
            stability_margins_raw.append(-1.0)
        else:
            base_gm_exact = _finite_float(base_load.gm_cm_exact)
            base_gm = base_gm_exact if base_gm_exact is not None else float(base_load.gm_cm)
            gm_diff = gm_val - base_gm
            _require(
                f"gm_{lb}_cm",
                gm_diff >= -GM_TOL_CM,
                gm_diff + GM_TOL_CM,
                f"GM at {lb} lb regressed: {gm_val:.3f} cm < {base_gm:.3f} - {GM_TOL_CM}",
                gm_val,
                base_gm,
            )
            stability_margins_raw.append(gm_diff)

        gz_seq = cand_load.gz_mm
        for idx, deg in enumerate((10, 20, 30)):
            raw_gz = gz_seq[idx] if gz_seq is not None and idx < len(gz_seq) else None
            gz_val = _finite_float(raw_gz)
            if gz_val is None:
                _require(
                    f"gz_{lb}_{deg}_mm",
                    False,
                    -1.0,
                    f"Missing or non-finite GZ({deg} deg) at {lb} lb: {raw_gz}",
                    "missing" if raw_gz is None else str(raw_gz),
                    base_load.gz_mm[idx],
                )
                stability_margins_raw.append(-1.0)
            else:
                gz_diff = gz_val - float(base_load.gz_mm[idx])
                _require(
                    f"gz_{lb}_{deg}_mm",
                    gz_diff >= -GZ_TOL_MM,
                    gz_diff + GZ_TOL_MM,
                    (
                        f"GZ({deg} deg) at {lb} lb regressed: "
                        f"{gz_val:.4f} mm < {base_load.gz_mm[idx]:.4f} - {GZ_TOL_MM}"
                    ),
                    gz_val,
                    base_load.gz_mm[idx],
                )
                stability_margins_raw.append(gz_diff)

    stability_min_raw = float(min(stability_margins_raw)) if stability_margins_raw else -1.0
    stability_min_margin = stability_min_raw if math.isfinite(stability_min_raw) else -1.0

    # 5b. 225 lb LCB shift cap (+-10 mm from baseline 225 lb LCG so trim/seat stays fixed)
    if 225 in baseline.loads and baseline.loads[225].lcg_m is not None and 225 in loads:
        base_lcg = float(baseline.loads[225].lcg_m)
        cand_lcg = _finite_float(loads[225].lcg_m)
        if cand_lcg is None:
            _require(
                "lcb_225_mm",
                False,
                -1.0,
                f"Missing or non-finite 225 lb LCB: {loads[225].lcg_m}",
                "missing" if loads[225].lcg_m is None else str(loads[225].lcg_m),
                LCB_225_TOL_MM,
            )
        else:
            shift_mm = abs(cand_lcg - base_lcg) * 1000.0
            _require(
                "lcb_225_mm",
                shift_mm <= LCB_225_TOL_MM,
                LCB_225_TOL_MM - shift_mm,
                f"225 lb LCB shifted {shift_mm:.2f} mm (limit +-{LCB_225_TOL_MM:.1f} mm)",
                shift_mm,
                LCB_225_TOL_MM,
            )

    # 6. Speed non-regression across 225 lb and 265 lb; race-pace objective = mean(90W, 120W) at 225 lb
    def _check_speed_load(
        prefix: str,
        cand_kn: dict[int, float] | None,
        cand_kn_exact: dict[int, float] | None,
        base_kn_tbl: dict[int, float],
        base_kn_exact_tbl: dict[int, float] | None,
        *,
        record_powers: tuple[int, ...] = (),
    ) -> list[float]:
        gains: list[float] = []
        if cand_kn is None:
            _require(
                f"{prefix}_missing",
                False,
                -1.0,
                f"Speed metrics {prefix} missing while skip_resistance=False",
                "missing",
                str(sorted(base_kn_tbl.keys())),
            )
            return gains
        for p_w, base_kn in base_kn_tbl.items():
            raw_kn = cand_kn.get(p_w)
            kn_val = _finite_float(raw_kn)
            if kn_val is None:
                _require(
                    f"{prefix}_{p_w}w_missing",
                    False,
                    -1.0,
                    f"Missing or non-finite {prefix} speed at {p_w} W",
                    "missing" if raw_kn is None else str(raw_kn),
                    base_kn,
                )
                continue
            vals = [kn_val]
            exact_val: float | None = None
            if cand_kn_exact is not None and p_w in cand_kn_exact:
                raw_exact = cand_kn_exact[p_w]
                exact_val = _finite_float(raw_exact)
                if exact_val is None:
                    _require(
                        f"{prefix}_{p_w}w_missing",
                        False,
                        -1.0,
                        f"Missing or non-finite exact {prefix} speed at {p_w} W",
                        "missing" if raw_exact is None else str(raw_exact),
                        base_kn,
                    )
                    continue
                vals.append(exact_val)
            base_exact = _finite_float((base_kn_exact_tbl or {}).get(p_w))
            if exact_val is not None and base_exact is not None:
                cand_kn_min = exact_val
                base_ref = base_exact
                cand_kn_gain = exact_val - base_exact
            else:
                cand_kn_min = min(vals)
                base_ref = float(base_kn)
                cand_kn_gain = (exact_val if exact_val is not None else kn_val) - base_kn
            if int(p_w) in record_powers:
                gains.append(float(cand_kn_gain))
            diff = cand_kn_min - base_ref
            _require(
                f"{prefix}_{p_w}w",
                diff >= -KN_TOL_KN,
                diff + KN_TOL_KN,
                f"{prefix} speed at {p_w} W regressed: {cand_kn_min:.4f} kn < {base_ref:.4f} - {KN_TOL_KN}",
                cand_kn_min,
                base_ref,
            )
        return gains

    speed_gains: list[float] = []
    if not metrics.skip_resistance:
        speed_gains = _check_speed_load(
            "kn_225",
            metrics.kn_225,
            metrics.kn_225_exact,
            baseline.kn_225,
            baseline.kn_225_exact,
            record_powers=RACE_PACE_POWERS_W,
        )
        if baseline.kn_265:
            _check_speed_load(
                "kn_265",
                metrics.kn_265,
                metrics.kn_265_exact,
                baseline.kn_265,
                baseline.kn_265_exact,
            )
    speed_gain_kn = float(sum(speed_gains) / len(speed_gains)) if speed_gains else 0.0

    # 7. Optional Tier 2 Capytaine seakeeping non-regression
    seakeeping_gains: list[float] = []
    if seakeeping is not None:
        if not seakeeping.valid:
            _require(
                "seakeeping_valid",
                False,
                -1.0,
                seakeeping.error or "Tier 2 Capytaine seakeeping failed",
                "invalid",
                "valid",
            )
        else:
            gm_t_val = _finite_float(seakeeping.gm_t_cm)
            if gm_t_val is None:
                _require(
                    "seakeeping_gm_t_cm",
                    False,
                    -1.0,
                    f"Missing or non-finite BEM GM_t: {seakeeping.gm_t_cm}",
                    "missing" if seakeeping.gm_t_cm is None else str(seakeeping.gm_t_cm),
                    baseline.capytaine.gm_t_cm,
                )
            else:
                gm_t_diff = gm_t_val - baseline.capytaine.gm_t_cm
                _require(
                    "seakeeping_gm_t_cm",
                    gm_t_diff >= -CAPYTAINE_GM_T_TOL_CM,
                    gm_t_diff + CAPYTAINE_GM_T_TOL_CM,
                    f"BEM GM_t regressed: {gm_t_val:.2f} cm < {baseline.capytaine.gm_t_cm:.2f}",
                    gm_t_val,
                    baseline.capytaine.gm_t_cm,
                )

            rel_tol = CAPYTAINE_QUICK_REL_TOL if seakeeping.quick else 0.0
            floor_map = {
                (r.speed_kn, r.heading, r.sea): r for r in baseline.capytaine.epic18x_conditions_225
            }
            row_map: dict[tuple[float, str, str], Mapping[str, Any]] = {}
            for idx, r in enumerate(seakeeping.rows or ()):
                if not isinstance(r, Mapping):
                    _require(
                        "seak_malformed_row",
                        False,
                        -1.0,
                        f"Malformed Capytaine seakeeping row at index {idx}: expected mapping, got {type(r).__name__}",
                        str(r),
                        "mapping with speed_kn, heading, sea",
                    )
                    continue
                speed_val = _finite_float(r.get("speed_kn"))
                heading_val = r.get("heading")
                sea_val = r.get("sea")
                if (
                    speed_val is None
                    or not isinstance(heading_val, str)
                    or not heading_val
                    or not isinstance(sea_val, str)
                    or not sea_val
                ):
                    _require(
                        "seak_malformed_row",
                        False,
                        -1.0,
                        f"Malformed Capytaine seakeeping row at index {idx}: {r}",
                        str(r),
                        "finite speed_kn and non-empty heading, sea",
                    )
                    continue
                row_map[(speed_val, heading_val, sea_val)] = r
            if seakeeping.quick:
                required_conditions = tuple(
                    r
                    for r in baseline.capytaine.target_conditions_225
                    if r.speed_kn == 4.5 and r.heading == "head"
                )
            else:
                required_conditions = baseline.capytaine.target_conditions_225

            for base_row in required_conditions:
                key = (base_row.speed_kn, base_row.heading, base_row.sea)
                tag = f"{base_row.speed_kn:.1f}kn_{base_row.heading}_{base_row.sea.split()[0].lower()}"
                cand_row = row_map.get(key)
                if cand_row is None:
                    _require(
                        f"seak_missing_{tag}",
                        False,
                        -1.0,
                        f"Missing required Capytaine condition {key}",
                        "missing",
                        str(key),
                    )
                    continue
                floor_row = floor_map.get(key)
                if floor_row is None:
                    _require(
                        f"seak_floor_missing_{tag}",
                        False,
                        -1.0,
                        f"No Epic 18X reference row for condition {key}",
                        "missing",
                        str(key),
                    )
                    continue
                dips_lim = max(floor_row.bow_dips_per_min, base_row.bow_dips_per_min) * (1.0 + rel_tol)
                raw_dips = cand_row.get("bow_dips_per_min")
                dips_val = _finite_float(raw_dips)
                _require(
                    f"seak_bow_dips_{tag}",
                    dips_val is not None and dips_val <= dips_lim,
                    (dips_lim - dips_val) if dips_val is not None else -1.0,
                    (
                        f"Bow dips at {key} worse than the Epic 18X floor: {dips_val:.2f} > {dips_lim:.2f}"
                        if dips_val is not None
                        else f"Missing or non-finite bow dips at {key}: {raw_dips}"
                    ),
                    dips_val if dips_val is not None else ("missing" if raw_dips is None else str(raw_dips)),
                    dips_lim,
                )
                acc_lim = max(floor_row.seat_acc_rms_ms2, base_row.seat_acc_rms_ms2) * (1.0 + rel_tol)
                raw_acc = cand_row.get("seat_acc_rms_ms2")
                acc_val = _finite_float(raw_acc)
                _require(
                    f"seak_seat_acc_{tag}",
                    acc_val is not None and acc_val <= acc_lim,
                    (acc_lim - acc_val) if acc_val is not None else -1.0,
                    (
                        f"Seat accel at {key} worse than the Epic 18X floor: {acc_val:.3f} > {acc_lim:.3f}"
                        if acc_val is not None
                        else f"Missing or non-finite seat accel at {key}: {raw_acc}"
                    ),
                    acc_val if acc_val is not None else ("missing" if raw_acc is None else str(raw_acc)),
                    acc_lim,
                )
                if dips_val is not None and base_row.bow_dips_per_min > 0:
                    seakeeping_gains.append(
                        (base_row.bow_dips_per_min - dips_val) / base_row.bow_dips_per_min
                    )
                if acc_val is not None and base_row.seat_acc_rms_ms2 > 0:
                    seakeeping_gains.append(
                        (base_row.seat_acc_rms_ms2 - acc_val) / base_row.seat_acc_rms_ms2
                    )
    seakeeping_gain = (
        float(sum(seakeeping_gains) / len(seakeeping_gains)) if seakeeping_gains else 0.0
    )

    # 8. Optional Tier 2 OpenFOAM CFD non-regression
    if cfd is not None:
        if not cfd.valid:
            _require(
                "cfd_valid",
                False,
                -1.0,
                cfd.error or "Tier 2 OpenFOAM CFD failed",
                "invalid",
                "valid",
            )
        if cfd.dry_run:
            _require(
                "cfd_dry_run",
                False,
                -1.0,
                "Tier 2 OpenFOAM CFD dry_run=True cannot satisfy promotion gate",
                "dry_run",
                "executed",
            )
        elif cfd.valid:
            if cfd.speeds is not None and not isinstance(cfd.speeds, Mapping):
                _require(
                    "cfd_speeds_malformed",
                    False,
                    -1.0,
                    f"Malformed OpenFOAM CFD speeds container: expected mapping, got {type(cfd.speeds).__name__}",
                    str(cfd.speeds),
                    "mapping",
                )
            speeds_map = cfd.speeds if isinstance(cfd.speeds, Mapping) else {}
            base_cfd_by_key = {
                "5.5kn": baseline.openfoam.speed_5_5kn,
                "6.2kn": baseline.openfoam.speed_6_2kn,
            }
            for sp_key, base_sp in base_cfd_by_key.items():
                sp_data = speeds_map.get(sp_key)
                if not isinstance(sp_data, Mapping) or "drag_total_N" not in sp_data:
                    _require(
                        f"cfd_missing_{sp_key}",
                        False,
                        -1.0,
                        f"Missing OpenFOAM CFD metrics for {sp_key}",
                        "missing" if sp_data is None else str(sp_data),
                        sp_key,
                    )
                    continue
                raw_drag = sp_data["drag_total_N"]
                drag = _finite_float(raw_drag)
                lim = base_sp.drag_total_N + CFD_DRAG_TOL_N
                _require(
                    f"cfd_drag_{sp_key}",
                    drag is not None and drag <= lim,
                    (lim - drag) if drag is not None else -1.0,
                    (
                        f"OpenFOAM drag at {sp_key} regressed: {drag:.3f} N > {lim:.3f} N"
                        if drag is not None
                        else f"Missing or non-finite OpenFOAM drag at {sp_key}: {raw_drag}"
                    ),
                    drag if drag is not None else ("missing" if raw_drag is None else str(raw_drag)),
                    lim,
                )

    sr_closeness = (
        float(-abs(sr_exact_obj - SR_4IN_TARGET))
        if sr_exact_obj is not None and math.isfinite(sr_exact_obj)
        else -10.0
    )
    objectives = {
        "sr_closeness_to_10_9": sr_closeness,
        "speed_gain_kn": speed_gain_kn if math.isfinite(speed_gain_kn) else -10.0,
        "stability_min_margin": stability_min_margin if math.isfinite(stability_min_margin) else -1.0,
        "seakeeping_gain": seakeeping_gain if math.isfinite(seakeeping_gain) else 0.0,
        "fairness_margin": fairness_margin if math.isfinite(fairness_margin) else -1.0,
    }
    return GateReport(
        passed=(len(violations) == 0),
        violations=tuple(violations),
        margins=margins,
        objectives=objectives,
    )
