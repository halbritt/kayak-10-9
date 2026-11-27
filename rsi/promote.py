"""Candidate comparison vs v0.2.3 baseline and strictly gated promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any

from kayakgen.model.hull import Hull

from rsi.baseline import DEFAULT_REPO_ROOT, BaselineReference, load_baseline_reference
from rsi.diagnose import DiagnosticReport, diagnose_candidate
from rsi.evaluator import CandidateMetrics, evaluate_tier0_tier1
from rsi.fidelity import (
    CfdMetrics,
    SeakeepingMetrics,
    evaluate_tier2_seakeeping,
    resolve_openfoam_bashrc,
)
from rsi.gates import GateReport, evaluate_gates
from rsi.genome import DesignSpace, hull_to_vector
from rsi.paired_cfd import (
    PAIRED_MODE,
    current_reference_key,
    evaluate_tier2_cfd_mode,
    expected_result_mode,
)


def _finite_or_none(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _delta(cand: Any, base: Any, ndigits: int = 6) -> float | None:
    c = _finite_or_none(cand)
    b = _finite_or_none(base)
    if c is None or b is None:
        return None
    return round(c - b, ndigits)


def build_hydro_3load_payload(metrics: CandidateMetrics) -> dict[str, Any]:
    """Build normalized hydro_3load structure matching data/kayak-10-9-specs.json."""
    out: dict[str, Any] = {}
    loads = metrics.loads or {}
    for lb in (225, 265, 300):
        ld = loads.get(lb)
        if ld is None:
            out[f"load_{lb}"] = None
            continue
        gz_vals = [
            _finite_or_none(x) for x in (ld.gz_mm or ())
        ]
        out[f"load_{lb}"] = {
            "load_lb": lb,
            "T_cm": _finite_or_none(ld.t_cm),
            "draft_mm": _finite_or_none(ld.draft_mm),
            "Bwl_cm": _finite_or_none(ld.bwl_cm),
            "wet_m2": _finite_or_none(ld.wet_m2),
            "GM_cm": _finite_or_none(ld.gm_cm),
            "gm_cm": _finite_or_none(ld.gm_cm),
            "gz_mm": gz_vals,
            "gz_10_20_30_mm": [round(x, 1) if x is not None else None for x in gz_vals],
            "lcg_m": _finite_or_none(ld.lcg_m),
        }
    out["kn_225"] = (
        {str(k): _finite_or_none(v) for k, v in metrics.kn_225.items()}
        if metrics.kn_225 is not None
        else None
    )
    out["kn_265"] = (
        {str(k): _finite_or_none(v) for k, v in metrics.kn_265.items()}
        if metrics.kn_265 is not None
        else None
    )
    out["shell_area_m2"] = _finite_or_none(metrics.shell_area_m2)
    return out


def augment_metrics_dict(metrics: CandidateMetrics) -> dict[str, Any]:
    """Serialize CandidateMetrics with convenience and TEST_INFRA contract aliases."""
    d = metrics.to_dict()
    b4_val = _finite_or_none(metrics.b4_m)
    d["b_4in_cm"] = round(b4_val * 100.0, 2) if b4_val is not None else None
    d["watertight"] = bool(metrics.is_watertight and metrics.is_volume)
    d["max_signed_turn_forward_deg"] = _finite_or_none(metrics.fwd_wl_max_turn_deg)
    d["buttock_075_max_turn_change_deg"] = _finite_or_none(metrics.buttock_0075_peak_deg)
    d["keel_groove_mm"] = _finite_or_none(metrics.forebody_max_keel_groove_mm)
    d["bilge_k_min"] = _finite_or_none(metrics.forebody_min_k)
    d["bilge_dk_ds_max"] = _finite_or_none(metrics.forebody_max_abs_dk_ds)
    fch = _finite_or_none(metrics.foot_clear_height_m)
    fcw = _finite_or_none(metrics.foot_clear_width_m)
    d["foot_clear_height_cm"] = round(fch * 100.0, 2) if fch is not None else None
    d["foot_clear_width_at_25cm"] = round(fcw * 100.0, 2) if fcw is not None else None
    return d


def _compute_deltas(
    metrics: CandidateMetrics,
    baseline: BaselineReference,
    seakeeping: SeakeepingMetrics | None,
    cfd: CfdMetrics | None,
) -> dict[str, Any]:
    b4_cm_cand = (
        round(metrics.b4_m * 100.0, 4) if _finite_or_none(metrics.b4_m) is not None else None
    )
    b4_cm_base = round(baseline.b4_m * 100.0, 4)
    class_form = {
        "sr_4in": _delta(metrics.sr_4in, baseline.sr_4in, 4),
        "sr_4in_exact": _delta(metrics.sr_4in_exact, baseline.sr_4in_exact, 6),
        "usca_4in_pct": _delta(metrics.usca_4in_pct, baseline.usca_4in_pct, 4),
        "usca_4in_pct_exact": _delta(metrics.usca_4in_pct_exact, baseline.usca_4in_pct_exact, 6),
        "l4_m": _delta(metrics.l4_m, baseline.l4_m, 6),
        "b4_m": _delta(metrics.b4_m, baseline.b4_m, 6),
        "b_4in_cm": _delta(b4_cm_cand, b4_cm_base, 4),
        "l_oa_m": _delta(metrics.l_oa_m, baseline.l_oa_m, 6),
        "b_oa_cm": _delta(metrics.b_oa_cm, baseline.b_oa_cm, 4),
    }

    kn_deltas: dict[str, Any] = {}
    gains: list[float] = []
    if metrics.kn_225 is not None:
        base_kn_exact = baseline.kn_225_exact or {}
        for p_w, base_kn in baseline.kn_225.items():
            cand_exact = (
                metrics.kn_225_exact.get(p_w)
                if metrics.kn_225_exact is not None
                else metrics.kn_225.get(p_w)
            )
            # Exact against exact where the spec file carries unrounded speeds (see rsi.gates).
            base_ref = (
                base_kn_exact.get(p_w, base_kn) if metrics.kn_225_exact is not None else base_kn
            )
            d_exact = _delta(cand_exact, base_ref, 6)
            kn_deltas[str(p_w)] = d_exact
            if d_exact is not None:
                gains.append(d_exact)
    kn_deltas["mean_speed_gain_kn"] = round(sum(gains) / len(gains), 6) if gains else None
    race_gains = [kn_deltas[k] for k in ("90", "120") if kn_deltas.get(k) is not None]
    kn_deltas["race_pace_90_120_gain_kn"] = (
        round(sum(race_gains) / len(race_gains), 6) if race_gains else None
    )

    kn_265_deltas: dict[str, Any] | None = None
    if metrics.kn_265 is not None and baseline.kn_265 is not None:
        kn_265_deltas = {}
        gains_265: list[float] = []
        base_265_exact = baseline.kn_265_exact or {}
        for p_w, base_kn in baseline.kn_265.items():
            cand_exact = (
                metrics.kn_265_exact.get(p_w)
                if metrics.kn_265_exact is not None
                else metrics.kn_265.get(p_w)
            )
            base_ref = (
                base_265_exact.get(p_w, base_kn) if metrics.kn_265_exact is not None else base_kn
            )
            d_exact = _delta(cand_exact, base_ref, 6)
            kn_265_deltas[str(p_w)] = d_exact
            if d_exact is not None:
                gains_265.append(d_exact)
        kn_265_deltas["mean_speed_gain_kn"] = (
            round(sum(gains_265) / len(gains_265), 6) if gains_265 else None
        )

    loads_delta: dict[str, Any] = {}
    cand_loads = metrics.loads or {}
    for lb, base_ld in baseline.loads.items():
        c_ld = cand_loads.get(lb)
        if c_ld is None:
            loads_delta[str(lb)] = None
            continue
        loads_delta[str(lb)] = {
            "gm_cm": _delta(c_ld.gm_cm, base_ld.gm_cm, 4),
            "gz_mm": [_delta(c_ld.gz_mm[i], base_ld.gz_mm[i], 4) for i in range(3)],
            "t_cm": _delta(c_ld.t_cm, base_ld.t_cm, 3),
            "draft_mm": _delta(c_ld.draft_mm, base_ld.draft_mm, 3),
            "bwl_cm": _delta(c_ld.bwl_cm, base_ld.bwl_cm, 3),
            "wet_m2": _delta(c_ld.wet_m2, base_ld.wet_m2, 4),
            "lcg_mm": (
                round((c_ld.lcg_m - base_ld.lcg_m) * 1000.0, 3)
                if base_ld.lcg_m is not None and _finite_or_none(c_ld.lcg_m) is not None
                else None
            ),
        }

    wl_cut_keys = (
        "Waterlines 0.05",
        "Waterlines 0.06",
        "Waterlines 0.07",
        "Waterlines 0.08",
        "Waterlines 0.09",
        "Waterlines 0.1016",
    )
    base_wl_vals = [
        cut.max_signed_turn_forward_deg
        for name in wl_cut_keys
        if (cut := baseline.fairness.cuts.get(name)) is not None
        and cut.max_signed_turn_forward_deg is not None
    ]
    if not base_wl_vals:
        base_wl_vals = [
            cut.max_signed_turn_forward_deg
            for name, cut in baseline.fairness.cuts.items()
            if name.startswith("Waterlines ") and cut.max_signed_turn_forward_deg is not None
        ]
    base_fwd_wl_turn = max(base_wl_vals) if base_wl_vals else None
    base_b75 = baseline.fairness.cuts.get("Buttocks 0.075")
    forebody_sections = [s for s in baseline.fairness.sections if 0.11 <= s.fraction <= 0.35]
    if not forebody_sections:
        forebody_sections = [s for s in baseline.fairness.sections if s.fraction <= 0.35]
    base_k_max = max(
        (s.peak_abs_dk_ds_per_m2 for s in forebody_sections if s.peak_abs_dk_ds_per_m2 is not None),
        default=None,
    )
    base_groove = max((s.keel_groove_mm for s in forebody_sections), default=0.0)
    fairness_delta = {
        "waterline_hollow_max_deg": _finite_or_none(metrics.waterline_hollow_max_deg),
        "fwd_wl_max_turn_deg": _delta(
            metrics.fwd_wl_max_turn_deg,
            base_fwd_wl_turn,
            6,
        ),
        "entry_zone_max_turn_deg": _finite_or_none(metrics.entry_zone_max_turn_deg),
        "buttock_0075_peak_deg": _delta(
            metrics.buttock_0075_peak_deg,
            base_b75.peak_turn_change_deg if base_b75 else None,
            6,
        ),
        "forebody_max_keel_groove_mm": _delta(metrics.forebody_max_keel_groove_mm, base_groove, 6),
        "forebody_min_k": _finite_or_none(metrics.forebody_min_k),
        "forebody_max_abs_dk_ds": _delta(metrics.forebody_max_abs_dk_ds, base_k_max, 4),
    }

    seak_delta: dict[str, Any] | None = None
    if seakeeping is not None and seakeeping.valid:
        row_map: dict[tuple[float, str, str], dict[str, Any]] = {}
        for r in seakeeping.rows or ():
            if isinstance(r, dict) and "speed_kn" in r and "heading" in r and "sea" in r:
                sp_v = _finite_or_none(r.get("speed_kn"))
                if sp_v is not None:
                    row_map[(sp_v, str(r["heading"]), str(r["sea"]))] = r
        cond_deltas: list[dict[str, Any]] = []
        for base_r in baseline.capytaine.rows_225:
            cand_r = row_map.get((base_r.speed_kn, base_r.heading, base_r.sea))
            if cand_r is None:
                continue
            cond_deltas.append(
                {
                    "speed_kn": base_r.speed_kn,
                    "heading": base_r.heading,
                    "sea": base_r.sea,
                    "bow_dips_per_min_delta": _delta(
                        cand_r.get("bow_dips_per_min"), base_r.bow_dips_per_min, 4
                    ),
                    "seat_acc_rms_ms2_delta": _delta(
                        cand_r.get("seat_acc_rms_ms2"), base_r.seat_acc_rms_ms2, 4
                    ),
                    "heave_sig_cm_delta": _delta(
                        cand_r.get("heave_sig_cm"), base_r.heave_sig_cm, 4
                    ),
                    "pitch_sig_deg_delta": _delta(
                        cand_r.get("pitch_sig_deg"), base_r.pitch_sig_deg, 4
                    ),
                    # Not gated. Reported because added form stability shows up here first.
                    "roll_sig_deg_delta": _delta(
                        cand_r.get("roll_sig_deg"), base_r.roll_sig_deg, 4
                    ),
                    "yaw_mom_rms_Nm_delta": _delta(
                        cand_r.get("yaw_mom_rms_Nm"), base_r.yaw_mom_rms_Nm, 4
                    ),
                }
            )
        seak_delta = {
            "gm_t_cm": _delta(seakeeping.gm_t_cm, baseline.capytaine.gm_t_cm, 4),
            "roll_period_s": _delta(seakeeping.roll_period_s, baseline.capytaine.roll_period_s, 4),
            "conditions": cond_deltas,
        }

    cfd_delta: dict[str, Any] | None = None
    if cfd is not None and cfd.valid and not cfd.dry_run and isinstance(cfd.speeds, dict):
        cfd_delta = {}
        base_map = {
            "5.5kn": baseline.openfoam.speed_5_5kn,
            "6.2kn": baseline.openfoam.speed_6_2kn,
        }
        for sp_key, base_sp in base_map.items():
            cand_sp = cfd.speeds.get(sp_key)
            if isinstance(cand_sp, dict) and cfd.mode == PAIRED_MODE:
                # A paired result carries its own differences, taken against the baseline on the
                # same mesh from the same state. Its absolute drag is at another step count than
                # the published 500-step baseline and must not be subtracted from it.
                d_total = cand_sp.get("drag_delta_N")
                u_mps = float(cand_sp.get("speed_kn", 0.0)) * 0.514444
                cfd_delta[sp_key] = {
                    "drag_total_N": _delta(d_total, 0.0, 4),
                    "effective_power_W": _delta(d_total * u_mps, 0.0, 3)
                    if isinstance(d_total, (int, float))
                    else None,
                    "drag_pressure_N": _delta(cand_sp.get("drag_delta_pressure_N"), 0.0, 4),
                    "drag_viscous_N": _delta(cand_sp.get("drag_delta_viscous_N"), 0.0, 4),
                }
            elif isinstance(cand_sp, dict):
                cfd_delta[sp_key] = {
                    "drag_total_N": _delta(cand_sp.get("drag_total_N"), base_sp.drag_total_N, 4),
                    "effective_power_W": _delta(
                        cand_sp.get("effective_power_W"), base_sp.effective_power_W, 3
                    ),
                    "drag_pressure_N": _delta(
                        cand_sp.get("drag_pressure_N"), base_sp.drag_pressure_N, 4
                    ),
                    "drag_viscous_N": _delta(
                        cand_sp.get("drag_viscous_N"), base_sp.drag_viscous_N, 4
                    ),
                }

    catch_delta: list[float | None] | None = None
    if metrics.catch_widths_mm is not None and len(metrics.catch_widths_mm) == len(
        baseline.catch_widths_mm
    ):
        catch_delta = [
            _delta(c, b, 3) for c, b in zip(metrics.catch_widths_mm, baseline.catch_widths_mm)
        ]

    ld_225_c = cand_loads.get(225)
    ld_225_b = baseline.loads.get(225)
    report_only = {
        "draft_225_mm": _finite_or_none(ld_225_c.draft_mm) if ld_225_c is not None else None,
        "draft_225_mm_delta": (
            _delta(ld_225_c.draft_mm, ld_225_b.draft_mm, 3)
            if ld_225_c is not None and ld_225_b is not None
            else None
        ),
        "b_oa_cm": _finite_or_none(metrics.b_oa_cm),
        "b_oa_cm_delta": _delta(metrics.b_oa_cm, baseline.b_oa_cm, 4),
        "shell_area_m2": _finite_or_none(metrics.shell_area_m2),
        "shell_area_m2_delta": _delta(metrics.shell_area_m2, baseline.shell_area_m2, 6),
        "aft_coaming_above_225wl_cm": _finite_or_none(metrics.aft_coaming_above_225wl_cm),
        "aft_coaming_above_225wl_cm_delta": _delta(
            metrics.aft_coaming_above_225wl_cm, baseline.aft_coaming_above_225wl_cm, 3
        ),
        "rear_deck_above_225wl_cm": _finite_or_none(metrics.rear_deck_above_225wl_cm),
        "rear_deck_above_225wl_cm_delta": _delta(
            metrics.rear_deck_above_225wl_cm, baseline.rear_deck_above_225wl_cm, 3
        ),
        "upper_hollow_max_deg": _finite_or_none(metrics.upper_hollow_max_deg),
    }

    return {
        "class_form": class_form,
        "catch_widths_mm": catch_delta,
        "kn_225": kn_deltas,
        "kn_265": kn_265_deltas,
        "loads": loads_delta,
        "fairness": fairness_delta,
        "seakeeping": seak_delta,
        "cfd": cfd_delta,
        "report_only": report_only,
    }


def _seakeeping_from_dict(raw: dict[str, Any]) -> SeakeepingMetrics:
    baja_raw = raw.get("baja_265")
    return SeakeepingMetrics(
        valid=bool(raw.get("valid", False)),
        error=raw.get("error"),
        quick=bool(raw.get("quick", False)),
        elapsed_s=float(raw.get("elapsed_s", 0.0)),
        gm_t_cm=_finite_or_none(raw.get("gm_t_cm")),
        roll_period_s=_finite_or_none(raw.get("roll_period_s")),
        added_mass_surge_kg=_finite_or_none(raw.get("added_mass_surge_kg")),
        added_mass_sway_kg=_finite_or_none(raw.get("added_mass_sway_kg")),
        info=raw.get("info"),
        rows=tuple(raw.get("rows") or ()),
        output_json_path=str(raw.get("output_json_path", "")),
        baja_265=dict(baja_raw) if isinstance(baja_raw, dict) else None,
    )


def _cfd_from_dict(raw: dict[str, Any]) -> CfdMetrics:
    return CfdMetrics(
        valid=bool(raw.get("valid", False)),
        error=raw.get("error"),
        dry_run=bool(raw.get("dry_run", True)),
        elapsed_s=float(raw.get("elapsed_s", 0.0)),
        bashrc=str(raw.get("bashrc", resolve_openfoam_bashrc())),
        bashrc_exists=bool(raw.get("bashrc_exists", False)),
        staging_dir=str(raw.get("staging_dir", "")),
        posed=raw.get("posed"),
        mesh_info=raw.get("mesh_info"),
        speeds=raw.get("speeds"),
        output_json_path=str(raw.get("output_json_path", "")),
        mode=str(raw.get("mode") or "cold_start"),
    )


def _lookup_verified_tier2_cache(
    work_dir: Path,
    design_hash: str,
    *,
    require_full_seakeeping: bool,
    require_executed_cfd: bool,
    cfd_result_mode: str | None = None,
    paired_reference: str | None = None,
) -> tuple[SeakeepingMetrics | None, CfdMetrics | None]:
    """Safely look up cached Tier 2 artifacts matching exact design_hash.

    ``cfd_result_mode`` restricts the CFD result to one kind (cold start or paired): the two are
    judged by different gates, so one must never stand in for the other. A paired result is a
    difference against one reference; with ``paired_reference`` given, a result taken against any
    other reference is not accepted.
    """

    def _usable(c: CfdMetrics) -> bool:
        if cfd_result_mode is not None and c.mode != cfd_result_mode:
            return False
        if c.mode == PAIRED_MODE and paired_reference is not None:
            return (c.mesh_info or {}).get("reference") == paired_reference
        return True

    seak_found: SeakeepingMetrics | None = None
    cfd_found: CfdMetrics | None = None

    pareto_path = work_dir / "pareto.json"
    if pareto_path.is_file():
        try:
            pareto_data = json.loads(pareto_path.read_text())
            entry = (pareto_data.get("tier2_verified") or {}).get(design_hash)
            if isinstance(entry, dict):
                if isinstance(entry.get("seakeeping"), dict):
                    s = _seakeeping_from_dict(entry["seakeeping"])
                    if s.valid and (not require_full_seakeeping or not s.quick):
                        seak_found = s
                if isinstance(entry.get("cfd"), dict):
                    c = _cfd_from_dict(entry["cfd"])
                    if (
                        c.valid
                        and (not require_executed_cfd or not c.dry_run)
                        and _usable(c)
                        and isinstance(c.speeds, dict)
                        and "5.5kn" in c.speeds
                        and "6.2kn" in c.speeds
                    ):
                        cfd_found = c
        except Exception:  # noqa: BLE001
            pass

    if seak_found is not None and cfd_found is not None:
        return seak_found, cfd_found

    # Check direct work_dir, evaluate/<hash16>/, and compare/<hash16>/ when candidate.hull.json matches design_hash
    hash16 = design_hash.split(":")[-1][:16]
    candidate_locations = (
        (
            work_dir,
            work_dir / "capytaine_seakeeping.json",
            work_dir / "openfoam_cfd_metrics.json",
        ),
        (
            work_dir / "evaluate" / hash16,
            work_dir / "evaluate" / hash16 / "seakeeping" / "capytaine_seakeeping.json",
            work_dir / "evaluate" / hash16 / "cfd" / "openfoam_cfd_metrics.json",
        ),
        (
            work_dir / "compare" / hash16,
            work_dir / "compare" / hash16 / "seakeeping" / "capytaine_seakeeping.json",
            work_dir / "compare" / hash16 / "cfd" / "openfoam_cfd_metrics.json",
        ),
    )
    for base_dir, cap_path, of_path in candidate_locations:
        if seak_found is not None and cfd_found is not None:
            break
        cand_file = base_dir / "candidate.hull.json"
        if not cand_file.is_file():
            continue
        try:
            cached_hull = Hull.model_validate_json(cand_file.read_text())
            if cached_hull.design_hash() != design_hash:
                continue
            if seak_found is None and cap_path.is_file():
                cap_raw = json.loads(cap_path.read_text())
                baja_raw = cap_raw.get("baja_265")
                s = SeakeepingMetrics(
                    valid=True,
                    error=None,
                    quick=bool(cap_raw.get("quick", False)),
                    elapsed_s=0.0,
                    gm_t_cm=_finite_or_none(cap_raw.get("gm_t_cm")),
                    roll_period_s=_finite_or_none(cap_raw.get("roll_period_s")),
                    added_mass_surge_kg=_finite_or_none(cap_raw.get("added_mass_surge_kg")),
                    added_mass_sway_kg=_finite_or_none(cap_raw.get("added_mass_sway_kg")),
                    info=cap_raw.get("info"),
                    rows=tuple(cap_raw.get("rows") or ()),
                    output_json_path=str(cap_path),
                    baja_265=dict(baja_raw) if isinstance(baja_raw, dict) else None,
                )
                if not require_full_seakeeping or not s.quick:
                    seak_found = s

            if cfd_found is None and of_path.is_file():
                of_raw = json.loads(of_path.read_text())
                speeds = of_raw.get("speeds")
                of_mode = str(of_raw.get("mode") or "cold_start")
                if isinstance(speeds, dict) and "5.5kn" in speeds and "6.2kn" in speeds:
                    found = CfdMetrics(
                        valid=True,
                        error=None,
                        dry_run=False,
                        elapsed_s=0.0,
                        bashrc=str(resolve_openfoam_bashrc()),
                        bashrc_exists=resolve_openfoam_bashrc().is_file(),
                        staging_dir=str(of_path.parent),
                        posed=of_raw.get("posed"),
                        mesh_info=of_raw.get("mesh_info"),
                        speeds=speeds,
                        output_json_path=str(of_path),
                        mode=of_mode,
                    )
                    if _usable(found):
                        cfd_found = found
        except Exception:  # noqa: BLE001
            continue

    return seak_found, cfd_found


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison of a candidate Hull against v0.2.3 baseline."""

    candidate_path: str
    design_hash: str
    baseline_design_hash: str
    passed: bool
    tier01_passed: bool
    tier2_complete: bool
    tier2_passed: bool
    promotion_eligible: bool
    metrics: CandidateMetrics
    seakeeping: SeakeepingMetrics | None
    cfd: CfdMetrics | None
    gate_report: GateReport
    diagnostic: DiagnosticReport
    deltas: dict[str, Any]
    baseline_ref: BaselineReference | None = None

    def to_dict(self) -> dict[str, Any]:
        base_section: dict[str, Any] = {"design_hash": self.baseline_design_hash}
        if self.baseline_ref is not None:
            b = self.baseline_ref
            base_section.update(
                {
                    "sr_4in": b.sr_4in,
                    "sr_4in_exact": b.sr_4in_exact,
                    "usca_4in_pct": b.usca_4in_pct,
                    "usca_4in_pct_exact": b.usca_4in_pct_exact,
                    "l4_m": b.l4_m,
                    "b4_m": b.b4_m,
                    "b_4in_cm": round(b.b4_m * 100.0, 2),
                    "kn_225": {str(k): v for k, v in b.kn_225.items()},
                    "loads": {str(k): v.to_dict() for k, v in b.loads.items()},
                    "capytaine": {
                        "gm_t_cm": b.capytaine.gm_t_cm,
                        "roll_period_s": b.capytaine.roll_period_s,
                    },
                    "openfoam": {
                        "5.5kn": b.openfoam.speed_5_5kn.to_dict(),
                        "6.2kn": b.openfoam.speed_6_2kn.to_dict(),
                    },
                }
            )

        gates_dict = self.gate_report.to_dict()
        return {
            "candidate_path": self.candidate_path,
            "design_hash": self.design_hash,
            "baseline_design_hash": self.baseline_design_hash,
            "passed": self.passed,
            "tier01_passed": self.tier01_passed,
            "tier2_complete": self.tier2_complete,
            "tier2_passed": self.tier2_passed,
            "promotion_eligible": self.promotion_eligible,
            "candidate": {
                "design_hash": self.design_hash,
                "path": self.candidate_path,
                "metrics": augment_metrics_dict(self.metrics),
                "seakeeping": self.seakeeping.to_dict() if self.seakeeping is not None else None,
                "cfd": self.cfd.to_dict() if self.cfd is not None else None,
            },
            "baseline": base_section,
            "hydro_3load": build_hydro_3load_payload(self.metrics),
            "deltas": self.deltas,
            "gates": gates_dict,
            "gate_report": gates_dict,
            "diagnostic": self.diagnostic.to_dict(),
        }


@dataclass(frozen=True)
class PromotionReport:
    """Result of promote --dry-run or --apply check."""

    candidate_path: str
    target_hull_path: str
    dry_run: bool
    applied: bool
    passed: bool
    promotion_eligible: bool
    reason: str
    comparison: ComparisonReport
    backup_path: str | None = None
    verified_design_hash: str | None = None
    preview_diff: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        comp_dict = self.comparison.to_dict()
        return {
            "candidate_path": self.candidate_path,
            "target_hull_path": self.target_hull_path,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "reason": self.reason,
            "backup_path": self.backup_path,
            "verified_design_hash": self.verified_design_hash,
            "preview_diff": self.preview_diff,
            "comparison": comp_dict,
            "gates": comp_dict["gates"],
            "deltas": comp_dict["deltas"],
            "diagnostic": comp_dict["diagnostic"],
        }


def load_candidate_hull(candidate_path: Path) -> Hull:
    """Load and validate a candidate Hull JSON file."""
    path = Path(candidate_path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate hull file not found: {path}")
    return Hull.model_validate_json(path.read_text())


def compare_candidate(
    candidate_hull: Hull,
    baseline: BaselineReference,
    work_dir: Path,
    *,
    candidate_path: Path | None = None,
    tier2: bool = False,
    quick_seakeeping: bool = False,
    cfd_dry_run: bool = False,
    cfd_mode: str = "paired",
    skip_resistance: bool = False,
    mesh_resolution: tuple[int, int] = (441, 129),
    tier2_n_jobs: int = 6,
    metrics: CandidateMetrics | None = None,
    seakeeping: SeakeepingMetrics | None = None,
    cfd: CfdMetrics | None = None,
) -> ComparisonReport:
    """Evaluate (or accept precomputed) Tier 0/1 & optional Tier 2 metrics and compute deltas."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    d_hash = candidate_hull.design_hash()
    cand_work = work / "compare" / d_hash.split(":")[-1][:16]
    cand_work.mkdir(parents=True, exist_ok=True)

    eval_metrics = (
        metrics
        if metrics is not None
        else evaluate_tier0_tier1(
            candidate_hull,
            cand_work,
            baseline,
            skip_resistance=skip_resistance,
            mesh_resolution=mesh_resolution,
        )
    )

    tier01_report = evaluate_gates(eval_metrics, baseline)
    tier01_passed = tier01_report.passed

    eff_seak = seakeeping
    eff_cfd = cfd
    if tier2:
        cached_seak, cached_cfd = _lookup_verified_tier2_cache(
            work,
            d_hash,
            require_full_seakeeping=not quick_seakeeping,
            require_executed_cfd=not cfd_dry_run,
            cfd_result_mode=expected_result_mode(cfd_mode, cfd_dry_run),
            paired_reference=current_reference_key(baseline),
        )
        if eff_seak is None:
            eff_seak = (
                cached_seak
                if cached_seak is not None
                else evaluate_tier2_seakeeping(
                    candidate_hull,
                    cand_work / "seakeeping",
                    baseline,
                    quick=quick_seakeeping,
                    n_jobs=tier2_n_jobs,
                )
            )
        if eff_cfd is None:
            eff_cfd = (
                cached_cfd
                if cached_cfd is not None
                else evaluate_tier2_cfd_mode(
                    candidate_hull,
                    cand_work / "cfd",
                    baseline,
                    mode=cfd_mode,
                    dry_run=cfd_dry_run,
                )
            )

    gate_report = evaluate_gates(eval_metrics, baseline, seakeeping=eff_seak, cfd=eff_cfd)
    space = DesignSpace.default()
    vec: list[float] | None = None
    try:
        vec = hull_to_vector(candidate_hull, space)
    except Exception:  # noqa: BLE001
        vec = None

    diagnostic = diagnose_candidate(
        eval_metrics,
        gate_report,
        seakeeping=eff_seak,
        cfd=eff_cfd,
        vector=vec,
        space=space,
    )

    tier2_complete = bool(
        eff_seak is not None
        and eff_seak.valid
        and not eff_seak.quick
        and eff_cfd is not None
        and eff_cfd.valid
        and not eff_cfd.dry_run
        and isinstance(eff_cfd.speeds, dict)
        and "5.5kn" in eff_cfd.speeds
        and "6.2kn" in eff_cfd.speeds
    )
    tier2_passed = bool(tier2_complete and gate_report.passed)
    promotion_eligible = tier2_passed and eff_cfd is not None and eff_cfd.mode == PAIRED_MODE
    deltas = _compute_deltas(eval_metrics, baseline, eff_seak, eff_cfd)

    return ComparisonReport(
        candidate_path=str(candidate_path) if candidate_path is not None else eval_metrics.hull_json_path,
        design_hash=d_hash,
        baseline_design_hash=baseline.design_hash,
        passed=gate_report.passed,
        tier01_passed=tier01_passed,
        tier2_complete=tier2_complete,
        tier2_passed=tier2_passed,
        promotion_eligible=promotion_eligible,
        metrics=eval_metrics,
        seakeeping=eff_seak,
        cfd=eff_cfd,
        gate_report=gate_report,
        diagnostic=diagnostic,
        deltas=deltas,
        baseline_ref=baseline,
    )


def _build_preview_diff(candidate_hull: Hull, baseline: BaselineReference) -> dict[str, Any]:
    space = DesignSpace.default()
    base_vec = hull_to_vector(baseline.hull, space)
    try:
        cand_vec = hull_to_vector(candidate_hull, space)
    except Exception:  # noqa: BLE001
        return {
            "baseline_design_hash": baseline.design_hash,
            "candidate_design_hash": candidate_hull.design_hash(),
            "parameter_deltas": {},
        }

    param_deltas: dict[str, dict[str, float]] = {}
    for spec, bv, cv in zip(space.params, base_vec, cand_vec):
        if abs(cv - bv) > 1e-12:
            param_deltas[spec.name] = {
                "baseline": round(bv, 8),
                "candidate": round(cv, 8),
                "delta": round(cv - bv, 8),
            }
    return {
        "baseline_design_hash": baseline.design_hash,
        "candidate_design_hash": candidate_hull.design_hash(),
        "parameter_deltas": param_deltas,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def promote_candidate(
    candidate_path: Path,
    work_dir: Path,
    *,
    repo_root: Path | None = None,
    dry_run: bool = True,
    apply: bool = False,
    tier2: bool = False,
    quick_seakeeping: bool = False,
    cfd_dry_run: bool = False,
    cfd_mode: str = "paired",
    tier2_n_jobs: int = 6,
    metrics: CandidateMetrics | None = None,
    seakeeping: SeakeepingMetrics | None = None,
    cfd: CfdMetrics | None = None,
) -> PromotionReport:
    """Safely check and optionally apply a verified candidate hull to hull/kayak-10-9.hull.json."""
    if apply and dry_run:
        raise ValueError("Cannot pass both dry_run=True and apply=True")

    root = DEFAULT_REPO_ROOT if repo_root is None else Path(repo_root).resolve()
    target_hull_path = root / "hull" / "kayak-10-9.hull.json"
    baseline = load_baseline_reference(root)
    cand_hull = load_candidate_hull(candidate_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    preview = _build_preview_diff(cand_hull, baseline)

    eff_seak = seakeeping
    eff_cfd = cfd
    if not tier2 and (eff_seak is None or eff_cfd is None):
        cached_seak, cached_cfd = _lookup_verified_tier2_cache(
            work,
            cand_hull.design_hash(),
            require_full_seakeeping=True,
            require_executed_cfd=True,
            cfd_result_mode=expected_result_mode(cfd_mode, False),
            paired_reference=current_reference_key(baseline),
        )
        if eff_seak is None:
            eff_seak = cached_seak
        if eff_cfd is None:
            eff_cfd = cached_cfd

    comp = compare_candidate(
        cand_hull,
        baseline,
        work,
        candidate_path=candidate_path,
        tier2=tier2,
        quick_seakeeping=quick_seakeeping,
        cfd_dry_run=cfd_dry_run,
        cfd_mode=cfd_mode,
        tier2_n_jobs=tier2_n_jobs,
        metrics=metrics,
        seakeeping=eff_seak,
        cfd=eff_cfd,
    )

    if not comp.passed:
        reason = f"Candidate failed {len(comp.gate_report.violations)} gate(s)."
    elif not comp.tier2_complete:
        reason = "Promotion requires full seakeeping and executed paired CFD at both speeds."
    elif not comp.promotion_eligible:
        reason = "The cold CFD screen passed. Promotion requires a paired CFD comparison."
    else:
        reason = "All promotion gates passed."

    if not apply or not comp.promotion_eligible:
        return PromotionReport(
            candidate_path=str(candidate_path),
            target_hull_path=str(target_hull_path),
            dry_run=not apply,
            applied=False,
            passed=comp.passed if not apply else False,
            promotion_eligible=comp.promotion_eligible,
            reason=reason,
            comparison=comp,
            preview_diff=preview,
        )

    backup_path = work / "backup.kayak-10-9.hull.json"
    original_text = target_hull_path.read_text(encoding="utf-8")
    _atomic_write_text(backup_path, original_text)

    new_text = cand_hull.model_dump_json(indent=2) + "\n"
    try:
        _atomic_write_text(target_hull_path, new_text)
        reloaded = Hull.model_validate_json(target_hull_path.read_text(encoding="utf-8"))
        if reloaded.design_hash() != cand_hull.design_hash():
            raise RuntimeError(
                f"Hash mismatch after write: {reloaded.design_hash()} != {cand_hull.design_hash()}"
            )
    except Exception:
        _atomic_write_text(target_hull_path, original_text)
        raise

    return PromotionReport(
        candidate_path=str(candidate_path),
        target_hull_path=str(target_hull_path),
        dry_run=False,
        applied=True,
        passed=True,
        promotion_eligible=True,
        reason="Candidate verified across Tier 0/1 + full Tier 2 and promoted atomically.",
        comparison=comp,
        backup_path=str(backup_path),
        verified_design_hash=reloaded.design_hash(),
        preview_diff=preview,
    )


__all__ = [
    "ComparisonReport",
    "PromotionReport",
    "augment_metrics_dict",
    "build_hydro_3load_payload",
    "compare_candidate",
    "load_candidate_hull",
    "promote_candidate",
]
