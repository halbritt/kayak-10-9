"""Catch-width gate, unrounded baseline comparisons, and the diagnostics that steer around them."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from rsi.baseline import BaselineReference, load_baseline_reference
from rsi.diagnose import AdaptationHints, DiagnosticReport, aggregate_diagnostics, diagnose_candidate
from rsi.evaluator import CATCH_OFFSETS_M, CandidateMetrics, LoadEvalMetrics, evaluate_tier0_tier1
from rsi.gates import CATCH_WIDTH_TOL_MM, GM_TOL_CM, KN_TOL_KN, evaluate_gates
from rsi.genome import DesignSpace, vector_to_hull
from rsi.promote import compare_candidate

# Forward section side exponent of campaign k109_rsi_main's iter046_cand006 (baseline 2.0). On its
# own this one parameter reproduces most of that hull's flare, stability gain and wider catch.
FLARED_SEC_SIDE_FWD = 1.657


def _baseline_like(ref: BaselineReference, **overrides: object) -> CandidateMetrics:
    """Metrics equal to the baseline on every gated quantity, built without a geometry run."""
    loads = {
        lb: LoadEvalMetrics(
            load_lb=lb,
            t_cm=b.t_cm,
            draft_mm=b.draft_mm,
            bwl_cm=b.bwl_cm,
            wet_m2=b.wet_m2,
            gm_cm=b.gm_cm_exact if b.gm_cm_exact is not None else b.gm_cm,
            gz_mm=b.gz_mm,
            lcg_m=0.0,
        )
        for lb, b in ref.loads.items()
    }
    fields: dict[str, object] = dict(
        valid=True,
        error=None,
        design_hash=ref.design_hash,
        hull_json_path="",
        mesh_resolution=(441, 129),
        elapsed_s=0.0,
        skip_resistance=False,
        is_watertight=True,
        is_volume=True,
        n_faces=1,
        z_min_m=0.0,
        l_oa_m=ref.l_oa_m,
        b_oa_cm=ref.b_oa_cm,
        l4_m=ref.l4_m,
        b4_m=ref.b4_m,
        sr_4in_exact=ref.sr_4in_exact,
        sr_4in=ref.sr_4in,
        usca_4in_pct_exact=ref.usca_4in_pct_exact,
        usca_4in_pct=ref.usca_4in_pct,
        catch_widths_mm=ref.catch_widths_mm,
        transition_dB_per_20cm=ref.transition_dB_per_20cm,
        has_surface_fairing=True,
        waterline_hollow_max_deg=0.0,
        fwd_wl_max_turn_deg=-0.03,
        entry_zone_max_turn_deg=0.02,
        buttock_0075_peak_deg=3.46,
        forebody_max_keel_groove_mm=0.0,
        forebody_min_k=2.0,
        forebody_max_abs_dk_ds=190.0,
        loads=loads,
        kn_225=dict(ref.kn_225),
        kn_225_exact=dict(ref.kn_225_exact or ref.kn_225),
    )
    fields.update(overrides)
    return CandidateMetrics(**fields)  # type: ignore[arg-type]


def _widened(ref: BaselineReference, by_offset_mm: dict[float, float]) -> tuple[float, ...]:
    return tuple(
        w + by_offset_mm.get(off, 0.0) for off, w in zip(CATCH_OFFSETS_M, ref.catch_widths_mm)
    )


def _violated(report, gate: str) -> bool:  # noqa: ANN001
    return any(v.gate == gate for v in report.violations)


def test_spec_file_carries_unrounded_baseline_that_rounds_to_the_published_table() -> None:
    ref = load_baseline_reference()
    assert ref.kn_225_exact is not None
    assert set(ref.kn_225_exact) == set(ref.kn_225)
    for p_w, published in ref.kn_225.items():
        assert round(ref.kn_225_exact[p_w], 2) == published
    for load in ref.loads.values():
        assert load.gm_cm_exact is not None
        assert round(load.gm_cm_exact, 2) == load.gm_cm

    # A hull identical to the baseline must score exactly zero against it: no speed "gain" and no
    # stability "loss" manufactured by comparing unrounded numbers with a rounded table.
    report = evaluate_gates(_baseline_like(ref), ref)
    assert report.passed is True, report.violations
    assert abs(report.objectives["speed_gain_kn"]) < 1e-12
    assert abs(report.objectives["stability_min_margin"]) < 1e-9


def test_catch_gate_is_one_sided_with_the_project_tolerance() -> None:
    ref = load_baseline_reference()
    assert len(ref.catch_widths_mm) == len(CATCH_OFFSETS_M)

    at_baseline = evaluate_gates(_baseline_like(ref), ref)
    assert at_baseline.passed is True
    assert math.isclose(at_baseline.margins["catch_width_mm"], CATCH_WIDTH_TOL_MM, abs_tol=1e-9)

    just_inside = evaluate_gates(
        _baseline_like(ref, catch_widths_mm=_widened(ref, {1.0: CATCH_WIDTH_TOL_MM - 0.1})), ref
    )
    assert just_inside.passed is True
    assert math.isclose(just_inside.margins["catch_width_mm"], 0.1, abs_tol=1e-9)

    just_outside = evaluate_gates(
        _baseline_like(ref, catch_widths_mm=_widened(ref, {1.0: CATCH_WIDTH_TOL_MM + 0.1})), ref
    )
    assert just_outside.passed is False
    assert [v.gate for v in just_outside.violations] == ["catch_width_mm"]
    assert "+1.0 m" in just_outside.violations[0].message
    assert math.isclose(just_outside.margins["catch_width_mm"], -0.1, abs_tol=1e-9)

    # The worst station decides, wherever it is along the stroke.
    worst_aft = evaluate_gates(
        _baseline_like(ref, catch_widths_mm=_widened(ref, {0.8: 1.0, 1.2: 5.0})), ref
    )
    assert _violated(worst_aft, "catch_width_mm")
    assert "+1.2 m" in worst_aft.violations[0].message

    # Narrower is an improvement, never a violation.
    narrower = evaluate_gates(
        _baseline_like(ref, catch_widths_mm=tuple(w - 25.0 for w in ref.catch_widths_mm)), ref
    )
    assert narrower.passed is True
    assert math.isclose(narrower.margins["catch_width_mm"], CATCH_WIDTH_TOL_MM + 25.0, abs_tol=1e-9)


def test_catch_gate_fails_closed_without_usable_widths() -> None:
    ref = load_baseline_reference()
    for bad in (None, ref.catch_widths_mm[:-1], (*ref.catch_widths_mm[:-1], float("nan"))):
        report = evaluate_gates(_baseline_like(ref, catch_widths_mm=bad), ref)
        assert report.passed is False, bad
        assert _violated(report, "catch_width_mm"), bad
        assert report.margins["catch_width_mm"] == -1.0


def test_speed_gate_compares_unrounded_against_unrounded() -> None:
    ref = load_baseline_reference()
    assert ref.kn_225_exact is not None
    exact = ref.kn_225_exact

    def with_speed(p_w: int, delta: float) -> CandidateMetrics:
        kn_exact = dict(exact)
        kn_exact[p_w] = exact[p_w] + delta
        return _baseline_like(
            ref, kn_225_exact=kn_exact, kn_225={k: round(v, 2) for k, v in kn_exact.items()}
        )

    # iter046_cand006's 40 W regression: its exact speed still rounds to the published 4.47, so the
    # comparison against the rounded table let a loss larger than the tolerance through.
    slow_40 = with_speed(40, -0.0056)
    assert slow_40.kn_225 is not None and slow_40.kn_225[40] == ref.kn_225[40]
    report = evaluate_gates(slow_40, ref)
    assert _violated(report, "kn_225_40w")
    assert math.isclose(report.margins["kn_225_40w"], KN_TOL_KN - 0.0056, abs_tol=1e-12)
    assert math.isclose(report.objectives["speed_gain_kn"], -0.0056 / 4.0, abs_tol=1e-12)

    assert evaluate_gates(with_speed(40, -(KN_TOL_KN - 1e-4)), ref).passed is True

    # The tolerance is the same at every power level. A 0.004 kn loss at 60 W changes the published
    # second decimal (5.08 -> 5.07), which the rounded comparison counted as a 0.01 kn regression.
    small_60 = with_speed(60, -0.004)
    assert small_60.kn_225 is not None and small_60.kn_225[60] < ref.kn_225[60]
    assert evaluate_gates(small_60, ref).passed is True
    assert _violated(evaluate_gates(with_speed(60, -(KN_TOL_KN + 1e-4)), ref), "kn_225_60w")

    # A spec file without unrounded speeds keeps the previous comparison.
    legacy_ref = replace(ref, kn_225_exact=None)
    assert evaluate_gates(slow_40, legacy_ref).passed is True


def test_gm_gate_uses_the_unrounded_baseline() -> None:
    ref = load_baseline_reference()
    base_metrics = _baseline_like(ref)
    assert base_metrics.loads is not None
    report = evaluate_gates(base_metrics, ref)
    for lb in ref.loads:
        assert math.isclose(report.margins[f"gm_{lb}_cm"], GM_TOL_CM, abs_tol=1e-12)

    def with_gm(lb: int, delta: float) -> CandidateMetrics:
        loads = dict(base_metrics.loads or {})
        loads[lb] = replace(loads[lb], gm_cm=loads[lb].gm_cm + delta)
        return _baseline_like(ref, loads=loads)

    assert evaluate_gates(with_gm(265, -(GM_TOL_CM - 1e-3)), ref).passed is True
    assert _violated(evaluate_gates(with_gm(265, -(GM_TOL_CM + 1e-3)), ref), "gm_265_cm")


def test_catch_violation_steers_forward_sections_back() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()["sec_side_fwd"]
    vec = space.baseline_vector()
    vec[idx] = FLARED_SEC_SIDE_FWD

    metrics = _baseline_like(ref, catch_widths_mm=_widened(ref, {o: 30.0 for o in CATCH_OFFSETS_M}))
    report = evaluate_gates(metrics, ref)
    diag = diagnose_candidate(metrics, report, vector=vec, space=space)

    assert diag.feasible is False
    assert "catch_width_widened" in diag.failure_modes
    assert diag.hints.preferred_directions["sec_side_fwd"] > 0.0
    rule = diag.hints.avoid_bounds["sec_side_fwd"]
    assert rule["side"] == "lower"
    assert math.isclose(rule["limit"], FLARED_SEC_SIDE_FWD)

    # No flare added relative to the baseline: there is no known-bad value to fence off.
    diag_base_vec = diagnose_candidate(metrics, report, vector=space.baseline_vector(), space=space)
    assert "sec_side_fwd" not in diag_base_vec.hints.avoid_bounds
    assert diag_base_vec.hints.preferred_directions["sec_side_fwd"] > 0.0


def test_flare_nudges_stop_when_the_catch_margin_is_tight() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()
    vec = space.baseline_vector()

    # Feasible, SR below target, no stability buffer: the stock hint asks for slacker forward sections.
    roomy = _baseline_like(ref)
    roomy_report = evaluate_gates(roomy, ref)
    assert roomy_report.passed is True
    roomy_diag = diagnose_candidate(roomy, roomy_report, vector=vec, space=space)
    assert roomy_diag.hints.preferred_directions["sec_side_fwd"] < 0.0

    # Same hull with most of the catch tolerance used up: still feasible, but no more flare.
    used = CATCH_WIDTH_TOL_MM * 0.75
    tight = _baseline_like(ref, catch_widths_mm=_widened(ref, {1.0: used}))
    tight_report = evaluate_gates(tight, ref)
    assert tight_report.passed is True
    tight_diag = diagnose_candidate(tight, tight_report, vector=vec, space=space)
    assert tight_diag.hints.preferred_directions.get("sec_side_fwd", 0.0) >= 0.0

    # Within a tenth of the tolerance the margin is reported as tight.
    edge = _baseline_like(ref, catch_widths_mm=_widened(ref, {1.0: CATCH_WIDTH_TOL_MM * 0.95}))
    edge_diag = diagnose_candidate(edge, evaluate_gates(edge, ref), vector=vec, space=space)
    assert "catch_width_mm" in edge_diag.hints.tight_margins


def test_aggregate_keeps_the_most_conservative_avoid_bound_in_any_order() -> None:
    def report(limit: float, side: str = "lower") -> DiagnosticReport:
        return DiagnosticReport(
            feasible=False,
            failure_modes=("catch_width_widened",),
            primary_groups=("section",),
            items=(),
            hints=AdaptationHints(
                avoid_bounds={"sec_side_fwd": {"side": side, "limit": limit, "reason": "test"}}
            ),
            summary="",
        )

    for order in ((1.70, 1.95), (1.95, 1.70)):
        merged = aggregate_diagnostics([report(v) for v in order])
        assert merged.avoid_bounds["sec_side_fwd"]["limit"] == 1.95
    for order in ((1.70, 1.95), (1.95, 1.70)):
        merged = aggregate_diagnostics([report(v, "upper") for v in order])
        assert merged.avoid_bounds["sec_side_fwd"]["limit"] == 1.70


def test_comparison_report_shows_catch_and_unrounded_speed_deltas() -> None:
    ref = load_baseline_reference()
    assert ref.kn_225_exact is not None
    work = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "catch_gate_compare"
    kn_exact = {p: v + 0.010 for p, v in ref.kn_225_exact.items()}
    metrics = _baseline_like(
        ref,
        catch_widths_mm=_widened(ref, {1.0: 28.3}),
        kn_225_exact=kn_exact,
        kn_225={p: round(v, 2) for p, v in kn_exact.items()},
    )
    report = compare_candidate(ref.hull, ref, work, metrics=metrics)

    assert report.passed is False
    deltas = report.deltas
    assert deltas["catch_widths_mm"] == [0.0, 0.0, 0.0, 28.3, 0.0, 0.0]
    assert deltas["kn_225"] == {
        "40": 0.01,
        "60": 0.01,
        "90": 0.01,
        "120": 0.01,
        "mean_speed_gain_kn": 0.01,
    }


def test_forward_flare_trips_the_catch_gate_on_real_geometry() -> None:
    """Slack forward sections buy heeled stability with flare and pay for it at the catch."""
    ref = load_baseline_reference()
    space = DesignSpace.default()
    vec = space.baseline_vector()
    vec[space.index_by_name()["sec_side_fwd"]] = FLARED_SEC_SIDE_FWD
    flared = vector_to_hull(vec, ref.hull, space)

    work = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "catch_gate_flare"
    metrics = evaluate_tier0_tier1(flared, work, ref, skip_resistance=True)
    assert metrics.valid is True, metrics.error
    assert metrics.catch_widths_mm is not None and metrics.loads is not None

    widening_at_1m = metrics.catch_widths_mm[3] - ref.catch_widths_mm[3]
    assert widening_at_1m > 10.0 * CATCH_WIDTH_TOL_MM
    # The reward that made this direction attractive to the search before the gate existed.
    assert metrics.loads[225].gz_mm[2] > ref.loads[225].gz_mm[2] + 1.0

    report = evaluate_gates(metrics, ref)
    assert report.passed is False
    assert _violated(report, "catch_width_mm")
