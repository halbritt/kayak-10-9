"""Catch-width gate, unrounded baseline comparisons, and the diagnostics that steer around them."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from rsi.baseline import BaselineReference, load_baseline_reference
from rsi.diagnose import AdaptationHints, DiagnosticReport, aggregate_diagnostics, diagnose_candidate
from rsi.evaluator import CATCH_OFFSETS_M, CandidateMetrics, LoadEvalMetrics, evaluate_tier0_tier1
from rsi.fidelity import SeakeepingMetrics
from rsi.gates import (
    CATCH_WIDTH_TOL_MM,
    GM_TOL_CM,
    KN_TOL_KN,
    RACE_PACE_POWERS_W,
    SEAKEEPING_ASTERN_HEADINGS,
    SEAKEEPING_ASTERN_ROLL_REL_ALLOWANCE,
    evaluate_gates,
)
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
            lcg_m=b.lcg_m if b.lcg_m is not None else 0.0,
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
        foot_clear_height_m=0.2624,
        foot_clear_width_m=0.2995,
        loads=loads,
        kn_225=dict(ref.kn_225),
        kn_225_exact=dict(ref.kn_225_exact or ref.kn_225),
        kn_265=dict(ref.kn_265 or {}),
        kn_265_exact=dict(ref.kn_265_exact or ref.kn_265 or {}),
        shell_area_m2=ref.shell_area_m2,
        aft_coaming_above_225wl_cm=ref.aft_coaming_above_225wl_cm,
        rear_deck_above_225wl_cm=ref.rear_deck_above_225wl_cm,
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
    # Owner, 2026-11-27: the objective is the mean gain at 60 and 90 W. 40 and 120 W are gated
    # against regression and carry no weight in it.
    assert RACE_PACE_POWERS_W == (60, 90)
    assert abs(report.objectives["speed_gain_kn"]) < 1e-12
    assert math.isclose(evaluate_gates(with_speed(60, 0.010), ref).objectives["speed_gain_kn"], 0.005, abs_tol=1e-12)
    assert math.isclose(evaluate_gates(with_speed(90, 0.006), ref).objectives["speed_gain_kn"], 0.003, abs_tol=1e-12)
    fast_120 = evaluate_gates(with_speed(120, 0.050), ref)
    assert fast_120.passed is True and abs(fast_120.objectives["speed_gain_kn"]) < 1e-12
    # 120 W may not be traded for the objective: a loss beyond the tolerance fails, a gain at
    # 60 and 90 W notwithstanding.
    assert evaluate_gates(with_speed(120, -(KN_TOL_KN - 1e-4)), ref).passed is True
    slow_120 = evaluate_gates(with_speed(120, -(KN_TOL_KN + 1e-4)), ref)
    assert [v.gate for v in slow_120.violations] == ["kn_225_120w"]
    assert math.isclose(slow_120.margins["kn_225_120w"], -1e-4, abs_tol=1e-12)

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
    assert math.isclose(report.margins["gm_225_cm"], GM_TOL_CM, abs_tol=1e-12)
    # Only the 225 lb race load is gated (owner, 2026-11-27); 265 and 300 lb are reported.
    assert not any(k.startswith(("gm_265", "gm_300", "gz_265", "gz_300")) for k in report.margins)

    def with_gm(lb: int, delta: float) -> CandidateMetrics:
        loads = dict(base_metrics.loads or {})
        loads[lb] = replace(loads[lb], gm_cm=loads[lb].gm_cm + delta)
        return _baseline_like(ref, loads=loads)

    assert evaluate_gates(with_gm(225, -(GM_TOL_CM - 1e-3)), ref).passed is True
    assert _violated(evaluate_gates(with_gm(225, -(GM_TOL_CM + 1e-3)), ref), "gm_225_cm")
    for lb in (265, 300):
        assert evaluate_gates(with_gm(lb, -1.0), ref).passed is True


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
        "race_pace_powers_w": [60, 90],
        "race_pace_gain_kn": 0.01,
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


def test_gates_foot_box_lcb_pure_speed_utility_and_report_only_sr_and_265_speed() -> None:
    from rsi.gates import FOOT_BOX_MIN_HEIGHT_M, FOOT_BOX_MIN_WIDTH_M, LCB_225_TOL_MM
    from rsi.loop import PARETO_OBJECTIVES, _composite_utility

    ref = load_baseline_reference()
    assert ref.kn_265_exact is not None and ref.loads[225].lcg_m is not None

    # 2b / 2d: stability and SR closeness are floors only, carrying zero weight in composite utility
    assert PARETO_OBJECTIVES == ("speed_gain_kn", "fairness_margin", "seakeeping_gain")
    u_base = _composite_utility(
        {"sr_closeness_to_10_9": -0.05, "speed_gain_kn": 0.01, "stability_min_margin": 0.0, "fairness_margin": 0.10, "seakeeping_gain": 0.0}
    )
    u_more_stab_and_sr = _composite_utility(
        {"sr_closeness_to_10_9": 0.0, "speed_gain_kn": 0.01, "stability_min_margin": 0.50, "fairness_margin": 0.10, "seakeeping_gain": 0.0}
    )
    assert u_base == u_more_stab_and_sr

    # Owner, 2026-11-27: the Sound Rowers band is no longer a requirement. The ratio is reported
    # and still feeds the report-only closeness objective.
    below_old_floor = evaluate_gates(_baseline_like(ref, sr_4in_exact=10.874, sr_4in=10.87), ref)
    assert below_old_floor.passed is True
    assert math.isclose(below_old_floor.objectives["sr_closeness_to_10_9"], -abs(10.874 - 10.90), abs_tol=1e-12)
    assert evaluate_gates(_baseline_like(ref, sr_4in_exact=11.40, sr_4in=11.40), ref).passed is True

    # Owner, 2026-11-27: the 265 lb speed gate is dropped; the speeds are reported.
    bad_265 = {**ref.kn_265_exact, 120: ref.kn_265_exact[120] - 0.2}
    r265 = evaluate_gates(_baseline_like(ref, kn_265_exact=bad_265, kn_265={k: round(v, 2) for k, v in bad_265.items()}), ref)
    assert r265.passed is True and not any(k.startswith("kn_265") for k in r265.margins)
    assert evaluate_gates(_baseline_like(ref, kn_265=None, kn_265_exact=None), ref).passed is True
    # The 225 lb race-load speed gate stays.
    bad_225 = {**ref.kn_225_exact, 40: ref.kn_225_exact[40] - (KN_TOL_KN + 0.001)}
    r225 = evaluate_gates(_baseline_like(ref, kn_225_exact=bad_225, kn_225={k: round(v, 2) for k, v in bad_225.items()}), ref)
    assert _violated(r225, "kn_225_40w")

    # 2f: Foot box clearances >= 28 x 25 cm (fail closed on small/None/NaN)
    assert _violated(evaluate_gates(_baseline_like(ref, foot_clear_height_m=FOOT_BOX_MIN_HEIGHT_M - 0.001), ref), "foot_clear_height_m")
    assert _violated(evaluate_gates(_baseline_like(ref, foot_clear_width_m=FOOT_BOX_MIN_WIDTH_M - 0.001), ref), "foot_clear_width_m")
    assert _violated(evaluate_gates(_baseline_like(ref, foot_clear_height_m=None), ref), "foot_clear_height_m")

    # 2g: 225 lb LCB shift <= +-10 mm
    base_loads = _baseline_like(ref).loads
    assert base_loads is not None
    ok_lcb = {**base_loads, 225: replace(base_loads[225], lcg_m=ref.loads[225].lcg_m + (LCB_225_TOL_MM - 0.2) / 1000.0)}
    assert evaluate_gates(_baseline_like(ref, loads=ok_lcb), ref).passed is True
    bad_lcb = {**base_loads, 225: replace(base_loads[225], lcg_m=ref.loads[225].lcg_m + (LCB_225_TOL_MM + 0.2) / 1000.0)}
    assert _violated(evaluate_gates(_baseline_like(ref, loads=bad_lcb), ref), "lcb_225_mm")


def test_fixed_frame_catch_and_report_only_fields_on_real_geometry() -> None:
    """2c/2e/2i: catch uses fixed baseline 225 lb LCG stations; 265 lb speed & report-only metrics populate."""
    import trimesh
    from rsi.evaluator import _ensure_cad_on_path

    _ensure_cad_on_path()
    from check_fairing_controls import catch_widths as controls_catch_widths
    from deck_geometry import SculptedKayakSurface
    from hydro import RHO_SEAWATER, float_at
    import numpy as np

    ref = load_baseline_reference()
    assert ref.loads[225].lcg_m is not None
    space = DesignSpace.default()
    vec = space.baseline_vector()
    vec[space.index_by_name()["wl_a1"]] += 0.012
    shifted_hull = vector_to_hull(vec, ref.hull, space)

    work = Path.home() / ".cache" / "kayak-10-9" / "test-work" / "v4_fixed_catch"
    m = evaluate_tier0_tier1(shifted_hull, work, ref, skip_resistance=False)
    assert m.valid is True, m.error
    assert m.loads is not None and abs(m.loads[225].lcg_m - ref.loads[225].lcg_m) > 0.002
    assert m.kn_265 is not None and m.kn_265_exact is not None
    # 230 lb (100-mile race load) is a reported case: between the 225 and 265 lb figures.
    assert m.kn_230_exact is not None and set(m.loads) == {225, 230, 265, 300}
    for p_w in (40, 60, 90, 120):
        assert m.kn_265_exact[p_w] < m.kn_230_exact[p_w] < m.kn_225_exact[p_w]
    assert m.loads[265].gm_cm < m.loads[230].gm_cm < m.loads[225].gm_cm
    assert m.to_dict()["kn_230_exact"]["90"] == m.kn_230_exact[90]
    assert m.shell_area_m2 is not None and m.aft_coaming_above_225wl_cm is not None and m.rear_deck_above_225wl_cm is not None

    surf = SculptedKayakSurface(Path(m.hull_json_path))
    closed = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
    closed.merge_vertices(merge_tex=True, merge_norm=True)
    V, F = np.asarray(closed.vertices, dtype=float), np.asarray(closed.faces, dtype=int)
    r_up = float_at(V, F, 225.0 * 0.45359237, RHO_SEAWATER, np.array([0.0, 0.0, float(closed.bounds[0, 2]) + 0.25]), free_trim=False)
    expected_fixed = tuple(float(x) for x in controls_catch_widths(closed, float(ref.loads[225].lcg_m), float(r_up["h"])))
    assert m.catch_widths_mm is not None
    for a, b in zip(m.catch_widths_mm, expected_fixed):
        assert math.isclose(a, b, abs_tol=1e-6)

    comp = compare_candidate(shifted_hull, ref, work, metrics=m)
    assert "report_only" in comp.deltas and comp.deltas["kn_265"] is not None
    assert comp.deltas["report_only"]["kn_230_exact"]["120"] == m.kn_230_exact[120]
    assert comp.deltas["report_only"]["load_230"]["gm_cm"] == m.loads[230].gm_cm
    assert "race_pace_gain_kn" in comp.deltas["kn_225"] and comp.deltas["kn_225"]["race_pace_powers_w"] == [60, 90]


def _seak_keyed(rows: tuple) -> dict:
    return {(r.speed_kn, r.heading, r.sea): r for r in rows}


def _seak_metrics(rows: list[dict], *, quick: bool = False, gm_t_cm: float = 0.0) -> SeakeepingMetrics:
    return SeakeepingMetrics(
        valid=True, error=None, quick=quick, elapsed_s=1.0, gm_t_cm=gm_t_cm, rows=tuple(rows)
    )


_AHEAD_HEADINGS = ("head", "bow_quarter")
_AHEAD_FIELDS = ("bow_dips_per_min", "seat_acc_rms_ms2")
_ASTERN_FIELDS = ("pitch_sig_deg", "bow_dips_per_min", "seat_acc_rms_ms2", "roll_sig_deg")


def _seak_rows_toward_floor(
    ref: BaselineReference, frac: float, *, headings: tuple[str, ...] | None = None, **scale: float
) -> list[dict]:
    """Rows frac of the way from v0.2.3 (0) to the seakeeping limit (1), optionally scaled per field.

    Head and bow-quarter rows are limited by the worse of the Epic 18X and v0.2.3; following and
    stern-quarter rows by the Epic 18X row alone, with 20 % more roll allowed. ``headings``
    restricts the scaling to those headings.
    """
    floor = _seak_keyed(ref.capytaine.epic18x_conditions_225)
    rows = []
    for key, base in _seak_keyed(ref.capytaine.target_conditions_225).items():
        astern = key[1] in SEAKEEPING_ASTERN_HEADINGS
        row: dict = {"speed_kn": key[0], "heading": key[1], "sea": key[2]}
        for fld in _ASTERN_FIELDS if astern else _AHEAD_FIELDS:
            b = getattr(base, fld)
            if astern:
                lim = getattr(floor[key], fld)
                if fld == "roll_sig_deg":
                    lim *= 1.0 + SEAKEEPING_ASTERN_ROLL_REL_ALLOWANCE
            else:
                lim = max(getattr(floor[key], fld), b)
            at = lim if frac >= 1.0 else b + frac * (lim - b)
            scaled = headings is None or key[1] in headings
            row[fld] = at * (scale.get(fld, 1.0) if scaled else 1.0)
        rows.append(row)
    return rows


def test_seakeeping_floor_is_the_epic_18x_not_the_baseline() -> None:
    """Owner, 2026-11-26: seakeeping may not regress to be worse than the Epic 18X."""
    ref = load_baseline_reference()
    base = _seak_keyed(ref.capytaine.target_conditions_225)
    floor = _seak_keyed(ref.capytaine.epic18x_conditions_225)
    assert set(floor) == set(base) and len(base) == 16
    metrics = _baseline_like(ref)
    gm = ref.capytaine.gm_t_cm

    # v0.2.3, a hull half-way to the floor and a hull exactly at the floor all pass.
    for frac in (0.0, 0.5, 1.0):
        rep = evaluate_gates(metrics, ref, seakeeping=_seak_metrics(_seak_rows_toward_floor(ref, frac), gm_t_cm=gm))
        assert rep.passed is True, (frac, rep.violations)
    # The half-way hull is well outside the earlier limit of v0.2.3 + 1.5 % in some condition.
    half = _seak_rows_toward_floor(ref, 0.5)
    assert any(r["bow_dips_per_min"] > 1.05 * base[(r["speed_kn"], r["heading"], r["sea"])].bow_dips_per_min for r in half)

    # One per cent beyond the floor fails, on the measure that crossed it and no other.
    for fld, gate_prefix in (("seat_acc_rms_ms2", "seak_seat_acc_"), ("bow_dips_per_min", "seak_bow_dips_")):
        rep = evaluate_gates(
            metrics,
            ref,
            seakeeping=_seak_metrics(
                _seak_rows_toward_floor(ref, 1.0, headings=_AHEAD_HEADINGS, **{fld: 1.01}), gm_t_cm=gm
            ),
        )
        assert rep.passed is False
        assert {v.gate[: len(gate_prefix)] for v in rep.violations} == {gate_prefix}
        assert len(rep.violations) == 8


def test_seakeeping_limit_is_the_baseline_where_the_baseline_is_behind_the_18x() -> None:
    """Head and bow-quarter seas: the baseline passes its own gate, and nothing may fall below both boats."""
    ref = load_baseline_reference()
    better_18x = tuple(
        replace(r, seat_acc_rms_ms2=0.5 * r.seat_acc_rms_ms2, bow_dips_per_min=0.5 * r.bow_dips_per_min)
        if r.heading in _AHEAD_HEADINGS
        else r
        for r in ref.capytaine.epic18x_conditions_225
    )
    ref2 = replace(ref, capytaine=replace(ref.capytaine, epic18x_conditions_225=better_18x))
    metrics = _baseline_like(ref2)
    gm = ref2.capytaine.gm_t_cm
    at_baseline = _seak_rows_toward_floor(ref2, 0.0)
    assert evaluate_gates(metrics, ref2, seakeeping=_seak_metrics(at_baseline, gm_t_cm=gm)).passed is True
    worse = _seak_rows_toward_floor(ref2, 0.0, headings=_AHEAD_HEADINGS, seat_acc_rms_ms2=1.01)
    rep = evaluate_gates(metrics, ref2, seakeeping=_seak_metrics(worse, gm_t_cm=gm))
    assert rep.passed is False and all(v.gate.startswith("seak_seat_acc_") for v in rep.violations)


def test_seakeeping_gate_fails_closed_without_18x_rows_and_allows_for_the_quick_grid() -> None:
    ref = load_baseline_reference()
    metrics = _baseline_like(ref)
    gm = ref.capytaine.gm_t_cm

    no_floor = replace(ref, capytaine=replace(ref.capytaine, epic18x_conditions_225=()))
    rep = evaluate_gates(metrics, no_floor, seakeeping=_seak_metrics(_seak_rows_toward_floor(ref, 0.0), gm_t_cm=gm))
    assert rep.passed is False
    assert {v.gate[:19] for v in rep.violations} == {"seak_floor_missing_"}

    def quick_rows(scale: float) -> list[dict]:
        return [
            r
            for r in _seak_rows_toward_floor(ref, 1.0, seat_acc_rms_ms2=scale, bow_dips_per_min=scale)
            if r["speed_kn"] == 4.5 and r["heading"] == "head"
        ]

    # Quick mode solves every second period, so it gets 5 % against rows from the full grid.
    assert evaluate_gates(metrics, ref, seakeeping=_seak_metrics(quick_rows(1.04), quick=True, gm_t_cm=gm)).passed is True
    assert evaluate_gates(metrics, ref, seakeeping=_seak_metrics(quick_rows(1.06), quick=True, gm_t_cm=gm)).passed is False


def test_following_and_stern_quarter_seas_are_gated_on_the_epic_18x_row_alone() -> None:
    """Owner, 2026-11-27: pitch, bow burying and seat motion no worse than the Epic; roll up to 20 % worse."""
    ref = load_baseline_reference()
    metrics = _baseline_like(ref)
    gm = ref.capytaine.gm_t_cm
    base = _seak_keyed(ref.capytaine.target_conditions_225)
    floor = _seak_keyed(ref.capytaine.epic18x_conditions_225)
    astern = [k for k in base if k[1] in SEAKEEPING_ASTERN_HEADINGS]
    assert len(astern) == 8 and {k[1] for k in astern} == {"following", "stern_quarter"}

    def gate(rows: list[dict], ref_: BaselineReference = ref):
        return evaluate_gates(metrics, ref_, seakeeping=_seak_metrics(rows, gm_t_cm=gm))

    # v0.2.3's own rows pass, and so does a hull exactly at every limit.
    for frac in (0.0, 1.0):
        rep = gate(_seak_rows_toward_floor(ref, frac))
        assert rep.passed is True, (frac, rep.violations)
    at_base = gate(_seak_rows_toward_floor(ref, 0.0))
    for prefix in ("seak_pitch_", "seak_bow_dips_", "seak_seat_acc_", "seak_roll_"):
        assert sum(k.startswith(prefix) and ("following" in k or "stern_quarter" in k) for k in at_base.margins) == 8
    # Pitch and roll are not gated in head and bow-quarter seas.
    assert not any(k.startswith(("seak_pitch_", "seak_roll_")) and ("head" in k or "bow_quarter" in k) for k in at_base.margins)

    # One per cent past the Epic 18X row fails on that measure, in all eight conditions.
    for fld, prefix in (("pitch_sig_deg", "seak_pitch_"), ("seat_acc_rms_ms2", "seak_seat_acc_")):
        rep = gate(_seak_rows_toward_floor(ref, 1.0, headings=SEAKEEPING_ASTERN_HEADINGS, **{fld: 1.01}))
        assert rep.passed is False
        assert {v.gate[: len(prefix)] for v in rep.violations} == {prefix} and len(rep.violations) == 8

    # Roll: 1.2 times the Epic row passes (frac 1.0 above); one per cent more fails in the four
    # stern-quarter conditions. Following seas have no roll in either boat.
    rep = gate(_seak_rows_toward_floor(ref, 1.0, headings=SEAKEEPING_ASTERN_HEADINGS, roll_sig_deg=1.01))
    assert rep.passed is False and len(rep.violations) == 4
    assert all(v.gate.startswith("seak_roll_") and "stern_quarter" in v.gate for v in rep.violations)
    assert sorted(float(v.limit) for v in rep.violations) == sorted(
        1.2 * floor[k].roll_sig_deg for k in astern if k[1] == "stern_quarter"
    )

    # Bow dips: the Epic 18X has none at 5.7 kn following, so any there fails.
    rows = _seak_rows_toward_floor(ref, 0.0)
    for r in rows:
        if r["heading"] == "following" and r["speed_kn"] == 5.7:
            assert floor[(r["speed_kn"], r["heading"], r["sea"])].bow_dips_per_min == 0.0
            r["bow_dips_per_min"] = 0.02
    rep = gate(rows)
    assert {v.gate for v in rep.violations} == {
        "seak_bow_dips_5.7kn_following_bay",
        "seak_bow_dips_5.7kn_following_steep",
    }

    # The limit is the Epic row, not the worse of the two boats: against a better Epic 18X the
    # baseline's own rows fail. (In head and bow-quarter seas they would still pass.)
    better_18x = tuple(
        replace(r, pitch_sig_deg=0.9 * r.pitch_sig_deg) if r.heading in SEAKEEPING_ASTERN_HEADINGS else r
        for r in ref.capytaine.epic18x_conditions_225
    )
    ref2 = replace(ref, capytaine=replace(ref.capytaine, epic18x_conditions_225=better_18x))
    rep = gate(_seak_rows_toward_floor(ref, 0.0), ref2)
    assert rep.passed is False
    assert {v.gate[:11] for v in rep.violations} == {"seak_pitch_"} and len(rep.violations) == 8

    # A following-sea row without a pitch reading fails closed.
    rows = _seak_rows_toward_floor(ref, 0.0)
    del rows[[r["heading"] for r in rows].index("following")]["pitch_sig_deg"]
    rep = gate(rows)
    assert rep.passed is False and [v.gate[:11] for v in rep.violations] == ["seak_pitch_"]

    # Forefoot emergence is reported, not gated: v0.2.3 exceeds the Epic 18X row in all eight
    # conditions, and whether to gate it is the owner's decision.
    assert all(base[k].forefoot_out_per_min > floor[k].forefoot_out_per_min for k in astern)
    rows = _seak_rows_toward_floor(ref, 0.0)
    for r in rows:
        r["forefoot_out_per_min"] = 100.0
    assert gate(rows).passed is True

    # The Pareto seakeeping objective still reads head and bow-quarter seas only.
    assert math.isclose(gate(_seak_rows_toward_floor(ref, 0.0)).objectives["seakeeping_gain"], 0.0, abs_tol=1e-12)
    astern_better = _seak_rows_toward_floor(ref, 0.0, headings=SEAKEEPING_ASTERN_HEADINGS, seat_acc_rms_ms2=0.5)
    assert math.isclose(gate(astern_better).objectives["seakeeping_gain"], 0.0, abs_tol=1e-12)
