"""Unit tests for rsi.genome design-space parameterization and Hull adapter."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from rsi.baseline import load_baseline_reference
from rsi.genome import DesignSpace, hull_to_vector, vector_to_hull


def test_design_space_default_and_round_trip_identity() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()

    assert len(space.params) >= 26
    assert {p.group for p in space.params} == {
        "waterline",
        "draft",
        "rocker",
        "section",
        "topsides",
        "fairing",
    }

    vec = space.baseline_vector()
    extracted = hull_to_vector(ref.hull, space)
    assert extracted == vec

    rebuilt = vector_to_hull(vec, ref.hull, space)
    assert rebuilt.model_dump() == ref.hull.model_dump()
    assert rebuilt.design_hash() == ref.hull.design_hash()
    assert rebuilt.record_hash() == ref.hull.record_hash()


def test_design_space_bounds_clamping() -> None:
    space = DesignSpace.default()
    vec = space.baseline_vector()
    too_low = [v - 10.0 for v in vec]
    too_high = [v + 10.0 for v in vec]

    clamped_low = space.clamp(too_low)
    clamped_high = space.clamp(too_high)
    assert clamped_low == [p.lower for p in space.params]
    assert clamped_high == [p.upper for p in space.params]


def test_single_parameter_bounds_validity_and_recovery() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()
    base_vec = space.baseline_vector()

    for idx, spec in enumerate(space.params):
        for bound in (spec.lower, spec.upper):
            vec = list(base_vec)
            vec[idx] = bound
            candidate = vector_to_hull(vec, ref.hull, space)
            assert candidate.bow_rake == 1.0
            assert candidate.stern_rake == 1.0
            recovered = hull_to_vector(candidate, space)
            assert abs(recovered[idx] - bound) < 1e-12


def test_vector_to_hull_rejects_invalid_inputs() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()
    base_vec = space.baseline_vector()

    with pytest.raises(ValueError):
        vector_to_hull(base_vec[:-1], ref.hull, space)

    # Non-finite parameter rejected by Pydantic finite_number check
    nan_vec = list(base_vec)
    nan_vec[idx["shoulder_round_m"]] = float("nan")
    with pytest.raises(ValidationError):
        vector_to_hull(nan_vec, ref.hull, space)

    # Overlapping ramp stations (forward_xi >= aft_xi) rejected
    bad_xi = list(base_vec)
    bad_xi[idx["sec_side_fwd_xi"]] = 0.50
    bad_xi[idx["sec_side_aft_xi"]] = 0.10
    with pytest.raises(ValidationError):
        vector_to_hull(bad_xi, ref.hull, space)

    # Sheer plan exceeding 1.5x waterline half-breadth rejected
    bad_sheer = list(base_vec)
    bad_sheer[idx["sheer_max_hb_m"]] = 0.45
    with pytest.raises(ValidationError):
        vector_to_hull(bad_sheer, ref.hull, space)


def test_sheer_height_frac_four_corners_valid() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()
    base_vec = space.baseline_vector()
    fwd_spec = space.params[idx["sheer_h_fwd_frac"]]
    aft_spec = space.params[idx["sheer_h_aft_frac"]]

    for fwd_val in (fwd_spec.lower, fwd_spec.upper):
        for aft_val in (aft_spec.lower, aft_spec.upper):
            vec = list(base_vec)
            vec[idx["sheer_h_fwd_frac"]] = fwd_val
            vec[idx["sheer_h_aft_frac"]] = aft_val
            candidate = vector_to_hull(vec, ref.hull, space)
            recovered = hull_to_vector(candidate, space)
            assert abs(recovered[idx["sheer_h_fwd_frac"]] - fwd_val) < 1e-12
            assert abs(recovered[idx["sheer_h_aft_frac"]] - aft_val) < 1e-12


def test_zero_deck_freeboard_endpoint_raises_value_error() -> None:
    ref = load_baseline_reference()
    space = DesignSpace.default()
    idx = space.index_by_name()
    base_vec = space.baseline_vector()

    zero_fwd_vec = list(base_vec)
    zero_fwd_vec[idx["deck_fb_fwd_scale"]] = 0.0
    zero_fwd_hull = vector_to_hull(zero_fwd_vec, ref.hull, space)

    with pytest.raises(ValueError, match="deck_freeboard"):
        hull_to_vector(zero_fwd_hull, space)

    with pytest.raises(ValueError, match="deck_freeboard"):
        vector_to_hull(base_vec, zero_fwd_hull, space)
