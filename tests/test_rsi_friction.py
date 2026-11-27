"""Friction charged by where the wetted area is (rsi.evaluator.local_friction_ratio)."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from rsi.baseline import DEFAULT_REPO_ROOT, FRICTION_REFERENCE_FILE, load_baseline_reference
from rsi.evaluator import RESISTANCE_SPEEDS_KN, evaluate_tier0_tier1, local_friction_ratio
from rsi.genome import DesignSpace, hull_to_vector, vector_to_hull

NU = 1.19e-6
U = 5.5 * 0.514444


def _strip(x0: float, x1: float, width: float, n: int = 400) -> np.ndarray:
    """A flat strip of skin from x0 to x1, as triangles 1/n of its length long."""
    xs = np.linspace(x0, x1, n + 1)
    tris = []
    for a, b in zip(xs[:-1], xs[1:]):
        tris += [((a, 0, -0.05), (b, 0, -0.05), (b, width, -0.05)), ((a, 0, -0.05), (b, width, -0.05), (a, width, -0.05))]
    return np.array(tris, dtype=float)


def test_a_plate_of_constant_width_gets_the_plain_ittc_value() -> None:
    plate = _strip(-2.7, 2.7, 0.4, n=2000)
    for speed_kn in (3.5, 5.5, 7.75):
        ratio = local_friction_ratio(plate, speed_kn * 0.514444, NU, x_fwd=-2.7, length_m=5.4)
        assert ratio == pytest.approx(1.0, abs=5e-4), speed_kn


def test_skin_near_the_bow_costs_more_than_skin_near_the_stern() -> None:
    plate = _strip(-2.7, 2.7, 0.4)

    def cost_of_patch(x0: float) -> float:
        """Friction of a 0.2 m long patch as a multiple of the same area at the mean coefficient."""
        patch = _strip(x0, x0 + 0.2, 0.4, n=40)
        both = np.concatenate([plate, patch])
        area_p = 0.2 * 0.4
        area = 5.4 * 0.4
        with_patch = local_friction_ratio(both, U, NU, x_fwd=-2.7, length_m=5.4) * (area + area_p)
        without = local_friction_ratio(plate, U, NU, x_fwd=-2.7, length_m=5.4) * area
        return (with_patch - without) / area_p

    bow, middle, stern = cost_of_patch(-2.6), cost_of_patch(-0.1), cost_of_patch(2.5)
    assert bow > 1.25 and stern < 0.90, (bow, middle, stern)
    assert bow > middle > stern


@pytest.fixture(scope="module")
def baseline_and_deeper_forefoot(tmp_path_factory: pytest.TempPathFactory) -> dict:
    ref = load_baseline_reference()
    plain = replace(ref, friction_ratio=None)  # one coefficient for the whole area, as before
    space = DesignSpace.default()
    vec = hull_to_vector(ref.hull, space)
    names = [spec.name for spec in space.params]
    vec[names.index("rocker_bow_end_m")] -= 0.006  # keel 6 mm deeper at the bow end
    cand = vector_to_hull(vec, ref.hull, space)
    work = tmp_path_factory.mktemp("friction")
    return {
        "ref": ref,
        "base_plain": evaluate_tier0_tier1(ref.hull, work / "b0", plain),
        "base": evaluate_tier0_tier1(ref.hull, work / "b1", ref),
        "cand_plain": evaluate_tier0_tier1(cand, work / "c0", plain),
        "cand": evaluate_tier0_tier1(cand, work / "c1", ref),
    }


def test_the_friction_reference_file_is_the_baselines_own_ratio(baseline_and_deeper_forefoot: dict) -> None:
    ref, base = baseline_and_deeper_forefoot["ref"], baseline_and_deeper_forefoot["base_plain"]
    assert base.valid, base.error
    stored = json.loads((DEFAULT_REPO_ROOT / "data" / FRICTION_REFERENCE_FILE).read_text())
    assert stored["design_hash"] == ref.design_hash
    assert stored["speeds_kn"] == [float(v) for v in RESISTANCE_SPEEDS_KN]
    assert sorted(ref.friction_ratio) == sorted(base.friction_ratio) == [225, 230, 265]
    for lb, vals in base.friction_ratio.items():
        assert vals == pytest.approx(ref.friction_ratio[lb], rel=1e-10), (
            "data/rsi-friction-reference.json is stale; run scripts/build_rsi_friction_reference.py"
        )
        assert 0.94 < min(vals) and max(vals) < 0.97, "a kayak carries less of its area at the ends than a plate"


def test_a_reference_that_does_not_fit_is_refused_not_ignored(tmp_path) -> None:
    ref = load_baseline_reference()
    no_265 = replace(ref, friction_ratio={lb: ref.friction_ratio[lb] for lb in (225, 230)})
    out = evaluate_tier0_tier1(ref.hull, tmp_path / "a", no_265)
    assert out.valid is False and "265 lb" in (out.error or "")
    other_speeds = replace(ref, friction_speeds_kn=tuple(v + 0.25 for v in ref.friction_speeds_kn))
    out = evaluate_tier0_tier1(ref.hull, tmp_path / "b", other_speeds)
    assert out.valid is False and "RESISTANCE_SPEEDS_KN" in (out.error or "")


def test_the_baseline_keeps_its_published_speeds(baseline_and_deeper_forefoot: dict) -> None:
    ref, base, plain = (baseline_and_deeper_forefoot[k] for k in ("ref", "base", "base_plain"))
    assert base.valid, base.error
    assert base.kn_225 == {40: 4.47, 60: 5.08, 90: 5.67, 120: 6.17}
    for p, v in base.kn_225_exact.items():
        assert v == pytest.approx(plain.kn_225_exact[p], abs=1e-9)
        assert v == pytest.approx(ref.kn_225_exact[p], abs=1e-6)
    for p, v in base.kn_265_exact.items():
        assert v == pytest.approx(plain.kn_265_exact[p], abs=1e-9)


def test_a_deeper_forefoot_pays_for_wetted_area_at_the_bow(baseline_and_deeper_forefoot: dict) -> None:
    d = baseline_and_deeper_forefoot
    ref, cand, cand_plain = d["ref"], d["cand"], d["cand_plain"]
    assert cand.valid and cand_plain.valid, (cand.error, cand_plain.error)
    # The change adds wetted area, and adds it forward.
    assert cand.loads[225].wet_m2 >= d["base"].loads[225].wet_m2
    i55 = int(np.argmin(np.abs(RESISTANCE_SPEEDS_KN - 5.5)))
    shift = cand.friction_ratio[225][i55] / ref.friction_ratio[225][i55] - 1.0
    assert shift > 3e-4, f"area moved towards the bow must raise the ratio, got {shift:+.2e}"
    # So at 40 W, where friction is nearly all of the drag, it is slower than one coefficient says.
    assert cand.kn_225_exact[40] < cand_plain.kn_225_exact[40] - 1e-4
    assert cand.kn_265_exact[40] < cand_plain.kn_265_exact[40] - 1e-4
    # The reported ratio does not depend on which reference was used.
    assert cand.friction_ratio[225] == pytest.approx(cand_plain.friction_ratio[225], rel=1e-12)
