"""Irregular-frequency points left by the BEM lid are found, repaired or refused."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))
cap_mod = pytest.importorskip("run_capytaine_seakeeping")


def _smooth_case(n: int = 30) -> dict:
    """A smooth 6-DOF case shaped like a head-sea forward-speed solve (periods 1.05 s upward)."""
    periods = np.linspace(1.05, 9.0, n)
    omega = np.sort(2.0 * np.pi / periods)
    A = np.zeros((n, 6, 6))
    B = np.zeros((n, 6, 6))
    F = np.zeros((n, 6), dtype=complex)
    for d, (a0, b0, f0) in {2: (190.0, 900.0, 4000.0), 3: (2.0, 1.0, 0.0), 4: (220.0, 1300.0, 9000.0)}.items():
        A[:, d, d] = a0 * (1.0 + 0.3 / omega)
        B[:, d, d] = b0 * (1.0 + 0.1 * omega)
        F[:, d] = f0 * np.exp(-0.25 * omega) * np.exp(1j * 0.4 * omega)
    return {"omega": omega, "omega_e": omega + omega**2 * 2.93 / 9.81, "A": A, "B": B, "F": F, "beta": 180.0}


def _spike(case: dict, i: int) -> dict:
    """Put the measured signature at index i: excitation 37x, negative heave added mass and damping."""
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in case.items()}
    out["F"][i, 2] *= 37.0
    out["F"][i, 4] *= 13.0
    out["A"][i, 2, 2] = -136.0
    out["B"][i, 2, 2] = -252.0
    return out


def test_clean_case_has_no_flags_and_is_returned_unchanged() -> None:
    case = _smooth_case()
    repaired, bad = cap_mod.repair_irregular_points(case)
    assert bad == []
    assert repaired is case


@pytest.mark.parametrize("index", [0, 7, 29])
def test_single_spike_is_found_and_bridged(index: int) -> None:
    clean = _smooth_case()
    case = _spike(clean, index)
    repaired, bad = cap_mod.repair_irregular_points(case)
    assert bad == [index]
    # The input is not modified.
    assert case["A"][index, 2, 2] == -136.0
    if 0 < index < 29:
        # Interior points are bridged close to the smooth value.
        assert repaired["A"][index, 2, 2] == pytest.approx(clean["A"][index, 2, 2], rel=0.02)
        assert repaired["B"][index, 2, 2] == pytest.approx(clean["B"][index, 2, 2], rel=0.02)
        assert abs(repaired["F"][index, 2]) == pytest.approx(abs(clean["F"][index, 2]), rel=0.02)
    else:
        # End points take the nearest clean value; nothing is extrapolated.
        nearest = 1 if index == 0 else 28
        assert repaired["A"][index, 2, 2] == clean["A"][nearest, 2, 2]
        assert repaired["B"][index, 2, 2] == clean["B"][nearest, 2, 2]
        assert repaired["F"][index, 2] == clean["F"][nearest, 2]
    others = [i for i in range(30) if i != index]
    np.testing.assert_array_equal(repaired["F"][others], clean["F"][others])
    np.testing.assert_array_equal(repaired["A"][others], clean["A"][others])
    assert not np.iscomplexobj(repaired["A"]) and np.iscomplexobj(repaired["F"])
    assert cap_mod.find_irregular_points(repaired) == []


def test_excitation_jump_alone_is_flagged() -> None:
    case = _smooth_case()
    case["F"][12, 2] *= 6.0
    assert cap_mod.find_irregular_points(case) == [12]


def test_excitation_dip_is_not_flagged() -> None:
    # Heave excitation passes near zero at some periods (77.8 N/m between 1500 and 2160 on the hull).
    case = _smooth_case()
    case["F"][12, 2] *= 0.04
    assert cap_mod.find_irregular_points(case) == []


def test_adjacent_or_many_spikes_are_refused() -> None:
    clean = _smooth_case()
    with pytest.raises(ValueError, match="too wide to repair"):
        cap_mod.repair_irregular_points(_spike(_spike(clean, 5), 6))
    many = clean
    for i in (3, 8, 13, 18):
        many = _spike(many, i)
    with pytest.raises(ValueError, match="too wide to repair"):
        cap_mod.repair_irregular_points(many)


def test_measured_spike_from_failing_hull_is_flagged() -> None:
    """Heave values solved for campaign hull iter075_cand007 at 5.7 kn head seas, default lid."""
    periods = np.array([2.25, 2.10, 1.95, 1.80, 1.65, 1.50, 1.35, 1.20, 1.05])
    f3 = np.array([8220.0, 6500.0, 4450.0, 2170.0, 46.6, 1460.0, 1170.0, 14700.0, 125.0])
    f5 = np.array([12900.0, 12800.0, 11900.0, 9920.0, 6430.0, 1680.0, 2340.0, 23300.0, 1730.0])
    a33 = np.array([247.0, 229.0, 212.0, 199.0, 189.0, 183.0, 180.0, -136.0, 193.0])
    b33 = np.array([1460.0, 1450.0, 1420.0, 1370.0, 1290.0, 1180.0, 1020.0, -252.0, 727.0])
    n = len(periods)
    A, B, F = np.zeros((n, 6, 6)), np.zeros((n, 6, 6)), np.zeros((n, 6), dtype=complex)
    A[:, 2, 2], B[:, 2, 2], F[:, 2], F[:, 4] = a33, b33, f3, f5
    A[:, 4, 4], B[:, 4, 4] = 250.0, 1500.0
    omega = 2.0 * np.pi / periods
    case = {"omega": omega, "omega_e": omega + omega**2 * 2.93 / 9.81, "A": A, "B": B, "F": F, "beta": 180.0}
    repaired, bad = cap_mod.repair_irregular_points(case)
    assert bad == [7]
    assert abs(repaired["F"][7, 2]) < 1200.0
    assert repaired["A"][7, 2, 2] > 180.0 and repaired["B"][7, 2, 2] > 700.0
