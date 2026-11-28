"""Unit tests for rsi.baseline reference loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi.baseline import load_baseline_reference


def test_load_baseline_reference_exact_values() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ref = load_baseline_reference(repo_root)

    assert ref.design_hash == (
        "distribution_v2_v2:a753b604c5348cf250a98ebcfba2b0c193ed5f166afdf34b4ecac75e0d039f2d"
    )
    assert ref.record_hash == (
        "bc952fcbef39aaf89618abb57323c1e051c124dfea3f5042b6a5dd2c741f1a5a"
    )
    assert (
        ref.outer_stl_sha256
        == "fcbf0bdbe06a7bff2d8d8e7d44469cbfac9311e74857d6f95099245e37366d2f"
    )
    assert ref.l_oa_m == 5.47
    assert ref.b_oa_cm == 56.26
    assert abs(ref.sr_4in_exact - 10.875770424030575) < 1e-12
    assert ref.sr_4in == 10.88
    assert ref.usca_4in_pct == 9.15

    uncal = load_baseline_reference(repo_root, with_panel_wave_reference=False)
    assert uncal.kn_225 == {40: 4.47, 60: 5.08, 90: 5.67, 120: 6.17}
    assert ref.panel_wave is not None
    assert set(ref.kn_225) == {40, 60, 90, 120}
    assert all(isinstance(k, int) for k in ref.kn_225)
    assert ref.kn_225_exact is not None
    for pw in (40, 60, 90, 120):
        assert round(ref.kn_225_exact[pw], 2) == ref.kn_225[pw]
    assert uncal.kn_265 == {40: 4.32, 60: 4.97, 90: 5.53, 120: 5.98}
    assert set(ref.kn_265) == {40, 60, 90, 120}
    assert ref.kn_265_exact is not None
    for pw in (40, 60, 90, 120):
        assert round(ref.kn_265_exact[pw], 2) == ref.kn_265[pw]
    assert ref.loads[225].lcg_m is not None
    assert abs(ref.shell_area_m2 - 5.80659) < 1e-5

    assert set(ref.loads.keys()) == {225, 265, 300}
    assert ref.loads[225].gm_cm == 4.11
    assert ref.loads[265].gm_cm == 3.14
    assert ref.loads[300].gm_cm == 2.43
    assert abs(ref.loads[225].gz_mm[0] - 6.543804143209861) < 1e-12
    assert abs(ref.loads[265].gz_mm[1] - 7.337712607391966) < 1e-12
    assert abs(ref.loads[300].gz_mm[2] - 1.413834066753522) < 1e-12

    assert ref.transition_dB_per_20cm == (3.4, 3.1, 3.0, 2.7, 2.1, 1.9, 1.5, 1.0)
    assert len(ref.catch_widths_mm) == 6
    assert abs(ref.catch_widths_mm[0] - 527.5320194709104) < 1e-9

    assert ref.fairness.sha256 == ref.outer_stl_sha256
    assert ref.fairness.cuts["Buttocks 0.075"].peak_turn_change_deg < 3.50
    assert ref.fairness.cuts["Waterlines 0.1016"].max_signed_turn_forward_deg < 0.0

    assert ref.capytaine.outer_stl_sha256 == ref.outer_stl_sha256
    assert ref.capytaine.roll_period_s == 2.07
    assert ref.capytaine.gm_t_cm == 4.09
    assert ref.capytaine.gm_t_cm_265_baja == 3.12
    # Head, bow-quarter, following and stern-quarter seas at 4.5 and 5.7 kn in two seas.
    assert len(ref.capytaine.target_conditions_225) == 16
    assert {r.heading for r in ref.capytaine.target_conditions_225} == {
        "head",
        "bow_quarter",
        "following",
        "stern_quarter",
    }
    # The seakeeping floor is the Epic 18X in the same conditions, and both sets of rows must come
    # from a body with irregular-frequency removal (the released dataset has none).
    assert [(r.speed_kn, r.heading, r.sea) for r in ref.capytaine.epic18x_conditions_225] == [
        (r.speed_kn, r.heading, r.sea) for r in ref.capytaine.target_conditions_225
    ]
    assert ref.capytaine.info_225["lid_panels"] > 0

    assert ref.openfoam.outer_stl_sha256 == ref.outer_stl_sha256
    assert ref.openfoam.speed_5_5kn.drag_total_N == 20.503
    assert ref.openfoam.speed_6_2kn.drag_total_N == 25.859


def test_baseline_reference_default_root_and_json_serializable() -> None:
    ref = load_baseline_reference()
    payload = ref.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["design_hash"] == ref.design_hash
    assert decoded["sr_4in"] == 10.88
    assert decoded["kn_225"] == {str(k): v for k, v in ref.kn_225.items()}
    assert decoded["panel_wave"] is not None


def test_a_seakeeping_reference_without_a_gated_row_is_refused(tmp_path) -> None:
    """Both boats need a row for every gated condition; a missing one must not pass silently."""
    import shutil

    from rsi.baseline import DEFAULT_REPO_ROOT, SEAKEEPING_REFERENCE_FILE

    for rel in (
        "hull/kayak-10-9.hull.json",
        "data/kayak-10-9-specs.json",
        "data/rsi-friction-reference.json",
        "data/openfoam-cfd-metrics.json",
        "docs/surface-fairing/controls.json",
        "docs/surface-fairing/fairness.json",
        "rsi/panel_wave_reference.json",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEFAULT_REPO_ROOT / rel, tmp_path / rel)
    raw = json.loads((DEFAULT_REPO_ROOT / "data" / SEAKEEPING_REFERENCE_FILE).read_text())
    raw["epic18x_225"]["rows"] = [
        r for r in raw["epic18x_225"]["rows"] if not (r["heading"] == "stern_quarter" and r["speed_kn"] == 5.7)
    ]
    (tmp_path / "data" / SEAKEEPING_REFERENCE_FILE).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="epic18x_225.*stern_quarter"):
        load_baseline_reference(tmp_path)
