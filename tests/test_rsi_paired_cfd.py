"""Paired warm-start CFD (rsi/paired_cfd.py): the comparison itself, without running OpenFOAM."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pytest

from rsi.baseline import load_baseline_reference
from rsi.evaluator import evaluate_tier0_tier1
from rsi.fidelity import CfdMetrics
from rsi.gates import (
    CFD_PAIRED_DRAG_TOL_N,
    CFD_PAIRED_DRIFT_TOL_N,
    CFD_PAIRED_MODE,
    evaluate_gates,
)
from rsi.paired_cfd import (
    COLD_MODE,
    PAIRED_MODE,
    PairedCfdSettings,
    apply_settle_treatment,
    expected_result_mode,
    paired_force_delta,
    prepare_pair_case,
    read_force_history,
    reference_key,
    settle_drift,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))

from openfoam_dicts import write_interfoam_case_files  # noqa: E402


def _history(steps: np.ndarray, total: np.ndarray, pressure_share: float = 0.2) -> dict[str, np.ndarray]:
    total = np.asarray(total, dtype=float)
    return {
        "step": np.asarray(steps, dtype=int),
        "pressure": pressure_share * total,
        "viscous": (1.0 - pressure_share) * total,
        "total": total,
    }


def _write_force_dat(path: Path, steps: np.ndarray, half_total_x: np.ndarray) -> None:
    """A force.dat as the forces function object writes it for the half model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Time (total_x total_y total_z) (pressure_x pressure_y pressure_z) (viscous_x viscous_y viscous_z)"]
    for s, f in zip(steps, half_total_x):
        lines.append(f"{s}\t({f:.9g} 0 400) ({0.2 * f:.9g} 0 400) ({0.8 * f:.9g} 0 0)")
    path.write_text("\n".join(lines) + "\n")


def test_difference_is_taken_step_for_step_over_the_tail() -> None:
    steps = np.arange(2001, 2601)
    drift = 18.0 - 0.002 * (steps - 2000)  # the baseline is still drifting by 1.2 N over the run
    base = _history(steps, drift + 0.05 * np.sin(steps / 7.0))
    cand = _history(steps, drift + 0.05 * np.sin(steps / 7.0) - 0.12)
    out = paired_force_delta(base, cand, tail_steps=200)
    assert out["drag_delta_N"] == pytest.approx(-0.12, abs=1e-12), "drift and slosh common to both must cancel"
    assert out["drag_delta_pressure_N"] == pytest.approx(-0.024, abs=1e-12)
    assert out["drag_delta_viscous_N"] == pytest.approx(-0.096, abs=1e-12)
    assert out["drag_delta_drift_N"] == pytest.approx(0.0, abs=1e-12)
    assert (out["first_step"], out["last_step"]) == (2401, 2600)
    assert out["baseline_drag_N"] == pytest.approx(float(base["total"][-200:].mean()))
    assert out["drag_total_N"] == pytest.approx(out["baseline_drag_N"] - 0.12)


def test_a_difference_that_is_still_moving_is_reported_as_drift() -> None:
    steps = np.arange(1, 401)
    base = _history(steps, np.full(400, 20.0))
    cand = _history(steps, 20.0 + 0.001 * steps)  # the difference grows by 0.1 N per 100 steps
    out = paired_force_delta(base, cand, tail_steps=200)
    assert out["drag_delta_drift_N"] == pytest.approx(0.1, abs=1e-9)
    assert out["drag_delta_N"] == pytest.approx(0.3005, abs=1e-9)


def test_histories_that_do_not_cover_the_same_steps_are_refused() -> None:
    base = _history(np.arange(1, 301), np.full(300, 20.0))
    with pytest.raises(ValueError, match="share"):
        paired_force_delta(base, _history(np.arange(250, 301), np.full(51, 20.0)), tail_steps=200)
    with pytest.raises(ValueError, match="same step"):
        paired_force_delta(base, _history(np.arange(1, 281), np.full(280, 20.0)), tail_steps=200)


def test_force_history_is_read_as_full_hull_drag(tmp_path: Path) -> None:
    f = tmp_path / "postProcessing/forces/2000/force.dat"
    _write_force_dat(f, np.arange(2001, 2011), np.linspace(9.0, 9.9, 10))
    h = read_force_history(f)
    assert h["step"].tolist() == list(range(2001, 2011))
    assert h["total"] == pytest.approx(2.0 * np.linspace(9.0, 9.9, 10))
    assert h["pressure"] == pytest.approx(0.4 * np.linspace(9.0, 9.9, 10))
    (tmp_path / "empty.dat").write_text("# only a header\n")
    with pytest.raises(ValueError, match="empty"):
        read_force_history(tmp_path / "empty.dat")


_OUTLETS = {
    "U": "type            outletPhaseMeanVelocity;\n        Umean           2.829442;\n"
    "        alpha           alpha.water;\n        value           uniform (2.829442 0 0);",
    "p_rgh": "type            zeroGradient;",
    "alpha.water": "type            variableHeightFlowRate;\n        lowerBound      0;\n"
    "        upperBound      1;\n        value           uniform 1;",
}


def _field_text(name: str, step: int) -> str:
    """A field file as interFoam writes it for one processor, with the case's own outlet."""
    kind = "volVectorField" if name == "U" else "volScalarField"
    wall = "type            movingWallVelocity;\n        value           uniform (0 0 0);" if name == "U" else "type            zeroGradient;"
    return (
        f"FoamFile\n{{\n    class       {kind};\n    location    \"{step}\";\n    object      {name};\n}}\n\n"
        "internalField   uniform 0;\n\nboundaryField\n{\n"
        "    inlet\n    {\n        type            fixedValue;\n        value           uniform 0;\n    }\n"
        f"    outlet\n    {{\n        {_OUTLETS[name]}\n    }}\n"
        "    midPlane\n    {\n        type            symmetryPlane;\n    }\n"
        f"    hull\n    {{\n        {wall}\n    }}\n"
        "    procBoundary0to1\n    {\n        type            processor;\n        value           uniform 0;\n    }\n}\n"
    )


def _write_state(proc_dir: Path, step: int) -> None:
    d = proc_dir / str(step)
    d.mkdir()
    for name in _OUTLETS:
        (d / name).write_text(_field_text(name, step))


def _finished_run(root: Path, step: int = 2000, n_proc: int = 2) -> Path:
    """The files a finished decomposed interFoam run leaves behind, as far as a pair case needs."""
    run = root / "run_candidate_5p5kn"
    (run / "constant").mkdir(parents=True)
    (run / "system").mkdir()
    write_interfoam_case_files(run, 2.829, [0.04, 0.0, 0.16], end_steps=step, np_ranks=n_proc)
    for i in range(n_proc):
        mesh = run / f"processor{i}" / "constant" / "polyMesh"
        mesh.mkdir(parents=True)
        (mesh / "points").write_text("points\n")
        for t in (0, step // 2, step):
            _write_state(run / f"processor{i}", t)
    return run


def test_pair_case_continues_the_finished_run_and_copies_only_its_last_state(tmp_path: Path) -> None:
    run = _finished_run(tmp_path)
    dst = tmp_path / "pair"
    prepare_pair_case(run, dst, 2000, 2600)
    ctrl = (dst / "system" / "controlDict").read_text()
    assert "startFrom latestTime;" in ctrl and "startFrom startTime" not in ctrl
    assert "stopAt endTime; endTime 2600;" in ctrl
    assert "writeControl timeStep; writeInterval 2600; purgeWrite" in ctrl, "fields are written at the end step"
    assert "writeControl timeStep; writeInterval 1; CofR" in ctrl, "forces are still written every step"
    for i in range(2):
        assert sorted(p.name for p in (dst / f"processor{i}").iterdir()) == ["2000", "constant"]
        assert (dst / f"processor{i}" / "2000" / "U").read_text() == _field_text("U", 2000)
    for name in ("g", "transportProperties", "turbulenceProperties"):
        assert (dst / "constant" / name).read_text() == (run / "constant" / name).read_text()
    assert (dst / "system" / "fvSolution").read_text() == (run / "system" / "fvSolution").read_text()
    assert "maxCo 10;" in (dst / "system" / "fvSolution").read_text()
    # the source is left as it was
    assert "startFrom startTime;" in (run / "system" / "controlDict").read_text()

    # preparing again replaces what was there; a Courant limit is set only when asked for
    (dst / "stale").write_text("x")
    prepare_pair_case(run, dst, 2000, 2300, max_co=50.0)
    assert not (dst / "stale").exists()
    assert "endTime 2300;" in (dst / "system" / "controlDict").read_text()
    sol = (dst / "system" / "fvSolution").read_text()
    assert "maxCo 50; maxAlphaCo 5;" in sol and "maxCo 10;" not in sol
    assert "maxCo 10;" in (run / "system" / "fvSolution").read_text()


def test_pair_case_refuses_a_missing_state_and_a_moved_reference_mesh(tmp_path: Path) -> None:
    run = _finished_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="time directory 1500"):
        prepare_pair_case(run, tmp_path / "pair", 1500, 2100)
    with pytest.raises(FileNotFoundError, match="no processor directories"):
        prepare_pair_case(tmp_path / "nothing", tmp_path / "pair", 2000, 2600)
    (run / "processor1" / "constant" / "polyMesh" / "points.orig").write_text("unmoved\n")
    with pytest.raises(ValueError, match="moved mesh"):
        prepare_pair_case(run, tmp_path / "pair", 2000, 2600)


def test_reference_key_changes_with_everything_the_reference_depends_on() -> None:
    bashrc = Path("/usr/lib/openfoam/openfoam2606/etc/bashrc")
    base = PairedCfdSettings()
    key, obj = reference_key("sha256:aaa", base, bashrc)
    assert key.startswith("cfdref_") and obj["baseline_design_hash"] == "sha256:aaa"
    assert reference_key("sha256:aaa", base, bashrc)[0] == key
    others = [
        reference_key("sha256:bbb", base, bashrc)[0],
        reference_key("sha256:aaa", replace(base, ref_steps=base.ref_steps + 100), bashrc)[0],
        reference_key("sha256:aaa", replace(base, pair_steps=base.pair_steps + 100), bashrc)[0],
        reference_key("sha256:aaa", replace(base, speeds_kn=(5.5,)), bashrc)[0],
        reference_key("sha256:aaa", replace(base, np_ranks=4), bashrc)[0],
        reference_key("sha256:aaa", replace(base, max_co=10.0), bashrc)[0],
        reference_key("sha256:aaa", replace(base, settle_steps=base.settle_steps + 100), bashrc)[0],
        reference_key("sha256:aaa", replace(base, settle_steps=0, sponge_frequency=0.0), bashrc)[0],
        reference_key("sha256:aaa", replace(base, sponge_frequency=0.3), bashrc)[0],
        reference_key("sha256:aaa", base, Path("/opt/other/bashrc"))[0],
    ]
    assert len({key, *others}) == 11
    # how the candidate's mesh is moved and how the tail is averaged do not change the reference
    assert reference_key("sha256:aaa", replace(base, tail_steps=100, morph_radius_m=0.3), bashrc)[0] == key


def test_settling_pins_the_outlet_level_and_adds_the_end_sponge(tmp_path: Path) -> None:
    run = _finished_run(tmp_path)
    case = tmp_path / "settle"
    prepare_pair_case(run, case, 2000, 3500, max_co=50.0)
    apply_settle_treatment(case, 2000, 2.829442, 0.15)

    for i in range(2):
        state = case / f"processor{i}" / "2000"
        u, p, alpha = ((state / n).read_text() for n in ("U", "p_rgh", "alpha.water"))
        assert "type            inletOutlet;\n        inletValue      uniform (2.829442 0 0);" in u
        assert "outletPhaseMeanVelocity" not in u and "Umean" not in u and "alpha.water;" not in u
        assert "outlet\n    {\n        type            fixedValue;\n        value           uniform 0;\n    }" in p
        assert "outlet\n    {\n        type            zeroGradient;\n        value           uniform 1;\n    }" in alpha
        assert "variableHeightFlowRate" not in alpha and "lowerBound" not in alpha
        # only the outlet changed
        assert "inlet\n    {\n        type            fixedValue;" in u and "type            movingWallVelocity;" in u
        ref = (state / "URef").read_text()
        assert "internalField uniform (2.829442 0 0);" in ref
        assert "midPlane { type symmetryPlane; }" in ref
        assert "procBoundary0to1 { type processor; value uniform (2.829442 0 0); }" in ref
        assert "outlet { type calculated; value uniform (2.829442 0 0); }" in ref

    opts = (case / "constant" / "fvOptions").read_text()
    assert "type acousticDampingSource;" in opts and "frequency 0.15; w 20; URef URef; U U;" in opts
    assert "origin (2.56 0 0); radius1 7.6; radius2 10.56;" in opts
    ctrl = (case / "system" / "controlDict").read_text()
    assert ctrl.count("readURef { type readFields;") == 1 and "fields (URef); readOnStart true;" in ctrl
    assert "type forces;" in ctrl

    # the reference run itself is left as the cold start wrote it
    assert "outletPhaseMeanVelocity" in (run / "processor0" / "2000" / "U").read_text()
    # a second application finds the outlet already changed and refuses to guess
    with pytest.raises(ValueError, match="not the case's own outlet"):
        apply_settle_treatment(case, 2000, 2.829442, 0.15)


def test_a_case_continued_from_a_settled_state_keeps_the_sponge_and_gets_its_reference_field(tmp_path: Path) -> None:
    run = _finished_run(tmp_path)
    settle = tmp_path / "settle"
    prepare_pair_case(run, settle, 2000, 2400, max_co=50.0)
    apply_settle_treatment(settle, 2000, 2.829442, 0.15)
    for i in range(2):  # interFoam writes its last fields, with the pinned outlet and without URef
        shutil.copytree(settle / f"processor{i}" / "2000", settle / f"processor{i}" / "2400")
        (settle / f"processor{i}" / "2400" / "URef").unlink()

    with pytest.raises(ValueError, match="stream speed"):
        prepare_pair_case(settle, tmp_path / "pair", 2400, 2700, max_co=50.0)
    prepare_pair_case(settle, tmp_path / "pair", 2400, 2700, max_co=50.0, u_mps=2.829442)
    pair = tmp_path / "pair"
    assert (pair / "constant" / "fvOptions").read_text() == (settle / "constant" / "fvOptions").read_text()
    assert (pair / "system" / "controlDict").read_text().count("readURef") == 1
    for i in range(2):
        assert "internalField uniform (2.829442 0 0);" in (pair / f"processor{i}" / "2400" / "URef").read_text()
        assert "type            inletOutlet;" in (pair / f"processor{i}" / "2400" / "U").read_text()


def test_a_pair_case_can_be_continued_again(tmp_path: Path) -> None:
    run = _finished_run(tmp_path)
    first = tmp_path / "settle"
    prepare_pair_case(run, first, 2000, 2400, max_co=50.0)
    for i in range(2):  # what interFoam leaves at the end of that run
        _write_state(first / f"processor{i}", 2400)
    second = tmp_path / "pair"
    prepare_pair_case(first, second, 2400, 2700, max_co=50.0)
    ctrl = (second / "system" / "controlDict").read_text()
    assert "startFrom latestTime;" in ctrl and "endTime 2700;" in ctrl and "writeInterval 2700; purgeWrite" in ctrl
    assert sorted(p.name for p in (second / "processor0").iterdir()) == ["2400", "constant"]


def test_settle_drift_is_the_change_between_the_last_two_windows(tmp_path: Path) -> None:
    steps = np.arange(1, 601)
    f = tmp_path / "force.dat"
    _write_force_dat(f, steps, 9.0 - 0.001 * steps)  # full-hull drag falls 0.2 N per 100 steps
    assert settle_drift(f, window=100) == pytest.approx(-0.2, abs=1e-6)
    assert settle_drift(f, window=1000) == pytest.approx(-0.6, abs=1e-6), "a short history is split in two"


def test_settings_reject_values_that_cannot_work() -> None:
    with pytest.raises(ValueError):
        PairedCfdSettings(speeds_kn=())
    with pytest.raises(ValueError):
        PairedCfdSettings(settle_steps=-1)
    with pytest.raises(ValueError, match="both zero or both positive"):
        PairedCfdSettings(settle_steps=0)  # a sponge that no stage would ever apply
    with pytest.raises(ValueError, match="both zero or both positive"):
        PairedCfdSettings(sponge_frequency=0.0)
    assert PairedCfdSettings(ref_steps=2000, settle_steps=1500, pair_steps=600, tail_steps=200).state_step == 3500
    assert PairedCfdSettings(ref_steps=2000, settle_steps=1500, pair_steps=600, tail_steps=200).end_step == 4100
    with pytest.raises(ValueError):
        PairedCfdSettings(pair_steps=100, tail_steps=200)
    with pytest.raises(ValueError):
        PairedCfdSettings(tail_steps=199)
    assert expected_result_mode("paired", dry_run=False) == PAIRED_MODE == CFD_PAIRED_MODE
    assert expected_result_mode("paired", dry_run=True) == COLD_MODE
    assert expected_result_mode("cold", dry_run=False) == COLD_MODE
    with pytest.raises(ValueError, match="cfd_mode"):
        expected_result_mode("warm", dry_run=False)


def _paired_metrics(delta_55: float, delta_62: float, drift_55: float = 0.0, drift_62: float = 0.0) -> CfdMetrics:
    def sp(v: float, delta: float, drift: float) -> dict[str, float]:
        return {"speed_kn": v, "drag_delta_N": delta, "drag_delta_drift_N": drift, "drag_total_N": 18.0 + delta}

    return CfdMetrics(
        valid=True,
        error=None,
        dry_run=False,
        elapsed_s=1.0,
        bashrc="/usr/lib/openfoam/openfoam2606/etc/bashrc",
        bashrc_exists=True,
        staging_dir="",
        mesh_info={"baseline_drift_N": {"5.5kn": 0.0, "6.2kn": 0.0}},
        speeds={"5.5kn": sp(5.5, delta_55, drift_55), "6.2kn": sp(6.2, delta_62, drift_62)},
        mode=PAIRED_MODE,
    )


@pytest.fixture(scope="module")
def baseline_and_metrics(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    ref = load_baseline_reference()
    metrics = evaluate_tier0_tier1(ref.hull, tmp_path_factory.mktemp("tier01"), ref, skip_resistance=True)
    assert metrics.valid, metrics.error
    return ref, metrics


def _cfd_gates(report) -> dict[str, float]:
    return {g: m for g, m in report.margins.items() if g.startswith("cfd_")}


def test_paired_result_is_judged_on_its_difference_not_on_absolute_drag(baseline_and_metrics: tuple) -> None:
    ref, metrics = baseline_and_metrics
    # 18 N is 2.5 N below the published 500-step baseline; only the difference may count
    report = evaluate_gates(metrics, ref, cfd=_paired_metrics(-0.10, +0.02))
    gates = _cfd_gates(report)
    assert set(gates) == {
        f"cfd_{gate}_{speed}kn"
        for gate in ("drag_delta", "settled", "baseline_settled")
        for speed in (5.5, 6.2)
    }
    assert gates["cfd_drag_delta_5.5kn"] == pytest.approx(CFD_PAIRED_DRAG_TOL_N + 0.10)
    assert gates["cfd_drag_delta_6.2kn"] == pytest.approx(CFD_PAIRED_DRAG_TOL_N - 0.02)
    assert not [v for v in report.violations if v.gate.startswith("cfd_")]

    worse = evaluate_gates(metrics, ref, cfd=_paired_metrics(-0.10, CFD_PAIRED_DRAG_TOL_N + 0.01))
    assert [v.gate for v in worse.violations if v.gate.startswith("cfd_")] == ["cfd_drag_delta_6.2kn"]


def test_paired_result_that_has_not_settled_cannot_pass(baseline_and_metrics: tuple) -> None:
    ref, metrics = baseline_and_metrics
    moving = _paired_metrics(-0.30, -0.30, drift_55=-(CFD_PAIRED_DRIFT_TOL_N + 0.01))
    report = evaluate_gates(metrics, ref, cfd=moving)
    assert [v.gate for v in report.violations if v.gate.startswith("cfd_")] == ["cfd_settled_5.5kn"]
    assert "still moving" in next(v.message for v in report.violations if v.gate == "cfd_settled_5.5kn")


def test_baseline_drift_alone_fails_the_affected_speed(baseline_and_metrics: tuple) -> None:
    from rsi.diagnose import diagnose_candidate

    ref, metrics = baseline_and_metrics
    cfd = replace(
        _paired_metrics(-0.30, -0.30),
        mesh_info={"baseline_drift_N": {"5.5kn": CFD_PAIRED_DRIFT_TOL_N, "6.2kn": -0.87}},
    )
    report = evaluate_gates(metrics, ref, cfd=cfd)
    assert [v.gate for v in report.violations if v.gate.startswith("cfd_")] == ["cfd_baseline_settled_6.2kn"]
    assert report.margins["cfd_baseline_settled_6.2kn"] == pytest.approx(CFD_PAIRED_DRIFT_TOL_N - 0.87)
    diagnosis = diagnose_candidate(metrics, report, cfd=cfd)
    item = next(i for i in diagnosis.items if i.gate == "cfd_baseline_settled_6.2kn")
    assert item.code == "cfd_baseline_drift" and not item.parameters and not item.groups


@pytest.mark.parametrize(
    "info",
    [None, {}, {"baseline_drift_N": []}, {"baseline_drift_N": {"6.2kn": float("nan"), "5.5kn": True}}],
)
def test_paired_result_without_finite_reference_drift_fails_closed(baseline_and_metrics, info) -> None:
    ref, metrics = baseline_and_metrics
    report = evaluate_gates(metrics, ref, cfd=replace(_paired_metrics(0.0, 0.0), mesh_info=info))
    assert {v.gate for v in report.violations if v.gate.startswith("cfd_")} == {
        "cfd_baseline_settled_5.5kn", "cfd_baseline_settled_6.2kn"
    }


def test_paired_result_with_a_missing_difference_fails_that_speed(baseline_and_metrics: tuple) -> None:
    ref, metrics = baseline_and_metrics
    cfd = _paired_metrics(-0.1, -0.1)
    speeds = {"5.5kn": cfd.speeds["5.5kn"], "6.2kn": {"drag_total_N": 17.9}}
    report = evaluate_gates(metrics, ref, cfd=replace(cfd, speeds=speeds))
    assert [v.gate for v in report.violations if v.gate.startswith("cfd_")] == ["cfd_missing_6.2kn"]
    nan = replace(cfd, speeds={**cfd.speeds, "5.5kn": {**cfd.speeds["5.5kn"], "drag_delta_N": float("nan")}})
    report = evaluate_gates(metrics, ref, cfd=nan)
    assert [v.gate for v in report.violations if v.gate.startswith("cfd_")] == ["cfd_missing_5.5kn"]


def test_cold_start_results_keep_the_absolute_screen(baseline_and_metrics: tuple) -> None:
    ref, metrics = baseline_and_metrics
    cold = replace(_paired_metrics(-0.1, -0.1), mode=COLD_MODE)
    report = evaluate_gates(metrics, ref, cfd=cold)
    assert set(_cfd_gates(report)) == {"cfd_drag_5.5kn", "cfd_drag_6.2kn"}


# --- the whole evaluation, with OpenFOAM replaced by stand-ins that write what it would write ---


def _stand_in_openfoam(monkeypatch: pytest.MonkeyPatch, calls: dict[str, list]) -> None:
    import run_openfoam_cfd

    import rsi.paired_cfd as paired
    from openfoam_morph import read_binary_stl, read_points, write_points

    header = b"FoamFile\n{\n    format      ascii;\n    class       vectorField;\n    object      points;\n}\n\n"

    def fake_build_mesh(hull_name, out_dir, bashrc, np_ranks=6):
        calls["mesh"].append(hull_name)
        return {"hull": hull_name, "ok": True}

    def fake_cold_run(hull_name, speed_kn, posed, out_dir, bashrc, end_steps=500, np_ranks=6):
        cdir = Path(out_dir) / f"run_{hull_name}_{speed_kn:.1f}kn".replace(".", "p")
        skin = read_binary_stl(Path(out_dir) / f"{hull_name}.stl")[::900, 0, :]
        skin = skin[skin[:, 1] >= 0.0][:200]
        pts = np.vstack([skin, skin + [0.0, 0.06, 0.0], [[9.0, 3.0, -1.0], [-7.0, 2.0, 0.5]]])
        mesh = cdir / "processor0" / "constant" / "polyMesh"
        mesh.mkdir(parents=True)
        write_points(mesh / "points", pts, header, b"\n")
        _write_state(cdir / "processor0", end_steps)
        (cdir / "logs").mkdir(exist_ok=True)
        (cdir / "logs" / "interFoam.log").write_text(f"Time = {end_steps}\n\nEnd\n")
        calls["cold"].append(speed_kn)
        return cdir, {}

    def fake_pair_run(case_dir, bashrc, end_step, np_ranks):
        case_dir = Path(case_dir)
        ctrl = (case_dir / "system" / "controlDict").read_text()
        assert "startFrom latestTime;" in ctrl and f"endTime {end_step};" in ctrl
        assert "maxCo 50;" in (case_dir / "system" / "fvSolution").read_text()
        start = max(int(p.name) for p in (case_dir / "processor0").iterdir() if p.name.isdigit())
        pts = read_points(case_dir / "processor0" / "constant" / "polyMesh" / "points")[0]
        fast = "6p2" in str(case_dir)
        half_drag = (12.0 if fast else 9.0) + 40.0 * pts[:, 2].mean()  # depends on the mesh it was given
        steps = np.arange(start + 1, end_step + 1)
        force = case_dir / "postProcessing" / "forces" / str(start) / "force.dat"
        _write_force_dat(force, steps, np.full(len(steps), half_drag))
        (case_dir / "logs" / "interFoam.log").write_text(f"Time = {end_step}\n\nEnd\n")
        # interFoam writes fields at steps that are multiples of writeInterval, and not otherwise:
        # same boundary types as it was started with, no URef
        assert (case_dir / "processor0" / str(start) / "URef").is_file(), "the sponge needs its reference field"
        assert "type            inletOutlet;" in (case_dir / "processor0" / str(start) / "U").read_text()
        interval = int(re.search(r"writeControl timeStep; writeInterval (\d+); purgeWrite", ctrl).group(1))
        for step in range(start + 1, end_step + 1):
            if step % interval == 0:
                shutil.copytree(case_dir / "processor0" / str(start), case_dir / "processor0" / str(step))
                (case_dir / "processor0" / str(step) / "URef").unlink()
        calls["pair"].append(str(case_dir))
        return force

    monkeypatch.setattr(run_openfoam_cfd, "build_mesh", fake_build_mesh)
    monkeypatch.setattr(run_openfoam_cfd, "run_interfoam_case", fake_cold_run)
    monkeypatch.setattr(paired, "run_pair_case", fake_pair_run)


def test_candidate_is_compared_with_the_baseline_on_the_baselines_own_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openfoam_morph import MorphSettings, SurfaceMorph, read_binary_stl, read_points

    from rsi.genome import DesignSpace, hull_to_vector, vector_to_hull
    from rsi.paired_cfd import evaluate_tier2_cfd_paired

    bashrc = tmp_path / "bashrc"
    bashrc.write_text("# stand-in\n")
    monkeypatch.setenv("OPENFOAM_BASHRC", str(bashrc))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("KAYAK_RSI_NO_TIER2_CACHE", raising=False)
    calls: dict[str, list] = {"mesh": [], "cold": [], "pair": []}
    _stand_in_openfoam(monkeypatch, calls)

    ref = load_baseline_reference()
    space = DesignSpace.default()
    vec = hull_to_vector(ref.hull, space)
    names = [spec.name for spec in space.params]
    deeper = list(vec)
    deeper[names.index("rocker_bow_end_m")] -= 0.004
    cand = vector_to_hull(deeper, ref.hull, space)
    # 20 + 14 = 34 is not a multiple of 14: the reference state must be written all the same
    settings = PairedCfdSettings(ref_steps=20, settle_steps=14, pair_steps=12, tail_steps=6)

    out = evaluate_tier2_cfd_paired(cand, tmp_path / "work" / "a", ref, settings=settings)
    assert out.valid, out.error
    assert out.mode == PAIRED_MODE and out.dry_run is False
    assert calls["mesh"] == ["candidate"] and calls["cold"] == [5.5, 6.2]
    assert len(calls["pair"]) == 6, "per speed: settling, the baseline's continuation, the candidate"
    assert [c.split("/")[-2] for c in calls["pair"]] == ["settle", "settle", "pair_base", "pair_base", "a", "a"]

    ref_root = Path(out.mesh_info["reference_dir"])
    assert (ref_root / "cold" / "stage.json").is_file()
    assert ref_root.parent == tmp_path / "cache" / "kayak-10-9" / "cfd-reference"
    assert (ref_root / "settle" / "5p5kn" / "processor0" / "34").is_dir(), "the reference state is kept"
    assert (ref_root / ".lock").is_file()
    assert (ref_root / "settle" / "5p5kn" / "constant" / "fvOptions").is_file()
    assert (tmp_path / "work" / "a" / "pair_5p5kn" / "constant" / "fvOptions").is_file()
    assert out.mesh_info["baseline_drift_N"] == {"5.5kn": pytest.approx(0.0), "6.2kn": pytest.approx(0.0)}
    base_tri = read_binary_stl(ref_root / "cold" / "candidate.stl")
    cand_tri = read_binary_stl(tmp_path / "work" / "a" / "candidate.stl")
    pts = read_points(ref_root / "cold" / "run_candidate_5p5kn" / "processor0" / "constant" / "polyMesh" / "points")[0]
    moved, _ = SurfaceMorph(base_tri, cand_tri, MorphSettings(r0_m=0.02, radius_m=0.2)).displacement(pts)
    expected = 2.0 * 40.0 * moved[:, 2].mean()
    assert abs(expected) > 1e-4, "the stand-in must be able to see the candidate's shape"
    for key in ("5.5kn", "6.2kn"):
        sp = out.speeds[key]
        assert sp["drag_delta_N"] == pytest.approx(expected, abs=1e-7)  # force.dat holds 9 digits
        assert sp["drag_delta_drift_N"] == pytest.approx(0.0, abs=1e-12)
        assert (sp["first_step"], sp["last_step"]) == (41, 46)
    assert out.speeds["6.2kn"]["baseline_drag_N"] - out.speeds["5.5kn"]["baseline_drag_N"] == pytest.approx(6.0)
    # the reference keeps its unmoved mesh; a finished candidate case keeps forces and logs only
    assert not list((ref_root / "cold").rglob("points.orig"))
    assert not list((tmp_path / "work" / "a" / "pair_5p5kn").glob("processor*"))
    assert list((tmp_path / "work" / "a" / "pair_5p5kn").rglob("force.dat"))
    assert out.mesh_info["mesh_move"]["points_within_1mm_of_hull"] == (len(pts) - 2) // 2, "the stand-in's skin points"

    # the same hull again comes from the cache; another hull reuses the reference
    again = evaluate_tier2_cfd_paired(cand, tmp_path / "work" / "b", ref, settings=settings)
    assert again.valid and again.speeds == out.speeds and len(calls["pair"]) == 6
    same = evaluate_tier2_cfd_paired(ref.hull, tmp_path / "work" / "c", ref, settings=settings)
    assert same.valid, same.error
    assert calls["cold"] == [5.5, 6.2] and len(calls["pair"]) == 8
    assert same.speeds["5.5kn"]["drag_delta_N"] == pytest.approx(0.0, abs=1e-7), "the baseline against itself is zero"

    report = evaluate_gates(evaluate_tier0_tier1(ref.hull, tmp_path / "t1", ref, skip_resistance=True), ref, cfd=same)
    assert not [v for v in report.violations if v.gate.startswith("cfd_")]


def test_a_hull_of_another_topology_is_reported_as_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rsi.paired_cfd as paired
    from rsi.paired_cfd import evaluate_tier2_cfd_paired

    bashrc = tmp_path / "bashrc"
    bashrc.write_text("# stand-in\n")
    monkeypatch.setenv("OPENFOAM_BASHRC", str(bashrc))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls: dict[str, list] = {"mesh": [], "cold": [], "pair": []}
    _stand_in_openfoam(monkeypatch, calls)
    ref = load_baseline_reference()
    real_stage = paired.stage_posed_hull_stl

    def coarse_candidate(hull, work, **kw):
        # only the candidate's own staging is coarse; the check of the reference's skin is not
        if Path(work) == tmp_path / "work":
            kw["mesh_resolution"] = (221, 65)
        return real_stage(hull, work, **kw)

    monkeypatch.setattr(paired, "stage_posed_hull_stl", coarse_candidate)
    out = evaluate_tier2_cfd_paired(
        ref.hull,
        tmp_path / "work",
        ref,
        settings=PairedCfdSettings(ref_steps=20, settle_steps=10, pair_steps=12, tail_steps=6),
    )
    assert out.valid is False and "same surface topology" in (out.error or "")
    assert out.mode == PAIRED_MODE


def test_a_reference_built_on_another_skin_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rsi.genome import DesignSpace, hull_to_vector, vector_to_hull
    from rsi.paired_cfd import evaluate_tier2_cfd_paired

    bashrc = tmp_path / "bashrc"
    bashrc.write_text("# stand-in\n")
    monkeypatch.setenv("OPENFOAM_BASHRC", str(bashrc))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls: dict[str, list] = {"mesh": [], "cold": [], "pair": []}
    _stand_in_openfoam(monkeypatch, calls)
    ref = load_baseline_reference()
    settings = PairedCfdSettings(ref_steps=20, settle_steps=10, pair_steps=12, tail_steps=6)
    assert evaluate_tier2_cfd_paired(ref.hull, tmp_path / "w1", ref, settings=settings).valid

    # the surface generator changes: the baseline staged today is no longer the skin the mesh sits on
    space = DesignSpace.default()
    vec = hull_to_vector(ref.hull, space)
    vec[[spec.name for spec in space.params].index("rocker_bow_end_m")] -= 0.003
    other = vector_to_hull(vec, ref.hull, space)
    ref_root = next((tmp_path / "cache" / "kayak-10-9" / "cfd-reference").iterdir())
    from rsi.fidelity import stage_posed_hull_stl

    stage_posed_hull_stl(other, ref_root / "cold", hull_name="candidate")
    out = evaluate_tier2_cfd_paired(other, tmp_path / "w2", ref, settings=settings)
    assert out.valid is False and "another baseline skin" in (out.error or "")
    assert len(calls["pair"]) == 6, "nothing was run on the stale reference"
