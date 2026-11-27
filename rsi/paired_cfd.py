"""Tier 2 CFD as a paired comparison: baseline and candidate on one mesh, from one flow state.

The cold-start screen (``rsi.fidelity.evaluate_tier2_cfd``) meshes every hull separately and starts
every run from rest. Its 500-step reading sits on a start-up transient of more than 1 N that differs
from hull to hull, and separate meshes differ by cut-cell pattern. Measured on 2026-11-26: the same
candidate read +0.56 N against v0.2.3 over steps 350-500 and -0.13 N over steps 1500-2000.

Here the baseline is run once from rest for ``ref_steps`` and then settled for ``settle_steps``
with the water level pinned at the outlet and a sponge at both ends of the tank (the reference
state). To judge a
candidate, the reference mesh is moved onto the candidate's skin (``openfoam_morph``), the
candidate continues from the reference flow state for ``pair_steps``, and its drag history is
compared step for step with the baseline continued from the same state for the same steps. Mesh
connectivity, wall spacing and decomposition are common to both. Drift in the reference can still
bias the difference if the two flows respond at different rates.

Why the settling: the start from rest leaves the tank's water level and waves several metres long
in motion. 1 mm of level is 15.6 N of vertical force, and a surface slope of 1 mm per metre over
the hull is 1 N of drag (the boat's weight times the slope). With the case's own outlet (mean
velocity prescribed, level free) v0.2.3's drag went from 18.2 N to 15.5 N between steps 2050 and
2400 and back to 18.3 N by 3600. A difference taken while the baseline moves 0.01 N per step is
contaminated by any lag between the two runs. Damping vertical velocity in zones did not help:
long waves are nearly horizontal motion. The excursion also appeared with the outlet level pinned
from the start. Its cause remains open.

A candidate must come from the same parametric surface as the baseline (same STL topology).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from kayakgen.model.hull import Hull

from rsi.baseline import BaselineReference
from rsi.fidelity import (
    CfdMetrics,
    _atomic_write_json,
    _ensure_cad_on_path,
    evaluate_tier2_cfd,
    resolve_openfoam_bashrc,
    resolve_tier2_cache_root,
    stage_openfoam_case,
    stage_posed_hull_stl,
)

# In every cache key. Raise it whenever a change in this file alters what a reference or a pair
# result holds (stages, text edits of case files, the difference itself); the keys do not hash this
# file, because a reference costs hours and a comment must not discard it.
PAIRED_SCHEMA_VERSION = 1
PAIRED_MODE = "paired_warm_start"
COLD_MODE = "cold_start"
# What a caller asks for: "paired" (default) or "cold" (each hull meshed alone and run from rest).
CFD_MODES = ("paired", "cold")
HULL_NAME = "candidate"
_CONSTANT_FILES = ("g", "transportProperties", "turbulenceProperties")
# Sponge at both ends of the tank of cad/openfoam_dicts.py (x = -8..13.12, hull at x = -2.74..2.74):
# velocity is pulled towards the uniform stream at a rate w * frequency * blend, where blend rises
# from 0 at radius1 to 1 at radius2 from the origin, the middle of the tank. Damping therefore
# starts at x = -5.04 and x = 10.16 and is full at the inlet and the outlet.
SPONGE_ORIGIN = (2.56, 0.0, 0.0)
SPONGE_RADIUS1_M = 7.6
SPONGE_RADIUS2_M = 10.56
SPONGE_W = 20
REFERENCE_VELOCITY_FIELD = "URef"
KNOT_MPS = 0.514444


@dataclass(frozen=True)
class PairedCfdSettings:
    """Everything a paired result depends on besides the two hulls."""

    speeds_kn: tuple[float, ...] = (5.5, 6.2)
    ref_steps: int = 2000  # steps of the baseline from rest, with the case's own settings
    settle_steps: int = 2000  # further steps with max_co and the settling treatment (see below)
    # Steps baseline and candidate both run from the reference state, and how many of the last of
    # them are averaged. The flow passes the hull once in about 300 steps at max_co 50, and the
    # baseline's drag swings about 0.3 N with a period near 300 steps; a lag between the two runs
    # then gives an error that averages out over whole periods. So: one pass discarded, two
    # periods averaged.
    pair_steps: int = 900
    tail_steps: int = 600
    np_ranks: int = 6
    # Settling treatment: the level is pinned at the outlet (p_rgh = 0, velocity and water fraction
    # free) and the end sponge runs at w * frequency, 3 per second at 0.15. 0 leaves the case's own
    # outlet and no sponge, and then no settling stage makes sense either.
    sponge_frequency: float = 0.15
    # Local-time-step Courant limit of the pair runs. The reference run from rest keeps the case's
    # own value (10). At 10 the step near the hull is about 3 ms and one pass of the hull takes
    # about 650 steps: 153 air cells over the bow wave are limited by the water they take in, and
    # the 5 % step smoothing spreads their limit over the whole hull. 50 raised the median step
    # 2.5 times and stayed stable; relaxing the smoothing to 0.2 diverged within 150 steps.
    max_co: float = 50.0
    morph_r0_m: float = 0.02
    morph_radius_m: float = 0.2

    def __post_init__(self) -> None:
        if not self.speeds_kn:
            raise ValueError("at least one speed is required")
        if self.ref_steps < 2 or self.pair_steps < 2 or self.np_ranks < 1:
            raise ValueError("ref_steps and pair_steps must be >= 2 and np_ranks >= 1")
        if self.settle_steps < 0 or self.sponge_frequency < 0.0:
            raise ValueError("settle_steps and sponge_frequency must not be negative")
        if (self.settle_steps > 0) != (self.sponge_frequency > 0.0):
            raise ValueError("settle_steps and sponge_frequency must be both zero or both positive")
        if self.max_co <= 0.0:
            raise ValueError("max_co must be positive")
        if not 2 <= self.tail_steps <= self.pair_steps or self.tail_steps % 2:
            raise ValueError("tail_steps must be even and between 2 and pair_steps")

    @property
    def state_step(self) -> int:
        """Step number of the reference state."""
        return self.ref_steps + self.settle_steps

    @property
    def end_step(self) -> int:
        """Step number at which a pair run ends."""
        return self.ref_steps + self.settle_steps + self.pair_steps


@dataclass(frozen=True)
class CfdReference:
    """The baseline's reference state and its continuation, on disk."""

    root: Path
    key: str
    key_obj: dict[str, Any]
    stl_path: Path
    posed: dict[str, Any]
    state_dirs: dict[str, Path]
    baseline_force_files: dict[str, Path]
    settle_drift_N: dict[str, float]


def resolve_cfd_reference_root() -> Path:
    """Where the baseline's reference lives. Not the result cache: it is a prerequisite that costs
    hours, so switching the result cache off must not make every candidate build its own."""
    env = os.environ.get("KAYAK_RSI_CFD_REFERENCE_ROOT", "").strip()
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg) if xdg else Path.home() / ".cache") / "kayak-10-9" / "cfd-reference"


@contextlib.contextmanager
def _reference_lock(ref_root: Path) -> Any:
    """One builder at a time: a second process must not wipe a stage the first is running."""
    ref_root.mkdir(parents=True, exist_ok=True)
    with open(ref_root / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _same_skin(stl_a: Path, stl_b: Path) -> bool:
    """Same triangles? The STL writer stamps the time into the header, so file hashes differ."""
    return Path(stl_a).read_bytes()[80:] == Path(stl_b).read_bytes()[80:]


def speed_key(speed_kn: float) -> str:
    return f"{speed_kn:.1f}kn"


def _speed_tag(speed_kn: float) -> str:
    return f"{speed_kn:.1f}kn".replace(".", "p")


def _run_tag(speed_kn: float) -> str:
    return f"run_{HULL_NAME}_{_speed_tag(speed_kn)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_digest(*module_names: str) -> str:
    _ensure_cad_on_path()
    import importlib

    h = hashlib.sha256()
    for name in module_names:
        h.update(Path(importlib.import_module(name).__file__).read_bytes())
    return h.hexdigest()[:16]


def _case_settings_digest() -> str:
    """Digest of what builds and runs the reference case, so a change there gets a new reference."""
    return _source_digest("openfoam_dicts", "run_openfoam_cfd")


def reference_key(
    baseline_design_hash: str,
    settings: PairedCfdSettings,
    bashrc: Path,
    mesh_resolution: tuple[int, int] = (441, 129),
) -> tuple[str, dict[str, Any]]:
    key_obj = {
        "solver": "openfoam_cfd_reference",
        "schema_version": PAIRED_SCHEMA_VERSION,
        "baseline_design_hash": baseline_design_hash,
        "speeds_kn": [float(s) for s in settings.speeds_kn],
        "ref_steps": int(settings.ref_steps),
        "settle_steps": int(settings.settle_steps),
        "pair_steps": int(settings.pair_steps),
        "max_co": float(settings.max_co),
        "sponge": [float(settings.sponge_frequency), SPONGE_W, list(SPONGE_ORIGIN), SPONGE_RADIUS1_M, SPONGE_RADIUS2_M],
        "outlet_pinned": settings.sponge_frequency > 0.0,
        "np_ranks": int(settings.np_ranks),
        "mesh_resolution": [int(mesh_resolution[0]), int(mesh_resolution[1])],
        "case_settings": _case_settings_digest(),
        "bashrc": str(bashrc),
    }
    digest = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return f"cfdref_{digest}", key_obj


def _pair_key(
    hull: Hull, ref_key: str, settings: PairedCfdSettings
) -> tuple[str, dict[str, Any]]:
    key_obj = {
        "solver": "openfoam_cfd_paired",
        "schema_version": PAIRED_SCHEMA_VERSION,
        "design_hash": hull.design_hash(),
        "reference": ref_key,
        "tail_steps": int(settings.tail_steps),
        "morph": asdict(_morph_settings(settings)),
        "morph_code": _source_digest("openfoam_morph"),
    }
    digest = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return f"cfdpair_{digest}", key_obj


def _morph_settings(settings: PairedCfdSettings) -> Any:
    _ensure_cad_on_path()
    from openfoam_morph import MorphSettings

    return MorphSettings(r0_m=settings.morph_r0_m, radius_m=settings.morph_radius_m)


def _log_finished(log_path: Path, end_step: int) -> bool:
    if not log_path.is_file():
        return False
    txt = log_path.read_text(errors="replace")
    return bool(re.search(rf"^Time = {end_step}\s*$", txt, re.M) and re.search(r"^End\s*$", txt, re.M))


def _read_stage(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def read_force_history(force_dat: Path) -> dict[str, np.ndarray]:
    """Full-hull drag history from a half-model ``force.dat`` (total, pressure, viscous, in x)."""
    _ensure_cad_on_path()
    from run_openfoam_cfd import _parse_forces

    steps, fp, fv = _parse_forces(Path(force_dat))
    if len(steps) == 0:
        raise ValueError(f"empty force history in {force_dat}")
    return {
        "step": np.rint(steps).astype(int),
        "pressure": fp[:, 0],
        "viscous": fv[:, 0],
        "total": fp[:, 0] + fv[:, 0],
    }


def paired_force_delta(
    base: dict[str, np.ndarray], cand: dict[str, np.ndarray], tail_steps: int
) -> dict[str, Any]:
    """Candidate minus baseline at matching steps, averaged over the last ``tail_steps``.

    ``drag_delta_drift_N`` is the second half of the tail minus the first half: how much the
    difference was still moving when the run ended.
    """
    common, ib, ic = np.intersect1d(base["step"], cand["step"], return_indices=True)
    if len(common) < tail_steps:
        raise ValueError(
            f"baseline and candidate share {len(common)} steps; {tail_steps} are needed"
        )
    if common[-1] != base["step"][-1] or common[-1] != cand["step"][-1]:
        raise ValueError("baseline and candidate force histories do not end at the same step")
    ib, ic = ib[-tail_steps:], ic[-tail_steps:]
    d_tot = cand["total"][ic] - base["total"][ib]
    half = tail_steps // 2
    return {
        "drag_delta_N": float(d_tot.mean()),
        "drag_delta_pressure_N": float((cand["pressure"][ic] - base["pressure"][ib]).mean()),
        "drag_delta_viscous_N": float((cand["viscous"][ic] - base["viscous"][ib]).mean()),
        "drag_delta_drift_N": float(d_tot[half:].mean() - d_tot[:half].mean()),
        "drag_delta_std_N": float(d_tot.std()),
        "baseline_drag_N": float(base["total"][ib].mean()),
        "drag_total_N": float(cand["total"][ic].mean()),
        "drag_std_N": float(cand["total"][ic].std()),
        "first_step": int(common[-tail_steps]),
        "last_step": int(common[-1]),
    }


def prepare_pair_case(
    state_case: Path,
    dst: Path,
    start_step: int,
    end_step: int,
    *,
    max_co: float | None = None,
    u_mps: float | None = None,
) -> None:
    """Copy a finished decomposed run at ``start_step`` into a case that continues to ``end_step``.

    ``max_co`` replaces the case's local-time-step Courant limit; ``None`` keeps it. A state that
    was settled with the sponge needs the stream speed ``u_mps`` for the sponge's reference field.
    """
    state_case, dst = Path(state_case), Path(dst)
    procs = sorted(p for p in state_case.glob("processor*") if p.is_dir())
    if not procs:
        raise FileNotFoundError(f"no processor directories in {state_case}")
    for p in procs:
        if not (p / str(start_step)).is_dir():
            raise FileNotFoundError(f"{p} has no time directory {start_step}")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "logs").mkdir(parents=True)
    (dst / "constant").mkdir()
    shutil.copytree(state_case / "system", dst / "system")
    for name in _CONSTANT_FILES:
        shutil.copyfile(state_case / "constant" / name, dst / "constant" / name)
    has_sponge = (state_case / "constant" / "fvOptions").is_file()
    if has_sponge:  # a settled state: the sponge stays on
        if u_mps is None:
            raise ValueError(f"{state_case} was settled with a sponge; its stream speed is needed")
        shutil.copyfile(state_case / "constant" / "fvOptions", dst / "constant" / "fvOptions")
    for p in procs:
        shutil.copytree(p / "constant", dst / p.name / "constant")
        orig = dst / p.name / "constant" / "polyMesh" / "points.orig"
        if orig.exists():  # the state itself must be an unmoved mesh
            raise ValueError(f"{p} holds a moved mesh; a reference state must not")
        shutil.copytree(p / str(start_step), dst / p.name / str(start_step))

    ctrl_path = dst / "system" / "controlDict"
    ctrl = ctrl_path.read_text()
    for pattern, repl in (
        (r"startFrom\s+(?:startTime|latestTime);", "startFrom latestTime;"),
        (r"stopAt\s+endTime;\s*endTime\s+\d+;", f"stopAt endTime; endTime {end_step};"),
        # interFoam writes fields when the step number is a multiple of the interval, not at the
        # end of a run. The end step itself is the one interval that always writes there, once.
        (r"writeControl\s+timeStep;\s*writeInterval\s+\d+;\s*purgeWrite",
         f"writeControl timeStep; writeInterval {end_step}; purgeWrite"),
    ):
        ctrl, n = re.subn(pattern, repl, ctrl)
        if n != 1:
            raise ValueError(f"controlDict of {state_case} does not have the expected layout: {pattern}")
    ctrl_path.write_text(ctrl)
    if max_co is not None:
        sol_path = dst / "system" / "fvSolution"
        sol, n = re.subn(r"\bmaxCo\s+[0-9.eE+-]+;", f"maxCo {max_co:g};", sol_path.read_text())
        if n != 1:
            raise ValueError(f"fvSolution of {state_case} does not set maxCo exactly once")
        sol_path.write_text(sol)
    if has_sponge:
        write_reference_velocity(dst, start_step, float(u_mps))


_FIELD_HEAD = "FoamFile { version 2.0; format ascii; class %s; object %s; }\n"


def _outlet_block(text: str, path: Path) -> tuple[int, int]:
    m = re.search(r"\n    outlet\n    \{\n", text)
    if m is None:
        raise ValueError(f"{path} has no outlet patch")
    return m.end(), text.index("\n    }", m.end())


def write_reference_velocity(case_dir: Path, step: int, u_mps: float) -> None:
    """Write the uniform stream the sponge pulls towards as ``URef`` in every processor's ``step``.

    interFoam does not write that field with its results, so every start state needs it again.
    """
    stream = f"({u_mps:.6f} 0 0)"
    procs = sorted(p for p in Path(case_dir).glob("processor*") if p.is_dir())
    for proc in procs:
        u_path = proc / str(step) / "U"
        text = u_path.read_text()
        patches = re.findall(r"\n    (\w+)\n    \{\n\s+type\s+(\w+);", text[text.index("boundaryField") :])
        if not patches:
            raise ValueError(f"{u_path} has no boundary patches")
        blocks = []
        for name, kind in patches:
            if kind == "symmetryPlane":
                blocks.append(f"    {name} {{ type symmetryPlane; }}")
            else:
                kept = "processor" if kind == "processor" else "calculated"
                blocks.append(f"    {name} {{ type {kept}; value uniform {stream}; }}")
        (proc / str(step) / REFERENCE_VELOCITY_FIELD).write_text(
            _FIELD_HEAD % ("volVectorField", REFERENCE_VELOCITY_FIELD)
            + f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform {stream};\n"
            + "boundaryField\n{\n" + "\n".join(blocks) + "\n}\n"
        )


def pin_outlet_level(case_dir: Path, step: int, u_mps: float) -> None:
    """Fix the water level at the outlet in the fields of ``step``.

    The case's own outlet prescribes the mean water velocity and leaves the level free, which let
    the tank's level wander for thousands of steps. Here p_rgh is 0 at the outlet (hydrostatic with
    the surface at z = 0), and velocity and water fraction leave freely.
    """
    edits = {
        "U": (
            (r"type\s+outletPhaseMeanVelocity;", f"type            inletOutlet;\n        inletValue      uniform ({u_mps:.6f} 0 0);"),
            (r"\n\s+Umean\s+[^;]+;", ""),
            (r"\n\s+alpha\s+alpha\.water;", ""),
        ),
        "p_rgh": ((r"type\s+zeroGradient;", "type            fixedValue;\n        value           uniform 0;"),),
        "alpha.water": (
            (r"type\s+variableHeightFlowRate;", "type            zeroGradient;"),
            (r"\n\s+lowerBound\s+[^;]+;", ""),
            (r"\n\s+upperBound\s+[^;]+;", ""),
        ),
    }
    for proc in sorted(p for p in Path(case_dir).glob("processor*") if p.is_dir()):
        for name, subs in edits.items():
            path = proc / str(step) / name
            text = path.read_text()
            a, b = _outlet_block(text, path)
            block = text[a:b]
            for pattern, repl in subs:
                block, n = re.subn(pattern, repl, block)
                if n != 1:
                    raise ValueError(f"outlet of {path} is not the case's own outlet: {pattern}")
            path.write_text(text[:a] + block + text[b:])


def apply_settle_treatment(case_dir: Path, step: int, u_mps: float, sponge_frequency: float) -> None:
    """Turn a prepared case into a settling run: outlet level pinned, sponge at both ends."""
    case_dir = Path(case_dir)
    (case_dir / "constant" / "fvOptions").write_text(
        _FIELD_HEAD % ("dictionary", "fvOptions")
        + "spongeEnds\n{\n    type acousticDampingSource; selectionMode all;\n"
        + f"    frequency {sponge_frequency:g}; w {SPONGE_W}; URef {REFERENCE_VELOCITY_FIELD}; U U;\n"
        + "    origin (%g %g %g); " % SPONGE_ORIGIN
        + f"radius1 {SPONGE_RADIUS1_M:g}; radius2 {SPONGE_RADIUS2_M:g};\n}}\n"
    )
    ctrl_path = case_dir / "system" / "controlDict"
    ctrl, n = re.subn(
        r"functions\s*\{\n",
        "functions\n{\n    readURef { type readFields; libs (fieldFunctionObjects); "
        f"fields ({REFERENCE_VELOCITY_FIELD}); readOnStart true; writeControl none; }}\n",
        ctrl_path.read_text(),
    )
    if n != 1:
        raise ValueError(f"controlDict of {case_dir} has no functions block")
    ctrl_path.write_text(ctrl)
    pin_outlet_level(case_dir, step, u_mps)
    write_reference_velocity(case_dir, step, u_mps)


def run_pair_case(case_dir: Path, bashrc: Path, end_step: int, np_ranks: int) -> Path:
    """Run interFoam on a prepared pair case and return its force history file."""
    _ensure_cad_on_path()
    from run_openfoam_cfd import _run_foam

    case_dir = Path(case_dir)
    cmd = f"mpirun -np {np_ranks} interFoam -parallel > logs/interFoam.log 2>&1"
    rc = _run_foam(bashrc, case_dir, cmd, case_dir / "logs" / "pipeline.log")
    log = case_dir / "logs" / "interFoam.log"
    if rc != 0 or not _log_finished(log, end_step):
        tail = log.read_text(errors="replace")[-1500:] if log.is_file() else ""
        raise RuntimeError(f"interFoam failed in {case_dir} (rc={rc}):\n{tail}")
    files = sorted((case_dir / "postProcessing" / "forces").glob("*/force.dat"))
    if not files:
        raise FileNotFoundError(f"no force.dat written in {case_dir}")
    return files[-1]


def _drop_fields(case_dir: Path) -> None:
    """Remove meshes and fields of a finished pair case; logs and force histories stay."""
    for p in Path(case_dir).glob("processor*"):
        if p.is_dir():
            shutil.rmtree(p)


def _map_speeds(fn: Any, settings: PairedCfdSettings) -> list[Any]:
    """Run the speeds one after the other.

    Two 6-rank runs at once on the 6-core host took 3.0 s per step each; one alone took 1.37 s
    (2026-11-26), so running them together gains nothing and takes every thread.
    """
    return [fn(sp) for sp in settings.speeds_kn]


def _cold_stage_complete(cold: Path, key: str, settings: PairedCfdSettings) -> bool:
    stage = _read_stage(cold / "stage.json")
    if stage is None or stage.get("key") != key:
        return False
    if not (cold / f"{HULL_NAME}.stl").is_file() or not (cold / "cfd_manifest.json").is_file():
        return False
    for sp in settings.speeds_kn:
        run = cold / _run_tag(sp)
        if not _log_finished(run / "logs" / "interFoam.log", settings.ref_steps):
            return False
        if not (run / "processor0" / str(settings.ref_steps)).is_dir():
            return False
    return True


def _force_file_of(case: Path, key: str, end_step: int) -> Path | None:
    """The force history a finished stage recorded, if it is there and reaches ``end_step``."""
    stage = _read_stage(case / "stage.json")
    if stage is None or stage.get("key") != key or not stage.get("force_file"):
        return None
    force = case / str(stage["force_file"])
    try:
        return force if int(read_force_history(force)["step"][-1]) == end_step else None
    except (OSError, ValueError):
        return None


def _settle_stage_complete(case: Path, key: str, state_step: int) -> bool:
    if _force_file_of(case, key, state_step) is None:
        return False
    procs = [p for p in case.glob("processor*") if p.is_dir()]
    return bool(procs) and all((p / str(state_step)).is_dir() for p in procs)


def settle_drift(force_dat: Path, window: int = 200) -> float:
    """Mean drag of the last ``window`` steps minus the ``window`` before: was it still moving?"""
    total = read_force_history(force_dat)["total"]
    window = min(window, len(total) // 2)
    return float(total[-window:].mean() - total[-2 * window : -window].mean())


def ensure_cfd_reference(
    baseline_hull: Hull,
    settings: PairedCfdSettings,
    bashrc: Path,
    root: Path,
) -> CfdReference:
    """Build (or find) the baseline's reference under ``root``; one process at a time."""
    key, _ = reference_key(baseline_hull.design_hash(), settings, bashrc)
    with _reference_lock(Path(root) / key):
        return _ensure_cfd_reference(baseline_hull, settings, bashrc, root)


def _ensure_cfd_reference(
    baseline_hull: Hull,
    settings: PairedCfdSettings,
    bashrc: Path,
    root: Path,
) -> CfdReference:
    """Build (or find) the baseline's reference state and its continuation under ``root``.

    Three stages, each skipped when its outputs are complete:

    * ``cold/``: from rest to ``ref_steps`` with the case's own settings, the same procedure as the
      cold-start screen;
    * ``settle/<speed>/``: on to ``state_step`` with ``max_co``, the outlet level pinned and the
      end sponge. Its last fields are the reference state;
    * ``pair_base/<speed>/``: the baseline on to ``end_step``, the history every candidate is
      compared with.
    """
    _ensure_cad_on_path()
    from run_openfoam_cfd import build_mesh, run_interfoam_case

    key, key_obj = reference_key(baseline_hull.design_hash(), settings, bashrc)
    ref_root = Path(root) / key
    cold = ref_root / "cold"
    state_step, end_step = settings.state_step, settings.end_step

    if not _cold_stage_complete(cold, key, settings):
        if cold.exists():
            shutil.rmtree(cold)
        with contextlib.redirect_stdout(sys.stderr):
            manifest = stage_openfoam_case(
                baseline_hull,
                cold,
                speeds_kn=settings.speeds_kn,
                end_steps=settings.ref_steps,
                np_ranks=settings.np_ranks,
                bashrc=bashrc,
                hull_name=HULL_NAME,
            )
            build_mesh(HULL_NAME, cold, bashrc, np_ranks=settings.np_ranks)
            _map_speeds(
                lambda sp: run_interfoam_case(
                    HULL_NAME,
                    sp,
                    manifest["posed"],
                    cold,
                    bashrc,
                    end_steps=settings.ref_steps,
                    np_ranks=settings.np_ranks,
                ),
                settings,
            )
        _atomic_write_json(
            cold / "stage.json", {"key": key, "completed_utc": _utc_now(), "source": "live_openfoam"}
        )
        if not _cold_stage_complete(cold, key, settings):
            raise RuntimeError(f"reference run from rest did not complete in {cold}")

    manifest = json.loads((cold / "cfd_manifest.json").read_text(encoding="utf-8"))
    # The mesh was built on the skin in cold/. If the surface generator has changed since, the
    # baseline staged today differs from it, and every candidate would be moved from the wrong skin.
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(sys.stderr):
        stage_posed_hull_stl(baseline_hull, Path(tmp), hull_name=HULL_NAME)
        if not _same_skin(Path(tmp) / f"{HULL_NAME}.stl", cold / f"{HULL_NAME}.stl"):
            raise ValueError(
                f"the reference in {ref_root} was built on another baseline skin than the one staged "
                "today; remove that directory to rebuild it"
            )

    def _stage(case: Path, force_file: Path) -> None:
        _atomic_write_json(
            case / "stage.json",
            {
                "key": key,
                "completed_utc": _utc_now(),
                "source": "live_openfoam",
                "force_file": str(force_file.relative_to(case)),
            },
        )

    def _settle(sp: float) -> None:
        case = ref_root / "settle" / _speed_tag(sp)
        if _settle_stage_complete(case, key, state_step):
            return
        prepare_pair_case(cold / _run_tag(sp), case, settings.ref_steps, state_step, max_co=settings.max_co)
        apply_settle_treatment(case, settings.ref_steps, sp * KNOT_MPS, settings.sponge_frequency)
        _stage(case, run_pair_case(case, bashrc, state_step, settings.np_ranks))

    def _continue(sp: float) -> None:
        case = ref_root / "pair_base" / _speed_tag(sp)
        if _force_file_of(case, key, end_step) is not None:
            return
        prepare_pair_case(
            state_dirs[speed_key(sp)], case, state_step, end_step, max_co=settings.max_co, u_mps=sp * KNOT_MPS
        )
        force_file = run_pair_case(case, bashrc, end_step, settings.np_ranks)
        _drop_fields(case)
        _stage(case, force_file)

    if settings.settle_steps > 0:
        _map_speeds(_settle, settings)
        state_dirs = {speed_key(sp): ref_root / "settle" / _speed_tag(sp) for sp in settings.speeds_kn}
    else:
        state_dirs = {speed_key(sp): cold / _run_tag(sp) for sp in settings.speeds_kn}
    _map_speeds(_continue, settings)

    force_files: dict[str, Path] = {}
    drift: dict[str, float] = {}
    for sp in settings.speeds_kn:
        case = ref_root / "pair_base" / _speed_tag(sp)
        force = _force_file_of(case, key, end_step)
        if force is None:
            raise RuntimeError(f"baseline continuation did not complete in {case}")
        force_files[speed_key(sp)] = force
        drift[speed_key(sp)] = settle_drift(force, window=settings.tail_steps)
    _atomic_write_json(ref_root / "meta.json", {"key": key_obj, "updated_utc": _utc_now()})
    return CfdReference(
        root=ref_root,
        key=key,
        key_obj=key_obj,
        stl_path=cold / f"{HULL_NAME}.stl",
        posed=dict(manifest["posed"]),
        state_dirs=state_dirs,
        baseline_force_files=force_files,
        settle_drift_N=drift,
    )


def _invalid(
    error: str,
    t0: float,
    bashrc: Path,
    work: Path,
    out_json: Path,
    *,
    dry_run: bool = False,
    posed: dict[str, Any] | None = None,
) -> CfdMetrics:
    return CfdMetrics(
        valid=False,
        error=error,
        dry_run=dry_run,
        elapsed_s=round(time.monotonic() - t0, 3),
        bashrc=str(bashrc),
        bashrc_exists=bashrc.is_file(),
        staging_dir=str(work),
        posed=posed,
        output_json_path=str(out_json),
        mode=PAIRED_MODE,
    )


def evaluate_tier2_cfd_paired(
    hull: Hull,
    work_dir: Path,
    baseline: BaselineReference,
    *,
    settings: PairedCfdSettings | None = None,
    bashrc: Path | None = None,
    reference_root: Path | None = None,
    keep_fields: bool = False,
) -> CfdMetrics:
    """Drag of ``hull`` relative to the baseline, both on one mesh from one flow state."""
    t0 = time.monotonic()
    settings = settings or PairedCfdSettings()
    work = Path(work_dir)
    resolved_bashrc = resolve_openfoam_bashrc(bashrc)
    out_json = work / "openfoam_cfd_metrics.json"
    if not resolved_bashrc.is_file():
        return _invalid(f"OpenFOAM bashrc not found: {resolved_bashrc}", t0, resolved_bashrc, work, out_json)

    baseline_hull = baseline.hull
    cache_root = resolve_tier2_cache_root()
    ref_root = Path(reference_root) if reference_root is not None else resolve_cfd_reference_root()
    ref_key, _ = reference_key(baseline.design_hash, settings, resolved_bashrc)
    pair_key, pair_key_obj = _pair_key(hull, ref_key, settings)

    if cache_root is not None:
        cached = _read_stage(cache_root / pair_key / "openfoam_cfd_metrics.json")
        meta = _read_stage(cache_root / pair_key / "meta.json")
        if cached is not None and meta is not None and meta.get("key") == pair_key_obj:
            work.mkdir(parents=True, exist_ok=True)
            (work / f"{HULL_NAME}.hull.json").write_text(hull.model_dump_json(indent=2) + "\n")
            _atomic_write_json(out_json, cached)
            return CfdMetrics(
                valid=True,
                error=None,
                dry_run=False,
                elapsed_s=round(time.monotonic() - t0, 3),
                bashrc=str(resolved_bashrc),
                bashrc_exists=True,
                staging_dir=str(work),
                posed=cached.get("posed"),
                mesh_info=cached.get("mesh_info"),
                speeds=cached.get("speeds"),
                output_json_path=str(out_json),
                mode=PAIRED_MODE,
            )

    posed: dict[str, Any] | None = None
    try:
        _ensure_cad_on_path()
        from openfoam_morph import (
            SurfaceMorph,
            morph_case_points,
            read_binary_stl,
            surface_change_report,
        )

        with contextlib.redirect_stdout(sys.stderr):
            reference = ensure_cfd_reference(baseline_hull, settings, resolved_bashrc, ref_root)
            work.mkdir(parents=True, exist_ok=True)
            (work / f"{HULL_NAME}.hull.json").write_text(hull.model_dump_json(indent=2) + "\n")
            posed = stage_posed_hull_stl(hull, work, hull_name=HULL_NAME)

        base_tri = read_binary_stl(reference.stl_path)
        cand_tri = read_binary_stl(work / f"{HULL_NAME}.stl")
        morph = SurfaceMorph(base_tri, cand_tri, _morph_settings(settings))
        end_step = settings.end_step

        def _solve(sp: float) -> tuple[str, dict[str, Any], dict[str, Any]]:
            case = work / f"pair_{_speed_tag(sp)}"
            prepare_pair_case(
                reference.state_dirs[speed_key(sp)],
                case,
                settings.state_step,
                end_step,
                max_co=settings.max_co,
                u_mps=sp * KNOT_MPS,
            )
            moved = morph_case_points(case, morph)
            force_file = run_pair_case(case, resolved_bashrc, end_step, settings.np_ranks)
            delta = paired_force_delta(
                read_force_history(reference.baseline_force_files[speed_key(sp)]),
                read_force_history(force_file),
                settings.tail_steps,
            )
            if not keep_fields:
                _drop_fields(case)
            return speed_key(sp), {"hull": HULL_NAME, "speed_kn": float(sp), **delta}, moved

        with contextlib.redirect_stdout(sys.stderr):
            solved = _map_speeds(_solve, settings)

        payload = {
            "mode": PAIRED_MODE,
            "posed": posed,
            "mesh_info": {
                "reference": reference.key,
                "reference_dir": str(reference.root),
                # Difference between the final two means of the baseline continuation. Each
                # uses tail_steps or half the available history, whichever is shorter.
                "baseline_drift_N": reference.settle_drift_N,
                "settings": asdict(settings),
                "surface_change": surface_change_report(base_tri, cand_tri),
                "mesh_move": solved[0][2],
            },
            "speeds": {k: v for k, v, _ in solved},
        }
        _atomic_write_json(out_json, payload)
        elapsed = round(time.monotonic() - t0, 3)
        if cache_root is not None:
            _atomic_write_json(cache_root / pair_key / "openfoam_cfd_metrics.json", payload)
            _atomic_write_json(
                cache_root / pair_key / "meta.json",
                {"source": "live_openfoam", "created_utc": _utc_now(), "elapsed_s": elapsed, "key": pair_key_obj},
            )
        return CfdMetrics(
            valid=True,
            error=None,
            dry_run=False,
            elapsed_s=elapsed,
            bashrc=str(resolved_bashrc),
            bashrc_exists=True,
            staging_dir=str(work),
            posed=posed,
            mesh_info=payload["mesh_info"],
            speeds=payload["speeds"],
            output_json_path=str(out_json),
            mode=PAIRED_MODE,
        )
    except Exception as exc:  # noqa: BLE001
        return _invalid(f"paired OpenFOAM comparison failed: {exc}", t0, resolved_bashrc, work, out_json, posed=posed)


def expected_result_mode(cfd_mode: str, dry_run: bool) -> str:
    """The ``CfdMetrics.mode`` a request for ``cfd_mode`` produces."""
    if cfd_mode not in CFD_MODES:
        raise ValueError(f"cfd_mode must be one of {CFD_MODES}, got {cfd_mode!r}")
    return COLD_MODE if (dry_run or cfd_mode == "cold") else PAIRED_MODE


def evaluate_tier2_cfd_mode(
    hull: Hull,
    work_dir: Path,
    baseline: BaselineReference,
    *,
    mode: str = "paired",
    dry_run: bool = False,
) -> CfdMetrics:
    """Tier 2 CFD in the requested mode. A dry run only stages the cold-start case."""
    if expected_result_mode(mode, dry_run) == COLD_MODE:
        return evaluate_tier2_cfd(hull, work_dir, baseline, dry_run=dry_run)
    return evaluate_tier2_cfd_paired(hull, work_dir, baseline)


def current_reference_key(baseline: BaselineReference) -> str:
    """Key of the reference a paired result must have been taken against to be valid today."""
    return reference_key(baseline.design_hash, PairedCfdSettings(), resolve_openfoam_bashrc())[0]
