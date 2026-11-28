"""Fine-grid runs of the free-surface panel solver; writes the JSON behind the "Fine grids" section
of docs/panel-solver/README.md.

Usage (from the checkout; the CuPy stages need an environment with the ``gpu`` extra and enough
free memory on the card, about 66 bytes per panel squared in double precision):

    python scripts/panel_gpu_ladder.py STAGE [STAGE ...] [--backend cupy] [--budget-s S]

Stages, each writing ``<out>/<prefix><stage>.json`` after every case so an interrupted run keeps
what it finished:

* ``ladder``    fixed-pose wave resistance of the baseline and the three campaign finalists on a
                family of station counts, with wall time and peak memory per size.
* ``thin``      uniform-stream linearisation on a thin Wigley hull against Michell's integral.
* ``precision`` single against double precision on the largest ladder grid that fits.
* ``free``      baseline and finalists free to sink and trim, in priority order, as far as the
                time budget allows.
* ``timing``    one fixed-pose baseline solve on one grid, for a CPU figure to set beside the GPU's.
* ``variants``  fixed-pose paired differences with the near-hull row width, the source height and
                the row count changed, on several station counts.

``--budget-s`` is a wall-clock allowance for the whole invocation: a case whose estimated time
does not fit in what is left is skipped and recorded as skipped. This script only computes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi import panel_solver as ps  # noqa: E402
from rsi.evaluator import KG_M, LB_TO_KG, RHO_SEAWATER, _michell_fast_curve  # noqa: E402
from rsi.panel_hull import hull_mesh  # noqa: E402

DEFAULT_CAMPAIGN = Path.home() / ".cache/kayak-10-9/campaigns/k109_rsi_v7_localfriction"
BASELINE = ROOT / "hull" / "kayak-10-9.hull.json"
FINALISTS = ("iter137_cand004", "iter139_cand009", "iter139_cand002")
SPEEDS_KN = (5.0, 5.5, 6.2)
FREE_SPEEDS_KN = (5.5, 6.2)
LOAD_LB = 225.0
RHO_FRESH = 1000.0
LADDER_NX = (40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115)
# (stations, rows, girth panels) for the thin-hull check
THIN_CASES = ((80, 48, 8), (100, 60, 8), (115, 69, 8), (120, 48, 8), (160, 48, 8))
# free runs in priority order: the pair the coarse grids left unresolved first, on two grids
FREE_PLAN = (
    (80, ("baseline", FINALISTS[0])),
    (100, ("baseline", FINALISTS[0])),
    (90, ("baseline", FINALISTS[0])),
    (80, FINALISTS[1:]),
    (100, FINALISTS[1:]),
)


class Clock:
    def __init__(self, budget_s: float | None):
        self.t0 = time.perf_counter()
        self.budget_s = budget_s

    def left(self) -> float:
        return math.inf if self.budget_s is None else self.budget_s - (time.perf_counter() - self.t0)


class Gpu:
    """Memory bookkeeping for the CuPy pool; every method is a no-op on the numpy backend."""

    def __init__(self, backend: str):
        self.xp = ps.get_backend(backend) if backend == "cupy" else None

    def reset(self) -> None:
        if self.xp is not None:
            self.xp.get_default_memory_pool().free_all_blocks()

    def pool_mib(self) -> float | None:
        # the pool keeps every block it was given until reset(), so this is its high-water mark
        if self.xp is None:
            return None
        return self.xp.get_default_memory_pool().total_bytes() / 2**20

    def process_mib(self) -> float | None:
        """Memory the driver charges this process (pool, CUDA context and library workspaces)."""
        if self.xp is None:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.splitlines():
            pid, _, mem = line.partition(",")
            if pid.strip() == str(os.getpid()):
                return float(mem)
        return None

    def is_oom(self, exc: BaseException) -> bool:
        return self.xp is not None and isinstance(exc, self.xp.cuda.memory.OutOfMemoryError)

    def describe(self) -> dict[str, Any]:
        if self.xp is None:
            return {}
        q = "name,driver_version,memory.total,memory.used"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
        return {"cupy": self.xp.__version__, "gpu": dict(zip(q.split(","), [s.strip() for s in out.split(",")]))}


def _stamp(args: argparse.Namespace, gpu: Gpu) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "backend": args.backend,
        "pair_budget": args.pair_budget,
        "threads": ps.host_threads(),
        "load_average": list(os.getloadavg()),
        "host_cpus": os.cpu_count(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    } | gpu.describe()


def _write(args: argparse.Namespace, gpu: Gpu, stage: str, result: dict[str, Any]) -> None:
    result["provenance"] = _stamp(args, gpu)
    path = args.out / f"{args.prefix}{stage}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")


def _cfg(args: argparse.Namespace, nx: int, **kw: Any) -> ps.PanelConfig:
    return ps.PanelConfig(**{"nx_hull": nx, "ny_fs": round(0.6 * nx), "pair_budget": args.pair_budget, **kw})


def _meshes(args: argparse.Namespace, names: tuple[str, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    paths = {"baseline": BASELINE} | {
        n: args.campaign / "candidates" / n / "candidate.hull.json" for n in FINALISTS
    }
    return {n: hull_mesh(paths[n]) for n in names}


def _static_tris(mesh: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    h = ps.static_pose(mesh[0], mesh[1], LOAD_LB * LB_TO_KG / RHO_SEAWATER)
    return ps.pose_triangles(mesh[0], mesh[1], h, 0.0)


def _timed_fixed(
    args: argparse.Namespace, gpu: Gpu, tris: np.ndarray, speeds: np.ndarray, cfg: ps.PanelConfig, rho: float
) -> tuple[list[ps.PanelResult], dict[str, Any]]:
    gpu.reset()
    t0 = time.perf_counter()
    res = ps.solve_fixed(tris, speeds, cfg, rho=rho, backend=args.backend)
    wall = time.perf_counter() - t0
    info = {
        "wall_s": wall,
        "assembly_s": res[0].assembly_s,
        "solve_s": [r.solve_s for r in res],
        "pool_peak_MiB": gpu.pool_mib(),
        "process_MiB": gpu.process_mib(),
    }
    return res, info


def ladder(args: argparse.Namespace, gpu: Gpu, clock: Clock) -> None:
    """Fixed-pose differences of every finalist from the baseline on a family of station counts."""
    speeds = np.array(SPEEDS_KN) * ps.KN
    tris = {k: _static_tris(m) for k, m in _meshes(args, ("baseline",) + FINALISTS).items()}
    out: dict[str, Any] = {"speeds_kn": SPEEDS_KN, "load_lb": LOAD_LB, "precision": "double", "rows": []}
    last_wall = 0.0
    for nx in args.nx or LADDER_NX:
        cfg = _cfg(args, nx)
        row: dict[str, Any] = {"nx_hull": nx, "ny_fs": cfg.ny_fs, "panels": cfg.n_panels()}
        if clock.left() < 6.0 * last_wall:  # four hulls on a grid at least as costly as the last
            row["skipped"] = "time budget"
        else:
            try:
                rw, timing = {}, {}
                for k, t in tris.items():
                    res, timing[k] = _timed_fixed(args, gpu, t, speeds, cfg, RHO_SEAWATER)
                    rw[k] = [r.rw_N for r in res]
                    if k == "baseline":
                        row["baseline_fz_dyn_N"] = [r.fz_dyn_N for r in res]
                        row["baseline_my_dyn_Nm"] = [r.my_dyn_Nm for r in res]
                    del res
                row["rw_N"] = rw
                row["difference_N"] = {k: [a - b for a, b in zip(v, rw["baseline"])] for k, v in rw.items() if k != "baseline"}
                row["timing"] = timing
                last_wall = max(t["wall_s"] for t in timing.values())
            except Exception as exc:  # noqa: BLE001 - only out-of-memory is a result; the rest re-raise
                if not gpu.is_oom(exc):
                    raise
                row["failed"] = f"out of GPU memory: {exc}"
                gpu.reset()
        out["rows"].append(row)
        print(json.dumps({k: row.get(k) for k in ("nx_hull", "panels", "difference_N", "skipped", "failed")}), flush=True)
        if "timing" in row:
            print("  baseline", [round(x, 4) for x in row["rw_N"]["baseline"]], row["timing"]["baseline"], flush=True)
        _write(args, gpu, "ladder", out)


def thin(args: argparse.Namespace, gpu: Gpu, clock: Clock) -> None:
    """Uniform-stream linearisation on a thin Wigley hull against Michell's integral."""
    beam, draft = 0.005, 0.0625
    v, f = ps.wigley_mesh(beam_ratio=beam)
    tris = ps.pose_triangles(v, f, 0.0)
    fns = (0.25, 0.30, 0.35, 0.40, 0.45)
    speeds = np.array(fns) * math.sqrt(ps.G)
    xs = np.linspace(-0.5, 0.5, 801)
    zs = np.linspace(-draft, 0.0, 161)
    yy = 0.5 * beam * (1.0 - (2.0 * xs[:, None]) ** 2) * (1.0 - (zs[None, :] / draft) ** 2)
    michell = _michell_fast_curve(xs, zs, yy, speeds, RHO_FRESH, g=ps.G)
    out: dict[str, Any] = {"beam_over_length": beam, "froude": fns, "linearisation": "kelvin", "rows": []}
    last_wall = 0.0
    for nx, ny, nz in [(nx, round(0.6 * nx), 8) for nx in args.nx] if args.nx else THIN_CASES:
        cfg = _cfg(args, nx, ny_fs=ny, nz_hull=nz, linearisation="kelvin")
        row: dict[str, Any] = {"nx_hull": nx, "ny_fs": ny, "nz_hull": nz, "panels": cfg.n_panels()}
        if clock.left() < 2.0 * last_wall:
            row["skipped"] = "time budget"
        else:
            try:
                res, row["timing"] = _timed_fixed(args, gpu, tris, speeds, cfg, RHO_FRESH)
                row["rw_over_michell"] = [r.rw_N / m for r, m in zip(res, michell)]
                last_wall = row["timing"]["wall_s"]
                del res
            except Exception as exc:  # noqa: BLE001
                if not gpu.is_oom(exc):
                    raise
                row["failed"] = f"out of GPU memory: {exc}"
                gpu.reset()
        out["rows"].append(row)
        print("thin", row["nx_hull"], ny, nz, row["panels"], [round(x, 4) for x in row.get("rw_over_michell", [])],
              row.get("skipped") or row.get("failed") or row["timing"], flush=True)
        _write(args, gpu, "thin-ship", out)


def precision(args: argparse.Namespace, gpu: Gpu, clock: Clock) -> None:
    """Single against double precision at a fine grid: resistance and paired difference."""
    speeds = np.array(SPEEDS_KN) * ps.KN
    tris = {k: _static_tris(m) for k, m in _meshes(args, ("baseline", FINALISTS[0])).items()}
    out: dict[str, Any] = {"speeds_kn": SPEEDS_KN, "candidate": FINALISTS[0], "rows": []}
    for nx in args.nx or (100,):
        row: dict[str, Any] = {"nx_hull": nx, "panels": _cfg(args, nx).n_panels()}
        try:
            for prec in ("double", "single"):
                cfg = _cfg(args, nx, precision=prec)
                b, tb = _timed_fixed(args, gpu, tris["baseline"], speeds, cfg, RHO_SEAWATER)
                c, _ = _timed_fixed(args, gpu, tris[FINALISTS[0]], speeds, cfg, RHO_SEAWATER)
                row[prec] = {
                    "baseline_rw_N": [r.rw_N for r in b],
                    "difference_N": [y.rw_N - x.rw_N for x, y in zip(b, c)],
                    "timing": tb,
                }
                del b, c
            row["single_minus_double_baseline_N"] = [x - y for x, y in zip(row["single"]["baseline_rw_N"], row["double"]["baseline_rw_N"])]
            row["single_minus_double_difference_N"] = [x - y for x, y in zip(row["single"]["difference_N"], row["double"]["difference_N"])]
        except Exception as exc:  # noqa: BLE001
            if not gpu.is_oom(exc):
                raise
            row["failed"] = f"out of GPU memory: {exc}"
            gpu.reset()
        out["rows"].append(row)
        print("precision", nx, row.get("single_minus_double_baseline_N"), row.get("single_minus_double_difference_N"), row.get("failed"), flush=True)
        _write(args, gpu, "precision", out)


def free(args: argparse.Namespace, gpu: Gpu, clock: Clock) -> None:
    """Hulls free to sink and trim, one (grid, hull, speed) at a time while the budget lasts."""
    meshes = _meshes(args, ("baseline",) + FINALISTS)
    volume = LOAD_LB * LB_TO_KG / RHO_SEAWATER
    out: dict[str, Any] = {"speeds_kn": FREE_SPEEDS_KN, "load_lb": LOAD_LB, "kg_m": KG_M, "runs": [], "skipped": []}
    per_solve: dict[int, float] = {}  # seconds per flow solution, measured on this run
    done: set[tuple[int, str]] = set()
    plan = [(nx, ("baseline",) + FINALISTS) for nx in args.nx] if args.nx else FREE_PLAN
    for nx, names in plan:
        cfg = _cfg(args, nx)
        for name in names:
            if (nx, name) in done:
                continue
            # seven flow solutions per speed is what the coarse grids needed; without a measured
            # time on this grid, scale the nearest measured one by the panel count squared
            n = sum(cfg.n_panels())
            if nx in per_solve:
                est = per_solve[nx]
            elif per_solve:
                ref = min(per_solve, key=lambda k: abs(k - nx))
                est = per_solve[ref] * (n / sum(_cfg(args, ref).n_panels())) ** 2
            else:
                est = 0.0
            need = est * 7.5 * len(FREE_SPEEDS_KN)
            if clock.left() < need:
                out["skipped"].append({"nx_hull": nx, "hull": name, "reason": "time budget", "needed_s": need, "left_s": clock.left()})
                print("free skipped", nx, name, f"needs {need:.0f} s, {clock.left():.0f} s left", flush=True)
                _write(args, gpu, "free", out)
                continue
            vcg = float(meshes[name][0][:, 2].min()) + KG_M
            gpu.reset()
            t0 = time.perf_counter()
            try:
                res = [
                    ps.solve_free(*meshes[name], float(kn) * ps.KN, cfg, volume_m3=volume, vcg_m=vcg,
                                  rho=RHO_SEAWATER, backend=args.backend)
                    for kn in FREE_SPEEDS_KN
                ]
            except Exception as exc:  # noqa: BLE001
                if not gpu.is_oom(exc):
                    raise
                out["skipped"].append({"nx_hull": nx, "hull": name, "reason": f"out of GPU memory: {exc}"})
                gpu.reset()
                _write(args, gpu, "free", out)
                continue
            wall = time.perf_counter() - t0
            per_solve[nx] = wall / sum(r.pose_iterations or 1 for r in res)
            done.add((nx, name))
            out["runs"].append(
                {"nx_hull": nx, "ny_fs": cfg.ny_fs, "panels": cfg.n_panels(), "hull": name, "wall_s": wall,
                 "pool_peak_MiB": gpu.pool_mib(), "results": [r.to_dict() for r in res]}
            )
            print("free", nx, name, [round(r.rw_N, 4) for r in res],
                  "sinkage mm", [round(1e3 * r.sinkage_m, 3) for r in res],
                  "trim deg", [round(math.degrees(r.trim_change_rad), 4) for r in res],
                  "iterations", [r.pose_iterations for r in res], "converged", [r.pose_converged for r in res],
                  f"{wall:.0f} s", flush=True)
            _write(args, gpu, "free", out)
    by = {(r["nx_hull"], r["hull"]): r for r in out["runs"]}
    out["difference_N"] = [
        {"nx_hull": nx, "hull": name,
         "difference_N": [c["rw_N"] - b["rw_N"] for b, c in zip(by[nx, "baseline"]["results"], r["results"])],
         "trim_difference_deg": [math.degrees(c["trim_rad"] - b["trim_rad"]) for b, c in zip(by[nx, "baseline"]["results"], r["results"])]}
        for (nx, name), r in by.items()
        if name != "baseline" and (nx, "baseline") in by
    ]
    for d in out["difference_N"]:
        print("free difference", d, flush=True)
    _write(args, gpu, "free", out)


def timing(args: argparse.Namespace, gpu: Gpu, clock: Clock) -> None:
    """One fixed-pose baseline solve per grid: the figure to set beside the other backend's."""
    speeds = np.array(SPEEDS_KN) * ps.KN
    tris = _static_tris(_meshes(args, ("baseline",))["baseline"])
    out: dict[str, Any] = {"speeds_kn": SPEEDS_KN, "hull": "baseline", "rows": []}
    for nx in args.nx or (100,):
        cfg = _cfg(args, nx)
        load0 = list(os.getloadavg())
        res, info = _timed_fixed(args, gpu, tris, speeds, cfg, RHO_SEAWATER)
        out["rows"].append({"nx_hull": nx, "panels": cfg.n_panels(), "rw_N": [r.rw_N for r in res], "timing": info,
                            "load_average_before": load0, "load_average_after": list(os.getloadavg())})
        print("timing", nx, cfg.n_panels(), [r.rw_N for r in res], info, flush=True)
        _write(args, gpu, "timing", out)


# Near-hull row width, source height and row count, each changed from the solver's defaults
VARIANTS: dict[str, dict[str, Any]] = {
    "nominal": {},
    "first_row 0.5625": {"first_row": 0.5625},
    "first_row 0.375": {"first_row": 0.375},
    "fs_raise 0.55": {"fs_raise": 0.55},
    "first_row 0.375, fs_raise 0.55": {"first_row": 0.375, "fs_raise": 0.55},
    "rows 0.8 nx": {"rows": 0.8},
    "first_row 0.375, rows 0.8 nx": {"first_row": 0.375, "rows": 0.8},
}
PAIR = ("baseline", FINALISTS[0])
OTHERS = FINALISTS[1:]
SIX = tuple(VARIANTS)[:5]  # the variants on the ladder's own row count
# (stations, hulls, variants) in priority order
VARIANT_PLAN = (
    (80, PAIR, tuple(VARIANTS)),
    (100, PAIR, SIX),
    (100, PAIR, tuple(VARIANTS)[5:]),
    (50, PAIR + OTHERS, tuple(VARIANTS)),
    (90, PAIR, tuple(VARIANTS)),
    (80, OTHERS, tuple(VARIANTS)),
    (65, PAIR, tuple(VARIANTS)),
    (100, OTHERS, SIX),
    (110, PAIR, SIX[:4]),
)


def variants(args: argparse.Namespace, gpu: Gpu, clock: Clock) -> None:
    """Fixed-pose paired differences under changes of the near-hull row width, the source height
    and the row count, on several station counts: do the variants approach one value?"""
    speeds = np.array(SPEEDS_KN) * ps.KN
    tris = {k: _static_tris(m) for k, m in _meshes(args, PAIR + OTHERS).items()}
    out: dict[str, Any] = {"speeds_kn": SPEEDS_KN, "load_lb": LOAD_LB, "variants": VARIANTS, "cases": [], "not_run": []}
    cases: dict[tuple[int, str], dict[str, Any]] = {}
    per_pair2: float | None = None  # seconds per hull per panel squared, measured on this run

    def run(nx: int, vname: str, hulls: tuple[str, ...], retry: bool = False) -> None:
        nonlocal per_pair2
        kw = dict(VARIANTS[vname])
        rows = kw.pop("rows", 0.6)
        cfg = _cfg(args, nx, ny_fs=round(rows * nx), **kw)
        n = sum(cfg.n_panels())
        case = cases.get((nx, vname))
        if case is None:
            case = {"nx_hull": nx, "ny_fs": cfg.ny_fs, "variant": vname, "panels": cfg.n_panels(),
                    "rw_N": {}, "rw_waterline_N": {}, "wall_s": {}}
            cases[nx, vname] = case
            out["cases"].append(case)
        for name in hulls:
            if name in case["rw_N"]:
                continue
            need = 0.0 if per_pair2 is None else 1.5 * per_pair2 * n * n
            if clock.left() < need:
                out["not_run"].append({"nx_hull": nx, "variant": vname, "hull": name, "reason": "time budget"})
                continue
            t0 = time.perf_counter()
            try:
                res = ps.solve_fixed(tris[name], speeds, cfg, rho=RHO_SEAWATER, backend=args.backend)
            except Exception as exc:  # noqa: BLE001
                if not gpu.is_oom(exc):
                    raise
                gpu.reset()
                out["not_run"].append({"nx_hull": nx, "variant": vname, "hull": name, "retry": retry,
                                       "reason": f"out of GPU memory: {exc}"})
                print("variants out of memory", nx, vname, name, flush=True)
                continue
            case["wall_s"][name] = time.perf_counter() - t0
            if n > 5000:
                per_pair2 = max(per_pair2 or 0.0, case["wall_s"][name] / (n * n))
            case["rw_N"][name] = [r.rw_N for r in res]
            case["rw_waterline_N"][name] = [r.rw_waterline_N for r in res]
            del res
        if "baseline" in case["rw_N"]:
            b = case["rw_N"]["baseline"]
            case["difference_N"] = {k: [x - y for x, y in zip(v, b)] for k, v in case["rw_N"].items() if k != "baseline"}
            print("variants", nx, cfg.ny_fs, vname, case["panels"],
                  {k: [round(x, 5) for x in v] for k, v in case["difference_N"].items()}, flush=True)
        _write(args, gpu, "variants", out)

    plan = [(nx, PAIR + OTHERS, tuple(VARIANTS)) for nx in args.nx] if args.nx else VARIANT_PLAN
    for nx, hulls, names in plan:
        for vname in names:
            run(nx, vname, hulls)
    # a case that ran out of memory while another process held the card gets one more attempt
    for miss in [m for m in out["not_run"] if m["reason"].startswith("out of GPU memory") and not m.get("retry")]:
        run(miss["nx_hull"], miss["variant"], (miss["hull"],), retry=True)
    _write(args, gpu, "variants", out)


STAGES = {"ladder": ladder, "thin": thin, "precision": precision, "free": free, "timing": timing, "variants": variants}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    ap.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "panel-solver" / "data")
    ap.add_argument("--prefix", default="fine-", help="file name prefix of the JSON written")
    ap.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    ap.add_argument("--nx", type=int, action="append", help="station counts, replacing the stage's own list")
    ap.add_argument("--budget-s", type=float, default=None)
    ap.add_argument("--pair-budget", type=int, default=None,
                    help="point-panel pairs per assembly chunk (default 2e6 on cupy, the solver's on numpy)")
    args = ap.parse_args()
    if args.pair_budget is None:
        args.pair_budget = 2_000_000 if args.backend == "cupy" else ps.PanelConfig().pair_budget
    args.out.mkdir(parents=True, exist_ok=True)
    gpu = Gpu(args.backend)
    clock = Clock(args.budget_s)
    if gpu.xp is not None:  # first use compiles the kernels; keep that out of the timings
        v, f = ps.wigley_mesh()
        ps.solve_fixed(ps.pose_triangles(v, f, 0.0), [1.0], ps.PanelConfig(nx_hull=16, ny_fs=8, nz_hull=4), backend="cupy")
    for name in args.stages:
        t0 = time.perf_counter()
        STAGES[name](args, gpu, clock)
        print(f"stage {name} done in {time.perf_counter() - t0:.0f} s; {clock.left():.0f} s of budget left", flush=True)


if __name__ == "__main__":
    main()
