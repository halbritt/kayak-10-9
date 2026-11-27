"""Validation runs of the free-surface panel solver; writes the JSON behind docs/panel-solver/README.md.

Usage (from the checkout, with its venv; keep the thread count to what the host can spare):

    OMP_NUM_THREADS=4 python scripts/panel_validation.py STUDY [STUDY ...] [--out DIR] [--campaign DIR]

Studies: thin-ship, wigley-fixed, wigley-free, wigley-convergence, paired, finalists, ensemble,
precision. ``paired`` also holds the kayak baseline's own convergence. Each writes
``DIR/<study>.json`` (default ``docs/panel-solver/data``). The reference numbers the document
compares against are cited there, not here: this script only computes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi import panel_solver as ps  # noqa: E402
from rsi.evaluator import LB_TO_KG, RHO_SEAWATER, _michell_fast_curve  # noqa: E402
from rsi.panel_hull import evaluate_mesh, hull_mesh, tier1_parts  # noqa: E402

DEFAULT_CAMPAIGN = Path.home() / ".cache/kayak-10-9/campaigns/k109_rsi_v7_localfriction"
BASELINE = ROOT / "hull" / "kayak-10-9.hull.json"
FINALISTS = ("iter137_cand004", "iter139_cand009", "iter139_cand002")
KAYAK_SPEEDS_KN = (5.5, 6.2)  # the grid studies
REPORT_SPEEDS_KN = (5.0, 5.5, 6.2)  # the finalist tables: 5.0 to 5.7 kn is the 60 to 90 W range
LOAD_LB = 225.0
RHO_FRESH = 1000.0

# Reference grids: the solver's defaults, with more stations for the Wigley hull because its
# lowest Froude numbers have shorter waves. The studies vary one thing at a time about these.
KAYAK: dict[str, Any] = {}
WIGLEY: dict[str, Any] = dict(nx_hull=60, ny_fs=36, nz_hull=10)
# Froude numbers of the published tables the document compares with
WIGLEY_FN = (0.25, 0.266, 0.267, 0.289, 0.30, 0.313, 0.316, 0.35, 0.354, 0.40, 0.408, 0.45)


def _stamp() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "threads": ps.host_threads(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _wigley_michell(speeds: np.ndarray, beam: float = 0.1, draft: float = 0.0625) -> np.ndarray:
    xs = np.linspace(-0.5, 0.5, 801)
    zs = np.linspace(-draft, 0.0, 161)
    yy = 0.5 * beam * (1.0 - (2.0 * xs[:, None]) ** 2) * (1.0 - (zs[None, :] / draft) ** 2)
    return _michell_fast_curve(xs, zs, yy, speeds, RHO_FRESH, g=ps.G)


def _row(r: ps.PanelResult, rho: float) -> dict[str, Any]:
    q = 0.5 * rho * r.speed_mps**2 * r.wetted_area_m2
    d = r.to_dict()
    d["cw_waterline"] = r.rw_waterline_N / q
    d["cw_farfield"] = None if r.rw_farfield_N is None else r.rw_farfield_N / q
    d["cw_pressure_linear"] = r.rw_pressure_linear_N / q
    return d


def thin_ship(_: argparse.Namespace) -> dict[str, Any]:
    """Uniform-stream linearisation on a thin Wigley hull against Michell's integral."""
    beam = 0.005
    v, f = ps.wigley_mesh(beam_ratio=beam)
    tris = ps.pose_triangles(v, f, 0.0)
    speeds = np.array([0.25, 0.30, 0.35, 0.40, 0.45]) * math.sqrt(ps.G)
    michell = _wigley_michell(speeds, beam=beam)
    base = dict(nx_hull=40, ny_fs=24, linearisation="kelvin")
    cases = [
        {},
        dict(nx_hull=60, ny_fs=36),
        dict(nx_hull=80, ny_fs=48),
        dict(nx_hull=60, ny_fs=36, nz_hull=16),
        dict(nx_hull=60, ny_fs=36, end_cluster=0.0),
        dict(nx_hull=60, ny_fs=48, half_width=0.8, upstream=0.8),
    ]
    rows = []
    for kw in cases:
        cfg = ps.PanelConfig(**{**base, **kw})
        res = ps.solve_fixed(tris, speeds, cfg, rho=RHO_FRESH)
        rows.append(
            {
                "change": kw,
                "panels": cfg.n_panels(),
                "froude": [r.froude for r in res],
                "rw_over_michell": [r.rw_N / m for r, m in zip(res, michell)],
                "assembly_s": res[0].assembly_s,
            }
        )
        print(kw, cfg.n_panels(), [round(x, 3) for x in rows[-1]["rw_over_michell"]], flush=True)
    return {"beam_over_length": beam, "base": base, "rows": rows}


def wigley_fixed(_: argparse.Namespace) -> dict[str, Any]:
    """Wigley hull held at its design pose: both linearisations, with the control-surface estimate."""
    v, f = ps.wigley_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    speeds = np.array(WIGLEY_FN) * math.sqrt(ps.G)
    michell = _wigley_michell(speeds)
    out: dict[str, Any] = {"config": WIGLEY, "michell_cw": None, "runs": {}}
    for lin in ("dawson", "kelvin"):
        cfg = ps.PanelConfig(**{**WIGLEY, "linearisation": lin})
        res = ps.solve_fixed(tris, speeds, cfg, rho=RHO_FRESH)
        out["runs"][lin] = [_row(r, RHO_FRESH) for r in res]
        if out["michell_cw"] is None:
            out["michell_cw"] = [
                float(m / (0.5 * RHO_FRESH * r.speed_mps**2 * r.wetted_area_m2)) for m, r in zip(michell, res)
            ]
            out["wetted_area_over_L2"] = res[0].wetted_area_m2
        for r in res:
            print(lin, f"Fn {r.froude:.3f} Cw {r.cw * 1e3:.3f}e-3", flush=True)
    # the control-surface estimate needs a longer and wider free surface
    ff = {**WIGLEY, "downstream": 1.0, "half_width": 0.8, "ny_fs": 48, "farfield": True, "ff_side": 0.85}
    cfg = ps.PanelConfig(**ff)
    sel = np.array([0.30, 0.35, 0.40, 0.45]) * math.sqrt(ps.G)
    res = ps.solve_fixed(tris, sel, cfg, rho=RHO_FRESH)
    out["farfield_config"] = ff
    out["farfield"] = [_row(r, RHO_FRESH) for r in res]
    for r in res:
        q = 0.5 * RHO_FRESH * r.speed_mps**2 * r.wetted_area_m2
        print(f"farfield Fn {r.froude:.3f} pressure {r.cw * 1e3:.3f} control surface {r.rw_farfield_N / q * 1e3:.3f}", flush=True)
    return out


def wigley_free(_: argparse.Namespace) -> dict[str, Any]:
    """Wigley hull free to sink and trim: resistance, sinkage and trim."""
    v, f = ps.wigley_mesh()
    volume = ps.hydrostatics(ps.pose_triangles(v, f, 0.0))["vol"]
    cfg = ps.PanelConfig(**WIGLEY)
    rows = []
    for fn in (0.25, 0.30, 0.35, 0.40, 0.45):
        u = fn * math.sqrt(ps.G)
        # centre of gravity at the still waterplane; the towing force acts through it
        try:
            r = ps.solve_free(v, f, u, cfg, volume_m3=volume, vcg_m=0.0, rho=RHO_FRESH)
        except ValueError as exc:  # a pose the solver refuses is a result, not a crash
            rows.append({"froude": fn, "rejected": str(exc)})
            print(f"Fn {fn:.2f} rejected: {exc}", flush=True)
            continue
        if not r.pose_converged:
            rows.append({"froude": fn, "rejected": f"free pose did not converge in {r.pose_iterations} iterations"})
            continue
        d = _row(r, RHO_FRESH)
        d["sinkage_coefficient"] = 2.0 * ps.G * r.sinkage_m / u**2  # 2 g s / U^2, s over L = 1
        d["trim_percent_L"] = 100.0 * math.tan(r.trim_change_rad)  # (dA - dF) / L x 100, bow up +
        rows.append(d)
        print(
            f"Fn {fn:.2f} Cw {r.cw * 1e3:.3f}e-3 sinkage coeff {d['sinkage_coefficient']:.4f} "
            f"trim {d['trim_percent_L']:+.3f} % L, {r.pose_iterations} iterations",
            flush=True,
        )
    return {"config": WIGLEY, "rows": rows}


def _vary(tris: np.ndarray, speeds: np.ndarray, base: dict[str, Any], cases: list[dict[str, Any]], rho: float) -> list[dict[str, Any]]:
    rows = []
    for kw in cases:
        cfg = ps.PanelConfig(**{**base, **kw})
        res = ps.solve_fixed(tris, speeds, cfg, rho=rho)
        rows.append(
            {
                "change": kw,
                "panels": cfg.n_panels(),
                "rw_N": [r.rw_N for r in res],
                "cw": [r.cw for r in res],
                "rw_waterline_N": [r.rw_waterline_N for r in res],
                "fz_dyn_N": [r.fz_dyn_N for r in res],
                "my_dyn_Nm": [r.my_dyn_Nm for r in res],
                "assembly_s": res[0].assembly_s,
                "solve_s": res[0].solve_s,
            }
        )
        print(kw, cfg.n_panels(), [round(r.rw_N, 5) for r in res], [round(r.cw * 1e3, 4) for r in res], flush=True)
    return rows


def wigley_convergence(_: argparse.Namespace) -> dict[str, Any]:
    v, f = ps.wigley_mesh()
    tris = ps.pose_triangles(v, f, 0.0)
    fns = (0.30, 0.35, 0.40, 0.45)
    speeds = np.array(fns) * math.sqrt(ps.G)
    cases = [
        dict(nx_hull=40, ny_fs=24),
        dict(nx_hull=50, ny_fs=30),
        {},
        dict(nx_hull=80, ny_fs=48),
        dict(nz_hull=6),
        dict(nz_hull=16),
        dict(ny_fs=48),
        dict(half_width=0.8, ny_fs=48),
        dict(upstream=0.8),
        dict(downstream=1.0),
        dict(end_cluster=0.0),
        dict(end_cluster=0.5),
        dict(fs_raise=1.0),
        dict(fs_shift=0.25),
    ]
    return {"froude": fns, "base": WIGLEY, "rows": _vary(tris, speeds, WIGLEY, cases, RHO_FRESH)}


KAYAK_CASES: list[dict[str, Any]] = [
    dict(nx_hull=40, ny_fs=24),
    {},
    dict(nx_hull=60, ny_fs=36),
    dict(nx_hull=70, ny_fs=42),
    dict(nz_hull=6),
    dict(nz_hull=12),
    dict(ny_fs=24),
    dict(ny_fs=40),
    dict(half_width=0.8, ny_fs=40),
    dict(upstream=0.8),
    dict(downstream=1.0),
    dict(end_cluster=0.7),
    dict(end_cluster=0.9),
    dict(fs_raise=1.0),
    dict(fs_shift=0.25),
    dict(linearisation="kelvin"),
    dict(first_row=0.375, waterline_eta="extrapolated"),
]


def _kayak_tris(path: Path) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    mesh = hull_mesh(path)
    h = ps.static_pose(mesh[0], mesh[1], LOAD_LB * LB_TO_KG / RHO_SEAWATER)
    return mesh, ps.pose_triangles(mesh[0], mesh[1], h, 0.0)


def paired(args: argparse.Namespace) -> dict[str, Any]:
    """Baseline against the campaign's best hull: is the difference stable under the grid?"""
    cand_path = args.campaign / "candidates" / FINALISTS[0] / "candidate.hull.json"
    base_mesh, base_tris = _kayak_tris(BASELINE)
    cand_mesh, cand_tris = _kayak_tris(cand_path)
    speeds = np.array(KAYAK_SPEEDS_KN) * ps.KN
    rows = []
    for kw in KAYAK_CASES:
        cfg = ps.PanelConfig(**{**KAYAK, **kw})
        b = ps.solve_fixed(base_tris, speeds, cfg, rho=RHO_SEAWATER)
        c = ps.solve_fixed(cand_tris, speeds, cfg, rho=RHO_SEAWATER)
        rows.append(
            {
                "change": kw,
                "panels": cfg.n_panels(),
                "baseline_rw_N": [r.rw_N for r in b],
                "baseline_fz_dyn_N": [r.fz_dyn_N for r in b],
                "baseline_my_dyn_Nm": [r.my_dyn_Nm for r in b],
                "baseline_wetted_area_m2": b[0].wetted_area_m2,
                "assembly_s": b[0].assembly_s,
                "solve_s": b[0].solve_s,
                "candidate_rw_N": [r.rw_N for r in c],
                "difference_N": [y.rw_N - x.rw_N for x, y in zip(b, c)],
                "difference_waterline_N": [y.rw_waterline_N - x.rw_waterline_N for x, y in zip(b, c)],
            }
        )
        print(kw, cfg.n_panels(), "base", [round(r.rw_N, 4) for r in b], "diff", [round(x, 4) for x in rows[-1]["difference_N"]], flush=True)
    free_rows = []
    for kw in ({}, dict(nx_hull=60, ny_fs=36)):
        cfg = ps.PanelConfig(**{**KAYAK, **kw})
        b = evaluate_mesh(*base_mesh, LOAD_LB, speeds, cfg, free=True)
        c = evaluate_mesh(*cand_mesh, LOAD_LB, speeds, cfg, free=True)
        free_rows.append(
            {
                "change": kw,
                "baseline": [r.to_dict() for r in b],
                "candidate": [r.to_dict() for r in c],
                "difference_N": [y.rw_N - x.rw_N for x, y in zip(b, c)],
            }
        )
        print("free", kw, "base", [round(r.rw_N, 4) for r in b], "diff", [round(x, 4) for x in free_rows[-1]["difference_N"]], flush=True)
    return {
        "baseline": "hull/kayak-10-9.hull.json",
        "candidate": FINALISTS[0],
        "load_lb": LOAD_LB,
        "speeds_kn": KAYAK_SPEEDS_KN,
        "base": KAYAK,
        "fixed_pose": rows,
        "free": free_rows,
    }


def finalists(args: argparse.Namespace) -> dict[str, Any]:
    """Each finalist against the baseline: panel wave, Michell wave and friction differences."""
    speeds = np.array(REPORT_SPEEDS_KN) * ps.KN
    cfg = ps.PanelConfig(**{**KAYAK, "nx_hull": 60, "ny_fs": 36})
    base_mesh = hull_mesh(BASELINE)

    def one(mesh: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        t0 = time.perf_counter()
        free = evaluate_mesh(*mesh, LOAD_LB, speeds, cfg, free=True)
        t_free = time.perf_counter() - t0
        fixed = evaluate_mesh(*mesh, LOAD_LB, speeds, cfg, free=False)
        return {
            "free": [r.to_dict() for r in free],
            "fixed": [r.to_dict() for r in fixed],
            "tier1": tier1_parts(*mesh, LOAD_LB, speeds, baseline=base_mesh),
            "free_wall_s": t_free,
        }

    out: dict[str, Any] = {
        "config": cfg.__dict__ | {},
        "panels": cfg.n_panels(),
        "load_lb": LOAD_LB,
        "speeds_kn": REPORT_SPEEDS_KN,
        "hulls": {"baseline": one(base_mesh)},
    }
    print("baseline", [round(r["rw_N"], 4) for r in out["hulls"]["baseline"]["free"]], flush=True)
    for name in FINALISTS:
        mesh = hull_mesh(args.campaign / "candidates" / name / "candidate.hull.json")
        out["hulls"][name] = one(mesh)
        b, c = out["hulls"]["baseline"], out["hulls"][name]
        for i, kn in enumerate(REPORT_SPEEDS_KN):
            print(
                name,
                f"{kn} kn: panel free {c['free'][i]['rw_N'] - b['free'][i]['rw_N']:+.4f} "
                f"fixed {c['fixed'][i]['rw_N'] - b['fixed'][i]['rw_N']:+.4f} "
                f"Michell {c['tier1']['michell_N'][i] - b['tier1']['michell_N'][i]:+.4f} "
                f"friction {c['tier1']['friction_by_position_N'][i] - b['tier1']['friction_by_position_N'][i]:+.4f}",
                flush=True,
            )
    return out


ENSEMBLE_NX = (40, 45, 50, 55, 60, 65, 70)


def ensemble(args: argparse.Namespace) -> dict[str, Any]:
    """Fixed-pose differences of every finalist on a family of grids: mean and scatter.

    The sample spread describes this station and row family at a fixed pose.
    """
    speeds = np.array(REPORT_SPEEDS_KN) * ps.KN
    tris = {"baseline": _kayak_tris(BASELINE)[1]}
    for name in FINALISTS:
        tris[name] = _kayak_tris(args.campaign / "candidates" / name / "candidate.hull.json")[1]
    rw: dict[str, list[list[float]]] = {k: [] for k in tris}
    for nx in ENSEMBLE_NX:
        cfg = ps.PanelConfig(**{**KAYAK, "nx_hull": nx, "ny_fs": round(0.6 * nx)})
        for k, t in tris.items():
            rw[k].append([r.rw_N for r in ps.solve_fixed(t, speeds, cfg, rho=RHO_SEAWATER)])
        print(nx, {k: [round(a - b, 4) for a, b in zip(v[-1], rw["baseline"][-1])] for k, v in rw.items() if k != "baseline"}, flush=True)
    base = np.array(rw["baseline"])
    out: dict[str, Any] = {
        "nx_hull": ENSEMBLE_NX,
        "speeds_kn": REPORT_SPEEDS_KN,
        "load_lb": LOAD_LB,
        "baseline_rw_N": base.tolist(),
        "baseline_mean_N": base.mean(0).tolist(),
        "baseline_std_N": base.std(0, ddof=1).tolist(),
        "finalists": {},
    }
    for name in FINALISTS:
        d = np.array(rw[name]) - base
        out["finalists"][name] = {
            "difference_N": d.tolist(),
            "mean_N": d.mean(0).tolist(),
            "std_N": d.std(0, ddof=1).tolist(),
            "min_N": d.min(0).tolist(),
            "max_N": d.max(0).tolist(),
        }
        print(name, "mean", np.round(d.mean(0), 4), "std", np.round(d.std(0, ddof=1), 4), flush=True)
    return out


def precision(args: argparse.Namespace) -> dict[str, Any]:
    """Single against double precision on the reference grid: resistance and paired difference."""
    speeds = np.array(REPORT_SPEEDS_KN) * ps.KN
    base = _kayak_tris(BASELINE)[1]
    cand = _kayak_tris(args.campaign / "candidates" / FINALISTS[0] / "candidate.hull.json")[1]
    out: dict[str, Any] = {"speeds_kn": REPORT_SPEEDS_KN, "candidate": FINALISTS[0], "runs": {}}
    for prec in ("double", "single"):
        cfg = ps.PanelConfig(**{**KAYAK, "precision": prec})
        b = ps.solve_fixed(base, speeds, cfg, rho=RHO_SEAWATER)
        c = ps.solve_fixed(cand, speeds, cfg, rho=RHO_SEAWATER)
        out["runs"][prec] = {
            "panels": cfg.n_panels(),
            "baseline_rw_N": [r.rw_N for r in b],
            "candidate_rw_N": [r.rw_N for r in c],
            "difference_N": [y.rw_N - x.rw_N for x, y in zip(b, c)],
            "baseline_fz_dyn_N": [r.fz_dyn_N for r in b],
            "assembly_s": b[0].assembly_s,
            "solve_s": b[0].solve_s,
        }
        print(prec, out["runs"][prec]["baseline_rw_N"], out["runs"][prec]["difference_N"], flush=True)
    d, sg = out["runs"]["double"], out["runs"]["single"]
    out["single_minus_double_difference_N"] = [x - y for x, y in zip(sg["difference_N"], d["difference_N"])]
    out["single_minus_double_baseline_N"] = [x - y for x, y in zip(sg["baseline_rw_N"], d["baseline_rw_N"])]
    return out


STUDIES = {
    "thin-ship": thin_ship,
    "wigley-fixed": wigley_fixed,
    "wigley-free": wigley_free,
    "wigley-convergence": wigley_convergence,
    "paired": paired,
    "finalists": finalists,
    "ensemble": ensemble,
    "precision": precision,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("studies", nargs="+", choices=sorted(STUDIES))
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "panel-solver" / "data")
    ap.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.studies:
        t0 = time.perf_counter()
        result = STUDIES[name](args)
        result["provenance"] = _stamp() | {"wall_s": round(time.perf_counter() - t0, 1)}
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {args.out / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
