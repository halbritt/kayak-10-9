"""Panel-solver evaluation of a hull JSON, with the Tier 1 drag parts for comparison.

``rsi.panel_solver`` works on any triangle mesh. This module feeds it the same closed outer shell
the Tier 1 evaluator builds from a hull JSON, at the same load and level static trim, and returns
the wave resistance next to Tier 1's own parts: Michell thin-ship wave drag and ITTC-57 friction
charged by position (``rsi.evaluator``). It is a diagnostic. Nothing here feeds the gates, the
campaign loop or promotion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from rsi import panel_solver as ps
from rsi.evaluator import (
    KG_M,
    LB_TO_KG,
    RHO_SEAWATER,
    _ensure_cad_on_path,
    _michell_fast_curve,
    local_friction_ratio,
)

DEFAULT_MESH_RESOLUTION = (441, 129)  # the Tier 1 evaluator's shell resolution


def hull_mesh(
    hull_json: Path, mesh_resolution: tuple[int, int] = DEFAULT_MESH_RESOLUTION
) -> tuple[np.ndarray, np.ndarray]:
    """Closed outer shell (hull and deck) of a hull JSON as vertices and faces; bow at -x."""
    _ensure_cad_on_path()
    from deck_geometry import SculptedKayakSurface

    surf = SculptedKayakSurface(
        Path(hull_json), n_stations=mesh_resolution[0], n_hull_half=mesh_resolution[1]
    )
    closed = trimesh.util.concatenate(
        (surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False))
    )
    closed.merge_vertices(merge_tex=True, merge_norm=True)
    if not closed.is_watertight:
        raise ValueError(f"{hull_json}: the outer shell is not watertight")
    return np.asarray(closed.vertices, dtype=float), np.asarray(closed.faces, dtype=int)


def tier1_parts(
    vertices: np.ndarray,
    faces: np.ndarray,
    load_lb: float,
    speeds_mps: np.ndarray,
    *,
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Tier 1's drag parts at exact speeds: Michell wave drag and friction, as the evaluator has them.

    Friction is ITTC-57 on the static wetted area times the hull's local-friction ratio over the
    baseline's at the same load and speed. With no ``baseline`` mesh the hull is its own baseline
    and the friction is plain ITTC-57.
    """
    _ensure_cad_on_path()
    from hydro import G, NU_SEAWATER, float_at, half_breadth_grid, ittc57

    u = np.atleast_1d(np.asarray(speeds_mps, dtype=float))

    def static(v: np.ndarray, f: np.ndarray) -> dict[str, Any]:
        cg = np.array([0.0, 0.0, float(v[:, 2].min()) + KG_M])
        return float_at(v, f, load_lb * LB_TO_KG, RHO_SEAWATER, cg, free_trim=False)

    def ratio(r: dict[str, Any]) -> np.ndarray:
        lwl = float(np.ptp(r["wl_x"]))
        return np.array(
            [
                local_friction_ratio(r["tris"], float(x), NU_SEAWATER, float(min(r["wl_x"])), lwl)
                for x in u
            ]
        )

    r_up = static(vertices, faces)
    lwl = float(np.ptp(r_up["wl_x"]))
    xs, zs, yy = half_breadth_grid(r_up["tris"], 401, 81)
    rw = _michell_fast_curve(xs, zs, yy, u, RHO_SEAWATER, g=G, ntheta=4000)
    rf_plain = np.array(
        [ittc57(float(x), lwl, r_up["wet"], RHO_SEAWATER, NU_SEAWATER)[0] for x in u]
    )
    own = ratio(r_up)
    base = own if baseline is None else ratio(static(*baseline))
    return {
        "lwl_m": lwl,
        "wet_m2": float(r_up["wet"]),
        "h_m": float(r_up["h"]),
        "michell_N": [float(x) for x in rw],
        "friction_ittc57_N": [float(x) for x in rf_plain],
        "friction_by_position_N": [float(x) for x in rf_plain * own / base],
    }


def evaluate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    load_lb: float,
    speeds_mps: np.ndarray,
    cfg: ps.PanelConfig,
    *,
    free: bool = True,
    backend: str = "numpy",
) -> list[ps.PanelResult]:
    """Panel solve of one shell at one load: free to sink and trim, or held at its static pose.

    The static pose is level at the displacement of ``load_lb`` in sea water, as in Tier 1, so the
    longitudinal centre of gravity is over the centre of buoyancy at rest. The centre of gravity
    is ``rsi.evaluator.KG_M`` above the keel.
    """
    volume = load_lb * LB_TO_KG / RHO_SEAWATER
    u = np.atleast_1d(np.asarray(speeds_mps, dtype=float))
    if not free:
        h = ps.static_pose(vertices, faces, volume)
        tris = ps.pose_triangles(vertices, faces, h, 0.0)
        return ps.solve_fixed(tris, u, cfg, rho=RHO_SEAWATER, backend=backend, h=h)
    vcg = float(np.asarray(vertices)[:, 2].min()) + KG_M
    results = []
    for speed in u:
        res = ps.solve_free(
            vertices,
            faces,
            float(speed),
            cfg,
            volume_m3=volume,
            vcg_m=vcg,
            rho=RHO_SEAWATER,
            backend=backend,
        )
        if not res.pose_converged:
            raise RuntimeError(
                f"Free pose did not converge at {speed / ps.KN:g} kn after {res.pose_iterations} iterations"
            )
        results.append(res)
    return results


def evaluate_hull(
    hull_json: Path,
    speeds_kn: list[float],
    cfg: ps.PanelConfig,
    *,
    load_lb: float = 225.0,
    baseline_json: Path | None = None,
    free: bool = True,
    backend: str = "numpy",
    mesh_resolution: tuple[int, int] = DEFAULT_MESH_RESOLUTION,
) -> dict[str, Any]:
    """Panel wave resistance and Tier 1 parts of a hull, and its differences from a baseline.

    Differences are candidate minus baseline, in newtons, with both hulls solved on grids built
    by the same rule from the same settings.
    """
    u = np.array(speeds_kn, dtype=float) * ps.KN
    mesh = hull_mesh(Path(hull_json), mesh_resolution)
    base_mesh = None if baseline_json is None else hull_mesh(Path(baseline_json), mesh_resolution)

    def one(m: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        res = evaluate_mesh(m[0], m[1], load_lb, u, cfg, free=free, backend=backend)
        return {
            "panel": [r.to_dict() for r in res],
            "tier1": tier1_parts(m[0], m[1], load_lb, u, baseline=base_mesh),
        }

    out: dict[str, Any] = {
        "hull": str(hull_json),
        "load_lb": load_lb,
        "speeds_kn": [float(x) for x in speeds_kn],
        "free_to_sink_and_trim": free,
        "config": cfg.__dict__ | {},
        "candidate": one(mesh),
    }
    if base_mesh is not None:
        out["baseline_hull"] = str(baseline_json)
        out["baseline"] = one(base_mesh)
        c, b = out["candidate"], out["baseline"]
        out["difference_N"] = [
            {
                "speed_kn": float(kn),
                "panel_wave": c["panel"][i]["rw_N"] - b["panel"][i]["rw_N"],
                "michell_wave": c["tier1"]["michell_N"][i] - b["tier1"]["michell_N"][i],
                "friction_by_position": c["tier1"]["friction_by_position_N"][i]
                - b["tier1"]["friction_by_position_N"][i],
            }
            for i, kn in enumerate(speeds_kn)
        ]
    return out
