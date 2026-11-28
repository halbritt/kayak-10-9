#!/usr/bin/env python3
"""Write rsi/panel_wave_reference.json: baseline v0.2.3 panel-solver and Michell wave reference.

Uses the 3-grid level-pose Dawson panel family ``((45, 27, 8), (50, 30, 8), (55, 33, 8))``
in single precision and leaves ``data/kayak-10-9-specs.json`` untouched.

Usage:
    python scripts/build_rsi_panel_wave_reference.py [--backend auto|cupy|numpy]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.baseline import (  # noqa: E402
    PANEL_WAVE_REFERENCE_FILE,
    PanelWaveReference,
    load_baseline_reference,
    panel_config_defaults,
    panel_solver_sha256,
)
from rsi.evaluator import (  # noqa: E402
    PANEL_WAVE_GRID_FAMILY,
    PANEL_WAVE_LINEARISATION,
    PANEL_WAVE_PRECISION,
    RESISTANCE_SPEEDS_KN,
    SPEED_LOADS_LB,
    complete_deferred_panel_wave,
    prepare_tier0_tier1_cpu,
    solve_panel_wave_curve,
)
from rsi.panel_hull import hull_mesh  # noqa: E402
from rsi.panel_solver import KN  # noqa: E402

VERSION = 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="auto", choices=("auto", "cupy", "numpy"))
    args = ap.parse_args()

    uncal = load_baseline_reference(ROOT, with_panel_wave_reference=False)
    V, F = hull_mesh(ROOT / "hull" / "kayak-10-9.hull.json")
    u_arr = RESISTANCE_SPEEDS_KN * KN

    panel_rw: dict[int, tuple[float, ...]] = {}
    backends_used: set[str] = set()
    for lb in SPEED_LOADS_LB:
        rw_p, bk = solve_panel_wave_curve(
            V,
            F,
            float(lb),
            u_arr,
            grid_family=PANEL_WAVE_GRID_FAMILY,
            precision=PANEL_WAVE_PRECISION,
            linearisation=PANEL_WAVE_LINEARISATION,
            backend=args.backend,
        )
        panel_rw[lb] = tuple(float(x) for x in rw_p)
        backends_used.update(bk)

    pw_ref = PanelWaveReference(
        version=VERSION,
        grid_family=PANEL_WAVE_GRID_FAMILY,
        precision=PANEL_WAVE_PRECISION,
        linearisation=PANEL_WAVE_LINEARISATION,
        speeds_kn=tuple(float(v) for v in RESISTANCE_SPEEDS_KN),
        baseline_panel_rw_N=panel_rw,
        baseline_michell_rw_N={},
        michell_kn_225=dict(uncal.kn_225),
        michell_kn_225_exact=dict(uncal.kn_225_exact or uncal.kn_225),
    )
    with tempfile.TemporaryDirectory() as tmp:
        partial, state = prepare_tier0_tier1_cpu(uncal.hull, Path(tmp), replace(uncal, panel_wave=pw_ref), defer_panel_wave=True)
        assert state is not None and partial.michell_rw_225_N is not None
        _, _, rf_by_load = state
        from rsi.evaluator import _ensure_cad_on_path, _michell_fast_curve  # noqa: E402
        _ensure_cad_on_path()
        from hydro import G, RHO_SEAWATER, float_at, half_breadth_grid  # noqa: E402
        cg = np.array([0.0, 0.0, float(V[:, 2].min()) + 0.25])
        mic_by_load: dict[str, list[float]] = {}
        for lb in SPEED_LOADS_LB:
            r_up = float_at(V, F, lb * 0.45359237, RHO_SEAWATER, cg, free_trim=False)
            xs, zs, Y = half_breadth_grid(r_up["tris"], 401, 81)
            mic_by_load[str(lb)] = [float(x) for x in _michell_fast_curve(xs, zs, Y, u_arr, RHO_SEAWATER, g=G, ntheta=4000)]
        full = complete_deferred_panel_wave(partial, V, F, rf_by_load, replace(uncal, panel_wave=pw_ref), panel_backend=args.backend)

    assert full.valid and full.kn_225 and full.kn_225_exact and full.kn_265 and full.kn_265_exact and full.kn_230 and full.kn_230_exact
    from rsi.loop import _panel_solver_sha  # noqa: E402

    payload = {
        "version": VERSION,
        "design_hash": uncal.design_hash,
        "panel_solver_git_sha": _panel_solver_sha(ROOT),
        "panel_solver_sha256": panel_solver_sha256(),
        "panel_config_defaults": panel_config_defaults(),
        "mesh_resolution": [441, 129],
        "grid_family": [list(g) for g in PANEL_WAVE_GRID_FAMILY],
        "precision": PANEL_WAVE_PRECISION,
        "linearisation": PANEL_WAVE_LINEARISATION,
        "backends_used": sorted(backends_used),
        "speeds_kn": [float(v) for v in RESISTANCE_SPEEDS_KN],
        "baseline_panel_rw_N": {str(lb): list(vals) for lb, vals in sorted(panel_rw.items())},
        "baseline_michell_rw_N": mic_by_load,
        "michell_kn_225": {str(k): v for k, v in sorted(uncal.kn_225.items())},
        "michell_kn_225_exact": {str(k): v for k, v in sorted((uncal.kn_225_exact or uncal.kn_225).items())},
        "baseline_kn_225": {str(k): v for k, v in sorted(full.kn_225.items())},
        "baseline_kn_225_exact": {str(k): v for k, v in sorted(full.kn_225_exact.items())},
        "baseline_kn_265": {str(k): v for k, v in sorted(full.kn_265.items())},
        "baseline_kn_265_exact": {str(k): v for k, v in sorted(full.kn_265_exact.items())},
        "baseline_kn_230": {str(k): v for k, v in sorted(full.kn_230.items())},
        "baseline_kn_230_exact": {str(k): v for k, v in sorted(full.kn_230_exact.items())},
    }
    out_path = ROOT / "rsi" / PANEL_WAVE_REFERENCE_FILE
    out_path.write_text(json.dumps(payload, indent=1) + "\n")
    print("WROTE", out_path, "backends=", sorted(backends_used))
    print("  225 lb panel kn:", full.kn_225_exact)
    print("  225 lb michell kn:", uncal.kn_225_exact)


if __name__ == "__main__":
    main()
