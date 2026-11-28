#!/usr/bin/env python3
"""Write data/rsi-friction-reference.json: the baseline hull's own local-friction ratios.

The RSI evaluator charges a candidate's friction relative to these (rsi.evaluator
local_friction_ratio), so the baseline keeps its published speeds. Run this again whenever
hull/kayak-10-9.hull.json changes; rsi.baseline refuses a file made for another hull.

Usage: python scripts/build_rsi_friction_reference.py [--work-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.baseline import FRICTION_REFERENCE_FILE, load_baseline_reference  # noqa: E402
from rsi.evaluator import RESISTANCE_SPEEDS_KN, evaluate_tier0_tier1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path)
    args = ap.parse_args()
    # Neither reference can be loaded after the hull changes; this one does not need the other.
    ref = load_baseline_reference(ROOT, with_friction_reference=False, with_panel_wave_reference=False)
    with tempfile.TemporaryDirectory() as tmp:
        metrics = evaluate_tier0_tier1(ref.hull, args.work_dir or Path(tmp), ref)
    if not metrics.valid or not metrics.friction_ratio:
        raise SystemExit(f"baseline evaluation failed: {metrics.error}")
    out = {
        "metadata": {
            "what": "local-friction ratio of the baseline hull per load (lb) and speed",
            "definition": "rsi.evaluator.local_friction_ratio",
        },
        "design_hash": ref.design_hash,
        "speeds_kn": [float(v) for v in RESISTANCE_SPEEDS_KN],
        "ratio": {str(lb): list(vals) for lb, vals in sorted(metrics.friction_ratio.items())},
    }
    path = ROOT / "data" / FRICTION_REFERENCE_FILE
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("WROTE", path)
    for lb, vals in out["ratio"].items():
        print(f"  {lb} lb: ratio {min(vals):.4f} .. {max(vals):.4f}")


if __name__ == "__main__":
    main()
