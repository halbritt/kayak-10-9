"""Time-resolved longitudinal hull force bins via OpenFOAM v2606 ``binField``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def hull_force_bins_dict(
    *,
    name: str = "hullForceBins",
    force_field: str = "force",
    patch: str = "hull",
    x_min: float = -2.75,
    x_max: float = 2.75,
    n_bin: int = 22,
    cofr: tuple[float, float, float] | list[float] = (0.03946, 0.0, 0.16136),
) -> str:
    """Return an OpenFOAM ``binField`` dictionary binning patch forces along x."""
    return (
        f"    {name}\n    {{\n"
        "        type                    binField;\n"
        "        libs                    (fieldFunctionObjects);\n"
        "        binModel                singleDirectionUniformBin;\n"
        f"        fields                  ({force_field});\n"
        f"        patches                 ({patch});\n"
        "        decomposePatchValues    true;\n"
        f"        CofR                    ({float(cofr[0]):g} {float(cofr[1]):g} {float(cofr[2]):g});\n"
        "        executeControl          timeStep;\n"
        "        executeInterval         1;\n"
        "        writeControl            timeStep;\n"
        "        writeInterval           1;\n"
        "        binData\n        {\n"
        f"            nBin        {int(n_bin)};\n"
        f"            min         {float(x_min):g};\n"
        f"            max         {float(x_max):g};\n"
        "            direction   (1 0 0);\n"
        "            cumulative  false;\n"
        "        }\n"
        "    }\n"
    )


def enable_binned_hull_forces(
    control_dict_text: str,
    *,
    x_min: float = -2.75,
    x_max: float = 2.75,
    n_bin: int = 22,
    cofr: tuple[float, float, float] | list[float] = (0.03946, 0.0, 0.16136),
) -> str:
    """Configure ``forces`` as an in-memory field producer and add ``hullForceBins``."""
    out, n = re.subn(
        r"(forces\s*\{\n[^}]*?)(\n\s*\})",
        r"\1\n        useNamePrefix false; executeControl timeStep; executeInterval 1;\2",
        control_dict_text,
        count=1,
    )
    if n != 1:
        raise ValueError("controlDict does not contain the expected forces functionObject block")
    bins_block = hull_force_bins_dict(x_min=x_min, x_max=x_max, n_bin=n_bin, cofr=cofr)
    out, n2 = re.subn(r"\n\}\s*$", "\n" + bins_block + "}\n", out, count=1)
    if n2 != 1:
        raise ValueError("controlDict does not end with functions closing brace")
    return out


def parse_force_bin_dat(path: Path, *, full_hull: bool = True) -> dict[str, Any]:
    """Parse ``postProcessing/<FO>/<time>/forceBin.dat`` into per-bin pressure/viscous arrays."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    n_bin, x_start, x_end = None, None, None
    steps, normal_x, tang_x, total_x = [], [], [], []
    scale = 2.0 if full_hull else 1.0
    num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            if m := re.match(r"^#\s*bins\s*:\s*(\d+)", s):
                n_bin = int(m.group(1))
            elif m := re.match(r"^#\s*start\s*:\s*(\S+)", s):
                x_start = float(m.group(1))
            elif m := re.match(r"^#\s*end\s*:\s*(\S+)", s):
                x_end = float(m.group(1))
            continue
        vals = [float(v) for v in num_re.findall(s)]
        if n_bin is None or len(vals) < 1 + 12 * n_bin:
            raise ValueError(f"Malformed data row in {path}: expected {1 + 12 * (n_bin or 0)} floats")
        # Each bin with cellZones=() & decomposePatchValues=true writes 4 vectors (12 floats):
        # total(3), internal(3), normal(3), tangential(3).
        arr = np.asarray(vals[1 : 1 + 12 * n_bin], dtype=float).reshape(n_bin, 4, 3)
        steps.append(int(round(vals[0])))
        total_x.append(scale * arr[:, 0, 0])
        normal_x.append(scale * arr[:, 2, 0])
        tang_x.append(scale * arr[:, 3, 0])
    if n_bin is None or not steps:
        raise ValueError(f"No bin data parsed from {path}")
    edges = np.linspace(float(x_start), float(x_end), n_bin + 1)
    return {
        "steps": np.asarray(steps, dtype=int),
        "bin_edges_x": edges,
        "total_x": np.asarray(total_x, dtype=float),
        "normal_x": np.asarray(normal_x, dtype=float),
        "tangential_x": np.asarray(tang_x, dtype=float),
    }
