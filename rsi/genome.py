"""Bounded scalar design-space parameterization and Hull adapter for Kayak 10.9 RSI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from kayakgen.model.hull import Hull

_BASE_DECK_FB_FWD = 0.234
_BASE_DECK_FB_AFT = 0.115
_BASE_SHEER_H_FWD = 0.76
_BASE_SHEER_H_AFT = 0.88


@dataclass(frozen=True)
class ParameterSpec:
    """Descriptor for a single scalar design parameter."""

    name: str
    lower: float
    upper: float
    baseline: float
    scale: float
    group: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "baseline": self.baseline,
            "scale": self.scale,
            "group": self.group,
        }


_DEFAULT_PARAMS: tuple[ParameterSpec, ...] = (
    # Group: waterline (waterline_half_breadth.coefficients[0..3], a4/a5 delta-derived)
    ParameterSpec("wl_a0", 0.242, 0.262, 0.249, 0.005, "waterline"),
    ParameterSpec("wl_a1", 0.038, 0.058, 0.04482, 0.005, "waterline"),
    ParameterSpec("wl_a2", -0.232, -0.205, -0.21912, 0.006, "waterline"),
    ParameterSpec("wl_a3", -0.052, -0.028, -0.0394416, 0.006, "waterline"),
    # Group: draft (d0=0.103 fixed for Z_KEEL_SHIFT=0.1030 compatibility)
    ParameterSpec("draft_a1", -0.008, 0.008, 0.0, 0.003, "draft"),
    ParameterSpec("draft_a2", -0.060, -0.028, -0.04532, 0.006, "draft"),
    # Group: rocker (end rockers mapped to r2, r3; r0=r1=0 fixed)
    ParameterSpec("rocker_bow_end_m", 0.0, 0.028, 0.012, 0.004, "rocker"),
    ParameterSpec("rocker_stern_end_m", 0.0, 0.020, 0.004, 0.004, "rocker"),
    # Group: section (section_side_exponent & bilge_round_m ramps, disjoint xi)
    ParameterSpec("sec_side_fwd", 1.30, 3.20, 2.0, 0.25, "section"),
    ParameterSpec("sec_side_aft", 1.05, 2.20, 1.1958772290315158, 0.20, "section"),
    ParameterSpec("sec_side_fwd_xi", -0.75, -0.10, -0.4589222038716594, 0.10, "section"),
    ParameterSpec("sec_side_aft_xi", 0.05, 0.75, 0.42936653865876717, 0.10, "section"),
    ParameterSpec("bilge_fwd_m", 0.025, 0.13, 0.07, 0.015, "section"),
    ParameterSpec("bilge_aft_m", 0.0, 0.05, 0.0, 0.01, "section"),
    ParameterSpec("bilge_fwd_xi", -0.70, -0.12, -0.38, 0.08, "section"),
    ParameterSpec("bilge_aft_xi", -0.05, 0.45, 0.05, 0.08, "section"),
    # Group: topsides (sheer_plan, deck_crown_exponent, shoulder_round_m, deck_freeboard, sheer_height_frac)
    ParameterSpec("sheer_max_hb_m", 0.275, 0.292, 0.285, 0.004, "topsides"),
    ParameterSpec("sheer_max_xi", 0.25, 0.41, 0.39631995663129627, 0.03, "topsides"),
    ParameterSpec("sheer_fwd_exp", 1.40, 1.58, 1.4901655203838866, 0.04, "topsides"),
    ParameterSpec("sheer_aft_exp", 1.85, 2.06, 1.990139555287078, 0.05, "topsides"),
    ParameterSpec("crown_fwd", 2.05, 5.0, 2.692816171190041, 0.4, "topsides"),
    ParameterSpec("crown_aft", 4.5, 11.5, 9.729547268321408, 1.0, "topsides"),
    ParameterSpec("crown_fwd_xi", -0.65, -0.10, -0.35, 0.08, "topsides"),
    ParameterSpec("crown_aft_xi", 0.0, 0.60, 0.2, 0.08, "topsides"),
    ParameterSpec("shoulder_round_m", 0.025, 0.120, 0.06812556509299181, 0.015, "topsides"),
    ParameterSpec("deck_fb_fwd_scale", 0.88, 1.12, 1.0, 0.03, "topsides"),
    ParameterSpec("deck_fb_aft_scale", 0.88, 1.12, 1.0, 0.03, "topsides"),
    ParameterSpec("sheer_h_fwd_frac", 0.68, 0.84, 0.76, 0.02, "topsides"),
    ParameterSpec("sheer_h_aft_frac", 0.80, 0.915, 0.88, 0.02, "topsides"),
    # Group: fairing
    ParameterSpec("target_beam_m", 0.490, 0.512, 0.500456, 0.002, "fairing"),
)


@dataclass(frozen=True)
class DesignSpace:
    """Ordered collection of bounded design parameters."""

    params: tuple[ParameterSpec, ...]

    @classmethod
    def default(cls) -> DesignSpace:
        return cls(params=_DEFAULT_PARAMS)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def baseline_vector(self) -> list[float]:
        return [p.baseline for p in self.params]

    def index_by_name(self) -> dict[str, int]:
        return {p.name: i for i, p in enumerate(self.params)}

    def clamp(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != len(self.params):
            raise ValueError(f"Expected vector of length {len(self.params)}, got {len(vector)}")
        out: list[float] = []
        for spec, raw in zip(self.params, vector):
            val = float(raw)
            if not math.isfinite(val):
                raise ValueError(f"Non-finite value for parameter {spec.name}: {raw!r}")
            out.append(min(spec.upper, max(spec.lower, val)))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"params": [p.to_dict() for p in self.params]}


def _require_distribution_v2(hull: Hull) -> Any:
    if hull.geometry_kind != "distribution_v2" or hull.distribution_v2 is None:
        raise ValueError("Hull must have geometry_kind='distribution_v2' and distribution_v2 spec")
    return hull.distribution_v2


def vector_to_hull(
    vector: Sequence[float],
    base_hull: Hull,
    space: DesignSpace | None = None,
    *,
    clamp: bool = False,
) -> Hull:
    """Map a scalar design vector into a validated DistributionV2 Hull."""
    sp = DesignSpace.default() if space is None else space
    if len(vector) != len(sp.params):
        raise ValueError(f"Expected vector of length {len(sp.params)}, got {len(vector)}")
    vals = sp.clamp(vector) if clamp else [float(v) for v in vector]
    by_name = {spec.name: val for spec, val in zip(sp.params, vals)}

    _require_distribution_v2(base_hull)
    data = base_hull.model_dump()
    dv2 = data["distribution_v2"]

    # Waterline polynomial with exact end-closure preservation
    base_wl = list(dv2["waterline_half_breadth"]["coefficients"])
    a0 = by_name["wl_a0"]
    a1 = by_name["wl_a1"]
    a2 = by_name["wl_a2"]
    a3 = by_name["wl_a3"]
    if (a0, a1, a2, a3) == (base_wl[0], base_wl[1], base_wl[2], base_wl[3]):
        dv2["waterline_half_breadth"]["coefficients"] = base_wl
    else:
        a4 = base_wl[4] - ((a0 - base_wl[0]) + (a2 - base_wl[2]))
        a5 = base_wl[5] - ((a1 - base_wl[1]) + (a3 - base_wl[3]))
        dv2["waterline_half_breadth"]["coefficients"] = [a0, a1, a2, a3, a4, a5]

    # Draft profile (d0 fixed at base_d[0] = 0.103)
    base_d = list(dv2["draft_profile"]["coefficients"])
    dv2["draft_profile"]["coefficients"] = [base_d[0], by_name["draft_a1"], by_name["draft_a2"]]

    # Rocker polynomial from end rockers
    r_bow = by_name["rocker_bow_end_m"]
    r_stern = by_name["rocker_stern_end_m"]
    r2 = 0.5 * (r_bow + r_stern)
    r3 = 0.5 * (r_stern - r_bow)
    dv2["rocker"]["coefficients"] = [0.0, 0.0, r2, r3]

    # Section side exponent ramp
    sec = dv2["section_side_exponent"]
    sec["forward"] = by_name["sec_side_fwd"]
    sec["aft"] = by_name["sec_side_aft"]
    sec["forward_xi"] = by_name["sec_side_fwd_xi"]
    sec["aft_xi"] = by_name["sec_side_aft_xi"]

    # Bilge round ramp
    top = dv2["topsides"]
    bilge = top["bilge_round_m"]
    bilge["forward"] = by_name["bilge_fwd_m"]
    bilge["aft"] = by_name["bilge_aft_m"]
    bilge["forward_xi"] = by_name["bilge_fwd_xi"]
    bilge["aft_xi"] = by_name["bilge_aft_xi"]

    # Sheer plan
    sheer = top["sheer_plan"]
    sheer["max_half_breadth_m"] = by_name["sheer_max_hb_m"]
    sheer["max_position_xi"] = by_name["sheer_max_xi"]
    sheer["forward_exponent"] = by_name["sheer_fwd_exp"]
    sheer["aft_exponent"] = by_name["sheer_aft_exp"]

    # Deck crown exponent ramp
    crown = top["deck_crown_exponent"]
    crown["forward"] = by_name["crown_fwd"]
    crown["aft"] = by_name["crown_aft"]
    crown["forward_xi"] = by_name["crown_fwd_xi"]
    crown["aft_xi"] = by_name["crown_aft_xi"]

    # Shoulder round
    top["shoulder_round_m"] = by_name["shoulder_round_m"]

    # Deck freeboard linear scaling across knots (idempotent relative to any base_hull)
    s_fwd = by_name["deck_fb_fwd_scale"]
    s_aft = by_name["deck_fb_aft_scale"]
    base_fb_knots = dv2["deck_freeboard"]["knots"]
    base_fb_fwd = float(base_fb_knots[0][1])
    base_fb_aft = float(base_fb_knots[-1][1])
    if abs(base_fb_fwd) <= 1e-12 or abs(base_fb_aft) <= 1e-12:
        raise ValueError(
            "base_hull deck_freeboard endpoints must be non-zero for scaling"
        )
    s_fwd_base = base_fb_fwd / _BASE_DECK_FB_FWD
    s_aft_base = base_fb_aft / _BASE_DECK_FB_AFT
    if s_fwd != s_fwd_base or s_aft != s_aft_base:
        scaled_knots: list[list[float]] = []
        for xi, z_m in base_fb_knots:
            denom = 0.5 * (1.0 - xi) * s_fwd_base + 0.5 * (1.0 + xi) * s_aft_base
            if abs(denom) <= 1e-12:
                raise ValueError(
                    f"base_hull deck_freeboard scaling denominator near zero at xi={xi}"
                )
            numer = 0.5 * (1.0 - xi) * s_fwd + 0.5 * (1.0 + xi) * s_aft
            scaled_knots.append([xi, z_m * numer / denom])
        dv2["deck_freeboard"]["knots"] = scaled_knots

    # Sheer height fraction linear shift across knots (idempotent relative to any base_hull)
    sh_fwd = by_name["sheer_h_fwd_frac"]
    sh_aft = by_name["sheer_h_aft_frac"]
    base_sh_knots = top["sheer_height_frac"]["knots"]
    sh_fwd_base = float(base_sh_knots[0][1])
    sh_aft_base = float(base_sh_knots[-1][1])
    if sh_fwd != sh_fwd_base or sh_aft != sh_aft_base:
        d_fwd = sh_fwd - sh_fwd_base
        d_aft = sh_aft - sh_aft_base
        top["sheer_height_frac"]["knots"] = [
            [xi, frac + (0.5 * (1.0 - xi) * d_fwd + 0.5 * (1.0 + xi) * d_aft)]
            for xi, frac in base_sh_knots
        ]

    # Surface fairing target beam
    if dv2.get("surface_fairing") is None:
        dv2["surface_fairing"] = {"target_beam_m": by_name["target_beam_m"]}
    else:
        dv2["surface_fairing"]["target_beam_m"] = by_name["target_beam_m"]

    return Hull.model_validate(data)


def hull_to_vector(hull: Hull, space: DesignSpace | None = None) -> list[float]:
    """Extract the scalar design vector from a validated DistributionV2 Hull."""
    sp = DesignSpace.default() if space is None else space
    dv2 = _require_distribution_v2(hull)

    wl_c = dv2.waterline_half_breadth.coefficients
    d_c = dv2.draft_profile.coefficients
    r_c = dv2.rocker.coefficients
    r2 = float(r_c[2]) if len(r_c) > 2 else 0.0
    r3 = float(r_c[3]) if len(r_c) > 3 else 0.0

    sec = dv2.section_side_exponent
    top = dv2.topsides
    bilge = top.bilge_round_m
    sheer = top.sheer_plan
    crown = top.deck_crown_exponent
    fb_knots = dv2.deck_freeboard.knots
    sh_knots = top.sheer_height_frac.knots

    if sheer is None:
        raise ValueError("Hull topsides.sheer_plan is required for DesignSpace extraction")
    if dv2.surface_fairing is None or dv2.surface_fairing.target_beam_m is None:
        raise ValueError("Hull distribution_v2.surface_fairing.target_beam_m is required")
    if len(fb_knots) < 2 or abs(float(fb_knots[0][1])) <= 1e-12 or abs(float(fb_knots[-1][1])) <= 1e-12:
        raise ValueError("Hull deck_freeboard endpoints must be non-zero for DesignSpace extraction")

    extracted: dict[str, float] = {
        "wl_a0": float(wl_c[0]),
        "wl_a1": float(wl_c[1]),
        "wl_a2": float(wl_c[2]),
        "wl_a3": float(wl_c[3]),
        "draft_a1": float(d_c[1]) if len(d_c) > 1 else 0.0,
        "draft_a2": float(d_c[2]) if len(d_c) > 2 else 0.0,
        "rocker_bow_end_m": r2 - r3,
        "rocker_stern_end_m": r2 + r3,
        "sec_side_fwd": float(sec.forward),
        "sec_side_aft": float(sec.aft),
        "sec_side_fwd_xi": float(sec.forward_xi),
        "sec_side_aft_xi": float(sec.aft_xi),
        "bilge_fwd_m": float(bilge.forward),
        "bilge_aft_m": float(bilge.aft),
        "bilge_fwd_xi": float(bilge.forward_xi),
        "bilge_aft_xi": float(bilge.aft_xi),
        "sheer_max_hb_m": float(sheer.max_half_breadth_m),
        "sheer_max_xi": float(sheer.max_position_xi),
        "sheer_fwd_exp": float(sheer.forward_exponent),
        "sheer_aft_exp": float(sheer.aft_exponent),
        "crown_fwd": float(crown.forward),
        "crown_aft": float(crown.aft),
        "crown_fwd_xi": float(crown.forward_xi),
        "crown_aft_xi": float(crown.aft_xi),
        "shoulder_round_m": float(top.shoulder_round_m),
        "deck_fb_fwd_scale": float(fb_knots[0][1]) / _BASE_DECK_FB_FWD,
        "deck_fb_aft_scale": float(fb_knots[-1][1]) / _BASE_DECK_FB_AFT,
        "sheer_h_fwd_frac": float(sh_knots[0][1]),
        "sheer_h_aft_frac": float(sh_knots[-1][1]),
        "target_beam_m": float(dv2.surface_fairing.target_beam_m),
    }
    return [extracted[p.name] for p in sp.params]
