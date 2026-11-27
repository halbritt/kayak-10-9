"""Tier 0 / Tier 1 hull evaluation bridging kayakgen and SculptedKayakSurface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import root
from scipy.spatial.transform import Rotation
import trimesh

from kayakgen.eval.contract import LoadCase
from kayakgen.eval.form_metrics import evaluate_form_metrics
from kayakgen.eval.high_angle_gz import build_high_angle_gz_block
from kayakgen.eval.hydrostatics import OperatingPose, evaluate as evaluate_hydrostatics
from kayakgen.eval.resistance import resistance_curve
from kayakgen.model.hull import Hull
from kayakgen.services.evaluation import solve_target_draft

from rsi.baseline import BaselineReference

KG_M = 0.25
HULL_MASS_KG = 18.0
LB_TO_KG = 0.45359237
RHO_SEAWATER = 1025.0
LOADS_LB = (225, 265, 300)
HEEL_DEGS = (10, 20, 30)
RESISTANCE_SPEEDS_KN = np.arange(3.5, 7.76, 0.25)
POWER_LEVELS_W = (40.0, 60.0, 90.0, 120.0)
KAYAKGEN_RES_SPEEDS_KN = np.array([4.5, 5.0, 5.5, 6.0, 6.2], dtype=float)
TRANSITION_OFFSETS_M = [1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4]
CATCH_OFFSETS_M = [0.0, 0.8, 0.9, 1.0, 1.1, 1.2]
FOREBODY_STATIONS_M = (-2.10, -1.65, -1.35, -0.95)
# Stretch of the 0.075 m buttock over which its turn change is judged, in metres aft of the bow. It is
# v0.2.3's own interior: that hull's cut is finite from 0.539 to 5.169 m and cad/inspect_fairness.py
# allows 100 mm at each end. Every candidate is read over this same stretch.
BUTTOCK_0075_WINDOW_AFT_M = (0.639, 5.069)
BUTTOCK_0075_STEP_M = 0.002
BUTTOCK_0075_SPAN_SAMPLES = 25  # turn change over 50 mm


def _ensure_cad_on_path() -> None:
    cad_dir = str((Path(__file__).resolve().parents[1] / "cad").resolve())
    if cad_dir not in sys.path:
        sys.path.insert(0, cad_dir)


@dataclass(frozen=True)
class LoadEvalMetrics:
    load_lb: int
    t_cm: float
    draft_mm: float
    bwl_cm: float
    wet_m2: float
    gm_cm: float
    gz_mm: tuple[float, float, float]
    lcg_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateMetrics:
    valid: bool
    error: str | None
    design_hash: str
    hull_json_path: str
    mesh_resolution: tuple[int, int]
    elapsed_s: float
    skip_resistance: bool
    # Tier 0 (kayakgen)
    kg_equilibrium_draft_m: float | None = None
    kg_sinkage_m: float | None = None
    kg_wetted_surface_m2: float | None = None
    kg_gm0_m: float | None = None
    kg_lcb_frac: float | None = None
    kg_rt_N: dict[str, float] | None = None
    waterline_hollow_max_deg: float | None = None
    plan_hollow_max_deg: float | None = None
    upper_hollow_max_deg: float | None = None
    upper_knuckle_max_deg: float | None = None
    waterline_knuckle_max_deg: float | None = None
    foot_clear_height_m: float | None = None
    foot_clear_width_m: float | None = None
    catch_zone_max_beam_m: float | None = None
    catch_zone_max_upper_beam_m: float | None = None
    kg_high_angle_gz_mm: dict[str, float] | None = None
    # Tier 1 (SculptedKayakSurface closed outer shell)
    is_watertight: bool = False
    is_volume: bool = False
    n_faces: int = 0
    outer_stl_sha256: str | None = None
    z_min_m: float | None = None
    l_oa_m: float | None = None
    b_oa_cm: float | None = None
    l4_m: float | None = None
    b4_m: float | None = None
    sr_4in_exact: float | None = None
    sr_4in: float | None = None
    usca_4in_pct_exact: float | None = None
    usca_4in_pct: float | None = None
    catch_widths_mm: tuple[float, ...] | None = None
    catch_beams_225_cm: dict[str, float] | None = None
    transition_dB_per_20cm: tuple[float, ...] | None = None
    has_surface_fairing: bool = False
    fwd_wl_max_turn_deg: float | None = None
    entry_zone_max_turn_deg: float | None = None
    buttock_0075_peak_deg: float | None = None
    forebody_max_keel_groove_mm: float | None = None
    forebody_min_k: float | None = None
    forebody_max_abs_dk_ds: float | None = None
    loads: dict[int, LoadEvalMetrics] | None = None
    kn_225: dict[int, float] | None = None
    kn_225_exact: dict[int, float] | None = None
    kn_265: dict[int, float] | None = None
    kn_265_exact: dict[int, float] | None = None
    # Per load: this hull's local-friction ratio at each speed of RESISTANCE_SPEEDS_KN (see
    # local_friction_ratio). Diagnostic; the speed tables already include it.
    friction_ratio: dict[int, tuple[float, ...]] | None = None
    shell_area_m2: float | None = None
    aft_coaming_above_225wl_cm: float | None = None
    rear_deck_above_225wl_cm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.loads is not None:
            d["loads"] = {str(k): v.to_dict() for k, v in self.loads.items()}
        if self.kn_225 is not None:
            d["kn_225"] = {str(k): v for k, v in self.kn_225.items()}
        if self.kn_225_exact is not None:
            d["kn_225_exact"] = {str(k): v for k, v in self.kn_225_exact.items()}
        if self.kn_265 is not None:
            d["kn_265"] = {str(k): v for k, v in self.kn_265.items()}
        if self.kn_265_exact is not None:
            d["kn_265_exact"] = {str(k): v for k, v in self.kn_265_exact.items()}
        return d


# Skin closer to the bow than this local Reynolds number (about 4 cm at 5.5 kn) is charged the
# line's own mean value there, which is what that first strip carries in total. The line has no
# meaning at a leading edge, and its local form turns negative under Re = 740.
FRICTION_RE_MIN = 1.0e5


def local_friction_ratio(
    tris: np.ndarray, u_mps: float, nu: float, x_fwd: float, length_m: float
) -> float:
    """Friction with the ITTC-57 line applied along the hull, over the line applied all at once.

    ``hydro.ittc57`` multiplies one coefficient, taken at the Reynolds number of the whole
    waterline, by the whole wetted area. That charges a patch of skin the same wherever it is. A
    turbulent boundary layer is thin at the bow and thick aft, so skin near the bow carries about
    1.4 times the mean friction and skin near the stern about 0.83 times. Here every immersed
    triangle is charged the local coefficient of the same line at its distance from the forward
    end of the waterline, ``d(x Cf(Re_x))/dx``. For a plate of constant width the result is the
    plain ITTC-57 value, so the ratio is 1; a hull with more of its area forward is above 1.

    The speed model normalizes this ratio to the baseline at the same load and speed. Moving
    wetted area forward can therefore increase a candidate's friction even at fixed total area.
    The mesh-v1 paired runs gave different signs for the viscous drag change of the same hull
    under different settling treatments. The magnitude of the correction remains unvalidated.

    The bow is at -x in the SculptedKayakSurface frame, so distance runs from ``x_fwd`` to +x.
    """
    area = 0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
    )
    dist = tris[:, :, 0].mean(axis=1) - x_fwd
    re_x = u_mps * dist / nu
    lg = np.log10(np.maximum(re_x, FRICTION_RE_MIN)) - 2.0
    cf_line = 0.075 / lg**2
    cf_local = np.where(re_x < FRICTION_RE_MIN, cf_line, cf_line * (1.0 - (2.0 / math.log(10.0)) / lg))
    cf_mean = 0.075 / (math.log10(u_mps * length_m / nu) - 2.0) ** 2
    return float((area * cf_local).sum() / (area.sum() * cf_mean))


def _michell_fast_curve(
    xs: np.ndarray,
    zs: np.ndarray,
    Y: np.ndarray,
    u_arr: np.ndarray,
    rho: float,
    *,
    g: float = 9.80665,
    ntheta: int = 4000,
) -> np.ndarray:
    """Vectorized real-GEMM Michell thin-ship wave resistance across speeds u_arr (m/s)."""
    th = (np.arange(ntheta) + 0.5) * (np.pi / 2.0) / ntheta
    sec = 1.0 / np.cos(th)
    sec2 = sec * sec
    sec3 = sec2 * sec
    dx = np.diff(xs)
    xm = 0.5 * (xs[:-1] + xs[1:])
    S = np.ascontiguousarray(np.diff(Y, axis=0) / dx[:, None])
    hz = np.diff(zs)
    out = np.empty(len(u_arr), dtype=float)
    for idx, u_val in enumerate(u_arr):
        u = float(u_val)
        k0 = g / (u * u)
        kap = k0 * sec
        a = k0 * sec2
        half_kd = 0.5 * kap[:, None] * dx[None, :]
        fac = dx[None, :] * np.sinc(half_kd / np.pi)
        ph = kap[:, None] * xm[None, :]
        Qr = (np.cos(ph) * fac) @ S
        Qi = (np.sin(ph) * fac) @ S
        t = a[:, None] * hz[None, :]
        em = np.exp(-t)
        e1 = np.exp(a[:, None] * zs[None, 1:])
        inv_t2 = hz[None, :] / (t * t)
        w_hi = e1 * (t - 1.0 + em) * inv_t2
        w_lo = e1 * (1.0 - em * (1.0 + t)) * inv_t2
        Ar = (Qr[:, 1:] * w_hi + Qr[:, :-1] * w_lo).sum(axis=1)
        Ai = (Qi[:, 1:] * w_hi + Qi[:, :-1] * w_lo).sum(axis=1)
        integrand = (Ar * Ar + Ai * Ai) * sec3
        out[idx] = 4.0 * rho * g * g / (np.pi * u * u) * integrand.sum() * (np.pi / 2.0) / ntheta
    return out


def _make_load_case_225() -> LoadCase:
    total_mass = 225.0 * LB_TO_KG
    return LoadCase(
        name="225lb",
        paddler_mass_kg=total_mass - HULL_MASS_KG,
        hull_mass_kg=HULL_MASS_KG,
        cargo_mass_kg=0.0,
        kg_above_keel_m=KG_M,
    )


def _solve_free_trim_gz_mm(
    closed: trimesh.Trimesh,
    immersed_fn: Any,
    volume: float,
    upright_height: float,
    lcg: float,
    z0: float,
) -> tuple[float, float, float]:
    cg = np.array([lcg, 0.0, z0 + KG_M], dtype=float)
    gz_list: list[float] = []
    for deg in HEEL_DEGS:
        heel = Rotation.from_euler("x", deg, degrees=True)

        def balance(pose: np.ndarray) -> list[float]:
            h, trim = float(pose[0]), float(pose[1])
            rot = heel * Rotation.from_euler("y", trim)
            vol, buoyancy = immersed_fn(closed, rot, h)
            if not np.isfinite(vol) or vol <= 0.0 or not np.all(np.isfinite(buoyancy)):
                raise ValueError(f"Degenerate immersed volume at heel={deg} deg")
            lever = rot.apply(cg) - buoyancy
            return [vol / volume - 1.0, float(lever[0])]

        sol = root(balance, [upright_height, 0.0])
        resid = balance(sol.x)
        if not sol.success or float(np.max(np.abs(resid))) > 1e-6:
            raise ValueError(f"Free-trim balance failed at {deg} deg: {sol.message}")
        rot = heel * Rotation.from_euler("y", float(sol.x[1]))
        _, buoyancy = immersed_fn(closed, rot, float(sol.x[0]))
        lever = rot.apply(cg) - buoyancy
        gz_list.append(float(1000.0 * lever[1]))
    return (gz_list[0], gz_list[1], gz_list[2])


def _buttock_peak_turn_change(ang: np.ndarray, aft_of_bow_m: np.ndarray) -> float:
    """Largest change of turning angle over 50 mm on the 0.075 m buttock, within a fixed stretch of hull.

    The reading used to run over each hull's own cut less 100 mm at either end. Its peak is at the
    aft edge of that allowance, where the buttock climbs to the stern and the turn change rises
    0.03 to 0.07 deg per 2 mm sample. A stern 2 mm longer at this buttock therefore added one sample
    and about 0.06 deg, against a margin of 0.04 deg to the limit, at any size of design step. Over
    a fixed stretch the reading follows the hull shape continuously, and v0.2.3 reads as before.
    """
    lo, hi = BUTTOCK_0075_WINDOW_AFT_M
    tol = 1e-9
    ins = np.flatnonzero((aft_of_bow_m >= lo - tol) & (aft_of_bow_m <= hi + tol))
    if len(ins) == 0 or ins[-1] + BUTTOCK_0075_SPAN_SAMPLES >= len(ang):
        raise ValueError("Buttocks 0.075 cut does not reach the stretch it is judged on")
    change = np.abs(ang[ins + BUTTOCK_0075_SPAN_SAMPLES] - ang[ins])
    if not np.isfinite(change).all():
        # The hull is narrower than 75 mm somewhere v0.2.3 is not. Treat it as unmeasurable rather
        # than judge it on a shorter stretch.
        raise ValueError("Buttocks 0.075 cut does not cover the stretch it is judged on")
    return float(change.max())


def _evaluate_mesh_fairness(
    closed: trimesh.Trimesh,
    hull_length_m: float,
    z_min_m: float,
    cut_line_fn: Any,
    turns_fn: Any,
    section_curve_fn: Any,
) -> tuple[float, float, float, float, float, float]:
    x_span = np.arange(0.645, 1.075, 0.002) - hull_length_m / 2.0
    fwd_wl_turns: list[float] = []
    for z in (0.050, 0.060, 0.070, 0.080, 0.090, 0.1016):
        y = cut_line_fn(closed, x_span, [0, 1], z + z_min_m)
        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite forward waterline cut at z={z}")
        t_vals = turns_fn(y, 0.002)[25:-25]
        if not np.isfinite(t_vals).any():
            raise ValueError(f"Empty forward waterline turn slice at z={z}")
        fwd_wl_turns.append(float(np.nanmax(t_vals)))
    fwd_wl_max_turn_deg = max(fwd_wl_turns)

    bow_x = float(closed.bounds[0, 0])
    stern_x = float(closed.bounds[1, 0])
    x_full = np.arange(bow_x + 0.001, stern_x, 0.002)
    entry_zone = (x_full - bow_x >= 0.40) & (x_full - bow_x <= 0.70)
    entry_turns: list[float] = []
    for z in (0.060, 0.070, 0.080, 0.090, 0.1016):
        ang = turns_fn(cut_line_fn(closed, x_full, [0, 1], z + z_min_m), 0.002)
        if not np.isfinite(ang[entry_zone]).any():
            raise ValueError(f"Empty entry zone waterline cut at z={z}")
        entry_turns.append(float(np.nanmax(ang[entry_zone])))
    entry_zone_max_turn_deg = max(entry_turns)

    ang_b75 = turns_fn(cut_line_fn(closed, x_full, [1, 0], 0.075), BUTTOCK_0075_STEP_M)
    buttock_0075_peak_deg = _buttock_peak_turn_change(ang_b75, x_full - bow_x)

    grooves: list[float] = []
    k_mins: list[float] = []
    dk_maxes: list[float] = []
    for x_sec in FOREBODY_STATIONS_M:
        arc, q, k = section_curve_fn(closed, x_sec)
        lower = (arc < arc[-1] / 2.0) & (q[:, 1] < 0.12)
        if not np.any(lower):
            raise ValueError(f"Empty lower forebody section at x={x_sec}")
        change = (np.roll(k, -2) - np.roll(k, 2)) / (4.0 * arc[1])
        grooves.append(float((q[0, 1] - q[:, 1].min()) * 1000.0))
        k_mins.append(float(k[lower].min()))
        dk_maxes.append(float(np.abs(change[lower]).max()))

    return (
        fwd_wl_max_turn_deg,
        entry_zone_max_turn_deg,
        buttock_0075_peak_deg,
        max(grooves),
        min(k_mins),
        max(dk_maxes),
    )


def evaluate_tier0_tier1(
    hull: Hull,
    work_dir: Path,
    baseline: BaselineReference,
    *,
    skip_resistance: bool = False,
    mesh_resolution: tuple[int, int] = (441, 129),
    fast_tier0: bool = False,
) -> CandidateMetrics:
    """Evaluate a candidate Hull using kayakgen (Tier 0) and SculptedKayakSurface (Tier 1)."""
    base_225 = baseline.loads.get(225) if baseline is not None else None
    fixed_catch_lcg_m = (
        float(base_225.lcg_m)
        if base_225 is not None and base_225.lcg_m is not None
        else None
    )
    t0 = time.monotonic()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    hull_path = work / "candidate.hull.json"
    hull_path.write_text(hull.model_dump_json(indent=2) + "\n")
    d_hash = hull.design_hash()

    def _fail(msg: str, **partial: Any) -> CandidateMetrics:
        return CandidateMetrics(
            valid=False,
            error=msg,
            design_hash=d_hash,
            hull_json_path=str(hull_path),
            mesh_resolution=mesh_resolution,
            elapsed_s=round(time.monotonic() - t0, 3),
            skip_resistance=skip_resistance,
            **partial,
        )

    # Tier 0: kayakgen form metrics (plus optional full Tier 0 hydrostatics/resistance/GZ)
    try:
        fm = evaluate_form_metrics(
            hull,
            catch_zone_forward_m=(0.8, 1.2),
            foot_box_width_m=0.28,
            foot_box_height_m=0.25,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"evaluate_form_metrics exception: {exc}")

    if fm.class_length_beam_ratio is None or fm.waterline_hollow_max_deg is None:
        return _fail("evaluate_form_metrics returned incomplete waterline metrics")

    if fast_tier0:
        if float(fm.waterline_hollow_max_deg) >= 0.025:
            return _fail(
                f"fast_tier0 waterline_hollow_max_deg={float(fm.waterline_hollow_max_deg):.4f} >= 0.025",
                waterline_hollow_max_deg=float(fm.waterline_hollow_max_deg),
            )
        tier0_fields: dict[str, Any] = {
            "waterline_hollow_max_deg": float(fm.waterline_hollow_max_deg),
            "plan_hollow_max_deg": (
                None if fm.plan_hollow_max_deg is None else float(fm.plan_hollow_max_deg)
            ),
            "upper_hollow_max_deg": float(fm.upper_hollow_max_deg),
            "upper_knuckle_max_deg": float(fm.upper_knuckle_max_deg),
            "waterline_knuckle_max_deg": float(fm.waterline_knuckle_max_deg),
            "foot_clear_height_m": float(fm.foot_clear_height_m),
            "foot_clear_width_m": float(fm.foot_clear_width_m),
            "catch_zone_max_beam_m": float(fm.catch_zone_max_beam_m),
            "catch_zone_max_upper_beam_m": float(fm.catch_zone_max_upper_beam_m),
        }
    else:
        lc_225 = _make_load_case_225()
        try:
            st = solve_target_draft(hull, lc_225)
        except Exception as exc:  # noqa: BLE001
            return _fail(f"solve_target_draft exception: {exc}")

        if st.status != "converged" or st.sinkage_m is None:
            return _fail(f"solve_target_draft not converged: status={st.status}, warnings={st.warnings}")

        pose = OperatingPose(sinkage_m=float(st.sinkage_m), trim_angle_deg=float(st.trim_angle_deg or 0.0))
        try:
            kg_hydro = evaluate_hydrostatics(hull, pose=pose, density_kg_m3=lc_225.seawater_density_kg_m3)
        except Exception as exc:  # noqa: BLE001
            return _fail(f"evaluate_hydrostatics exception: {exc}")

        if (
            kg_hydro.GM0_m is None
            or "operating_pose_dry" in kg_hydro.notes
            or "operating_pose_fully_submerged" in kg_hydro.notes
        ):
            return _fail(
                f"invalid kayakgen hydrostatics at pose: GM0={kg_hydro.GM0_m}, notes={kg_hydro.notes}"
            )

        kg_rt: dict[str, float] | None = None
        if not skip_resistance:
            try:
                curve = resistance_curve(
                    hull,
                    V_knots=KAYAKGEN_RES_SPEEDS_KN,
                    n_stations=400,
                    n_depths=20,
                    n_theta=30,
                    pose=pose,
                )
                rt_arr = np.asarray(curve.Rt_N, dtype=float)
                if not np.all(np.isfinite(rt_arr)):
                    return _fail("non-finite kayakgen resistance_curve output")
                kg_rt = {f"{float(v):.1f}": float(r) for v, r in zip(curve.V_knots, rt_arr)}
            except Exception as exc:  # noqa: BLE001
                return _fail(f"resistance_curve exception: {exc}")

        try:
            gz_block = build_high_angle_gz_block(hull, lc_225, heel_grid_deg=(10.0, 20.0, 30.0))
        except Exception as exc:  # noqa: BLE001
            return _fail(f"build_high_angle_gz_block exception: {exc}")

        if gz_block.get("available") is not True or gz_block.get("summary") is None:
            return _fail(f"high_angle_gz unavailable: {gz_block.get('unavailable_reason')}")
        kg_gz_mm = {
            str(int(round(float(pt["heel_deg"])))): float(pt["gz_m"]) * 1000.0
            for pt in gz_block["heel_points"]
        }

        tier0_fields = {
            "kg_equilibrium_draft_m": (
                None if st.equilibrium_draft_m is None else float(st.equilibrium_draft_m)
            ),
            "kg_sinkage_m": float(st.sinkage_m),
            "kg_wetted_surface_m2": float(kg_hydro.wetted_surface_m2),
            "kg_gm0_m": float(kg_hydro.GM0_m),
            "kg_lcb_frac": float(kg_hydro.LCB_frac),
            "kg_rt_N": kg_rt,
            "waterline_hollow_max_deg": float(fm.waterline_hollow_max_deg),
            "plan_hollow_max_deg": (
                None if fm.plan_hollow_max_deg is None else float(fm.plan_hollow_max_deg)
            ),
            "upper_hollow_max_deg": float(fm.upper_hollow_max_deg),
            "upper_knuckle_max_deg": float(fm.upper_knuckle_max_deg),
            "waterline_knuckle_max_deg": float(fm.waterline_knuckle_max_deg),
            "foot_clear_height_m": float(fm.foot_clear_height_m),
            "foot_clear_width_m": float(fm.foot_clear_width_m),
            "catch_zone_max_beam_m": float(fm.catch_zone_max_beam_m),
            "catch_zone_max_upper_beam_m": float(fm.catch_zone_max_upper_beam_m),
            "kg_high_angle_gz_mm": kg_gz_mm,
        }

    # Tier 1: SculptedKayakSurface finished closed outer shell
    _ensure_cad_on_path()
    from build_specs_and_ga import _station_beams_ge_10cm
    from check_fairing_controls import catch_widths as controls_catch_widths
    from check_fairing_controls import immersed as controls_immersed
    from deck_geometry import SculptedKayakSurface
    from hydro import G, KN, NU_SEAWATER, float_at, half_breadth_grid, ittc57
    from inspect_fairness import cut_line, section_curve, turns

    try:
        surf = SculptedKayakSurface(
            hull_path, n_stations=mesh_resolution[0], n_hull_half=mesh_resolution[1]
        )
        hull_shell = surf.build_hull_shell()
        deck_shell = surf.build_deck_shell(cut_openings=False)
        closed = trimesh.util.concatenate((hull_shell, deck_shell))
        closed.merge_vertices(merge_tex=True, merge_norm=True)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"SculptedKayakSurface build exception: {exc}", **tier0_fields)

    is_wt = bool(closed.is_watertight)
    is_vol = bool(closed.is_volume)
    n_faces = int(len(closed.faces))
    if not is_wt or not is_vol or n_faces == 0:
        return _fail(
            f"Closed shell invalid: is_watertight={is_wt}, is_volume={is_vol}, faces={n_faces}",
            is_watertight=is_wt,
            is_volume=is_vol,
            n_faces=n_faces,
            **tier0_fields,
        )

    outer_sha = hashlib.sha256(closed.export(file_type="stl")).hexdigest()
    V = np.asarray(closed.vertices, dtype=float)
    F = np.asarray(closed.faces, dtype=int)
    z_min_m = float(closed.bounds[0, 2])
    l_oa_m = float(np.ptp(V[:, 0]))
    b_oa_cm = float(np.ptp(V[:, 1]) * 100.0)
    shell_area_m2 = round(float(closed.area), 6)

    r4 = trimesh.intersections.mesh_plane(
        closed, [0.0, 0.0, 1.0], [0.0, 0.0, z_min_m + 0.1016]
    ).reshape(-1, 3)
    if len(r4) < 4:
        return _fail(
            "4-inch waterline slice returned insufficient points",
            is_watertight=is_wt,
            is_volume=is_vol,
            n_faces=n_faces,
            outer_stl_sha256=outer_sha,
            z_min_m=z_min_m,
            **tier0_fields,
        )
    l4_m = float(np.ptp(r4[:, 0]))
    b4_m = float(np.ptp(r4[:, 1]))
    if b4_m <= 1e-6 or not (math.isfinite(l4_m) and math.isfinite(b4_m)):
        return _fail("Degenerate 4-inch waterline dimensions", **tier0_fields)

    sr_4in_exact = l4_m / b4_m
    sr_4in = round(sr_4in_exact, 2)
    usca_4in_pct_exact = 100.0 * b4_m / l_oa_m
    usca_4in_pct = round(usca_4in_pct_exact, 2)

    has_sf = hull.distribution_v2 is not None and hull.distribution_v2.surface_fairing is not None
    try:
        (
            fwd_wl_max_turn_deg,
            entry_zone_max_turn_deg,
            buttock_0075_peak_deg,
            forebody_max_keel_groove_mm,
            forebody_min_k,
            forebody_max_abs_dk_ds,
        ) = _evaluate_mesh_fairness(
            closed, surf.L, z_min_m, cut_line, turns, section_curve
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"Mesh fairness evaluation failed: {exc}", **tier0_fields)

    loads_eval: dict[int, LoadEvalMetrics] = {}
    catch_widths_mm: tuple[float, ...] | None = None
    catch_beams_225_cm: dict[str, float] | None = None
    transition_dB: tuple[float, ...] | None = None
    kn_225: dict[int, float] | None = None
    kn_225_exact: dict[int, float] | None = None
    kn_265: dict[int, float] | None = None
    kn_265_exact: dict[int, float] | None = None
    friction_ratio: dict[int, tuple[float, ...]] = {}
    aft_coaming_above_225wl_cm: float | None = None
    rear_deck_above_225wl_cm: float | None = None

    def _speed_tables(r_up: dict[str, Any], lb: int) -> tuple[dict[int, float], dict[int, float]]:
        lwl_m = float(np.ptp(r_up["wl_x"]))
        xs, zs, Y = half_breadth_grid(r_up["tris"], 401, 81)
        u_arr = RESISTANCE_SPEEDS_KN * KN
        rf_arr = np.array(
            [ittc57(float(u), lwl_m, r_up["wet"], RHO_SEAWATER, NU_SEAWATER)[0] for u in u_arr],
            dtype=float,
        )
        # Charge friction where the area is, relative to the baseline's own distribution: the
        # baseline keeps its published speeds, and a candidate pays for moving area forward.
        x_fwd = float(min(r_up["wl_x"]))
        ratio = np.array(
            [local_friction_ratio(r_up["tris"], float(u), NU_SEAWATER, x_fwd, lwl_m) for u in u_arr]
        )
        friction_ratio[lb] = tuple(float(x) for x in ratio)
        if baseline.friction_ratio is not None:
            base_ratio = baseline.friction_ratio.get(lb)
            if base_ratio is None:
                raise ValueError(f"baseline friction reference has no {lb} lb load")
            speeds = baseline.friction_speeds_kn
            if speeds is None or not np.array_equal(np.asarray(speeds), RESISTANCE_SPEEDS_KN):
                raise ValueError("baseline friction reference is not on RESISTANCE_SPEEDS_KN")
            rf_arr = rf_arr * ratio / np.asarray(base_ratio, dtype=float)
        rw_arr = _michell_fast_curve(xs, zs, Y, u_arr, RHO_SEAWATER, g=G, ntheta=4000)
        pwr_w = (rf_arr + rw_arr) * u_arr
        exact_tbl = {
            int(p): float(np.interp(p, pwr_w, RESISTANCE_SPEEDS_KN))
            for p in POWER_LEVELS_W
        }
        return {int(p): round(val, 2) for p, val in exact_tbl.items()}, exact_tbl

    try:
        for lb in LOADS_LB:
            mass_kg = lb * LB_TO_KG
            volume = mass_kg / RHO_SEAWATER
            r_up = float_at(
                V, F, mass_kg, RHO_SEAWATER, np.array([0.0, 0.0, z_min_m + KG_M]), free_trim=False
            )
            lcg = float(r_up["B"][0])
            t_m = float(-r_up["zmin"])
            upright_height = float(r_up["h"])
            bwl_m = float(np.ptp(r_up["wl_y"]))
            gm_m = float(r_up["B"][2] - r_up["zmin"] + r_up["IT"] / r_up["vol"] - KG_M)
            gz_mm = _solve_free_trim_gz_mm(
                closed, controls_immersed, volume, upright_height, lcg, z_min_m
            )
            loads_eval[lb] = LoadEvalMetrics(
                load_lb=lb,
                t_cm=round(t_m * 100.0, 2),
                draft_mm=t_m * 1000.0,
                bwl_cm=round(bwl_m * 100.0, 2),
                wet_m2=round(float(r_up["wet"]), 3),
                gm_cm=gm_m * 100.0,
                gz_mm=gz_mm,
                lcg_m=lcg,
            )

            if lb == 225:
                # Fixed cockpit frame relative to baseline 225 lb LCG, sliced 100 mm above this hull's 225 lb WL.
                catch_ref_x = fixed_catch_lcg_m if fixed_catch_lcg_m is not None else lcg
                catch_widths_mm = tuple(
                    float(x) for x in controls_catch_widths(closed, catch_ref_x, upright_height)
                )
                catch_beams_225_cm = _station_beams_ge_10cm(
                    V, F, catch_ref_x, upright_height, -1, CATCH_OFFSETS_M
                )
                grid_b = _station_beams_ge_10cm(
                    V, F, catch_ref_x, upright_height, -1, TRANSITION_OFFSETS_M
                )
                vals = [grid_b[f"{o:+.1f}"] for o in TRANSITION_OFFSETS_M]
                transition_dB = tuple(round(vals[i + 1] - vals[i], 1) for i in range(len(vals) - 1))
                aft_coaming_above_225wl_cm = round(
                    (float(surf.sculpted_deck_z(np.array([0.38]), np.array([0.0]))[0]) + 0.0185 - upright_height) * 100.0,
                    3,
                )
                rear_deck_above_225wl_cm = round(
                    (float(surf.base_deck_z(np.array([0.68]), np.array([0.0]))[0]) - upright_height) * 100.0,
                    3,
                )
                if not skip_resistance:
                    kn_225, kn_225_exact = _speed_tables(r_up, lb)
            elif lb == 265 and not skip_resistance:
                kn_265, kn_265_exact = _speed_tables(r_up, lb)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"Hydrostatics / stability / resistance failed: {exc}", **tier0_fields)

    return CandidateMetrics(
        valid=True,
        error=None,
        design_hash=d_hash,
        hull_json_path=str(hull_path),
        mesh_resolution=mesh_resolution,
        elapsed_s=round(time.monotonic() - t0, 3),
        skip_resistance=skip_resistance,
        is_watertight=is_wt,
        is_volume=is_vol,
        n_faces=n_faces,
        outer_stl_sha256=outer_sha,
        z_min_m=z_min_m,
        l_oa_m=l_oa_m,
        b_oa_cm=b_oa_cm,
        l4_m=l4_m,
        b4_m=b4_m,
        sr_4in_exact=sr_4in_exact,
        sr_4in=sr_4in,
        usca_4in_pct_exact=usca_4in_pct_exact,
        usca_4in_pct=usca_4in_pct,
        catch_widths_mm=catch_widths_mm,
        catch_beams_225_cm=catch_beams_225_cm,
        transition_dB_per_20cm=transition_dB,
        has_surface_fairing=has_sf,
        fwd_wl_max_turn_deg=fwd_wl_max_turn_deg,
        entry_zone_max_turn_deg=entry_zone_max_turn_deg,
        buttock_0075_peak_deg=buttock_0075_peak_deg,
        forebody_max_keel_groove_mm=forebody_max_keel_groove_mm,
        forebody_min_k=forebody_min_k,
        forebody_max_abs_dk_ds=forebody_max_abs_dk_ds,
        loads=loads_eval,
        kn_225=kn_225,
        kn_225_exact=kn_225_exact,
        kn_265=kn_265,
        kn_265_exact=kn_265_exact,
        friction_ratio=dict(friction_ratio) or None,
        shell_area_m2=shell_area_m2,
        aft_coaming_above_225wl_cm=aft_coaming_above_225wl_cm,
        rear_deck_above_225wl_cm=rear_deck_above_225wl_cm,
        **tier0_fields,
    )
