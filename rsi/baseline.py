"""Baseline v0.2.3 hull and reference metric loader for Kayak 10.9 RSI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from kayakgen.model.hull import Hull

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

FAIRNESS_FWD_WL_TURN_MAX_DEG = 0.0
FAIRNESS_ENTRY_ZONE_TURN_MAX_DEG = 0.03
FAIRNESS_BUTTOCK_0075_PEAK_MAX_DEG = 3.50
FAIRNESS_FOREBODY_KEEL_GROOVE_MAX_MM = 1e-4  # 1e-7 m
FAIRNESS_FOREBODY_K_MIN_PER_M = 0.0
FAIRNESS_FOREBODY_DK_DS_MAX_PER_M2 = 210.0

TARGET_CAPYTAINE_SPEEDS_KN = (4.5, 5.7)
TARGET_CAPYTAINE_HEADINGS = ("head", "bow_quarter")
TARGET_CAPYTAINE_SEAS = (
    "Bay chop (Hs 0.35m, Tp 2.5s)",
    "Steep whitecaps (Hs 0.65m, Tp 3.8s)",
)


@dataclass(frozen=True)
class LoadBaseline:
    load_lb: int
    draft_mm: float
    t_cm: float
    bwl_cm: float
    wet_m2: float
    gm_cm: float
    gz_mm: tuple[float, float, float]
    gz_rounded_10_20_30_40_mm: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessCutBaseline:
    peak_turn_change_deg: float
    distance_aft_m: float
    rms_turn_change_deg: float
    max_signed_turn_forward_deg: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessSectionBaseline:
    fraction: float
    keel_groove_mm: float
    peak_abs_dk_ds_per_m2: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FairnessBaseline:
    sha256: str
    cuts: dict[str, FairnessCutBaseline]
    sections: tuple[FairnessSectionBaseline, ...]
    fwd_wl_turn_max_deg: float = FAIRNESS_FWD_WL_TURN_MAX_DEG
    entry_zone_turn_max_deg: float = FAIRNESS_ENTRY_ZONE_TURN_MAX_DEG
    buttock_0075_peak_max_deg: float = FAIRNESS_BUTTOCK_0075_PEAK_MAX_DEG
    forebody_keel_groove_max_mm: float = FAIRNESS_FOREBODY_KEEL_GROOVE_MAX_MM
    forebody_k_min_per_m: float = FAIRNESS_FOREBODY_K_MIN_PER_M
    forebody_dk_ds_max_per_m2: float = FAIRNESS_FOREBODY_DK_DS_MAX_PER_M2

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "cuts": {k: v.to_dict() for k, v in self.cuts.items()},
            "sections": [s.to_dict() for s in self.sections],
            "thresholds": {
                "fwd_wl_turn_max_deg": self.fwd_wl_turn_max_deg,
                "entry_zone_turn_max_deg": self.entry_zone_turn_max_deg,
                "buttock_0075_peak_max_deg": self.buttock_0075_peak_max_deg,
                "forebody_keel_groove_max_mm": self.forebody_keel_groove_max_mm,
                "forebody_k_min_per_m": self.forebody_k_min_per_m,
                "forebody_dk_ds_max_per_m2": self.forebody_dk_ds_max_per_m2,
            },
        }


@dataclass(frozen=True)
class SeakeepingConditionBaseline:
    speed_kn: float
    heading: str
    sea: str
    cov: float
    heave_sig_cm: float
    pitch_sig_deg: float
    roll_sig_deg: float
    bow_rel_sig_cm: float
    bow_dips_per_min: float
    aft_dips_per_min: float
    forefoot_out_per_min: float
    seat_acc_rms_ms2: float
    yaw_mom_rms_Nm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapytaineBaseline:
    outer_stl_sha256: str
    capytaine_version: str
    gm_t_cm: float
    roll_period_s: float
    added_mass_surge_kg: float
    added_mass_sway_kg: float
    draft_cm: float
    info_225: dict[str, Any]
    target_conditions_225: tuple[SeakeepingConditionBaseline, ...]
    rows_225: tuple[SeakeepingConditionBaseline, ...]
    gm_t_cm_265_baja: float
    roll_period_s_265_baja: float
    rows_265_baja: tuple[SeakeepingConditionBaseline, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_stl_sha256": self.outer_stl_sha256,
            "capytaine_version": self.capytaine_version,
            "gm_t_cm": self.gm_t_cm,
            "roll_period_s": self.roll_period_s,
            "added_mass_surge_kg": self.added_mass_surge_kg,
            "added_mass_sway_kg": self.added_mass_sway_kg,
            "draft_cm": self.draft_cm,
            "info_225": dict(self.info_225),
            "target_conditions_225": [r.to_dict() for r in self.target_conditions_225],
            "rows_225": [r.to_dict() for r in self.rows_225],
            "gm_t_cm_265_baja": self.gm_t_cm_265_baja,
            "roll_period_s_265_baja": self.roll_period_s_265_baja,
            "rows_265_baja": [r.to_dict() for r in self.rows_265_baja],
        }


@dataclass(frozen=True)
class OpenFoamSpeedBaseline:
    speed_kn: float
    u_mps: float
    drag_total_N: float
    drag_std_N: float
    drag_pressure_N: float
    drag_viscous_N: float
    effective_power_W: float
    vertical_force_N: float
    pitch_moment_My_Nm: float
    bow_up_running_trim_deg: float
    bow_wave_crest_mm: float
    mid_wave_trough_mm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenFoamBaseline:
    outer_stl_sha256: str
    openfoam_version: str
    pose_kayak_10_9: dict[str, Any]
    speed_5_5kn: OpenFoamSpeedBaseline
    speed_6_2kn: OpenFoamSpeedBaseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_stl_sha256": self.outer_stl_sha256,
            "openfoam_version": self.openfoam_version,
            "pose_kayak_10_9": dict(self.pose_kayak_10_9),
            "speed_5_5kn": self.speed_5_5kn.to_dict(),
            "speed_6_2kn": self.speed_6_2kn.to_dict(),
        }


@dataclass(frozen=True)
class BaselineReference:
    hull: Hull
    design_hash: str
    record_hash: str
    outer_stl_sha256: str
    l_oa_m: float
    b_oa_cm: float
    l4_m: float
    b4_m: float
    sr_4in_exact: float
    sr_4in: float
    usca_4in_pct_exact: float
    usca_4in_pct: float
    mass_at_4in_kg: float
    kn_225: dict[int, float]
    loads: dict[int, LoadBaseline]
    catch_widths_mm: tuple[float, ...]
    catch_beams_225_cm: dict[str, float]
    transition_beams_cm: dict[str, float]
    transition_dB_per_20cm: tuple[float, ...]
    fairness: FairnessBaseline
    capytaine: CapytaineBaseline
    openfoam: OpenFoamBaseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "hull": self.hull.model_dump(mode="json"),
            "design_hash": self.design_hash,
            "record_hash": self.record_hash,
            "outer_stl_sha256": self.outer_stl_sha256,
            "l_oa_m": self.l_oa_m,
            "b_oa_cm": self.b_oa_cm,
            "l4_m": self.l4_m,
            "b4_m": self.b4_m,
            "sr_4in_exact": self.sr_4in_exact,
            "sr_4in": self.sr_4in,
            "usca_4in_pct_exact": self.usca_4in_pct_exact,
            "usca_4in_pct": self.usca_4in_pct,
            "mass_at_4in_kg": self.mass_at_4in_kg,
            "kn_225": {int(k): v for k, v in self.kn_225.items()},
            "loads": {str(k): v.to_dict() for k, v in self.loads.items()},
            "catch_widths_mm": list(self.catch_widths_mm),
            "catch_beams_225_cm": dict(self.catch_beams_225_cm),
            "transition_beams_cm": dict(self.transition_beams_cm),
            "transition_dB_per_20cm": list(self.transition_dB_per_20cm),
            "fairness": self.fairness.to_dict(),
            "capytaine": self.capytaine.to_dict(),
            "openfoam": self.openfoam.to_dict(),
        }


def _parse_seakeeping_rows(raw_rows: list[dict[str, Any]]) -> tuple[SeakeepingConditionBaseline, ...]:
    return tuple(
        SeakeepingConditionBaseline(
            speed_kn=float(r["speed_kn"]),
            heading=str(r["heading"]),
            sea=str(r["sea"]),
            cov=float(r["cov"]),
            heave_sig_cm=float(r["heave_sig_cm"]),
            pitch_sig_deg=float(r["pitch_sig_deg"]),
            roll_sig_deg=float(r["roll_sig_deg"]),
            bow_rel_sig_cm=float(r["bow_rel_sig_cm"]),
            bow_dips_per_min=float(r["bow_dips_per_min"]),
            aft_dips_per_min=float(r["aft_dips_per_min"]),
            forefoot_out_per_min=float(r["forefoot_out_per_min"]),
            seat_acc_rms_ms2=float(r["seat_acc_rms_ms2"]),
            yaw_mom_rms_Nm=float(r["yaw_mom_rms_Nm"]),
        )
        for r in raw_rows
    )


def _parse_openfoam_speed(raw: dict[str, Any]) -> OpenFoamSpeedBaseline:
    return OpenFoamSpeedBaseline(
        speed_kn=float(raw["speed_kn"]),
        u_mps=float(raw["u_mps"]),
        drag_total_N=float(raw["drag_total_N"]),
        drag_std_N=float(raw["drag_std_N"]),
        drag_pressure_N=float(raw["drag_pressure_N"]),
        drag_viscous_N=float(raw["drag_viscous_N"]),
        effective_power_W=float(raw["effective_power_W"]),
        vertical_force_N=float(raw["vertical_force_N"]),
        pitch_moment_My_Nm=float(raw["pitch_moment_My_Nm"]),
        bow_up_running_trim_deg=float(raw["bow_up_running_trim_deg"]),
        bow_wave_crest_mm=float(raw["bow_wave_crest_mm"]),
        mid_wave_trough_mm=float(raw["mid_wave_trough_mm"]),
    )


def load_baseline_reference(repo_root: Path | None = None) -> BaselineReference:
    """Load immutable baseline v0.2.3 hull and reference metrics from repo JSON files."""
    root = DEFAULT_REPO_ROOT if repo_root is None else Path(repo_root).resolve()

    hull = Hull.model_validate_json((root / "hull" / "kayak-10-9.hull.json").read_text())
    specs = json.loads((root / "data" / "kayak-10-9-specs.json").read_text())
    controls = json.loads((root / "docs" / "surface-fairing" / "controls.json").read_text())["candidate"]
    fairness_raw = json.loads((root / "docs" / "surface-fairing" / "fairness.json").read_text())["v0.2.2"]
    capytaine_raw = json.loads((root / "data" / "capytaine-seakeeping.json").read_text())
    openfoam_raw = json.loads((root / "data" / "openfoam-cfd-metrics.json").read_text())

    hydro_3 = specs["hydro_3load"]
    l_oa_m = float(specs["L_oa_m"])
    b_oa_cm = float(specs["B_oa_cm"])
    l4_m = float(controls["waterline_mm"][0]) / 1000.0
    b4_m = float(controls["waterline_mm"][1]) / 1000.0
    sr_4in_exact = float(controls["waterline_ratio"])
    sr_4in = float(specs["SR_4in"])
    usca_4in_pct_exact = 100.0 * b4_m / l_oa_m
    usca_4in_pct = float(specs["USCA_4in_pct"])

    kn_225 = {int(k): float(v) for k, v in hydro_3["kn_225"].items()}

    loads: dict[int, LoadBaseline] = {}
    for lb in (225, 265, 300):
        ctrl_load = controls["loads"][str(lb)]
        spec_load = hydro_3[f"load_{lb}"]
        gz_raw = tuple(float(x) for x in ctrl_load["gz_mm"])
        gz_rnd = tuple(float(x) for x in spec_load["gz_10_20_30_40_mm"])
        loads[lb] = LoadBaseline(
            load_lb=lb,
            draft_mm=float(ctrl_load["draft_mm"]),
            t_cm=float(spec_load["T_cm"]),
            bwl_cm=float(spec_load["Bwl_cm"]),
            wet_m2=float(spec_load["wet_m2"]),
            gm_cm=float(spec_load["GM_cm"]),
            gz_mm=(gz_raw[0], gz_raw[1], gz_raw[2]),
            gz_rounded_10_20_30_40_mm=(gz_rnd[0], gz_rnd[1], gz_rnd[2], gz_rnd[3]),
        )

    cuts = {
        name: FairnessCutBaseline(
            peak_turn_change_deg=float(vals["peak_turn_change_deg"]),
            distance_aft_m=float(vals["distance_aft_m"]),
            rms_turn_change_deg=float(vals["rms_turn_change_deg"]),
            max_signed_turn_forward_deg=(
                None
                if vals.get("max_signed_turn_forward_deg") is None
                else float(vals["max_signed_turn_forward_deg"])
            ),
        )
        for name, vals in fairness_raw["cuts"].items()
    }
    sections = tuple(
        FairnessSectionBaseline(
            fraction=float(s["fraction"]),
            keel_groove_mm=float(s["keel_groove_mm"]),
            peak_abs_dk_ds_per_m2=(
                None
                if s.get("peak_abs_dk_ds_per_m2") is None
                else float(s["peak_abs_dk_ds_per_m2"])
            ),
        )
        for s in fairness_raw["sections"]
    )
    fairness = FairnessBaseline(
        sha256=str(fairness_raw["sha256"]),
        cuts=cuts,
        sections=sections,
    )

    k225 = capytaine_raw["k10_9_225"]
    k265 = capytaine_raw["k10_9_265_baja"]
    rows_225 = _parse_seakeeping_rows(k225["rows"])
    target_225 = tuple(
        r
        for r in rows_225
        if r.speed_kn in TARGET_CAPYTAINE_SPEEDS_KN
        and r.heading in TARGET_CAPYTAINE_HEADINGS
        and r.sea in TARGET_CAPYTAINE_SEAS
    )
    rows_265 = _parse_seakeeping_rows(k265["rows"])
    capytaine = CapytaineBaseline(
        outer_stl_sha256=str(capytaine_raw["metadata"]["kayak_10_9_outer_stl_sha256"]),
        capytaine_version=str(capytaine_raw["metadata"]["capytaine_version"]),
        gm_t_cm=float(k225["gm_t_cm"]),
        roll_period_s=float(k225["roll_period_s"]),
        added_mass_surge_kg=float(k225["added_mass_surge_kg"]),
        added_mass_sway_kg=float(k225["added_mass_sway_kg"]),
        draft_cm=float(k225["info"]["draft_cm"]),
        info_225=dict(k225["info"]),
        target_conditions_225=target_225,
        rows_225=rows_225,
        gm_t_cm_265_baja=float(k265["gm_t_cm"]),
        roll_period_s_265_baja=float(k265["roll_period_s"]),
        rows_265_baja=rows_265,
    )

    of_meta = openfoam_raw["metadata"]
    openfoam = OpenFoamBaseline(
        outer_stl_sha256=str(of_meta["kayak_10_9_outer_stl_sha256"]),
        openfoam_version=str(of_meta["openfoam_version"]),
        pose_kayak_10_9=dict(of_meta["poses"]["kayak_10_9"]),
        speed_5_5kn=_parse_openfoam_speed(openfoam_raw["kayak_10_9_5.5kn"]),
        speed_6_2kn=_parse_openfoam_speed(openfoam_raw["kayak_10_9_6.2kn"]),
    )

    return BaselineReference(
        hull=hull,
        design_hash=hull.design_hash(),
        record_hash=hull.record_hash(),
        outer_stl_sha256=str(controls["sha256"]),
        l_oa_m=l_oa_m,
        b_oa_cm=b_oa_cm,
        l4_m=l4_m,
        b4_m=b4_m,
        sr_4in_exact=sr_4in_exact,
        sr_4in=sr_4in,
        usca_4in_pct_exact=usca_4in_pct_exact,
        usca_4in_pct=usca_4in_pct,
        mass_at_4in_kg=float(controls["mass_at_4in_kg"]),
        kn_225=kn_225,
        loads=loads,
        catch_widths_mm=tuple(float(x) for x in controls["catch_widths_mm"]),
        catch_beams_225_cm={str(k): float(v) for k, v in hydro_3["catch_beams_225_cm"].items()},
        transition_beams_cm={str(k): float(v) for k, v in hydro_3["transition_beams_cm"].items()},
        transition_dB_per_20cm=tuple(float(x) for x in hydro_3["transition_dB_per_20cm"]),
        fairness=fairness,
        capytaine=capytaine,
        openfoam=openfoam,
    )
