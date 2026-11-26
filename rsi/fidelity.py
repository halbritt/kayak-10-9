"""Tier 2 high-fidelity verification: Capytaine 3.0 BEM and OpenFOAM-v2606 RANS."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import trimesh

from kayakgen.model.hull import Hull

from rsi.baseline import BaselineReference

DEFAULT_OPENFOAM_BASHRC = Path("/usr/lib/openfoam/openfoam2606/etc/bashrc")


def _ensure_cad_on_path() -> None:
    cad_dir = str((Path(__file__).resolve().parents[1] / "cad").resolve())
    if cad_dir not in sys.path:
        sys.path.insert(0, cad_dir)


def resolve_openfoam_bashrc(bashrc: Path | None = None) -> Path:
    if bashrc is not None:
        return Path(bashrc)
    env_val = os.environ.get("OPENFOAM_BASHRC")
    if env_val:
        return Path(env_val)
    return DEFAULT_OPENFOAM_BASHRC


@dataclass(frozen=True)
class SeakeepingMetrics:
    valid: bool
    error: str | None
    quick: bool
    elapsed_s: float
    gm_t_cm: float | None = None
    roll_period_s: float | None = None
    added_mass_surge_kg: float | None = None
    added_mass_sway_kg: float | None = None
    info: dict[str, Any] | None = None
    rows: tuple[dict[str, Any], ...] = ()
    output_json_path: str = ""
    baja_265: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rows"] = list(self.rows or ())
        return d


def resolve_tier2_cache_root() -> Path | None:
    """Resolve content-addressed SSD3 Tier 2 cache root (None when disabled)."""
    if os.environ.get("KAYAK_RSI_NO_TIER2_CACHE", "").strip().lower() in ("1", "true", "yes"):
        return None
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    cache_base = Path(xdg) if xdg else (Path.home() / ".cache")
    return cache_base / "kayak-10-9" / "tier2-cache"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _single_thread_blas_env(enabled: bool):
    if not enabled:
        yield
        return
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    prev = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _clean_spectra_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in r.items()
        }
        for r in raw_rows
    ]


def _seakeeping_cache_key(hull: Hull, *, quick: bool) -> tuple[str, dict[str, Any]]:
    key_obj = {
        "solver": "capytaine_seakeeping",
        # 3: build_bem_body adds a lid mesh (irregular-frequency removal). Rows cached under 1 and 2
        # came from a body without one and are wrong at 4.5 kn bow-quarter.
        "schema_version": 3,
        "design_hash": hull.design_hash(),
        "quick": bool(quick),
        "loads": ["225"] if quick else ["225", "265_baja"],
        "speeds_225_kn": [0.0, 4.5] if quick else [0.0, 4.5, 5.7],
        "speeds_265_kn": [] if quick else [0.0, 4.5],
        "kg_m": 0.25,
        "nx": 64,
        "ng": 10,
    }
    digest = hashlib.sha256(
        json.dumps(key_obj, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()[:24]
    return f"seak_{digest}", key_obj


def _cfd_cache_key(
    hull: Hull,
    *,
    speeds_kn: tuple[float, ...],
    end_steps: int,
    bashrc: Path,
    mesh_resolution: tuple[int, int] = (441, 129),
) -> tuple[str, dict[str, Any]]:
    key_obj = {
        "solver": "openfoam_cfd",
        "schema_version": 1,
        "design_hash": hull.design_hash(),
        "speeds_kn": [float(s) for s in speeds_kn],
        "end_steps": int(end_steps),
        "mesh_resolution": [int(mesh_resolution[0]), int(mesh_resolution[1])],
        "bashrc": str(bashrc),
    }
    digest = hashlib.sha256(
        json.dumps(key_obj, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()[:24]
    return f"cfd_{digest}", key_obj


@dataclass(frozen=True)
class CfdMetrics:
    valid: bool
    error: str | None
    dry_run: bool
    elapsed_s: float
    bashrc: str
    bashrc_exists: bool
    staging_dir: str
    posed: dict[str, Any] | None = None
    mesh_info: dict[str, Any] | None = None
    speeds: dict[str, dict[str, Any]] | None = None
    output_json_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _solve_bem_subset(
    body: Any,
    I_mat: np.ndarray,
    info: dict[str, Any],
    *,
    periods: np.ndarray,
    zero_labels: tuple[str, ...],
    fwd_cases: tuple[tuple[float, tuple[str, ...]], ...],
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Shared Capytaine subset BEM solver for quick and gated-only Tier 2 modes."""
    _ensure_cad_on_path()
    import capytaine as cpt
    import xarray as xr
    from hydro import G, KN, RHO_SEAWATER
    from run_capytaine_seakeeping import HEADINGS

    K = body.compute_hydrostatic_stiffness(rho=RHO_SEAWATER, g=G)
    Ixr = xr.DataArray(I_mat, coords=K.coords, dims=K.dims)
    solver = cpt.BEMSolver()
    kw = dict(body=body, rho=RHO_SEAWATER, g=G, water_depth=np.inf)
    out: dict[str, Any] = {
        "info": info,
        "K": K.values,
        "I": I_mat,
        "gm_t_cm": round(float(K.values[3, 3] / (info["mass_kg"] * G)) * 100.0, 2),
        "cases": {},
    }

    def _run_ds(u_ms: float, per: np.ndarray, betas: list[float]) -> xr.Dataset:
        pbs = []
        for per_s in per:
            om = 2.0 * np.pi / float(per_s)
            for b in betas:
                pbs.append(
                    cpt.DiffractionProblem(
                        omega=om, wave_direction=np.radians(b), forward_speed=u_ms, **kw
                    )
                )
                if u_ms > 0.0 or b == betas[0]:
                    pbs.extend(
                        [
                            cpt.RadiationProblem(
                                omega=om,
                                wave_direction=np.radians(b),
                                forward_speed=u_ms,
                                radiating_dof=d,
                                **kw,
                            )
                            for d in body.dofs
                        ]
                    )
        res = solver.solve_all(
            pbs,
            n_jobs=n_jobs,
            n_threads=1 if n_jobs > 1 else None,
            progress_bar=False,
        )
        ds = cpt.assemble_dataset(res, hydrostatics=False)
        ds["inertia_matrix"], ds["hydrostatic_stiffness"] = Ixr, K
        return ds

    def _pack(ds: xr.Dataset, u_kn: float, label: str, b: float) -> None:
        d = (
            ds.sel(wave_direction=np.radians(b), method="nearest")
            if "wave_direction" in ds.dims
            else ds
        )
        om = d.omega.values
        ome = (
            np.squeeze(d.encounter_omega.transpose("omega", ...).values)
            if "encounter_omega" in d.coords
            else om
        )
        d = d.squeeze(drop=True)
        A = d.added_mass.transpose("omega", "influenced_dof", "radiating_dof").values
        B_mat = d.radiation_damping.transpose("omega", "influenced_dof", "radiating_dof").values
        Fx = d.excitation_force.transpose("omega", "influenced_dof").values
        o = np.argsort(om)
        out["cases"][(u_kn, label)] = {
            "omega": om[o],
            "omega_e": np.asarray(ome)[o] if np.ndim(ome) else np.full(len(om), ome),
            "A": A[o],
            "B": B_mat[o],
            "F": Fx[o],
            "beta": b,
        }

    ds0 = _run_ds(0.0, periods, [HEADINGS[lbl] for lbl in zero_labels])
    for lbl in zero_labels:
        _pack(ds0, 0.0, lbl, HEADINGS[lbl])

    for u_kn, labels in fwd_cases:
        ds_u = _run_ds(u_kn * KN, periods, [HEADINGS[lbl] for lbl in labels])
        for lbl in labels:
            _pack(ds_u, u_kn, lbl, HEADINGS[lbl])

    return out


def _solve_bem_quick(
    body: Any,
    I_mat: np.ndarray,
    info: dict[str, Any],
    *,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Targeted subset Capytaine solve that satisfies evaluate_spectra dependencies."""
    _ensure_cad_on_path()
    from run_capytaine_seakeeping import PERIODS

    return _solve_bem_subset(
        body,
        I_mat,
        info,
        periods=PERIODS[::2],
        zero_labels=("head", "beam"),
        fwd_cases=((4.5, ("head",)),),
        n_jobs=n_jobs,
    )


def _solve_bem_full_fast(
    body: Any,
    I_mat: np.ndarray,
    info: dict[str, Any],
    *,
    speeds_kn: tuple[float, ...],
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Full-spectrum 5-heading BEM solver matching solve_bem_case with single solve_all per speed."""
    _ensure_cad_on_path()
    import capytaine as cpt
    import xarray as xr
    from hydro import G, KN, RHO_SEAWATER
    from run_capytaine_seakeeping import HEADINGS, PERIODS

    K = body.compute_hydrostatic_stiffness(rho=RHO_SEAWATER, g=G)
    Ixr = xr.DataArray(I_mat, coords=K.coords, dims=K.dims)
    solver = cpt.BEMSolver()
    kw = dict(body=body, rho=RHO_SEAWATER, g=G, water_depth=np.inf)
    out: dict[str, Any] = {
        "info": info,
        "K": K.values,
        "I": I_mat,
        "gm_t_cm": round(float(K.values[3, 3] / (info["mass_kg"] * G)) * 100.0, 2),
        "cases": {},
    }

    def _pack(ds: xr.Dataset, u_kn: float, label: str, b: float) -> None:
        d = (
            ds.sel(wave_direction=np.radians(b), method="nearest")
            if "wave_direction" in ds.dims
            else ds
        )
        om = d.omega.values
        ome = (
            np.squeeze(d.encounter_omega.transpose("omega", ...).values)
            if "encounter_omega" in d.coords
            else om
        )
        d = d.squeeze(drop=True)
        A = d.added_mass.transpose("omega", "influenced_dof", "radiating_dof").values
        B_mat = d.radiation_damping.transpose("omega", "influenced_dof", "radiating_dof").values
        Fx = d.excitation_force.transpose("omega", "influenced_dof").values
        o = np.argsort(om)
        out["cases"][(u_kn, label)] = {
            "omega": om[o],
            "omega_e": np.asarray(ome)[o] if np.ndim(ome) else np.full(len(om), ome),
            "A": A[o],
            "B": B_mat[o],
            "F": Fx[o],
            "beta": b,
        }

    for u_kn in speeds_kn:
        u_ms = float(u_kn) * KN
        if u_kn == 0.0:
            betas = list(HEADINGS.values())
            pbs = []
            for per_s in PERIODS:
                om = 2.0 * np.pi / float(per_s)
                for b in betas:
                    pbs.append(
                        cpt.DiffractionProblem(
                            omega=om, wave_direction=np.radians(b), forward_speed=0.0, **kw
                        )
                    )
                    if b == betas[0]:
                        pbs.extend(
                            cpt.RadiationProblem(
                                omega=om,
                                wave_direction=np.radians(b),
                                forward_speed=0.0,
                                radiating_dof=d,
                                **kw,
                            )
                            for d in body.dofs
                        )
            res = solver.solve_all(
                pbs,
                n_jobs=n_jobs,
                n_threads=1 if n_jobs > 1 else None,
                progress_bar=False,
            )
            ds0 = cpt.assemble_dataset(res, hydrostatics=False)
            ds0["inertia_matrix"], ds0["hydrostatic_stiffness"] = Ixr, K
            for lbl, b in HEADINGS.items():
                _pack(ds0, 0.0, lbl, b)
        else:
            pbs = []
            for lbl, b in HEADINGS.items():
                min_p = (
                    1.05
                    if b >= 90.0
                    else max(1.90, 2.0 * np.pi * u_ms * np.cos(np.radians(b)) / G * 1.35)
                )
                for per_s in PERIODS[PERIODS >= min_p]:
                    om = 2.0 * np.pi / float(per_s)
                    pbs.append(
                        cpt.DiffractionProblem(
                            omega=om, wave_direction=np.radians(b), forward_speed=u_ms, **kw
                        )
                    )
                    pbs.extend(
                        cpt.RadiationProblem(
                            omega=om,
                            wave_direction=np.radians(b),
                            forward_speed=u_ms,
                            radiating_dof=d,
                            **kw,
                        )
                        for d in body.dofs
                    )
            res = solver.solve_all(
                pbs,
                n_jobs=n_jobs,
                n_threads=1 if n_jobs > 1 else None,
                progress_bar=False,
            )
            for lbl, b in HEADINGS.items():
                b_rad = float(np.radians(b))
                res_b = [r for r in res if abs(float(r.wave_direction) - b_rad) < 1e-9]
                ds_b = cpt.assemble_dataset(res_b, hydrostatics=False)
                ds_b["inertia_matrix"], ds_b["hydrostatic_stiffness"] = Ixr, K
                _pack(ds_b, u_kn, lbl, b)

    return out


def _load_seakeeping_cache(
    cache_root: Path | None,
    cache_key: str,
    expected_key: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any] | None:
    if cache_root is None:
        return None
    entry_dir = cache_root / cache_key
    meta_path = entry_dir / "meta.json"
    data_path = entry_dir / "capytaine_seakeeping.json"
    if not meta_path.is_file() or not data_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            not isinstance(meta, dict)
            or meta.get("source") != "live_capytaine"
            or meta.get("key") != expected_key
        ):
            return None
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or bool(raw.get("quick", False)) != bool(quick):
            return None
        rows = raw.get("rows")
        if not isinstance(rows, list) or len(rows) == 0:
            return None
        for val_name in ("gm_t_cm", "roll_period_s", "added_mass_surge_kg", "added_mass_sway_kg"):
            v = raw.get(val_name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                return None
        if not quick:
            baja = raw.get("baja_265")
            if not isinstance(baja, dict) or not isinstance(baja.get("rows"), list) or not baja["rows"]:
                return None
            for b_name in ("gm_t_cm", "roll_period_s"):
                bv = baja.get(b_name)
                if not isinstance(bv, (int, float)) or isinstance(bv, bool) or not math.isfinite(float(bv)):
                    return None
        return raw
    except Exception:  # noqa: BLE001
        return None


def evaluate_tier2_seakeeping(
    hull: Hull,
    work_dir: Path,
    baseline: BaselineReference,
    *,
    quick: bool = False,
    n_jobs: int = 1,
) -> SeakeepingMetrics:
    """Run Capytaine 3.0 3D BEM seakeeping on a candidate hull in SSD3 work_dir."""
    del baseline
    t0 = time.monotonic()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    hull_path = work / "candidate.hull.json"
    hull_path.write_text(hull.model_dump_json(indent=2) + "\n")
    out_json = work / "capytaine_seakeeping.json"

    cache_root = resolve_tier2_cache_root()
    cache_key, key_obj = _seakeeping_cache_key(hull, quick=quick)
    cached_payload = _load_seakeeping_cache(cache_root, cache_key, key_obj, quick=quick)
    if cached_payload is not None:
        _atomic_write_json(out_json, cached_payload)
        return SeakeepingMetrics(
            valid=True,
            error=None,
            quick=quick,
            elapsed_s=round(time.monotonic() - t0, 3),
            gm_t_cm=float(cached_payload["gm_t_cm"]),
            roll_period_s=float(cached_payload["roll_period_s"]),
            added_mass_surge_kg=float(cached_payload["added_mass_surge_kg"]),
            added_mass_sway_kg=float(cached_payload["added_mass_sway_kg"]),
            info=dict(cached_payload.get("info") or {}),
            rows=tuple(cached_payload["rows"]),
            output_json_path=str(out_json),
            baja_265=dict(cached_payload["baja_265"]) if cached_payload.get("baja_265") else None,
        )

    try:
        _ensure_cad_on_path()
        import capytaine as cpt
        from deck_geometry import SculptedKayakSurface
        from hydro import LB
        from run_capytaine_seakeeping import build_bem_body, evaluate_spectra

        surf = SculptedKayakSurface(hull_path)
        closed = trimesh.util.concatenate(
            (surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False))
        )
        closed.merge_vertices(merge_tex=True, merge_norm=True)
        if not closed.is_watertight or not closed.is_volume:
            return SeakeepingMetrics(
                valid=False,
                error="Closed outer shell is not watertight/volume",
                quick=quick,
                elapsed_s=round(time.monotonic() - t0, 3),
                output_json_path=str(out_json),
            )

        # SculptedKayakSurface has bow at -X; Capytaine BEM requires bow at +X
        V = np.asarray(closed.vertices, dtype=float).copy()
        F = np.asarray(closed.faces, dtype=int).copy()
        V[:, 0] = -V[:, 0]
        F = F[:, ::-1]
        if trimesh.Trimesh(V, F, process=False).volume < 0:
            F = F[:, ::-1]
        V[:, 0] -= V[:, 0].min()
        V[:, 1] -= np.median(V[:, 1])
        V[:, 2] -= V[:, 2].min()

        has_joblib = importlib.util.find_spec("joblib") is not None
        cpu_cnt = os.cpu_count() or 1
        eff_jobs = min(max(1, int(n_jobs)), cpu_cnt) if (n_jobs > 1 and has_joblib) else 1

        baja_265_payload: dict[str, Any] | None = None
        with contextlib.redirect_stdout(sys.stderr), _single_thread_blas_env(eff_jobs > 1):
            mass_225 = 225.0 * LB
            body, I_mat, info = build_bem_body("candidate_225", V, F, mass_kg=mass_225, kg_m=0.25)

            if quick:
                bem_out = _solve_bem_quick(body, I_mat, info, n_jobs=eff_jobs)
            else:
                bem_out = _solve_bem_full_fast(
                    body, I_mat, info, speeds_kn=(0.0, 4.5, 5.7), n_jobs=eff_jobs
                )
            spectra = evaluate_spectra(bem_out)

            if not quick:
                mass_265 = 265.0 * LB
                body_265, I_mat_265, info_265 = build_bem_body(
                    "candidate_265_baja", V, F, mass_kg=mass_265, kg_m=0.25
                )
                bem_out_265 = _solve_bem_full_fast(
                    body_265, I_mat_265, info_265, speeds_kn=(0.0, 4.5), n_jobs=eff_jobs
                )
                spectra_265 = evaluate_spectra(bem_out_265)
                baja_265_payload = {
                    "info": spectra_265["info"],
                    "gm_t_cm": float(spectra_265["gm_t_cm"]),
                    "roll_period_s": float(spectra_265["roll_period_s"]),
                    "added_mass_surge_kg": float(spectra_265["added_mass_surge_kg"]),
                    "added_mass_sway_kg": float(spectra_265["added_mass_sway_kg"]),
                    "rows": _clean_spectra_rows(spectra_265["rows"]),
                }

        clean_rows = _clean_spectra_rows(spectra["rows"])
        payload: dict[str, Any] = {
            "quick": quick,
            "info": spectra["info"],
            "gm_t_cm": float(spectra["gm_t_cm"]),
            "roll_period_s": float(spectra["roll_period_s"]),
            "added_mass_surge_kg": float(spectra["added_mass_surge_kg"]),
            "added_mass_sway_kg": float(spectra["added_mass_sway_kg"]),
            "rows": clean_rows,
            "baja_265": baja_265_payload,
        }
        _atomic_write_json(out_json, payload)
        elapsed = round(time.monotonic() - t0, 3)

        if cache_root is not None:
            entry_dir = cache_root / cache_key
            _atomic_write_json(entry_dir / "capytaine_seakeeping.json", payload)
            _atomic_write_json(
                entry_dir / "meta.json",
                {
                    "source": "live_capytaine",
                    "capytaine_version": getattr(cpt, "__version__", "unknown"),
                    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "elapsed_s": elapsed,
                    "n_jobs": eff_jobs,
                    "key": key_obj,
                },
            )

        return SeakeepingMetrics(
            valid=True,
            error=None,
            quick=quick,
            elapsed_s=elapsed,
            gm_t_cm=float(spectra["gm_t_cm"]),
            roll_period_s=float(spectra["roll_period_s"]),
            added_mass_surge_kg=float(spectra["added_mass_surge_kg"]),
            added_mass_sway_kg=float(spectra["added_mass_sway_kg"]),
            info=dict(spectra["info"]),
            rows=tuple(clean_rows),
            output_json_path=str(out_json),
            baja_265=baja_265_payload,
        )
    except Exception as exc:  # noqa: BLE001
        return SeakeepingMetrics(
            valid=False,
            error=str(exc),
            quick=quick,
            elapsed_s=round(time.monotonic() - t0, 3),
            output_json_path=str(out_json),
        )


def stage_openfoam_case(
    hull: Hull,
    work_dir: Path,
    *,
    speeds_kn: tuple[float, ...] = (5.5, 6.2),
    end_steps: int = 500,
    np_ranks: int = 6,
    bashrc: Path | None = None,
    hull_name: str = "candidate",
    mesh_resolution: tuple[int, int] = (441, 129),
) -> dict[str, Any]:
    """Stage OpenFOAM-v2606 STL, mesh dicts, and interFoam case directories on SSD3."""
    _ensure_cad_on_path()
    from deck_geometry import SculptedKayakSurface
    from hydro import KN
    from openfoam_dicts import write_interfoam_case_files, write_mesh_dicts
    from run_openfoam_cfd import pose_for_cfd

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    hull_path = work / f"{hull_name}.hull.json"
    hull_path.write_text(hull.model_dump_json(indent=2) + "\n")

    surf = SculptedKayakSurface(
        hull_path, n_stations=mesh_resolution[0], n_hull_half=mesh_resolution[1]
    )
    closed = trimesh.util.concatenate(
        (surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False))
    )
    closed.merge_vertices(merge_tex=True, merge_norm=True)
    if not closed.is_watertight or not closed.is_volume:
        raise ValueError("Closed outer shell must be watertight and a valid volume for CFD")

    posed = pose_for_cfd(
        hull_name,
        np.asarray(closed.vertices, dtype=float),
        np.asarray(closed.faces, dtype=int),
        work,
    )

    mesh_dir = work / f"mesh_{hull_name}"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    for sub in ("system", "constant/triSurface", "logs"):
        (mesh_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / f"{hull_name}.stl", mesh_dir / "constant/triSurface/hull.stl")
    write_mesh_dicts(mesh_dir, np_ranks=np_ranks)

    run_dirs: dict[str, str] = {}
    for sp in speeds_kn:
        tag = f"run_{hull_name}_{sp:.1f}kn".replace(".", "p")
        cdir = work / tag
        if cdir.exists():
            shutil.rmtree(cdir)
        for sub in ("system", "constant", "logs"):
            (cdir / sub).mkdir(parents=True, exist_ok=True)
        write_interfoam_case_files(
            cdir, sp * KN, posed["cg_cfd_m"], end_steps=end_steps, np_ranks=np_ranks
        )
        shutil.copytree(cdir / "0.orig", cdir / "0")
        run_dirs[f"{sp:.1f}kn"] = str(cdir)

    resolved_bashrc = resolve_openfoam_bashrc(bashrc)
    manifest = {
        "hull_name": hull_name,
        "bashrc": str(resolved_bashrc),
        "bashrc_exists": resolved_bashrc.is_file(),
        "speeds_kn": list(speeds_kn),
        "end_steps": int(end_steps),
        "np_ranks": int(np_ranks),
        "posed": posed,
        "mesh_dir": str(mesh_dir),
        "run_dirs": run_dirs,
    }
    (work / "cfd_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_openfoam_case_metrics(
    case_dir: Path,
    posed: dict[str, Any],
    speed_kn: float,
    *,
    hull_name: str = "candidate",
) -> dict[str, Any]:
    """Parse force.dat, moment.dat, and optional wave fields from an OpenFOAM case dir."""
    _ensure_cad_on_path()
    from hydro import G, KN
    from run_openfoam_cfd import MASS_225, _free_surface_and_hull_wave, _parse_forces

    cdir = Path(case_dir)
    force_files = sorted((cdir / "postProcessing/forces").glob("*/force.dat"))
    moment_files = sorted((cdir / "postProcessing/forces").glob("*/moment.dat"))
    if not force_files or not moment_files:
        raise FileNotFoundError(
            f"Missing OpenFOAM postProcessing/forces/force.dat or moment.dat in {cdir}"
        )

    st, fp, fv = _parse_forces(force_files[-1])
    mt, mp, mv = _parse_forces(moment_files[-1])
    if len(st) == 0:
        raise ValueError(f"Empty force history in {force_files[-1]}")
    if len(mt) == 0:
        raise ValueError(f"Empty moment history in {moment_files[-1]}")

    ftot = fp + fv
    mtot = mp + mv
    tail = st >= max(st[-1] - 150.0, st[0])
    tail_m = mt >= max(mt[-1] - 150.0, mt[0])
    fx_mean = float(ftot[tail, 0].mean())
    fx_std = float(ftot[tail, 0].std())
    fpx_mean = float(fp[tail, 0].mean())
    fvx_mean = float(fv[tail, 0].mean())
    fz_mean = float(ftot[tail, 2].mean())
    my_mean = float(mtot[tail_m, 1].mean())
    u_mps = float(speed_kn) * KN

    bow_wave_crest_mm: float | None = None
    mid_wave_trough_mm: float | None = None
    step_dir = str(int(round(float(st[-1]))))
    tdir = cdir / step_dir
    if (tdir / "C").is_file() and (tdir / "V").is_file() and (tdir / "alpha.water").is_file():
        _, (xs_w, zw) = _free_surface_and_hull_wave(cdir, step_dir)
        fwd_mask = xs_w < -1.5
        if np.any(np.isfinite(zw[fwd_mask])):
            bow_wave_crest_mm = round(float(np.nanmax(zw[fwd_mask]) * 1000.0), 1)
        if np.any(np.isfinite(zw)):
            mid_wave_trough_mm = round(float(np.nanmin(zw) * 1000.0), 1)

    return {
        "hull": hull_name,
        "speed_kn": float(speed_kn),
        "u_mps": round(u_mps, 4),
        "drag_total_N": round(fx_mean, 3),
        "drag_std_N": round(fx_std, 3),
        "drag_pressure_N": round(fpx_mean, 3),
        "drag_viscous_N": round(fvx_mean, 3),
        "effective_power_W": round(fx_mean * u_mps, 2),
        "vertical_force_N": round(fz_mean, 2),
        "pitch_moment_My_Nm": round(my_mean, 3),
        "bow_up_running_trim_deg": round(
            float(np.degrees(my_mean / (MASS_225 * G * float(posed["gm_l_m"])))), 3
        ),
        "bow_wave_crest_mm": bow_wave_crest_mm,
        "mid_wave_trough_mm": mid_wave_trough_mm,
    }


def _load_cfd_cache(
    cache_root: Path | None,
    cache_key: str,
    expected_key: dict[str, Any],
    speeds_kn: tuple[float, ...],
) -> dict[str, Any] | None:
    if cache_root is None:
        return None
    entry_dir = cache_root / cache_key
    meta_path = entry_dir / "meta.json"
    data_path = entry_dir / "openfoam_cfd_metrics.json"
    if not meta_path.is_file() or not data_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            not isinstance(meta, dict)
            or meta.get("source") != "live_openfoam"
            or meta.get("key") != expected_key
        ):
            return None
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        speeds = raw.get("speeds")
        if not isinstance(speeds, dict):
            return None
        for sp in speeds_kn:
            sp_key = f"{sp:.1f}kn"
            sp_entry = speeds.get(sp_key)
            if not isinstance(sp_entry, dict):
                return None
            drag = sp_entry.get("drag_total_N")
            if (
                not isinstance(drag, (int, float))
                or isinstance(drag, bool)
                or not math.isfinite(float(drag))
            ):
                return None
        return raw
    except Exception:  # noqa: BLE001
        return None


def evaluate_tier2_cfd(
    hull: Hull,
    work_dir: Path,
    baseline: BaselineReference,
    *,
    speeds_kn: tuple[float, ...] = (5.5, 6.2),
    end_steps: int = 500,
    np_ranks: int = 6,
    bashrc: Path | None = None,
    dry_run: bool = False,
) -> CfdMetrics:
    """Stage and optionally execute OpenFOAM-v2606 RANS CFD in SSD3 work_dir."""
    del baseline
    t0 = time.monotonic()
    work = Path(work_dir)
    resolved_bashrc = resolve_openfoam_bashrc(bashrc)
    out_json = work / "openfoam_cfd_metrics.json"
    hull_name = "candidate"

    cache_root = None if dry_run else resolve_tier2_cache_root()
    cache_key, key_obj = _cfd_cache_key(
        hull,
        speeds_kn=speeds_kn,
        end_steps=end_steps,
        bashrc=resolved_bashrc,
    )
    if not dry_run and resolved_bashrc.is_file():
        cached_payload = _load_cfd_cache(cache_root, cache_key, key_obj, speeds_kn)
        if cached_payload is not None:
            work.mkdir(parents=True, exist_ok=True)
            (work / f"{hull_name}.hull.json").write_text(hull.model_dump_json(indent=2) + "\n")
            _atomic_write_json(out_json, cached_payload)
            return CfdMetrics(
                valid=True,
                error=None,
                dry_run=False,
                elapsed_s=round(time.monotonic() - t0, 3),
                bashrc=str(resolved_bashrc),
                bashrc_exists=True,
                staging_dir=str(work),
                posed=cached_payload.get("posed"),
                mesh_info=cached_payload.get("mesh_info"),
                speeds=cached_payload.get("speeds"),
                output_json_path=str(out_json),
            )

    try:
        with contextlib.redirect_stdout(sys.stderr):
            manifest = stage_openfoam_case(
                hull,
                work,
                speeds_kn=speeds_kn,
                end_steps=end_steps,
                np_ranks=np_ranks,
                bashrc=resolved_bashrc,
                hull_name=hull_name,
            )
    except Exception as exc:  # noqa: BLE001
        return CfdMetrics(
            valid=False,
            error=f"OpenFOAM staging failed: {exc}",
            dry_run=dry_run,
            elapsed_s=round(time.monotonic() - t0, 3),
            bashrc=str(resolved_bashrc),
            bashrc_exists=resolved_bashrc.is_file(),
            staging_dir=str(work),
            output_json_path=str(out_json),
        )

    posed = manifest["posed"]
    if dry_run:
        return CfdMetrics(
            valid=True,
            error=None,
            dry_run=True,
            elapsed_s=round(time.monotonic() - t0, 3),
            bashrc=str(resolved_bashrc),
            bashrc_exists=resolved_bashrc.is_file(),
            staging_dir=str(work),
            posed=posed,
            speeds={},
            output_json_path=str(out_json),
        )

    if not resolved_bashrc.is_file():
        return CfdMetrics(
            valid=False,
            error=f"OpenFOAM bashrc not found: {resolved_bashrc}",
            dry_run=False,
            elapsed_s=round(time.monotonic() - t0, 3),
            bashrc=str(resolved_bashrc),
            bashrc_exists=False,
            staging_dir=str(work),
            posed=posed,
            output_json_path=str(out_json),
        )

    try:
        _ensure_cad_on_path()
        from run_openfoam_cfd import build_mesh, run_interfoam_case

        with contextlib.redirect_stdout(sys.stderr):
            mesh_info = build_mesh(hull_name, work, resolved_bashrc, np_ranks=np_ranks)

            def _solve_speed(sp: float) -> tuple[str, dict[str, Any]]:
                cdir, _ = run_interfoam_case(
                    hull_name,
                    sp,
                    posed,
                    work,
                    resolved_bashrc,
                    end_steps=end_steps,
                    np_ranks=np_ranks,
                )
                return f"{sp:.1f}kn", parse_openfoam_case_metrics(
                    cdir, posed, sp, hull_name=hull_name
                )

            cpu_cnt = os.cpu_count() or 1
            concurrent_slots = min(len(speeds_kn), cpu_cnt // max(1, int(np_ranks)))
            if len(speeds_kn) > 1 and concurrent_slots >= 2:
                with ThreadPoolExecutor(max_workers=concurrent_slots) as pool:
                    speed_pairs = list(pool.map(_solve_speed, speeds_kn))
                speed_metrics = dict(speed_pairs)
            else:
                speed_metrics = dict(_solve_speed(sp) for sp in speeds_kn)

        payload = {
            "posed": posed,
            "mesh_info": mesh_info,
            "speeds": speed_metrics,
        }
        _atomic_write_json(out_json, payload)
        elapsed = round(time.monotonic() - t0, 3)

        if cache_root is not None:
            entry_dir = cache_root / cache_key
            _atomic_write_json(entry_dir / "openfoam_cfd_metrics.json", payload)
            _atomic_write_json(
                entry_dir / "meta.json",
                {
                    "source": "live_openfoam",
                    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "elapsed_s": elapsed,
                    "np_ranks": int(np_ranks),
                    "key": key_obj,
                },
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
            mesh_info=mesh_info,
            speeds=speed_metrics,
            output_json_path=str(out_json),
        )
    except Exception as exc:  # noqa: BLE001
        return CfdMetrics(
            valid=False,
            error=f"OpenFOAM execution failed: {exc}",
            dry_run=False,
            elapsed_s=round(time.monotonic() - t0, 3),
            bashrc=str(resolved_bashrc),
            bashrc_exists=resolved_bashrc.is_file(),
            staging_dir=str(work),
            posed=posed,
            output_json_path=str(out_json),
        )
