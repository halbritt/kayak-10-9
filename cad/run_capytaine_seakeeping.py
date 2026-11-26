"""Run Capytaine 3.0 3D linear BEM seakeeping, Munk/aero windage balance & Plate 6 for Kayak 10.9."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import capytaine as cpt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import xarray as xr

from deck_geometry import SculptedKayakSurface
from hydro import G, KN, LB, RHO_SEAWATER, STAGE9_PRE_FINISH_JSON, float_at, half_breadth_grid, load_epic_18x_arrays
from kayakgen.eval.closed_volume import generated_hull_plus_deck_body
from kayakgen.model.hull import Hull

cpt.set_logging("ERROR")
RHO_AIR = 1.225
HEADINGS = {"head": 180.0, "bow_quarter": 135.0, "beam": 90.0, "stern_quarter": 45.0, "following": 0.0}
PERIODS = np.concatenate([np.arange(1.05, 3.0, 0.15), np.arange(3.0, 5.0, 0.25), np.arange(5.0, 9.01, 0.5)])
# Highest forward speed any case here is solved at. With PERIODS it fixes the highest encounter
# frequency the BEM sees (head seas, shortest period), which is what the lid has to cover.
LID_SPEED_KN = 5.7
LID_OMEGA_MARGIN = 1.25
SEAS = {
    "Bay chop (Hs 0.35m, Tp 2.5s)": (0.35, 2.5),
    "Steep whitecaps (Hs 0.65m, Tp 3.8s)": (0.65, 3.8),
    "Cortez swell (Hs 1.10m, Tp 6.5s)": (1.10, 6.5),
}


def _load_kayak_10_9_pos_bow() -> tuple[np.ndarray, np.ndarray, str]:
    surf = SculptedKayakSurface(REPO_ROOT / "hull" / "kayak-10-9.hull.json")
    closed = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
    closed.merge_vertices(merge_tex=True, merge_norm=True)
    sha = hashlib.sha256(closed.export(file_type="stl")).hexdigest()
    V = np.asarray(closed.vertices, float).copy()
    F = np.asarray(closed.faces, int).copy()
    V[:, 0] = -V[:, 0]
    F = F[:, ::-1]
    if trimesh.Trimesh(V, F, process=False).volume < 0:
        F = F[:, ::-1]
    V[:, 0] -= V[:, 0].min()
    V[:, 1] -= np.median(V[:, 1])
    V[:, 2] -= V[:, 2].min()
    return V, F, sha


def _deck_freeboard_at(V: np.ndarray, F: np.ndarray, xv: float, draft: float) -> float:
    sl = trimesh.intersections.mesh_plane(trimesh.Trimesh(V, F, process=False), [1.0, 0.0, 0.0], [float(xv), 0.0, 0.0]).reshape(-1, 3)
    return float(sl[:, 2].max() - draft)


def _keel_depth_at(V: np.ndarray, F: np.ndarray, xv: float, draft: float) -> float:
    sl = trimesh.intersections.mesh_plane(trimesh.Trimesh(V, F, process=False), [1.0, 0.0, 0.0], [float(xv), 0.0, 0.0]).reshape(-1, 3)
    return max(float(draft - sl[:, 2].min()), 0.002)


def lid_omega_max() -> float:
    """Encounter frequency the irregular-frequency lid must clear, with margin [rad/s]."""
    om = 2.0 * np.pi / float(PERIODS.min())
    return LID_OMEGA_MARGIN * (om + om**2 / G * LID_SPEED_KN * KN)


def build_bem_body(name: str, V: np.ndarray, F: np.ndarray, mass_kg: float, kg_m: float = 0.25, nx: int = 64, ng: int = 10) -> tuple[cpt.FloatingBody, np.ndarray, dict]:
    L = float(np.ptp(V[:, 0]))
    r0 = float_at(V, F, mass_kg, RHO_SEAWATER, np.array([0.5 * L, 0.0, kg_m]), free_trim=False)
    lcg = float(r0["B"][0])
    r = float_at(V, F, mass_kg, RHO_SEAWATER, np.array([lcg, 0.0, kg_m]), free_trim=False)
    xs, zs, Y = half_breadth_grid(r["tris"], 801, 321)
    x0, x1 = r["wl_x"]
    X = x0 + (x1 - x0) * (0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nx + 1))))
    has = Y > 1e-7
    keel = np.array([zs[has[i]].min() if has[i].any() else 0.0 for i in range(len(xs))])
    kz = np.interp(X, xs, keel)
    kz[0] = kz[-1] = 0.0
    t = np.linspace(0.0, 1.0, ng + 1) ** 1.5
    pts = np.zeros((nx + 1, ng + 2, 3))
    for i, xv in enumerate(X):
        zz = kz[i] * (1.0 - t)
        j = int(np.clip(np.searchsorted(xs, xv) - 1, 0, len(xs) - 2))
        w = (xv - xs[j]) / (xs[j + 1] - xs[j])
        yy = np.interp(zz, zs, (1.0 - w) * Y[j] + w * Y[j + 1])
        if i in (0, nx):
            yy[:] = 0.0
        pts[i, 0] = (xv, 0.0, kz[i])
        pts[i, 1:, 0] = xv
        pts[i, 1:, 1] = yy
        pts[i, 1:, 2] = zz
    n = ng + 2

    def idx(i: int, j: int, side: int) -> int:
        return (i * n + j) + side * (nx + 1) * n

    Vp = np.concatenate([pts.reshape(-1, 3), pts.reshape(-1, 3) * [1.0, -1.0, 1.0]])
    Fp = []
    for i in range(nx):
        for j in range(n - 1):
            Fp.append([idx(i, j, 0), idx(i, j + 1, 0), idx(i + 1, j + 1, 0), idx(i + 1, j, 0)])
            Fp.append([idx(i, j, 1), idx(i + 1, j, 1), idx(i + 1, j + 1, 1), idx(i, j + 1, 1)])
    Vp[:, 0] -= lcg
    mesh = cpt.Mesh(Vp, np.array(Fp), name=f"{name}_mesh")
    draft = float(-r["zmin"])
    cg = np.array([0.0, 0.0, kg_m - draft])
    # Irregular-frequency removal. A panel solution of a surface-piercing hull is wrong near the
    # sloshing frequencies of the water that would fill its inside. For this hull (0.48 m beam,
    # 0.09 m draft) the first is near 11 to 12 rad/s, inside the encounter-frequency range of every
    # forward-speed case. Without the lid, v0.2.3 at 4.5 kn bow-quarter read a heave excitation of
    # 10740 N/m at T = 1.05 s (90 N/m with it), and that one point was 28 % of the seat-acceleration
    # integral. Below 5.5 rad/s encounter the lid changes A33, B33 and F3 by under 0.5 %.
    lid_mesh = mesh.generate_lid(z=mesh.lowest_lid_position(omega_max=lid_omega_max()))
    body = cpt.FloatingBody(mesh=mesh, lid_mesh=lid_mesh, center_of_mass=cg, mass=mass_kg, name=name)
    body.add_all_rigid_body_dofs(rotation_center=cg)
    assert body.volume > 0, f"Expected outward BEM normals on {name}"

    x_bow_abs, x_stern_abs = float(V[:, 0].max()), float(V[:, 0].min())
    info = {
        "panels": int(mesh.nb_faces),
        "lid_panels": int(lid_mesh.nb_faces),
        "mass_kg": round(mass_kg, 3),
        "kg_m": round(kg_m, 4),
        "vol_mesh_L": round(float(body.volume) * 1000.0, 2),
        "vol_exact_L": round(float(r["vol"]) * 1000.0, 2),
        "draft_cm": round(draft * 100.0, 2),
        "L_m": round(L, 4),
        "lcg_abs_m": round(lcg, 4),
        "bow_x": float(x_bow_abs - lcg),
        "stern_x": float(x_stern_abs - lcg),
        "bow_fb_3pct_m": _deck_freeboard_at(V, F, x_bow_abs - 0.03 * L, draft),
        "aft_fb_25pct_m": _deck_freeboard_at(V, F, x_stern_abs + 0.25 * L, draft),
        "forefoot_10pct_m": _keel_depth_at(V, F, x_bow_abs - 0.10 * L, draft),
    }
    I_mat = np.diag([mass_kg, mass_kg, mass_kg, mass_kg * 0.20**2, mass_kg * (0.15 * L) ** 2, mass_kg * (0.15 * L) ** 2])
    return body, I_mat, info


def solve_bem_case(body: cpt.FloatingBody, I_mat: np.ndarray, info: dict, speeds_kn: tuple[float, ...], n_jobs: int = 6) -> dict:
    t0 = time.monotonic()
    K = body.compute_hydrostatic_stiffness(rho=RHO_SEAWATER, g=G)
    Ixr = xr.DataArray(I_mat, coords=K.coords, dims=K.dims)
    solver = cpt.BEMSolver()
    kw = dict(body=body, rho=RHO_SEAWATER, g=G, water_depth=np.inf)
    out: dict = dict(info=info, K=K.values, I=I_mat, gm_t_cm=round(float(K.values[3, 3] / (info["mass_kg"] * G)) * 100.0, 2), cases={})

    def _run_ds(u_ms: float, per: np.ndarray, betas: list[float]) -> xr.Dataset:
        pbs = []
        for per_s in per:
            om = 2.0 * np.pi / float(per_s)
            for b in betas:
                pbs.append(cpt.DiffractionProblem(omega=om, wave_direction=np.radians(b), forward_speed=u_ms, **kw))
                if u_ms > 0.0 or b == betas[0]:
                    pbs.extend([cpt.RadiationProblem(omega=om, wave_direction=np.radians(b), forward_speed=u_ms, radiating_dof=d, **kw) for d in body.dofs])
        res = solver.solve_all(pbs, n_jobs=n_jobs, progress_bar=False)
        ds = cpt.assemble_dataset(res, hydrostatics=False)
        ds["inertia_matrix"], ds["hydrostatic_stiffness"] = Ixr, K
        return ds

    def _pack(ds: xr.Dataset, u_kn: float, label: str, b: float) -> None:
        d = ds.sel(wave_direction=np.radians(b), method="nearest") if "wave_direction" in ds.dims else ds
        om = d.omega.values
        ome = np.squeeze(d.encounter_omega.transpose("omega", ...).values) if "encounter_omega" in d.coords else om
        d = d.squeeze(drop=True)
        A = d.added_mass.transpose("omega", "influenced_dof", "radiating_dof").values
        B_mat = d.radiation_damping.transpose("omega", "influenced_dof", "radiating_dof").values
        Fx = d.excitation_force.transpose("omega", "influenced_dof").values
        o = np.argsort(om)
        out["cases"][(u_kn, label)] = dict(omega=om[o], omega_e=np.asarray(ome)[o] if np.ndim(ome) else np.full(len(om), ome), A=A[o], B=B_mat[o], F=Fx[o], beta=b)

    for u_kn in speeds_kn:
        u_ms = u_kn * KN
        if u_kn == 0.0:
            ds0 = _run_ds(0.0, PERIODS, list(HEADINGS.values()))
            for lbl, b in HEADINGS.items():
                _pack(ds0, 0.0, lbl, b)
        else:
            for lbl, b in HEADINGS.items():
                min_p = 1.05 if b >= 90.0 else max(1.90, 2.0 * np.pi * u_ms * np.cos(np.radians(b)) / G * 1.35)
                ds_u = _run_ds(u_ms, PERIODS[PERIODS >= min_p], [b])
                _pack(ds_u, u_kn, lbl, b)
        print(f"  [bem] {body.name} @ {u_kn:.1f} kn solved ({time.monotonic() - t0:.1f}s)", flush=True)
    return out


def _jonswap(w: np.ndarray, hs: float, tp: float, gam: float = 3.3) -> np.ndarray:
    wp = 2.0 * np.pi / tp
    s = np.where(w <= wp, 0.07, 0.09)
    spec = (w**-5) * np.exp(-1.25 * (wp / w) ** 4) * (gam ** np.exp(-((w - wp) ** 2) / (2.0 * s**2 * wp**2)))
    wf = np.linspace(0.3, 12.0, 6000)
    sf = np.where(wf <= wp, 0.07, 0.09)
    spec_f = (wf**-5) * np.exp(-1.25 * (wp / wf) ** 4) * (gam ** np.exp(-((wf - wp) ** 2) / (2.0 * sf**2 * wp**2)))
    return spec * (hs**2 / 16.0) / np.trapezoid(spec_f, wf)


def _interp_complex(w: np.ndarray, w0: np.ndarray, arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(len(w0), -1)
    out = np.stack([np.interp(w, w0, flat[:, j].real) + 1j * np.interp(w, w0, flat[:, j].imag) for j in range(flat.shape[1])], axis=1)
    return out.reshape((len(w),) + arr.shape[1:])


def evaluate_spectra(bem_out: dict, zeta_roll: float = 0.10) -> dict:
    inf = bem_out["info"]
    L = inf["L_m"]
    c_beam = bem_out["cases"][(0.0, "beam")]
    wn = 3.0
    for _ in range(40):
        wn = float(np.sqrt(bem_out["K"][3, 3] / (bem_out["I"][3, 3] + np.interp(wn, c_beam["omega"], c_beam["A"][:, 3, 3].real))))
    c_head0 = bem_out["cases"][(0.0, "head")]
    a11_low, a22_low = float(c_head0["A"][0, 0, 0].real), float(c_head0["A"][0, 1, 1].real)
    xb, x_aft, xf = inf["bow_x"] - 0.03 * L, inf["stern_x"] + 0.25 * L, inf["bow_x"] - 0.10 * L

    rows = []
    for (u_kn, label), c in bem_out["cases"].items():
        w0 = c["omega"]
        w = np.linspace(float(w0.min()), float(w0.max()), 600)
        k, b_rad = (w**2) / G, np.radians(c["beta"])
        A, B_mat, Fx = _interp_complex(w, w0, c["A"]).real, _interp_complex(w, w0, c["B"]).real, _interp_complex(w, w0, c["F"])
        we = np.interp(w, w0, c["omega_e"])
        Bv = np.zeros((6, 6))
        Bv[3, 3] = 2.0 * zeta_roll * bem_out["K"][3, 3] / wn
        X = np.array([np.linalg.solve(-(we[i] ** 2) * (bem_out["I"] + A[i]) - 1j * we[i] * (B_mat[i] + Bv) + bem_out["K"], Fx[i]) for i in range(len(w))])

        def rel_z(x_pt: float) -> np.ndarray:
            return X[:, 2] - x_pt * X[:, 4] - np.exp(1j * k * x_pt * np.cos(b_rad))

        for sea_name, (hs, tp) in SEAS.items():
            S = _jonswap(w, hs, tp)
            cov = float(np.trapezoid(S, w) / (hs**2 / 16.0))

            def mom(H: np.ndarray, p: int = 0) -> float:
                return max(float(np.trapezoid((np.abs(H) ** 2) * S * (np.abs(we) ** p), w)), 1e-18)

            def rate(H: np.ndarray, lim: float) -> float:
                return float(60.0 / (2.0 * np.pi) * np.sqrt(mom(H, 2) / mom(H, 0)) * np.exp(-(lim**2) / (2.0 * mom(H, 0))))

            rows.append({
                "speed_kn": u_kn, "heading": label, "sea": sea_name, "cov": round(cov, 3),
                "heave_sig_cm": round(200.0 * np.sqrt(mom(X[:, 2])), 2),
                "pitch_sig_deg": round(2.0 * np.degrees(np.sqrt(mom(X[:, 4]))), 2),
                "roll_sig_deg": round(2.0 * np.degrees(np.sqrt(mom(X[:, 3]))), 2),
                "bow_rel_sig_cm": round(200.0 * np.sqrt(mom(rel_z(xb))), 2),
                "bow_dips_per_min": round(rate(rel_z(xb), inf["bow_fb_3pct_m"]), 2),
                "aft_dips_per_min": round(rate(rel_z(x_aft), inf["aft_fb_25pct_m"]), 2),
                "forefoot_out_per_min": round(rate(rel_z(xf), inf["forefoot_10pct_m"]), 2),
                "seat_acc_rms_ms2": round(float(np.sqrt(np.trapezoid((np.abs(X[:, 2]) ** 2) * S * (we**4), w))), 3),
                "yaw_mom_rms_Nm": round(float(np.sqrt(np.trapezoid((np.interp(w, w0, np.abs(c["F"][:, 5])) ** 2) * S, w))), 2),
            })
    return {"info": inf, "gm_t_cm": bem_out["gm_t_cm"], "roll_period_s": round(2.0 * np.pi / wn, 2), "added_mass_surge_kg": round(a11_low, 2), "added_mass_sway_kg": round(a22_low, 2), "rows": rows}


def _profile_areas_and_centroids(V: np.ndarray, F: np.ndarray, mass_kg: float) -> dict[str, float]:
    L = float(np.ptp(V[:, 0]))
    r0 = float_at(V, F, mass_kg, RHO_SEAWATER, np.array([0.5 * L, 0.0, 0.25]), free_trim=False)
    lcg = float(r0["B"][0])
    r = float_at(V, F, mass_kg, RHO_SEAWATER, np.array([lcg, 0.0, 0.25]), free_trim=False)
    draft = float(-r["zmin"])
    tm = trimesh.Trimesh(V, F, process=False)
    xs = np.linspace(float(V[:, 0].min()) + 0.015, float(V[:, 0].max()) - 0.015, 220)
    h_air, h_sub = np.zeros_like(xs), np.zeros_like(xs)
    for i, xv in enumerate(xs):
        sl = trimesh.intersections.mesh_plane(tm, [1.0, 0.0, 0.0], [float(xv), 0.0, 0.0]).reshape(-1, 3)
        if len(sl) >= 2:
            h_air[i] = max(float(sl[:, 2].max()) - draft, 0.0)
            h_sub[i] = max(draft - float(sl[:, 2].min()), 0.0)
    x_rel = xs - lcg
    a_air, a_sub = float(np.trapezoid(h_air, xs)), float(np.trapezoid(h_sub, xs))
    return {
        "lcg_m": round(lcg, 4),
        "draft_cm": round(draft * 100.0, 2),
        "A_air_hull_m2": round(a_air, 4),
        "x_CE_hull_cm": round(float(np.trapezoid(h_air * x_rel, xs) / max(a_air, 1e-9)) * 100.0, 2),
        "A_sub_hull_m2": round(a_sub, 4),
        "x_CLR_geom_cm": round(float(np.trapezoid(h_sub * x_rel, xs) / max(a_sub, 1e-9)) * 100.0, 2),
    }


def evaluate_windage_and_rudder(seak_data: dict, Ve: np.ndarray, Fe: np.ndarray, V10: np.ndarray, F10: np.ndarray) -> dict:
    h9 = Hull.model_validate_json(STAGE9_PRE_FINISH_JSON.read_text())
    p9 = generated_hull_plus_deck_body(h9, stations=161, section_points=32).parts[0]
    m9 = trimesh.Trimesh(np.array(p9.vertices, float), np.array(p9.faces, int), process=True)
    V9, F9 = np.asarray(m9.vertices).copy(), np.asarray(m9.faces).copy()
    if m9.volume < 0:
        F9 = F9[:, ::-1]
    V9[:, 0] = -V9[:, 0]; F9 = F9[:, ::-1]
    V9[:, 0] -= V9[:, 0].min(); V9[:, 1] -= np.median(V9[:, 1]); V9[:, 2] -= V9[:, 2].min()

    m225, u_boat = 225.0 * LB, 5.0 * KN
    a_rud, cl_alpha, lever_rud = 0.166 * 0.042, 2.0 * np.pi / (1.0 + 2.0 / 3.9), 2.76
    q_water = 0.5 * RHO_SEAWATER * (u_boat**2)
    cd_a_torso, x_torso = 0.22 * 1.05, -0.05

    out = {}
    for label, V, F, seak_key in [
        ("Epic 18X", Ve, Fe, "epic18x_225"),
        ("Stage 9 (18.5 cm rear slab)", V9, F9, "k10_9_225"),
        ("10.9 (14.8 cm low rear deck)", V10, F10, "k10_9_225"),
    ]:
        prof = _profile_areas_and_centroids(V, F, m225)
        a22, a11 = seak_data[seak_key]["added_mass_sway_kg"], seak_data[seak_key]["added_mass_surge_kg"]
        cd_a_hull = prof["A_air_hull_m2"] * 0.72
        cd_a_tot = cd_a_hull + cd_a_torso
        x_ce_tot = (cd_a_hull * (prof["x_CE_hull_cm"] / 100.0) + cd_a_torso * x_torso) / cd_a_tot
        ar_hull = 2.0 * ((prof["draft_cm"] / 100.0) ** 2) / max(prof["A_sub_hull_m2"], 1e-4)
        cy_beta = 0.5 * np.pi * ar_hull + 0.20
        dx_munk = 2.0 * (a22 - a11) / (RHO_SEAWATER * prof["A_sub_hull_m2"] * cy_beta) * 0.65
        x_cp_dyn = (prof["x_CLR_geom_cm"] / 100.0) + dx_munk
        prof.update(x_CE_total_cm=round(x_ce_tot * 100.0, 2), x_CP_hydro_dyn_cm=round(x_cp_dyn * 100.0, 2), weathercock_lever_cm=round((x_cp_dyn - x_ce_tot) * 100.0, 2))
        wind_cases = {}
        for vw_kn in (15.0, 22.0):
            fy = 0.5 * RHO_AIR * ((vw_kn * KN) ** 2) * cd_a_tot
            n_yaw = fy * (x_cp_dyn - x_ce_tot)
            delta_rad = n_yaw / (q_water * a_rud * cl_alpha * lever_rud)
            cl_r = cl_alpha * delta_rad
            drag_rud = q_water * a_rud * (0.009 + (cl_r**2) / (np.pi * 3.9 * 0.85))
            wind_cases[f"{int(vw_kn)}kn_beam"] = {"side_force_N": round(fy, 2), "weathercock_Nm": round(n_yaw, 2), "rudder_trim_deg": round(float(np.degrees(delta_rad)), 2), "rudder_power_W": round(drag_rud * u_boat, 2)}
        prof["wind_cases"] = wind_cases
        out[label] = prof
    return out


def evaluate_rear_deck_stiffener() -> dict:
    L_open, L_sub, b_eff, P_load = 0.80, 0.50, 0.30, 800.0
    E_weave, E_ud, t_top, t_bot, arch_factor = 68e9, 135e9, 0.00018, 0.00012, 2.6
    d3 = 0.003 + 0.5 * (t_top + t_bot)
    ei_3mm = arch_factor * E_weave * b_eff * (t_top * t_bot / (t_top + t_bot)) * (d3**2)
    k_crown, k_arch_94 = 18500.0, 2.20e6
    k_bare_80 = (110.0 * ei_3mm / (L_open**3)) + k_crown
    k_beam_only = (110.0 * ei_3mm / (L_sub**3)) + 0.35 * k_crown
    k_deep_rib = (160.0 * (ei_3mm + E_ud * (0.025 * 0.00044) * (0.0255**2)) / (L_open**3)) + k_crown
    ei_cruc_rib = ei_3mm + E_ud * (0.022 * 0.00044) * (0.0165**2)
    defl_sub = P_load * (L_sub**3) / (155.0 * ei_cruc_rib) + (0.42 * P_load) / k_arch_94
    return {
        "load_N": P_load,
        "open_span_m": L_open,
        "bare_3mm_no_day_bh": {"added_mass_g": 0, "deflection_mm": round(float(P_load / k_bare_80 * 1000.0), 2)},
        "transverse_arch_beam_20x20mm_only": {"added_mass_g": 26, "deflection_mm": round(float(P_load / k_beam_only * 1000.0), 2)},
        "deep_center_ud_rib_18x24mm_only": {"added_mass_g": 48, "deflection_mm": round(float(P_load / k_deep_rib * 1000.0), 2)},
        "cruciform_ud_rib_plus_arch_beam": {"added_mass_g": 64, "deflection_mm": round(float(defl_sub * 1000.0), 2)},
    }


def _find_row(rows: list[dict], spd: float, hdg: str, sea_prefix: str) -> dict:
    for r in rows:
        if abs(r["speed_kn"] - spd) < 0.05 and r["heading"] == hdg and r["sea"].startswith(sea_prefix):
            return r
    raise KeyError((spd, hdg, sea_prefix))


def render_plate6(seak: dict, wind: dict, stiff: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8), dpi=190, facecolor="white")
    ax1, ax2, ax3, ax4 = axes.ravel()
    cases = [
        ("Epic 18X\n(225 lb)", seak["epic18x_225"]["rows"], "#5c6773"),
        ("10.9 v0.2.2\n(225 lb matched)", seak["k10_9_225"]["rows"], "#1f6fa8"),
        ("10.9 Ultralight\n(213 lb carbon)", seak["k10_9_213_ultralight"]["rows"], "#0b8a6a"),
        ("Epic 18X\n(265 lb Baja)", seak["epic18x_265_baja"]["rows"], "#8d99ae"),
        ("10.9 v0.2.2\n(265 lb Baja)", seak["k10_9_265_baja"]["rows"], "#2b4c7e"),
    ]
    x_idx, w = np.arange(len(cases)), 0.36
    b1 = ax1.bar(x_idx - w / 2, [_find_row(r, 4.5, "head", "Bay chop")["bow_dips_per_min"] for _, r, _ in cases], width=w, color=[c for _, _, c in cases], alpha=0.88, label="Bay chop (Hs 0.35m, Tp 2.5s)")
    b2 = ax1.bar(x_idx + w / 2, [_find_row(r, 4.5, "head", "Steep whitecaps")["bow_dips_per_min"] for _, r, _ in cases], width=w, color=[c for _, _, c in cases], alpha=0.48, hatch="//", edgecolor="#222", label="Steep whitecaps (Hs 0.65m, Tp 3.8s)")
    for bars in (b1, b2):
        ax1.bar_label(bars, fmt="%.1f", fontsize=7.8, padding=2)
    ax1.set_xticks(x_idx, [lbl for lbl, _, _ in cases], fontsize=8.2)
    ax1.set_ylabel("Bow deck green-water dips / min (at 3% L)", fontsize=8.8)
    ax1.set_title("1. Head-Sea Bow Dryness at 4.5 kn (Lower = Drier Foredeck)", fontsize=10.2, fontweight="bold")
    ax1.legend(fontsize=8.0, loc="upper left"); ax1.grid(axis="y", alpha=0.25)

    hdgs = ["head", "bow_quarter", "beam", "stern_quarter", "following"]
    hx = np.arange(len(hdgs))
    for lbl, key, col, mk in [
        ("Epic 18X (225 lb)", "epic18x_225", "#5c6773", "o--"),
        ("10.9 v0.2.2 (225 lb)", "k10_9_225", "#1f6fa8", "s-"),
        ("10.9 Ultralight (213 lb)", "k10_9_213_ultralight", "#0b8a6a", "^-"),
        ("Epic 18X (265 lb Baja)", "epic18x_265_baja", "#8d99ae", "d:"),
        ("10.9 v0.2.2 (265 lb Baja)", "k10_9_265_baja", "#2b4c7e", "P-"),
    ]:
        ax2.plot(hx, [_find_row(seak[key]["rows"], 4.5, h, "Steep whitecaps")["seat_acc_rms_ms2"] for h in hdgs], mk, color=col, lw=2.0, ms=5.8, label=lbl)
    ax2.set_xticks(hx, ["Head\n180°", "Bow Qtr\n135°", "Beam\n90°", "Stern Qtr\n45°", "Following\n0°"], fontsize=8.5)
    ax2.set_ylabel("Seat vertical acceleration RMS (m/s²)", fontsize=8.8)
    ax2.set_title("2. Seat Ride Slamming Across Headings (4.5 kn, Steep Whitecaps)", fontsize=10.2, fontweight="bold")
    ax2.legend(fontsize=7.8, loc="upper right"); ax2.grid(True, alpha=0.25)

    w_names = list(wind.keys())
    wx = np.arange(len(w_names))
    bw1 = ax3.bar(wx - 0.18, [wind[k]["wind_cases"]["15kn_beam"]["rudder_trim_deg"] for k in w_names], width=0.34, color="#2a6f9e", label="15 kn beam wind")
    bw2 = ax3.bar(wx + 0.18, [wind[k]["wind_cases"]["22kn_beam"]["rudder_trim_deg"] for k in w_names], width=0.34, color="#d97724", label="22 kn strong beam wind")
    ax3.bar_label(bw1, fmt="%.2f°", fontsize=8.3, padding=2); ax3.bar_label(bw2, fmt="%.2f°", fontsize=8.3, padding=2)
    ax3.axhline(12.5, color="#b83227", ls=":", lw=1.2, label="NACA-0012 stall (~12.5°)")
    ax3.set_xticks(wx, ["Epic 18X", "10.9 Pre-Finish\n(18.5 cm rear slab)", "10.9 v0.2.2\n(14.8 cm rear deck)"], fontsize=8.5)
    ax3.set_ylabel("Required rudder trim angle (deg)", fontsize=8.8); ax3.set_ylim(0, 14.5)
    ax3.set_title("3. Weathercocking Rudder Angle at 5.0 kn (Low Rear Deck Payoff)", fontsize=10.2, fontweight="bold")
    ax3.legend(fontsize=8.0, loc="upper right"); ax3.grid(axis="y", alpha=0.25)

    s_keys = ["bare_3mm_no_day_bh", "transverse_arch_beam_20x20mm_only", "deep_center_ud_rib_18x24mm_only", "cruciform_ud_rib_plus_arch_beam"]
    bs = ax4.bar(np.arange(4), [stiff[k]["deflection_mm"] for k in s_keys], width=0.52, color=["#c0564b", "#7b8a97", "#2a6f9e", "#0b8a6a"])
    ax4.bar_label(bs, fmt="%.2f mm", fontsize=8.5, padding=5)
    ax4.axhline(2.5, color="#0b8a6a", ls="--", lw=1.1, label="Drum-tight comfort target (<= 2.5 mm)")
    ax4.set_xticks(np.arange(4), ["Bare 3 mm\nNo Day BH (+0 g)", "Transverse Arch\nBeam Only (+26 g)", "Deep 18×24 mm\nUD Rib Only (+48 g)", "Cruciform (+) Rib\n+ Arch Beam (+64 g)"], fontsize=8.0)
    ax4.set_ylabel("Peak deck deflection under 180 lbf re-entry (mm)", fontsize=8.8)
    ax4.set_title("4. Single-BH Open Rear Deck (80 cm Span): Cruciform (+) Grid", fontsize=10.0, fontweight="bold")
    ax4.legend(fontsize=8.0); ax4.grid(axis="y", alpha=0.25)

    fig.suptitle("PLATE 6 — Capytaine 3.0 3D Linear Seakeeping, Windage Balance & Rear-Deck Stiffener Check (Calculated, Unvalidated)", fontsize=11.6, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=190)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seak-out", type=Path, default=REPO_ROOT / "data" / "capytaine-seakeeping.json")
    ap.add_argument("--wind-out", type=Path, default=REPO_ROOT / "data" / "windage-and-stiffener.json")
    ap.add_argument("--plate-out", type=Path, default=REPO_ROOT / "build" / "plates" / "plate6_seakeeping_and_windage.png")
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()

    Ve, Fe = load_epic_18x_arrays(bow_at_pos_x=True)
    V10, F10, sha10 = _load_kayak_10_9_pos_bow()
    configs = [
        ("epic18x_225", Ve, Fe, 225.0 * LB, 0.250, (0.0, 4.5, 5.7)),
        ("k10_9_225", V10, F10, 225.0 * LB, 0.250, (0.0, 4.5, 5.7)),
        ("k10_9_213_ultralight", V10, F10, 213.0 * LB, 0.242, (0.0, 4.5, 5.7)),
        ("epic18x_265_baja", Ve, Fe, 265.0 * LB, 0.250, (0.0, 4.5)),
        ("k10_9_265_baja", V10, F10, 265.0 * LB, 0.250, (0.0, 4.5)),
    ]
    summary: dict = {"metadata": {"capytaine_version": cpt.__version__, "claim_state": "raw_unvalidated", "kayak_10_9_outer_stl_sha256": sha10}}
    for name, V, F, mass_kg, kg_m, spds in configs:
        print(f"=== Capytaine 3D BEM: {name} ({mass_kg / LB:.0f} lb) ===", flush=True)
        body, I_mat, info = build_bem_body(name, V, F, mass_kg, kg_m)
        bem_out = solve_bem_case(body, I_mat, info, spds, n_jobs=args.n_jobs)
        summary[name] = evaluate_spectra(bem_out)

    args.seak_out.parent.mkdir(parents=True, exist_ok=True)
    args.seak_out.write_text(json.dumps(summary, indent=2) + "\n")

    wind = evaluate_windage_and_rudder(summary, Ve, Fe, V10, F10)
    stiff = evaluate_rear_deck_stiffener()
    wind_full = {"metadata": {"claim_state": "raw_unvalidated", "kayak_10_9_outer_stl_sha256": sha10}, "windage_and_rudder": wind, "rear_deck_stiffener": stiff}
    args.wind_out.parent.mkdir(parents=True, exist_ok=True)
    args.wind_out.write_text(json.dumps(wind_full, indent=2) + "\n")
    render_plate6(summary, wind, stiff, args.plate_out)
    print("WROTE", args.seak_out, args.wind_out, "AND", args.plate_out, flush=True)


if __name__ == "__main__":
    main()
