"""Run OpenFOAM-v2606 snappyHexMesh + interFoam (kOmegaSST) on Kayak 10.9 vs. Epic 18X (5.5 & 6.2 kn)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stl import mesh as stl_mesh
import trimesh

from deck_geometry import SculptedKayakSurface
from hydro import G, KN, LB, RHO_SEAWATER, float_at, immersed, load_epic_18x_arrays
from openfoam_dicts import write_interfoam_case_files, write_mesh_dicts

MASS_225 = 225.0 * LB
KG_225 = 0.25
DEFAULT_BASHRC = Path("/usr/lib/openfoam/openfoam2606/etc/bashrc")


def _run_foam(bashrc: Path, cwd: Path, shell_cmd: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as out:
        res = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", 'setup=$1; cmd=$2; source "$setup" "" >/dev/null 2>&1 && eval "$cmd"', "foam", str(bashrc), shell_cmd],
            cwd=cwd,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return res.returncode


def _write_stl(V: np.ndarray, F: np.ndarray, path: Path) -> str:
    m = stl_mesh.Mesh(np.zeros(F.shape[0], dtype=stl_mesh.Mesh.dtype))
    m.vectors[:] = V[F]
    m.save(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_for_cfd(name: str, V_in: np.ndarray, F_in: np.ndarray, out_dir: Path) -> dict:
    """Float-pose watertight closed shell at 225 lb in 1025 kg/m³ seawater (bow at -x, calm WL at z=0)."""
    tm0 = trimesh.Trimesh(V_in.copy(), F_in.copy(), process=True)
    assert tm0.is_watertight, f"{name} closed shell must be watertight"
    V, F = np.asarray(tm0.vertices, float).copy(), np.asarray(tm0.faces, int).copy()
    if tm0.volume < 0:
        F = F[:, ::-1]
    V[:, 2] -= V[:, 2].min()
    V[:, 1] -= 0.5 * (V[:, 1].min() + V[:, 1].max())
    V[:, 0] -= 0.5 * (V[:, 0].min() + V[:, 0].max())

    r = float_at(V, F, MASS_225, RHO_SEAWATER, np.array([0.0, 0.0, KG_225]), free_trim=False)
    T = float(-r["zmin"])
    lcg = float(r["B"][0])
    gm_l = float(r["B"][2] + T + r["IL"] / r["vol"] - KG_225)
    V[:, 2] -= T
    chk = immersed(V, F, heel=0.0, trim=0.0, h=0.0)
    stl_path = out_dir / f"{name}.stl"
    sha = _write_stl(V, F, stl_path)
    return {
        "name": name,
        "stl_sha256": sha,
        "n_faces": int(len(F)),
        "x_range_m": [round(float(V[:, 0].min()), 4), round(float(V[:, 0].max()), 4)],
        "y_range_m": [round(float(V[:, 1].min()), 4), round(float(V[:, 1].max()), 4)],
        "z_range_m": [round(float(V[:, 2].min()), 4), round(float(V[:, 2].max()), 4)],
        "draft_225_m": round(T, 5),
        "lcg_x_m": round(lcg, 5),
        "gm_l_m": round(gm_l, 3),
        "cg_cfd_m": [round(lcg, 5), 0.0, round(KG_225 - T, 5)],
        "wetted_area_m2": round(float(chk["wet"]), 4),
        "displaced_mass_kg": round(float(chk["vol"] * RHO_SEAWATER), 4),
    }


def build_mesh(hull_name: str, out_dir: Path, bashrc: Path, np_ranks: int = 6) -> dict:
    t0 = time.monotonic()
    cdir = out_dir / f"mesh_{hull_name}"
    if cdir.exists():
        shutil.rmtree(cdir)
    for sub in ("system", "constant/triSurface", "logs"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / f"{hull_name}.stl", cdir / "constant/triSurface/hull.stl")
    write_mesh_dicts(cdir, np_ranks=np_ranks)

    cmd = (
        "blockMesh > logs/blockMesh.log 2>&1 && "
        "surfaceFeatureExtract > logs/sfe.log 2>&1 && "
        "cp constant/extendedFeatureEdgeMesh/hull.eMesh constant/triSurface/hull.eMesh 2>/dev/null || true && "
        "decomposePar > logs/decomp.log 2>&1 && "
        f"mpirun -np {np_ranks} snappyHexMesh -parallel -overwrite > logs/snappy.log 2>&1 && "
        "reconstructParMesh -constant > logs/reconMesh.log 2>&1 && "
        "rm -rf processor* && "
        "checkMesh > logs/checkMesh.log 2>&1"
    )
    rc = _run_foam(bashrc, cdir, cmd, cdir / "logs/pipeline.log")
    chk_txt = (cdir / "logs/checkMesh.log").read_text() if (cdir / "logs/checkMesh.log").is_file() else ""
    ok = ("Mesh OK." in chk_txt) and (rc == 0)
    cells = next((ln.strip() for ln in chk_txt.splitlines() if "cells:" in ln), "?")
    nonortho = next((ln.strip() for ln in chk_txt.splitlines() if "Mesh non-orthogonality" in ln), "?")
    sec = round(time.monotonic() - t0, 1)
    print(f"[mesh] {hull_name}: ok={ok} {cells} | {nonortho} ({sec}s)", flush=True)
    if not ok:
        raise RuntimeError(f"Mesh check failed for {hull_name}:\n{chk_txt[-1200:]}")
    return {"hull": hull_name, "ok": ok, "cells_line": cells, "nonortho": nonortho, "seconds": sec}


def run_interfoam_case(hull_name: str, speed_kn: float, posed: dict, out_dir: Path, bashrc: Path, end_steps: int = 500, np_ranks: int = 6) -> tuple[Path, dict]:
    t0 = time.monotonic()
    u_mps = speed_kn * KN
    tag = f"run_{hull_name}_{speed_kn:.1f}kn".replace(".", "p")
    cdir = out_dir / tag
    if cdir.exists():
        shutil.rmtree(cdir)
    for sub in ("system", "constant", "logs"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copytree(out_dir / f"mesh_{hull_name}/constant/polyMesh", cdir / "constant/polyMesh")
    write_interfoam_case_files(cdir, u_mps, posed["cg_cfd_m"], end_steps=end_steps, np_ranks=np_ranks)
    shutil.copytree(cdir / "0.orig", cdir / "0")

    step_str = str(end_steps)
    cmd = (
        "setFields > logs/setFields.log 2>&1 && "
        "decomposePar > logs/decomposePar.log 2>&1 && "
        f"mpirun -np {np_ranks} renumberMesh -parallel -overwrite > logs/renumber.log 2>&1 && "
        f"mpirun -np {np_ranks} interFoam -parallel > logs/interFoam.log 2>&1 && "
        "reconstructPar -latestTime > logs/reconstructPar.log 2>&1 && "
        f"postProcess -func writeCellCentres -time {step_str} > logs/cellC.log 2>&1 && "
        f"postProcess -func writeCellVolumes -time {step_str} >> logs/cellC.log 2>&1"
    )
    rc = _run_foam(bashrc, cdir, cmd, cdir / "logs/pipeline.log")
    log_txt = (cdir / "logs/interFoam.log").read_text() if (cdir / "logs/interFoam.log").is_file() else ""
    if rc != 0 or not re.search(rf"^Time = {end_steps}\s*$", log_txt, re.M) or not re.search(r"^End\s*$", log_txt, re.M):
        raise RuntimeError(f"interFoam failed for {tag} (rc={rc}):\n{log_txt[-1500:]}")
    using_line = next((ln.strip() for ln in log_txt.splitlines() if ln.startswith("Using:")), "")
    build_line = next((ln.strip() for ln in log_txt.splitlines() if ln.startswith("Build")), "")
    sec = round(time.monotonic() - t0, 1)
    print(f"[solve] {tag}: ok ({sec}s) | {using_line} | {build_line}", flush=True)
    return cdir, {"tag": tag, "hull": hull_name, "speed_kn": speed_kn, "seconds": sec, "using": using_line, "build": build_line}


def _parse_forces(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
    steps, fp, fv = [], [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = [float(x) for x in num_re.findall(s)]
        if len(vals) >= 10:
            steps.append(vals[0])
            fp.append(vals[4:7])
            fv.append(vals[7:10])
    return np.array(steps), 2.0 * np.array(fp), 2.0 * np.array(fv)


def _read_internal_field(path: Path, is_vector: bool) -> np.ndarray:
    txt = path.read_text()
    m_uni = re.search(r"internalField\s+uniform\s+([^\n;]+);", txt)
    if m_uni:
        return np.array([float(v) for v in m_uni.group(1).strip(" ()").split()])
    sub = txt[txt.find("internalField") :]
    m_n = re.search(r"nonuniform\s+List<[^>]+>\s+(\d+)\s*\(", sub)
    assert m_n is not None, f"could not parse internalField in {path}"
    n = int(m_n.group(1))
    b0 = sub.find("(", m_n.end() - 1) + 1
    if not is_vector:
        return np.fromstring(sub[b0 : sub.find(")", b0)], sep=" ", count=n)
    vec_re = re.compile(r"\(\s*([^\)]+)\s*\)")
    out, pos = np.empty((n, 3), float), b0
    for i in range(n):
        mv = vec_re.search(sub, pos)
        assert mv is not None
        out[i] = [float(w) for w in mv.group(1).split()]
        pos = mv.end()
    return out


def _read_patch_vectors(path: Path, patch: str) -> np.ndarray:
    txt = path.read_text()
    sub = txt[txt.find(patch, txt.find("boundaryField")) :]
    m_n = re.search(r"nonuniform\s+List<[^>]+>\s+(\d+)\s*\(", sub)
    if m_n is None:
        return np.zeros((0, 3))
    n, pos = int(m_n.group(1)), sub.find("(", m_n.end() - 1) + 1
    vec_re = re.compile(r"\(\s*([^\)]+)\s*\)")
    out = np.empty((n, 3), float)
    for i in range(n):
        mv = vec_re.search(sub, pos)
        assert mv is not None
        out[i] = [float(w) for w in mv.group(1).split()]
        pos = mv.end()
    return out


def _free_surface_and_hull_wave(cdir: Path, step_dir: str = "500") -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    tdir = cdir / step_dir
    C = _read_internal_field(tdir / "C", is_vector=True)
    V = _read_internal_field(tdir / "V", is_vector=False)
    alpha = _read_internal_field(tdir / "alpha.water", is_vector=False)
    Ch = _read_patch_vectors(tdir / "C", "hull")

    z_lo, z_hi = -0.055, 0.055
    slab = (C[:, 2] > z_lo) & (C[:, 2] < z_hi) & (C[:, 0] >= -3.3) & (C[:, 0] <= 8.5) & (C[:, 1] <= 2.0)
    xb, yb = np.linspace(-3.3, 8.5, 119), np.linspace(0.0, 2.0, 41)
    vol_sum, _, _ = np.histogram2d(C[slab, 0], C[slab, 1], bins=[xb, yb], weights=V[slab])
    wat_sum, _, _ = np.histogram2d(C[slab, 0], C[slab, 1], bins=[xb, yb], weights=V[slab] * alpha[slab])
    eta = (z_lo + np.where(vol_sum > 1e-12, wat_sum / np.maximum(vol_sum, 1e-12), np.nan) * (z_hi - z_lo)).T
    gx, gy = 0.5 * (xb[:-1] + xb[1:]), 0.5 * (yb[:-1] + yb[1:])

    band = (C[:, 2] > -0.055) & (C[:, 2] < 0.055) & (C[:, 0] >= -2.8) & (C[:, 0] <= 2.8) & (C[:, 1] <= 0.35)
    Cb, ab = C[band], alpha[band]
    xs = np.linspace(-2.70, 2.70, 109)
    zw = np.full_like(xs, np.nan)
    for i, x0 in enumerate(xs):
        sh = (np.abs(Ch[:, 0] - x0) <= 0.035) & (np.abs(Ch[:, 2]) <= 0.03)
        if sh.sum() < 2:
            continue
        yh = float(Ch[sh, 1].max())
        sc = (np.abs(Cb[:, 0] - x0) <= 0.035) & (Cb[:, 1] >= yh - 0.01) & (Cb[:, 1] <= yh + 0.035) & (ab >= 0.10) & (ab <= 0.90)
        if sc.sum() >= 1:
            zw[i] = float(np.median(Cb[sc, 2] + (ab[sc] - 0.5) * 0.014))
    return (gx, gy, eta), (xs, zw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "build" / "openfoam")
    ap.add_argument("--openfoam-bashrc", type=Path, default=DEFAULT_BASHRC)
    ap.add_argument("--metrics-out", type=Path, default=REPO_ROOT / "data" / "openfoam-cfd-metrics.json")
    ap.add_argument("--plate-out", type=Path, default=REPO_ROOT / "build" / "plates" / "plate5_openfoam_cfd.png")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--np", type=int, default=6)
    args = ap.parse_args()

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    args.plate_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    surf = SculptedKayakSurface(REPO_ROOT / "hull" / "kayak-10-9.hull.json")
    closed = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
    closed.merge_vertices(merge_tex=True, merge_norm=True)
    outer_stl = work / "kayak-10-9-outer-shell.stl"
    closed.export(outer_stl)
    outer_sha = hashlib.sha256(outer_stl.read_bytes()).hexdigest()

    Ve, Fe = load_epic_18x_arrays(bow_at_pos_x=False)
    poses = {
        "kayak_10_9": pose_for_cfd("kayak_10_9", np.asarray(closed.vertices), np.asarray(closed.faces), work),
        "epic_18x": pose_for_cfd("epic_18x", Ve, Fe, work),
    }
    meshes = {h: build_mesh(h, work, args.openfoam_bashrc, np_ranks=args.np) for h in ("epic_18x", "kayak_10_9")}

    fig, axs = plt.subplots(3, 1, figsize=(14.5, 12.2), layout="constrained")
    styles = {
        ("epic_18x", 5.5): ("#4b5563", "--", "Epic 18X @ 5.5 kn"),
        ("kayak_10_9", 5.5): ("#0284c7", "-", "Kayak 10.9 (v0.2.2) @ 5.5 kn"),
        ("epic_18x", 6.2): ("#111111", "--", "Epic 18X @ 6.2 kn"),
        ("kayak_10_9", 6.2): ("#dc2626", "-", "Kayak 10.9 (v0.2.2) @ 6.2 kn"),
    }
    wave62: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary: dict = {"metadata": {"claim_state": "raw_unvalidated", "kayak_10_9_outer_stl_sha256": outer_sha, "poses": poses, "meshes": meshes}}
    ver_tag = "OpenFOAM"

    for sp in (5.5, 6.2):
        for h in ("epic_18x", "kayak_10_9"):
            cdir, run_info = run_interfoam_case(h, sp, poses[h], work, args.openfoam_bashrc, end_steps=args.steps, np_ranks=args.np)
            if run_info["using"] or run_info["build"]:
                hdr = f"{run_info['using']} {run_info['build']}".strip()
                m_v = re.search(r"(?:OpenFOAM-|OPENFOAM=)([0-9A-Za-z._-]+)", hdr)
                ver_tag = f"OpenFOAM-v{m_v.group(1)}" if m_v else hdr
                summary["metadata"]["openfoam_version"] = ver_tag
                summary["metadata"]["openfoam_using"] = run_info["using"] or ver_tag
                summary["metadata"]["openfoam_build"] = run_info["build"]
            f_path = sorted((cdir / "postProcessing/forces").glob("*/force.dat"))[-1]
            m_path = sorted((cdir / "postProcessing/forces").glob("*/moment.dat"))[-1]
            st, fp, fv = _parse_forces(f_path)
            _, mp, mv = _parse_forces(m_path)
            ftot, mtot = fp + fv, mp + mv
            tail = st >= max(st[-1] - 150, st[0])
            fx_mean, fx_std = float(ftot[tail, 0].mean()), float(ftot[tail, 0].std())
            fpx_mean, fvx_mean = float(fp[tail, 0].mean()), float(fv[tail, 0].mean())
            fz_mean, my_mean = float(ftot[tail, 2].mean()), float(mtot[tail, 1].mean())
            u_mps = sp * KN
            (gx, gy, eta), (xs_w, zw) = _free_surface_and_hull_wave(cdir, str(int(st[-1])))
            col, ls, lbl = styles[(h, sp)]
            axs[0].plot(st, ftot[:, 0], color=col, ls=ls, lw=1.6, label=f"{lbl}: {fx_mean:.2f} N (p={fpx_mean:.2f}, v={fvx_mean:.2f})")
            axs[1].plot(xs_w, zw * 1000.0, color=col, ls=ls, lw=1.7, label=lbl)
            if sp == 6.2:
                wave62[h] = (gx, gy, eta)
            summary[f"{h}_{sp}kn"] = {
                "hull": h,
                "speed_kn": sp,
                "u_mps": round(u_mps, 4),
                "drag_total_N": round(fx_mean, 3),
                "drag_std_N": round(fx_std, 3),
                "drag_pressure_N": round(fpx_mean, 3),
                "drag_viscous_N": round(fvx_mean, 3),
                "effective_power_W": round(fx_mean * u_mps, 2),
                "vertical_force_N": round(fz_mean, 2),
                "pitch_moment_My_Nm": round(my_mean, 3),
                "bow_up_running_trim_deg": round(float(np.degrees(my_mean / (MASS_225 * G * poses[h]["gm_l_m"]))), 3),
                "bow_wave_crest_mm": round(float(np.nanmax(zw[xs_w < -1.5]) * 1000.0), 1),
                "mid_wave_trough_mm": round(float(np.nanmin(zw) * 1000.0), 1),
            }

    axs[0].set_ylim(0, 35); axs[0].grid(alpha=0.25); axs[0].legend(loc="upper right", fontsize=8.5)
    axs[0].set_ylabel("Full-Hull Drag Force (N)")
    axs[0].set_title(f"1. {ver_tag} interFoam (LTS + k-omega SST) Drag Convergence (raw_unvalidated)")
    axs[1].axhline(0.0, color="royalblue", ls=":", lw=0.8); axs[1].grid(alpha=0.25); axs[1].legend(loc="upper right", fontsize=8.5)
    axs[1].set_ylabel("Wave Height (mm)"); axs[1].set_xlabel("x (m; bow at -2.74 m, stern at +2.74 m)")
    axs[1].set_title("2. Wetted Waterline Wave Profile Along Hull Skin (mm above calm 225 lb WL)")
    gx, gy, z_ep = wave62["epic_18x"]
    _, _, z_k10 = wave62["kayak_10_9"]
    levels = np.linspace(-24, 24, 25)
    cf = axs[2].contourf(gx, gy, z_ep * 1000.0, levels=levels, cmap="RdBu_r", extend="both")
    axs[2].contourf(gx, -gy, z_k10 * 1000.0, levels=levels, cmap="RdBu_r", extend="both")
    axs[2].axhline(0.0, color="black", lw=1.2)
    axs[2].set_aspect("equal"); axs[2].set_xlim(-3.2, 8.2); axs[2].set_ylim(-1.9, 1.9)
    axs[2].set_xlabel("x (m; flow left-to-right)"); axs[2].set_ylabel("y (m)")
    axs[2].set_title("3. 6.2 kn Kelvin Free-Surface Wave Elevation (mm): TOP (y>0) = Epic 18X | BOTTOM (y<0) = Kayak 10.9 (v0.2.2)")
    fig.colorbar(cf, ax=axs[2], label="Wave elevation (mm)", shrink=0.85)
    fig.savefig(args.plate_out, dpi=145)
    plt.close(fig)
    args.metrics_out.write_text(json.dumps(summary, indent=2) + "\n")
    print("WROTE", args.metrics_out, "AND", args.plate_out, flush=True)


if __name__ == "__main__":
    main()
