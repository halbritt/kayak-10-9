"""Mesh-based hydrostatics, stability, and ITTC-57 + Michell thin-ship resistance.

Body frame: x along length, y transverse (symmetric about y=0), z up. Units: m, kg, N.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import trimesh

G = 9.80665
KN = 0.514444
LB = 0.45359237
RHO_SEAWATER = 1025.0
NU_SEAWATER = 1.19e-6

REPO_ROOT = Path(__file__).resolve().parents[1]
EPIC_18X_NPZ = REPO_ROOT / "data" / "benchmarks" / "epic-18x-reference.npz"
STAGE9_PRE_FINISH_JSON = REPO_ROOT / "data" / "benchmarks" / "stage9-pre-finish.hull.json"


def load_epic_18x_arrays(*, bow_at_pos_x: bool) -> tuple[np.ndarray, np.ndarray]:
    """Load Epic 18X benchmark mesh with keel min at z=0 and centered y."""
    d = np.load(EPIC_18X_NPZ)
    V = np.asarray(d["vertices"], dtype=float).copy()
    F = np.asarray(d["faces"], dtype=int).copy()
    if bow_at_pos_x:
        V[:, 0] -= V[:, 0].min()
    else:
        V[:, 0] = -V[:, 0]
        F = F[:, ::-1]
        if trimesh.Trimesh(V, F, process=False).volume < 0:
            F = F[:, ::-1]
        V[:, 0] -= 0.5 * (V[:, 0].min() + V[:, 0].max())
    V[:, 1] -= np.median(V[:, 1])
    V[:, 2] -= V[:, 2].min()
    return V, F


def rot(heel: float, trim: float) -> np.ndarray:
    ch, sh, ct, st = np.cos(heel), np.sin(heel), np.cos(trim), np.sin(trim)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ch, -sh], [0.0, sh, ch]])
    ry = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return rx @ ry


def clip_below(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Return triangles (n, 3, 3) of the surface below z <= 0, orientation preserved."""
    T = V[F]
    z = T[:, :, 2]
    below = z < 0
    nb = below.sum(1)
    out = [T[nb == 3]]
    for k in (1, 2):
        Tk, bk = T[nb == k], below[nb == k]
        if not len(Tk):
            continue
        odd = (bk if k == 1 else ~bk).argmax(1)
        idx = (odd[:, None] + np.arange(3)[None]) % 3
        Tk = np.take_along_axis(Tk, idx[:, :, None], 1)
        a, b, c = Tk[:, 0], Tk[:, 1], Tk[:, 2]
        tb = (a[:, 2] / (a[:, 2] - b[:, 2]))[:, None]
        tc = (a[:, 2] / (a[:, 2] - c[:, 2]))[:, None]
        ab = a + tb * (b - a)
        ac = a + tc * (c - a)
        ab[:, 2] = 0.0
        ac[:, 2] = 0.0
        if k == 1:
            out.append(np.stack([a, ab, ac], 1))
        else:
            out.append(np.stack([ab, b, c], 1))
            out.append(np.stack([ab, c, ac], 1))
    return np.concatenate(out)


def immersed(V: np.ndarray, F: np.ndarray, heel: float = 0.0, trim: float = 0.0, h: float = 0.0) -> dict:
    """Properties of the immersed volume in the water frame (origin height h above body origin)."""
    W = V @ rot(heel, trim).T
    W[:, 2] -= h
    T = clip_below(W, F)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    n = np.cross(b - a, c - a) / 2.0
    az = n[:, 2]
    x, y, z = T[:, :, 0], T[:, :, 1], T[:, :, 2]

    def m2(p: np.ndarray, q: np.ndarray) -> float:
        return float((az / 12.0 * (p.sum(1) * q.sum(1) + (p * q).sum(1))).sum())

    vol = float((az * z.mean(1)).sum())
    r: dict = dict(vol=vol, wet=float(np.linalg.norm(n, axis=1).sum()), ntri=len(T))
    if vol <= 0.0:
        return r
    r["B"] = np.array([m2(x, z), m2(y, z), m2(z, z) / 2.0]) / vol
    awp = float(-az.sum())
    xf = float(-(az * x.mean(1)).sum() / awp)
    yf = float(-(az * y.mean(1)).sum() / awp)
    r.update(awp=awp, xf=xf, yf=yf, IT=-m2(y, y) - awp * yf**2, IL=-m2(x, x) - awp * xf**2)
    on = np.abs(T[:, :, 2]) < 1e-12
    P = T[on]
    r.update(wl_x=(float(P[:, 0].min()), float(P[:, 0].max())), wl_y=(float(P[:, 1].min()), float(P[:, 1].max())), zmin=float(T[:, :, 2].min()), tris=T)
    return r


def float_at(V: np.ndarray, F: np.ndarray, mass: float, rho: float, cg: np.ndarray, heel: float = 0.0, free_trim: bool = True, trim: float = 0.0) -> dict:
    """Solve sinkage (and optional free trim) so displaced mass = mass and LCB = LCG."""
    vol0 = mass / rho

    def h_for(tr: float) -> float:
        W = V @ rot(heel, tr).T
        return float(brentq(lambda h: immersed(V, F, heel, tr, h)["vol"] - vol0, float(W[:, 2].min()) + 1e-6, float(W[:, 2].max()) - 1e-6, xtol=1e-10))

    def resid(tr: float) -> float:
        h_val = h_for(tr)
        r_val = immersed(V, F, heel, tr, h_val)
        g_val = rot(heel, tr) @ cg
        return float(r_val["B"][0] - g_val[0])

    if free_trim:
        trim = float(brentq(resid, np.radians(-6.0), np.radians(6.0), xtol=1e-9))
    h = h_for(trim)
    r = immersed(V, F, heel, trim, h)
    g = rot(heel, trim) @ cg
    g[2] -= h
    r.update(heel=heel, trim=trim, h=h, G=g, GZ=float(r["B"][1] - g[1]), dx=float(r["B"][0] - g[0]))
    return r


def half_breadth_grid(tris: np.ndarray, nx: int = 401, nz: int = 81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half-breadth y(x, z) of the immersed body (water frame) plus the coordinate grids."""
    P = tris.reshape(-1, 3)
    xs = np.linspace(float(P[:, 0].min()), float(P[:, 0].max()), nx)
    zs = np.linspace(float(P[:, 2].min()), 0.0, nz)
    Y = np.zeros((nx, nz))
    for i, xc in enumerate(xs[1:-1], 1):
        pos = tris[:, :, 0] >= xc
        npos = pos.sum(1)
        sel = (npos == 1) | (npos == 2)
        T, p_ = tris[sel], pos[sel]
        if not len(T):
            continue
        odd = np.where(p_.sum(1) == 1, p_.argmax(1), (~p_).argmax(1))
        idx = (odd[:, None] + np.arange(3)[None]) % 3
        T = np.take_along_axis(T, idx[:, :, None], 1)
        a_, b_, c_ = T[:, 0], T[:, 1], T[:, 2]
        tb = ((xc - a_[:, 0]) / (b_[:, 0] - a_[:, 0]))[:, None]
        tc = ((xc - a_[:, 0]) / (c_[:, 0] - a_[:, 0]))[:, None]
        S = np.stack([a_[:, 1:] + tb * (b_[:, 1:] - a_[:, 1:]), a_[:, 1:] + tc * (c_[:, 1:] - a_[:, 1:])], 1)
        ya, za, yb, zb = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]
        lo, hi = np.minimum(za, zb), np.maximum(za, zb)
        cross = (lo[None] <= zs[:, None] + 1e-12) & (hi[None] >= zs[:, None] - 1e-12)
        dz = zb - za
        t = np.clip((zs[:, None] - za[None]) / np.where(dz == 0, 1.0, dz)[None], 0.0, 1.0)
        yy = np.abs(ya[None] + t * (yb - ya)[None])
        yy = np.where(dz[None] == 0, np.maximum(np.abs(ya), np.abs(yb))[None], yy)
        Y[i] = np.where(cross, yy, 0.0).max(1)
    return xs, zs, Y


def michell(xs: np.ndarray, zs: np.ndarray, Y: np.ndarray, U: float, rho: float, ntheta: int = 4000) -> float:
    """Michell thin-ship wave resistance; Y piecewise linear in x and z, integrated analytically."""
    k0 = G / U**2
    th = (np.arange(ntheta) + 0.5) * (np.pi / 2.0) / ntheta
    sec = 1.0 / np.cos(th)
    kap = k0 * sec
    a = k0 * sec**2
    S = np.diff(Y, axis=0) / np.diff(xs)[:, None]
    E = np.exp(1j * kap[:, None] * xs[None])
    Ex = (E[:, 1:] - E[:, :-1]) / (1j * kap[:, None])
    Q = Ex @ S
    hz = np.diff(zs)
    t = a[:, None] * hz[None]
    e1 = np.exp(a[:, None] * zs[None, 1:])
    w_hi = e1 * hz * (t - 1.0 + np.exp(-t)) / t**2
    w_lo = e1 * hz * (1.0 - np.exp(-t) * (1.0 + t)) / t**2
    A = (Q[:, 1:] * w_hi + Q[:, :-1] * w_lo).sum(1)
    integrand = np.abs(A) ** 2 * sec**3
    return float(4.0 * rho * G**2 / (np.pi * U**2) * integrand.sum() * (np.pi / 2.0) / ntheta)


def ittc57(U: float, L: float, S: float, rho: float, nu: float, k: float = 0.0) -> tuple[float, float, float]:
    re = U * L / nu
    cf = 0.075 / (np.log10(re) - 2.0) ** 2
    return float(0.5 * rho * S * U**2 * cf * (1.0 + k)), float(cf), float(re)
