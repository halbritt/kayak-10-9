"""Steady free-surface potential flow about a hull: a Rankine-source panel solver.

What it computes
----------------
Wave resistance, dynamic sinkage and trim of a hull moving at constant speed in calm, deep water,
from a triangle mesh of the hull. It is independent of Michell's thin-ship assumption
(``rsi.evaluator._michell_fast_curve``): the hull boundary condition is applied on the real hull
surface, and the hull changes the flow the waves ride on. It is deterministic: the same mesh and
settings give the same number, so the difference between two near-identical hulls carries no
run-to-run noise. Its discretisation error is studied in ``docs/panel-solver/README.md``.

Method
------
* Constant-strength Rankine source panels (flat quadrilaterals, Hess and Smith) on the wetted hull
  below the still waterplane and above a patch of that waterplane around the hull. Half model
  with an image in the centreplane y = 0.
* The hull is re-panelled from the triangle mesh: transverse sections are cut at stations between
  the waterline ends and divided into panels of equal girth. The stations are gathered towards
  the two ends (``end_cluster``), where a hull with a steep stem changes fastest. The
  free-surface grid uses the same stations, follows the waterline and continues ahead and astern.
* Linearisation (``linearisation="dawson"``, the default): Dawson's double-body linearisation.
  The flow with a rigid lid on z = 0 (double-body flow, potential Phi) is solved first. The
  wave part phi' is then solved with the free-surface condition linearised about that flow and
  applied on z = 0::

      D(D phi') + [D(q^2) / (2 q^2)] D phi' + g dphi'/dz = -1/2 D(q^2),
      D = grad(Phi) . grad (in the plane z = 0),  q^2 = |grad Phi|^2,

  which is Dawson's condition ``(Phi_l^2 phi_l)_l + g phi_z = 2 Phi_l^2 Phi_ll`` written for the
  perturbation in Cartesian form. ``linearisation="kelvin"`` replaces the double-body flow by the
  uniform stream in the condition (Neumann-Kelvin: ``U^2 phi_xx + g phi_z = 0``); it is the
  panel-method counterpart of Michell's theory with the exact hull condition.
* Radiation condition: the outer derivative ``D`` uses Dawson's four-point upstream difference
  along the grid lines, and its value is zero at the upstream edge. No waves travel upstream.
* Free-surface panels are raised above z = 0 by ``fs_raise`` panel lengths and the collocation
  points, on z = 0, lie ``fs_shift`` panel lengths upstream of the panel centres (Raven's and
  Jensen's arrangement). With panels in the plane and collocation under their centres a
  point-to-point oscillation of the source density is almost free: it reached the upstream edge
  and changed the resistance by up to a tenth with the parity of the column count.
* Wave elevation: ``eta = (U^2 - q^2 - 2 D phi') / (2 g)``.
* Wave resistance ``rw_N``: Bernoulli pressure (all quadratic terms kept) integrated over the
  hull below z = 0, less the same integral for the double-body flow (zero in exact arithmetic,
  so this removes a discretisation error). This is Dawson's practice. Reported beside it:
  - ``rw_waterline_N``, the term ``rho g sum(eta^2 dy)`` for the strip between z = 0 and the
    wave surface. Hull pressure plus this term is what the momentum balance of the linear
    problem gives; it is of higher order for a thin hull and is not included in ``rw_N``.
  - ``rw_farfield_N`` (``farfield=True``), the momentum flux through a transverse plane behind
    the hull, a plane ahead of it and a longitudinal plane beside it, from the velocities and
    wave elevation there. It needs a free surface reaching well behind the hull and loses the
    short diverging waves the grid cannot carry, so it is a lower bound.
* Sinkage and trim: the hull pose is iterated until weight, buoyancy and the dynamic vertical
  force balance, and their moments about the centre of gravity balance, at fixed displacement
  and longitudinal centre of gravity. The hull is re-panelled at each pose. The towing force is
  taken to act through the centre of gravity.

What is linearised or left out
------------------------------
* The free-surface condition is linear and is applied on the still waterplane z = 0, not on the
  wave surface. Hull panels stop at z = 0: the change of wetted surface under the wave profile
  enters only through the separately reported waterline term. No breaking, no spray.
* The pose is not linearised: sinkage and trim move the real hull, which is re-panelled.
* Potential flow: no friction, no boundary layer, no separation at the stern. Friction is taken
  from the existing ITTC-57 model (``rsi.evaluator.local_friction_ratio``) by the callers.
* Deep, unbounded water. Waterline ends must be pointed (no immersed transom) and the immersed
  hull must not reach beyond the waterline ends (no bulb). Each immersed half section must be one
  curve from the centreline to the waterline. A hull outside these limits is rejected.

Frames and signs
----------------
Body frame as in ``cad/hydro.py``: x along the hull with the bow at -x, y to starboard, z up.
The water frame has z = 0 on the still waterplane; the stream runs towards +x. ``h`` is the
height of the waterplane above the body origin and ``trim`` (radians) is positive bow up.
Sinkage is positive when the hull goes down.

Backends
--------
The dense assembly and solve are written against the array interface shared by numpy and CuPy
(``backend="numpy"`` or ``"cupy"``; CuPy is the optional ``gpu`` extra). Geometry and equilibrium
run on the host. The numpy assembly runs its chunks on ``KAYAK109_PANEL_THREADS`` threads
(default ``OMP_NUM_THREADS``, else 4). Memory is about 80 N^2 bytes for N panels.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import math
import os
import time
from typing import Any

import numpy as np
from scipy.optimize import brentq

G = 9.80665
KN = 0.514444
FOUR_PI = 4.0 * math.pi

__all__ = [
    "PanelConfig",
    "PanelResult",
    "PanelSet",
    "get_backend",
    "hydrostatics",
    "make_panels",
    "solve_fixed",
    "solve_free",
    "static_pose",
    "velocity_influence",
    "wigley_mesh",
]


# --------------------------------------------------------------------------------------------
# configuration and results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelConfig:
    """Discretisation of one solve. Lengths are in waterline lengths unless stated."""

    nx_hull: int = 50  # panels along the waterline; the free surface uses the same stations
    end_cluster: float = 0.8  # 0: equal stations; towards 1: stations gathered at the two ends
    end_growth: float = 1.15  # growth of the column length ahead of and behind a clustered hull
    nz_hull: int = 8  # panels along the girth of each half section
    ny_fs: int = 30  # free-surface panel rows on one side
    upstream: float = 0.5  # free surface ahead of the bow
    downstream: float = 0.5  # free surface behind the stern (1.0 or more with farfield)
    half_width: float = 0.6  # free surface from the centreplane outwards
    first_row: float = 0.75  # width of the row next to the hull, in station spacings
    fs_raise: float = 0.8  # height of the free-surface panels above z = 0, in station spacings
    fs_shift: float = 0.5  # collocation points upstream of the panel centres, in station spacings
    linearisation: str = "dawson"  # or "kelvin"
    waterline_eta: str = "row0"  # elevation used in the waterline term: "row0" or "extrapolated"
    farfield: bool = False  # also compute the control-surface estimate
    ff_behind_stern: float = 0.5  # transverse control plane, behind the stern
    ff_ahead_bow: float = 0.3  # transverse control plane, ahead of the bow
    ff_side: float = 0.75  # longitudinal control plane, fraction of half_width
    ff_ny: int = 64
    ff_nz: int = 20
    ff_depth_wavelengths: float = 2.0
    pair_budget: int = 250_000  # point-panel pairs per assembly chunk (memory control)
    precision: str = "double"  # "single" halves the memory of the dense work; see the document

    def n_panels(self) -> tuple[int, int]:
        """(hull panels, free-surface panels) for this configuration."""
        frac = _station_fractions(self.nx_hull, self.end_cluster)
        n_up = len(_outer_columns(frac[1] - frac[0], 1.0 / self.nx_hull, self.upstream, self.end_growth))
        n_dn = len(_outer_columns(frac[-1] - frac[-2], 1.0 / self.nx_hull, self.downstream, self.end_growth))
        return self.nx_hull * self.nz_hull, (n_up + self.nx_hull + n_dn) * self.ny_fs

    @classmethod
    def for_froude(cls, fn: float, panels_per_wavelength: float = 30.0, **kw: Any) -> "PanelConfig":
        """Station spacing fine enough for ``panels_per_wavelength`` along the transverse wave."""
        lam = 2.0 * math.pi * fn * fn
        nx = max(40, int(math.ceil(panels_per_wavelength / lam / 4.0)) * 4)
        return cls(**{"nx_hull": nx, "ny_fs": round(0.6 * nx), **kw})


@dataclass
class PanelResult:
    """One speed at one pose. Forces in newtons for the whole hull (both sides)."""

    speed_mps: float
    froude: float
    lwl_m: float
    wetted_area_m2: float  # panelled wetted surface below z = 0, both sides
    rw_N: float  # wave resistance: hull pressure below z = 0, double-body zero removed
    rw_waterline_N: float  # waterline term, reported apart (see the module docstring)
    rw_doublebody_zero_N: float  # the discretisation error that was removed
    rw_pressure_linear_N: float  # as rw_N with the pressure linearised about the base flow
    rw_farfield_N: float | None  # control-surface estimate
    farfield_parts_N: dict[str, float] | None
    fz_dyn_N: float  # dynamic vertical force, positive up
    my_dyn_Nm: float  # dynamic moment about the water-frame origin, positive bow up
    cw: float  # rw_N / (0.5 rho U^2 wetted_area)
    h_m: float
    trim_rad: float
    sinkage_m: float | None = None
    trim_change_rad: float | None = None
    pose_iterations: int | None = None
    pose_converged: bool | None = None
    n_hull: int = 0
    n_fs: int = 0
    assembly_s: float = 0.0
    solve_s: float = 0.0
    farfield_s: float = 0.0
    backend: str = "numpy"
    wave_profile: dict[str, list[float]] | None = field(default=None, repr=False)

    def to_dict(self, *, profile: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not profile:
            d.pop("wave_profile")
        return d


# --------------------------------------------------------------------------------------------
# backend
# --------------------------------------------------------------------------------------------


def get_backend(name: str = "numpy") -> Any:
    """Array module for the dense work: numpy, or CuPy when it is installed."""
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("backend 'cupy' requested but CuPy is not installed") from exc
        return cupy
    raise ValueError(f"unknown backend {name!r}; use 'numpy' or 'cupy'")


def host_threads() -> int:
    """Threads for the numpy assembly: ``KAYAK109_PANEL_THREADS``, else ``OMP_NUM_THREADS``, else 4."""
    for name in ("KAYAK109_PANEL_THREADS", "OMP_NUM_THREADS"):
        val = os.environ.get(name, "")
        if val.isdigit() and int(val) > 0:
            return int(val)
    return 4


def _host(a: Any) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _sync(xp: Any) -> None:
    if xp is not np:  # pragma: no cover - GPU only
        xp.cuda.Stream.null.synchronize()


# --------------------------------------------------------------------------------------------
# panels and their influence
# --------------------------------------------------------------------------------------------


@dataclass
class PanelSet:
    """Flat quadrilateral source panels. Corners run anticlockwise seen from the normal's side."""

    corners: np.ndarray  # (n, 4, 3), projected on each panel's mean plane
    centre: np.ndarray  # (n, 3) area centroid: the collocation point
    normal: np.ndarray  # (n, 3) unit normal, into the fluid
    area: np.ndarray  # (n,)

    @property
    def n(self) -> int:
        return int(len(self.area))


def make_panels(corners: np.ndarray) -> PanelSet:
    """Hess-Smith flat panels from corner quadruples; a triangle is a quad with a repeated corner.

    The normal is the cross product of the diagonals, so the corner order fixes its side.
    """
    c = np.asarray(corners, dtype=float)
    d1 = c[:, 2] - c[:, 0]
    d2 = c[:, 3] - c[:, 1]
    nvec = np.cross(d1, d2)
    nn = np.linalg.norm(nvec, axis=1)
    if not np.all(nn > 0.0):
        raise ValueError("degenerate panel (zero area)")
    normal = nvec / nn[:, None]
    mean = c.mean(axis=1)
    off = np.einsum("nkc,nc->nk", c - mean[:, None, :], normal)
    flat = c - off[:, :, None] * normal[:, None, :]
    # area centroid from the two triangles (0,1,2) and (0,2,3)
    t1 = 0.5 * np.einsum("nc,nc->n", np.cross(flat[:, 1] - flat[:, 0], flat[:, 2] - flat[:, 0]), normal)
    t2 = 0.5 * np.einsum("nc,nc->n", np.cross(flat[:, 2] - flat[:, 0], flat[:, 3] - flat[:, 0]), normal)
    area = t1 + t2
    if not np.all(area > 0.0):
        raise ValueError("panel with non-positive area (folded quadrilateral)")
    g1 = (flat[:, 0] + flat[:, 1] + flat[:, 2]) / 3.0
    g2 = (flat[:, 0] + flat[:, 2] + flat[:, 3]) / 3.0
    centre = (t1[:, None] * g1 + t2[:, None] * g2) / area[:, None]
    return PanelSet(corners=flat, centre=centre, normal=normal, area=area)


def _panel_edges(xp: Any, pan: PanelSet, dtype: Any) -> tuple[Any, Any, Any, Any]:
    # edge lengths and directions in double precision, then cast
    edge = np.roll(pan.corners, -1, axis=1) - pan.corners
    dh = np.sqrt((edge * edge).sum(-1))
    th = edge / np.maximum(dh, 1e-300)[:, :, None]
    c = xp.asarray(pan.corners, dtype=dtype)
    nrm = xp.asarray(pan.normal, dtype=dtype)
    d = xp.asarray(dh, dtype=dtype)
    t = xp.asarray(th, dtype=dtype)
    nu = xp.cross(t, nrm[:, None, :])  # in-plane, outwards across the edge
    return c, nrm, d, nu


def _influence_block(
    xp: Any,
    pts: Any,
    pan: PanelSet,
    *,
    coplanar: bool = False,
    pair_budget: int = 250_000,
    dtype: Any = np.float64,
) -> Any:
    """Velocity at ``pts`` (m, 3) induced by unit source density on each panel: (m, n, 3).

    The potential of a panel is ``-1/(4 pi) int dS / r``. In the panel's plane the velocity is
    ``1/(4 pi) sum_k nu_k ln[(r_k + r_k+1 + d_k) / (r_k + r_k+1 - d_k)]`` over the edges, and along
    the normal it is the solid angle the panel subtends, over 4 pi, signed by the side.
    ``coplanar`` skips the solid angle for points in the plane of every panel and outside them.
    """
    c, nrm, d, nu = _panel_edges(xp, pan, dtype)
    m, n = int(pts.shape[0]), pan.n
    out = xp.empty((m, n, 3), dtype=dtype)
    tiny = float(np.finfo(dtype).tiny)
    step = max(1, pair_budget // max(n, 1))

    def chunk(s: int) -> None:
        p = pts[s : s + step]
        rv = c[None, :, :, :] - p[:, None, None, :]
        r = xp.sqrt((rv * rv).sum(-1))
        ssum = r + xp.roll(r, -1, axis=2)
        lg = xp.log((ssum + d[None]) / xp.maximum(ssum - d[None], tiny))
        v = xp.einsum("mnk,nkc->mnc", lg, nu)
        if not coplanar:
            a, b, cc, e = rv[:, :, 0], rv[:, :, 1], rv[:, :, 2], rv[:, :, 3]
            ra, rb, rc, re = r[:, :, 0], r[:, :, 1], r[:, :, 2], r[:, :, 3]
            ac = (a * cc).sum(-1)
            num1 = (a * xp.cross(b, cc)).sum(-1)
            den1 = ra * rb * rc + (a * b).sum(-1) * rc + ac * rb + (b * cc).sum(-1) * ra
            num2 = (a * xp.cross(cc, e)).sum(-1)
            den2 = ra * rc * re + ac * re + (a * e).sum(-1) * rc + (cc * e).sum(-1) * ra
            omega = 2.0 * (xp.arctan2(num1, den1) + xp.arctan2(num2, den2))
            v -= omega[:, :, None] * nrm[None, :, :]
        out[s : s + step] = v

    starts = range(0, m, step)
    threads = host_threads()
    if xp is np and threads > 1 and len(starts) > 1:
        # numpy releases the interpreter lock inside its array loops, so chunks run in parallel
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(chunk, starts))
    else:
        for s in starts:
            chunk(s)
    out *= 1.0 / FOUR_PI
    return out


def velocity_influence(
    xp: Any,
    points: np.ndarray,
    pan: PanelSet,
    *,
    self_term: bool = False,
    coplanar: bool = False,
    z_image: bool = False,
    pair_budget: int = 250_000,
    dtype: Any = np.float64,
) -> Any:
    """Velocity influence with the image in y = 0: (m, n, 3).

    ``self_term``: ``points`` are the panels' own collocation points; the normal velocity of a
    panel on itself is set to +1/2 (the fluid side). ``z_image`` returns a second array holding
    the images in z = 0 (of the panel and of its y image), used for the double-body flow.
    """
    pts = xp.asarray(points, dtype=dtype)
    kw = dict(coplanar=coplanar, pair_budget=pair_budget, dtype=dtype)
    v = _influence_block(xp, pts, pan, **kw)
    if self_term:
        if pts.shape[0] != pan.n:
            raise ValueError("self_term needs one point per panel")
        idx = xp.arange(pan.n)
        nrm = xp.asarray(pan.normal, dtype=dtype)
        diag = v[idx, idx, :]
        vn = (diag * nrm).sum(-1)
        v[idx, idx, :] = diag + (0.5 - vn)[:, None] * nrm
    my = xp.asarray([1.0, -1.0, 1.0], dtype=dtype)
    v += _influence_block(xp, pts * my, pan, **kw) * my
    if not z_image:
        return v
    mz = xp.asarray([1.0, 1.0, -1.0], dtype=dtype)
    kz = dict(pair_budget=pair_budget, dtype=dtype)
    w = _influence_block(xp, pts * mz, pan, **kz) * mz
    w += _influence_block(xp, pts * (my * mz), pan, **kz) * (my * mz)
    return v, w


# --------------------------------------------------------------------------------------------
# geometry: pose, clipping, hydrostatics, sections
# --------------------------------------------------------------------------------------------


def _rot_trim(trim: float) -> np.ndarray:
    ct, st = math.cos(trim), math.sin(trim)
    return np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])


def pose_triangles(vertices: np.ndarray, faces: np.ndarray, h: float, trim: float = 0.0) -> np.ndarray:
    """Triangles (n, 3, 3) of the mesh in the water frame."""
    w = np.asarray(vertices, dtype=float) @ _rot_trim(trim).T
    w[:, 2] -= h
    return w[np.asarray(faces, dtype=int)]


def _clip(tris: np.ndarray, axis: int, sign: float) -> np.ndarray:
    """The part of each triangle where ``sign * coordinate < 0``; orientation is kept."""
    val = sign * tris[:, :, axis]
    below = val < 0.0
    nb = below.sum(1)
    out = [tris[nb == 3]]
    for k in (1, 2):
        sel = nb == k
        if not sel.any():
            continue
        tk, bk, vk = tris[sel], below[sel], val[sel]
        odd = (bk if k == 1 else ~bk).argmax(1)
        idx = (odd[:, None] + np.arange(3)[None]) % 3
        tk = np.take_along_axis(tk, idx[:, :, None], 1)
        vk = np.take_along_axis(vk, idx, 1)
        a, b, c = tk[:, 0], tk[:, 1], tk[:, 2]
        tb = (vk[:, 0] / (vk[:, 0] - vk[:, 1]))[:, None]
        tc = (vk[:, 0] / (vk[:, 0] - vk[:, 2]))[:, None]
        ab = a + tb * (b - a)
        ac = a + tc * (c - a)
        ab[:, axis] = 0.0
        ac[:, axis] = 0.0
        if k == 1:
            out.append(np.stack([a, ab, ac], 1))
        else:
            out.append(np.stack([ab, b, c], 1))
            out.append(np.stack([ab, c, ac], 1))
    return np.concatenate(out)


def hydrostatics(tris: np.ndarray) -> dict[str, Any]:
    """Immersed volume, centre of buoyancy and waterline extent of posed triangles (water frame).

    The mesh must be closed below the waterplane and consistently oriented (either way).
    """
    t = _clip(tris, 2, 1.0)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    n = np.cross(b - a, c - a) / 2.0
    az = n[:, 2]
    x, z = t[:, :, 0], t[:, :, 2]
    vol = float((az * z.mean(1)).sum())
    if vol < 0.0:
        az, vol = -az, -vol
    out: dict[str, Any] = {"vol": vol, "wet": float(np.linalg.norm(n, axis=1).sum()), "tris": t}
    if vol <= 0.0:
        return out

    def m2(p: np.ndarray, q: np.ndarray) -> float:
        return float((az / 12.0 * (p.sum(1) * q.sum(1) + (p * q).sum(1))).sum())

    out["xb"] = m2(x, z) / vol
    out["zb"] = m2(z, z) / (2.0 * vol)
    on = np.abs(t[:, :, 2]) < 1e-12
    px = t[:, :, 0][on]
    out["wl_x"] = (float(px.min()), float(px.max()))
    out["x_range"] = (float(t[:, :, 0].min()), float(t[:, :, 0].max()))
    out["z_min"] = float(t[:, :, 2].min())
    return out


def static_pose(vertices: np.ndarray, faces: np.ndarray, volume_m3: float, trim: float = 0.0) -> float:
    """Waterplane height ``h`` at which the mesh displaces ``volume_m3`` at the given trim."""
    w = np.asarray(vertices, dtype=float) @ _rot_trim(trim).T
    lo, hi = float(w[:, 2].min()) + 1e-6, float(w[:, 2].max()) - 1e-6

    def f(h: float) -> float:
        return hydrostatics(pose_triangles(vertices, faces, h, trim))["vol"] - volume_m3

    if f(hi) < 0.0:
        raise ValueError("the mesh cannot displace the requested volume")
    return float(brentq(f, lo, hi, xtol=1e-11))


def _section(half_tris: np.ndarray, xc: float) -> np.ndarray:
    """Half section at x = xc as a polyline (k, 2) of (y, z) from the centreline to the waterline."""
    pos = half_tris[:, :, 0] >= xc
    npos = pos.sum(1)
    sel = (npos == 1) | (npos == 2)
    t, p_ = half_tris[sel], pos[sel]
    if not len(t):
        raise ValueError(f"no hull section at x = {xc:.6f}")
    odd = np.where(p_.sum(1) == 1, p_.argmax(1), (~p_).argmax(1))
    idx = (odd[:, None] + np.arange(3)[None]) % 3
    t = np.take_along_axis(t, idx[:, :, None], 1)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    tb = ((xc - a[:, 0]) / (b[:, 0] - a[:, 0]))[:, None]
    tc = ((xc - a[:, 0]) / (c[:, 0] - a[:, 0]))[:, None]
    seg = np.stack([a[:, 1:] + tb * (b[:, 1:] - a[:, 1:]), a[:, 1:] + tc * (c[:, 1:] - a[:, 1:])], 1)
    # join the segments into one curve
    tol = 1e-9
    keys = np.round(seg.reshape(-1, 2) / tol).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    pts = np.zeros((len(uniq), 2))
    pts[inv.reshape(-1)] = seg.reshape(-1, 2)
    inv = inv.reshape(-1, 2)
    inv = inv[inv[:, 0] != inv[:, 1]]
    if not len(inv):
        raise ValueError(f"hull section at x = {xc:.6f} is a single point")
    inv = np.unique(np.sort(inv, axis=1), axis=0)
    nbr: dict[int, list[int]] = {}
    for u, v in inv:
        nbr.setdefault(int(u), []).append(int(v))
        nbr.setdefault(int(v), []).append(int(u))
    ends = [k for k, v in nbr.items() if len(v) == 1]
    if len(ends) != 2 or any(len(v) > 2 for v in nbr.values()):
        raise ValueError(
            f"hull section at x = {xc:.6f} is not one open curve from centreline to waterline "
            "(appendage, tunnel or several hulls?)"
        )
    start = min(ends, key=lambda k: pts[k, 0])  # the centreline end has the smaller y
    chain = [start]
    prev = -1
    while True:
        nxt = [k for k in nbr[chain[-1]] if k != prev]
        if not nxt:
            break
        prev = chain[-1]
        chain.append(nxt[0])
    if len(chain) != len(nbr):
        raise ValueError(f"hull section at x = {xc:.6f} has more than one piece")
    return pts[chain]


def _end_section(half: np.ndarray, x0: float, x1: float, bow: bool, nz: int) -> np.ndarray:
    """Section at a waterline end, drawn onto the centreplane: a point or a vertical stem line.

    The cut is taken just inside the end. A mesh can be ragged there on the scale of micrometres
    (slivers of the waterplane beside a keel that is just dry), so the cut moves inwards until it
    is one curve; the end must close within a thousandth of the waterline length.
    """
    lwl = x1 - x0
    err: ValueError | None = None
    for inset in (1e-6, 1e-5, 1e-4, 1e-3):
        try:
            sec = _resample(_section(half, x0 + inset * lwl if bow else x1 - inset * lwl), nz)
        except ValueError as exc:
            err = exc
            continue
        if sec[:, 0].max() > 2.0 * max(inset, 1e-3) * lwl:
            raise ValueError("the waterline does not close at its end (immersed transom?)")
        sec[:, 0] = 0.0
        return sec
    raise ValueError(f"no usable section at the waterline end: {err}")


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    si = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(si, s, poly[:, 0]), np.interp(si, s, poly[:, 1])])


def _geometric_fractions(n: int, first: float) -> np.ndarray:
    """Node fractions 0..1 of n intervals growing geometrically from a first interval ``first``."""
    if first * n >= 1.0 - 1e-3:
        return np.linspace(0.0, 1.0, n + 1)

    def total(r: float) -> float:
        return first * (r**n - 1.0) / (r - 1.0) - 1.0

    r = brentq(total, 1.0 + 1e-6, 50.0)
    w = first * r ** np.arange(n)
    s = np.concatenate([[0.0], np.cumsum(w)])
    return s / s[-1]


def _station_fractions(nx: int, cluster: float) -> np.ndarray:
    """Station positions 0..1 along the waterline: equal, blended with a cosine distribution."""
    if not 0.0 <= cluster < 1.0:
        raise ValueError("end_cluster must be in [0, 1)")
    t = np.arange(nx + 1) / nx
    return (1.0 - cluster) * t + cluster * 0.5 * (1.0 - np.cos(math.pi * t))


def _outer_columns(first: float, nominal: float, extent: float, growth: float) -> np.ndarray:
    """Column lengths beyond a waterline end: growing from ``first`` to ``nominal``, then equal."""
    out: list[float] = []
    w = first
    while sum(out) < extent - 1e-9 or len(out) < 3:
        w = min(w * growth, nominal) if w < nominal else nominal
        out.append(w)
    return np.array(out)


@dataclass
class _Grid:
    hull: PanelSet
    fs: PanelSet
    fs_pts: np.ndarray  # (n_fs, 3) free-surface collocation points, on z = 0
    nxf: int
    ny: int
    n_up: int
    nx: int
    x_bow: float
    lwl: float
    dx: float  # nominal station spacing, lwl / nx
    x_nodes: np.ndarray  # (nxf + 1,) free-surface station lines
    wl_dy: np.ndarray  # (nx,) change of waterline half-breadth over each hull station interval
    half_width: float
    wetted_area: float


def _build_grid(tris_water: np.ndarray, cfg: PanelConfig) -> _Grid:
    """Hull and free-surface panels for posed triangles (both sides, water frame)."""
    below = _clip(tris_water, 2, 1.0)
    half = _clip(below, 1, -1.0)
    if not len(half):
        raise ValueError("nothing immersed")
    on = np.abs(half[:, :, 2]) < 1e-12
    wl_x = half[:, :, 0][on]
    if not len(wl_x):
        raise ValueError("the hull does not cut the waterplane")
    x0, x1 = float(wl_x.min()), float(wl_x.max())
    lwl = x1 - x0
    lo, hi = float(half[:, :, 0].min()), float(half[:, :, 0].max())
    if x0 - lo > 1e-3 * lwl or hi - x1 > 1e-3 * lwl:
        raise ValueError(
            "the immersed hull reaches beyond the waterline ends (bulb or overhang under water); "
            "this solver needs pointed waterline ends"
        )
    nx, nz = cfg.nx_hull, cfg.nz_hull
    dx = lwl / nx
    frac_x = _station_fractions(nx, cfg.end_cluster)
    xs = x0 + lwl * frac_x
    xs[-1] = x1
    nodes = np.zeros((nx + 1, nz + 1, 3))
    for i, xc in enumerate(xs):
        if i in (0, nx):
            sec = _end_section(half, x0, x1, i == 0, nz)
        else:
            sec = _resample(_section(half, xc), nz)
        nodes[i, :, 0] = xc
        nodes[i, :, 1:] = sec
    a = nodes[:-1, :-1].reshape(-1, 3)
    b = nodes[1:, :-1].reshape(-1, 3)
    c = nodes[1:, 1:].reshape(-1, 3)
    d = nodes[:-1, 1:].reshape(-1, 3)
    hull = make_panels(np.stack([a, d, c, b], axis=1))  # normal out of the hull

    up = lwl * _outer_columns(frac_x[1] - frac_x[0], 1.0 / nx, cfg.upstream, cfg.end_growth)
    dn = lwl * _outer_columns(frac_x[-1] - frac_x[-2], 1.0 / nx, cfg.downstream, cfg.end_growth)
    n_up, n_dn = len(up), len(dn)
    nxf = n_up + nx + n_dn
    xn = np.concatenate([x0 - np.cumsum(up)[::-1], xs, x1 + np.cumsum(dn)])
    col = np.repeat(np.diff(xn), cfg.ny_fs)  # length of each free-surface panel along x
    ywl = np.zeros(nxf + 1)
    ywl[n_up : n_up + nx + 1] = nodes[:, nz, 1]
    width = cfg.half_width * lwl
    if width <= 2.0 * float(ywl.max()):
        raise ValueError("free-surface half width is too small for this hull")
    frac = _geometric_fractions(cfg.ny_fs, cfg.first_row * dx / width)
    yn = ywl[:, None] + (width - ywl[:, None]) * frac[None, :]
    fn = np.zeros((nxf + 1, cfg.ny_fs + 1, 3))
    fn[:, :, 0] = xn[:, None]
    fn[:, :, 1] = yn
    a = fn[:-1, :-1].reshape(-1, 3)
    b = fn[1:, :-1].reshape(-1, 3)
    c = fn[1:, 1:].reshape(-1, 3)
    d = fn[:-1, 1:].reshape(-1, 3)
    fs = make_panels(np.stack([a, d, c, b], axis=1))  # normal down, into the water
    fs_pts = fs.centre.copy()  # collocation on z = 0
    if cfg.fs_shift != 0.0:
        if cfg.fs_raise <= 0.0:
            raise ValueError("fs_shift needs raised free-surface panels (fs_raise > 0)")
        ymid = 0.5 * (yn[:, :-1] + yn[:, 1:])  # row centrelines at the station lines
        slope = (np.diff(ymid, axis=0) / np.diff(xn)[:, None]).reshape(-1)
        fs_pts[:, 0] -= cfg.fs_shift * col
        fs_pts[:, 1] -= cfg.fs_shift * col * slope
    if cfg.fs_raise > 0.0:
        lift = np.zeros((fs.n, 3))
        lift[:, 2] = cfg.fs_raise * col
        fs = PanelSet(
            corners=fs.corners + lift[:, None, :], centre=fs.centre + lift, normal=fs.normal, area=fs.area
        )
    return _Grid(
        hull=hull,
        fs=fs,
        fs_pts=fs_pts,
        nxf=nxf,
        ny=cfg.ny_fs,
        n_up=n_up,
        nx=nx,
        x_bow=x0,
        lwl=lwl,
        dx=dx,
        x_nodes=xn,
        wl_dy=np.diff(nodes[:, nz, 1]),
        half_width=width,
        wetted_area=2.0 * float(hull.area.sum()),
    )


# --------------------------------------------------------------------------------------------
# difference operators on the free-surface grid
# --------------------------------------------------------------------------------------------


def _d_xi(xp: Any, f: Any, *, upstream_zero: bool) -> Any:
    """Upstream difference along the grid lines (axis 0): Dawson's four-point operator.

    ``upstream_zero`` sets the derivative to zero in the first column (the radiation condition
    for the flow quantities); otherwise the first column uses a forward difference (metrics).
    """
    out = xp.empty_like(f)
    out[3:] = (10.0 * f[3:] - 15.0 * f[2:-1] + 6.0 * f[1:-2] - f[:-3]) / 6.0
    out[2] = 1.5 * f[2] - 2.0 * f[1] + 0.5 * f[0]
    out[1] = f[1] - f[0]
    out[0] = 0.0 if upstream_zero else f[1] - f[0]
    return out


def _d_eta(xp: Any, f: Any) -> Any:
    """Central difference across the grid lines (axis 1), one-sided at the two edges."""
    out = xp.empty_like(f)
    out[:, 1:-1] = 0.5 * (f[:, 2:] - f[:, :-2])
    out[:, 0] = -1.5 * f[:, 0] + 2.0 * f[:, 1] - 0.5 * f[:, 2]
    out[:, -1] = 1.5 * f[:, -1] - 2.0 * f[:, -2] + 0.5 * f[:, -3]
    return out


# --------------------------------------------------------------------------------------------
# the flow solve at one pose
# --------------------------------------------------------------------------------------------


def solve_fixed(
    tris_water: np.ndarray,
    speeds_mps: Any,
    cfg: PanelConfig = PanelConfig(),
    *,
    rho: float = 1025.0,
    g: float = G,
    backend: str = "numpy",
    h: float = 0.0,
    trim: float = 0.0,
    keep_profile: bool = False,
) -> list[PanelResult]:
    """Flow about a hull held in the pose its triangles already have (water frame, bow at -x).

    ``tris_water`` are both sides of the hull, (n, 3, 3). The influence matrices do not depend
    on speed, so several speeds at one pose share one assembly. ``h`` and ``trim`` are only
    recorded in the results.
    """
    if cfg.linearisation not in ("dawson", "kelvin"):
        raise ValueError("linearisation must be 'dawson' or 'kelvin'")
    if cfg.waterline_eta not in ("row0", "extrapolated"):
        raise ValueError("waterline_eta must be 'row0' or 'extrapolated'")
    if cfg.precision not in ("double", "single"):
        raise ValueError("precision must be 'double' or 'single'")
    xp = get_backend(backend)
    dt = np.float64 if cfg.precision == "double" else np.float32
    speeds = [float(u) for u in np.atleast_1d(np.asarray(speeds_mps, dtype=float))]
    t0 = time.perf_counter()
    grid = _build_grid(tris_water, cfg)
    hull, fs = grid.hull, grid.fs
    nh, nf = hull.n, fs.n
    n = nh + nf
    kw: dict[str, Any] = dict(pair_budget=cfg.pair_budget, dtype=dt)
    nrm_h = xp.asarray(hull.normal, dtype=dt)
    area_h = xp.asarray(hull.area, dtype=dt)
    ctr_h = xp.asarray(hull.centre, dtype=dt)

    # influence of the hull panels on the hull, with and without the image in z = 0
    v_hh, v_hh_z = velocity_influence(xp, hull.centre, hull, self_term=True, z_image=True, **kw)
    # double-body flow for a unit stream
    a_db = ((v_hh + v_hh_z) * nrm_h[:, None, :]).sum(-1)
    sig_b = xp.linalg.solve(a_db, -nrm_h[:, 0])
    del a_db
    ex = xp.asarray([1.0, 0.0, 0.0], dtype=dt)
    vel_db_h = ex[None, :] + xp.einsum("mnc,n->mc", v_hh + v_hh_z, sig_b)
    del v_hh_z
    cp_db = 1.0 - (vel_db_h * vel_db_h).sum(-1)
    cfx_db = float(-2.0 * (cp_db * nrm_h[:, 0] * area_h).sum())

    v_hf = velocity_influence(xp, hull.centre, fs, **kw)  # at the hull, from the surface
    v_fh = velocity_influence(xp, grid.fs_pts, hull, **kw)  # at the surface, from the hull
    raised = cfg.fs_raise > 0.0
    v_ff = velocity_influence(
        xp, grid.fs_pts, fs, self_term=not raised, coplanar=not raised, **kw
    )

    if cfg.linearisation == "dawson":
        base_fx = 1.0 + 2.0 * (v_fh[:, :, 0] @ sig_b)
        base_fy = 2.0 * (v_fh[:, :, 1] @ sig_b)
        base_h = vel_db_h
    else:
        base_fx = xp.ones(nf, dtype=dt)
        base_fy = xp.zeros(nf, dtype=dt)
        base_h = xp.broadcast_to(ex, (nh, 3))

    # F sigma = D phi' at the free-surface points
    fmat = xp.empty((nf, n), dtype=dt)
    fmat[:, :nh] = base_fx[:, None] * v_fh[:, :, 0] + base_fy[:, None] * v_fh[:, :, 1]
    fmat[:, nh:] = base_fx[:, None] * v_ff[:, :, 0] + base_fy[:, None] * v_ff[:, :, 1]
    # d/dz at the surface points: from the hull, and from the surface panels (-1/2 I if not raised)
    zmat = xp.empty((nf, n), dtype=dt)
    zmat[:, :nh] = v_fh[:, :, 2]
    zmat[:, nh:] = v_ff[:, :, 2]
    del v_ff, v_fh

    shape = (grid.nxf, grid.ny)
    xc = xp.asarray(grid.fs_pts[:, 0], dtype=dt).reshape(shape)
    yc = xp.asarray(grid.fs_pts[:, 1], dtype=dt).reshape(shape)
    x_xi, y_xi = _d_xi(xp, xc, upstream_zero=False), _d_xi(xp, yc, upstream_zero=False)
    x_et, y_et = _d_eta(xp, xc), _d_eta(xp, yc)
    jac = x_xi * y_et - x_et * y_xi
    bx, by = base_fx.reshape(shape), base_fy.reshape(shape)
    ca = (bx * y_et - by * x_et) / jac
    cb = (by * x_xi - bx * y_xi) / jac
    q2 = bx * bx + by * by
    dq2 = ca * _d_xi(xp, q2, upstream_zero=True) + cb * _d_eta(xp, q2)
    cc = dq2 / (2.0 * q2)

    f3 = fmat.reshape(grid.nxf, grid.ny, n)
    lmat = ca[:, :, None] * _d_xi(xp, f3, upstream_zero=True)
    lmat += cb[:, :, None] * _d_eta(xp, f3)
    lmat += cc[:, :, None] * f3
    lmat = lmat.reshape(nf, n)

    a_mat = xp.empty((n, n), dtype=dt)
    a_mat[:nh, :nh] = (v_hh * nrm_h[:, None, :]).sum(-1)
    a_mat[:nh, nh:] = (v_hf * nrm_h[:, None, :]).sum(-1)
    rhs = xp.empty(n, dtype=dt)
    rhs[:nh] = -(base_h * nrm_h).sum(-1)
    rhs[nh:] = -0.5 * dq2.reshape(-1)
    _sync(xp)
    assembly_s = time.perf_counter() - t0

    ff_geom = None
    if cfg.farfield:
        t2 = time.perf_counter()
        ff_geom = _farfield_points(grid, cfg, max(speeds), g)
        ff_pts = np.concatenate([ff_geom["aft"], ff_geom["fwd"], ff_geom["side"]])
        if cfg.linearisation == "dawson":
            ff_geom["v_h"], ff_geom["w_h"] = velocity_influence(xp, ff_pts, hull, z_image=True, **kw)
        else:
            ff_geom["v_h"] = velocity_influence(xp, ff_pts, hull, **kw)
        ff_geom["v_f"] = velocity_influence(xp, ff_pts, fs, **kw)
        _sync(xp)
        ff_geom["setup_s"] = time.perf_counter() - t2
    results: list[PanelResult] = []
    for u in speeds:
        t1 = time.perf_counter()
        k0 = g / (u * u)
        a_mat[nh:, :] = lmat + k0 * zmat
        sig = xp.linalg.solve(a_mat, rhs)
        _sync(xp)
        solve_s = time.perf_counter() - t1

        vel_h = base_h + xp.einsum("mnc,n->mc", v_hh, sig[:nh]) + xp.einsum("mnc,n->mc", v_hf, sig[nh:])
        cp = 1.0 - (vel_h * vel_h).sum(-1)
        dphi = vel_h - base_h
        cp_lin = 1.0 - (base_h * base_h).sum(-1) - 2.0 * (base_h * dphi).sum(-1)
        qd = 0.5 * rho * u * u
        fx = float(-2.0 * (cp * nrm_h[:, 0] * area_h).sum())
        fx_lin = float(-2.0 * (cp_lin * nrm_h[:, 0] * area_h).sum())
        fz = float(-2.0 * (cp * nrm_h[:, 2] * area_h).sum())
        my = float(
            -2.0 * (cp * area_h * (ctr_h[:, 2] * nrm_h[:, 0] - ctr_h[:, 0] * nrm_h[:, 2])).sum()
        )
        eta = (u * u / (2.0 * g)) * (1.0 - q2.reshape(-1) - 2.0 * (fmat @ sig))
        eta2 = _host(eta).reshape(shape)
        # elevation beside the hull at the middle of each station interval: the collocation
        # points sit fs_shift spacings upstream of there
        rows = slice(grid.n_up, grid.n_up + grid.nx)
        nxt = slice(grid.n_up + 1, grid.n_up + grid.nx + 1)
        eta_hull = (1.0 - cfg.fs_shift) * eta2[rows, :2] + cfg.fs_shift * eta2[nxt, :2]
        if cfg.waterline_eta == "row0":
            eta_wl = eta_hull[:, 0]
        else:
            ych = (1.0 - cfg.fs_shift) * _host(yc)[rows, :2] + cfg.fs_shift * _host(yc)[nxt, :2]
            wl_mid = _waterline_mid(grid)
            d0, d1 = ych[:, 0] - wl_mid, ych[:, 1] - wl_mid
            eta_wl = eta_hull[:, 0] - (eta_hull[:, 1] - eta_hull[:, 0]) * d0 / (d1 - d0)
        rw_wl = rho * g * float((eta_wl**2 * grid.wl_dy).sum())
        rw_p = qd * (fx - cfx_db)
        ff_total, ff_parts, ff_s = None, None, 0.0
        if ff_geom is not None:
            t2 = time.perf_counter()
            ff_total, ff_parts = _farfield(
                xp, grid, cfg, ff_geom, sig, sig_b, eta2, _host(yc), u, rho, g
            )
            _sync(xp)
            ff_s = time.perf_counter() - t2 + ff_geom["setup_s"] / len(speeds)
        res = PanelResult(
            speed_mps=u,
            froude=u / math.sqrt(g * grid.lwl),
            lwl_m=grid.lwl,
            wetted_area_m2=grid.wetted_area,
            rw_N=rw_p,
            rw_waterline_N=rw_wl,
            rw_doublebody_zero_N=qd * cfx_db,
            rw_pressure_linear_N=qd * (fx_lin - cfx_db),
            rw_farfield_N=ff_total,
            farfield_parts_N=ff_parts,
            fz_dyn_N=qd * fz,
            my_dyn_Nm=qd * my,
            cw=rw_p / (qd * grid.wetted_area),
            h_m=h,
            trim_rad=trim,
            n_hull=nh,
            n_fs=nf,
            assembly_s=assembly_s,
            solve_s=solve_s,
            farfield_s=ff_s,
            backend=backend,
        )
        if keep_profile:
            res.wave_profile = {
                "x_over_L": (
                    (0.5 * (grid.x_nodes[rows] + grid.x_nodes[nxt]) - grid.x_bow) / grid.lwl
                ).tolist(),
                "eta_m": [float(v) for v in eta_hull[:, 0]],
            }
        results.append(res)
    return results


def _waterline_mid(grid: _Grid) -> np.ndarray:
    """Waterline half-breadth at the middle of each hull station interval."""
    nodes_y = grid.fs.corners[:, :, 1].reshape(grid.nxf, grid.ny, 4)[grid.n_up : grid.n_up + grid.nx, 0]
    return 0.5 * (nodes_y[:, 0] + nodes_y[:, 3])


# --------------------------------------------------------------------------------------------
# control-surface (far-field) estimate
# --------------------------------------------------------------------------------------------


def _farfield_points(grid: _Grid, cfg: PanelConfig, u_max: float, g: float) -> dict[str, Any]:
    """Quadrature points of the three control planes."""
    xcol = grid.fs_pts[:, 0].reshape(grid.nxf, grid.ny)[:, 0]
    x_stern = grid.x_bow + grid.lwl
    i_aft = int(np.argmin(np.abs(xcol - (x_stern + cfg.ff_behind_stern * grid.lwl))))
    i_fwd = int(np.argmin(np.abs(xcol - (grid.x_bow - cfg.ff_ahead_bow * grid.lwl))))
    i_aft = min(max(i_aft, grid.n_up + grid.nx), grid.nxf - 2)
    i_fwd = min(max(i_fwd, 1), grid.n_up - 1)
    y_side = cfg.ff_side * grid.half_width
    depth = cfg.ff_depth_wavelengths * 2.0 * math.pi * u_max * u_max / g
    tq, wq = np.polynomial.legendre.leggauss(cfg.ff_nz)
    tq, wq = 0.5 * (tq + 1.0), 0.5 * wq
    zq = -depth * tq * tq
    wz = depth * 2.0 * tq * wq
    yq = (np.arange(cfg.ff_ny) + 0.5) * y_side / cfg.ff_ny
    wy = np.full(cfg.ff_ny, y_side / cfg.ff_ny)

    def plane(x: float) -> np.ndarray:
        yy, zz = np.meshgrid(yq, zq, indexing="ij")
        return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])

    xs_side = xcol[i_fwd : i_aft + 1]  # collocation columns between the two planes
    xx, zz = np.meshgrid(xs_side, zq, indexing="ij")
    side = np.column_stack([xx.ravel(), np.full(xx.size, y_side), zz.ravel()])
    # trapezoid rule along the side plane, from plane to plane
    wx = np.zeros(len(xs_side))
    wx[:-1] += 0.5 * np.diff(xs_side)
    wx[1:] += 0.5 * np.diff(xs_side)
    return {
        "i_aft": i_aft,
        "i_fwd": i_fwd,
        "aft": plane(float(xcol[i_aft])),
        "fwd": plane(float(xcol[i_fwd])),
        "side": side,
        "wy": wy,
        "wz": wz,
        "wx": wx,
        "yq": yq,
        "y_side": y_side,
    }


def _farfield(
    xp: Any,
    grid: _Grid,
    cfg: PanelConfig,
    geo: dict[str, Any],
    sig: Any,
    sig_b: Any,
    eta2: np.ndarray,
    yc: np.ndarray,
    u: float,
    rho: float,
    g: float,
) -> tuple[float, dict[str, float]]:
    """Wave resistance from the momentum flux through the control planes.

    With (u, v, w) the velocity less the stream, for both sides of the hull::

        R = [rho g int eta^2 dy + rho int int (v^2 + w^2 - u^2) dz dy] (aft plane - forward plane)
            - 2 rho int int u v dz dx (side plane)

    with y from 0 to the side plane and z from the depth cut-off to 0.
    """
    nh = grid.hull.n
    vel = xp.einsum("mnc,n->mc", geo["v_f"], sig[nh:])
    if cfg.linearisation == "dawson":
        vel += xp.einsum("mnc,n->mc", geo["v_h"], sig[:nh] + sig_b)
        vel += xp.einsum("mnc,n->mc", geo["w_h"], sig_b)
    else:
        vel += xp.einsum("mnc,n->mc", geo["v_h"], sig[:nh])
    vel_all = _host(vel) * u
    n_plane = len(geo["aft"])
    pert = {"aft": vel_all[:n_plane], "fwd": vel_all[n_plane : 2 * n_plane], "side": vel_all[2 * n_plane :]}

    ny, nz = cfg.ff_ny, cfg.ff_nz
    parts: dict[str, float] = {}
    for name, col in (("aft", geo["i_aft"]), ("fwd", geo["i_fwd"])):
        vel = pert[name].reshape(ny, nz, 3)
        kin = vel[:, :, 1] ** 2 + vel[:, :, 2] ** 2 - vel[:, :, 0] ** 2
        t2 = rho * float((kin * geo["wy"][:, None] * geo["wz"][None, :]).sum())
        eta_q = np.interp(geo["yq"], yc[col], eta2[col])
        t1 = rho * g * float((eta_q**2 * geo["wy"]).sum())
        parts[f"{name}_potential"] = t1
        parts[f"{name}_kinetic"] = t2
    vel = pert["side"].reshape(len(geo["wx"]), nz, 3)
    t3 = -2.0 * rho * float((vel[:, :, 0] * vel[:, :, 1] * geo["wx"][:, None] * geo["wz"][None, :]).sum())
    parts["side"] = t3
    total = (
        parts["aft_potential"] + parts["aft_kinetic"] - parts["fwd_potential"] - parts["fwd_kinetic"] + t3
    )
    return total, parts


# --------------------------------------------------------------------------------------------
# free to sink and trim
# --------------------------------------------------------------------------------------------


def solve_free(
    vertices: np.ndarray,
    faces: np.ndarray,
    speed_mps: float,
    cfg: PanelConfig = PanelConfig(),
    *,
    volume_m3: float,
    vcg_m: float,
    lcg_m: float | None = None,
    rho: float = 1025.0,
    g: float = G,
    backend: str = "numpy",
    max_iterations: int = 12,
    tol_m: float = 1e-6,
    keep_profile: bool = False,
) -> PanelResult:
    """Wave resistance with the hull free to sink and trim at fixed displacement and LCG.

    ``vertices``/``faces``: the hull in its body frame (bow at -x, z up), closed below any
    waterplane it will reach. ``volume_m3`` is the displaced volume at rest. ``lcg_m`` defaults
    to the centre of buoyancy at level trim, so the hull floats level at rest. ``vcg_m`` is the
    height of the centre of gravity in the body frame. The dynamic pressure force and moment
    from the flow at the current pose are held while the pose is re-balanced hydrostatically;
    the flow is then solved again, until the pose moves less than ``tol_m`` (sinkage, and trim
    times half the waterline length).
    """
    h0 = static_pose(vertices, faces, volume_m3, 0.0)
    hs0 = hydrostatics(pose_triangles(vertices, faces, h0, 0.0))
    lcg = float(hs0["xb"]) if lcg_m is None else float(lcg_m)
    weight = rho * g * volume_m3
    cg_body = np.array([lcg, 0.0, vcg_m])

    def balance(pose: np.ndarray, fz: float, my_o: float, fx: float) -> np.ndarray:
        h, tr = float(pose[0]), float(pose[1])
        hs = hydrostatics(pose_triangles(vertices, faces, h, tr))
        gw = _rot_trim(tr) @ cg_body
        gw[2] -= h
        buoy = rho * g * hs["vol"]
        my_g = my_o - (gw[2] * fx - gw[0] * fz)
        return np.array([(buoy + fz - weight) / weight, (-(hs["xb"] - gw[0]) * buoy + my_g) / weight])

    def rebalance(start: np.ndarray, fz: float, my_o: float, fx: float) -> np.ndarray:
        # Newton with fixed difference steps: the residual of a faceted mesh is only piecewise
        # smooth, so steps much smaller than a facet give a useless Jacobian
        pose = np.array(start, dtype=float)
        steps = np.array([1e-5, 1e-5])
        for _ in range(40):
            r0 = balance(pose, fz, my_o, fx)
            jac = np.empty((2, 2))
            for k in range(2):
                dp = np.zeros(2)
                dp[k] = steps[k]
                jac[:, k] = (balance(pose + dp, fz, my_o, fx) - balance(pose - dp, fz, my_o, fx)) / (
                    2.0 * steps[k]
                )
            delta = np.linalg.solve(jac, -r0)
            pose = pose + delta
            if float(np.abs(delta).max()) < 1e-11:
                return pose
        raise RuntimeError("hydrostatic balance did not converge")

    if lcg_m is None:
        pose_static = np.array([h0, 0.0])
    else:
        pose_static = rebalance(np.array([h0, 0.0]), 0.0, 0.0, 0.0)
    pose = pose_static.copy()
    res: PanelResult | None = None
    history: list[np.ndarray] = []
    converged = False
    it = 0
    lwl_half = 0.5 * float(np.ptp(hs0["wl_x"]))
    for it in range(1, max_iterations + 1):
        tris = pose_triangles(vertices, faces, float(pose[0]), float(pose[1]))
        res = solve_fixed(
            tris,
            speed_mps,
            cfg,
            rho=rho,
            g=g,
            backend=backend,
            h=float(pose[0]),
            trim=float(pose[1]),
            keep_profile=keep_profile,
        )[0]
        new = rebalance(pose, res.fz_dyn_N, res.my_dyn_Nm, res.rw_N)
        step = max(abs(new[0] - pose[0]), abs(new[1] - pose[1]) * lwl_half)
        if step < tol_m:
            converged = True
            break
        history.append(new)
        pose = new
        if len(history) == 3:
            # the plain iteration converges linearly: extrapolate its limit (Aitken) and restart
            p0, p1, p2 = history
            d1, d2 = p1 - p0, p2 - p1
            den = d2 - d1
            ok = np.abs(den) > 1e-14
            pose = np.where(ok, p2 - d2 * d2 / np.where(ok, den, 1.0), p2)
            history = []
    assert res is not None
    res.sinkage_m = float(pose[0] - pose_static[0])
    res.trim_change_rad = float(pose[1] - pose_static[1])
    res.pose_iterations = it
    res.pose_converged = converged
    return res


# --------------------------------------------------------------------------------------------
# test hulls
# --------------------------------------------------------------------------------------------


def wigley_mesh(
    length: float = 1.0,
    beam_ratio: float = 0.1,
    draft_ratio: float = 0.0625,
    freeboard_ratio: float = 0.03,
    nx: int = 240,
    nz: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle mesh of the Wigley parabolic hull, wall-sided above the design waterplane.

    ``y = (B/2) (1 - (2x/L)^2) (1 - (z/T)^2)`` for -T <= z <= 0, with the still waterplane at
    z = 0 and the hull centred on x = 0. Both sides; open at the top. Normals point outwards.
    """
    b, t = beam_ratio * length, draft_ratio * length
    fb = freeboard_ratio * length
    xs = 0.5 * length * np.cos(np.linspace(math.pi, 0.0, nx + 1))  # finer at the ends
    nfb = max(2, round(nz * fb / (t + fb)))
    zs = np.concatenate([-t * np.cos(np.linspace(0.0, 0.5 * math.pi, nz - nfb + 1)), np.linspace(0.0, fb, nfb + 1)[1:]])
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    yy = 0.5 * b * (1.0 - (2.0 * xx / length) ** 2) * np.where(zz < 0.0, 1.0 - (zz / t) ** 2, 1.0)
    yy[0, :] = yy[-1, :] = 0.0
    yy[:, 0] = 0.0
    ni, nk = xx.shape
    stb = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    port = stb * np.array([1.0, -1.0, 1.0])
    idx = np.arange(ni * nk).reshape(ni, nk)
    a, b_, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    f_stb = np.concatenate([np.stack([a, d, c], 1), np.stack([a, c, b_], 1)])
    f_port = f_stb[:, ::-1] + ni * nk
    return np.concatenate([stb, port]), np.concatenate([f_stb, f_port])
