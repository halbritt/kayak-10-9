"""Stage 10 sculpted hull + conformal Delaunay deck shell with keyhole & hatch recesses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
import trimesh
from kayakgen.model.hull import Hull
from surface_fairing import fit_half_surface

Z_KEEL_SHIFT = 0.1030  # shifts DistributionV2 z so keel min sits at z = 0.000 m
Z_WL_225 = 0.08811     # 225 lb seawater calm waterline above keel baseline
Z_WL_265 = 0.09780     # 265 lb seawater calm waterline above keel baseline


@dataclass(frozen=True)
class HatchSpec:
    name: str
    x0: float
    rx_lid: float
    ry_lid: float
    depth: float
    y0: float = 0.0


HATCHES: tuple[HatchSpec, ...] = (
    HatchSpec("bow_oval", x0=-1.44, rx_lid=0.170, ry_lid=0.072, depth=0.013, y0=0.0),
    HatchSpec("day_round", x0=+0.66, rx_lid=0.085, ry_lid=0.085, depth=0.013, y0=+0.118),
    HatchSpec("stern_oval", x0=+1.46, rx_lid=0.210, ry_lid=0.120, depth=0.014, y0=0.0),
)

KEYHOLE_X_FWD = -0.460
KEYHOLE_X_AFT = +0.380
KEYHOLE_X_MID = 0.5 * (KEYHOLE_X_FWD + KEYHOLE_X_AFT)  # -0.040 m
KEYHOLE_HALF_L = 0.5 * (KEYHOLE_X_AFT - KEYHOLE_X_FWD)  # 0.420 m (84.0 cm inner length)


def keyhole_inner_curve(theta: np.ndarray) -> np.ndarray:
    """Smooth C-infinity 84.0 x 42.0 cm racing/ocean keyhole inner loop (x, y)."""
    th = np.asarray(theta, dtype=float)
    u = -np.cos(th)  # -1 at forward nose (-0.46 m), +1 at aft rim (+0.38 m)
    x = KEYHOLE_X_MID + KEYHOLE_HALF_L * u
    # Calibrated so max inner width = 42.00 cm (half-width 0.2100 m) around x = +0.02..+0.12 m,
    # tapering gently over thighs to a rounded forward knee arch.
    half_w = (0.1935 + 0.0255 * np.tanh(1.65 * (u + 0.18)) - 0.0090 * (0.5 * (1.0 - u)) ** 2) * (0.2100 / 0.20226)
    y = np.sin(th) * half_w
    return np.column_stack((x, y))


def keyhole_offset_curve(theta: np.ndarray, delta: float) -> np.ndarray:
    """Parallel offset curve delta (m) outward from keyhole_inner_curve."""
    eps = 1e-4
    p0 = keyhole_inner_curve(theta)
    dp = (keyhole_inner_curve(theta + eps) - keyhole_inner_curve(theta - eps)) / (2.0 * eps)
    tang_norm = np.maximum(np.linalg.norm(dp, axis=1, keepdims=True), 1e-12)
    t = dp / tang_norm
    # As theta increases 0 -> pi (u goes -1 -> +1, y > 0), tangent points (+x),
    # so outward normal (+y on starboard upper half) is (-t_y, +t_x)? Let's check:
    # At theta = pi/2: x increases (+1, 0), outward is (0, +1) -> (-t_y, +t_x) = (0, +1)!
    # Wait: at theta = 0: x=-0.46, y increases (0, +1), (-t_y, +t_x) = (-1, 0) -> forward (-x)! Exact!
    n_out = np.column_stack((-t[:, 1], t[:, 0]))
    return p0 + delta * n_out


def keyhole_signed_distance(xy: np.ndarray, n_ref: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Signed outward distance from inner keyhole rim (< 0 inside hole, > 0 outside) & nearest u."""
    th = np.linspace(0.0, 2.0 * np.pi, n_ref, endpoint=False)
    loop = keyhole_inner_curve(th)
    dp = np.roll(loop, -1, axis=0) - np.roll(loop, 1, axis=0)
    t = dp / np.maximum(np.linalg.norm(dp, axis=1, keepdims=True), 1e-12)
    n_out = np.column_stack((-t[:, 1], t[:, 0]))
    diff = xy[:, None, :] - loop[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    idx = np.argmin(d2, axis=1)
    vec = xy - loop[idx]
    dist = np.sqrt(d2[np.arange(len(xy)), idx])
    sgn = np.where(np.sum(vec * n_out[idx], axis=1) >= 0.0, 1.0, -1.0)
    u_near = -np.cos(th[idx])
    return sgn * dist, u_near


def smoothstep(t: np.ndarray) -> np.ndarray:
    c = np.clip(t, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


FIT_STATIONS = 441
FIT_HALF_RING = 129
BEAM_AT_FOUR_INCHES_M = .500456  # Finished Kayak 10.9 design control.


def _station_grid(length, count):
    u = np.linspace(-1., 1., count)
    return .5 * length * (.02 * u + .98 * np.sin(.5 * np.pi * u))


class SculptedKayakSurface:
    """Precomputed tip-clustered parametric lookup for hull, sheer seam, and sculpted deck."""

    def __init__(self, hull_json_path: Path, n_stations: int = FIT_STATIONS, n_hull_half: int = FIT_HALF_RING) -> None:
        self.hull = Hull.model_validate_json(hull_json_path.read_text())
        self.geom = self.hull.to_geometry()
        self.L = self.hull.length_m
        # Cosine tip clustering (~0.28 mm first step at tips) to resolve 3.8 mm stem bullnose & 24 mm nose fillet smoothly
        if min(n_stations, n_hull_half) < 2:
            raise ValueError("At least two stations and two half-ring points are required")
        output_stations, output_half = n_stations, n_hull_half
        n_stations, n_hull_half = FIT_STATIONS, FIT_HALF_RING
        self.xs = _station_grid(self.L, n_stations)
        self.xs[0] = -0.5 * self.L
        self.xs[-1] = +0.5 * self.L
        self.n_hull_half = output_half

        seam_y = np.zeros(n_stations)
        seam_z = np.zeros(n_stations)
        keel_z = np.zeros(n_stations)
        deck_cl_z = np.zeros(n_stations)
        half_rings: list[np.ndarray] = []  # keel to starboard seam
        deck_r_grid = np.linspace(0.0, 1.0, 65)  # r = |y| / max(seam_y, 1e-5)
        deck_z_table = np.zeros((n_stations, len(deck_r_grid)))

        for i, x in enumerate(self.xs):
            xc = float(np.clip(x, -0.5 * self.L + 1e-5, 0.5 * self.L - 1e-5))
            lower, upper = self.geom.half_section_outline(xc)
            pts_up, mid = self.geom._upper_outline(xc)
            idx_mid = int(np.argmin(np.linalg.norm(pts_up - mid, axis=1)))

            # Monotonic circular-cap leading-edge stem bullnose (y_cap=3.8 mm bow / 2.4 mm stern; zero aft bulge!)
            d_end = max(0.5 * self.L - abs(float(x)), 0.0)
            y_cap = 0.0038 if x < 0.0 else 0.0024
            fade_b = float(1.0 - smoothstep(np.array((d_end - 0.080) / 0.140)))
            y_bull = y_cap * float(np.sqrt(np.tanh(d_end / (0.5 * y_cap)))) * fade_b
            z_lo, z_hi = float(lower[0, 1]), float(pts_up[-1, 1])
            span_z = max(z_hi - z_lo, 1e-4)

            # Starboard keel -> waterline -> shoulder mold seam
            stbd_low = np.vstack((lower, pts_up[1 : idx_mid + 1]))
            if y_bull > 0.0 and 0 < i < n_stations - 1:
                tz = np.clip((stbd_low[:, 1] - z_lo) / span_z, 0.0, 1.0)
                stbd_low[:, 0] = np.hypot(stbd_low[:, 0], y_bull * (np.sin(np.pi * tz) ** 0.55))
            stbd_low[:, 1] += Z_KEEL_SHIFT
            seg = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(stbd_low, axis=0), axis=1))))
            s_u = np.linspace(0.0, float(seg[-1]), n_hull_half)
            y_h = np.interp(s_u, seg, stbd_low[:, 0])
            z_h = np.interp(s_u, seg, stbd_low[:, 1])
            if i == 0 or i == n_stations - 1:
                y_h[:] = 0.0
            else:
                y_h[0] = 0.0

            seam_y[i] = float(y_h[-1])
            seam_z[i] = float(z_h[-1])
            keel_z[i] = float(z_h[0])

            # Starboard shoulder seam -> deck centerline
            deck_half = pts_up[idx_mid:].copy()
            if y_bull > 0.0 and 0 < i < n_stations - 1:
                tz_d = np.clip((deck_half[:, 1] - z_lo) / span_z, 0.0, 1.0)
                deck_half[:, 0] = np.hypot(deck_half[:, 0], y_bull * (np.sin(np.pi * tz_d) ** 0.55))
                deck_half[-1, 0] = 0.0
            deck_half[:, 1] += Z_KEEL_SHIFT
            if i == 0 or i == n_stations - 1:
                deck_half[:, 0] = 0.0
                z_d = np.linspace(float(deck_half[-1, 1]), float(seam_z[i]), len(deck_r_grid))
                deck_cl_z[i] = float(deck_half[-1, 1])
                deck_z_table[i] = z_d
            else:
                y_d = np.minimum.accumulate(deck_half[:, 0])[::-1]  # 0 -> seam_y
                z_d = deck_half[:, 1][::-1]                         # cl -> seam_z
                deck_cl_z[i] = float(z_d[0])
                ym = max(float(y_d[-1]), 1e-5)
                deck_z_table[i] = np.interp(deck_r_grid * ym, y_d, z_d)

            half_rings.append(np.column_stack((y_h, z_h)))

        half = fit_half_surface(self.xs, np.asarray(half_rings), .1016, BEAM_AT_FOUR_INCHES_M)
        # Tessellation samples the same design net at every output resolution.
        if output_stations != FIT_STATIONS:
            target = _station_grid(self.L, output_stations)
            seam_y, seam_z, keel_z, deck_cl_z = (
                np.interp(target, self.xs, v) for v in (seam_y, seam_z, keel_z, deck_cl_z)
            )
            deck_z_table = interp1d(self.xs, deck_z_table, axis=0)(target)
            half = interp1d(self.xs, half, axis=0)(target)
            self.xs = target
        if output_half != FIT_HALF_RING:
            half = interp1d(np.linspace(0, 1, FIT_HALF_RING), half, axis=1)(np.linspace(0, 1, output_half))

        self.seam_y = seam_y
        self.seam_z = seam_z
        self.keel_z = keel_z
        self.deck_cl_z = deck_cl_z
        self.deck_r_grid = deck_r_grid
        self.deck_z_table = deck_z_table
        port = half[:, :0:-1] * [-1, 1]
        self.hull_rings = np.concatenate((port, half), axis=1)

    def seam_at(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xa = np.asarray(x, dtype=float)
        return np.interp(xa, self.xs, self.seam_y), np.interp(xa, self.xs, self.seam_z)

    def keel_at(self, x: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), self.xs, self.keel_z)

    def base_deck_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        sy = np.maximum(np.interp(xa, self.xs, self.seam_y), 1e-5)
        r = np.clip(np.abs(ya) / sy, 0.0, 1.0)
        pos = np.clip(np.interp(xa, self.xs, np.arange(len(self.xs), dtype=float)), 0.0, len(self.xs) - 1 - 1e-9)
        i0 = pos.astype(int)
        fx = pos - i0
        rpos = np.clip(r * (len(self.deck_r_grid) - 1), 0.0, len(self.deck_r_grid) - 1 - 1e-9)
        j0 = rpos.astype(int)
        fr = rpos - j0
        z00 = self.deck_z_table[i0, j0]
        z01 = self.deck_z_table[i0, j0 + 1]
        z10 = self.deck_z_table[i0 + 1, j0]
        z11 = self.deck_z_table[i0 + 1, j0 + 1]
        return (1.0 - fx) * ((1.0 - fr) * z00 + fr * z01) + fx * ((1.0 - fr) * z10 + fr * z11)

    def sculpted_deck_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Deck z before end_shape_xyz: base crown + wing-knee center arch - aft coaming well - hatch gutters."""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        sy = np.maximum(np.interp(xa, self.xs, self.seam_y), 1e-5)
        edge_fade = smoothstep((1.0 - np.abs(ya) / sy) / 0.14)
        z = self.base_deck_z(xa, ya)

        # 1. Subtle centerline wing-knee arch (+12 mm at x=-0.38 m, |y|<13 cm; zero change to side catch!)
        wx = np.exp(-(((xa + 0.38) / 0.24) ** 2))
        wy = (1.0 - smoothstep(np.abs(ya) / 0.145))
        z = z + 0.012 * wx * wy * edge_fade

        # 2. Smooth recessed aft coaming well (drops aft rim so skirt lip sits flush with 14.5 cm rear deck)
        xy = np.column_stack((xa.ravel(), ya.ravel()))
        d_kh, _ = keyhole_signed_distance(xy)
        d_kh = d_kh.reshape(xa.shape)
        aft_ramp = smoothstep((xa + 0.04) / 0.34) * (1.0 - smoothstep((xa - 0.41) / 0.08))
        well_rad = 1.0 - smoothstep((d_kh - 0.020) / 0.036)
        z = z - 0.021 * aft_ramp * well_rad * edge_fade

        # 3. Three recessed flush hatch gutters (flat pocket floor + raised inner sealing lip)
        for hs in HATCHES:
            rho = np.hypot((xa - hs.x0) / hs.rx_lid, (ya - hs.y0) / hs.ry_lid)
            outer_w = 1.0 - smoothstep((rho - 1.09) / 0.14)
            lip_rise = 0.006 * (1.0 - smoothstep((rho - 0.88) / 0.08))
            z = z - (hs.depth * outer_w - lip_rise) * edge_fade
        return z

    def end_shape_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Shared rounded bow/stern nose fillets + smooth forefoot + stem rake + stern rudder sweep."""
        out = np.asarray(xyz, dtype=float).copy()
        x, y, z = out[:, 0].copy(), out[:, 1].copy(), out[:, 2].copy()
        half_L = 0.5 * self.L
        zk = self.keel_at(x)
        z_top = np.interp(x, self.xs, self.deck_cl_z)

        # 1. Upper bow nose round-over (Rx=26 mm, Rz=23 mm smooth tangent bullnose, Epic 18X / S18R style)
        d_bow = x + half_L
        rx_n, rz_n = 0.026, 0.023
        m_bn = (d_bow < rx_n) & (z > z_top - rz_n)
        if np.any(m_bn):
            ux = (rx_n - d_bow[m_bn]) / rx_n
            uz = np.clip((z[m_bn] - (z_top[m_bn] - rz_n)) / rz_n, 0.0, 1.5)
            sc = np.maximum(ux, uz) / np.maximum(np.hypot(ux, uz), 1e-12)
            d_bow[m_bn] = rx_n * (1.0 - sc * ux)
            z[m_bn] = (z_top[m_bn] - rz_n) + rz_n * (sc * uz)

        # The product vanishes quadratically in the scale factor at each join.
        # On the stem or keel one coordinate is 1, preserving the elliptical profile.
        rx_f, rz_f = 0.135, 0.056
        m_bf = (d_bow < rx_f) & (z < zk + rz_f)
        if np.any(m_bf):
            fx = (rx_f - d_bow[m_bf]) / rx_f
            fz = np.clip(((zk[m_bf] + rz_f) - z[m_bf]) / rz_f, 0.0, 1.5)
            scf = 1.0 / np.hypot(1.0, fx * fz)
            d_bow[m_bf] = rx_f * (1.0 - scf * fx)
            z[m_bf] = (zk[m_bf] + rz_f) - rz_f * (scf * fz)

        # 3. Gentle ~8 deg wave-piercing stem rake anchored at rounded nose apex (keeps x_min == -2.7350 m!)
        z_apex = z_top - rz_n
        tau_rake = np.clip((z - zk) / np.maximum(z_apex - zk, 1e-4), 0.0, 1.0)
        u_rake = 1.0 - smoothstep(d_bow / 0.25)
        x_new = (-half_L + d_bow) + 0.030 * (u_rake ** 1.35) * ((1.0 - tau_rake) ** 1.15)

        # 4. Stern upper nose round-over (11 mm tangent fillet) + rudder keel sweep (keeps x_max == +2.7350 m!)
        d_st = half_L - x_new
        rs_x, rs_z = 0.011, 0.011
        m_sn = (d_st < rs_x) & (z > z_top - rs_z)
        if np.any(m_sn):
            sx = (rs_x - d_st[m_sn]) / rs_x
            sz = np.clip((z[m_sn] - (z_top[m_sn] - rs_z)) / rs_z, 0.0, 1.5)
            scs = np.maximum(sx, sz) / np.maximum(np.hypot(sx, sz), 1e-12)
            x_new[m_sn] = half_L - rs_x * (1.0 - scs * sx)
            z[m_sn] = (z_top[m_sn] - rs_z) + rs_z * (scs * sz)

        tau_st = np.clip((z - zk) / np.maximum(z_top - zk, 1e-4), 0.0, 1.0)
        u_st = np.clip((x_new - 2.50) / (half_L - 2.50), 0.0, 1.0)
        dz_target = 0.0930 - float(self.keel_z[-1])
        z = z + dz_target * (u_st ** 1.80) * (1.0 - tau_st)

        out[:, 0] = x_new
        out[:, 2] = z
        return out

    def mold_station_rings_mm(self) -> np.ndarray:
        """Sample mold cavities at physical X after rounding and raking the bow."""
        x = np.broadcast_to(self.xs[:, None, None], self.hull_rings.shape[:2] + (1,))
        rings = self.end_shape_xyz(np.concatenate((x, self.hull_rings), axis=2).reshape(-1, 3))
        rings = rings.reshape(len(self.xs), -1, 3) * 1000.0

        # Bow rings bend longitudinally during fairing. The tooling loft uses
        # flat stations, so intersect those planes with the finished surface.
        shell = trimesh.util.concatenate((self.build_hull_shell(), self.build_deck_shell(cut_openings=False)))
        shell.merge_vertices()
        end_x = self.xs[0] + 0.250
        bow = shell.submesh([shell.triangles[:, :, 0].min(axis=1) <= end_x], append=True)
        for i in np.flatnonzero(self.xs < end_x):
            station = max(self.xs[i], self.xs[0] + 0.00005)
            sec = bow.section([1, 0, 0], [station, 0, 0])
            loop = max(sec.discrete, key=len)[:, 1:]
            if np.linalg.norm(loop[0] - loop[-1]) < 1e-12:
                loop = loop[:-1]
            area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
            if area < 0.0:
                loop = loop[::-1]
            aug: list[np.ndarray] = []
            for k in range(len(loop)):
                p0, p1 = loop[k], loop[(k + 1) % len(loop)]
                aug.append(p0)
                if (p0[0] < -1e-7 and p1[0] > 1e-7) or (p0[0] > 1e-7 and p1[0] < -1e-7):
                    zc = float(p0[1] - p0[0] * (p1[1] - p0[1]) / (p1[0] - p0[0]))
                    aug.append(np.array([0.0, zc]))
            loop = np.asarray(aug, dtype=float)
            cl = np.flatnonzero(np.abs(loop[:, 0]) <= 1e-7)
            rolled = np.roll(loop, -int(cl[np.argmin(loop[cl, 1])]), axis=0)
            i_top = int(np.flatnonzero(np.abs(rolled[1:, 0]) <= 1e-7)[0]) + 1
            side = rolled[: i_top + 1].copy()
            side[0, 0] = 0.0
            z_seam = float(self.seam_z[i])
            # Where rake moves the stem above the mold parting line, retain
            # its lowest point so the tooling can close the cavity there.
            if side[0, 1] >= z_seam:
                side = side[:1]
            elif side[-1, 1] > z_seam:
                k = int(np.flatnonzero(side[:, 1] >= z_seam)[0])
                alpha = (z_seam - side[k - 1, 1]) / (side[k, 1] - side[k - 1, 1])
                side = np.vstack((side[:k], [side[k - 1] + alpha * (side[k] - side[k - 1])]))
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))]
            t = np.linspace(0.0, arc[-1], self.n_hull_half)
            starboard = np.column_stack([np.interp(t, arc, side[:, j]) for j in (0, 1)])
            port = starboard[:0:-1] * [-1, 1]
            rings[i, :, 1:] = 1000.0 * np.vstack((port, starboard))
            rings[i, :, 0] = 1000.0 * station
        return rings

    @staticmethod
    def mold_stbd_section_mm(ring_mm: np.ndarray, z_part_mm: float) -> np.ndarray:
        """Extract ordered keel-to-parting-line starboard cavity curve (mm) without z-sorting."""
        stbd = ring_mm[len(ring_mm) // 2 :, 1:].copy()
        stbd[0, 0] = 0.0
        if stbd[0, 1] >= z_part_mm:
            hs = np.array([[0.0, z_part_mm - 1.0], [max(float(stbd[0, 0]), 0.5), z_part_mm]])
        elif stbd[-1, 1] <= z_part_mm:
            hs = np.vstack((stbd, [[float(stbd[-1, 0]), z_part_mm]]))
        else:
            k = int(np.flatnonzero(stbd[:, 1] >= z_part_mm)[0])
            dz = float(stbd[k, 1] - stbd[k - 1, 1])
            alpha = (z_part_mm - float(stbd[k - 1, 1])) / dz if dz > 1e-9 else 0.0
            pt = stbd[k - 1] + alpha * (stbd[k] - stbd[k - 1])
            pt[1] = z_part_mm
            hs = np.vstack((stbd[:k], [pt]))
        hs[:, 0] = np.maximum.accumulate(np.maximum(hs[:, 0], 0.0))
        return hs

    def build_hull_shell(self) -> trimesh.Trimesh:
        """Build outer hull + topside shell below the sheer mold seam."""
        n_x, m_ring, _ = self.hull_rings.shape
        verts = np.zeros((n_x, m_ring, 3))
        verts[:, :, 0] = self.xs[:, None]
        verts[:, :, 1:] = self.hull_rings
        flat_v = self.end_shape_xyz(verts.reshape(-1, 3))

        faces: list[list[int]] = []
        for i in range(n_x - 1):
            s = i * m_ring
            t = (i + 1) * m_ring
            for j in range(m_ring - 1):
                c1, c2, n1, n2 = s + j, s + j + 1, t + j, t + j + 1
                faces.append([c1, c2, n1])
                faces.append([c2, n2, n1])
        mesh = trimesh.Trimesh(vertices=flat_v, faces=np.asarray(faces, dtype=int), process=True)
        return mesh

    def _structured_deck_patch(self, i_start: int, i_end: int, u_ring: np.ndarray, sculpted: bool) -> trimesh.Trimesh:
        """Structured quad-lofted deck surface across stations [i_start..i_end] (exact bow/stern tip closure)."""
        idx_slice = np.arange(i_start, i_end + 1)
        m_dk = len(u_ring)
        r_abs = np.abs(u_ring)
        rpos = np.clip(r_abs * (len(self.deck_r_grid) - 1), 0.0, len(self.deck_r_grid) - 1 - 1e-9)
        j0 = rpos.astype(int)
        fr = rpos - j0

        verts = np.zeros((len(idx_slice), m_dk, 3))
        for k, ii in enumerate(idx_slice):
            xv = float(self.xs[ii])
            yv = u_ring * float(self.seam_y[ii])
            if sculpted and 1 < ii < len(self.xs) - 2:
                zv = self.sculpted_deck_z(np.full(m_dk, xv), yv)
            else:
                zv = (1.0 - fr) * self.deck_z_table[ii, j0] + fr * self.deck_z_table[ii, j0 + 1]
            zv[0] = float(self.seam_z[ii])
            zv[-1] = float(self.seam_z[ii])
            verts[k, :, 0] = xv
            verts[k, :, 1] = yv
            verts[k, :, 2] = zv
        flat_v = self.end_shape_xyz(verts.reshape(-1, 3))
        faces: list[list[int]] = []
        for k in range(len(idx_slice) - 1):
            s, t = k * m_dk, (k + 1) * m_dk
            for j in range(m_dk - 1):
                c1, c2, n1, n2 = s + j, s + j + 1, t + j, t + j + 1
                faces.append([c1, n2, c2])
                faces.append([c1, n1, n2])
        return trimesh.Trimesh(vertices=flat_v, faces=np.asarray(faces, dtype=int), process=True)

    def build_deck_shell(self, cut_openings: bool = True) -> trimesh.Trimesh:
        """Structured bow/stern caps + conformal 2D Delaunay mid-deck with keyhole & hatch recesses."""
        u_ring = np.linspace(-1.0, +1.0, 65)
        if not cut_openings:
            return self._structured_deck_patch(0, len(self.xs) - 1, u_ring, sculpted=False)

        i_fwd = int(np.argmin(np.abs(self.xs - (-1.85))))
        i_aft = len(self.xs) - 1 - i_fwd  # x in [-1.85, +1.85] m contains all cutouts & recesses
        bow_cap = self._structured_deck_patch(0, i_fwd, u_ring, sculpted=True)
        stern_cap = self._structured_deck_patch(i_aft, len(self.xs) - 1, u_ring, sculpted=True)

        # Boundary of mid-body Delaunay patch: exact transverse rings at i_fwd & i_aft + side seams
        fwd_ring_xy = np.column_stack((np.full(len(u_ring), self.xs[i_fwd]), u_ring * self.seam_y[i_fwd]))
        aft_ring_xy = np.column_stack((np.full(len(u_ring), self.xs[i_aft]), u_ring * self.seam_y[i_aft]))
        side_pts: list[list[float]] = []
        for ii in range(i_fwd + 1, i_aft):
            sy = float(self.seam_y[ii])
            side_pts.append([float(self.xs[ii]), -sy])
            side_pts.append([float(self.xs[ii]), +sy])
        seam_boundary = np.vstack((fwd_ring_xy, aft_ring_xy, np.asarray(side_pts)))

        ring_pts: list[np.ndarray] = []
        th_kh = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        for delta in (0.0040, 0.0160, 0.0280):
            rc = keyhole_offset_curve(th_kh, delta)
            if delta > 0.005:
                sy_lim = np.interp(rc[:, 0], self.xs, self.seam_y) - 0.0055
                rc[:, 1] = np.clip(rc[:, 1], -sy_lim, +sy_lim)
            ring_pts.append(rc)
        th_h = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        for hs in HATCHES:
            for rho in (0.82, 0.94, 1.09, 1.22):
                ring_pts.append(
                    np.column_stack((hs.x0 + rho * hs.rx_lid * np.cos(th_h), hs.y0 + rho * hs.ry_lid * np.sin(th_h)))
                )

        gx = np.linspace(float(self.xs[i_fwd]) + 0.012, float(self.xs[i_aft]) - 0.012, 260)
        grid_list: list[np.ndarray] = []
        for xv in gx:
            sy = float(np.interp(xv, self.xs, self.seam_y))
            ny = max(1, int(np.floor((sy - 0.006) / 0.012)))
            ys = np.linspace(-(sy - 0.007), +(sy - 0.007), 2 * ny + 1)
            grid_list.append(np.column_stack((np.full(len(ys), xv), ys)))
        cand = np.vstack(grid_list)

        fixed = np.vstack([seam_boundary] + ring_pts)
        from scipy.spatial import cKDTree
        keep = cKDTree(fixed).query(cand, k=1)[0] > 0.0055
        d_kh, _ = keyhole_signed_distance(cand)
        keep &= d_kh > 0.0095
        for hs in HATCHES:
            rho = np.hypot((cand[:, 0] - hs.x0) / hs.rx_lid, (cand[:, 1] - hs.y0) / hs.ry_lid)
            keep &= rho > 0.86

        pts_xy = np.vstack((fixed, cand[keep]))
        simplices = Delaunay(pts_xy).simplices
        cent = pts_xy[simplices].mean(axis=1)
        sy_c = np.interp(cent[:, 0], self.xs, self.seam_y)
        valid = (cent[:, 0] > self.xs[i_fwd] + 1e-6) & (cent[:, 0] < self.xs[i_aft] - 1e-6) & (np.abs(cent[:, 1]) < sy_c - 1e-5)
        d_c, _ = keyhole_signed_distance(cent)
        valid &= d_c > 0.00405
        for hs in HATCHES:
            rho_c = np.hypot((cent[:, 0] - hs.x0) / hs.rx_lid, (cent[:, 1] - hs.y0) / hs.ry_lid)
            valid &= rho_c > 0.8205

        faces = simplices[valid].copy()
        z_vals = self.sculpted_deck_z(pts_xy[:, 0], pts_xy[:, 1])
        # Exact seam height on port/starboard sheer boundary points
        z_vals[0] = self.seam_z[i_fwd]
        z_vals[len(u_ring) - 1] = self.seam_z[i_fwd]
        z_vals[len(u_ring)] = self.seam_z[i_aft]
        z_vals[2 * len(u_ring) - 1] = self.seam_z[i_aft]
        n_rings = 2 * len(u_ring)
        n_side = len(seam_boundary) - n_rings
        z_vals[n_rings : n_rings + n_side] = np.interp(pts_xy[n_rings : n_rings + n_side, 0], self.xs, self.seam_z)

        xyz = self.end_shape_xyz(np.column_stack((pts_xy[:, 0], pts_xy[:, 1], z_vals)))
        v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
        flip = np.cross(v1 - v0, v2 - v0)[:, 2] < 0.0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        mid_patch = trimesh.Trimesh(vertices=xyz, faces=faces, process=True)

        full_deck = trimesh.util.concatenate([bow_cap, mid_patch, stern_cap])
        full_deck.merge_vertices(merge_tex=True, merge_norm=True)
        return full_deck
