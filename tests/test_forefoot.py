"""Check the forefoot surface on both sides of the old diagonal crease."""
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad"))
from deck_geometry import SculptedKayakSurface
import build_hull_mold_strongback_assy as full_mold
import build_stepped_cradle_strongback as stepped_mold


class ForefootTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surface = SculptedKayakSurface(ROOT / "hull/kayak-10-9.hull.json")

    def test_tangent_across_forefoot(self):
        surf = self.surface
        step = 1e-7
        for fraction in (0.15, 0.35, 0.55, 0.75, 0.90):
            x = -surf.L / 2 + 0.135 * fraction
            z = surf.keel_at(x) + 0.056 * fraction
            points = np.array([[x, 0.001, z + offset] for offset in (-step, 0, step)])
            left, right = np.diff(surf.end_shape_xyz(points), axis=0) / step
            with self.subTest(fraction=fraction):
                np.testing.assert_allclose(left, right, atol=1e-5, rtol=0)

    def test_keel_follows_original_elliptical_arc(self):
        surf = self.surface
        d = np.linspace(0, 0.135, 51)
        x = -surf.L / 2 + d
        keel = surf.keel_at(x)
        points = np.column_stack((x, np.zeros_like(x), keel))
        actual_z = surf.end_shape_xyz(points)[:, 2]
        expected_z = keel + 0.056 * (1 - 1 / np.hypot(1, 1 - d / 0.135))
        np.testing.assert_allclose(actual_z, expected_z, atol=1e-12)

    def test_mold_station_rings_follow_outer_shell_curve(self):
        surf = self.surface
        shell = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
        rings_mm = surf.mold_station_rings_mm()
        mid = surf.n_hull_half - 1
        for i in np.flatnonzero((surf.xs > -2.725) & (surf.xs < -2.485)):
            x_m = float(rings_mm[i, 0, 0]) / 1000.0
            section = shell.section([1, 0, 0], [x_m, 0, 0])
            curve_mm = LineString(max(section.discrete, key=len)[:, 1:] * 1000.0)
            stbd_y = rings_mm[i, mid:, 1]
            max_dist_mm = max(curve_mm.distance(Point(pt)) for pt in rings_mm[i, :, 1:])
            with self.subTest(station_mm=round(x_m * 1000.0, 2)):
                self.assertAlmostEqual(float(stbd_y[0]), 0.0, places=5)
                self.assertTrue(np.all(np.diff(stbd_y[:12]) >= -1e-6))
                self.assertLess(max_dist_mm, 0.02)

    def test_forward_mold_cavities_match_outer_shell(self):
        surf = self.surface
        shell = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
        tbl_step = stepped_mold._precompute_tables(surf)
        tbl_full = full_mold._precompute_mold_tables(surf)
        for x_mm in (-2615.0, -2580.0, -2550.0, -2495.0, -2460.0):
            section = shell.section([1, 0, 0], [x_mm / 1000.0, 0, 0])
            curve_mm = LineString(max(section.discrete, key=len)[:, 1:] * 1000.0)
            cav_stbd, zm, _ = stepped_mold._hull_stbd_curve_at_x(tbl_step, x_mm)
            loop_full = full_mold._mold_loop_at_x_mm(tbl_full, x_mm)
            cav_full = loop_full[(np.abs(loop_full[:, 0]) <= float(cav_stbd[-1, 0]) - 0.5) & (loop_full[:, 1] > 40.0)]
            max_dev_step = max(curve_mm.distance(Point(pt)) for pt in cav_stbd[1:])
            max_dev_full = max(curve_mm.distance(Point(pt)) for pt in cav_full)
            with self.subTest(x_mm=x_mm, zm_mm=round(zm, 2)):
                self.assertLess(max_dev_step, 0.15)
                self.assertLess(max_dev_full, 0.15)

    def test_forward_bilge_and_keel_are_faired(self):
        geom = self.surface.geom
        for x_m in (-2.10, -1.65, -1.35, -0.95):
            lo, up = geom.half_section_outline(x_m)
            comb = np.vstack((lo, up[1:]))
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(comb, axis=0), axis=1))]
            su = np.linspace(0.0, float(s[-1]), 64)
            yu, zu = np.interp(su, s, comb[:, 0]), np.interp(su, s, comb[:, 1])
            sm = np.r_[-su[:0:-1], su]
            ym, zm = np.r_[-yu[:0:-1], yu], np.r_[zu[:0:-1], zu]
            dy, dz = np.gradient(ym, sm), np.gradient(zm, sm)
            ddy, ddz = np.gradient(dy, sm), np.gradient(dz, sm)
            kap = (dy * ddz - dz * ddy) / np.maximum((dy * dy + dz * dz) ** 1.5, 1e-12)
            k_stbd = kap[len(su) - 1 : len(su) - 1 + 36]
            dk_cm = float(np.abs(np.gradient(k_stbd, su[:36])).max() * 0.01)
            with self.subTest(x_m=x_m):
                self.assertLessEqual(float(lo[0, 1] - lo[:, 1].min()), 1e-9)
                self.assertGreater(float(k_stbd.min()), 0.4)
                self.assertLess(dk_cm, 2.5)
