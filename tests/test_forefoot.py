"""Check the forefoot surface on both sides of the old diagonal crease."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad"))
from deck_geometry import SculptedKayakSurface


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
