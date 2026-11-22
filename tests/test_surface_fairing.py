"""Read the final fitted surface through the same sections used for finished CAD."""

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad"))
from deck_geometry import SculptedKayakSurface
from inspect_fairness import cut_line, turns, section_curve


class SurfaceFairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ROOT / "hull/kayak-10-9.hull.json"
        cls.surface = SculptedKayakSurface(cls.spec)
        cls.shell = trimesh.util.concatenate(
            (cls.surface.build_hull_shell(), cls.surface.build_deck_shell(cut_openings=False))
        )
        cls.shell.merge_vertices()

    def test_forward_waterlines_through_span_transition(self):
        x = np.arange(0.645, 1.075, 0.002) - self.surface.L / 2
        for z in (0.050, 0.060, 0.070, 0.080, 0.090, 0.1016):
            with self.subTest(height=z):
                y = cut_line(self.shell, x, [0, 1], z)
                self.assertTrue(np.isfinite(y).all())
                self.assertLess(np.max(turns(y, 0.002)[25:-25]), 0.02)

    def test_finished_bilge_keeps_transverse_fairing(self):
        for x in (-2.10, -1.65, -1.35, -0.95):
            arc, q, k = section_curve(self.shell, x)
            lower = (arc < arc[-1] / 2) & (q[:, 1] < 0.12)
            change = (np.roll(k, -2) - np.roll(k, 2)) / (4 * arc[1])
            with self.subTest(x=x):
                self.assertLess(q[0, 1] - q[:, 1].min(), 1e-7)
                self.assertGreater(k[lower].min(), 0)
                self.assertLess(abs(change[lower]).max(), 210)

    def test_coarser_tessellation_samples_same_net(self):
        coarse = SculptedKayakSurface(self.spec, n_stations=221, n_hull_half=65)
        np.testing.assert_allclose(coarse.hull_rings, self.surface.hull_rings[::2, ::2], atol=1e-12)
