"""Regression tests for the self-contained Kayak 10.9 naval-architecture, CFD, BEM & plate toolchain."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest

import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "cad") not in sys.path:
    sys.path.insert(0, str(ROOT / "cad"))

from deck_geometry import SculptedKayakSurface  # noqa: E402
from hydro import LB, RHO_SEAWATER, STAGE9_PRE_FINISH_JSON, float_at, load_epic_18x_arrays  # noqa: E402


class NavalArchPipelineTests(unittest.TestCase):
    def test_no_hardcoded_home_or_cache_paths_in_scripts(self) -> None:
        for folder in ("cad", "site", "tests"):
            for path in sorted((ROOT / folder).glob("*.py")):
                if path.name == Path(__file__).name:
                    continue
                text = path.read_text()
                self.assertNotIn("/home/", text, f"Hardcoded /home/ path in {path.name}")
                self.assertNotIn(".cache/kayak-gen", text, f"Hardcoded .cache path in {path.name}")

    def test_checked_in_pipeline_modules_import_and_benchmarks_float(self) -> None:
        for mod_name in ("hydro", "openfoam_dicts", "build_specs_and_ga", "build_plates", "package_release"):
            importlib.import_module(mod_name)
        self.assertTrue(STAGE9_PRE_FINISH_JSON.is_file())
        ve, fe = load_epic_18x_arrays(bow_at_pos_x=False)
        tm = trimesh.Trimesh(ve, fe, process=True)
        self.assertTrue(tm.is_watertight)
        res = float_at(ve, fe, 225.0 * LB, RHO_SEAWATER, [0.0, 0.0, 0.25], free_trim=False)
        self.assertAlmostEqual(float(res["vol"] * RHO_SEAWATER), 225.0 * LB, places=2)

    def test_cfd_and_seakeeping_datasets_match_finished_outer_shell_and_openfoam_2606(self) -> None:
        surf = SculptedKayakSurface(ROOT / "hull" / "kayak-10-9.hull.json")
        closed = trimesh.util.concatenate((surf.build_hull_shell(), surf.build_deck_shell(cut_openings=False)))
        closed.merge_vertices(merge_tex=True, merge_norm=True)
        expected_sha = hashlib.sha256(closed.export(file_type="stl")).hexdigest()

        cfd = json.loads((ROOT / "data" / "openfoam-cfd-metrics.json").read_text())
        seak = json.loads((ROOT / "data" / "capytaine-seakeeping.json").read_text())
        wind = json.loads((ROOT / "data" / "windage-and-stiffener.json").read_text())

        self.assertEqual(cfd["metadata"]["kayak_10_9_outer_stl_sha256"], expected_sha)
        self.assertIn("2606", cfd["metadata"]["openfoam_using"])
        self.assertIn("kayak_10_9_5.5kn", cfd)
        self.assertIn("kayak_10_9_6.2kn", cfd)
        self.assertLess(cfd["kayak_10_9_6.2kn"]["drag_total_N"], cfd["epic_18x_6.2kn"]["drag_total_N"])

        self.assertEqual(seak["metadata"]["kayak_10_9_outer_stl_sha256"], expected_sha)
        self.assertEqual(wind["metadata"]["kayak_10_9_outer_stl_sha256"], expected_sha)


if __name__ == "__main__":
    unittest.main()
