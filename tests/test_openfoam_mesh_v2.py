"""Unit tests for Reference Mesh v2 OpenFOAM dicts and moveDynamicMesh helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))
from openfoam_dicts import BLOCK_MESH_DICT, SNAPPY_DICT  # noqa: E402
from openfoam_mesh_v2 import (  # noqa: E402
    build_stl_vertex_displacement,
    extract_patch_mesh_points,
    write_mesh_dicts_v2,
    write_point_displacement_field,
)


def test_write_mesh_dicts_v2_preserves_baseline_dicts(tmp_path: Path) -> None:
    write_mesh_dicts_v2(tmp_path, np_ranks=4)
    blk = (tmp_path / "system/blockMeshDict").read_text(encoding="utf-8")
    snp = (tmp_path / "system/snappyHexMeshDict").read_text(encoding="utf-8")
    mqd = (tmp_path / "system/meshQualityDict").read_text(encoding="utf-8")
    assert "addLayers       true;" in snp
    assert "relativeSizes false;" in snp
    assert "firstLayerThickness 0.00105;" in snp
    assert "layers { hull { nSurfaceLayers 5; } }" in snp
    assert "maxBoundarySkewness 3.8;" in mqd
    assert blk != BLOCK_MESH_DICT
    assert "addLayers       false;" in SNAPPY_DICT


def test_extract_patch_mesh_points_preserves_first_appearance_order(tmp_path: Path) -> None:
    pm = tmp_path / "constant/polyMesh"
    pm.mkdir(parents=True)
    pts_lines = "\n".join(f"({i}.0 0.0 0.0)" for i in range(12))
    (pm / "points").write_text(f"// *\n12\n(\n{pts_lines}\n)\n\n", encoding="utf-8")
    (pm / "faces").write_text(
        "// *\n3\n(\n3(0 1 2)\n4(10 3 7 3)\n3(7 2 10)\n)\n", encoding="utf-8"
    )
    (pm / "boundary").write_text(
        "2\n(\n inlet { type patch; nFaces 1; startFace 0; }\n"
        " hull { type wall; nFaces 2; startFace 1; }\n)\n",
        encoding="utf-8",
    )
    hull_pts, ids = extract_patch_mesh_points(pm, "hull")
    assert ids == [10, 3, 7, 2]
    assert np.allclose(hull_pts[:, 0], [10.0, 3.0, 7.0, 2.0])


def test_seam_blended_normal_displacement_tracks_keel_and_minimizes_wl_slide(tmp_path: Path) -> None:
    # Half-hull strip: keel at y=0, z=-0.04; chine/wl at y=0.05, z=0.0; flare top at y=0.05, z=+0.04
    v_base = np.array(
        [
            [0.0, 0.0, -0.04],
            [1.0, 0.0, -0.04],
            [0.0, 0.05, 0.0],
            [1.0, 0.05, 0.0],
            [0.0, 0.05, 0.04],
            [1.0, 0.05, 0.04],
        ]
    )
    f_base = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    # Target drops keel by -9.5 mm while slightly shifting WL node parametrically (-4 mm z, +1 mm y)
    v_target = v_base.copy()
    v_target[0:2, 2] -= 0.0095
    v_target[2:4, 2] -= 0.0040
    v_target[2:4, 1] += 0.0010

    query = np.array([[0.5, 0.0, -0.04], [0.5, 0.05, 0.0]])
    d_norm = build_stl_vertex_displacement(v_base, f_base, v_target, query, mode="normal")
    d_param = build_stl_vertex_displacement(v_base, f_base, v_target, query, mode="parametric")
    assert d_norm[0, 1] == 0.0
    assert abs(d_norm[0, 2] - (-0.0095)) < 1e-4
    assert abs(d_norm[1, 2]) < abs(d_param[1, 2])

    out_pd = tmp_path / "0/pointDisplacement"
    write_point_displacement_field(out_pd, d_norm)
    assert f"List<vector> {len(d_norm)}" in out_pd.read_text(encoding="utf-8")


def test_morph_polymesh_points_rigid_core_and_midplane_symmetry(tmp_path: Path) -> None:
    from openfoam_mesh_v2 import morph_polymesh_points

    pm = tmp_path / "constant/polyMesh"
    pm.mkdir(parents=True)
    # hull points at idx 0, 1; nearby prism point 1mm below at idx 2; far outer point at idx 3
    pts = np.array(
        [
            [0.0, 0.0, -0.035],
            [0.02, 0.01, -0.030],
            [0.0, 0.0, -0.036],
            [0.0, 0.80, -0.035],
        ]
    )
    (pm / "points").write_text(
        "// *\n4\n(\n" + "\n".join(f"({x} {y} {z})" for x, y, z in pts) + "\n)\n\n",
        encoding="utf-8",
    )
    hull_disp = np.array([[0.0, 0.0, -0.0095], [0.0, 0.0005, -0.0095]])
    morphed = morph_polymesh_points(
        pm, [0, 1], hull_disp, out_points_path=pm / "points_morphed"
    )
    assert np.allclose(morphed[0], [0.0, 0.0, -0.0445])
    assert morphed[2, 1] == 0.0  # midPlane y=0 preserved
    assert morphed[2, 2] < morphed[0, 2]  # 1mm keel prism cell did NOT invert!
    assert np.allclose(morphed[3], pts[3])  # beyond r_outer=0.45 is untouched
    assert (pm / "points_morphed").is_file()
