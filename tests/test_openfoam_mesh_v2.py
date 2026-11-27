"""Unit tests for Reference Mesh v2 OpenFOAM dicts and configurable case writer."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))
from openfoam_bin_forces import parse_force_bin_dat  # noqa: E402
from openfoam_dicts import BLOCK_MESH_DICT, SNAPPY_DICT  # noqa: E402
from openfoam_mesh_v2 import write_interfoam_case_files_v2, write_mesh_dicts_v2  # noqa: E402


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


def test_write_interfoam_case_files_v2_maxco_damping_and_potential(tmp_path: Path) -> None:
    cmds = write_interfoam_case_files_v2(
        tmp_path,
        u_mps=2.8294,
        cg=[0.04, 0.0, 0.16],
        end_steps=2000,
        np_ranks=6,
        max_co=50.0,
        max_alpha_co=5.0,
        damping_zones=True,
        calm_start="potential",
    )
    fv_sol = (tmp_path / "system/fvSolution").read_text(encoding="utf-8")
    ctrl = (tmp_path / "system/controlDict").read_text(encoding="utf-8")
    assert "maxCo 50;" in fv_sol
    assert "maxAlphaCo 5;" in fv_sol
    assert "potentialFlow" in fv_sol
    assert (tmp_path / "0.orig/p").is_file()
    assert (tmp_path / "0/U").is_file()
    assert (tmp_path / "0/alpha.water").is_file()
    assert "type fixedValue; value uniform 0;" in (tmp_path / "0.orig/p_rgh").read_text(encoding="utf-8")
    assert "type inletOutlet; inletValue uniform (2.829400 0 0);" in (tmp_path / "0.orig/U").read_text(encoding="utf-8")
    assert "outlet     { type zeroGradient; }" in (tmp_path / "0.orig/alpha.water").read_text(encoding="utf-8")
    assert "readFields" in ctrl and "fields (URef)" in ctrl
    fv_opt = (tmp_path / "constant/fvOptions").read_text(encoding="utf-8")
    assert "acousticDampingSource" in fv_opt
    assert "radius1         7.6;" in fv_opt and "radius2         10.56;" in fv_opt
    joined = " && ".join(cmds)
    assert "setExprFields -parallel -time 0 -field URef" in joined
    assert "potentialFoam -initialiseUBCs -parallel" in joined
    assert "useNamePrefix false; executeControl timeStep; executeInterval 1;" in ctrl
    assert "type                    binField;" in ctrl
    assert "binModel                singleDirectionUniformBin;" in ctrl
    assert "nBin        22;" in ctrl


def test_write_interfoam_case_files_v2_ramp_and_no_damping(tmp_path: Path) -> None:
    cmds = write_interfoam_case_files_v2(
        tmp_path,
        u_mps=3.1896,
        cg=[0.04, 0.0, 0.16],
        end_steps=1500,
        np_ranks=4,
        max_co=25.0,
        damping_zones=False,
        calm_start="ramp",
        ramp_steps=300,
    )
    assert not (tmp_path / "constant/fvOptions").exists()
    assert not (tmp_path / "0.orig/p").exists()
    u_txt = (tmp_path / "0.orig/U").read_text(encoding="utf-8")
    assert "uniformFixedValue" in u_txt
    assert "300 (3.189600 0 0)" in u_txt
    assert "maxCo 25;" in (tmp_path / "system/fvSolution").read_text(encoding="utf-8")
    assert all("potentialFoam" not in c for c in cmds)
    assert all("setExprFields" not in c for c in cmds)


def test_parse_force_bin_dat_v2606_format(tmp_path: Path) -> None:
    # Exact header/column structure emitted by OpenFOAM v2606 binField (singleDirectionUniformBin)
    row = (
        "10 "
        "1.0 0.0 0.0 0.0 0.0 0.0 0.9 0.0 0.0 0.1 0.0 0.0 "
        "2.0 0.0 0.0 0.0 0.0 0.0 0.2 0.0 0.0 1.8 0.0 0.0\n"
    )
    sample = (
        "# bins         : 2\n"
        "# start        : -2.7500000e+00\n"
        "# end          : 2.7500000e+00\n"
        "# delta        : 2.7500000e+00\n"
        "# direction    : (1.0000000e+00 0.0000000e+00 0.0000000e+00)\n"
        "# Time\ttotal_0_x total_0_y total_0_z\tinternal_0_x ...\n"
        + row
    )
    p = tmp_path / "forceBin.dat"
    p.write_text(sample, encoding="utf-8")
    out = parse_force_bin_dat(p, full_hull=True)
    assert out["steps"].tolist() == [10]
    assert out["bin_edges_x"].tolist() == [-2.75, 0.0, 2.75]
    assert out["total_x"].tolist() == [[2.0, 4.0]]
    assert out["normal_x"].tolist() == [[1.8, 0.4]]
    assert out["tangential_x"].tolist() == [[0.2, 3.6]]
