"""OpenFOAM-v2606 Reference Mesh v2 dictionaries and configurable interFoam case writer."""

from __future__ import annotations

from pathlib import Path
import shutil

from openfoam_dicts import CTRL_MESH, SFE_DICT, decompose_dict

BLOCK_MESH_DICT_V2 = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    (-8.0 0.0 -2.20) (13.12 0.0 -2.20) (13.12 3.96 -2.20) (-8.0 3.96 -2.20)
    (-8.0 0.0 -0.16) (13.12 0.0 -0.16) (13.12 3.96 -0.16) (-8.0 3.96 -0.16)
    (-8.0 0.0  0.16) (13.12 0.0  0.16) (13.12 3.96  0.16) (-8.0 3.96  0.16)
    (-8.0 0.0  1.10) (13.12 0.0  1.10) (13.12 3.96  1.10) (-8.0 3.96  1.10)
);
blocks
(
    hex (0 1 2 3 4 5 6 7)       (96 18 10) simpleGrading (1 1 0.22)
    hex (4 5 6 7 8 9 10 11)     (96 18 4)  simpleGrading (1 1 1)
    hex (8 9 10 11 12 13 14 15) (96 18 6)  simpleGrading (1 1 2.8)
);
edges ();
boundary
(
    inlet      { type patch;         faces ( (0 4 7 3) (4 8 11 7) (8 12 15 11) ); }
    outlet     { type patch;         faces ( (1 2 6 5) (5 6 10 9) (9 10 14 13) ); }
    midPlane   { type symmetryPlane; faces ( (0 1 5 4) (4 5 9 8) (8 9 13 12) ); }
    side       { type symmetryPlane; faces ( (3 7 6 2) (7 11 10 6) (11 15 14 10) ); }
    bottom     { type symmetryPlane; faces ( (0 3 2 1) ); }
    atmosphere { type patch;         faces ( (12 13 14 15) ); }
);
mergePatchPairs ();
"""

SNAPPY_DICT_V2 = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       true;
geometry
{
    hull.stl         { type triSurfaceMesh; name hull; }
    outerWakeBox     { type searchableBox; min (-3.8 0.0 -0.22); max (8.8 1.9 0.14); }
    wakeSurfaceBox   { type searchableBox; min (-3.6 0.0 -0.055); max (8.2 1.5 0.055); }
    hullWaterlineBox { type searchableBox; min (-2.9 0.0 -0.044); max (3.1 0.36 0.052); }
}
castellatedMeshControls
{
    maxLocalCells       300000;
    maxGlobalCells      2500000;
    minRefinementCells  5;
    nCellsBetweenLevels 3;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    features ( { file "hull.eMesh"; level 4; } );
    refinementSurfaces { hull { level (4 4); patchInfo { type wall; } } }
    refinementRegions
    {
        outerWakeBox     { mode inside; levels ((1E15 2)); }
        wakeSurfaceBox   { mode inside; levels ((1E15 3)); }
        hullWaterlineBox { mode inside; levels ((1E15 4)); }
    }
    locationInMesh (-5.0 1.5 0.5);
}
snapControls
{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes false; thicknessModel firstAndExpansion;
    expansionRatio 1.25; firstLayerThickness 0.00105; minThickness 0.00035;
    layers { hull { nSurfaceLayers 5; } }
    nGrow 0; featureAngle 140; slipFeatureAngle 30; nRelaxIter 5;
    nSmoothSurfaceNormals 3; nSmoothNormals 5; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.4; minMedialAxisAngle 75;
    nBufferCellsNoExtrude 0; nLayerIter 50; nRelaxedIter 20;
}
meshQualityControls { #include "meshQualityDict" nSmoothScale 4; errorReduction 0.75; }
mergeTolerance 1e-6;
"""

MESH_QUALITY_DICT_V2 = """FoamFile { version 2.0; format ascii; class dictionary; object meshQualityDict; }
#includeEtc "caseDicts/meshQualityDict"
maxNonOrtho 62; maxBoundarySkewness 3.8; maxInternalSkewness 3.8; maxConcave 80;
minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001;
minFaceWeight 0.05; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
"""

TOPO_SET_DICT_DAMPING = ""  # End sponge uses selectionMode all (no topoSet zones needed).


def fv_options_end_sponge(
    *,
    frequency: float = 0.15,
    w: float = 20.0,
    origin_x: float = 2.56,
    radius1: float = 7.6,
    radius2: float = 10.56,
) -> str:
    """Horizontal/all-component end sponge (acousticDampingSource) pulling U toward URef."""
    return (
        "FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\n"
        "spongeEnds\n{\n"
        "    type            acousticDampingSource;\n"
        "    selectionMode   all;\n"
        f"    frequency       {frequency:g};\n"
        f"    w               {w:g};\n"
        "    URef            URef;\n"
        f"    origin          ({origin_x:g} 0 0);\n"
        f"    radius1         {radius1:g};\n"
        f"    radius2         {radius2:g};\n"
        "    U               U;\n}\n"
    )


def fv_schemes_v2(include_potential_foam: bool = True) -> str:
    """Return fvSchemes for interFoam LTS + optional potentialFoam calm-start init."""
    from openfoam_dicts import FV_SCHEMES

    if not include_potential_foam:
        return FV_SCHEMES
    return FV_SCHEMES.replace(
        "div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;",
        "div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n    div(div(phi,U)) Gauss linear;",
    )


def fv_solution_v2(
    max_co: float = 50.0,
    max_alpha_co: float = 5.0,
    r_delta_t_smoothing: float = 0.05,
    include_potential_foam: bool = True,
) -> str:
    """Return fvSolution with configurable maxCo/maxAlphaCo and optional potentialFlow block."""
    from openfoam_dicts import FV_SOLUTION

    txt = FV_SOLUTION.replace(
        "maxCo 10; maxAlphaCo 5; rDeltaTSmoothingCoeff 0.05;",
        f"maxCo {max_co:g}; maxAlphaCo {max_alpha_co:g}; rDeltaTSmoothingCoeff {r_delta_t_smoothing:g};",
    )
    if include_potential_foam:
        txt = txt.replace(
            "    p_rghFinal { $p_rgh; relTol 0; }",
            "    p_rghFinal { $p_rgh; relTol 0; }\n    Phi { solver GAMG; smoother DIC; tolerance 1e-6; relTol 0.01; }\n    PhiFinal { $Phi; relTol 0; }",
        ) + "\npotentialFlow { nNonOrthogonalCorrectors 4; }\n"
    return txt


def write_mesh_dicts_v2(cdir: Path, np_ranks: int = 4) -> None:
    """Write Reference Mesh v2 OpenFOAM meshing dictionaries without touching baseline dicts."""
    from openfoam_dicts import FV_SCHEMES, FV_SOLUTION

    (cdir / "system").mkdir(parents=True, exist_ok=True)
    (cdir / "system/blockMeshDict").write_text(BLOCK_MESH_DICT_V2, encoding="utf-8")
    (cdir / "system/snappyHexMeshDict").write_text(SNAPPY_DICT_V2, encoding="utf-8")
    (cdir / "system/meshQualityDict").write_text(MESH_QUALITY_DICT_V2, encoding="utf-8")
    (cdir / "system/surfaceFeatureExtractDict").write_text(SFE_DICT, encoding="utf-8")
    (cdir / "system/controlDict").write_text(CTRL_MESH, encoding="utf-8")
    (cdir / "system/decomposeParDict").write_text(decompose_dict(np_ranks), encoding="utf-8")
    (cdir / "system/fvSchemes").write_text(FV_SCHEMES, encoding="utf-8")
    (cdir / "system/fvSolution").write_text(FV_SOLUTION, encoding="utf-8")


def write_interfoam_case_files_v2(
    cdir: Path,
    u_mps: float,
    cg: list[float],
    *,
    end_steps: int = 2000,
    np_ranks: int = 6,
    max_co: float = 50.0,
    max_alpha_co: float = 5.0,
    r_delta_t_smoothing: float = 0.05,
    pin_outlet: bool = True,
    damping_zones: bool = False,
    sponge_frequency: float = 0.15,
    sponge_w: float = 20.0,
    calm_start: str = "potential",
    ramp_steps: int = 300,
    binned_forces: bool = True,
) -> list[str]:
    """Write a Mesh v2 interFoam case reusing baseline fields without modifying cad/openfoam_dicts.py."""
    from openfoam_bin_forces import enable_binned_hull_forces
    from openfoam_dicts import write_interfoam_case_files

    if calm_start not in ("none", "potential", "ramp"):
        raise ValueError(f"Unsupported calm_start mode: {calm_start!r}")
    for sub in ("constant", "system", "logs"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)
    write_interfoam_case_files(cdir, u_mps, cg, end_steps=end_steps, np_ranks=np_ranks)
    use_pot = calm_start == "potential"
    (cdir / "system/fvSchemes").write_text(fv_schemes_v2(use_pot), encoding="utf-8")
    (cdir / "system/fvSolution").write_text(
        fv_solution_v2(max_co, max_alpha_co, r_delta_t_smoothing, use_pot), encoding="utf-8"
    )
    if pin_outlet and calm_start != "ramp":
        (cdir / "0.orig/p_rgh").write_text(
            'FoamFile { version 2.0; format ascii; class volScalarField; object p_rgh; }\ndimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      { type fixedFluxPressure; value uniform 0; }\n    outlet     { type fixedValue; value uniform 0; }\n    atmosphere { type totalPressure; p0 uniform 0; }\n    hull       { type fixedFluxPressure; value uniform 0; }\n}\n',
            encoding="utf-8",
        )
        (cdir / "0.orig/U").write_text(
            f'FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}\ndimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({u_mps:.6f} 0 0);\nboundaryField\n{{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      {{ type fixedValue; value uniform ({u_mps:.6f} 0 0); }}\n    outlet     {{ type inletOutlet; inletValue uniform ({u_mps:.6f} 0 0); value uniform ({u_mps:.6f} 0 0); }}\n    atmosphere {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}\n    hull       {{ type movingWallVelocity; value uniform (0 0 0); }}\n}}\n',
            encoding="utf-8",
        )
        (cdir / "0.orig/alpha.water").write_text(
            'FoamFile { version 2.0; format ascii; class volScalarField; object alpha.water; }\ndimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      { type fixedValue; value $internalField; }\n    outlet     { type zeroGradient; }\n    atmosphere { type inletOutlet; inletValue uniform 0; value uniform 0; }\n    hull       { type zeroGradient; }\n}\n',
            encoding="utf-8",
        )
    if damping_zones:
        (cdir / "constant/fvOptions").write_text(
            fv_options_end_sponge(frequency=sponge_frequency, w=sponge_w), encoding="utf-8"
        )
        ctrl = (cdir / "system/controlDict").read_text(encoding="utf-8")
        ctrl = ctrl.replace(
            "functions\n{\n    forces",
            "functions\n{\n    readURef { type readFields; libs (fieldFunctionObjects); fields (URef); readOnStart true; writeControl none; }\n    forces",
        )
        (cdir / "system/controlDict").write_text(ctrl, encoding="utf-8")
    if use_pot:
        (cdir / "0.orig/pPotential").write_text(
            'FoamFile { version 2.0; format ascii; class volScalarField; object pPotential; }\ndimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      { type zeroGradient; }\n    outlet     { type fixedValue; value uniform 0; }\n    atmosphere { type zeroGradient; }\n    hull       { type zeroGradient; }\n}\n',
            encoding="utf-8",
        )
    elif calm_start == "ramp":
        (cdir / "0.orig/U").write_text(
            f'FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}\ndimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      {{ type uniformFixedValue; uniformValue table ( (0 (0 0 0)) ({ramp_steps} ({u_mps:.6f} 0 0)) ); value uniform (0 0 0); }}\n    outlet     {{ type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0); }}\n    atmosphere {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}\n    hull       {{ type movingWallVelocity; value uniform (0 0 0); }}\n}}\n',
            encoding="utf-8",
        )
    if binned_forces:
        ctrl_path = cdir / "system/controlDict"
        ctrl_path.write_text(
            enable_binned_hull_forces(ctrl_path.read_text(encoding="utf-8"), cofr=cg),
            encoding="utf-8",
        )
    shutil.rmtree(cdir / "0", ignore_errors=True)
    shutil.copytree(cdir / "0.orig", cdir / "0")
    cmds = [
        "setFields > logs/setFields.log 2>&1",
        "decomposePar > logs/decomposePar.log 2>&1",
        f"mpirun -np {np_ranks} renumberMesh -parallel -overwrite > logs/renumber.log 2>&1",
    ]
    if damping_zones:
        cmds.append(
            f'mpirun -np {np_ranks} setExprFields -parallel -time 0 -field URef -create -expression "vector({u_mps:.6f}, 0, 0)" -dimensions "[0 1 -1 0 0 0 0]" > logs/setExpr.log 2>&1'
        )
    if use_pot:
        cmds.append(
            f"mpirun -np {np_ranks} potentialFoam -initialiseUBCs -pName pPotential -parallel -noFunctionObjects > logs/potentialFoam.log 2>&1"
        )
    cmds.append(f"mpirun -np {np_ranks} interFoam -parallel > logs/interFoam.log 2>&1")
    return cmds
