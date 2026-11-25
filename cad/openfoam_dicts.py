"""Self-contained OpenFOAM (v2606 / v2512) meshing and interFoam RAS kOmegaSST dictionaries."""

from __future__ import annotations

from pathlib import Path

BLOCK_MESH_DICT = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    (-8.0 0.0 -2.20) (13.12 0.0 -2.20) (13.12 3.96 -2.20) (-8.0 3.96 -2.20)
    (-8.0 0.0  1.10) (13.12 0.0  1.10) (13.12 3.96  1.10) (-8.0 3.96  1.10)
);
blocks ( hex (0 1 2 3 4 5 6 7) (96 18 15) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet      { type patch;         faces ( (0 4 7 3) ); }
    outlet     { type patch;         faces ( (1 2 6 5) ); }
    midPlane   { type symmetryPlane; faces ( (0 1 5 4) ); }
    side       { type symmetryPlane; faces ( (3 7 6 2) ); }
    bottom     { type symmetryPlane; faces ( (0 3 2 1) ); }
    atmosphere { type patch;         faces ( (4 5 6 7) ); }
);
mergePatchPairs ();
"""

SNAPPY_DICT = """FoamFile { version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       false;
geometry
{
    hull.stl       { type triSurfaceMesh; name hull; }
    outerWakeBox   { type searchableBox; min (-3.8 0.0 -0.32); max (9.2 2.0 0.16); }
    wakeSurfaceBox { type searchableBox; min (-3.6 0.0 -0.055); max (8.6 1.8 0.055); }
    hullSurfaceBox { type searchableBox; min (-3.0 0.0 -0.040); max (4.0 0.55 0.040); }
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
        outerWakeBox   { mode inside; levels ((1E15 2)); }
        wakeSurfaceBox { mode inside; levels ((1E15 3)); }
        hullSurfaceBox { mode inside; levels ((1E15 4)); }
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
    relativeSizes true; layers {} expansionRatio 1.2; finalLayerThickness 0.3; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3;
    nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}
meshQualityControls
{
    #include "meshQualityDict"
    nSmoothScale 4;
    errorReduction 0.75;
}
mergeTolerance 1e-6;
"""

MESH_QUALITY_DICT = """FoamFile { version 2.0; format ascii; class dictionary; object meshQualityDict; }
#includeEtc "caseDicts/meshQualityDict"
maxNonOrtho 65;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-13;
minTetQuality 1e-30;
minArea -1;
minTwist 0.02;
minDeterminant 0.001;
minFaceWeight 0.05;
minVolRatio 0.01;
minTriangleTwist -1;
nSmoothScale 4;
errorReduction 0.75;
"""

SFE_DICT = """FoamFile { version 2.0; format ascii; class dictionary; object surfaceFeatureExtractDict; }
hull.stl { extractionMethod extractFromSurface; includedAngle 150; writeObj no; }
"""

CTRL_MESH = """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat binary; writePrecision 6; timePrecision 6;
"""

FV_SCHEMES = """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default localEuler; }
gradSchemes { default Gauss linear; limitedGrad cellLimited Gauss linear 1; }
divSchemes
{
    div(rhoPhi,U)   Gauss linearUpwind grad(U);
    div(phi,alpha)  Gauss vanLeer;
    div(phirb,alpha) Gauss linear;
    div(phi,k)      Gauss linearUpwind limitedGrad;
    div(phi,omega)  Gauss linearUpwind limitedGrad;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
"""

FV_SOLUTION = """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    "alpha.water.*"
    {
        nAlphaCorr 2; nAlphaSubCycles 1; cAlpha 1; icAlpha 0;
        MULESCorr yes; nLimiterIter 10; alphaApplyPrevCorr yes;
        solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; minIter 1;
    }
    "pcorr.*"
    {
        solver PCG;
        preconditioner { preconditioner GAMG; smoother GaussSeidel; tolerance 1e-5; relTol 0; };
        tolerance 1e-5; relTol 0;
    }
    p_rgh { solver GAMG; smoother DIC; tolerance 1e-7; relTol 0.01; }
    p_rghFinal { $p_rgh; relTol 0; }
    "(U|k|omega).*" { solver smoothSolver; smoother symGaussSeidel; nSweeps 1; tolerance 1e-7; relTol 0.1; minIter 1; }
}
PIMPLE
{
    momentumPredictor no; nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0;
    maxCo 10; maxAlphaCo 5; rDeltaTSmoothingCoeff 0.05; rDeltaTDampingCoeff 0.5;
    nAlphaSpreadIter 0; nAlphaSweepIter 0; maxDeltaT 1;
}
relaxationFactors { equations { ".*" 1; } }
cache { grad(U); }
"""


def decompose_dict(np_ranks: int = 6) -> str:
    return f"FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}\nnumberOfSubdomains {np_ranks}; method scotch;\n"


def write_mesh_dicts(cdir: Path, np_ranks: int = 6) -> None:
    (cdir / "system/blockMeshDict").write_text(BLOCK_MESH_DICT)
    (cdir / "system/snappyHexMeshDict").write_text(SNAPPY_DICT)
    (cdir / "system/meshQualityDict").write_text(MESH_QUALITY_DICT)
    (cdir / "system/surfaceFeatureExtractDict").write_text(SFE_DICT)
    (cdir / "system/controlDict").write_text(CTRL_MESH)
    (cdir / "system/decomposeParDict").write_text(decompose_dict(np_ranks))
    (cdir / "system/fvSchemes").write_text(FV_SCHEMES)
    (cdir / "system/fvSolution").write_text(FV_SOLUTION)


def write_interfoam_case_files(cdir: Path, u_mps: float, cg: list[float], end_steps: int = 500, np_ranks: int = 6) -> None:
    z0 = cdir / "0.orig"
    z0.mkdir(parents=True, exist_ok=True)
    k_val = round(1.5 * (u_mps * 0.01) ** 2, 6)
    om_val = round(k_val / (2.0 * 1.19e-6), 1)

    (z0 / "U").write_text(
        f'FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}\ndimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({u_mps:.6f} 0 0);\nboundaryField\n{{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      {{ type fixedValue; value uniform ({u_mps:.6f} 0 0); }}\n    outlet     {{ type outletPhaseMeanVelocity; alpha alpha.water; Umean {u_mps:.6f}; value uniform ({u_mps:.6f} 0 0); }}\n    atmosphere {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}\n    hull       {{ type movingWallVelocity; value uniform (0 0 0); }}\n}}\n'
    )
    (z0 / "alpha.water").write_text(
        'FoamFile { version 2.0; format ascii; class volScalarField; object alpha.water; }\ndimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      { type fixedValue; value $internalField; }\n    outlet     { type variableHeightFlowRate; lowerBound 0; upperBound 1; value $internalField; }\n    atmosphere { type inletOutlet; inletValue uniform 0; value uniform 0; }\n    hull       { type zeroGradient; }\n}\n'
    )
    (z0 / "p_rgh").write_text(
        'FoamFile { version 2.0; format ascii; class volScalarField; object p_rgh; }\ndimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      { type fixedFluxPressure; value uniform 0; }\n    outlet     { type zeroGradient; }\n    atmosphere { type totalPressure; p0 uniform 0; }\n    hull       { type fixedFluxPressure; value uniform 0; }\n}\n'
    )
    (z0 / "k").write_text(
        f'FoamFile {{ version 2.0; format ascii; class volScalarField; object k; }}\ndimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k_val};\nboundaryField\n{{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      {{ type fixedValue; value $internalField; }}\n    outlet     {{ type inletOutlet; inletValue $internalField; value $internalField; }}\n    atmosphere {{ type inletOutlet; inletValue $internalField; value $internalField; }}\n    hull       {{ type kqRWallFunction; value $internalField; }}\n}}\n'
    )
    (z0 / "omega").write_text(
        f'FoamFile {{ version 2.0; format ascii; class volScalarField; object omega; }}\ndimensions [0 0 -1 0 0 0 0];\ninternalField uniform {om_val};\nboundaryField\n{{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      {{ type fixedValue; value $internalField; }}\n    outlet     {{ type inletOutlet; inletValue $internalField; value $internalField; }}\n    atmosphere {{ type inletOutlet; inletValue $internalField; value $internalField; }}\n    hull       {{ type omegaWallFunction; value $internalField; }}\n}}\n'
    )
    (z0 / "nut").write_text(
        'FoamFile { version 2.0; format ascii; class volScalarField; object nut; }\ndimensions [0 2 -1 0 0 0 0];\ninternalField uniform 2.4e-06;\nboundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n    inlet      { type fixedValue; value $internalField; }\n    outlet     { type zeroGradient; }\n    atmosphere { type zeroGradient; }\n    hull       { type nutUSpaldingWallFunction; value $internalField; }\n}\n'
    )
    (cdir / "constant/transportProperties").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }\nphases (water air);\nwater { transportModel Newtonian; nu 1.19e-06; rho 1025.0; }\nair   { transportModel Newtonian; nu 1.48e-05; rho 1.20; }\nsigma 0.072;\n"
    )
    (cdir / "constant/g").write_text(
        "FoamFile { version 2.0; format ascii; class uniformDimensionedVectorField; object g; }\ndimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -9.80665);\n"
    )
    (cdir / "constant/turbulenceProperties").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs off; }\n"
    )
    (cdir / "system/setFieldsDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object setFieldsDict; }\ndefaultFieldValues ( volScalarFieldValue alpha.water 0 );\nregions\n(\n    boxToCell { box (-999 -999 -999) (999 999 0.0); fieldValues ( volScalarFieldValue alpha.water 1 ); }\n    boxToFace { box (-999 -999 -999) (999 999 0.0); fieldValues ( volScalarFieldValue alpha.water 1 ); }\n);\n"
    )
    (cdir / "system/controlDict").write_text(
        f"FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}\napplication interFoam; startFrom startTime; startTime 0; stopAt endTime; endTime {end_steps}; deltaT 1;\nwriteControl timeStep; writeInterval {end_steps // 2}; purgeWrite 0; writeFormat ascii; writePrecision 7;\nwriteCompression off; timeFormat general; timePrecision 6; runTimeModifiable no;\nfunctions\n{{\n    forces\n    {{\n        type forces; libs (forces); patches (hull); rhoInf 1025.0; log on;\n        writeControl timeStep; writeInterval 1; CofR ({cg[0]} {cg[1]} {cg[2]});\n    }}\n}}\n"
    )
    (cdir / "system/decomposeParDict").write_text(decompose_dict(np_ranks))
    (cdir / "system/fvSchemes").write_text(FV_SCHEMES)
    (cdir / "system/fvSolution").write_text(FV_SOLUTION)
