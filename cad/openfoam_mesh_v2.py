"""OpenFOAM-v2606 Reference Mesh v2 (graded waterline + hull prism layers) & morph helpers."""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np

from openfoam_dicts import FV_SCHEMES, FV_SOLUTION, SFE_DICT, CTRL_MESH, decompose_dict

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
    relativeSizes false;
    thicknessModel firstAndExpansion;
    expansionRatio 1.25;
    firstLayerThickness 0.00105;
    minThickness 0.00035;
    layers { hull { nSurfaceLayers 5; } }
    nGrow 0;
    featureAngle 140;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.4;
    minMedialAxisAngle 75;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}
meshQualityControls
{
    #include "meshQualityDict"
    nSmoothScale 4;
    errorReduction 0.75;
}
mergeTolerance 1e-6;
"""

MESH_QUALITY_DICT_V2 = """FoamFile { version 2.0; format ascii; class dictionary; object meshQualityDict; }
#includeEtc "caseDicts/meshQualityDict"
maxNonOrtho 62;
maxBoundarySkewness 3.8;
maxInternalSkewness 3.8;
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

DYNAMIC_MESH_DICT_MORPH = """FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }
dynamicFvMesh   dynamicMotionSolverFvMesh;
motionSolverLibs ("libfvMotionSolvers.so");
solver          displacementLaplacian;
displacementLaplacianCoeffs
{
    diffusivity quadratic inverseDistance 1(hull);
}
"""

MORPH_FV_SCHEMES = """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes
{
    default Gauss linear corrected;
    laplacian(diffusivity,cellDisplacement) Gauss linear corrected;
}
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""

MORPH_FV_SOLUTION = """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    "cellDisplacement.*"
    {
        solver PCG;
        preconditioner DIC;
        tolerance 1e-6;
        relTol 0;
        minIter 2;
    }
}
"""


def write_mesh_dicts_v2(cdir: Path, np_ranks: int = 4) -> None:
    """Write Reference Mesh v2 OpenFOAM meshing dictionaries without touching baseline dicts."""
    (cdir / "system").mkdir(parents=True, exist_ok=True)
    (cdir / "system/blockMeshDict").write_text(BLOCK_MESH_DICT_V2)
    (cdir / "system/snappyHexMeshDict").write_text(SNAPPY_DICT_V2)
    (cdir / "system/meshQualityDict").write_text(MESH_QUALITY_DICT_V2)
    (cdir / "system/surfaceFeatureExtractDict").write_text(SFE_DICT)
    (cdir / "system/controlDict").write_text(CTRL_MESH)
    (cdir / "system/decomposeParDict").write_text(decompose_dict(np_ranks))
    (cdir / "system/fvSchemes").write_text(FV_SCHEMES)
    (cdir / "system/fvSolution").write_text(FV_SOLUTION)


def _project_points_to_triangles(
    pts: np.ndarray, verts: np.ndarray, faces: np.ndarray, k: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto closest STL triangles using scipy cKDTree (no rtree required)."""
    from scipy.spatial import cKDTree

    tri = verts[faces]
    tree = cKDTree(tri.mean(axis=1))
    _, cand = tree.query(pts, k=min(k, len(faces)))
    cand = np.atleast_2d(cand)
    a, b, c = tri[cand, 0], tri[cand, 1], tri[cand, 2]
    ab, ac, ap = b - a, c - a, pts[:, None, :] - a
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    d3 = np.sum(ab * (pts[:, None, :] - b), axis=-1)
    d4 = np.sum(ac * (pts[:, None, :] - b), axis=-1)
    d5 = np.sum(ab * (pts[:, None, :] - c), axis=-1)
    d6 = np.sum(ac * (pts[:, None, :] - c), axis=-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    v = np.zeros_like(d1)
    w = np.zeros_like(d1)
    # Vertex A
    m_a = (d1 <= 0) & (d2 <= 0)
    # Vertex B
    m_b = ~m_a & (d3 >= 0) & (d4 <= d3)
    v[m_b] = 1.0
    # Edge AB
    m_ab = ~m_a & ~m_b & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v[m_ab] = np.clip(d1[m_ab] / np.maximum(d1[m_ab] - d3[m_ab], 1e-30), 0.0, 1.0)
    # Vertex C
    m_c = ~m_a & ~m_b & ~m_ab & (d6 >= 0) & (d5 <= d6)
    w[m_c] = 1.0
    # Edge AC
    m_ac = ~m_a & ~m_b & ~m_ab & ~m_c & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w[m_ac] = np.clip(d2[m_ac] / np.maximum(d2[m_ac] - d6[m_ac], 1e-30), 0.0, 1.0)
    # Edge BC
    m_bc = ~m_a & ~m_b & ~m_ab & ~m_c & ~m_ac & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w[m_bc] = np.clip((d4[m_bc] - d3[m_bc]) / np.maximum((d4[m_bc] - d3[m_bc]) + (d5[m_bc] - d6[m_bc]), 1e-30), 0.0, 1.0)
    v[m_bc] = 1.0 - w[m_bc]
    # Interior
    m_in = ~(m_a | m_b | m_ab | m_c | m_ac | m_bc)
    inv = 1.0 / np.maximum(va[m_in] + vb[m_in] + vc[m_in], 1e-30)
    v[m_in] = vb[m_in] * inv
    w[m_in] = vc[m_in] * inv
    u = 1.0 - v - w
    proj = u[..., None] * a + v[..., None] * b + w[..., None] * c
    dists = np.linalg.norm(proj - pts[:, None, :], axis=-1)
    best = np.argmin(dists, axis=1)
    idx = np.arange(len(pts))
    tri_idx = cand[idx, best]
    bary = np.stack([u[idx, best], v[idx, best], w[idx, best]], axis=-1)
    return tri_idx, bary, dists[idx, best]


def read_polymesh_points_ascii(points_path: Path) -> np.ndarray:
    """Parse an ASCII OpenFOAM polyMesh/points vectorField into an (N, 3) float64 array."""
    ptxt = points_path.read_text(encoding="utf-8")
    anchor = ptxt.find("// *")
    psub = ptxt[anchor:] if anchor >= 0 else ptxt
    pm0 = re.search(r"(\d+)\s*\(", psub)
    if pm0 is None:
        raise ValueError(f"Could not parse points count in {points_path}")
    n_pts = int(pm0.group(1))
    pbody = psub[pm0.end() :]
    pend = pbody.find(")\n\n") if ")\n\n" in pbody else pbody.rfind(")")
    return np.fromstring(
        pbody[:pend].replace("(", " ").replace(")", " "), sep=" ", count=n_pts * 3
    ).reshape(n_pts, 3)


def extract_patch_mesh_points(
    poly_mesh_dir: Path, patch_name: str = "hull"
) -> tuple[np.ndarray, list[int]]:
    """Read ASCII polyMesh and return patch vertices in OpenFOAM first-appearance meshPoints order."""
    btxt = (poly_mesh_dir / "boundary").read_text(encoding="utf-8")
    mb = re.search(
        rf"\b{re.escape(patch_name)}\s*\{{[^}}]*?nFaces\s+(\d+);[^}}]*?startFace\s+(\d+);",
        btxt,
        re.S,
    )
    if mb is None:
        raise ValueError(f"Patch {patch_name!r} not found in {poly_mesh_dir / 'boundary'}")
    n_faces, start_face = int(mb.group(1)), int(mb.group(2))

    ftxt = (poly_mesh_dir / "faces").read_text(encoding="utf-8")
    anchor = ftxt.find("// *")
    fsub = ftxt[anchor:] if anchor >= 0 else ftxt
    fm0 = re.search(r"(\d+)\s*\(", fsub)
    if fm0 is None:
        raise ValueError(f"Could not parse faces count in {poly_mesh_dir / 'faces'}")
    face_pat = re.compile(r"\d+\(([^)]+)\)")
    mesh_point_ids = list(
        dict.fromkeys(
            int(idx)
            for face_idx, m_face in enumerate(face_pat.finditer(fsub, fm0.end()))
            if start_face <= face_idx < start_face + n_faces
            for idx in m_face.group(1).split()
        )
    )
    pts = read_polymesh_points_ascii(poly_mesh_dir / "points")
    return pts[mesh_point_ids], mesh_point_ids


def build_stl_vertex_displacement(
    v_base: np.ndarray,
    f_base: np.ndarray,
    v_target: np.ndarray,
    hull_pts: np.ndarray,
    *,
    mode: str = "normal",
) -> np.ndarray:
    """Compute patch displacements from base STL to target STL ('normal' adds a 10 mm keel-seam blend)."""
    f_pos_mask = v_base[f_base].mean(axis=1)[:, 1] >= -1e-6
    f_pos = f_base[f_pos_mask] if np.any(f_pos_mask) else f_base
    tri_idx, bary, _ = _project_points_to_triangles(hull_pts, v_base, f_pos, k=32)
    d_param = np.einsum("ij,ijk->ik", bary, v_target[f_pos[tri_idx]]) - hull_pts
    if mode == "parametric":
        return d_param
    if mode == "normal":
        w_seam = np.exp(-((np.maximum(hull_pts[:, 1], 0.0) / 0.010) ** 2))[:, None]
        seed_pts = hull_pts + w_seam * d_param
        t_idx, t_bary, _ = _project_points_to_triangles(seed_pts, v_target, f_pos, k=32)
        target_pts = np.einsum("ij,ijk->ik", t_bary, v_target[f_pos[t_idx]])
        target_pts[hull_pts[:, 1] <= 1e-6, 1] = 0.0
        return target_pts - hull_pts
    raise ValueError(f"Unknown displacement mode: {mode}")


def write_point_displacement_field(path: Path, hull_disp: np.ndarray) -> None:
    """Write 0/pointDisplacement fixedValue boundary field for OpenFOAM moveDynamicMesh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vec_lines = "\n".join(f"({d[0]:.8e} {d[1]:.8e} {d[2]:.8e})" for d in hull_disp)
    path.write_text(
        "FoamFile { version 2.0; format ascii; class pointVectorField; object pointDisplacement; }\n"
        "dimensions [0 1 0 0 0 0 0];\n"
        "internalField uniform (0 0 0);\n"
        "boundaryField\n{\n"
        "    inlet      { type fixedValue; value uniform (0 0 0); }\n"
        "    outlet     { type fixedValue; value uniform (0 0 0); }\n"
        "    midPlane   { type symmetryPlane; }\n"
        "    side       { type symmetryPlane; }\n"
        "    bottom     { type symmetryPlane; }\n"
        "    atmosphere { type fixedValue; value uniform (0 0 0); }\n"
        f"    hull       {{ type fixedValue; value nonuniform List<vector> {len(hull_disp)} (\n{vec_lines}\n); }}\n"
        "}\n",
        encoding="utf-8",
    )


def write_polymesh_points_ascii(points_path: Path, pts: np.ndarray) -> None:
    """Write an (N, 3) float array as an ASCII OpenFOAM polyMesh/points file."""
    pts_arr = np.asarray(pts, dtype=float)
    lines = [f"({x:.9g} {y:.9g} {z:.9g})" for x, y, z in pts_arr]
    header = (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        '    class       vectorField;\n    location    "constant/polyMesh";\n'
        "    object      points;\n}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )
    points_path.write_text(f"{header}\n{len(pts_arr)}\n(\n" + "\n".join(lines) + "\n)\n", encoding="utf-8")


def morph_polymesh_points(
    poly_mesh_dir: Path,
    hull_pt_ids: list[int] | np.ndarray,
    hull_disp: np.ndarray,
    *,
    r_core: float = 0.015,
    r_outer: float = 0.45,
    k: int = 16,
    out_points_path: Path | None = None,
) -> np.ndarray:
    """Morph all mesh points via cKDTree k-NN IDW with rigid prism core and C^2 quintic cutoff."""
    from scipy.spatial import cKDTree

    pts = read_polymesh_points_ascii(poly_mesh_dir / "points")
    hull_ids_arr = np.asarray(hull_pt_ids, dtype=int)
    hull_pts = pts[hull_ids_arr]
    dists, idx = cKDTree(hull_pts).query(pts, k=min(int(k), len(hull_pts)))
    dists = np.atleast_2d(dists)
    idx = np.atleast_2d(idx)
    w = 1.0 / (dists + 2e-4) ** 2
    w /= w.sum(axis=1, keepdims=True)
    d_local = np.einsum("ij,ijk->ik", w, hull_disp[idx])
    s = np.clip((dists[:, 0] - r_core) / max(r_outer - r_core, 1e-9), 0.0, 1.0)
    decay = 1.0 - s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
    disp_all = d_local * decay[:, None]
    disp_all[hull_ids_arr] = hull_disp
    disp_all[pts[:, 1] <= 1e-7, 1] = 0.0
    morphed = pts + disp_all
    if out_points_path is not None:
        write_polymesh_points_ascii(out_points_path, morphed)
    return morphed
