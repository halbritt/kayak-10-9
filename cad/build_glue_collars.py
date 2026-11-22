"""Build removable bar-clamp collars for the released 1:8 model, in print mm."""

import manifold3d as mf
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

from build_model_1to8_h2s import BUILD_ROOT, write_bambu_3mf

PARTS = BUILD_ROOT / "model_1to8_h2s"
OUT = BUILD_ROOT / "glue_collars"
SETBACK = 3.0
THICKNESS = 10.0
CLEARANCE = 0.36
SPLIT_GAP = 2.4


def bore_points(boat, seam, direction, protect_rim):
    """Convex envelope of every triangle clipped to the collar's axial span."""
    near = seam + direction * SETBACK
    far = near + direction * THICKNESS
    lo, hi = sorted((near, far))
    vertices = boat.vertices
    between = vertices[(vertices[:, 0] >= lo) & (vertices[:, 0] <= hi)]
    ends = [
        trimesh.intersections.mesh_plane(boat, [1, 0, 0], [x, 0, 0]).reshape(-1, 3)
        for x in (near, far)
    ]
    xyz = np.vstack([between, *ends])
    local = np.column_stack((xyz[:, 1], xyz[:, 2], (xyz[:, 0] - near) * direction))
    if protect_rim:
        # Bridge the aft cockpit edge. Hold the midbody on its flanks.
        roof = local[:, 1].max() + 1.5
        local = np.vstack(
            [local, [[y, roof, z] for y in (-17, 17) for z in (0, THICKNESS)]]
        )
    return local[ConvexHull(local).vertices]


def bore(points):
    # Dilate sideways only. The 32-sided offset has >0.35 mm normal clearance.
    a = np.arange(32) * (2 * np.pi / 32)
    offsets = CLEARANCE * np.column_stack((np.cos(a), np.sin(a), np.zeros_like(a)))
    p = (points[:, None, :] + offsets).reshape(-1, 3)
    start = p[np.isclose(p[:, 2], 0)] - [0, 0, 0.1]
    end = p[np.isclose(p[:, 2], THICKNESS)] + [0, 0, 0.1]
    return mf.Manifold.hull_points(np.vstack([p, start, end]))


def outer_blank(half_width, bottom, top):
    """Rounded outside and a central tie channel with printable 45 degree lips."""
    center = (top + bottom) / 2
    radius = 7
    core = mf.CrossSection.square(
        (2 * (half_width - radius), top - bottom - 2 * radius), center=True
    )

    def band(inset, start, height):
        return (
            core.offset(radius - inset, mf.JoinType.Round, circular_segments=48)
            .extrude(height)
            .translate((0, center, start))
        )

    lower = band(0, 0, 1.3)
    upper = band(0, THICKNESS - 1.3, 1.3)
    channel = band(1, 2.3, THICKNESS - 4.6)
    lower_lip = (lower + band(1, 2.3, 0.1)).hull()
    upper_lip = (upper + band(1, THICKNESS - 2.4, 0.1)).hull()
    return lower_lip + channel + upper_lip


def make_pair(points, half_width, bottom, top, number):
    # Each jaw's outside axial face receives the bar-clamp pad. Count the
    # dimples on that face to identify the station and distinguish its ends.
    cutout = bore(points)
    blank = outer_blank(half_width, bottom, top)
    ring = blank - cutout
    jaws = []
    for side in (-1, 1):
        jaw = ring.trim_by_plane((side, 0, 0), SPLIT_GAP / 2)
        for i in range(number):
            hole = mf.Manifold.cylinder(1.4, 1.25, circular_segments=24)
            jaw -= hole.translate((side * (6 + 4 * i), top - 3.2, THICKNESS - 1.2))
        # Remove sub-micron facets before the 3MF writer rounds coordinates.
        m = jaw.simplify(0.001).to_mesh64()
        part = trimesh.Trimesh(m.vert_properties[:, :3].round(5), m.tri_verts)
        if (
            not part.is_volume
            or part.body_count != 1
            or not part.nondegenerate_faces().all()
        ):
            raise ValueError(
                f"Invalid collar {number} jaw {side}: {jaw.status()}, watertight={part.is_watertight}, winding={part.is_winding_consistent}, bodies={part.body_count}"
            )
        jaws.append(part)
    return jaws


def main():
    boat = trimesh.load(PARTS / "kayak-10-9-scale-1to8-uncut-full.stl")
    bow = trimesh.load(PARTS / "01_bow_section_1to8.stl")
    stern = trimesh.load(PARTS / "03_stern_section_1to8.stl")
    seams = [boat.bounds[0, 0] + bow.extents[2], boat.bounds[1, 0] - stern.extents[2]]
    labels = ["bow", "midbody_forward", "midbody_aft", "stern"]
    items = []
    for joint, seam in enumerate(seams):
        envelopes = [
            bore_points(boat, seam, direction, joint == 1 and direction == -1)
            for direction in (-1, 1)
        ]
        union = np.vstack(envelopes)
        half_width = np.max(np.abs(union[:, 0])) + 20
        bottom, top = np.min(union[:, 1]) - 7, np.max(union[:, 1]) + 7
        for i, points in enumerate(envelopes):
            index = 2 * joint + i
            for side, jaw in zip(
                (-1, 1), make_pair(points, half_width, bottom, top, index + 1)
            ):
                # Two halves of each numbered collar sit next to one another.
                pos = (
                    86 + 168 * i + side * 3,
                    86 + 148 * joint - (bottom + top) / 2,
                    0,
                )
                name = f"{index + 1}_{labels[index]}_{'left' if side < 0 else 'right'}"
                items.append((name, jaw, pos))
    OUT.mkdir(exist_ok=True, parents=True)
    write_bambu_3mf(
        items,
        OUT / "kayak-10-9-scale-1to8-h2s-glue-collars.3mf",
        title="Kayak 10.9 glue collars for small bar clamps",
        plate_name="Four split glue collars",
    )


if __name__ == "__main__":
    main()
