"""Export the finished outer surface, outfitted assembly, cutaway and builder sections."""

from pathlib import Path

import ezdxf
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from deck_geometry import SculptedKayakSurface
from outfitting_geometry import (
    build_coaming_and_thigh_hooks, build_day_pod_and_structural_grid,
    build_hatch_covers, build_interior_and_bulkheads, build_overstern_rudder,
    build_seat_and_footboard,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "kayak-10-9"


def export_sections(shell: trimesh.Trimesh, path: Path) -> None:
    """3D polylines in mm, with 10 mm stations through the finished forefoot."""
    doc = ezdxf.new(units=ezdxf.units.MM)
    sections = doc.modelspace()
    doc.layers.new("FINISHED_STATIONS")
    doc.layers.new("STEM_AND_KEEL")
    bow, stern = shell.bounds[:, 0]
    stations = np.unique(np.r_[bow + np.arange(0.005, 0.256, 0.010), np.linspace(bow, stern, 31)[1:-1]])
    for x in stations:
        section = shell.section([1, 0, 0], [x, 0, 0])
        for line in section.discrete:
            sections.add_polyline3d(line * 1000, dxfattribs={"layer": "FINISHED_STATIONS"})
    profile = shell.section([0, 1, 0], [0, 0, 0])
    for line in profile.discrete:
        sections.add_polyline3d(line * 1000, dxfattribs={"layer": "STEM_AND_KEEL"})
    doc.saveas(path)


def main() -> None:
    out = ROOT / "build" / "finished"
    out.mkdir(parents=True, exist_ok=True)
    surf = SculptedKayakSurface(ROOT / "hull" / f"{MODEL}.hull.json")
    hull = surf.build_hull_shell()
    shell = trimesh.util.concatenate((hull, surf.build_deck_shell(cut_openings=False)))
    shell.merge_vertices()
    if not shell.is_volume:
        raise ValueError("Expected a closed, outward-facing outer shell")
    shell.export(out / f"{MODEL}-outer-shell.stl")
    export_sections(shell, out / "finished-stations.dxf")

    parts = (
        ("01_hull_shell", hull, [238, 242, 246], 0.28, 0.05),
        ("02_deck_shell", surf.build_deck_shell(cut_openings=True), [88, 148, 192], 0.28, 0.08),
        ("03_cockpit_coaming_and_thigh_hooks", build_coaming_and_thigh_hooks(surf), [38, 43, 50], 0.45, 0.18),
        ("04_recessed_hatch_covers", build_hatch_covers(surf), [58, 66, 76], 0.55, 0.10),
        ("05_cockpit_interior_and_bulkheads", build_interior_and_bulkheads(surf), [208, 196, 168], 0.52, 0.05),
        ("06_seat_pan_and_footboard", build_seat_and_footboard(surf), [50, 56, 66], 0.42, 0.15),
        ("07_overstern_rudder", build_overstern_rudder(surf), [112, 122, 134], 0.32, 0.45),
        ("08_day_pod_and_deck_ribs_keel_strap", build_day_pod_and_structural_grid(surf), [42, 48, 56], 0.38, 0.16),
    )
    cutaway = {
        "02_deck_shell": ([88, 148, 192, 76], 0.22, 0.05),
        "04_recessed_hatch_covers": ([58, 66, 76, 128], 0.45, 0.08),
        "08_day_pod_and_deck_ribs_keel_strap": ([20, 138, 108, 255], 0.34, 0.14),
    }
    for view, overrides in (("finished", {}), ("xray-cutaway", cutaway)):
        scene = trimesh.Scene()
        for name, mesh, rgb, roughness, metal in parts:
            rgba, rough, metallic = overrides.get(name, (rgb + [255], roughness, metal))
            part = mesh.copy()
            part.visual = TextureVisuals(material=PBRMaterial(
                name=f"{name}_mat", baseColorFactor=rgba, roughnessFactor=rough,
                metallicFactor=metallic, doubleSided=True,
                alphaMode="BLEND" if rgba[-1] < 255 else None,
            ))
            scene.add_geometry(part, node_name=name, geom_name=name)
        scene.export(out / f"{MODEL}-{view}.glb", include_normals=True)


if __name__ == "__main__":
    main()
