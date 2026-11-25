"""Package all Kayak 10.9 release archives, site-assets tarball, and SHA256SUMS.txt from build outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_zip(zip_path: Path, members: dict[str, Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname in sorted(members):
            zf.write(members[arcname], arcname)
    with ZipFile(zip_path) as zf:
        assert zf.testzip() is None, f"Corrupt zip: {zip_path}"


def _glob_map(directory: Path, pattern: str, prefix: str = "") -> dict[str, Path]:
    return {f"{prefix}{p.name}": p for p in sorted(directory.glob(pattern)) if p.is_file()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    ap.add_argument("--build-export-dir", type=Path, required=True)
    ap.add_argument("--site-stage-dir", type=Path, required=True)
    ap.add_argument("--release-dir", type=Path, required=True)
    args = ap.parse_args()

    build, bexp, site, rel = args.build_dir, args.build_export_dir, args.site_stage_dir, args.release_dir
    rel.mkdir(parents=True, exist_ok=True)
    finished, printdir = build / "finished", build / "model_1to8_h2s"

    models = {
        **_glob_map(finished, "*.glb"),
        **_glob_map(finished, "*.stl"),
        **_glob_map(build / "mold_stepped_cradle_assy", "*.glb"),
        **_glob_map(build / "mold_strongback_assy", "*.glb"),
    }
    _write_zip(rel / "kayak-10-9-3d-assemblies.zip", models)

    builder_files = {
        "finished-stations.dxf": finished / "finished-stations.dxf",
        "kayak-10-9-specs.json": REPO_ROOT / "data" / "kayak-10-9-specs.json",
        "kayak-10-9.hull.json": REPO_ROOT / "hull" / "kayak-10-9.hull.json",
        **{n: bexp / n for n in ("offsets.csv", "keel.svg", "sheer.svg", "waterline.svg", "deck_centreline.svg", "sections.dxf", "station_molds.dxf")},
    }
    _write_zip(rel / "kayak-10-9-builder-dxf-offsets.zip", builder_files)

    mold_members: dict[str, Path] = {}
    for sub in ("mold_h2s_dovetail_midship", "mold_h2s_fullwidth", "mold_station_06"):
        mold_members.update(_glob_map(build / sub, "*", prefix=f"{sub}/"))
    _write_zip(rel / "kayak-10-9-female-mold-stls.zip", mold_members)

    _write_zip(rel / "kayak-10-9-scale-1to8-stls.zip", {**_glob_map(printdir, "*.stl"), **_glob_map(printdir, "*.json")})

    for p in sorted(printdir.glob("*.3mf")):
        shutil.copy2(p, rel / p.name)
    for p in sorted((build / "hull_inspection_h2s").glob("*.3mf")):
        shutil.copy2(p, rel / p.name)

    models.update(_glob_map(printdir, "*.glb"))
    for p in (site / "models").glob("*.glb"):
        shutil.copy2(models[p.name], p)
    if (printdir / "h2s_1to8_plate_and_model_preview.png").is_file():
        shutil.copy2(printdir / "h2s_1to8_plate_and_model_preview.png", site / "plates" / "h2s_1to8_plate_and_model_preview.png")
    for p in (build / "mold_stepped_cradle_assy").glob("*.png"):
        shutil.copy2(p, site / ("plates" if "cross_section" in p.name else "renders") / p.name)

    with tarfile.open(rel / "kayak-10-9-site-assets.tar.gz", "w:gz") as tar:
        for folder in sorted(site.iterdir()):
            tar.add(folder, arcname=folder.name)

    (rel / "SHA256SUMS.txt").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(rel.iterdir()) if p.name != "SHA256SUMS.txt"))
    print("PACKAGED RELEASE ASSETS:", sorted(p.name for p in rel.iterdir()))


if __name__ == "__main__":
    main()
