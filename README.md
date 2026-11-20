# Kayak 10.9 — 18-Foot Fast Sea Kayak Design, Composite Tooling & 3D-Print Package

**Live Monograph & Interactive 3D Viewer:** **[https://halbritt.github.io/kayak-10-9/](https://halbritt.github.io/kayak-10-9/)**

**Kayak 10.9** is an open-hardware 18-foot Fast Sea Kayak (`L_oa = 5.470 m` CAD / `5.480 m` with over-stern gudgeon, `B_oa = 56.23 cm`, `B_4in = 50.05 cm`, Sound Rowers 4"-draft ratio **`10.88 <= 10.99`**, USCA 4"-beam **`9.15% >= 8.50%`**) generated and evaluated with [`halbritt/kayak-gen`](https://github.com/halbritt/kayak-gen) (`cabbd21`). It combines a wing-cut forward catch (`39.6 cm` at `+1.0 m`) and low `14.8 cm` rear deck with `253.3 L` of sealed dry storage (`98.3 L` bow + `5.5 L` contoured starboard carbon day pod + `149.5 L` unified stern) for day racing and minimalist watermaker expeditions.

## Latest Release Downloads (Auto-Advancing `/releases/latest/`)

Heavy 3D meshes, slicer plates, and full-scale CAD exports are published on **[GitHub Releases](https://github.com/halbritt/kayak-10-9/releases/latest)** to keep repository clones lightweight:

| Deliverable | Direct `"Latest"` Download | Contents |
| --- | --- | --- |
| **1:8 Scale Bambu Lab H2S `.3MF`** | [`kayak-10-9-scale-1to8-h2s-plate.3mf`](https://github.com/halbritt/kayak-10-9/releases/latest/download/kayak-10-9-scale-1to8-h2s-plate.3mf) | Single-plate (`340 x 320 x 340 mm`) support-free vertical bulkhead-split `1:8` scale model (`683.8 x 70.3 x 42.1 mm`), `6x` octagonal shear keys, and `2x` Bessel-station desk stands |
| **1:8 Scale Watertight `.STL` Set** | [`kayak-10-9-scale-1to8-stls.zip`](https://github.com/halbritt/kayak-10-9/releases/latest/download/kayak-10-9-scale-1to8-stls.zip) | Individual watertight `.stl` sections (`01_bow`, `02_cockpit`, `03_stern`, keys, stands) + uncut `683.75 mm` solid `.stl` |
| **Full-Scale Builder DXFs & Offsets** | [`kayak-10-9-builder-dxf-offsets.zip`](https://github.com/halbritt/kayak-10-9/releases/latest/download/kayak-10-9-builder-dxf-offsets.zip) | `station_molds.dxf` (`29` stations), `sections.dxf`, `offsets.csv`, profile SVGs, and `kayak-10-9.hull.json` |
| **Full-Scale 3D-Printed Mold `.STL` Samples** | [`kayak-10-9-female-mold-stls.zip`](https://github.com/halbritt/kayak-10-9/releases/latest/download/kayak-10-9-female-mold-stls.zip) | Watertight zero-undercut female hull & deck mold stations for Bambu Lab H2S (unsplit forward + midship slicer-dovetail) and Voron Phoenix `600³` (`45°` diagonal Station 06) |
| **Interactive 3D `.GLB` & Outer OML `.STL`** | [`kayak-10-9-3d-assemblies.zip`](https://github.com/halbritt/kayak-10-9/releases/latest/download/kayak-10-9-3d-assemblies.zip) | 8-part finished `.glb`, smoked-deck X-ray `.glb`, H2S stepped wedge-clamping Baltic-birch strongback `.glb`, and `kayak-10-9-outer-shell.stl` |
| **Checksums** | [`SHA256SUMS.txt`](https://github.com/halbritt/kayak-10-9/releases/latest/download/SHA256SUMS.txt) | SHA-256 digests for all release archives and `.3mf` assets |

## Repository Layout

- [`hull/kayak-10-9.hull.json`](hull/kayak-10-9.hull.json) — Canonical `kayak-gen` `DistributionV2` parametric hull & deck definition.
- [`cad/`](cad/) — Parametric Python generators (`deck_geometry.py`, `outfitting_geometry.py`, `build_stepped_cradle_strongback.py`, `build_hull_mold_strongback_assy.py`, `build_model_1to8_h2s.py`, and female mold station builders).
- [`data/`](data/) — Hydrostatic, OpenFOAM-v2512 RANS (`raw_unvalidated`), Capytaine 3.0 3D BEM seakeeping, windage/stiffener JSON datasets, station `offsets.csv`, and profile SVGs.
- [`site/`](site/) — Static site builder (`build_site.py`, `styles.css`) deployed to [GitHub Pages](https://halbritt.github.io/kayak-10-9/) via [`.github/workflows/pages.yml`](.github/workflows/pages.yml).

## Rebuilding Locally

With a Python 3.12 environment containing [`kayak-gen`](https://github.com/halbritt/kayak-gen), `numpy`, `scipy`, `trimesh`, `vtk`, `pyvista`, and `matplotlib`:

```bash
python3 cad/build_stepped_cradle_strongback.py   # -> build/mold_stepped_cradle_assy/
python3 cad/build_model_1to8_h2s.py              # -> build/model_1to8_h2s/ (.3mf + .stl)
python3 site/build_site.py --dist _site          # -> _site/index.html
```

## Epistemological Scope & Licenses

Per `kayak-gen` release discipline, all analytical (`ITTC-57 + Michell`), OpenFOAM-v2512 `interFoam` RANS (`raw_unvalidated`), Capytaine 3.0 3D linear BEM seakeeping, hydrostatic `GZ`, and laminate deflection results are **calculated comparative predictions**, not tow-tank-calibrated guarantees. Code in `cad/` and `site/` is [MIT licensed](LICENSE); hull geometry, CAD/mold/print exports, and documentation are licensed under [CC BY 4.0](LICENSE-HARDWARE.md).
