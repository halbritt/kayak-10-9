# Changelog

All notable changes to **Kayak 10.9** are documented in this file.

## [v0.1.2] - 2026-11-21

- Traversed forward bow mold stations along the connected outer-shell section curve instead of sorting vertices by height, eliminating the `1.56 mm` keel zigzag where the bow rocker lifts `z(y=0)` slightly above adjacent keel points.
- Preserved curve order in both strongback mold generators via `SculptedKayakSurface.mold_stbd_section_mm` and regenerated the full-scale stepped-cradle and full-width mold assemblies.

## [v0.1.1] - 2026-11-21

- Faired the diagonal pinch where the stem turns into the keel. The surface now has a continuous tangent across the original crease and retains the 135 × 56 mm elliptical forefoot profile.
- Added a bow-only H2S plate for existing 1:8 models. The bulkhead and alignment sockets fit the v0.1.0 midbody and keys.
- Regenerated both outfitted boat views, the outer shell, complete scale model, tooling assemblies, hydrostatic summary and monograph. The forward mold cavities now use transverse cuts through the shaped bow. The builder package includes a finished-surface DXF with closely spaced bow sections.

## [v0.1.0] - 2026-11-20

### Added
- **Canonical Hull & Outfitting Specification (`hull/kayak-10-9.hull.json`, `cad/`):**
  - 18-foot USCA & Sound Rowers Fast Sea Kayak (`L_oa = 5.470 m` CAD / `5.480 m` with over-stern gudgeon, `B_oa = 56.23 cm`, `B_4in = 50.05 cm`, Sound Rowers 4"-draft ratio `10.88 <= 10.99`, USCA 4"-beam `9.15% >= 8.50%`), generated via [`kayak-gen`](https://github.com/halbritt/kayak-gen) (`cabbd21`).
  - Epic 18X / Stellar S18R-style 3D rounded bow (`r_LE = 3.8 mm` leading-edge bullnose, `26 x 23 mm` nose crown fillet, `135 x 56 mm` forefoot curve) and `11 mm` stern tip fillet.
  - 2-bulkhead (`-0.95 m` forward BH + `+0.44 m` slanted seat BH) + contoured `5.5 L` starboard DIY carbon day pod (`+0.66 m, y = +11.8 cm`) + co-cured `+64 g` cruciform (`+`) rear-deck UD-carbon stiffener grid (`1.79 mm` calculated deflection @ `180 lbf`) + flush `60 mm` internal UD-carbon keel strap (`253.3 L` total sealed dry storage).
- **Comparative Hydrodynamics & Seakeeping Datasets (`data/`):**
  - Matched OpenFOAM-v2512 `snappyHexMesh` + `interFoam` free-surface RANS (`5.50` & `6.20 kn`, `raw_unvalidated` comparative), Capytaine 3.0 3D linear BEM JONSWAP seakeeping (`1,408` panels), and 4-load (`213 / 225 / 265 / 300 lb`) hydrostatic righting-arm (`GZ`) datasets.
- **3D-Printed Female Mold, Stepped Clamping Strongback & 1:8 Scale H2S `.3MF` Tooling:**
  - Full-length 17-station (`324.0 mm` H2S Z-height) split-half female hull mold seated in a stepped, `8°` gib-wedge-clamping `18 mm` Baltic-birch torsion-box strongback (`cad/build_stepped_cradle_strongback.py`), plus Voron `600³` and Bambu Lab H2S dovetail/full-width sample mold `.stl` generators.
  - Single-plate Bambu Lab H2S (`340 x 320 x 340 mm`) `1:8` scale (`683.75 x 70.29 x 42.07 mm`) support-free vertical bulkhead-split `.3mf` & watertight `.stl` print package (`cad/build_model_1to8_h2s.py`).
- **Self-Contained Public Monograph & Interactive 3D Viewer (`site/`, `.github/workflows/pages.yml`):**
  - GitHub Pages site (`https://halbritt.github.io/kayak-10-9/`) with multi-model in-browser `Online3DViewer`, 4K orthographic studio renders, 8 naval-architecture plates, tooling cross-sections, and permanent `/releases/latest/download/...` asset links.
