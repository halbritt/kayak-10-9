# Changelog

All notable changes to **Kayak 10.9** are documented in this file.

## [v0.2.1] - 2026-11-22

- Pinned keel depth and scaled `D³` bilge-batten span by submerged girth (`s_wl`) in `kayak-gen`, and replaced the uniform `0.08 m` bilge round with a longitudinal ramp (`forward: 0.07 -> aft: 0.0` over `xi = -0.38..+0.05`). Restores the full `10.88` Sound Rowers 4"-draft ratio (`5.4428 m / 50.05 cm`), `4.47 / 5.08 / 5.67 / 6.17 kn` calm-water speeds at `40 / 60 / 90 / 120 W` (`225 lb`), and `10.0 mm` `GZ(20°)` while keeping the forebody bilge and forward keel groove faired.
- Added a regression test (`test_sound_rowers_ratio_and_keel_baseline_preserved`) pinning `SR_4in == 10.88`, `L_4in >= 5.442 m`, and zero keel lift, and regenerated all CAD, mold, `1:8` `.3mf`/`.stl`, hydrostatic, plate, studio render, and monograph deliverables.

## [v0.2.0] - 2026-11-22

- Faired the submerged quarter-ellipse bilge ridge and forward centreline keel groove by setting `topsides.bilge_round_m = 0.08` (`kayak-gen` RFC 0089 stage 9 symmetric `D³` batten fairing). Sound Rowers 4"-draft ratio moves from `10.88` to `10.79 <= 10.99`, USCA 4"-beam is `9.16% >= 8.50%`, and wing-paddle catch beam stays within `1 mm` (`43.0 / 39.7 / 36.2 cm` at `+0.8 / +1.0 / +1.2 m`).
- Added smooth vertex normals to `kayak-10-9-finished.glb` and `kayak-10-9-xray-cutaway.glb`, exact `y = 0` plane-slice crossing interpolation in `SculptedKayakSurface.mold_station_rings_mm`, and dynamic alignment-bore clearance in `cad/build_mold_h2s_fullwidth.py`.
- Regenerated all builder patterns/offsets, finished outer shell, 8-part boat `.glb`s, sample female molds, both full-scale strongback mold assemblies, `1:8` H2S `.3mf`/`.stl` packages, hydrostatic specs, plates, 4K studio renders, and monograph. Because `v0.2.0` fairs the full hull bilge across all three `1:8` sections, print the full `v0.2.0` plate for a matched model (existing `v0.1.0` prints can still pair with the [`v0.1.2` bow-only `.3mf`](https://github.com/halbritt/kayak-10-9/releases/download/v0.1.2/kayak-10-9-scale-1to8-h2s-bow.3mf)).

## Glue collar accessory · 2026-11-21

- Added a standalone H2S plate with four split collars for clamping both joints of the 1:8 model with small bar clamps.

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
