# Changelog

All notable changes to **Kayak 10.9** are documented in this file.

## Unreleased

- **Seakeeping correction.** `cad/run_capytaine_seakeeping.py::build_bem_body` now builds the Capytaine body with a lid mesh (irregular-frequency removal). Without it the interior free surface of the hull resonates from about 11 rad/s, inside the encounter-frequency range of every forward-speed case. At 4.5 kn bow-quarter the shortest period of the grid (1.05 s) sat on such a frequency and read a heave excitation of 10,740 N/m where the corrected body gives 90 N/m. `data/capytaine-seakeeping.json`, Plate 6 and the site figures were released with `v0.2.3` before this fix and are unchanged here. Against the corrected rows they overstate Kayak 10.9's seat acceleration at 4.5 kn bow-quarter by 10 to 15 % (bay chop `1.476` against `1.260 m/s²`) and its bow dips there by 12 %, and they overstate the Epic 18X's seat acceleration at 265 lb in 4.5 kn head seas by up to 16 % (bay chop `1.662` against `1.391 m/s²`), which halves the published seat-acceleration advantage at that load (26 % becomes 12 %). Other gated figures move by under 3 %.
- **RSI harness, gates version 5.** The Tier 2 seakeeping gates now use the Epic 18X as the floor: in each gated condition a candidate's seat acceleration and bow dips may not exceed the Epic 18X's in the same model (or v0.2.3's, where v0.2.3 is itself behind the 18X). They replace the limit of v0.2.3 plus 1.5 %. The corrected rows for both boats are in `data/rsi-seakeeping-reference.json`, which only the harness reads.

## [v0.2.3] - 2026-11-24

- Re-ran matched **OpenFOAM-v2606** (`_481094f-20260618`, `snappyHexMesh` `354,132` cells + 6-rank `interFoam` `kOmegaSST`) and **Capytaine 3.0.0** (`1,408`-panel 3D linear BEM JONSWAP seakeeping + Munk/aerodynamic crosswind rudder trim) on the `v0.2.2` whole-hull faired surface (`fcbf0bdb...`) vs. the Epic 18X scan, updating `data/openfoam-cfd-metrics.json`, `data/capytaine-seakeeping.json`, `data/windage-and-stiffener.json`, and Plates 1–8 (`20.50 ± 0.10 N @ 5.50 kn`, `-4.1%` vs. 18X; `25.86 ± 0.04 N @ 6.20 kn`, `-4.8%` total / `-22.2%` wave drag vs. 18X; `-22%` to `-26%` fewer head-chop bow dips; `2.11° / 4.54°` rudder trim at `15 / 22 kn` beam wind). Hull and CAD geometry are unchanged from `v0.2.2`.
- Checked the complete naval-architecture, OpenFOAM-v2606 CFD, Capytaine 3.0 BEM, 8-plate, 4K studio-render, and release-packaging toolchain into `cad/` (`hydro.py`, `openfoam_dicts.py`, `run_openfoam_cfd.py`, `run_capytaine_seakeeping.py`, `build_specs_and_ga.py`, `build_plates.py`, `render_studio.py`, `package_release.py`) and `data/benchmarks/` with zero external `~/.cache` or `openfoam2512` dependencies, wired `site/build_site.py` directly to the JSON datasets, and added `tests/test_naval_arch_pipeline.py`.

## [v0.2.2] - 2026-11-22

- Wired `hull/kayak-10-9.hull.json` to `kayak-gen` `DistributionV2Spec.surface_fairing` (`b67aa6c`, RFC 0091) so the parametric hull, `kayakgen build-export` station offsets/molds, all H2S/Voron female mold cavities, scale prints, and finished boat shells share one cubic B-spline hull surface from keel to sheer and bow to stern.
- Excluded singular raked stem tips (`end_distance < 0.22 m`) from the `12 x 22` least-squares B-spline domain (`end_blend_m = 0.35`, `target_beam_m = 0.500456`). Eliminates the `+0.506°` outward waterline reversal at `0.867 m` aft of bow from `v0.2.1` AND removes the `+0.322°` `0.48 m` entry ripple and `3.45° -> 3.63°` `75 mm` aft-buttock sharpening (`3.46°` in `v0.2.2`) from `7b5f2d2`, while keeping the low aft coaming (`15.6 cm` above `225 lb` WL), `10.88` Sound Rowers 4"-draft ratio, `225/265/300 lb` drafts (`<= 0.017 mm` delta), and `GZ` righting arms (`<= 0.009 mm` delta) intact. [Inspection and design controls](docs/surface-fairing/README.md)
- Regenerated all builder DXF/offset exports, finished outer shell (`fcbf0bdb...`) and 8-part `.glb`s, sample female molds, stepped-cradle and full-width strongbacks, `1:8` and `1:12` `45°` H2S `.3mf`/`.stl` packages, hydrostatic specs, plates, 4K studio renders, and monograph assets.

## [v0.2.1] - 2026-11-22

- Added `cad/build_hull_inspection_h2s.py` (`kayak-10-9-hull-inspection-h2s.3mf` and `kayak-10-9-hull-only-h2s.3mf`): a one-shot, single-piece `1:12` (`1" = 1'`, `454.9 mm` long, `326.2 mm` Z) Bambu Lab H2S bare-hull inspection print oriented at `45°` across the `340 x 320 mm` plate diagonal with keel, forefoot, and bilges facing UP (`+Z`, support-blocked) and organic tree supports enabled along 6 sheer-back stations (`-Z`) plus two `1:12` Bessel cradles.
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
