# Whole-hull fairing of Kayak 10.9

The v0.2.2 hull uses a single cubic B-spline fit from keel to sheer and bow to stern via `kayak-gen` `DistributionV2Spec.surface_fairing` (RFC 0091). The v0.2.1 forebody had a small knuckle 0.867 m aft of the bow where a girth-dependent section fairing reached its maximum span. At 70 mm above the keel, the line briefly turned outward by 0.506° across adjoining 50 mm spans. The v0.2.2 hull keeps turning inward through both the 0.40–0.70 m stem blend and the 0.70–1.02 m span-cap interval across all nine waterlines.

![Forward waterline tangent change](entry-waterline.png)

## Geometry and controls

`hull/kayak-10-9.hull.json` sets `distribution_v2.surface_fairing.target_beam_m = 0.500456`. The design net has 441 longitudinal stations and 129 equal-girth samples per half section. The least-squares cubic surface uses 12 longitudinal and 22 transverse spans. Excluding singular stem tips within 220 mm of the ends (`0.88 * end_preserve_m`, where `dy/dx` diverges on the raked bow and stern) removes the +0.322° entry ripple near 0.48 m aft of the bow and the 3.45° -> 3.63° sharpening on the 75 mm aft buttock observed in the initial 10-span fit (`7b5f2d2`, STL `50cfb370...`). The fit tapers into the original keel and deck seam across the first and last 12% of half-girth, retains the stem treatment in the first and last 250 mm, and blends over the next 350 mm. One symmetric breadth adjustment constrains the 4-inch beam to 500.456 mm. Output tessellation, `kayakgen build-export` station offsets and molds, the outfitted assembly, finished station DXF, and all H2S/Voron female mold builders sample the same net.

The deck and low aft coaming retain their existing geometry (`24.53 cm` above keel / `15.72 cm` above the 225 lb waterline, `+3.70 mm` proud of the `x = +0.50 m` rear deck). Through the strongly raked bow, finer tessellation moves the exported centreplane outline by up to 0.103 mm in nearest-profile distance. Use the finished outer shell (`fcbf0bdb...`) for sculpted-end hull measurements.

## What changed

- Nine waterlines, five buttocks, five 45° diagonals and 23 transverse cuts cover the hull. The lines sheet includes normal-based reflection bands. Turn is the signed angle between adjoining 50 mm chords measured along X. The section comb uses a three-point circle with approximately 15 mm spacing along girth. Across 0.70–1.02 m aft of the bow, signed waterline turn stays between -0.033° and -0.129°. At the 75 mm buttock near the stern, peak change of turn over 50 mm is 3.46° (versus 3.45° in v0.2.1 and 3.63° in `7b5f2d2`).
- At the four earlier forebody test stations (`-2.10`, `-1.65`, `-1.35` and `-0.95 m`), peak transverse curvature change on the finished mesh is 118–190 m⁻². The corresponding v0.2.1 values are 132–203 m⁻² and the v0.1.2 values are 599–724 m⁻². Both directions of the sampled surface-distance comparison against v0.2.1 stay below 0.50 mm (`p95` 0.093 and 0.113 mm; maximum `v0.2.1 -> v0.2.2` distance 0.359 mm on the forward bilge, `v0.2.2 -> v0.2.1` 0.490 mm at the stern tip).
- At 4 inches the finished length and beam are 5442.889 and 500.460 mm (`SR_4in = 10.88`). Seawater displacement at that height changes from 125.989 to 126.021 kg. At equal loads of 225, 265 and 300 lb, maximum draft change relative to v0.2.1 is 0.017 mm. With KG 250 mm above the keel, righting arms change by at most 0.009 mm at 10°, 20° and 30°. The largest upper-beam change in the 225 lb catch measurements is 0.239 mm.

[Full hull lines and reflections](hull-lines.png) · [Surface displacement map](displacement.png) · [Cut-by-cut readings and STL hashes](fairness.json) · [Mesh-distance readings](deviation.json) · [Load cases and comparison tolerances](controls.json)

## Reproduction

Extract each tagged release's outer-shell STL from its 3D assemblies archive. Keep the files under separate paths. From the repository root run the following commands in the CAD environment.

```sh
python3 cad/build_finished.py
python3 cad/inspect_fairness.py v0.1.2=old-012.stl v0.2.1=old-021.stl v0.2.2=build/finished/kayak-10-9-outer-shell.stl --out build/fairness
python3 cad/check_fairing_controls.py old-021.stl build/finished/kayak-10-9-outer-shell.stl
python3 -m unittest discover -s tests -v
```

The load check balances displaced volume and longitudinal moment on the inclined mesh with free trim at heel, seawater density 1025 kg/m³ and zero transverse CG. It clips the upper catch contour at 100 mm above the loaded waterline and measures widths at the upright LCG and 0.8–1.2 m forward in 0.1 m steps. The comparison limits are 1 mm for 4-inch dimensions and loaded draft, 2 mm for catch widths and 0.5 mm for righting arms. The evidence concerns the digital hull.

This run used Python 3.12.3, kayak-gen revision `b67aa6c`, NumPy 2.5.4, SciPy 1.18.1, trimesh 5.2.0, VTK 9.7.2, PyVista 0.49.2, Matplotlib 3.11.2, ezdxf 1.4.4, NetworkX 3.7 and Shapely 2.2.0.
