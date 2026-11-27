# Free-surface panel solver: method and validation

`rsi/panel_solver.py` computes the wave resistance, dynamic sinkage and trim of a hull at steady speed in calm water with a Rankine-source panel method. It is a third opinion beside the two tiers the RSI harness already has:

- Tier 1 wave drag is Michell's thin-ship integral (`rsi.evaluator._michell_fast_curve`). It assumes a hull so thin that its boundary condition can be applied on the centreplane.
- Tier 2 is paired OpenFOAM CFD (`rsi/paired_cfd.py`). It has the full physics, but its reading of one hull pair moved by 0.35 N with the way the run was settled.

The panel solver applies the hull condition on the real hull surface, so it does not share Michell's assumption, and it has no time stepping and no random element: the same mesh and settings give the same number. What it has instead is discretisation error, whose sensitivity to the choice of grid is measured below.

It is a diagnostic. It is not wired into `rsi/gates.py`, the campaign loop or promotion.

```sh
kayak109-rsi panel-wave --fixed --hull CANDIDATE.hull.json --baseline hull/kayak-10-9.hull.json --speed-kn 5.0 --speed-kn 5.5 --speed-kn 6.2
```

## Method, and what is linearised

The module docstring is the full statement. In short:

- Constant-strength Rankine source panels on the wetted hull below the still waterplane and above a patch of that waterplane; half model with a centreplane image. The hull is re-panelled from its triangle mesh at stations gathered towards the waterline ends.
- **Linearisation: Dawson's double-body linearisation** (default). The flow under a rigid lid is solved first; the wave part is solved with the free-surface condition linearised about that flow and applied on the still waterplane `z = 0`. The alternative `--linearisation kelvin` linearises about the uniform stream (Neumann-Kelvin). Dawson's form was chosen because it lets the hull's own flow field carry the waves, which is the part Michell leaves out; the Kelvin form is kept because in the thin-hull limit it must reproduce Michell exactly, which gives a check with a known answer.
- Radiation condition: Dawson's four-point upstream difference. Free-surface panels are raised above the waterplane and the collocation points are shifted half a panel upstream.
- Wave resistance is the hull pressure integral, with the same integral for the double-body flow subtracted. Two more numbers are reported beside it and are not part of it: the waterline term `rho g sum(eta^2 dy)` and, on request, a control-surface (momentum flux) estimate.
- Sinkage and trim: the pose is iterated to force and moment balance at fixed displacement and LCG, re-panelling the hull at each pose.

Linearised or left out: the free-surface condition is linear and applied at `z = 0`, not on the wave surface; hull panels stop at `z = 0`, so the wetted surface does not follow the wave profile; no breaking or spray; no viscosity, so no boundary layer and no stern separation; deep unbounded water. The pose is not linearised. Hulls with an immersed transom or a bulb are rejected.

## Summary of what was found

Verified by the runs below:

- In the thin-hull limit, where Michell's integral is the exact answer, the solver approaches it from below as the grid is refined and is still 5 to 12 % short at 80 stations. Convergence to Michell is the trend, not something these runs reach.
- On the fixed Wigley hull the solver is within 7 % of an independent linear Rankine-panel calculation at seven Froude numbers. It differs from four measured residuary coefficients by 1 to 17 % and exceeds two wake-survey wave coefficients by 6 and 20 %.
- Free to sink and trim, sinkage on the Wigley hull is 8 to 12 % below the experiments and **trim is biased bow up by up to 0.4 % of length**. This matters for the kayak; see "Limitations".
- Across seven grids the baseline kayak's fixed-pose wave resistance has a sample standard deviation of 0.092, 0.132 and 0.233 N at 5.0, 5.5 and 6.2 kn (5.8, 4.4 and 4.1 % of the respective means).
- For candidate minus baseline, the largest sample standard deviation among the three finalists is 0.006, 0.009 and 0.014 N at those speeds. The seven grids vary both longitudinal stations and transverse rows. Their spread measures sensitivity within this grid family. Halving the near-hull free-surface row width reverses one fixed-pose comparison at 5.5 kn and moves it by 0.0397 N at 6.2 kn. Accuracy against physical kayak resistance remains unmeasured.
- **Paired differences free to sink and trim are not resolved.** On two grids the same pair read +0.031 and -0.007 N at 5.5 kn, and +0.065 and +0.125 N at 6.2 kn.
- Every finalist shows a fixed-pose wave-drag reduction on all seven grids in the station-count study at 5.5 and 6.2 kn. The seven-grid means are 22 to 37 % of the corresponding Michell reductions. Adding Tier 1 friction gives both signs on the seven grids for each finalist at both speeds. At 5.0 kn the same fixed-pose calculation gives a drag increase in all 21 cases. Free-running total drag remains unresolved.

## Verification against theory: the thin-hull limit

With the uniform stream as base flow (`kelvin`) and a hull thin enough, the panel solver and Michell's integral solve the same problem, so Michell is the exact answer. Wigley form with `B/L = 0.005`, `T/L = 0.0625`; entries are panel wave resistance over Michell (`rsi.evaluator._michell_fast_curve`, the Tier 1 routine).

| Grid | Panels (hull, surface) | Fn 0.25 | Fn 0.30 | Fn 0.35 | Fn 0.40 | Fn 0.45 |
|---|---|---|---|---|---|---|
| stations 40, rows 24 | 320, 2160 | 0.711 | 0.854 | 0.878 | 0.911 | 0.921 |
| stations 60, rows 36 | 480, 4680 | 0.830 | 0.895 | 0.915 | 0.931 | 0.944 |
| stations 80, rows 48 | 640, 8160 | 0.884 | 0.913 | 0.933 | 0.941 | 0.954 |
| stations 60, rows 36, girth panels 16 | 960, 4680 | 0.839 | 0.905 | 0.926 | 0.940 | 0.952 |
| stations 60, rows 36, end clustering 0.0 | 480, 4320 | 0.850 | 0.897 | 0.917 | 0.930 | 0.943 |
| stations 60, rows 48, half width 0.8, ahead of the bow 0.8 | 480, 7104 | 0.831 | 0.898 | 0.918 | 0.927 | 0.945 |

The ratio rises with station count (0.71, 0.83, 0.88 at Fn 0.25; 0.92, 0.94, 0.95 at Fn 0.45) and with girth panels, and changes less when the free-surface extent or end clustering changes on the grids tested. At 80 stations it is 5 to 12 % below Michell, worst at the lowest speed, where the waves are shortest. Read with the Wigley results below, the solver on its reference grids probably under-reads wave resistance by several percent. The next section checks the Tier 1 Michell routine against two published calculations for the standard Wigley hull.

## Wigley hull: comparison with published data

Hull: `y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2)`, `L/B = 10`, `B/T = 1.6`. Coefficients are `Cw = Rw / (0.5 rho U^2 S)` on the wetted surface at rest; the panelled value is `S = 0.14870 L^2`, against 0.1487 in [S2] and `0.661 L (2T + B) = 0.1487 L^2` in [S1].

### Sources

Every reference number below was read from the source named, in this session. Nothing is quoted from memory.

- **[S1]** Kajitani, Miyata, Ikehata, Tanaka, Adachi, Namimatsu, Ogiwara (1983), "The Summary of the Cooperative Experiment on Wigley Parabolic Model in Japan", 2nd DTNSRDC Workshop on Ship Wave-Resistance Computations, DTIC AD-P003 037. <https://apps.dtic.mil/sti/pdfs/ADP003037.pdf> (read from the Internet Archive copy of that file). No DOI. Resistance, sinkage and trim are **figures only** (Figs. 1, 2 and 4). The values quoted from it here were read off 250 dpi renders of a poor scan: about ±0.07e-3 in Cw, ±0.004 in sinkage coefficient, ±0.03 % L in trim. Treat them as approximate.
- **[S2]** Shahshahan and Landweber (1986), “Interactions Between Wavemaking and the Boundary Layer and Wake of a Ship Model”, IIHR Report 302, University of Iowa, DTIC ADA170160. Table 6 (printed p. 122) gives the fixed-model residuary coefficient `Cr = Ct - C(ITTC-57)` and Michell values at four Froude numbers, plus the wake-survey wave coefficient `Cwv` at two. L = 3.048 m. The authors note tank blockage above Fn 0.30. [Report DOI](https://doi.org/10.21236/ada170160)
- **[S3]** Peri (2026), "SIREN: A nonlinear Rankine-source boundary element solver for the stationary motion of a marine vehicle in calm water", arXiv:2610.14027v1. <https://arxiv.org/abs/2610.14027>. **Tabulated**: Table 3 gives a linearised Rankine-panel `Cw` at fixed attitude on three grids, and Michell values from the author's own script. A recent preprint, not peer reviewed; used here as an independent code of the same class, not as truth.

Looked for and not obtained in tabulated, open form: Chen and Noblesse (1983, J. Ship Research 27, doi:10.5957/jsr.1983.27.4.215, paywalled), McCarthy (1985, DTNSRDC-85/011), Bai and McCarthy (1979), Dawson (1977), Raven's thesis (1996; opened, no Wigley tables found in the parts read). No tabulated experimental sinkage or trim was found anywhere.

### Hull fixed at its design pose

Grid: the solver's defaults with 60 stations, 36 rows and 10 girth panels (600 hull and 4680 free-surface panels). `Cw x 10^3`.

| Fn | Experiment, Cr fixed [S2] Table 6 | Experiment, Cw fixed, SRI 4 m model [S1] Fig. 2, digitised | Michell [S2] / [S3] | Michell, Tier 1 routine | Linear Rankine panel [S3] Table 3, fine grid | This solver, Dawson | This solver, Kelvin |
|---|---|---|---|---|---|---|---|
| 0.250 | | 0.64 to 0.71 | – / 1.064 | 1.065 | 0.995 | 0.939 | 0.909 |
| 0.266 | 0.77 | 0.69 | 0.972 / – | 0.944 | | 0.897 | 0.938 |
| 0.267 | | 0.69 | – / 0.976 | 0.976 | 0.855 | 0.911 | 0.953 |
| 0.289 | | 1.06 | – / 1.987 | 1.988 | 1.435 | 1.431 | 1.412 |
| 0.300 | | 1.25 to 1.32 | | 2.143 | | 1.590 | 1.499 |
| 0.313 | 1.46 | 1.45 | 1.912 / – | 1.919 | | 1.584 | 1.459 |
| 0.316 | | 1.47 | – / 1.831 | 1.833 | 1.657 | 1.560 | 1.442 |
| 0.350 | 1.44 | 1.30 | 1.249 / – | 1.249 | | 1.281 | 1.328 |
| 0.354 | | | – / 1.284 | 1.285 | 1.294 | 1.289 | 1.335 |
| 0.400 | 2.12 | 2.11 to 2.16 | 2.737 / – | 2.736 | | 2.141 | 2.275 |
| 0.408 | | | – / 3.030 | 3.032 | 2.381 | 2.360 | 2.500 |
| 0.450 | | | – / 4.154 | 4.157 | 3.367 | 3.293 | 3.106 |

Reading it:

- Against [S3]'s linear Rankine code: -5.6, +6.5, -0.3, -5.9, -0.4, -0.9 and -2.2 % at Fn 0.250, 0.267, 0.289, 0.316, 0.354, 0.408 and 0.450.
- Against the tabulated residuary coefficients `Cr` in [S2], Dawson gives +16.5, +8.5, -11.0 and +1.0 % at Fn 0.266, 0.313, 0.350 and 0.400. Those measurements include form drag. The separate wake-survey wave measurements `Cwv × 10³` are 0.750 at Fn 0.266 and 1.490 at Fn 0.313. Dawson exceeds them by 19.6 and 6.3 % respectively. Table 6 leaves the other two wake-survey entries blank.
- Michell is 26 to 39 % above the panel result at the humps (Fn 0.289, 0.30, 0.40 to 0.45) and equal to it in the hollow at 0.35. Both panel codes and the experiments agree on that. It is the known thin-ship overshoot for a hull of this fullness, not a defect of either tier.
- The Tier 1 Michell routine agrees with the published Michell columns to within 0.2 % except at two [S2] points. It is 2.9 % lower at Fn 0.266 and 0.35 % higher at Fn 0.313.

### Other estimates of the same force

`Cw x 10^3`, Dawson, same grid for the first three columns; the control-surface estimate needs a longer and wider free surface (60 stations, 48 rows, half width 0.8 L, 1.0 L behind the stern).

| Fn | Hull pressure (the reported value) | Hull pressure with pressure linearised | Hull pressure + waterline term | Hull pressure on the wider grid | Control surface on the wider grid |
|---|---|---|---|---|---|
| 0.30 | 1.590 | 1.678 | 1.893 | 1.600 | 1.338 |
| 0.35 | 1.281 | 1.406 | 1.650 | 1.296 | 1.074 |
| 0.40 | 2.141 | 2.261 | 2.463 | 2.171 | 1.957 |
| 0.45 | 3.293 | 3.389 | 3.452 | 3.236 | 3.198 |

- On the wider grid the control-surface estimate falls below hull pressure by 1 % at Fn 0.45 and by 10 to 17 % at Fn 0.30 to 0.40. Wave damping and limited grid resolution could contribute. The sign of the error against the physical force is unknown.
- Adding the waterline term moves the result away from both the experiments and the control-surface estimate. The linear momentum balance says hull pressure plus this term equals the wave flux, and the numbers do not bear that out, so the term is reported and left out of `rw_N`. This choice was made on the Wigley hull before any kayak hull was run.

### Free to sink and trim

Centre of gravity at the still waterplane, towing force through it ([S1] towed 0.004 to 0.021 L below the waterplane). Experimental values are the digitised [S1] Fig. 1 and Fig. 4 points for the SRI 4 m model (UT 2.5 m model in brackets). Sinkage coefficient is `2 g s / U^2` with `s` the mean sinkage; trim is `(dA - dF)/L` in percent, positive bow up.

| Fn | Cw x 10^3, free | Experiment Cw x 10^3, free [S1] | Sinkage coefficient | Experiment [S1] | Trim, % L | Experiment [S1] | Pose iterations |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.954 | 0.80 (0.89) | 0.043 | 0.049 (0.044) | +0.01 | -0.06 (-0.02) | 3 |
| 0.30 | 1.673 | 1.51 | 0.046 | 0.051 (0.048) | +0.11 | -0.04 (0.00) | 4 |
| 0.35 | 1.346 | 1.56 (1.65) | 0.049 | 0.053 (0.054) | +0.20 | -0.10 (0.00) | 4 |
| 0.40 | 2.532 | 2.56 (2.73) | 0.056 | 0.062 (0.063) | +1.18 | +0.81 (+0.93) | 6 |
| 0.45 | rejected | | | | | | |

- At Fn 0.45 the run stopped with the solver's own refusal: at the computed bow-up trim the foot of the Wigley hull's vertical stem lies ahead of the waterline end, which the panelling does not support. No free result exists at that speed.
- Sinkage is 8 to 12 % below the SRI points.
- Trim is too far bow up at every speed: by 0.03 to 0.15 % L at Fn 0.25 to 0.30, 0.2 to 0.3 % L at 0.35, 0.25 to 0.37 % L at 0.40. The experiments show the bow going slightly down below Fn 0.36; the solver does not.
- Free minus fixed `Cw` at Fn 0.40 is +18 % here, against +19 to +21 % between the digitised free and fixed SRI points.

## Convergence

### Wigley hull, fixed, Dawson

One thing changed at a time about 60 stations, 36 rows, 10 girth panels. `Cw x 10^3`.

| Change | Panels (hull, surface) | Fn 0.30 | Fn 0.35 | Fn 0.40 | Fn 0.45 | Assembly, s | Solve per speed, s |
|---|---|---|---|---|---|---|---|
| stations 40, rows 24 | 400, 2160 | 1.565 | 1.243 | 2.138 | 3.269 | 7 | 0.3 |
| stations 50, rows 30 | 500, 3300 | 1.582 | 1.264 | 2.134 | 3.294 | 19 | 1.1 |
| reference grid | 600, 4680 | 1.590 | 1.281 | 2.141 | 3.293 | 44 | 3.6 |
| stations 80, rows 48 | 800, 8160 | 1.599 | 1.292 | 2.148 | 3.303 | 148 | 12.0 |
| girth panels 6 | 360, 4680 | 1.573 | 1.272 | 2.120 | 3.260 | 35 | 3.2 |
| girth panels 16 | 960, 4680 | 1.596 | 1.284 | 2.146 | 3.301 | 46 | 3.6 |
| rows 48 | 600, 6240 | 1.590 | 1.280 | 2.140 | 3.295 | 67 | 5.2 |
| half width 0.8, rows 48 | 600, 6240 | 1.592 | 1.283 | 2.152 | 3.298 | 74 | 4.8 |
| ahead of the bow 0.8 | 600, 5328 | 1.596 | 1.282 | 2.124 | 3.287 | 56 | 4.4 |
| behind the stern 1.0 | 600, 5760 | 1.598 | 1.290 | 2.167 | 3.232 | 64 | 4.4 |
| end clustering 0.0 | 600, 4320 | 1.592 | 1.284 | 2.135 | 3.287 | 32 | 2.6 |
| end clustering 0.5 | 600, 4464 | 1.595 | 1.285 | 2.134 | 3.290 | 29 | 1.8 |
| panels raised 1.0 | 600, 4680 | 1.592 | 1.275 | 2.136 | 3.299 | 36 | 2.7 |
| collocation shift 0.25 | 600, 4680 | 1.589 | 1.283 | 2.139 | 3.292 | 42 | 3.5 |

Every change moves the result by less than 2 %, and the station count moves it monotonically (+2 to +4 % from 40 to 80 stations).

### Kayak baseline at its static pose (225 lb, level)

`hull/kayak-10-9.hull.json` (v0.2.3), one thing changed at a time about the defaults (50 stations, 30 rows, 8 girth panels, free surface 0.5 L ahead and astern and 0.6 L to the side, end clustering 0.8). Wave resistance in newtons; the two right-hand columns are the best campaign hull (`iter137_cand004`) minus the baseline on the same grid rule.

| Change | Panels (hull, surface) | Baseline 5.5 kn | Baseline 6.2 kn | Difference 5.5 kn | Difference 6.2 kn |
|---|---|---|---|---|---|
| 40 stations, 24 rows | 320, 2160 | 3.047 | 5.494 | -0.0229 | -0.0363 |
| reference grid | 400, 3300 | 3.012 | 5.593 | -0.0170 | -0.0513 |
| 60 stations, 36 rows | 480, 4680 | 2.935 | 5.608 | -0.0397 | -0.0231 |
| 70 stations, 42 rows | 560, 6300 | 3.132 | 5.764 | -0.0130 | -0.0526 |
| 6 girth panels | 300, 3300 | 3.003 | 5.581 | -0.0166 | -0.0511 |
| 12 girth panels | 600, 3300 | 3.011 | 5.590 | -0.0169 | -0.0514 |
| 24 rows | 400, 2640 | 2.917 | 5.764 | -0.0190 | -0.0473 |
| 40 rows | 400, 4400 | 3.054 | 5.504 | -0.0180 | -0.0505 |
| half width 0.8 L, 40 rows | 400, 4400 | 3.020 | 5.601 | -0.0172 | -0.0509 |
| 0.8 L ahead of the bow | 400, 3750 | 2.999 | 5.581 | -0.0165 | -0.0513 |
| 1.0 L behind the stern | 400, 4050 | 3.052 | 5.553 | -0.0176 | -0.0520 |
| end clustering 0.7 | 400, 3180 | 3.069 | 5.535 | -0.0157 | -0.0533 |
| end clustering 0.9 | 400, 3480 | 2.956 | 5.659 | -0.0189 | -0.0476 |
| panels raised 1.0 (default 0.8) | 400, 3300 | 3.119 | 5.429 | -0.0161 | -0.0542 |
| collocation shift 0.25 (default 0.5) | 400, 3300 | 2.999 | 5.489 | -0.0248 | -0.0452 |
| Kelvin linearisation | 400, 3300 | 2.722 | 4.936 | -0.0229 | -0.0639 |

- Baseline resistance changes nonmonotonically as stations and transverse rows increase together. The rising keel near each stem may make the result sensitive to station placement. These runs vary several spacings together, so they cannot isolate the cause.
- Across the Dawson grids here the dynamic vertical force is −115 to −111 N at 5.5 kn and −155 to −148 N at 6.2 kn. The corresponding trimming moments are 84 to 89 N·m and 160 to 169 N·m.

## Paired differences: what the solver resolves

The same pair as above. With the station count fixed, nine changes of girth panels, rows, half width, upstream and downstream extent and end clustering move the difference between -0.0157 and -0.0190 N at 5.5 kn and between -0.0473 and -0.0533 N at 6.2 kn: a spread of ±0.002 and ±0.003 N. Errors common to the two hulls cancel well.

The table shows fixed-pose wave resistance differences in newtons, candidate minus baseline, on seven grids with 40, 45, 50, 55, 60, 65 and 70 stations. The number of transverse rows is 0.6 times the station count. Each standard deviation is the sample standard deviation across seven paired differences.

| Hull | 5.0 kn mean | std | range | 5.5 kn mean | std | range | 6.2 kn mean | std | range |
|---|---|---|---|---|---|---|---|---|---|
| iter137_cand004 | -0.0075 | 0.0059 | -0.0136 to +0.0016 | -0.0254 | 0.0091 | -0.0397 to -0.0130 | -0.0371 | 0.0140 | -0.0526 to -0.0225 |
| iter139_cand009 | -0.0053 | 0.0056 | -0.0107 to +0.0034 | -0.0237 | 0.0086 | -0.0368 to -0.0120 | -0.0375 | 0.0130 | -0.0513 to -0.0221 |
| iter139_cand002 | -0.0083 | 0.0055 | -0.0137 to -0.0000 | -0.0295 | 0.0080 | -0.0419 to -0.0184 | -0.0492 | 0.0118 | -0.0617 to -0.0358 |

The baseline itself over the same seven grids: 1.585 ± 0.092, 2.998 ± 0.132 and 5.658 ± 0.233 N.

**Across the three finalists, the largest seven-grid sample standard deviations are 0.006, 0.009 and 0.014 N at 5.0, 5.5 and 6.2 kn.** The widest ranges for an individual finalist are 0.015, 0.027 and 0.030 N. At 5.5 and 6.2 kn all seven grids in this family predict lower wave resistance for each finalist. At 5.0 kn two finalists change sign with the grid and the third comes within 0.000006 N of zero. The largest standard deviations are 5 to 12 times smaller than the 0.07 N target. The independent near-hull spacing check below measures one effect this family missed.

An independent check varied the near-hull free-surface spacing while holding 50 stations, 8 girth panels, 30 rows and the static pose fixed. The first row controls the distribution of all transverse rows. Both hulls used the same settings. Values are for `iter137_cand004` minus baseline.

| First row (nominal station spacings) | Pressure difference at 5.5 kn (N) | Pressure difference at 6.2 kn (N) |
|---|---:|---:|
| 0.750 | −0.01703 | −0.05134 |
| 0.375 | +0.00194 | −0.09104 |

Changing this spacing reverses the pressure difference at 5.5 kn and shifts it by 0.0397 N at 6.2 kn, more than the entire seven-grid range of 0.0301 N. The narrower-row case also extrapolated the waterline elevation. That option affects only the separately reported waterline force. Including it gives +0.00891 and −0.08128 N on the narrower grid. The `paired` validation study includes this control.

For comparison, changing the shift from 0.50 to 0.25 panel lengths changes the pressure difference by −0.00778 and +0.00614 N at 5.5 and 6.2 kn. Moving the end clustering between 0.7 and 0.9 changes the nominal difference by at most 0.00370 N across those speeds. Refining the input hull shell from 441 × 129 to 661 × 161 with the flow grid held at 40 × 24 and eight girth panels changes the paired pressure result by at most 0.00014 N across all three speeds. This separates the sensitivity to the flow grid from one test of hull tessellation.

The control-surface force integral gives another check on the difference. At 40 stations and 32 rows, with half-width 0.8 L and downstream extent 1.0 L, the 6.2 kn pressure difference is −0.03883 N. Pressure plus the waterline strip gives −0.04880 N and the control-surface integral gives −0.01608 N. All three use the same two computed flows. Agreement of their signs on this grid leaves a 0.03272 N spread in magnitude.

Free to sink and trim the difference is **not** resolved:

| Grid | Speed, kn | Baseline, N | Sinkage, mm | Trim, deg (bow up +) | Candidate trim, deg | Difference, N |
|---|---|---|---|---|---|---|
| reference | 5.5 | 4.180 | 6.78 | +0.2523 | +0.2551 | +0.0311 |
| reference | 6.2 | 9.401 | 9.16 | +0.5458 | +0.5495 | +0.0653 |
| 60 stations, 36 rows | 5.5 | 4.111 | 6.87 | +0.2594 | +0.2616 | -0.0071 |
| 60 stations, 36 rows | 6.2 | 9.453 | 9.16 | +0.5372 | +0.5417 | +0.1245 |

The computed wave resistance rises by 40 % at 5.5 kn and 68 % at 6.2 kn when the hull is released, almost all of it through trim (about 5 N per degree at 5.5 kn). A difference in trim of 0.003 degrees between two hulls is then worth 0.015 N, and the two grids disagree by 0.04 to 0.06 N on the difference. The pose itself converged on every run (6 or 7 flow solutions, tolerance 1 micrometre).

### Holding both hulls at the baseline’s running pose

On the 50-station, 30-row grid, impose the baseline’s computed sinkage and trim on both hulls relative to each hull’s own level static draft. The resulting wave resistance differences for `iter137_cand004` are candidate minus baseline.

| Speed (kn) | Baseline sinkage (mm) | Baseline trim (degrees bow up) | Both level (N) | Both at common sinkage and trim (N) | Each free (N) |
|---|---:|---:|---:|---:|---:|
| 5.5 | 6.7793 | 0.25233 | −0.01703 | +0.02389 | +0.03108 |
| 6.2 | 9.164 | 0.54579 | −0.05134 | +0.04600 | +0.06529 |

Changing the common pose reverses the predicted wave-drag difference at both speeds, even before either hull is allowed to find a different running trim. In the individually free column each hull carries its longitudinal centre of gravity at its own level centre of buoyancy. At 225 lb that position is x = 39.463 mm on the baseline and 45.347 mm on the finalist, a 5.884 mm shift aft. A comparison with the paddler and ballast kept at fixed stations requires a separate load case. The true running pose has not been measured. The level fixed-pose comparison therefore cannot confirm a 0.07 N drag gain underway.

To reproduce the middle force column, take `sinkage_m`, `trim_rad` and `rw_N` from the corresponding `free[0].baseline` record in `data/paired.json`. At 225 lb compute the candidate’s level height `h0` with `static_pose`. Pass `pose_triangles(v, f, h0 + sinkage_m, trim_rad)` to `solve_fixed` with `PanelConfig()`, seawater density 1025 kg/m³ and the corresponding speed in m/s. Subtract the stored baseline `rw_N`.

### Single precision

Reference grid, numpy, single against double precision (`precision="single"`): the baseline moved by -2.6e-5, -1.1e-4 and +7.6e-5 N at 5.0, 5.5 and 6.2 kn and the paired difference by +3e-6, +2.0e-4 and +1.6e-4 N. That is 350 times below 0.07 N and 40 times below the grid scatter, so single precision does not limit the paired difference at this size (3700 panels). Rounding error grows with the panel count, and this was not measured above 3700 panels.

## Cross-check with the Michell tier on the kayak

Baseline, 225 lb, newtons. Michell and friction are Tier 1's own (`rsi.panel_hull.tier1_parts`); the panel values are on 60 stations and 36 rows.

| Speed, kn | Fn | Michell (Tier 1) | Panel, fixed pose | Panel / Michell | Panel, free | Sinkage, mm | Trim, deg | Friction (Tier 1) |
|---|---|---|---|---|---|---|---|---|
| 5.0 | 0.352 | 2.051 | 1.660 | 0.81 | 2.095 | 5.08 | +0.106 | 19.914 |
| 5.5 | 0.388 | 4.703 | 2.935 | 0.62 | 4.111 | 6.87 | +0.259 | 23.707 |
| 6.2 | 0.438 | 8.660 | 5.608 | 0.65 | 9.453 | 9.16 | +0.537 | 29.522 |

- At the level static pose the kayak's panel resistance is 62 to 81 % of Michell. The fixed Wigley hull gives 78 to 79 % of Michell at Fn 0.40 to 0.45. At these speeds the kayak has waterline beam/draft about 5.4, compared with 1.6 for the Wigley hull. Michell’s thin-ship assumption is one possible contributor to the difference. Neither comparison isolates it from panel discretisation and other modelling choices.
- The Kelvin linearisation, which keeps Michell's free-surface condition and changes only the hull condition, gives 2.72 and 4.94 N at 5.5 and 6.2 kn, 0.58 and 0.57 of Michell. The double-body base flow accounts for the other 0.3 to 0.7 N.
- Free to sink and trim the panel value rises to 0.87 to 1.09 of Michell. Tier 1 holds the hull level at its static draft and so leaves this out; see the limitation on trim below before using these free values.

## Campaign v7 finalists against the baseline

All forces are in newtons at 225 lb, candidate minus baseline. A negative difference indicates less drag. The seven-grid columns use the mean and sample standard deviation above. Single-grid results use 60 stations and 36 transverse rows. The Tier 1 components are evaluated at each exact speed using the campaign formulas, including 6.2 kn between the campaign’s 0.25 kn grid points.

| Hull | Speed, kn | Michell wave (Tier 1) | Friction by position (Tier 1) | Tier 1 total | Panel wave, fixed pose, 7 grids | Panel wave, fixed, one grid | Panel wave, free, one grid | Panel fixed (7 grids) + friction |
|---|---|---|---|---|---|---|---|---|
| iter137_cand004 | 5.0 | -0.055 | +0.024 | -0.031 | -0.008 ± 0.006 | +0.002 | +0.023 | +0.016 |
| iter137_cand004 | 5.5 | -0.114 | +0.028 | -0.086 | -0.025 ± 0.009 | -0.040 | -0.007 | +0.003 |
| iter137_cand004 | 6.2 | -0.158 | +0.035 | -0.122 | -0.037 ± 0.014 | -0.023 | +0.125 | -0.002 |
| iter139_cand009 | 5.0 | -0.049 | +0.023 | -0.027 | -0.005 ± 0.006 | +0.003 | +0.020 | +0.018 |
| iter139_cand009 | 5.5 | -0.104 | +0.027 | -0.076 | -0.024 ± 0.009 | -0.037 | -0.019 | +0.004 |
| iter139_cand009 | 6.2 | -0.144 | +0.034 | -0.110 | -0.038 ± 0.013 | -0.027 | +0.083 | -0.004 |
| iter139_cand002 | 5.0 | -0.046 | +0.026 | -0.020 | -0.008 ± 0.006 | -0.000 | +0.007 | +0.018 |
| iter139_cand002 | 5.5 | -0.096 | +0.031 | -0.065 | -0.030 ± 0.008 | -0.042 | -0.056 | +0.001 |
| iter139_cand002 | 6.2 | -0.134 | +0.038 | -0.096 | -0.049 ± 0.012 | -0.039 | +0.006 | -0.011 |

- **Wave resistance.** At 5.5 and 6.2 kn all 42 fixed-pose differences are negative. For each finalist, its mean over seven grids is 22 to 37 % of its Michell reduction at those speeds. At 5.0 kn the corresponding means are 11 to 18 % of the Michell reductions.
- **Fixed-pose wave resistance plus Tier 1 friction.** The modelled friction penalties of 0.023 to 0.038 N give 5.0 kn drag increases of 0.010 to 0.026 N across all finalists and grids. At 5.5 and 6.2 kn each finalist’s range crosses zero. At 5.5 kn replacing Michell wave drag with the panel mean removes 0.067 to 0.089 N, slightly more than the entire 0.065 to 0.086 N Tier 1 net gain. The calculation supplies no free-running friction measurement.
- **Free to sink and trim** the single-grid differences range from -0.056 to +0.125 N and are within the unresolved band described above. They are listed for completeness and support no conclusion.
- **Ranking on matched grids.** The lowest mean panel wave resistance belongs to `iter139_cand002` at all three speeds. Including friction gives `iter137_cand004` the lowest mean at 5.0 kn. At 6.2 kn the fixed-pose estimate including friction favors `iter139_cand002` over `iter137_cand004` by 0.0047 to 0.0129 N and over `iter139_cand009` by 0.0058 to 0.0091 N on the same seven grids. The panel results establish this ordering within the sampled grid family only.
- The hull mass changed from 18 to 13.6 kg at the same 225 lb total while this was being run. Displacement, static draft and waterline are unchanged by that. The height of the centre of gravity enters only the free runs' moment balance; it was left at the evaluator's `KG_M = 0.25 m` above the keel, since no new value was given.

## Timings and memory

Kayak baseline, three speeds at one pose (one assembly, three solves), wall time. The host was heavily loaded by other work throughout (load average 15 to 27 on 12 threads), so the CPU times are pessimistic; assembly used 4 threads.

| Grid | Panels | Precision | numpy, 4 threads | CuPy, RTX 3090 | GPU memory pool |
|---|---|---|---|---|---|
| 40 stations, 24 rows | 2480 | double | 13.9 s (assembly 11.3, solve 0.76 per speed) | 0.80 s (assembly 0.66, solve 0.04 per speed) | 408 MiB |
| 50 stations, 30 rows (default) | 3700 | single | 17.4 s (assembly 12.9, solve 1.2 per speed) | 1.62 s (assembly 1.53, solve 0.03 per speed) | 450 MiB |
| 50 stations, 30 rows (default) | 3700 | double | 20 to 25 s | did not fit in the free memory | about 0.9 GiB needed |
| 60 stations, 36 rows | 5160 | double | 45 to 50 s | not run | about 1.8 GiB |
| 70 stations, 42 rows | 6860 | double | about 90 s | not run | about 3.1 GiB |

- A hull free to sink and trim needs 6 or 7 flow solutions per speed: 3.5 to 4.5 minutes per speed on the CPU at 60 stations, 10 to 13.5 minutes for the three speeds.
- The GPU is 11 to 17 times faster than four loaded CPU threads at the sizes that fit. It is worth having: measuring variation across the seven grids costs seven fixed-pose runs per hull, and the free runs seven solves per speed.
- GPU memory: about 66 bytes per panel squared in double precision (measured: 408 MiB at 2480 panels), half that in single, plus roughly 0.3 GiB of CUDA context. The card had 0.93 GiB free during this work. A double-precision run at 60 stations needs about 2.1 GiB in all, at 70 stations about 3.4 GiB, and at 100 stations (about 14 000 panels) about 12.5 GiB. The control-surface estimate adds about 100 kB per panel. **No full-size double-precision GPU run has been made.**
- The CuPy backend gives the same wave resistance as numpy to 1e-9 relative in double precision (`tests/test_panel_solver_gpu.py`, run in a separate environment with CuPy 14.2.0 on CUDA 12.8: 4 passed). CuPy is not installed in the repository's environment, where those tests skip.

## Limitations and failed checks

- **Trim is biased bow up**, by up to 0.4 % of length on the Wigley hull against experiment. On the kayak the computed trim is 0.19 % L at 5.0 kn, 0.45 % at 5.5 kn and 0.94 % at 6.2 kn, and the computed rise in wave resistance on release (26, 40 and 68 %) comes almost entirely from it. The bias on the Wigley hull is of the same size as the kayak's trim at 5.0 and 5.5 kn. The free-running absolute values for the kayak are therefore not validated, and the Wigley comparison gives a reason to expect them to be too high. Potential flow has no boundary layer to relieve the pressure recovery at the stern, and the towing force is assumed to act through the centre of gravity.
- **Free-running paired differences are unresolved** (above).
- **Fixed-pose paired differences** have sample standard deviations up to 0.006 to 0.014 N depending on speed in this sweep. The grid family can share discretisation and model bias.
- **Absolute wave resistance of the baseline kayak** has a grid-to-grid standard deviation of 4.1 to 5.8 % of its mean across the three speeds. Refinement gives a nonmonotonic sequence.
- **The control-surface estimate** differs from computed Wigley hull pressure by 10 to 17 % at Fn 0.30 to 0.40 and by 1 % at Fn 0.45. Its error relative to the physical force remains unknown.
- **The waterline term** is omitted from reported resistance on the evidence of the Wigley comparison. On the 60-station grid its candidate-minus-baseline change has magnitude up to 0.0111 N at a fixed pose. In the free-pose comparison it reaches −0.0555 N for `iter137_cand004` at 6.2 kn.
- **Linear free surface.** At 6.2 kn the computed bow wave beside the hull is about 20 mm against a draft of 89 mm. Nothing here measures what the linearisation costs on this hull. [S3] reports that iterating to the nonlinear condition changed the Wigley `Cw` by 1 to 9 % on its coarse grid.
- **Experimental validation** covers the Wigley hull, whose beam is 1.6 times its draft. The kayak’s waterline beam is about 5.4 times its draft. These studies contain no measured kayak resistance.
- Reference data: the experimental `Cw`, sinkage and trim from [S1] are digitised from figures. Only [S2] Table 6 and [S3] are tabulated.
- The solver was first written with free-surface panels in the waterplane and collocation at their centres (Dawson's original arrangement). That version changed its answer by up to 10 % with the parity of the number of columns ahead of the bow. It was replaced by raised panels with a half-panel collocation shift; `tests/test_panel_solver.py` guards against the regression.

## Reproduction

```sh
OMP_NUM_THREADS=4 python scripts/panel_validation.py precision paired finalists ensemble wigley-fixed wigley-free wigley-convergence thin-ship
python -m pytest tests/test_panel_solver.py            # numpy backend
python -m pytest tests/test_panel_solver_gpu.py        # needs the gpu extra (CuPy) and a CUDA device
```

The original validation outputs are in `docs/panel-solver/data/*.json`, each with the versions and time of its run. The common-pose control above uses the archived baseline and one additional candidate solve per speed. The finalist hulls are read from the campaign directory `~/.cache/kayak-10-9/campaigns/k109_rsi_v7_localfriction/` (`--campaign` to change it); they are not in the repository. The runs used Python 3.12.3, NumPy 2.5.5, SciPy 1.18.1 and, for the GPU rows, CuPy 14.2.0 (cuda12x) with CUDA 12.8 on driver 615.71.10.
