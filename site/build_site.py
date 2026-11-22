"""Build the self-contained GitHub Pages site (_site/) for Kayak 10.9."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
HULL_DIR = REPO_ROOT / "hull"
SITE_DIR = REPO_ROOT / "site"
REL_BASE = "https://github.com/halbritt/kayak-10-9/releases/latest/download"
REPO_URL = "https://github.com/halbritt/kayak-10-9"
KAYAKGEN_URL = "https://github.com/halbritt/kayak-gen"

SVG_CALIPER = """
<svg viewBox="0 0 760 225" width="100%" style="background:#f8fbff;border:1px solid #d6e0ea;border-radius:8px;margin-top:0.6rem">
  <text x="380" y="24" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f1e2e">Twin Foil-Track Seat Caliper &amp; Rear Shoulder-Bushing Schematic (Side View)</text>
  <rect x="65" y="152" width="630" height="32" rx="3" fill="#1e293b" stroke="#0f172a"/>
  <rect x="85" y="160" width="590" height="16" fill="#e2e8f0"/>
  <text x="380" y="203" text-anchor="middle" font-size="11" fill="#334155" font-weight="600">12.6&quot; (320 mm) Solid Carbon Hydrofoil Track Bonded Outboard @ y = &plusmn;12.5 cm (Zero Galvanic Corrosion, Clear Center Keel)</text>
  <path d="M 145,134 L 220,134 Q 370,142 545,126 L 615,126 L 615,140 L 545,140 Q 370,152 220,146 L 145,146 Z" fill="#0ea5e9" fill-opacity="0.28" stroke="#0369a1" stroke-width="2"/>
  <text x="385" y="122" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0c4a6e">Carbon Seat Pan Outboard Wing (Center Belly Drops Between Tracks to 10 mm Above Keel)</text>
  <rect x="160" y="146" width="44" height="6" fill="#dc2626"/>
  <rect x="162" y="162" width="40" height="12" rx="2" fill="#64748b" stroke="#0f172a"/>
  <rect x="178" y="102" width="8" height="72" fill="#475569"/>
  <rect x="164" y="92" width="36" height="12" rx="3" fill="#0f172a"/>
  <text x="182" y="83" text-anchor="middle" font-size="11" font-weight="700" fill="#991b1b">FRONT: 316/Ti Clamping T-Nut</text>
  <text x="182" y="69" text-anchor="middle" font-size="10" fill="#475569">+ 1 mm EPDM Shim (520+ lbf Caliper Lock)</text>
  <rect x="548" y="161" width="52" height="14" rx="2" fill="#f8fafc" stroke="#0d9488" stroke-width="2"/>
  <rect x="569" y="138" width="10" height="24" fill="#94a3b8" stroke="#0f172a"/>
  <rect x="571" y="124" width="6" height="48" fill="#334155"/>
  <rect x="565" y="119" width="18" height="6" rx="1" fill="#0f172a"/>
  <text x="574" y="83" text-anchor="middle" font-size="11" font-weight="700" fill="#0f766e">REAR: Delrin Slider + SS Shoulder Sleeve</text>
  <text x="574" y="69" text-anchor="middle" font-size="10" fill="#475569">+0.2 mm Clearance (Glides Free, Zero Roll Uplift)</text>
</svg>
"""


def make_svg_trim(bow_l: float, pod_l: float, stern_l: float) -> str:
    return f"""
<svg viewBox="0 0 760 195" width="100%" style="background:#f8fbff;border:1px solid #d6e0ea;border-radius:8px;margin-top:0.6rem">
  <text x="380" y="24" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f1e2e">+40 lb Baja / Watermaker Trim-Balance Vector Diagram (265 lb Total, 2 Bulkheads + Carbon Day Pod, 0.00&deg; Trim)</text>
  <line x1="60" y1="112" x2="700" y2="112" stroke="#0d9488" stroke-width="1.6" stroke-dasharray="6,4"/>
  <path d="M 80,96 Q 380,82 680,104 Q 380,136 80,110 Z" fill="#dbeafe" stroke="#1e3a8a" stroke-width="2"/>
  <line x1="272" y1="90" x2="272" y2="123" stroke="#b91c1c" stroke-width="2.6"/>
  <line x1="432" y1="90" x2="436" y2="124" stroke="#b91c1c" stroke-width="2.6"/>
  <rect x="448" y="90" width="28" height="17" rx="4" fill="#ccfbf1" stroke="#0f766e" stroke-width="1.6" stroke-dasharray="3,2"/>
  <rect x="489" y="89" width="6" height="7" fill="#0f766e"/>
  <line x1="210" y1="52" x2="210" y2="96" stroke="#1d4ed8" stroke-width="3"/>
  <polygon points="210,102 205,92 215,92" fill="#1d4ed8"/>
  <text x="210" y="45" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">Bow Oval: 16.0 lb</text>
  <text x="210" y="143" text-anchor="middle" font-size="10" fill="#475569">x &lt; -0.95 m ({bow_l:.1f} L)</text>
  <line x1="462" y1="62" x2="462" y2="94" stroke="#0f766e" stroke-width="3"/>
  <polygon points="462,100 457,90 467,90" fill="#0f766e"/>
  <text x="462" y="55" text-anchor="middle" font-size="11" font-weight="700" fill="#0f766e">Carbon Pod: 4.0 lb</text>
  <text x="462" y="143" text-anchor="middle" font-size="10" fill="#475569">x = +0.66 m ({pod_l:.1f} L)</text>
  <line x1="548" y1="48" x2="548" y2="98" stroke="#1d4ed8" stroke-width="3"/>
  <polygon points="548,104 543,94 553,94" fill="#1d4ed8"/>
  <text x="548" y="41" text-anchor="middle" font-size="11" font-weight="700" fill="#1e3a8a">Unified Stern: 20.0 lb</text>
  <text x="548" y="143" text-anchor="middle" font-size="10" fill="#475569">x &gt; +0.44 m ({stern_l:.1f} L)</text>
  <polygon points="383,128 375,144 391,144" fill="#d97706"/>
  <text x="383" y="162" text-anchor="middle" font-size="11" font-weight="700" fill="#b45309">LCB = LCG = +0.025 m (Heavy Stern Bags Slide Forward Against +0.44 m Seat Bulkhead)</text>
</svg>
"""


def make_master_table(specs: dict, sr_4in: float, b_oa: float, bow_l: float, pod_l: float, stern_l: float, tot_l: float) -> str:
    h3 = specs["hydro_3load"]
    b4 = float(h3["b_4in_cm"])
    usca = float(specs["USCA_4in_pct"])
    cb = h3["catch_beams_225_cm"]
    kn = h3["kn_225"]
    l225, l265, l300 = h3["load_225"], h3["load_265"], h3["load_300"]
    g225 = " / ".join(f"{v:.1f}" for v in l225["gz_10_20_30_40_mm"][:3])
    g265 = " / ".join(f"{v:.1f}" for v in l265["gz_10_20_30_40_mm"][:3])
    return f"""
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Parameter / Metric</th>
      <th>Kayak 10.9 (v0.2.1 Faired Forebody)</th>
      <th>Epic 18X / 18X Sport</th>
      <th>Stellar S18R G2</th>
      <th>Basis / Source</th>
    </tr>
  </thead>
  <tbody>
    <tr class="group-hdr"><td colspan="5">1. Principal Dimensions &amp; Race-Class Eligibility</td></tr>
    <tr><td>Overall Length (L_oa)</td><td class="hi">5.470 m CAD (5.480 m w/ gudgeon)</td><td>5.480 m (18&apos; 0&quot;)</td><td>5.480 m (17&apos; 11.8&quot;)</td><td><span class="badge b-mesh">3D Mesh + Maker</span></td></tr>
    <tr><td>Overall Max Beam (B_oa)</td><td class="hi">{b_oa:.2f} cm ({b_oa / 2.54:.2f}&quot;)</td><td>55.88 cm (22.00&quot;)</td><td>51.20 cm (20.16&quot;)</td><td><span class="badge b-mesh">3D Mesh + Maker</span></td></tr>
    <tr><td>4&quot; Caliper Beam (B_4in @ 10.16 cm)</td><td class="hi">{b4:.2f} cm ({b4 / 2.54:.2f}&quot;)</td><td>51.80 cm (scan)</td><td>~50.5 cm (derived)</td><td><span class="badge b-mesh">4&quot; Plane Slice</span></td></tr>
    <tr><td>Sound Rowers 4&quot;-Draft Ratio (&le; 10.99 FSK)</td><td class="hi">{sr_4in:.2f} (FSK Legal)</td><td>10.58 scan (10.16 SR list)</td><td>10.74 (SR Official List)</td><td><span class="badge b-mesh">SR Class Rule</span></td></tr>
    <tr><td>USCA 4&quot; Beam / L_oa (&ge; 8.50% Sea Kayak)</td><td class="hi">{usca:.2f}% (Pass)</td><td>9.45% (Pass)</td><td>~9.2% (USCA Listed)</td><td><span class="badge b-mesh">USCA Class Rule</span></td></tr>
    <tr><td>Max Depth / Bow / Rear Deck Above 225 lb WL</td><td class="hi">{h3["D_max_cm"]:.1f} cm / 24.6 cm / 14.8 cm</td><td>33.1 cm / 23.2 cm / 13.1 cm</td><td>32.0 cm OA (maker)</td><td><span class="badge b-mesh">3D Mesh + Maker</span></td></tr>
    <tr class="group-hdr"><td colspan="5">2. Layup Mass, Storage &amp; Cockpit Ergonomics</td></tr>
    <tr><td>Lightest Composite Weight (Ready to Paddle)</td><td class="hi">27.0 lb prepreg / 29.4 lb vac-bag</td><td>39.0 lb Ultra (41.7 lb Perf.)</td><td>34.2 lb Alpha (34.4 lb Excel)</td><td><span class="badge b-spec">BOM / Maker</span></td></tr>
    <tr><td>Sealed Dry Storage &amp; Bulkhead Layout</td><td class="hi">{tot_l:.1f} L ({bow_l:.1f} Bow + {pod_l:.1f} Carbon Pod + {stern_l:.1f} Stern; 2 BHs)</td><td>206.0 L (87 Bow + 36 Ctr Day + 83 Stern)</td><td>206.0 L (81 Bow + 125 Stern; NO Day Hatch)</td><td><span class="badge b-mesh">3D Volume / Maker</span></td></tr>
    <tr><td>Cockpit Inner Keyhole Opening</td><td class="hi">84.0 &times; 42.0 cm (recessed aft well)</td><td>85.1 &times; 40.6 cm (33.5&quot;&times;16.0&quot;)</td><td>85.1 &times; 43.9 cm (33.5&quot;&times;17.3&quot;)</td><td><span class="badge b-spec">CAD / Maker</span></td></tr>
    <tr><td>Catch Beam &ge; 10 cm Above WL (+0.8 / +1.0 / +1.2 m)</td><td class="hi">{cb["+0.8"]:.1f} / {cb["+1.0"]:.1f} / {cb["+1.2"]:.1f} cm</td><td>45.3 / 42.4 / 38.3 cm</td><td>Kayak Cut (~40–42 cm)</td><td><span class="badge b-mesh">3D Mesh Slice</span></td></tr>
    <tr><td>Wing-Knee Channel &amp; Thigh Control</td><td class="hi">31.2 cm open center + 4.1 cm side hooks</td><td>Standard touring rim + spine ridge</td><td>Open keyhole + cheek-hung seat</td><td><span class="badge b-spec">CAD / Maker</span></td></tr>
    <tr><td>Seat, Footbrace &amp; Keel Architecture</td><td class="hi">10 mm pan on twin &plusmn;12.5 cm carbon foil tracks + 12&deg; wing footboard (-0.82 m) + 60 mm flush UD keel strap</td><td>Center plastic notched rail + SmartTrack toe pegs</td><td>Cheek-bolted 3-pos seat + 3-cam QR full footplate</td><td><span class="badge b-spec">CAD / Maker</span></td></tr>
    <tr class="group-hdr"><td colspan="5">3. Calm-Water Speed &amp; Prior-Design OpenFOAM-v2512 Free-Surface RANS CFD (225 lb Seawater, Unvalidated Comparative)</td></tr>
    <tr><td>Speed @ 40 / 60 / 90 / 120 W (ITTC-57 + Michell)</td><td class="hi">{kn["40"]:.2f} / {kn["60"]:.2f} / {kn["90"]:.2f} / {kn["120"]:.2f} kn</td><td>4.47 / 5.02 / 5.58 / 6.06 kn</td><td>&mdash; (no 3D scan)</td><td><span class="badge b-cfd">Analytical</span></td></tr>
    <tr><td>OpenFOAM Total Drag @ 5.50 kn (Fn 0.386)</td><td class="hi">20.58 &plusmn; 0.24 N (-3.7%, 58.2 W)</td><td>21.38 &plusmn; 0.09 N (60.5 W)</td><td>&mdash;</td><td><span class="badge b-cfd">OpenFOAM RANS</span></td></tr>
    <tr><td>OpenFOAM Pressure/Wave Drag @ 5.50 kn</td><td class="hi">3.82 N (-23.5% wave drag)</td><td>5.00 N</td><td>&mdash;</td><td><span class="badge b-cfd">OpenFOAM RANS</span></td></tr>
    <tr><td>OpenFOAM Total Drag @ 6.20 kn (Fn 0.435)</td><td class="hi">26.02 &plusmn; 0.22 N (-4.2%, 83.0 W)</td><td>27.17 &plusmn; 0.07 N (86.7 W)</td><td>&mdash;</td><td><span class="badge b-cfd">OpenFOAM RANS</span></td></tr>
    <tr><td>OpenFOAM Pressure/Wave Drag @ 6.20 kn</td><td class="hi">5.30 N (-25.9% wave drag)</td><td>7.15 N</td><td>&mdash;</td><td><span class="badge b-cfd">OpenFOAM RANS</span></td></tr>
    <tr class="group-hdr"><td colspan="5">4. Multi-Load Stability &amp; Prior-Design Capytaine Wave Response</td></tr>
    <tr><td>Initial GM_T: 225 lb / 265 lb / 300 lb</td><td class="hi">{l225["GM_cm"]:.2f} / {l265["GM_cm"]:.2f} / {l300["GM_cm"]:.2f} cm</td><td>5.68 / 4.30 / 3.01 cm</td><td>~(S18S ski hull: lower flare)</td><td><span class="badge b-mesh">3D Float / BEM</span></td></tr>
    <tr><td>Righting Arm GZ @ 10&deg; / 20&deg; / 30&deg; (225 lb, KG=0.25m)</td><td class="hi">{g225} mm</td><td>8.9 / 13.4 / 12.2 mm</td><td>&mdash;</td><td><span class="badge b-mesh">3D Float</span></td></tr>
    <tr><td>Righting Arm GZ @ 10&deg; / 20&deg; / 30&deg; (265 lb Baja)</td><td class="hi">{g265} mm</td><td>6.8 / 10.6 / 8.9 mm</td><td>&mdash; (330 lb max cap.)</td><td><span class="badge b-mesh">3D Float</span></td></tr>
    <tr><td>Head-Chop Bow Dips/min (4.5 kn, Hs 0.35m Tp 2.5s)</td><td class="hi">11.2 /min (213 lb) | 11.8 /min (225 lb)</td><td>15.3 /min (225 lb)</td><td>&mdash;</td><td><span class="badge b-cfd">Capytaine BEM</span></td></tr>
    <tr><td>Seat Vertical Slamming Accel RMS (4.5 kn Head Chop)</td><td class="hi">1.23 m/s&sup2; (-15% empty, -26% @ 265 lb)</td><td>1.44 m/s&sup2; (1.66 @ 265 lb)</td><td>&mdash;</td><td><span class="badge b-cfd">Capytaine BEM</span></td></tr>
    <tr><td>Beam-Wind Rudder Trim @ 5.0 kn (15 kn / 22 kn Wind)</td><td class="hi">2.12&deg; / 4.55&deg; (-44% vs 18X)</td><td>3.77&deg; / 8.11&deg;</td><td>Over-stern / Under-stern opt.</td><td><span class="badge b-cfd">Munk + Aero</span></td></tr>
  </tbody>
</table>
</div>
"""


def build_html(specs: dict, stiff: dict) -> str:
    sr_4in = float(specs["SR_4in"])
    b_oa = float(specs["B_oa_cm"])
    vols = specs["compartments_liters"]
    bow_l, cock_l, pod_l, stern_l = vols["bow_compartment_L"], vols["cockpit_foot_and_seat_well_L"], vols["day_pod_L"], vols["stern_compartment_L"]
    tot_l = float(specs["total_sealed_buoyancy_liters"])
    defl_mm = float(stiff["rear_deck_stiffener"]["cruciform_ud_rib_plus_arch_beam"]["deflection_mm"])
    master_tbl = make_master_table(specs, sr_4in, b_oa, bow_l, pod_l, stern_l, tot_l)
    svg_trim = make_svg_trim(bow_l, pod_l, stern_l)
    usca_pct = float(specs["USCA_4in_pct"])
    catch_1m = float(specs["hydro_3load"]["catch_beams_225_cm"]["+1.0"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Kayak 10.9 — Naval Architecture, 3D-Printed Tooling &amp; Build Monograph</title>
  <link rel="stylesheet" href="styles.css"/>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <span class="eyebrow">Open-Hardware Design Monograph &bull; USCA &amp; Sound Rowers FSK Class</span>
    <h1>Kayak 10.9 — Naval Architecture &amp; Composite Build Package</h1>
    <p class="subtitle">
      A class-maxed 18-foot Fast Sea Kayak (5.470 m CAD / 5.480 m w/ gudgeon &times; {b_oa:.2f} cm, Sound Rowers 4&quot; ratio {sr_4in:.2f} &le; 10.99, USCA 4&quot; beam {usca_pct:.2f}% &ge; 8.5%)
      designed with <a href="{KAYAKGEN_URL}" style="color:#a7f3d0">kayak-gen</a> for wing-paddle flatwater/bay speed and minimalist watermaker expeditions, benchmarked against the
      <strong>Epic 18X / 18X Sport</strong> and <strong>Stellar S18R G2</strong>. Source &amp; releases: <a href="{REPO_URL}" style="color:#a7f3d0">github.com/halbritt/kayak-10-9</a>.
    </p>
    <div class="kpi-grid">
      <div class="kpi"><div class="val">{sr_4in:.2f}</div><div class="lbl">Sound Rowers 4&quot; Ratio (&le; 10.99 FSK Cap)</div></div>
      <div class="kpi"><div class="val">27.0–29.4 lb</div><div class="lbl">TeXtreme + Rohacell Target Mass</div></div>
      <div class="kpi"><div class="val">-25.9%</div><div class="lbl">Prior-design OpenFOAM Wave Drag vs. 18X @ 6.2 kn</div></div>
      <div class="kpi"><div class="val">-23% to -26%</div><div class="lbl">Calc. Bow Plunges in Steep Chop (BEM)</div></div>
      <div class="kpi"><div class="val">{tot_l:.1f} L</div><div class="lbl">Sealed Dry Storage (2 BHs + 5.5 L Pod)</div></div>
      <div class="kpi"><div class="val">{defl_mm:.2f} mm</div><div class="lbl">Calc. Aft-Deck Flex @ 180 lbf (+64 g Grid)</div></div>
    </div>
  </div>
</header>
<nav class="topnav">
  <div class="wrap">
    <a href="#gallery">Studio Renders</a>
    <a href="#viewer">Interactive 3D</a>
    <a href="#benchmarks">10.9 vs. 18X vs. S18R</a>
    <a href="#lines">Lines &amp; GA</a>
    <a href="#hydro">Hydro &amp; Ergonomics</a>
    <a href="#cfd">OpenFOAM &amp; BEM</a>
    <a href="#strongback">Strongback &amp; Molds</a>
    <a href="#scale1to8">1:8 H2S .3MF</a>
    <a href="#downloads">Latest Downloads</a>
  </div>
</nav>
<main class="wrap">
  <section id="gallery" class="card">
    <h2>Photorealistic PBR Orthographic Studio Renders (Click Any Image to Zoom)</h2>
    <div class="render-stack">
      <div class="render-two">
        <div class="render-card">
          <img src="renders/render_ortho_bow_quarter.png" alt="Orthographic Bow-Quarter Studio Render"/>
          <div class="rcap"><span><strong>Orthographic Bow-Quarter (3400 &times; 1550 px)</strong> &mdash; Rounded Epic-style bow stem, narrow wing-paddle catch scuppers, and carbon keyhole/hooks.</span><a href="renders/render_ortho_bow_quarter.png" download>PNG &darr;</a></div>
        </div>
        <div class="render-card">
          <img src="renders/render_ortho_stern_quarter.png" alt="Orthographic Stern-Quarter Studio Render"/>
          <div class="rcap"><span><strong>Orthographic Stern-Quarter (3400 &times; 1550 px)</strong> &mdash; 14.8 cm low rear deck, recessed aft layback well, offset day hatch, and over-stern NACA-0012 foil.</span><a href="renders/render_ortho_stern_quarter.png" download>PNG &darr;</a></div>
        </div>
      </div>
      <div class="render-card">
        <img src="renders/render_top_plan.png" alt="True Orthographic Top Plan Studio Render"/>
        <div class="rcap"><span><strong>True Orthographic Top Plan (3500 &times; 980 px)</strong> &mdash; Fair {b_oa:.1f} cm sheer outline, narrow catch entry, 84 &times; 42 cm keyhole, 60 mm flush UD keel strap, and {tot_l:.1f} L 3-hatch layout.</span><a href="renders/render_top_plan.png" download>PNG &darr;</a></div>
      </div>
      <div class="render-card">
        <img src="renders/render_side_profile.png" alt="True Orthographic Side Profile Studio Render"/>
        <div class="rcap"><span><strong>True Orthographic Side Profile (3500 &times; 900 px)</strong> &mdash; Rounded leading-edge stem &amp; 135 &times; 56 mm forefoot, arched footbox, flush aft coaming lip, and 14.8 cm low rear deck.</span><a href="renders/render_side_profile.png" download>PNG &darr;</a></div>
      </div>
    </div>
  </section>
""" + _body_sections_1(sr_4in, b_oa, catch_1m, bow_l, cock_l, pod_l, stern_l, tot_l, defl_mm, master_tbl, svg_trim) + _body_sections_2()


def _body_sections_1(sr_4in: float, b_oa: float, catch_1m: float, bow_l: float, cock_l: float, pod_l: float, stern_l: float, tot_l: float, defl_mm: float, master_tbl: str, svg_trim: str) -> str:
    return f"""
  <section id="viewer" class="card">
    <h2>Interactive In-Browser 3D CAD &amp; Tooling Assemblies</h2>
    <p class="lead">
      Switch between all four multi-part 3D assemblies right in the browser (powered by self-hosted MIT <code>Online3DViewer</code>).
      Use the left sidebar inside the viewer to toggle individual meshes (e.g., hide <code>02_deck_shell</code> on the boat or hide <code>01b_smoked_stbd_h2s_mold_halves</code> on the stepped-cradle strongback).
    </p>
    <div class="viewer-bar">
      <div class="viewer-tabs">
        <button class="vtab active" data-model="models/kayak-10-9-xray-cutaway.glb">1. Smoked-Deck X-Ray Boat</button>
        <button class="vtab" data-model="models/kayak-10-9-finished.glb">2. Opaque Finished Boat</button>
        <button class="vtab" data-model="models/kayak-10-9-h2s-stepped-cradle-strongback.glb">3. H2S Stepped-Cradle Strongback &amp; Smoked Mold</button>
        <button class="vtab" data-model="models/kayak-10-9-scale-1to8-h2s-plate-preview.glb">4. 1:8 Scale Bambu H2S .3MF Plate</button>
      </div>
      <a id="viewer-fullscreen" class="viewer-open" href="viewer/index.html" target="_blank" rel="noopener">Open Full-Screen 3D Viewer &nearr;</a>
    </div>
    <iframe id="viewer-iframe" class="viewer-frame" src="about:blank" allowfullscreen></iframe>
  </section>

  <section id="benchmarks" class="card">
    <h2>Master Benchmark Comparison — Kayak 10.9 vs. Epic 18X vs. Stellar S18R G2</h2>
    <p class="lead">
      <strong>10.9</strong> pairs a <strong>{sr_4in:.2f}</strong> 4&quot; waterline ratio and <strong>{catch_1m:.1f} cm</strong> wing-cut catch forward with a <strong>{b_oa:.1f} cm</strong> flared aft sheer,
      <strong>14.8 cm</strong> low rear deck, and <strong>{tot_l:.1f} L</strong> 2-bulkhead + carbon day-pod dry storage layout.
    </p>
    {master_tbl}
  </section>

  <section id="lines" class="card">
    <h2>Plate 1 &amp; Plate 2 — Lines Plan, Rounded Bow Stem &amp; General Arrangement</h2>
    <div class="plate-box">
      <img src="plates/plate1_lines_plan.png" alt="Plate 1 Lines Plan"/>
      <div class="caption"><strong>Plate 1:</strong> Profile with 6/12/18 cm buttock lines, half-breadth waterlines (4.0&quot; Sound Rowers caliper contour vs. Epic 18X and Stellar S18R), 20-station split body plan, and 225 lb submerged Sectional Area Curve.</div>
    </div>
    <div class="plate-box">
      <img src="plates/bilge-fairing-v020.png" alt="v0.2.1 submerged bilge fairing comparison"/>
      <div class="caption"><strong>v0.2.1 Forebody Bilge Fairing (<code>forward: 0.07 &rarr; aft: 0.0</code>):</strong> Keel-pinned symmetric <code>D&sup3;</code> batten fairing removes the submerged quarter-ellipse chine and forward centreline keel groove while keeping the <code>10.88</code> Sound Rowers ratio and <code>5.08 / 5.67 / 6.17 kn</code> speeds intact.</div>
    </div>
    <div class="plate-box">
      <img src="plates/forefoot-revision.png" alt="Original and revised bow STL under the same grazing light"/>
      <div class="caption">The v0.1.1 forefoot keeps the rounded keel profile and fairs the diagonal crease visible on the original print.</div>
    </div>
    <div class="plate-box">
      <img src="plates/bow_roundover_vs_epic18x.png" alt="Bow Roundover vs Epic 18X"/>
      <div class="caption"><strong>Bow Stem Detail (1:1 mm):</strong> <code>r_LE = 3.8 mm</code> monotonic semicircular stem bullnose, <code>26 &times; 23 mm</code> upper nose crown fillet, and <code>135 &times; 56 mm</code> forefoot curve overlaid against Epic 18X scan geometry.</div>
    </div>
    <div class="plate-box">
      <img src="plates/plate2_general_arrangement.png" alt="Plate 2 General Arrangement"/>
      <div class="caption"><strong>Plate 2:</strong> 2-bulkhead arrangement (<code>-0.95 m</code> forward BH + <code>+0.44 m</code> slanted seat BH, <code>{cock_l:.1f} L</code> cockpit well), contoured <code>{pod_l:.1f} L</code> starboard carbon day pod, co-cured <code>+64 g</code> cruciform (<code>+</code>) rear-deck UD grid, and flush <code>60 mm</code> internal UD keel strap.</div>
    </div>
    {svg_trim}
  </section>

  <section id="hydro" class="card">
    <h2>Plate 3 &amp; Plate 4 — Hydrostatics, Multi-Load GZ &amp; Wing-Paddle Ergonomics</h2>
    <div class="plate-box">
      <img src="plates/plate3_hydrostatics_and_gz.png" alt="Plate 3 Hydrostatics and GZ"/>
      <div class="caption"><strong>Plate 3:</strong> Hydrostatic curves vs. draft (5.8–11.6 cm) and 0&deg;–45&deg; calculated GZ curves across 213 lb carbon ultralight (<code>GM = 5.53 cm</code>), 225 lb day, 265 lb Baja watermaker, and 300 lb heavy expedition loads.</div>
    </div>
    <div class="plate-box">
      <img src="plates/plate4_ergonomics_and_3boat.png" alt="Plate 4 Wing Ergonomics"/>
      <div class="caption"><strong>Plate 4:</strong> Monotonic upper-beam transition from catch (+1.3 m) to aft seat (-0.4 m), 75&deg; wing-blade catch clearance, and normalized 3-boat comparison.</div>
    </div>
    {SVG_CALIPER}
  </section>

  <section id="cfd" class="card">
    <p>Plates 5 and 6 report earlier v0.1.0 simulations. The v0.2.1 faired-forebody hull has fresh hydrostatic and analytical resistance calculations; its RANS drag and BEM wave response have yet to be re-simulated.</p>
    <h2>Plates 5, 6 &amp; 7 — OpenFOAM RANS CFD, Capytaine 3D Seakeeping &amp; X-Ray Cutaways</h2>
    <div class="plate-box">
      <img src="plates/plate5_openfoam_cfd.png" alt="Plate 5 OpenFOAM RANS CFD"/>
      <div class="caption"><strong>Plate 5 (Calculated / Unvalidated Comparative):</strong> Prior-design OpenFOAM-v2512 <code>snappyHexMesh</code> + <code>interFoam</code> VOF RANS at 225 lb seawater equilibrium (5.50 &amp; 6.20 kn).</div>
    </div>
    <div class="plate-box">
      <img src="plates/plate6_seakeeping_and_windage.png" alt="Plate 6 Capytaine 3D Seakeeping"/>
      <div class="caption"><strong>Plate 6 (Calculated / Unvalidated Comparative):</strong> Capytaine 3.0 1,408-panel 3D linear BEM JONSWAP wave response, crosswind weathercocking balance, and rear-deck cruciform stiffener flex ({defl_mm:.2f} mm @ 180 lbf).</div>
    </div>
    <div class="plate-box">
      <img src="plates/plate7_3d_views.png" alt="Plate 7 3D Cutaways"/>
      <div class="caption"><strong>Plate 7:</strong> Shaded 3/4 exterior views plus cockpit and aft-deck X-ray cutaways.</div>
    </div>
  </section>
"""


def _body_sections_2() -> str:
    return f"""
  <section id="strongback" class="card">
    <h2>Plate 8 &amp; Stepped Wedge-Clamping Baltic-Birch Strongback for H2S Female Molds</h2>
    <p class="lead">
      Splitting each <code>324.0 mm</code> station at <code>y = 0</code> lets all 34 Port and Starboard female hull mold halves (<code>316.6 &times; 216.7 &times; 322.6 mm</code> max)
      print upright at <code>0&deg;</code> inside the Bambu Lab H2S (<code>340 &times; 320 &times; 340 mm</code>). Two dead-flat horizontal <code>Z</code>-datum steps lock hull rocker regardless of print shrinkage,
      while top-access <code>8&deg;</code> gib draw-wedges (<code>7.1:1</code> mechanical advantage) high up the hull walls and lower <code>M8</code> stainless keel cross-tie rods clamp the interlocking <code>y = 0</code> tongue-and-groove seam in compression.
    </p>
    <div class="plate-box">
      <img src="plates/stepped_cradle_cross_section.png" alt="Stepped Cradle Cross Section"/>
      <div class="caption"><strong>Midship Clamping Section (1:1 mm):</strong> Port &amp; Starboard H2S half-molds nested on <code>0&deg;</code> horizontal Baltic-birch datum steps with top-access <code>8&deg;</code> draw-wedges and lower <code>M8</code> keel cross-tie rod.</div>
    </div>
    <div class="render-two" style="margin-top:1rem">
      <div class="render-card">
        <img src="renders/stepped_cradle_closeup.png" alt="Stepped Cradle X-Ray Closeup"/>
        <div class="rcap"><span><strong>Smoked Mold &amp; Cradle Close-Up</strong> &mdash; 168 mm birch L-saddles, 8&deg; wedges, M8 keel tie-rods, and &Oslash;10 mm seam dowels.</span><a href="renders/stepped_cradle_closeup.png" download>PNG &darr;</a></div>
      </div>
      <div class="render-card">
        <img src="renders/stepped_cradle_cutaway_port_only.png" alt="Port Only Cutaway"/>
        <div class="rcap"><span><strong>Centerline y = 0 Cutaway</strong> &mdash; Starboard half removed to show interlocking keel tongue-and-groove and stepped birch pocket.</span><a href="renders/stepped_cradle_cutaway_port_only.png" download>PNG &darr;</a></div>
      </div>
    </div>
    <div class="plate-box">
      <img src="renders/stepped_cradle_studio_quarter.png" alt="Full 17-Station Stepped Cradle Strongback"/>
      <div class="caption"><strong>Full 17-Station Assembly:</strong> 5,540 &times; 240 &times; 176 mm 18 mm Baltic-birch torsion-box spine on Bessel-station (&plusmn;0.28 L) rolling shop trestles.</div>
    </div>
    <div class="plate-box">
      <img src="plates/plate8_construction_and_tooling.png" alt="Plate 8 Laminate and Tooling"/>
      <div class="caption"><strong>Plate 8:</strong> TeXtreme Spread-Tow Carbon/Innegra + Rohacell IG-F 51 sandwich schedule, weight budget, Voron 600&sup3; diagonal mold option, and H2S track-mounting jigs.</div>
    </div>
  </section>

  <section id="scale1to8" class="card">
    <h2>1:8 Scale (683.8 mm) Bambu Lab H2S Single-Plate .3MF Print Package</h2>
    <p class="lead">
      Splits Kayak 10.9 cleanly at its two real watertight bulkheads (<code>-0.95 m</code> forward BH &amp; <code>+0.44 m</code> seat BH) into three self-supporting vertical CoreXY towers
      (<code>223.2 mm</code> Bow, <code>174.1 mm</code> Cockpit Midbody, <code>286.4 mm</code> Stern) that print together on one H2S plate with <strong>0 g of supports</strong>, zero longitudinal hull/deck seam,
      an unbroken 1-piece keyhole cockpit/coaming/seat, 6 octagonal shear keys, and 2 Bessel-station desk display stands.
    </p>
    <div class="plate-box">
      <img src="plates/h2s_1to8_plate_and_model_preview.png" alt="1:8 Scale H2S Plate and Model Preview"/>
      <div class="caption"><strong>PETG / PLA Slicer Notes (0.4 mm Nozzle):</strong> Use <code>Arachne</code> walls (3 loops), <code>10–12% Gyroid</code> or <code>Cross Hatch</code> infill (never crossing Grid), <code>8–10 mm</code> outer brim (<code>0.18–0.20 mm</code> brim gap for PETG), <code>Avoid crossing wall</code> enabled, and bond scuffed bulkhead faces with 5-minute / G/flex epoxy.</div>
    </div>
  </section>

  <section id="downloads" class="card">
    <h2>Latest Release CAD, Mold, .3MF &amp; Solver Downloads</h2>
    <p class="lead">Binary CAD/print packages resolve automatically to the newest tagged release on <a href="{REPO_URL}/releases/latest">GitHub Releases</a>; lightweight JSON/CSV/SVG datasets are also hosted directly alongside this page.</p>
    <div class="dl-grid">
      <a class="dl-item" href="{REL_BASE}/kayak-10-9-hull-inspection-h2s.3mf"><strong>One-Shot Bare Hull &middot; 1:12 H2S 45&deg; 3MF</strong><span>Single-piece 454.9 mm (1:12) bare hull at 45&deg; across the H2S diagonal; keel/forefoot/bilges UP, tree supports enabled on sheer back + cradles.</span></a>
      <a class="dl-item" href="{REL_BASE}/kayak-10-9-scale-1to8-h2s-bow.3mf"><strong>Bow section &middot; 1:8 H2S 3MF</strong><span>Fore section matching the v0.2.1 faired midbody (for v0.1.0 prints, use the v0.1.2 release bow).</span></a>
      <a class="dl-item" href="{REL_BASE}/kayak-10-9-scale-1to8-h2s-plate.3mf"><strong>kayak-10-9-scale-1to8-h2s-plate.3mf</strong><span>Bambu Lab H2S single-plate 1:8 scale model (6 objects, 0 g supports)</span></a>
      <a class="dl-item" href="{REL_BASE}/kayak-10-9-scale-1to8-stls.zip"><strong>kayak-10-9-scale-1to8-stls.zip</strong><span>Individual watertight 1:8 .stl sections + 683.75 mm uncut solid .stl</span></a>
      <a class="dl-item" href="{REL_BASE}/kayak-10-9-builder-dxf-offsets.zip"><strong>kayak-10-9-builder-dxf-offsets.zip</strong><span>Finished 3D station curves and stem profile, base-hull patterns and offsets, and hull JSON</span></a>
      <a class="dl-item" href="{REL_BASE}/kayak-10-9-female-mold-stls.zip"><strong>kayak-10-9-female-mold-stls.zip</strong><span>Full-scale watertight female hull/deck mold .stl samples (H2S &amp; Voron 600&sup3;)</span></a>
      <a class="dl-item" href="{REL_BASE}/kayak-10-9-3d-assemblies.zip"><strong>kayak-10-9-3d-assemblies.zip</strong><span>All .glb 3D assemblies (boat, X-ray, strongback) + outer-shell.stl</span></a>
      <a class="dl-item" href="downloads/kayak-10-9.hull.json" download><strong>kayak-10-9.hull.json</strong><span>Canonical kayak-gen DistributionV2 parametric hull specification</span></a>
      <a class="dl-item" href="downloads/offsets.csv" download><strong>offsets.csv</strong><span>Parametric base-hull offsets. Use finished-stations.dxf in the builder archive for the faired surface.</span></a>
      <a class="dl-item" href="downloads/openfoam-cfd-metrics.json" download><strong>openfoam-cfd-metrics.json</strong><span>OpenFOAM-v2512 interFoam RANS comparative drag &amp; wave dataset</span></a>
      <a class="dl-item" href="downloads/capytaine-seakeeping.json" download><strong>capytaine-seakeeping.json</strong><span>Capytaine 3.0 3D BEM JONSWAP seakeeping dataset</span></a>
    </div>
    <div class="note-box">
      <strong>Methodological &amp; Epistemological Scope:</strong> Per <a href="{KAYAKGEN_URL}">kayak-gen</a> release discipline, analytical calm-water resistance (<code>ITTC-57 + Michell</code>),
      OpenFOAM-v2512 free-surface RANS CFD (<code>raw_unvalidated</code>), Capytaine 3.0 3D linear BEM seakeeping, hydrostatic righting-arm curves (<code>GZ</code>), and laminate flex calculations
      are matched comparative physics predictions rather than tow-tank-calibrated guarantees.
    </div>
  </section>
</main>
<div id="lightbox" onclick="this.style.display='none'"><img id="lb-img" src="" alt="Zoomed plate"/></div>
<script>
document.querySelectorAll('.plate-box img, .render-card img').forEach(img => {{
  img.addEventListener('click', () => {{
    document.getElementById('lb-img').src = img.src;
    document.getElementById('lightbox').style.display = 'flex';
  }});
}});
function loadModel(relPath) {{
  const modelUrl = new URL(relPath, window.location.href).href;
  const embedBase = new URL('viewer/embed.html', window.location.href).href;
  const fullBase = new URL('viewer/index.html', window.location.href).href;
  document.getElementById('viewer-iframe').src = embedBase + '#model=' + modelUrl;
  document.getElementById('viewer-fullscreen').href = fullBase + '#model=' + modelUrl;
}}
document.querySelectorAll('.vtab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.vtab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    loadModel(btn.dataset.model);
  }});
}});
loadModel('models/kayak-10-9-xray-cutaway.glb');
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=REPO_ROOT / "_site")
    args = ap.parse_args()
    dist: Path = args.dist
    (dist / "downloads").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SITE_DIR / "styles.css", dist / "styles.css")
    shutil.copy2(HULL_DIR / "kayak-10-9.hull.json", dist / "downloads" / "kayak-10-9.hull.json")
    for p in DATA_DIR.iterdir():
        if p.is_file():
            shutil.copy2(p, dist / "downloads" / p.name)
    specs = json.loads((DATA_DIR / "kayak-10-9-specs.json").read_text())
    stiff = json.loads((DATA_DIR / "windage-and-stiffener.json").read_text())
    (dist / "index.html").write_text(build_html(specs, stiff))
    (dist / ".nojekyll").write_text("")
    print("BUILT SITE:", dist / "index.html")


if __name__ == "__main__":
    main()
