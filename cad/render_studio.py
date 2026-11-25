"""Render 4K orthographic VTK PBR + Clearcoat + Studio IBL views of Kayak 10.9."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image, ImageFilter
import pyvista as pv
import trimesh

from deck_geometry import SculptedKayakSurface
from outfitting_geometry import (
    build_coaming_and_thigh_hooks,
    build_day_pod_and_structural_grid,
    build_hatch_covers,
    build_interior_and_bulkheads,
    build_overstern_rudder,
    build_seat_and_footboard,
)


def make_studio_cubemap(tmp_dir: Path, size: int = 512) -> pv.Texture:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    u = np.linspace(-1.0, 1.0, size)
    uu, vv = np.meshgrid(u, -u)
    dirs = {"px": (np.ones_like(uu), -vv, -uu), "nx": (-np.ones_like(uu), -vv, uu), "py": (uu, np.ones_like(uu), vv), "ny": (uu, -np.ones_like(uu), -vv), "pz": (uu, -vv, np.ones_like(uu)), "nz": (-uu, -vv, -np.ones_like(uu))}
    sky, horizon, floor = np.array([150.0, 176.0, 206.0]), np.array([68.0, 86.0, 108.0]), np.array([28.0, 38.0, 50.0])
    paths = []
    for name in ("px", "nx", "py", "ny", "pz", "nz"):
        dx, dy, dz = dirs[name]
        nrm = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / nrm, dy / nrm, dz / nrm
        upness = np.clip(0.65 * dz + 0.35 * dy, -1.0, 1.0)
        t_up, t_dn = np.clip(upness, 0.0, 1.0)[:, :, None], np.clip(-upness, 0.0, 1.0)[:, :, None]
        rgb = horizon[None, None, :] * (1.0 - t_up - t_dn) + sky[None, None, :] * t_up + floor[None, None, :] * t_dn
        sb1 = np.exp(-(((dy - 0.38) / 0.14) ** 2)) * np.exp(-(((dz - 0.82) / 0.24) ** 2)) * np.exp(-((dx / 0.92) ** 6))
        sb2 = np.exp(-(((dy + 0.42) / 0.18) ** 2)) * np.exp(-(((dz - 0.72) / 0.26) ** 2)) * np.exp(-((dx / 0.92) ** 6))
        sb_y = np.exp(-(((dz - 0.35) / 0.16) ** 2)) * np.exp(-(((dy - 0.82) / 0.24) ** 2)) * np.exp(-((dx / 0.92) ** 6))
        img = np.clip(rgb + (145.0 * sb1 + 110.0 * sb2 + 120.0 * sb_y)[:, :, None], 0, 255).astype(np.uint8)
        p = tmp_dir / f"{name}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p))
    return pv.cubemap_from_filenames(paths)


def make_textreme_texture(base_dark: tuple[int, int, int] = (28, 32, 38), base_light: tuple[int, int, int] = (48, 55, 64)) -> pv.Texture:
    x = np.linspace(0, 8 * np.pi, 512, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    u45, v45 = (xx + yy) / np.sqrt(2.0), (xx - yy) / np.sqrt(2.0)
    chk = (np.sin(u45) * np.sin(v45)) > 0
    sheen = np.where(chk, 0.72 + 0.28 * (0.5 + 0.5 * np.sin(18.0 * u45)), 0.72 + 0.28 * (0.5 + 0.5 * np.sin(18.0 * v45)))[:, :, None]
    arr = np.where(chk[:, :, None], np.array(base_light, float)[None, None, :], np.array(base_dark, float)[None, None, :]) * sheen
    tex = pv.numpy_to_texture(np.clip(arr, 0, 255).astype(np.uint8))
    tex.repeat = True
    return tex


def make_kevlar_carbon_texture() -> pv.Texture:
    ii, jj = np.meshgrid(np.arange(256), np.arange(256))
    is_kevlar = (((ii // 16) % 4 + (jj // 16) % 4) % 4) < 2
    grain = (0.88 + 0.12 * np.sin(0.4 * (ii + jj)))[:, :, None]
    kev, car = np.array([214.0, 184.0, 118.0])[None, None, :], np.array([62.0, 66.0, 72.0])[None, None, :]
    tex = pv.numpy_to_texture(np.clip(np.where(is_kevlar[:, :, None], kev, 0.78 * kev + 0.22 * car) * grain, 0, 255).astype(np.uint8))
    tex.repeat = True
    return tex


def trimesh_to_pv(tm_obj: trimesh.Trimesh, feature_angle: float = 52.0, uv_scale: float = 7.0) -> pv.PolyData:
    v, f = np.asarray(tm_obj.vertices, float), np.asarray(tm_obj.faces, np.int64)
    pd = pv.PolyData(v, np.column_stack((np.full(len(f), 3, dtype=np.int64), f)).ravel())
    pd = pd.compute_normals(cell_normals=False, point_normals=True, split_vertices=True, feature_angle=feature_angle, auto_orient_normals=False)
    pts = np.asarray(pd.points)
    pd.active_texture_coordinates = np.column_stack((pts[:, 0] * uv_scale, (pts[:, 1] + 0.65 * pts[:, 2]) * uv_scale))
    return pd


def make_shadow_floor(outer_verts: np.ndarray, z_floor: float = -0.086) -> tuple[pv.PolyData, pv.Texture]:
    w_img, h_img = 1024, 512
    x_min, x_max, y_min, y_max = -3.35, +3.45, -1.15, +1.15
    alpha = np.zeros((h_img, w_img), np.float32)
    px = np.clip(((outer_verts[:, 0] - x_min) / (x_max - x_min) * (w_img - 1)).astype(int), 0, w_img - 1)
    py = np.clip(((outer_verts[:, 1] - y_min) / (y_max - y_min) * (h_img - 1)).astype(int), 0, h_img - 1)
    prox = np.clip(1.0 - outer_verts[:, 2] / 0.30, 0.30, 1.0)
    for ix, iy, pr in zip(px, py, prox):
        alpha[max(0, iy - 5) : min(h_img, iy + 6), max(0, ix - 5) : min(w_img, ix + 6)] = np.maximum(alpha[max(0, iy - 5) : min(h_img, iy + 6), max(0, ix - 5) : min(w_img, ix + 6)], 165.0 * pr)
    a_arr = np.asarray(Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=22)), dtype=np.uint8)
    rgba = np.zeros((h_img, w_img, 4), dtype=np.uint8)
    rgba[:, :, :3] = (8, 12, 18); rgba[:, :, 3] = a_arr
    plane = pv.Plane(center=(0.05, 0.0, z_floor), direction=(0, 0, 1), i_size=x_max - x_min, j_size=y_max - y_min, i_resolution=2, j_resolution=2)
    plane.texture_map_to_plane(origin=(x_min, y_min, z_floor), point_u=(x_max, y_min, z_floor), point_v=(x_min, y_max, z_floor), inplace=True)
    return plane, pv.numpy_to_texture(rgba)


def build_seam_tape_and_deck_lines(surf: SculptedKayakSurface) -> tuple[pv.PolyData, pv.PolyData]:
    seam_splines = [pv.Spline(surf.end_shape_xyz(np.column_stack((surf.xs, sgn * surf.seam_y, surf.seam_z))), n_points=320).tube(radius=0.0028, n_sides=12) for sgn in (-1.0, +1.0)]
    seam_pd = seam_splines[0].merge(seam_splines[1])
    line_parts: list[pv.PolyData] = []
    for sgn in (-1.0, +1.0):
        for xa, xb in ((-2.28, -0.52), (+0.48, +2.28)):
            xl = np.linspace(xa, xb, 48)
            yl = sgn * np.maximum(np.interp(xl, surf.xs, surf.seam_y) - 0.024, 0.012)
            zl = surf.sculpted_deck_z(xl, yl) + 0.0032
            line_parts.append(pv.Spline(surf.end_shape_xyz(np.column_stack((xl, yl, zl))), n_points=90).tube(radius=0.0020, n_sides=10))
            for xf in (xa, 0.5 * (xa + xb), xb):
                yf = float(np.interp(xf, xl, yl))
                zf = float(surf.sculpted_deck_z(np.array([xf]), np.array([yf]))[0])
                line_parts.append(pv.Cylinder(center=surf.end_shape_xyz(np.array([[xf, yf, zf + 0.001]]))[0], direction=(1, 0, 0), radius=0.006, height=0.018))
    for p_a, p_b in ((np.array([-2.732, 0.0, 0.312]), np.array([-2.780, 0.0, 0.294])), (np.array([+2.732, 0.0, 0.206]), np.array([+2.772, 0.0, 0.190]))):
        line_parts.append(pv.Cylinder(center=p_b, direction=(0, 1, 0), radius=0.0055, height=0.075))
        line_parts.append(pv.Spline(np.array([p_a, 0.5 * (p_a + p_b) + [0, 0, 0.004], p_b]), 20).tube(radius=0.0018))
    lines_pd = line_parts[0]
    for lp in line_parts[1:]:
        lines_pd = lines_pd.merge(lp)
    return seam_pd, lines_pd


def _add_pbr(pl: pv.Plotter, pd: pv.PolyData, color: str | None = None, texture: pv.Texture | None = None, metallic: float = 0.08, roughness: float = 0.18, coat: float = 0.85, coat_rough: float = 0.05) -> None:
    act = pl.add_mesh(pd, color=color if color is not None else "white", pbr=True, metallic=metallic, roughness=roughness, smooth_shading=True)
    prop = act.GetProperty()
    if texture is not None:
        texture.SetUseSRGBColorSpace(True)
        prop.SetBaseColorTexture(texture)
    if hasattr(prop, "SetCoatStrength"):
        prop.SetCoatStrength(coat)
        prop.SetCoatRoughness(coat_rough)
    if hasattr(prop, "SetBaseIOR"):
        prop.SetBaseIOR(1.52)


def render_studio_view(parts: dict[str, pv.PolyData], textures: dict[str, pv.Texture], cubemap: pv.Texture, floor_tuple: tuple[pv.PolyData, pv.Texture] | None, cam_pos: tuple[float, float, float], focal_pt: tuple[float, float, float], up_vec: tuple[float, float, float], parallel_scale: float, out_path: Path, win_size: tuple[int, int] = (3400, 1550)) -> None:
    pl = pv.Plotter(off_screen=True, window_size=win_size, lighting="none")
    pl.set_background("#1a2431", top="#2a394c")
    pl.set_environment_texture(cubemap, is_srgb=True)
    for pos, inten, col in [((-4.5, -5.5, 6.5), 1.15, "#fffdf9"), ((+5.5, -3.5, 4.8), 0.65, "#e6f2ff"), ((+1.5, +6.0, 5.2), 0.60, "#ffffff"), ((0.0, -7.0, -0.5), 0.38, "#cbd5e1")]:
        lt = pv.Light(position=pos, focal_point=(0.0, 0.0, 0.12), color=col, intensity=inten)
        lt.positional = False
        pl.add_light(lt)
    if floor_tuple is not None:
        pl.add_mesh(floor_tuple[0], texture=floor_tuple[1], lighting=False)
    _add_pbr(pl, parts["hull"], color="#eef2f7", metallic=0.04, roughness=0.17, coat=0.92, coat_rough=0.04)
    _add_pbr(pl, parts["deck"], color="#3b82b6", metallic=0.14, roughness=0.19, coat=0.90, coat_rough=0.045)
    _add_pbr(pl, parts["coaming"], texture=textures["carbon"], metallic=0.22, roughness=0.24, coat=0.88, coat_rough=0.06)
    _add_pbr(pl, parts["hatches"], color="#232830", metallic=0.05, roughness=0.65, coat=0.06, coat_rough=0.35)
    _add_pbr(pl, parts["interior"], texture=textures["kevlar"], metallic=0.06, roughness=0.40, coat=0.42, coat_rough=0.12)
    _add_pbr(pl, parts["pod_grid"], texture=textures["carbon"], metallic=0.20, roughness=0.26, coat=0.80, coat_rough=0.07)
    _add_pbr(pl, parts["seat"], texture=textures["carbon"], metallic=0.22, roughness=0.25, coat=0.82, coat_rough=0.07)
    _add_pbr(pl, parts["rudder"], texture=textures["carbon"], metallic=0.28, roughness=0.22, coat=0.88, coat_rough=0.05)
    _add_pbr(pl, parts["seam"], color="#161a20", metallic=0.15, roughness=0.32, coat=0.70, coat_rough=0.08)
    _add_pbr(pl, parts["lines"], color="#12151a", metallic=0.08, roughness=0.58, coat=0.08, coat_rough=0.30)
    pl.camera.position, pl.camera.focal_point, pl.camera.up = cam_pos, focal_pt, up_vec
    pl.enable_parallel_projection(); pl.camera.parallel_scale = parallel_scale
    pl.enable_ssao(radius=0.09, bias=0.008); pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(return_img=True); pl.close()
    Image.fromarray(img).save(out_path, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders-dir", type=Path, default=REPO_ROOT / "build" / "renders")
    args = ap.parse_args()
    args.renders_dir.mkdir(parents=True, exist_ok=True)

    surf = SculptedKayakSurface(REPO_ROOT / "hull" / "kayak-10-9.hull.json")
    m_hull, m_deck = surf.build_hull_shell(), surf.build_deck_shell(cut_openings=True)
    m_rud = build_overstern_rudder(surf)
    seam_pd, lines_pd = build_seam_tape_and_deck_lines(surf)
    parts = {
        "hull": trimesh_to_pv(m_hull, 75.0), "deck": trimesh_to_pv(m_deck, 88.0),
        "coaming": trimesh_to_pv(build_coaming_and_thigh_hooks(surf), 55.0, 12.0),
        "hatches": trimesh_to_pv(build_hatch_covers(surf), 55.0),
        "interior": trimesh_to_pv(build_interior_and_bulkheads(surf), 55.0, 10.0),
        "pod_grid": trimesh_to_pv(build_day_pod_and_structural_grid(surf), 55.0, 12.0),
        "seat": trimesh_to_pv(build_seat_and_footboard(surf), 55.0, 12.0),
        "rudder": trimesh_to_pv(m_rud, 45.0, 14.0),
        "seam": seam_pd.compute_normals(auto_orient_normals=False),
        "lines": lines_pd.compute_normals(auto_orient_normals=False),
    }
    textures = {"carbon": make_textreme_texture(), "kevlar": make_kevlar_carbon_texture()}
    floor_sh = make_shadow_floor(np.vstack((np.asarray(m_hull.vertices), np.asarray(m_rud.vertices))), z_floor=-0.086)
    with tempfile.TemporaryDirectory(prefix="kayak109-cubemap-", dir=str(args.renders_dir.parent)) as tmp:
        cubemap = make_studio_cubemap(Path(tmp))
        render_studio_view(parts, textures, cubemap, floor_sh, (-5.4, -5.2, 2.85), (0.04, 0.0, 0.08), (0, 0, 1), 1.08, args.renders_dir / "render_ortho_bow_quarter.png", (3400, 1550))
        render_studio_view(parts, textures, cubemap, floor_sh, (+5.5, -5.0, 2.75), (0.02, 0.0, 0.08), (0, 0, 1), 1.08, args.renders_dir / "render_ortho_stern_quarter.png", (3400, 1550))
        render_studio_view(parts, textures, cubemap, floor_sh, (0.02, 0.0, 8.0), (0.02, 0.0, 0.10), (0, 1, 0), 0.81, args.renders_dir / "render_top_plan.png", (3500, 980))
        render_studio_view(parts, textures, cubemap, None, (0.02, -8.0, 0.11), (0.02, 0.0, 0.11), (0, 0, 1), 0.77, args.renders_dir / "render_side_profile.png", (3500, 900))
    print("RENDERED ALL 4 LARGE PHOTOREALISTIC STUDIO VIEWS ->", args.renders_dir)


if __name__ == "__main__":
    main()
