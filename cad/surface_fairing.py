"""Re-export kayak-gen's whole-surface fairing helpers (RFC 0091)."""

from kayakgen.model.surface_fairing import cosine_station_grid, fit_half_surface, interp_linear_axis

__all__ = ["cosine_station_grid", "fit_half_surface", "interp_linear_axis"]
