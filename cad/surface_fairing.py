"""Fit the hull and topsides together on a longitudinal × sectional-girth net."""

import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.optimize import brentq


def _cubic_fit(t, points, spans):
    """Least-squares cubic on a uniform knot grid, accounting for sample spacing."""
    spans = min(spans, len(t) - 3)
    knots = np.r_[np.repeat(t[0], 3), np.linspace(t[0], t[-1], spans + 1), np.repeat(t[-1], 3)]
    weights = np.sqrt(np.gradient(t))
    return make_lsq_spline(t, points, knots, k=3, w=weights)(t)


def _fade(t):
    t = np.clip(t, 0, 1)
    return t**3 * (10 + t * (-15 + 6 * t))


def fit_half_surface(x, raw, beam_height, design_beam):
    """Fit Y and Z in metres on rings running from keel to deck seam."""
    girth = np.linspace(0, 1, raw.shape[1])
    # A separable least-squares solve gives a tensor-product cubic B-spline.
    along = _cubic_fit(x, raw, 10)
    fitted = _cubic_fit(girth, along.swapaxes(0, 1), 22).swapaxes(0, 1)
    # Carry the original stem fillets, keel and deck tangency into the fit.
    end_distance = np.minimum(x - x[0], x[-1] - x)
    longitudinal = _fade((end_distance - 0.25) / 0.30)
    transverse = _fade(girth / 0.12) * _fade((1 - girth) / 0.12)
    weight = longitudinal[:, None] * transverse[None, :]
    half = raw + weight[:, :, None] * (fitted - raw)
    # One breadth scale for the entire fitted surface, with the same protected
    # boundaries. Constrain the finished 4-inch beam (v0.2.1 design datum).
    # Z is independent of the scale, so each edge-plane hit is affine in it.
    y, z = half[:, :, 0], half[:, :, 1]
    i, j = np.nonzero((z[:, :-1] <= beam_height) & (z[:, 1:] > beam_height))
    f = (beam_height - z[i, j]) / (z[i, j + 1] - z[i, j])

    def intercept(values):
        return (1 - f) * values[i, j] + f * values[i, j + 1]

    beam_y = intercept(y)
    mode = y * weight
    lever = intercept(mode)
    scale = brentq(lambda k: 2 * np.max(beam_y + k * lever) - design_beam, -0.05, 0.05)
    half[:, :, 0] += scale * mode
    return half
