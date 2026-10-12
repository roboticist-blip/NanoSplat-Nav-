"""
Adaptive, self-calibrating depth scale.

MiDaS outputs affine-invariant disparity: disp ≈ a * rel + b, with unknown a, b
per frame. We fit (a, b) per keyframe against sparse landmarks whose inverse
depth is known from the triangulated map. The map's scale is fixed by the first
baseline, so every keyframe lands in the same units. No user-supplied distance
is needed.
"""
from typing import Tuple

import numpy as np


def fit_affine_disparity(rel_samples: np.ndarray, inv_depth_samples: np.ndarray,
                         iters: int = 10, huber_delta: float = 0.05) -> Tuple[float, float]:
    """Robust (IRLS, Huber) fit of inv_depth ≈ a * rel + b."""
    x = np.asarray(rel_samples, np.float64).ravel()
    y = np.asarray(inv_depth_samples, np.float64).ravel()
    if x.size < 2 or np.ptp(x) < 1e-9:
        raise ValueError("need at least 2 landmarks with varying MiDaS values")
    A = np.stack([x, np.ones_like(x)], 1)
    w = np.ones_like(x)
    coef = np.zeros(2)
    for _ in range(iters):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ coef
        scale = max(np.median(np.abs(r)) * 1.4826, 1e-9)
        z = np.abs(r) / max(huber_delta, scale)
        w = np.where(z <= 1.0, 1.0, 1.0 / z)
    return float(coef[0]), float(coef[1])


def disparity_to_depth(rel: np.ndarray, a: float, b: float,
                       min_disp: float = 1e-3) -> np.ndarray:
    """Convert MiDaS output to depth in map units."""
    disp = np.maximum(a * rel.astype(np.float64) + b, min_disp)
    return (1.0 / disp).astype(np.float32)
