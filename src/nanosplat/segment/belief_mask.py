"""
Temporal mask stabilisation with a float belief map.

Replaces the v1 rule, which was an identity map (audit 2.1). The belief is

    b_t = alpha * raw_t + (1 - alpha) * warp(b_{t-1})      (float, 0..1)
    mask_t = b_t > tau

With alpha = 0.3 and tau = 0.5, a pixel that appears in a single frame reaches
0.3 and is rejected. A pixel that persists for two frames reaches 0.51 and is
admitted. A pixel that disappears decays out after a few frames.
"""
from typing import Optional

import cv2
import numpy as np


class BeliefMask:
    def __init__(self, alpha: float = 0.3, tau: float = 0.5,
                 flow_params: Optional[dict] = None):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.tau = float(tau)
        self._fb = flow_params or dict(pyr_scale=0.5, levels=2, winsize=13,
                                       iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        self._prev_gray: Optional[np.ndarray] = None
        self._belief: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None
        self._belief = None

    def update(self, raw_mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """raw_mask: uint8 0/255 or bool, same shape as gray. Returns uint8 0/255."""
        raw = (raw_mask > 0).astype(np.float32)
        if self._belief is None or self._belief.shape != raw.shape:
            self._belief = raw.copy()
        else:
            flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, **self._fb)
            h, w = raw.shape
            gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
            mx = np.clip(gx + flow[..., 0], 0, w - 1).astype(np.float32)
            my = np.clip(gy + flow[..., 1], 0, h - 1).astype(np.float32)
            warped = cv2.remap(self._belief, mx, my, cv2.INTER_LINEAR)
            self._belief = self.alpha * raw + (1.0 - self.alpha) * warped
        self._prev_gray = gray.copy()
        return np.where(self._belief > self.tau, 255, 0).astype(np.uint8)
