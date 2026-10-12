"""
Spatially indexed, confidence-weighted Gaussian map.

Implements the fusion rule of the NanoSplat paper (Eqs. 10-12) with the
association fixed: a new observation is fused with the Gaussian that occupies
the same voxel, not with whatever sits in the next ring-buffer slot (v1 bug).

Memory is bounded by `capacity`. When full, the lowest-confidence fraction is
pruned and the arrays are compacted, so nothing is reallocated during a session.
"""
from typing import Dict

import numpy as np

_OFF = 1 << 20          # coordinate offset so voxel indices pack into int64
_BITS = 21
_MASK = (1 << _BITS) - 1


class VoxelFusion:
    def __init__(self, voxel_size: float, capacity: int,
                 omega_max: float = 8.0, prune_keep: float = 0.85):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.capacity = int(capacity)
        self.omega_max = float(omega_max)
        self.prune_keep = float(prune_keep)

        self.pos = np.zeros((capacity, 3), np.float32)
        self.col = np.zeros((capacity, 3), np.float32)
        self.scale = np.zeros((capacity, 3), np.float32)
        self.conf = np.zeros(capacity, np.float32)
        self.opacity = np.zeros(capacity, np.float32)
        self.keys = np.zeros(capacity, np.int64)
        self.n = 0
        self._index: Dict[int, int] = {}

    # ── keys ────────────────────────────────────────────────────────────────
    def _pack(self, pts: np.ndarray) -> np.ndarray:
        v = np.floor(pts / self.voxel_size).astype(np.int64) + _OFF
        if np.any(v < 0) or np.any(v > _MASK):
            raise ValueError("point outside representable voxel range")
        return (v[:, 0] << (2 * _BITS)) | (v[:, 1] << _BITS) | v[:, 2]

    # ── public API ──────────────────────────────────────────────────────────
    def add(self, pts: np.ndarray, cols: np.ndarray, scales: np.ndarray,
            opacity: float = 0.85) -> int:
        """Fuse a batch of observations. Returns number of voxels touched."""
        if len(pts) == 0:
            return 0
        keys = self._pack(pts)
        uniq, inv = np.unique(keys, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float32)

        def _mean(a):
            s = np.zeros((len(uniq),) + a.shape[1:], np.float32)
            np.add.at(s, inv, a.astype(np.float32))
            return s / cnt.reshape((-1,) + (1,) * (a.ndim - 1))

        m_pos, m_col, m_scl = _mean(pts), _mean(cols), _mean(scales)

        slots = self._lookup(uniq)
        n_new = int(np.count_nonzero(slots < 0))
        if n_new and self.n + n_new > self.capacity:
            self.prune()                       # prune moves slots: look up again
            slots = self._lookup(uniq)

        # ── existing voxels: vectorised Eqs. 10-12 with batch count c ──────
        ex = np.nonzero(slots >= 0)[0]
        s = slots[ex]
        w = self.conf[s]
        c = cnt[ex]
        wc = (w + c)[:, None]
        self.pos[s] = (w[:, None] * self.pos[s] + c[:, None] * m_pos[ex]) / wc
        self.col[s] = (w[:, None] * self.col[s] + c[:, None] * m_col[ex]) / wc
        self.scale[s] = (w[:, None] * self.scale[s] + c[:, None] * m_scl[ex]) / wc
        self.conf[s] = np.minimum(w + c, self.omega_max)
        touched = len(ex)

        # ── new voxels: append as far as capacity allows ───────────────────
        nw = np.nonzero(slots < 0)[0][: self.capacity - self.n]
        if len(nw):
            dst = np.arange(self.n, self.n + len(nw))
            self.pos[dst] = m_pos[nw]
            self.col[dst] = m_col[nw]
            self.scale[dst] = m_scl[nw]
            self.conf[dst] = cnt[nw]
            self.opacity[dst] = opacity
            self.keys[dst] = uniq[nw]
            for k, d in zip(uniq[nw].tolist(), dst.tolist()):
                self._index[k] = d
            self.n += len(nw)
            touched += len(nw)
        return touched

    def _lookup(self, uniq: np.ndarray) -> np.ndarray:
        """Slot index per voxel key, or -1 if the voxel is not in the map."""
        get = self._index.get
        return np.fromiter((get(k, -1) for k in uniq.tolist()),
                           dtype=np.int64, count=len(uniq))

    def prune(self) -> int:
        """Keep the most confident `prune_keep` fraction and compact the arrays."""
        if self.n == 0:
            return 0
        keep_n = max(1, int(self.n * self.prune_keep))
        order = np.argsort(-self.conf[:self.n], kind="stable")[:keep_n]
        removed = self.n - keep_n
        for name in ("pos", "col", "scale", "conf", "opacity", "keys"):
            arr = getattr(self, name)
            arr[:keep_n] = arr[order]
        self.n = keep_n
        self._index = {int(k): i for i, k in enumerate(self.keys[:keep_n])}
        return removed

    def points(self) -> np.ndarray:
        return self.pos[:self.n]

    def colors(self) -> np.ndarray:
        return self.col[:self.n]

    def confidences(self) -> np.ndarray:
        return self.conf[:self.n]
