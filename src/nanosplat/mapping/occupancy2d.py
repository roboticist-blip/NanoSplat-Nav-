"""
2D log-odds occupancy grid for ground-robot navigation.

Each range scan marks cells along each beam as free and the endpoint as occupied (when the
beam hit something inside max_range). Cell updates are applied once per scan, so a cell hit by
many beams is not counted many times. The output feeds a planner directly.
"""
import numpy as np

L_OCC = 0.85
L_FREE = -0.4
L_MIN, L_MAX = -5.0, 5.0


class OccupancyGrid2D(object):
    def __init__(self, width_m, height_m, res_m, origin_xy=(-5.0, -5.0)):
        self.res = float(res_m)
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        self.nx = int(np.ceil(width_m / res_m))
        self.ny = int(np.ceil(height_m / res_m))
        self.L = np.zeros((self.ny, self.nx))

    def _index(self, x, y):
        ix = np.floor((x - self.ox) / self.res).astype(int)
        iy = np.floor((y - self.oy) / self.res).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix[inside], iy[inside]

    def update(self, origin_xy, yaw, bearings, ranges, max_range):
        """bearings (rad, body frame), ranges (m). Ranges >= max_range mean no hit."""
        bearings = np.asarray(bearings, float)
        ranges = np.asarray(ranges, float)
        ox, oy = origin_xy
        ang = yaw + bearings
        hit = ranges < max_range
        n_steps = max(2, int(np.ceil(max_range / (0.5 * self.res))))
        s = np.linspace(0.0, 1.0, n_steps, endpoint=False)[None, :]
        r = np.minimum(ranges, max_range)[:, None] * s
        xs = ox + r * np.cos(ang)[:, None]
        ys = oy + r * np.sin(ang)[:, None]
        fx, fy = self._index(xs.ravel(), ys.ravel())
        free_flat = np.unique(fy * self.nx + fx)
        occ_x, occ_y = self._index(ox + ranges[hit] * np.cos(ang[hit]), oy + ranges[hit] * np.sin(ang[hit]))
        occ_flat = np.unique(occ_y * self.nx + occ_x)
        free_flat = np.setdiff1d(free_flat, occ_flat)
        self.L.flat[free_flat] += L_FREE
        self.L.flat[occ_flat] += L_OCC
        np.clip(self.L, L_MIN, L_MAX, out=self.L)

    def probability(self):
        return 1.0 / (1.0 + np.exp(-self.L))

    def occupied(self, threshold=0.65):
        return self.probability() > threshold

    def free(self, threshold=0.35):
        return self.probability() < threshold

    def world_to_cell(self, x, y):
        return int(np.floor((x - self.ox) / self.res)), int(np.floor((y - self.oy) / self.res))
