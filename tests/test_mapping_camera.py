"""Occupancy grid, initial attitude, and camera wrapper tests (no hardware)."""
import math
import unittest

import numpy as np

from nanosplat.mapping.occupancy2d import OccupancyGrid2D
from nanosplat.estimator.init import initial_quat_from_gravity
from nanosplat.math3d import quat_to_R
from nanosplat.camera.usb_camera import UsbCamera


class TestOccupancy(unittest.TestCase):
    def _wall_scan(self, grid, origin, yaw, wall_x=2.0):
        bearings = np.radians(np.linspace(-30, 30, 61))
        ranges = np.full(len(bearings), 10.0)
        for i, b in enumerate(bearings):
            a = yaw + b
            if math.cos(a) > 0:
                r = (wall_x - origin[0]) / math.cos(a)
                if r < 10.0:
                    ranges[i] = r
        for _ in range(4):
            grid.update(origin, yaw, bearings, ranges, max_range=10.0)

    def test_wall_is_occupied_and_space_before_it_is_free(self):
        g = OccupancyGrid2D(12.0, 12.0, 0.05, origin_xy=(-3.0, -6.0))
        self._wall_scan(g, origin=(0.0, 0.0), yaw=0.0)
        ix_wall, iy0 = g.world_to_cell(2.0, 0.0)
        ix_mid, _ = g.world_to_cell(1.0, 0.0)
        self.assertGreater(g.probability()[iy0, ix_wall], 0.8)
        self.assertLess(g.probability()[iy0, ix_mid], 0.2)

    def test_unobserved_cells_stay_unknown(self):
        g = OccupancyGrid2D(12.0, 12.0, 0.1, origin_xy=(-6.0, -6.0))
        self._wall_scan(g, origin=(0.0, 0.0), yaw=0.0)
        ix_behind, iy = g.world_to_cell(-4.0, 4.0)
        self.assertAlmostEqual(g.probability()[iy, ix_behind], 0.5)

    def test_log_odds_are_clamped(self):
        g = OccupancyGrid2D(12.0, 12.0, 0.1, origin_xy=(-3.0, -6.0))
        self._wall_scan(g, origin=(0.0, 0.0), yaw=0.0)
        for _ in range(50):
            self._wall_scan(g, origin=(0.0, 0.0), yaw=0.0)
        self.assertLessEqual(g.L.max(), 5.0)
        self.assertGreaterEqual(g.L.min(), -5.0)


class TestInitialAttitude(unittest.TestCase):
    def test_level_gives_identity(self):
        q = initial_quat_from_gravity(np.tile([0.0, 0.0, 9.80665], (100, 1)))
        np.testing.assert_allclose(quat_to_R(q), np.eye(3), atol=1e-9)

    def test_roll_and_pitch_recovered(self):
        roll, pitch = math.radians(10), math.radians(-5)
        # specific force in body = R^T * (0,0,g) for body tilted by (roll, pitch)
        from nanosplat.math3d import exp_map
        Rt = quat_to_R(exp_map([roll, 0, 0]))
        Rt = quat_to_R(exp_map([0, pitch, 0])).dot(Rt)
        f = Rt.T.dot([0.0, 0.0, 9.80665])
        q = initial_quat_from_gravity(np.tile(f, (50, 1)))
        Rq = quat_to_R(q)
        # gravity direction in body should map back to world up
        np.testing.assert_allclose(Rq.dot(f), [0, 0, 9.80665], atol=1e-6)


class TestCamera(unittest.TestCase):
    class _Fake(object):
        def __init__(self, ok=True):
            self.ok = ok
            self.props = {}

        def isOpened(self):
            return True

        def set(self, prop, val):
            self.props[prop] = val
            return True

        def read(self):
            return self.ok, (np.zeros((480, 640, 3), np.uint8) if self.ok else None)

        def release(self):
            self.released = True

    def test_returns_timestamp_and_frame(self):
        cam = UsbCamera(capture_factory=lambda: self._Fake())
        t, frame = cam.read()
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(t, 0.0)

    def test_read_failure_raises(self):
        cam = UsbCamera(capture_factory=lambda: self._Fake(ok=False))
        with self.assertRaises(RuntimeError):
            cam.read()


if __name__ == "__main__":
    unittest.main()
