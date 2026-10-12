"""Quaternion and rotation helpers."""
import unittest

import numpy as np

from nanosplat.math3d import quat_mul, quat_to_R, R_to_quat, exp_map, log_map, quat_conj, yaw_quat


class TestMath3d(unittest.TestCase):
    def test_R_round_trip(self):
        rng = np.random.RandomState(0)
        for _ in range(50):
            q = exp_map(rng.normal(0, 1, 3))
            q2 = R_to_quat(quat_to_R(q))
            self.assertLess(min(np.linalg.norm(q - q2), np.linalg.norm(q + q2)), 1e-9)

    def test_exp_log_inverse(self):
        phi = np.array([0.3, -0.2, 0.5])
        np.testing.assert_allclose(log_map(exp_map(phi)), phi, atol=1e-10)

    def test_composition_matches_matrices(self):
        a = exp_map([0.1, 0.4, -0.2])
        b = exp_map([-0.3, 0.1, 0.25])
        np.testing.assert_allclose(quat_to_R(quat_mul(a, b)), quat_to_R(a).dot(quat_to_R(b)), atol=1e-12)

    def test_yaw_quat_rotates_x_axis(self):
        R = quat_to_R(yaw_quat(np.pi / 2))
        np.testing.assert_allclose(R.dot([1, 0, 0]), [0, 1, 0], atol=1e-12)

    def test_conjugate_is_inverse(self):
        q = exp_map([0.2, 0.1, -0.4])
        np.testing.assert_allclose(quat_mul(q, quat_conj(q)), [1, 0, 0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
