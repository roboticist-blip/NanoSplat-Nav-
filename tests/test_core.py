"""
NanoSplat v2 core tests. Synthetic data with known ground truth, no hardware needed.
Run from the repo root:   PYTHONPATH=src python3 -m unittest discover -s tests -v
"""
import unittest

import cv2
import numpy as np

from nanosplat.map.voxel_fusion import VoxelFusion
from nanosplat.segment.belief_mask import BeliefMask
from nanosplat.pose.geometry import (chain_relative, triangulate,
                                     pose_from_landmarks, projection_matrix)
from nanosplat.depth.alignment import fit_affine_disparity, disparity_to_depth

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
RNG = np.random.default_rng(7)


def orbit_pose(theta, radius=1.0):
    c = np.array([radius * np.sin(theta), 0.0, -radius * np.cos(theta)])
    z = -c / np.linalg.norm(c)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1), c


def project(R_wc, c, X):
    Xc = (R_wc.T @ (X - c).T).T
    p = (K @ Xc.T).T
    return (p[:, :2] / p[:, 2:3]).astype(np.float64)


def rot_err_deg(A, B):
    return np.rad2deg(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


class TestVoxelFusion(unittest.TestCase):
    def test_reobserved_surface_is_fused_not_duplicated(self):
        f = VoxelFusion(voxel_size=0.005, capacity=5000)
        P = RNG.uniform(-0.1, 0.1, (1000, 3))
        C = RNG.uniform(0, 1, (1000, 3))
        S = np.full((1000, 3), 0.01)
        f.add(P, C, S)
        n1 = f.n
        f.add(P, C, S)
        self.assertEqual(f.n, n1)
        self.assertGreater(f.conf[0], 1.0)            # confidence rose

    def test_spatially_separate_points_never_merge(self):
        f = VoxelFusion(voxel_size=0.005, capacity=5000)
        A = RNG.uniform(-1, -0.9, (500, 3))
        B = RNG.uniform(5, 5.1, (500, 3))
        C = np.ones((500, 3)) * 0.5
        S = np.full((500, 3), 0.01)
        f.add(A, C, S)
        f.add(B, C, S)
        # Expected: unique voxels per batch, with no cross-batch fusion.
        expect_A = len(np.unique(f._pack(A)))
        expect_B = len(np.unique(f._pack(B)))
        self.assertEqual(f.n, expect_A + expect_B)
        # Nothing from cluster B has been averaged into cluster A.
        self.assertGreater(f.points()[expect_A:].min(), 4.0)
        self.assertLess(f.points()[:expect_A].max(), -0.8)

    def test_capacity_is_bounded_and_prunes_low_confidence(self):
        f = VoxelFusion(voxel_size=0.01, capacity=300, prune_keep=0.5)
        strong = RNG.uniform(0, 0.2, (100, 3))
        for _ in range(4):                           # make these high confidence
            f.add(strong, np.ones((100, 3)), np.full((100, 3), 0.01))
        for i in range(10):                          # flood with one-off points
            f.add(RNG.uniform(10, 20, (200, 3)), np.zeros((200, 3)),
                  np.full((200, 3), 0.01))
        self.assertLessEqual(f.n, f.capacity)
        self.assertGreater(f.confidences().max(), 3.0)


class TestBeliefMask(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((60, 80), 128, np.uint8)

    def _blob(self, on):
        m = np.zeros((60, 80), np.uint8)
        if on:
            cv2.circle(m, (40, 30), 12, 255, -1)
        return m

    def test_single_frame_flip_is_rejected(self):
        bm = BeliefMask(alpha=0.3, tau=0.5)
        bm.update(self._blob(True), self.gray)
        out = bm.update(self._blob(True), self.gray)     # stable object
        self.assertGreater(np.count_nonzero(out), 0)
        spike = np.zeros((60, 80), np.uint8)
        cv2.circle(spike, (5, 5), 4, 255, -1)            # one-frame noise far away
        out = bm.update(spike, self.gray)
        self.assertEqual(np.count_nonzero(out[0:12, 0:12]), 0)

    def test_persistent_region_is_admitted_after_two_frames(self):
        bm = BeliefMask(alpha=0.3, tau=0.5)
        patch = np.zeros((60, 80), np.uint8)
        cv2.circle(patch, (5, 5), 4, 255, -1)
        bm.update(np.zeros((60, 80), np.uint8), self.gray)
        o1 = bm.update(patch, self.gray)
        o2 = bm.update(patch, self.gray)
        self.assertEqual(np.count_nonzero(o1), 0)
        self.assertGreater(np.count_nonzero(o2), 0)

    def test_departed_object_decays(self):
        bm = BeliefMask(alpha=0.3, tau=0.5)
        for _ in range(5):
            bm.update(self._blob(True), self.gray)
        for _ in range(6):
            out = bm.update(self._blob(False), self.gray)
        self.assertEqual(np.count_nonzero(out), 0)


class TestPoseGeometry(unittest.TestCase):
    def setUp(self):
        self.X = RNG.uniform(-0.15, 0.15, (300, 3))
        self.thetas = np.linspace(0, np.deg2rad(40), 6)
        self.gt = [orbit_pose(t) for t in self.thetas]

    def test_chain_recovers_rotation_on_orbit(self):
        R_wc, c = np.eye(3), np.zeros(3)
        for i in range(1, len(self.gt)):
            p0 = project(*self.gt[i - 1], self.X)
            p1 = project(*self.gt[i], self.X)
            E, _ = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, p0, p1, K)
            R_wc, c = chain_relative(R_wc, c, R, t)
        # Ground-truth orientation expressed in the first camera's frame
        R0, c0 = self.gt[0]
        R_expected = R0.T @ self.gt[-1][0]
        self.assertLess(rot_err_deg(R_wc, R_expected), 0.5)

    def test_pose_from_landmarks_is_consistent_with_map_scale(self):
        # Map frame: camera 0 at origin, axes of camera 0, first baseline = 1 unit.
        p0 = project(*self.gt[0], self.X)
        p1 = project(*self.gt[1], self.X)
        E, _ = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, p0, p1, K)
        R_map1, c_map1 = chain_relative(np.eye(3), np.zeros(3), R, t)
        landmarks = triangulate(K, np.eye(3), np.zeros(3), R_map1, c_map1, p0, p1)

        R0, c0 = self.gt[0]
        baseline = np.linalg.norm(self.gt[1][1] - c0)
        for i in (2, 3, 4):
            Ri, ci = self.gt[i]
            pi = project(Ri, ci, self.X)
            res = pose_from_landmarks(K, landmarks, pi)
            self.assertIsNotNone(res)
            R_est, c_est, _ = res
            R_exp = R0.T @ Ri
            c_exp = R0.T @ (ci - c0) / baseline
            self.assertLess(rot_err_deg(R_est, R_exp), 0.5)
            np.testing.assert_allclose(c_est, c_exp, atol=2e-3)

    def test_triangulation_reprojects_exactly(self):
        R1, c1 = self.gt[0]
        R2, c2 = self.gt[2]
        p1 = project(R1, c1, self.X)
        p2 = project(R2, c2, self.X)
        Xr = triangulate(K, R1, c1, R2, c2, p1, p2)
        np.testing.assert_allclose(Xr, self.X, atol=1e-6)


class TestAdaptiveDepth(unittest.TestCase):
    def test_recovers_unknown_affine_disparity(self):
        a_true, b_true = 2.7, -0.4                 # arbitrary MiDaS scale and shift
        rel = RNG.uniform(0.1, 1.0, 400)
        inv_z = a_true * rel + b_true
        inv_z[:40] += RNG.uniform(0.5, 1.5, 40)    # 10% gross outliers
        a, b = fit_affine_disparity(rel, inv_z)
        self.assertAlmostEqual(a, a_true, delta=0.05)
        self.assertAlmostEqual(b, b_true, delta=0.05)

    def test_depth_conversion_is_positive(self):
        rel = RNG.uniform(0, 1, (8, 8))
        d = disparity_to_depth(rel, a=2.0, b=0.5)
        self.assertTrue(np.all(d > 0))


if __name__ == "__main__":
    unittest.main()
