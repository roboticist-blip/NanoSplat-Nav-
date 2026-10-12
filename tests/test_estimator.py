"""
Estimator tests against the synthetic rover with known ground truth.
These verify the filter's behaviour; hardware validation is a separate step.
"""
import socket
import unittest

import numpy as np

from nanosplat.estimator.ekf import ImuEkf
from nanosplat.pipeline.nav import NavPipeline
from nanosplat.pipeline.udp_sink import UdpPoseSink
from nanosplat.output.pose_frame import decode
from nanosplat.sim.synthetic import simulate

SIG_A_DENSITY = 3.9e-3     # MPU6050 datasheet order, m/s^2/sqrt(Hz)
SIG_G_DENSITY = 8.7e-5     # rad/s/sqrt(Hz)


def _run(data, use_pose=True, dropout=None, pose_timeout=1.0, sink=None):
    t0, p0, v0, q0 = data["truth"][0]
    ekf = ImuEkf(p0, v0, q0, SIG_A_DENSITY, SIG_G_DENSITY)
    pipe = NavPipeline(ekf, sink=sink, pose_timeout=pose_timeout, keep_history=True)
    poses = {round(t, 6): (p, q) for t, p, q in data["pose"]} if use_pose else {}
    for t, acc, gyro in data["imu"]:
        pipe.on_imu(t, acc, gyro)
        key = round(t, 6)
        if key in poses:
            p, q = poses[key]
            pipe.on_pose(t, p, q)
    return pipe


def _errors(pipe, truth):
    est = np.array([e.position for e in pipe.history])
    gt = np.array([p for _, p, _, _ in truth])
    return np.linalg.norm(est - gt, axis=1), est, gt


class TestRoverEstimate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = simulate(duration=60.0, seed=1)

    def test_fused_estimate_tracks_truth(self):
        pipe = _run(self.data)
        err, _, _ = _errors(pipe, self.data["truth"])
        self.assertLess(np.sqrt(np.mean(err ** 2)), 0.15)
        self.assertLess(err[-1], 0.3)

    def test_imu_only_drifts_badly(self):
        pipe = _run(self.data, use_pose=False)
        err, _, _ = _errors(pipe, self.data["truth"])
        self.assertGreater(err[-1], 5.0)

    def test_scale_is_metric(self):
        # Path length is compared at 1 Hz on both estimate and truth. Summing the 200 Hz
        # estimate directly counts the measurement noise as distance and is not a valid check.
        pipe = _run(self.data)
        _, est, gt = _errors(pipe, self.data["truth"])
        step = 200
        est_len = np.sum(np.linalg.norm(np.diff(est[::step], axis=0), axis=1))
        gt_len = np.sum(np.linalg.norm(np.diff(gt[::step], axis=0), axis=1))
        self.assertLess(abs(est_len - gt_len) / gt_len, 0.02)

    def test_recovers_after_pose_dropout(self):
        data = simulate(duration=60.0, seed=2, dropout=(20.0, 30.0))
        pipe = _run(data)
        err, _, _ = _errors(pipe, data["truth"])
        t = np.array([s[0] for s in data["truth"]])
        after = err[t > 35.0]
        self.assertLess(np.max(after), 0.3)

    def test_validity_drops_without_pose(self):
        pipe = _run(self.data, use_pose=False, pose_timeout=1.0)
        self.assertFalse(pipe.history[-1].valid)

    def test_validity_high_with_pose(self):
        pipe = _run(self.data, pose_timeout=1.0)
        self.assertTrue(all(e.valid for e in pipe.history[int(2 / 0.005):]))

    def test_bias_estimate_moves_toward_truth(self):
        data = self.data
        pipe = _run(data)
        ba_true = data["bias_a"]
        self.assertLess(np.linalg.norm(pipe.ekf.ba - ba_true), np.linalg.norm(ba_true))


class TestUdpOutput(unittest.TestCase):
    def test_frames_arrive_and_decode(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(2.0)
        port = rx.getsockname()[1]
        sink = UdpPoseSink("127.0.0.1", port)
        data = simulate(duration=2.0, seed=3)
        _run(data, sink=sink)
        sink.close()
        buf, _ = rx.recvfrom(256)
        est = decode(buf)
        self.assertEqual(est.seq, 0)
        np.testing.assert_allclose(est.position, data["truth"][0][1], atol=0.5)
        rx.close()


if __name__ == "__main__":
    unittest.main()
