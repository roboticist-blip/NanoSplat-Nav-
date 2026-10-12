"""
Synthetic rover run with known ground truth, for testing the estimator without hardware.

Path: circle of radius R with a small vertical wave, level body, heading follows the velocity.
IMU: specific force and yaw rate, with constant bias, white noise at the MPU6050 datasheet density.
Pose: position and attitude measurements at a lower rate, with noise and optional dropout.
"""
import math

import numpy as np

from nanosplat.math3d import yaw_quat, quat_to_R

G = 9.80665


class Trajectory(object):
    def __init__(self, radius=5.0, period=40.0, wave=0.1):
        self.R = radius
        self.w = 2 * math.pi / period
        self.wave = wave

    def at(self, t):
        w, R = self.w, self.R
        p = np.array([R * math.cos(w * t), R * math.sin(w * t), self.wave * math.sin(w * t)])
        v = np.array([-R * w * math.sin(w * t), R * w * math.cos(w * t), self.wave * w * math.cos(w * t)])
        a = np.array([-R * w * w * math.cos(w * t), -R * w * w * math.sin(w * t), -self.wave * w * w * math.sin(w * t)])
        yaw = math.atan2(v[1], v[0])
        yaw_rate = (v[0] * a[1] - v[1] * a[0]) / (v[0] ** 2 + v[1] ** 2)
        return p, v, a, yaw, yaw_rate


def simulate(duration=60.0, imu_rate=200.0, pose_rate=10.0, sig_a=0.039, sig_g=8.7e-4,
             bias_a=(0.05, -0.03, 0.02), bias_g=(0.002, -0.001, 0.0005),
             sig_p=0.02, sig_att=0.01, dropout=None, seed=0):
    """
    Returns dict with imu (list of (t, acc, gyro)), pose (list of (t, p, q) or skipped),
    truth (list of (t, p, v)), and the bias used.
    sig_a, sig_g are per-sample standard deviations at imu_rate.
    dropout: optional (t_start, t_end) window with no pose measurements.
    """
    rng = np.random.RandomState(seed)
    traj = Trajectory()
    ba, bg = np.array(bias_a), np.array(bias_g)
    imu, pose, truth = [], [], []
    n_imu = int(duration * imu_rate)
    for k in range(n_imu + 1):
        t = k / imu_rate
        p, v, a, yaw, yaw_rate = traj.at(t)
        q = yaw_quat(yaw)
        R = quat_to_R(q)
        f_w = a + np.array([0.0, 0.0, G])           # specific force in world frame
        f_b = R.T.dot(f_w) + ba + rng.normal(0, sig_a, 3)
        w_b = np.array([0.0, 0.0, yaw_rate]) + bg + rng.normal(0, sig_g, 3)
        imu.append((t, f_b, w_b))
        truth.append((t, p, v, q))
    step = int(round(imu_rate / pose_rate))
    for k in range(0, n_imu + 1, step):
        t = k / imu_rate
        if dropout and dropout[0] <= t <= dropout[1]:
            continue
        p, v, a, yaw, yaw_rate = traj.at(t)
        dp = rng.normal(0, sig_p, 3)
        dth = rng.normal(0, sig_att, 3)
        q = quat_mul_safe(yaw_quat(yaw), exp_vec(dth))
        pose.append((t, p + dp, q))
    return {"imu": imu, "pose": pose, "truth": truth, "bias_a": ba, "bias_g": bg}


def quat_mul_safe(a, b):
    from nanosplat.math3d import quat_mul, quat_normalize
    return quat_normalize(quat_mul(a, b))


def exp_vec(v):
    from nanosplat.math3d import exp_map
    return exp_map(v)
