"""
Navigation pipeline: time-ordered IMU and pose inputs in, PoseEstimate frames out.

Inputs are called in time order:
    on_imu(t, accel, gyro)                  every IMU sample (drives propagation and output)
    on_pose(t, position, quat, sig_p, sig_att)   when an absolute pose measurement arrives

Each IMU sample produces one PoseEstimate. `valid` is set only while a pose measurement
arrived within `pose_timeout` seconds, so a consumer can tell when the estimate has gone
IMU-only and is drifting.
"""
import numpy as np

from nanosplat.output.pose_frame import PoseEstimate


class NavPipeline(object):
    def __init__(self, ekf, sink=None, pose_timeout=1.0, keep_history=False):
        self.ekf = ekf
        self.sink = sink
        self.pose_timeout = float(pose_timeout)
        self.keep_history = keep_history
        self.history = []
        self.seq = 0
        self.last_t = None
        self.last_pose_t = None

    def on_imu(self, t, accel, gyro):
        if self.last_t is not None:
            dt = t - self.last_t
            if dt > 0:
                self.ekf.propagate(accel, gyro, dt)
        self.last_t = t
        est = self._estimate(t)
        if self.keep_history:
            self.history.append(est)
        if self.sink is not None:
            self.sink(est)
        return est

    def on_pose(self, t, position, quat, sig_p=0.02, sig_att=0.01):
        self.ekf.update_pose(position, quat, sig_p, sig_att)
        self.last_pose_t = t

    def _estimate(self, t):
        valid = self.last_pose_t is not None and (t - self.last_pose_t) <= self.pose_timeout
        quality = 1.0 / (1.0 + self.ekf.position_std())
        est = PoseEstimate(seq=self.seq, t=t, position=self.ekf.p, quat=self.ekf.q,
                           velocity=self.ekf.v, quality=quality, valid=valid)
        self.seq += 1
        return est
