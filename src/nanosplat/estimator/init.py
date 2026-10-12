"""Initial attitude from gravity while the vehicle is stationary at start-up."""
import math

import numpy as np

from nanosplat.math3d import yaw_quat, exp_map, quat_mul, quat_normalize


def initial_quat_from_gravity(acc_samples, yaw=0.0):
    """
    acc_samples: (N, 3) specific-force samples taken while still. At rest the accelerometer
    reads +g along the body 'up' axis, so roll and pitch follow from its direction.
    Yaw is not observable from gravity; it is set by the argument.
    """
    a = np.mean(np.asarray(acc_samples, float), axis=0)
    ax, ay, az = a
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    q_rp = quat_normalize(quat_mul(yaw_quat(0.0), exp_map([roll, 0.0, 0.0])))
    q_rp = quat_normalize(quat_mul(exp_map([0.0, pitch, 0.0]), q_rp))
    return quat_normalize(quat_mul(yaw_quat(yaw), q_rp))
