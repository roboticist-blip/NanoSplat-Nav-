"""Quaternion (w, x, y, z) and small-rotation helpers. Python 3.6 compatible."""
import math

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_normalize(q):
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], np.float64)


def quat_to_R(q):
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, np.float64)
    return q if q[0] >= 0 else -q


def exp_map(phi):
    """Rotation vector to unit quaternion."""
    phi = np.asarray(phi, np.float64)
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return quat_normalize([1.0, *(phi / 2.0)])
    return np.array([math.cos(th / 2), *(math.sin(th / 2) * phi / th)])


def log_map(q):
    """Unit quaternion to rotation vector, shortest path."""
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2.0 * v
    return 2.0 * math.atan2(n, q[0]) * v / n


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
