"""
Allan deviation and IMU noise parameters from a stationary recording.

Use: record 2-3 hours of stationary IMU data (or 30+ min minimum), then run this
to get gyro/accel noise density (white noise) and bias instability (the curve minimum).
These numbers set the estimator's process and measurement noise.
"""
import math

import numpy as np


def allan_deviation(x, fs, m_list=None):
    """Overlapping Allan deviation of signal x sampled at fs Hz. Returns (taus, adev)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    if m_list is None:
        m_list = np.unique(np.logspace(0, np.log10(max(2, n // 9)), 60).astype(int))
    c = np.concatenate([[0.0], np.cumsum(x)])           # integrated signal
    taus, adev = [], []
    for m in m_list:
        if 2 * m >= n:
            break
        # averages over windows of length m, overlapping by one sample
        y = (c[m:] - c[:-m]) / m
        d = y[m:] - y[:-m]
        taus.append(m / fs)
        adev.append(math.sqrt(0.5 * np.mean(d * d)))
    return np.array(taus), np.array(adev)


def noise_density(taus, adev, fs):
    """White-noise density from the tau = 1/fs point: N = sigma * sqrt(2 / fs)."""
    sigma_sample = adev[0]              # tau = 1 sample
    return sigma_sample * math.sqrt(2.0 / fs)


def bias_instability(taus, adev):
    """Minimum of the Allan curve: (tau at minimum, minimum deviation)."""
    i = int(np.argmin(adev))
    return taus[i], adev[i]
