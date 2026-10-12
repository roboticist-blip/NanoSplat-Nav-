"""
IMU recording format and per-axis noise report.

CSV columns: t, ax, ay, az, gx, gy, gz, temp_c   (t in seconds, host monotonic clock)
Accel in m/s^2, gyro in rad/s, as produced by the MPU6050 driver.
"""
import csv

import numpy as np

from nanosplat.calib.allan import allan_deviation, noise_density, bias_instability

COLUMNS = ["t", "ax", "ay", "az", "gx", "gy", "gz", "temp_c"]
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]


def write_csv(path, rows):
    """rows: iterable of (t, ax, ay, az, gx, gy, gz, temp_c)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(["%.6f" % r[0]] + ["%.7f" % v for v in r[1:7]] + ["%.2f" % r[7]])


def read_csv(path):
    """Return a dict of column name -> numpy array."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != COLUMNS:
            raise ValueError("unexpected header: %s" % header)
        data = np.array([[float(v) for v in row] for row in reader], np.float64)
    return {name: data[:, i] for i, name in enumerate(COLUMNS)}


def sample_rate(t):
    """Mean rate and jitter of the timestamps, in Hz and seconds (std of intervals)."""
    dt = np.diff(t)
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return 1.0 / dt.mean(), dt.std()


def noise_report(log):
    """
    Per-axis noise parameters from a stationary log.
    Returns a list of dicts: axis, noise_density, bias_tau_s, bias_instability.
    The rate used is the measured mean rate; the host-polling jitter is reported separately.
    """
    fs, jitter = sample_rate(log["t"])
    out = []
    for axis in AXES:
        x = log[axis] - np.mean(log[axis])
        taus, adev = allan_deviation(x, fs)
        N = noise_density(taus, adev, fs)
        tau_b, bi = bias_instability(taus, adev)
        out.append({"axis": axis, "noise_density": N, "bias_tau_s": tau_b,
                    "bias_instability": bi})
    return fs, jitter, out


def format_report(fs, jitter, rows, duration_s):
    lines = ["Recording: %.0f s, measured rate %.2f Hz, interval jitter %.2f ms"
             % (duration_s, fs, jitter * 1e3),
             "",
             "| Axis | Noise density | Bias instability | at tau (s) |",
             "|---|---|---|---|"]
    for r in rows:
        unit_n = "rad/s/sqrt(Hz)" if r["axis"].startswith("g") else "m/s^2/sqrt(Hz)"
        unit_b = "rad/s" if r["axis"].startswith("g") else "m/s^2"
        lines.append("| %s | %.3e %s | %.3e %s | %.1f |" % (
            r["axis"], r["noise_density"], unit_n, r["bias_instability"], unit_b,
            r["bias_tau_s"]))
    return "\n".join(lines)


G = 9.80665
GYRO_FULL_SCALE = 500.0 * 3.141592653589793 / 180.0    # rad/s at +/-500 dps
ACCEL_FULL_SCALE = 8.0 * G                             # m/s^2 at +/-8 g


def health_check(log, gyro_fs=GYRO_FULL_SCALE, accel_fs=ACCEL_FULL_SCALE):
    """
    Checks that must pass before a log is used for noise estimation.
    Returns a list of (severity, message); severity is "FAIL" or "WARN".
    FAIL means the Allan numbers from this log are not trustworthy.
    """
    issues = []
    sat_limit = 0.999
    for axis in ("gx", "gy", "gz"):
        n_sat = int(np.sum(np.abs(log[axis]) >= sat_limit * gyro_fs))
        if n_sat > 0:
            issues.append(("FAIL", "%s is at full scale (+/-500 dps) for %d samples (%.1f%%). "
                           "The sensor or its bus connection is faulty." %
                           (axis, n_sat, 100.0 * n_sat / len(log[axis]))))
    for axis in ("ax", "ay", "az"):
        n_sat = int(np.sum(np.abs(log[axis]) >= sat_limit * accel_fs))
        if n_sat > 0:
            issues.append(("FAIL", "%s is at full scale (+/-8 g) for %d samples." % (axis, n_sat)))

    acc = np.c_[log["ax"], log["ay"], log["az"]]
    mag = np.linalg.norm(acc, axis=1)
    mean_mag = float(np.mean(mag))
    if abs(mean_mag - G) > 0.3:
        issues.append(("WARN", "accelerometer magnitude at rest is %.2f m/s^2, expected %.2f. "
                       "Per-axis offset/scale calibration is needed." % (mean_mag, G)))

    gyro_mean = np.array([np.mean(log[a]) for a in ("gx", "gy", "gz")])
    bias_dps = np.degrees(np.linalg.norm(gyro_mean))
    if bias_dps > 5.0:
        issues.append(("WARN", "static gyro offset is %.1f dps. Typical MPU6050 offsets are under "
                       "about 20 dps, and the recording should be checked for a fault." % bias_dps))

    for axis in ("gx", "gy", "gz"):
        x = log[axis]
        step = np.abs(np.diff(x))
        if np.percentile(step, 99.9) > 0.5:
            issues.append(("WARN", "%s has large sample-to-sample jumps (99.9th percentile %.2f rad/s)."
                           % (axis, np.percentile(step, 99.9))))

    block = 5 * 60 * 100
    if len(log["t"]) > 2 * block:
        for axis in ("gx", "gy", "gz"):
            means = [np.mean(log[axis][i:i + block]) for i in range(0, len(log[axis]) - block + 1, block)]
            if np.ptp(means) > 0.05:
                issues.append(("WARN", "%s drifts by %.3f rad/s across 5-minute blocks (not stationary)."
                               % (axis, np.ptp(means))))

    repeats = int(np.sum(np.all(np.diff(np.c_[log["ax"], log["ay"], log["az"],
                                             log["gx"], log["gy"], log["gz"]], axis=0) == 0, axis=1)))
    if repeats > 0.001 * len(log["t"]):
        issues.append(("WARN", "%d identical consecutive samples (stale reads)." % repeats))
    return issues
