"""
IMU noise parameters used by the estimator, with an explicit provenance flag.

Two sources are allowed:
- "provisional": datasheet-typical placeholders, used only until the sensor is fixed and
  a passing stationary recording exists. Never treated as calibrated.
- "measured": produced from an Allan report whose health check passed.

from_measured() refuses any report with a FAIL, so a faulty log cannot become calibration.
Units: gyro noise density rad/s/sqrt(Hz), accel m/s^2/sqrt(Hz); bias instability in rad/s and m/s^2.
"""
import json
import math

DEG = math.pi / 180.0
G = 9.80665


class NoiseParams(object):
    def __init__(self, gyro_nd, accel_nd, gyro_bi, accel_bi, source, note=""):
        if source not in ("provisional", "measured"):
            raise ValueError("source must be 'provisional' or 'measured'")
        self.gyro_nd = float(gyro_nd)
        self.accel_nd = float(accel_nd)
        self.gyro_bi = float(gyro_bi)
        self.accel_bi = float(accel_bi)
        self.source = source
        self.note = note

    @property
    def is_calibrated(self):
        return self.source == "measured"

    def to_json(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        return cls(d["gyro_nd"], d["accel_nd"], d["gyro_bi"], d["accel_bi"],
                   d["source"], d.get("note", ""))


def provisional_mpu6050():
    """Datasheet-typical MPU6050 values. Placeholders only; recalibrate on final hardware."""
    return NoiseParams(
        gyro_nd=0.005 * DEG,            # 0.005 dps/sqrt(Hz)
        accel_nd=400e-6 * G,            # 400 ug/sqrt(Hz)
        gyro_bi=0.1 * DEG,              # placeholder: bias instability not specified by datasheet
        accel_bi=0.02 * G,              # placeholder
        source="provisional",
        note="Datasheet-typical placeholder. Not calibrated. Replace with a measured recording.")


def from_measured(rows, issues):
    """
    rows: output of imu_log.noise_report (list of dicts with axis, noise_density, bias_instability).
    issues: output of imu_log.health_check. Any FAIL blocks calibration.
    """
    fails = [m for s, m in issues if s == "FAIL"]
    if fails:
        raise ValueError("log failed the health check; refusing to calibrate: " + "; ".join(fails))
    by = {r["axis"]: r for r in rows}
    gyro = [by[a] for a in ("gx", "gy", "gz")]
    accel = [by[a] for a in ("ax", "ay", "az")]
    return NoiseParams(
        gyro_nd=max(r["noise_density"] for r in gyro),
        accel_nd=max(r["noise_density"] for r in accel),
        gyro_bi=max(r["bias_instability"] for r in gyro),
        accel_bi=max(r["bias_instability"] for r in accel),
        source="measured",
        note="Worst axis of a passing stationary recording.")
