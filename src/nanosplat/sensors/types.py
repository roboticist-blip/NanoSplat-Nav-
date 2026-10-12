"""Hardware-independent sensor sample types (Python 3.6 compatible)."""


class ImuSample(object):
    """One IMU reading in SI units. t is host monotonic seconds."""
    __slots__ = ("t", "accel", "gyro", "temp_c")

    def __init__(self, t, accel, gyro, temp_c=None):
        self.t = float(t)
        self.accel = tuple(float(v) for v in accel)   # m/s^2, body frame
        self.gyro = tuple(float(v) for v in gyro)     # rad/s, body frame
        self.temp_c = temp_c

    def __repr__(self):
        return "ImuSample(t=%.4f, accel=%s, gyro=%s)" % (self.t, self.accel, self.gyro)
