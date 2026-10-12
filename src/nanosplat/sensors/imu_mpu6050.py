"""
MPU6050 driver (6-axis IMU, I2C). Python 3.6 compatible.

The bus access is injected as a `read_regs(reg, n)` / `write_reg(reg, value)` pair,
so the decoding and scaling logic can be tested without hardware. On the Jetson,
`open_smbus(bus=1)` provides these via smbus2.

Known limits of the MPU6050 (plan the estimator around these):
- No magnetometer: yaw is gyro-integrated only and drifts.
- No hardware timestamps: sample time is the host read time, with I2C jitter.
- Gyro bias drifts with temperature; calibrate at start-up (stationary) and log temperature.
"""
import math
import time

from nanosplat.sensors.types import ImuSample

ADDR = 0x68
REG_WHO_AM_I = 0x75
REG_PWR_MGMT_1 = 0x6B
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_DATA_START = 0x3B          # ACCEL_X_H .. GYRO_Z_L, 14 bytes

GYRO_FS = {250: (0, 131.0), 500: (1, 65.5), 1000: (2, 32.8), 2000: (3, 16.4)}   # dps -> LSB per dps
ACCEL_FS = {2: (0, 16384.0), 4: (1, 8192.0), 8: (2, 4096.0), 16: (3, 2048.0)}  # g -> LSB per g
G = 9.80665
DEG = math.pi / 180.0


def _s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


def decode_frame(raw14, gyro_fs_dps=500, accel_fs_g=8, t=None):
    """Decode the 14-byte burst read from 0x3B into an ImuSample (SI units)."""
    if len(raw14) != 14:
        raise ValueError("expected 14 bytes")
    b = list(raw14)
    ax, ay, az = _s16(b[0], b[1]), _s16(b[2], b[3]), _s16(b[4], b[5])
    temp_raw = _s16(b[6], b[7])
    gx, gy, gz = _s16(b[8], b[9]), _s16(b[10], b[11]), _s16(b[12], b[13])
    _, g_lsb = GYRO_FS[gyro_fs_dps]
    _, a_lsb = ACCEL_FS[accel_fs_g]
    accel = (ax / a_lsb * G, ay / a_lsb * G, az / a_lsb * G)
    gyro = (gx / g_lsb * DEG, gy / g_lsb * DEG, gz / g_lsb * DEG)
    temp_c = temp_raw / 340.0 + 36.53
    return ImuSample(time.monotonic() if t is None else t, accel, gyro, temp_c)


class MPU6050(object):
    def __init__(self, read_regs, write_reg, sample_rate_hz=200,
                 gyro_fs_dps=500, accel_fs_g=8, dlpf=3):
        self._read = read_regs
        self._write = write_reg
        self.gyro_fs_dps = gyro_fs_dps
        self.accel_fs_g = accel_fs_g
        self._configure(sample_rate_hz, dlpf)

    def _configure(self, rate_hz, dlpf):
        who = self._read(REG_WHO_AM_I, 1)[0]
        if who != 0x68:
            raise RuntimeError("MPU6050 not found (WHO_AM_I=0x%02X)" % who)
        self._write(REG_PWR_MGMT_1, 0x00)            # wake, internal 8 MHz clock
        time.sleep(0.1)
        self._write(REG_CONFIG, dlpf & 0x07)         # digital low-pass filter
        base = 1000 if dlpf not in (0, 7) else 8000  # Hz before divider
        self._write(REG_SMPLRT_DIV, max(0, int(round(base / float(rate_hz))) - 1))
        self._write(REG_GYRO_CONFIG, GYRO_FS[self.gyro_fs_dps][0] << 3)
        self._write(REG_ACCEL_CONFIG, ACCEL_FS[self.accel_fs_g][0] << 3)

    def read_sample(self):
        raw = bytes(self._read(REG_DATA_START, 14))
        return decode_frame(raw, self.gyro_fs_dps, self.accel_fs_g)


def open_smbus(bus=1):
    """Return (read_regs, write_reg) bound to smbus2 on the given I2C bus."""
    import smbus2                                   # MIT licence, pure Python
    b = smbus2.SMBus(bus)

    def read_regs(reg, n):
        return b.read_i2c_block_data(ADDR, reg, n)

    def write_reg(reg, value):
        b.write_byte_data(ADDR, reg, value)

    return read_regs, write_reg
