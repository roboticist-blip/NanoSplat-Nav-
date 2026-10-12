"""IMU driver and Allan tests. No hardware needed. Python 3.6 compatible."""
import math
import unittest

import numpy as np

from nanosplat.sensors.imu_mpu6050 import decode_frame, MPU6050, G, DEG
from nanosplat.calib.allan import allan_deviation, noise_density, bias_instability


def _pack(vals):
    out = []
    for v in vals:
        v &= 0xFFFF
        out += [(v >> 8) & 0xFF, v & 0xFF]
    return bytes(out)


class TestDecode(unittest.TestCase):
    def test_gravity_on_z_at_8g_range(self):
        # 1 g at +/-8 g range is 4096 LSB
        raw = _pack([0, 0, 4096, 0, 0, 0, 0])
        s = decode_frame(raw, gyro_fs_dps=500, accel_fs_g=8, t=0.0)
        self.assertAlmostEqual(s.accel[2], G, places=4)
        self.assertAlmostEqual(s.accel[0], 0.0, places=6)

    def test_gyro_scale_500dps(self):
        # 65.5 LSB per deg/s at +/-500 dps
        raw = _pack([0, 0, 0, 0, 655, 0, 0])   # gx = 655 LSB at 500 dps
        s = decode_frame(raw, gyro_fs_dps=500, accel_fs_g=8, t=0.0)
        self.assertAlmostEqual(s.gyro[0], 10.0 * DEG, places=6)

    def test_negative_values_are_signed(self):
        raw = _pack([-4096, 0, 0, 0, 0, 0, 0])
        s = decode_frame(raw, gyro_fs_dps=500, accel_fs_g=8, t=0.0)
        self.assertAlmostEqual(s.accel[0], -G, places=4)

    def test_temperature(self):
        raw = _pack([0, 0, 0, 340 * 10, 0, 0, 0])   # 10 C above the 36.53 reference
        s = decode_frame(raw, t=0.0)
        self.assertAlmostEqual(s.temp_c, 10 + 36.53, places=3)

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            decode_frame(b"\x00" * 12)


class FakeMPU(object):
    """Register-level fake: WHO_AM_I = 0x68, and writes are recorded."""
    def __init__(self):
        self.regs = {0x75: 0x68}
        self.writes = {}

    def read(self, reg, n):
        if reg == 0x75:
            return bytes([0x68])
        return bytes(14)

    def write(self, reg, val):
        self.writes[reg] = val


class TestDriverConfig(unittest.TestCase):
    def test_range_bits_written(self):
        fake = FakeMPU()
        MPU6050(fake.read, fake.write, sample_rate_hz=200, gyro_fs_dps=500, accel_fs_g=8, dlpf=3)
        self.assertEqual(fake.writes[0x1B], 1 << 3)     # GYRO_CONFIG FS_SEL=1 (500 dps)
        self.assertEqual(fake.writes[0x1C], 2 << 3)     # ACCEL_CONFIG AFS_SEL=2 (8 g)
        self.assertEqual(fake.writes[0x6B], 0x00)       # wake

    def test_wrong_device_rejected(self):
        def read(reg, n):
            return bytes([0x00])
        with self.assertRaises(RuntimeError):
            MPU6050(read, lambda r, v: None)


class TestAllan(unittest.TestCase):
    def test_white_noise_density_recovered(self):
        rng = np.random.default_rng(0)
        fs, sigma = 100.0, 0.02                  # per-sample std
        x = rng.normal(0, sigma, 200000)
        taus, adev = allan_deviation(x, fs)
        N = noise_density(taus, adev, fs)
        expected = sigma * math.sqrt(2.0 / fs)
        self.assertLess(abs(N - expected) / expected, 0.05)

    def test_white_noise_slope_is_minus_half(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1.0, 100000)
        taus, adev = allan_deviation(x, 100.0)
        slope = np.polyfit(np.log(taus[:10]), np.log(adev[:10]), 1)[0]
        self.assertAlmostEqual(slope, -0.5, delta=0.05)

    def test_bias_instability_detected_on_random_walk_plus_white(self):
        rng = np.random.default_rng(2)
        n = 200000
        white = rng.normal(0, 0.01, n)
        walk = np.cumsum(rng.normal(0, 0.0005, n))
        x = white + walk
        taus, adev = allan_deviation(x, 100.0)
        tau_min, adev_min = bias_instability(taus, adev)
        self.assertGreater(tau_min, 0.0)
        self.assertLess(adev_min, adev[0])


if __name__ == "__main__":
    unittest.main()
