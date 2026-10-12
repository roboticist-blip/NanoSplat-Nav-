"""IMU log format and noise report tests. No hardware needed."""
import os
import tempfile
import unittest

import numpy as np

from nanosplat.calib.imu_log import (write_csv, read_csv, sample_rate, noise_report, format_report,
                                     health_check, GYRO_FULL_SCALE)


def _synthetic_log(n=60000, fs=100.0, sig_a=0.02, sig_g=0.001, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    rows = []
    cols = {
        "ax": rng.normal(0, sig_a, n), "ay": rng.normal(0, sig_a, n), "az": 9.81 + rng.normal(0, sig_a, n),
        "gx": rng.normal(0, sig_g, n), "gy": rng.normal(0, sig_g, n), "gz": rng.normal(0, sig_g, n),
    }
    for i in range(n):
        rows.append((t[i], cols["ax"][i], cols["ay"][i], cols["az"][i],
                     cols["gx"][i], cols["gy"][i], cols["gz"][i], 25.0))
    return rows


class TestLogFormat(unittest.TestCase):
    def test_round_trip(self):
        rows = _synthetic_log(n=200)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "log.csv")
            write_csv(p, rows)
            log = read_csv(p)
        self.assertEqual(len(log["t"]), 200)
        self.assertAlmostEqual(log["az"][0], rows[0][3], places=5)

    def test_wrong_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.csv")
            with open(p, "w") as f:
                f.write("a,b,c\n1,2,3\n")
            with self.assertRaises(ValueError):
                read_csv(p)


class TestRateAndReport(unittest.TestCase):
    def test_rate_measured(self):
        t = np.arange(1000) / 100.0
        fs, jitter = sample_rate(t)
        self.assertAlmostEqual(fs, 100.0, places=6)
        self.assertLess(jitter, 1e-9)

    def test_non_monotonic_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            sample_rate(np.array([0.0, 0.01, 0.005]))

    def test_noise_report_recovers_known_noise(self):
        rows = _synthetic_log(n=60000, fs=100.0, sig_a=0.02, sig_g=0.001)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "still.csv")
            write_csv(p, rows)
            log = read_csv(p)
        fs, jitter, out = noise_report(log)
        by = {r["axis"]: r for r in out}
        expect_a = 0.02 * np.sqrt(2.0 / 100.0)
        expect_g = 0.001 * np.sqrt(2.0 / 100.0)
        self.assertLess(abs(by["ax"]["noise_density"] - expect_a) / expect_a, 0.08)
        self.assertLess(abs(by["gz"]["noise_density"] - expect_g) / expect_g, 0.08)

    def test_report_is_markdown_table(self):
        rows = _synthetic_log(n=20000)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.csv")
            write_csv(p, rows)
            log = read_csv(p)
        fs, jitter, out = noise_report(log)
        text = format_report(fs, jitter, out, 200.0)
        self.assertIn("| Axis |", text)
        self.assertIn("| gz |", text)


if __name__ == "__main__":
    unittest.main()


class TestHealthCheck(unittest.TestCase):
    def _log(self, **over):
        rng = np.random.default_rng(3)
        n = 60000
        base = {
            "t": np.arange(n) / 100.0,
            "ax": rng.normal(0, 0.02, n), "ay": rng.normal(0, 0.02, n),
            "az": 9.81 + rng.normal(0, 0.02, n),
            "gx": rng.normal(0, 0.001, n), "gy": rng.normal(0, 0.001, n),
            "gz": rng.normal(0, 0.001, n), "temp_c": np.full(n, 25.0),
        }
        base.update(over)
        return base

    def test_healthy_log_passes(self):
        self.assertEqual([s for s, _ in health_check(self._log()) if s == "FAIL"], [])

    def test_gyro_saturation_fails(self):
        g = np.random.default_rng(4).normal(0, 0.001, 60000)
        g[20000:] = GYRO_FULL_SCALE
        issues = health_check(self._log(gy=g))
        self.assertTrue(any(s == "FAIL" and "gy" in m for s, m in issues))

    def test_accel_magnitude_offset_warns(self):
        issues = health_check(self._log(az=np.full(60000, 12.5) + np.random.default_rng(5).normal(0, 0.02, 60000)))
        self.assertTrue(any(s == "WARN" and "magnitude" in m for s, m in issues))

    def test_repeated_samples_warn(self):
        log = self._log()
        for k in ("ax", "ay", "az", "gx", "gy", "gz"):
            log[k][1000:1100] = log[k][999]
        issues = health_check(log)
        self.assertTrue(any("identical" in m for s, m in issues))
