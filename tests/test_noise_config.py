"""Noise parameter provenance: placeholders are flagged, faulty logs cannot calibrate."""
import unittest

from nanosplat.calib.noise_config import NoiseParams, provisional_mpu6050, from_measured, DEG, G


def _rows(nd=1e-4):
    axes = ["ax", "ay", "az", "gx", "gy", "gz"]
    return [{"axis": a, "noise_density": nd, "bias_instability": nd * 10} for a in axes]


class TestNoiseConfig(unittest.TestCase):
    def test_provisional_is_flagged_not_calibrated(self):
        p = provisional_mpu6050()
        self.assertEqual(p.source, "provisional")
        self.assertFalse(p.is_calibrated)
        self.assertAlmostEqual(p.gyro_nd, 0.005 * DEG, places=12)

    def test_failed_log_cannot_calibrate(self):
        with self.assertRaises(ValueError):
            from_measured(_rows(), [("FAIL", "gy is at full scale")])

    def test_warnings_do_not_block(self):
        p = from_measured(_rows(), [("WARN", "accelerometer magnitude off")])
        self.assertTrue(p.is_calibrated)

    def test_worst_axis_used(self):
        rows = _rows()
        rows[4]["noise_density"] = 9e-3          # gy noisier than the rest
        p = from_measured(rows, [])
        self.assertAlmostEqual(p.gyro_nd, 9e-3)

    def test_json_round_trip(self):
        p = provisional_mpu6050()
        q = NoiseParams.from_json(p.to_json())
        self.assertEqual(q.source, "provisional")
        self.assertAlmostEqual(q.accel_nd, p.accel_nd)

    def test_invalid_source_rejected(self):
        with self.assertRaises(ValueError):
            NoiseParams(1, 1, 1, 1, source="guess")


if __name__ == "__main__":
    unittest.main()
