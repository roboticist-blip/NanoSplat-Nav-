"""Pose frame tests: known-answer CRC, round trip, and corruption detection."""
import math
import struct
import unittest

from nanosplat.output.pose_frame import (encode, decode, crc16_ccitt_false,
                                         PoseEstimate, SIZE, FLAG_VALID)


class TestCrc(unittest.TestCase):
    def test_known_answer(self):
        # Standard check value for CRC-16/CCITT-FALSE
        self.assertEqual(crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_empty(self):
        self.assertEqual(crc16_ccitt_false(b""), 0xFFFF)


class TestPoseFrame(unittest.TestCase):
    def _est(self, valid=True):
        h = math.radians(30) / 2
        return PoseEstimate(seq=42, t=1234.5678, position=(1.25, -0.5, 0.0),
                            quat=(math.cos(h), 0.0, 0.0, math.sin(h)),
                            velocity=(0.4, 0.0, 0.0), quality=0.9, valid=valid)

    def test_size(self):
        self.assertEqual(SIZE, 74)
        self.assertEqual(len(encode(self._est())), 74)

    def test_round_trip(self):
        e = self._est()
        d = decode(encode(e))
        self.assertEqual(d.seq, 42)
        self.assertAlmostEqual(d.t, 1234.5678, places=9)
        for a, b in zip(d.position, e.position):
            self.assertAlmostEqual(a, b, places=9)
        for a, b in zip(d.quat, e.quat):
            self.assertAlmostEqual(a, b, places=6)
        self.assertTrue(d.valid)
        self.assertAlmostEqual(d.quality, 0.9, places=6)

    def test_invalid_flag_round_trips(self):
        self.assertFalse(decode(encode(self._est(valid=False))).valid)

    def test_corruption_detected(self):
        buf = bytearray(encode(self._est()))
        buf[20] ^= 0x01
        with self.assertRaises(ValueError):
            decode(bytes(buf))

    def test_bad_magic_rejected(self):
        buf = bytearray(encode(self._est()))
        buf[0] = ord("X")
        with self.assertRaises(ValueError):
            decode(bytes(buf))

    def test_non_unit_quaternion_rejected(self):
        with self.assertRaises(ValueError):
            PoseEstimate(0, 0.0, (0, 0, 0), (2, 0, 0, 0), (0, 0, 0), 1.0, True)

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            decode(b"\x00" * 10)


if __name__ == "__main__":
    unittest.main()
