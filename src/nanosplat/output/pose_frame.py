"""
Controller-agnostic pose frame: a fixed-size little-endian binary packet with CRC-16.

Layout (74 bytes, little-endian, no padding):
    2s  magic      b"NS"
    B   version    1
    B   flags      bit0 = estimate valid
    I   seq        frame counter, wraps at 2^32
    d   t          timestamp, seconds (monotonic clock of the sender)
    3d  position   x, y, z in metres, map/world frame
    4f  quat       w, x, y, z, unit quaternion (world-from-body)
    3f  velocity   m/s, world frame
    f   quality    0..1, estimator's own confidence
    H   crc16      CRC-16/CCITT-FALSE over all preceding bytes

The frame is transport-agnostic: send it over UDP, serial, or a shared file.
Adapters for ROS 2 or MAVLink convert from PoseEstimate and do not change this format.
"""
import math
import struct

MAGIC = b"NS"
VERSION = 1
FMT = "<2sBBId3d4f3ffH"
SIZE = struct.calcsize(FMT)            # 74
FLAG_VALID = 0x01
_QUAT_TOL = 1e-3


def crc16_ccitt_false(data):
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class PoseEstimate(object):
    __slots__ = ("seq", "t", "position", "quat", "velocity", "quality", "valid")

    def __init__(self, seq, t, position, quat, velocity, quality, valid):
        self.seq = int(seq) & 0xFFFFFFFF
        self.t = float(t)
        self.position = tuple(float(v) for v in position)
        self.quat = tuple(float(v) for v in quat)
        self.velocity = tuple(float(v) for v in velocity)
        self.quality = float(quality)
        self.valid = bool(valid)
        if len(self.position) != 3 or len(self.quat) != 4 or len(self.velocity) != 3:
            raise ValueError("position and velocity need 3 values, quat needs 4")
        norm = math.sqrt(sum(q * q for q in self.quat))
        if abs(norm - 1.0) > _QUAT_TOL:
            raise ValueError("quaternion is not unit length (|q| = %.6f)" % norm)
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")


def encode(est):
    flags = FLAG_VALID if est.valid else 0
    body = struct.pack("<2sBBId3d4f3ff", MAGIC, VERSION, flags, est.seq, est.t,
                       *(est.position + est.quat + est.velocity + (est.quality,)))
    return body + struct.pack("<H", crc16_ccitt_false(body))


def decode(buf):
    if len(buf) != SIZE:
        raise ValueError("frame must be %d bytes, got %d" % (SIZE, len(buf)))
    vals = struct.unpack(FMT, buf)
    magic, version, flags, seq, t = vals[0], vals[1], vals[2], vals[3], vals[4]
    if magic != MAGIC:
        raise ValueError("bad magic %r" % magic)
    if version != VERSION:
        raise ValueError("unsupported version %d" % version)
    crc_rx = vals[-1]
    if crc16_ccitt_false(buf[:-2]) != crc_rx:
        raise ValueError("CRC mismatch")
    position = vals[5:8]
    quat = vals[8:12]
    velocity = vals[12:15]
    quality = vals[15]
    return PoseEstimate(seq, t, position, quat, velocity, quality, bool(flags & FLAG_VALID))
