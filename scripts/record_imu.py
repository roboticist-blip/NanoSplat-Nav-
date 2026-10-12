#!/usr/bin/env python3
"""
Record a stationary MPU6050 log for Allan variance.

Place the sensor still on a stable surface, away from fans and vibration, then run:
    PYTHONPATH=src python3 scripts/record_imu.py --minutes 30 --rate 100 --out imu_still.csv

Samples are polled at the requested rate with host timestamps. The measured rate and the
interval jitter are printed; a large jitter means the host is busy and the noise numbers
are less reliable (see docs/CALIBRATION.md).
"""
import argparse
import sys
import time

from nanosplat.sensors.imu_mpu6050 import MPU6050, open_smbus
from nanosplat.calib.imu_log import write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=100.0, help="target polling rate, Hz")
    ap.add_argument("--out", default="imu_still.csv")
    ap.add_argument("--bus", type=int, default=1)
    args = ap.parse_args()

    rd, wr = open_smbus(bus=args.bus)
    imu = MPU6050(rd, wr, sample_rate_hz=int(args.rate))
    period = 1.0 / args.rate
    total = args.minutes * 60.0
    rows = []
    t0 = time.monotonic()
    next_t = t0
    print("Recording %.1f min at %.0f Hz -> %s (keep the sensor still)" % (args.minutes, args.rate, args.out))
    try:
        while True:
            now = time.monotonic()
            if now - t0 >= total:
                break
            if now < next_t:
                time.sleep(next_t - now)
                continue
            s = imu.read_sample()
            rows.append((now, s.accel[0], s.accel[1], s.accel[2],
                         s.gyro[0], s.gyro[1], s.gyro[2], s.temp_c))
            next_t += period
            if now - next_t > 10 * period:      # fell far behind: resynchronise
                next_t = now
    except KeyboardInterrupt:
        print("Stopped early.")
    write_csv(args.out, rows)
    print("Wrote %d samples to %s" % (len(rows), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
