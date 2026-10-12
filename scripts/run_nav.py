#!/usr/bin/env python3
"""
Run the navigation pipeline.

  Simulation (no hardware, checks the whole chain against ground truth):
      PYTHONPATH=src python3 scripts/run_nav.py --sim --seconds 60 --udp 127.0.0.1:14550

  Live IMU (MPU6050 on the Jetson). Pose measurements from the camera are NOT yet wired in,
  so the output is IMU-only and `valid` stays 0. This verifies the sensor, output and timing path:
      PYTHONPATH=src python3 scripts/run_nav.py --imu --rate 100 --udp 127.0.0.1:14550
"""
import argparse
import math
import sys
import time

import numpy as np

from nanosplat.estimator.ekf import ImuEkf
from nanosplat.estimator.init import initial_quat_from_gravity
from nanosplat.pipeline.nav import NavPipeline
from nanosplat.pipeline.udp_sink import UdpPoseSink
from nanosplat.calib.noise_config import provisional_mpu6050

SIG_A_DENSITY, SIG_G_DENSITY = 3.9e-3, 8.7e-5     # provisional (datasheet); see calib/noise_config.py


def parse_udp(s):
    host, port = s.rsplit(":", 1)
    return host, int(port)


def run_sim(args):
    from nanosplat.sim.synthetic import simulate
    data = simulate(duration=args.seconds, seed=args.seed)
    t0, p0, v0, q0 = data["truth"][0]
    ekf = ImuEkf(p0, v0, q0, SIG_A_DENSITY, SIG_G_DENSITY)
    sink = UdpPoseSink(*parse_udp(args.udp)) if args.udp else None
    pipe = NavPipeline(ekf, sink=sink, keep_history=True)
    poses = {round(t, 6): (p, q) for t, p, q in data["pose"]}
    for t, acc, gyro in data["imu"]:
        pipe.on_imu(t, acc, gyro)
        k = round(t, 6)
        if k in poses:
            pipe.on_pose(t, poses[k][0], poses[k][1])
    if sink:
        sink.close()
    est = np.array([e.position for e in pipe.history])
    gt = np.array([s[1] for s in data["truth"]])
    err = np.linalg.norm(est - gt, axis=1)
    print("samples: %d   RMS position error: %.3f m   final error: %.3f m" %
          (len(est), math.sqrt(np.mean(err ** 2)), err[-1]))
    return 0


def run_imu(args):
    from nanosplat.sensors.imu_mpu6050 import MPU6050, open_smbus
    rd, wr = open_smbus(bus=args.bus)
    imu = MPU6050(rd, wr, sample_rate_hz=int(args.rate))
    sink = UdpPoseSink(*parse_udp(args.udp)) if args.udp else None
    print("Keep the vehicle still for 3 s while the start attitude is set from gravity.")
    still = []
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        s = imu.read_sample()
        still.append(s.accel)
    q0 = initial_quat_from_gravity(np.array(still))
    ekf = ImuEkf(np.zeros(3), np.zeros(3), q0, SIG_A_DENSITY, SIG_G_DENSITY)
    pipe = NavPipeline(ekf, sink=sink)
    period = 1.0 / args.rate
    t0 = time.monotonic()
    n = 0
    print("Running IMU-only at %.0f Hz for %.0f s (Ctrl-C to stop)" % (args.rate, args.seconds))
    try:
        while time.monotonic() - t0 < args.seconds:
            s = imu.read_sample()
            pipe.on_imu(s.t, s.accel, s.gyro)
            n += 1
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    if sink:
        sink.close()
    elapsed = time.monotonic() - t0
    print("samples: %d over %.1f s (%.1f Hz effective)" % (n, elapsed, n / elapsed))
    return 0


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sim", action="store_true")
    mode.add_argument("--imu", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--udp", default=None, help="host:port for pose frames")
    args = ap.parse_args()
    return run_sim(args) if args.sim else run_imu(args)


if __name__ == "__main__":
    sys.exit(main())
