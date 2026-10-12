#!/usr/bin/env python3
"""
Noise report from a stationary IMU log.

    PYTHONPATH=src python3 scripts/allan_report.py imu_still.csv
Prints a markdown table of noise density and bias instability per axis.
Copy the numbers into the estimator configuration.
"""
import sys

from nanosplat.calib.imu_log import read_csv, noise_report, format_report, health_check


def main(path):
    log = read_csv(path)
    duration = log["t"][-1] - log["t"][0]
    if duration < 600:
        print("warning: only %.0f s recorded; 30 min or more is recommended" % duration)
    issues = health_check(log)
    for sev, msg in issues:
        print("[%s] %s" % (sev, msg))
    if any(sev == "FAIL" for sev, _ in issues):
        print("\nNoise numbers NOT produced: the log failed the health check. Fix the hardware and record again.")
        return 1
    fs, jitter, rows = noise_report(log)
    print(format_report(fs, jitter, rows, duration))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: allan_report.py <log.csv>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
