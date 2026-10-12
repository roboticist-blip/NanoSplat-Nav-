# Calibration

Three calibrations are needed before the estimator can run. Each is either implemented or planned:

| Calibration | Status | Tool |
|---|---|---|
| IMU noise (noise density, bias instability) | Implemented (`calib/allan.py`) | stationary recording, 30 min minimum |
| Camera intrinsics and distortion | Planned | checkerboard, OpenCV (Apache-2.0) |
| Camera-IMU extrinsics and time offset | Planned | rotation sequence, Kalibr-style (check licence first) |

## IMU noise (implemented)

Record and report in two commands:
```bash
PYTHONPATH=src python3 scripts/record_imu.py --minutes 30 --rate 100 --out imu_still.csv
PYTHONPATH=src python3 scripts/allan_report.py imu_still.csv
```

Known caveat: the recorder polls the sensor from Python, so the host introduces timing jitter. The
report prints the measured jitter. If it is large (more than a few ms), close other programs and
rerun, and treat the noise numbers as approximate.


1. Place the MPU6050 still on a stable surface, away from fans and vibration.
2. Log raw samples at the driver's rate for at least 30 minutes (2 to 3 hours is better).
3. Compute the Allan deviation of the gyro and accel axes:
   ```python
   from nanosplat.calib.allan import allan_deviation, noise_density, bias_instability
   taus, adev = allan_deviation(gyro_x, fs=200.0)
   N = noise_density(taus, adev, fs=200.0)          # white noise density
   tau_b, bi = bias_instability(taus, adev)         # bias instability
   ```
4. Record the results per axis, with temperature. Use them as the estimator's noise parameters.

The MPU6050 datasheet typical values are about 0.005 deg/s/√Hz for the gyro and about 400 µg/√Hz for the
accel. Use your measured values, not the datasheet values.

## Camera intrinsics (planned)

Use a printed checkerboard, with several views covering the image edges and several tilts. Estimate the
camera matrix and distortion coefficients. Record the image size used for calibration, because the
intrinsics are only valid at that resolution.

## Camera-IMU extrinsics and time offset (planned)

The USB camera is not hardware-synchronised with the IMU. The calibration must estimate:
- the rigid transform from IMU to camera frame (rotation and translation), and
- the time offset between the camera and IMU clocks.

Excite the rotations along all three axes and keep the camera pointed at a textured target. Store the
results in one file that the pipeline loads at start-up.
