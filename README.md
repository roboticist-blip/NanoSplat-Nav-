# NanoSplat Nav

Real-time localisation and navigation mapping for ground robots and drones, running on a Jetson Nano
with one camera and one IMU. The output is a controller-agnostic pose stream and an obstacle map in metres.

> **Status: integrated but not yet validated on hardware.** The full software chain (IMU driver, navigation
> filter, pipeline, output, mapping) is implemented and tested in simulation. The **visual front-end**
> (camera frames to absolute pose measurements) and **camera-IMU calibration** are not yet built, and the
> camera and the MPU6050 have not yet been run together on the Nano. See the status table.

## Status

| Component | Path | Status | Verified by |
|---|---|---|---|
| Quaternion and rotation helpers | `src/nanosplat/math3d.py` | Implemented | unit tests |
| Error-state EKF (IMU propagation, bias estimation, pose and velocity updates) | `src/nanosplat/estimator/ekf.py` | Implemented | simulator tests: 3.5 cm RMS over 60 s |
| Initial attitude from gravity | `src/nanosplat/estimator/init.py` | Implemented | unit tests |
| Navigation pipeline (IMU in, PoseEstimate out, validity flag) | `src/nanosplat/pipeline/nav.py` | Implemented | simulator + UDP loopback test |
| UDP pose output | `src/nanosplat/pipeline/udp_sink.py` | Implemented | loopback test (same machine) |
| Pose frame protocol (74-byte, CRC-16) | `src/nanosplat/output/pose_frame.py` | Implemented | CRC known-answer and round-trip tests |
| 2D log-odds occupancy grid | `src/nanosplat/mapping/occupancy2d.py` | Implemented | synthetic wall tests |
| Synthetic rover simulator (ground truth) | `src/nanosplat/sim/synthetic.py` | Implemented | used by estimator tests |
| MPU6050 driver | `src/nanosplat/sensors/imu_mpu6050.py` | Implemented | fake-bus tests; **hardware not yet validated with a passing recording** |
| Allan variance and IMU health check | `src/nanosplat/calib/allan.py`, `imu_log.py` | Implemented | synthetic tests; the first real recording failed the health check |
| Noise parameters with provenance flag | `src/nanosplat/calib/noise_config.py` | Implemented (currently provisional values) | unit tests |
| USB camera capture with host timestamps | `src/nanosplat/camera/usb_camera.py` | Implemented | injected fake capture; hardware test script `scripts/test_camera.py` ready, **not yet run on a camera** |
| Live run script (sim and IMU modes) | `scripts/run_nav.py` | Implemented | sim mode verified; IMU mode not run on hardware |
| UDP pose listener (checks CRC and sequence gaps) | `scripts/listen_pose.py` | Implemented | verified against the simulator |
| Visual front-end (features, tracking, pose from camera) | — | **Not started** | — |
| Camera intrinsic and camera-IMU extrinsic calibration | — | **Not started** | — |
| Keyframes and loop closure | — | **Not started** | — |
| MiDaS depth on keyframes | — | **Not started** | — |
| 3D occupancy and ESDF for drones | — | **Not started** | — |
| ROS 2 and MAVLink adapters | — | **Deferred** (board and controller decision) | — |

Tests run on the Nano's toolchain (Python 3.6.9, NumPy 1.19.5, OpenCV 4.1.1). Run `./scripts/run_tests.sh`.

## Quick start

```bash
./scripts/run_tests.sh                                   # environment check, then all tests
PYTHONPATH=src python3 scripts/run_nav.py --sim --seconds 60 --udp 127.0.0.1:14550
```

For the device setup, follow [docs/SETUP.md](docs/SETUP.md).

## How the pieces fit

```
MPU6050 ──► ImuEkf.propagate ─────────────┐
                                          ▼
Camera ──► (visual front-end, TODO) ──► ImuEkf.update_pose ──► NavPipeline ──► UDP pose frames
                                          ▲                       │
                                          └── velocity updates    └──► OccupancyGrid2D (planned input: range data)
```

The pipeline calls the filter's propagation on every IMU sample, so the output rate follows the IMU.
Absolute corrections arrive when a measurement is available. Without them, `valid` is 0 and the
estimate is IMU-only, which drifts quickly (over 5 m after 60 s in simulation).

## Repository layout

```
src/nanosplat/
  math3d.py         quaternions, rotation vectors
  estimator/        error-state EKF, initial attitude
  pipeline/         navigation pipeline, UDP output
  output/           pose frame protocol
  sensors/          IMU types and MPU6050 driver
  camera/           USB camera wrapper
  calib/            Allan variance, IMU log format and health check, noise parameters
  mapping/          2D occupancy grid
  sim/              synthetic rover with ground truth
  map/ segment/ pose/ depth/   earlier mapping primitives (see docs/DESIGN.md)
tests/              unit and integration tests (no hardware needed)
scripts/            run_tests.sh, check_env.py, run_nav.py, record_imu.py, allan_report.py
docs/               setup, design, protocol, calibration, roadmap, rover test plan, v1 audit
```

## Documentation

- [docs/SETUP.md](docs/SETUP.md): Jetson Nano setup, Python 3.6, OpenCV, I2C, running tests
- [docs/DESIGN.md](docs/DESIGN.md): system architecture and requirements
- [docs/PROTOCOL.md](docs/PROTOCOL.md): pose frame wire format
- [docs/CALIBRATION.md](docs/CALIBRATION.md): IMU noise, camera intrinsics, camera-IMU extrinsics
- [docs/ROVER_TEST_PLAN.md](docs/ROVER_TEST_PLAN.md): test ladder with pass criteria
- [docs/ROADMAP.md](docs/ROADMAP.md): build order and open decisions
- [docs/V1_AUDIT.md](docs/V1_AUDIT.md): defects found in the earlier object-scan code

## Licence

Not yet chosen. Do not publish or distribute until a licence is selected. Third-party components are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
