# NanoSplat v3 — Real-Time Navigation Mapping (Design)

Purpose: run on a small embedded computer on a drone or ground robot, using one camera and one IMU,
and produce a map a planner can use: a pose, an obstacle representation in metres, and an export
to the flight controller or ROS 2 navigation stack.

Session types: indoor rooms, factories, and open outdoor areas (including nadir or oblique views from a drone).

This document replaces the object-scan design of v1/v2. The v2 core modules are reused where
they fit (see Section 6).

---

## 1. Requirements

| Requirement | Target (to be measured on hardware) |
|---|---|
| Pose output | 100 Hz+ from IMU propagation, camera-corrected |
| Metric scale | Yes, from IMU (no user-entered distance) |
| Local obstacle map | Metric, updated at 5–10 Hz, short range (about 5–10 m) |
| Global map | Persistent pose graph with loop closure, drift-corrected |
| Drift | Room scale: a few cm; open area: stated as % of distance travelled |
| Output | Pose and velocity to flight controller or ROS 2; occupancy grid (2D) and voxel/ESDF (3D) |
| Priority | Reconstruction and localisation quality, then frame rate, then memory |

---

## 2. Architecture

Two loops run in parallel. The real-time loop must never wait for the slow loop.

```
IMU (200–400 Hz) ──► Preintegration ──────────────────────────┐
                                                              ▼
Camera (20–30 Hz) ─► KLT tracking ─► Feature bucketing ─► ESTIMATOR (MSCKF-style EKF)
                         │                                    │  pose, velocity, biases
                         │                                    ▼  (every IMU sample)
                         │                           Real-time pose output ──► flight controller / ROS 2
                         │                                    │
                         │             Keyframe selection (parallax, time, track loss)
                         │                                    │  (async)
                         ▼                                    ▼
                    Descriptors ──────────────► BACKEND (pose graph, loop closure)
                                                              │
                              MiDaS depth on keyframes ───────┤
                              aligned to metric landmarks     ▼
                                                    MAPPING: log-odds occupancy
                                                    (local window + global), ESDF
                                                              │
                                                 Export: OccupancyGrid, voxel, PLY
```

### 2.1 Real-time loop (every camera frame or IMU sample)

- **IMU preintegration.** Integrate gyro and accelerometer between camera frames with bias states.
  Gives metric scale and gravity direction (roll and pitch observable), which fixes the scale ambiguity
  that rules out pure monocular methods.
- **Feature tracking.** Shi–Tomasi or FAST corners, grid-bucketed for even spread, tracked with
  pyramidal Lucas–Kanade optical flow. This is cheaper on the CPU than descriptor matching for frame-to-frame tracking.
- **Estimator (MSCKF-style EKF).** The state holds IMU pose, velocity, biases, and a sliding window of
  past camera poses (clones). Feature tracks that span several frames are used as measurements and then
  discarded, so the cost per frame stays fixed. That suits the Nano's budget. Outputs pose at IMU rate.

### 2.2 Slow loop (keyframes, asynchronous)

- **Keyframe selection.** A new keyframe when the camera has moved enough (parallax), enough time has passed,
  or tracked features drop below a threshold.
- **Backend.** A pose graph built from keyframes. Loop closure by a bag-of-words place-recognition index
  (DBoW2-style, BSD-3) on ORB descriptors computed only for keyframes. Optimised with a sparse solver
  (Ceres or g2o, both permissive).
- **Dense depth on keyframes.** MiDaS-Small gives relative depth. The affine fit is anchored to metric
  landmarks from the estimator, so depth is metric (this is the adaptive alignment from v2, now with a
  metric anchor from the IMU). Each depth estimate carries a residual-based confidence.

### 2.3 Mapping

- **Local obstacle map.** Log-odds occupancy in a voxel grid centred on the vehicle, resolution
  5–10 cm indoors and 0.2–0.5 m outdoors. Each keyframe's aligned depth is ray-cast into the grid.
  Only the window around the vehicle is kept active, so memory is bounded.
- **Global map.** Keyframe-attached occupancy tiles, re-anchored when the pose graph is corrected after
  loop closure. Tiles are stored and reloaded as needed.
- **Planner outputs.** A 2D occupancy grid for ground robots, and an ESDF (signed distance field) for drones.
- **Visual map (optional).** The v2 Gaussian/voxel colour map remains available as a viewer layer, but it is
  not used for planning.

---

## 3. Interfaces

| Output | Format | Consumer |
|---|---|---|
| Pose + velocity (100 Hz) | `odometry` (ROS 2 nav_msgs) or MAVLink `ODOMETRY` / `VISION_POSITION_ESTIMATE` | Nav2 / flight controller (PX4, ArduPilot) |
| Occupancy grid (2D, 2 Hz) | `nav_msgs/OccupancyGrid` | ground robot planner |
| ESDF / voxel (3D, 2 Hz) | custom or octomap-style | drone planner |
| Keyframe graph | JSON / binary | offline review, relocalisation |
| Colour map | PLY | visualisation |

Timestamps are synchronised between camera and IMU in hardware or by a shared clock. Camera-IMU
extrinsics and time offset must be calibrated offline and loaded at start-up.

---

## 4. Real-time budget (targets; to be measured)

Estimated on the Nano's class of hardware. Measured values replace these numbers when available.

| Stage | Rate | Budget |
|---|---|---|
| IMU preintegration | 200–400 Hz | < 0.1 ms per sample |
| KLT tracking, 200 features, 640×480 | 20–30 Hz | ~5–10 ms |
| Estimator update | 20–30 Hz | ~2–5 ms |
| Keyframe selection | per frame | < 0.5 ms |
| Descriptors + BoW query | per keyframe, 2–5 Hz | ~10–20 ms |
| MiDaS-Small on keyframe | 2–5 Hz | 30–100 ms (async) |
| Occupancy ray-cast, local window | 5–10 Hz | 10–30 ms (async) |
| Backend optimisation | on loop closure | async, no real-time cost |

The real-time output (pose) depends only on the first three rows. The slow loop can fall behind
without stopping pose output. That is the main design guarantee.

---

## 5. Licensing

Avoid GPL-licensed VIO and SLAM code for a commercial product. OpenVINS and ORB-SLAM3 are GPL-3.0, and VINS-Mono is GPL-3.0.
Write the MSCKF-style estimator in-house, or use a permissive library after checking its licence at source.

| Component | Licence | Use |
|---|---|---|
| OpenCV (core, Lucas–Kanade, Farnebäck) | Apache-2.0 | yes |
| Eigen | MPL-2.0 (file-level) | yes |
| Ceres Solver | BSD-3 | yes (backend) |
| g2o | BSD-2 | yes (alternative backend) |
| DBoW2 | BSD-3 | yes (loop closure) |
| GTSAM | BSD-3 | candidate (backend) |
| Sophus | MIT | yes (Lie groups) |
| MiDaS-Small | MIT | yes |
| NanoDet | Apache-2.0 (verify at source) | object mode only |
| ROS 2 packages | Apache-2.0 | yes |
| GPL VIO / SLAM (OpenVINS, ORB-SLAM3, VINS-Mono) | GPL-3.0 | **no**, unless licensing is accepted |

---

## 6. What happens to v1 and v2 modules

| Module | Status in v3 |
|---|---|
| `map/voxel_fusion.py` | kept, as the visual colour map layer |
| `segment/belief_mask.py` | kept for object mode and for masking dynamic regions |
| `pose/geometry.py` | kept for triangulation and PnP relocalisation; the chaining is replaced by the estimator |
| `depth/alignment.py` | kept, now anchored to metric landmarks from the IMU-aided estimator |
| v1 tracker (MOSSE) and K-means segmentation | dropped from the navigation path |
| v1 ring buffer, Python-loop insertion | dropped |

---

## 7. Failure modes and safety

- **Feature-poor scenes** (white walls, sky, water, uniform ground): the estimator degrades to IMU-only,
  which drifts quickly. The system must report a confidence and must stop claiming a valid pose after a set time.
- **Fast motion or blur:** KLT loses tracks. Mitigation: IMU propagation carries the pose across the gap,
  and tracks are re-detected on recovery.
- **Monocular depth is not a certified obstacle sensor.** Use it to supplement a proper ranging sensor
  (time-of-flight or lidar) for collision avoidance. Do not rely on it alone for safety-critical stops.
- **Outdoor drones:** GNSS fused with VIO corrects long-term drift. Recommended if flying outdoors.
- **Power and thermal limits:** the system must degrade, with the pose loop kept alive, when the slow loop is starved.

---

## 8. Open decisions

1. **Board.** The Jetson Nano is on JetPack 4.6 (Ubuntu 18.04). Modern ROS 2 needs Ubuntu 22.04 or
   newer, and the MSCKF estimator is light enough for any modern Jetson. A Jetson Orin Nano (JetPack 6,
   Ubuntu 22.04, 8 GB) would make ROS 2 and the backend much simpler, at a higher cost. Which board, or
   both (Nano for development benchmarks, Orin for flight)?
2. **IMU model and interface.** Rate, noise specs, and whether the IMU is hardware-timestamped. Its noise parameters drive the estimator's tuning.
3. **Camera.** Field of view, global or rolling shutter, and resolution. Rolling shutter needs modelling for fast motion.
4. **Flight controller.** PX4 or ArduPilot, and whether the vehicle is a drone, a ground robot, or both.
5. **GNSS.** Is GNSS available for outdoor sessions?

---

## 9. Roadmap

1. Calibration tooling: camera intrinsics and camera-IMU extrinsics with time offset.
2. Real-time loop: IMU preintegration, KLT tracking, MSCKF estimator. Benchmark on the chosen board. Test on
   recorded rosbag-style datasets before live hardware.
3. Keyframes and backend: pose graph and loop closure. Test drift on a loop-closing recording.
4. Mapping: local log-odds occupancy, 2D grid export first, then ESDF for drones.
5. Integration: ROS 2 odometry and OccupancyGrid, and MAVLink for the flight controller.
6. Field tests: indoor room, factory aisle, outdoor open area.

Each stage has a pass criterion measured on the device before the next stage starts.
