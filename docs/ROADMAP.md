# Roadmap

Each stage has a pass criterion, measured on the device, before the next stage starts.

## Done
1. Spatial voxel fusion, belief mask, pose geometry, adaptive depth alignment (earlier mapping primitives)
2. MPU6050 driver, Allan variance, IMU health check, noise parameters with provenance
3. Pose frame protocol (74 bytes, CRC-16)
4. Quaternion math and error-state EKF with bias estimation (simulator: 3.5 cm RMS over 60 s with pose updates)
5. Navigation pipeline, UDP output, validity flag (loopback tested)
6. 2D occupancy grid (synthetic tests)
7. USB camera wrapper (injected-capture tested)
8. Run script for simulation and live IMU

## Next, on the device
9. **Passing IMU recording.** Fix the faulty gyro axis and the wiring, run the diagnostic, record 30 minutes, run `allan_report.py`. Pass: health check passes, and the noise numbers replace the provisional values.
10. **Live IMU run.** `run_nav.py --imu`. Pass: effective rate at or above 100 Hz, and a still sensor reports position near zero for 60 s without drift beyond the IMU-only budget.
11. **Camera on the Nano.** `UsbCamera` at 640×480, 20+ fps, with no dropped-frame growth over 10 minutes.
12. **Camera intrinsics and camera-IMU extrinsics with time offset.** Pass: reprojection error below 1 px; time-offset estimate repeatable to within 5 ms.

## Next, in software
13. **Visual front-end.** Shi-Tomasi features, KLT tracking, and a pose measurement against keyframe landmarks (triangulation and PnP already exist in `pose/geometry.py`). Output: `on_pose` measurements at camera rate. Pass: on the rover test T2, position error under 5%.
14. **Keyframes and loop closure.** Pass on rover test T4.
15. **MiDaS depth on keyframes** (TensorRT FP16 or ONNX), aligned to metric landmarks. Pass: latency within budget.
16. **Obstacle mapping from depth.** 2D grid first, then 3D ESDF. Pass: obstacles placed within 10 cm on a known layout.
17. **Adapters.** ROS 2 and MAVLink, once the board and controller are chosen.

## Open decisions
- Board: stay on Jetson Nano (JetPack 4.6, Python 3.6, no modern ROS 2), or move to an Orin Nano for flight.
- IMU: MPU6050 for rover tests, then BMI270.
- Camera: USB, or CSI with a verified pinout.
- Scale on rover: add wheel encoders if the test ladder shows scale drift.
- Controller: undecided. The output is generic until one is chosen.
- Project licence.
