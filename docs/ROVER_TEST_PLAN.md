# Rover-first plan (Jetson Nano + MPU6050 + USB camera)

## Hardware reality
- **MPU6050:** 6-axis, I2C at 0x68, no magnetometer, no hardware timestamps. Gyro noise is
  about 0.005 deg/s/sqrt(Hz) and accel about 400 ug/sqrt(Hz) per datasheet, so calibrate it
  on the device. Expect gyro bias to shift with temperature. Plan to swap for BMI270 later:
  the driver interface (`ImuSample`) stays the same.
- **USB camera:** UVC, usually rolling shutter, and not hardware-synced to the IMU. Timestamps
  arrive with variable host latency, so the camera-IMU time offset must be estimated, not assumed.
- **CSI camera:** the IMX219 modules shipped for the Jetson Nano work through nvarguscamerasrc.
  Raspberry Pi camera modules often don't work directly, because the ribbon pinout differs. Check
  the module's pinout before buying. The USB camera is enough for the rover tests.

## Scale on a rover: the observability caveat
The IMU observes scale only through acceleration changes. A rover driving at constant speed gives
almost no scale information, so drift in scale can grow unnoticed. Mitigations, in order of cost:
1. Drive with regular starts, stops and turns during calibration runs. This excites scale.
2. Add wheel encoders on the drive motors. They give speed directly and are cheap.
3. Use the rover's known wheel geometry as a metric reference.

## Calibration sequence (before any mapping)
1. IMU static recording, 30 min minimum: Allan variance gives noise density and bias instability
   (`calib/allan.py`). These set the estimator's noise parameters.
2. Camera intrinsics and distortion, from a checkerboard (OpenCV, Apache-2.0).
3. Camera-IMU extrinsics and time offset, from a shaken or rotated sequence (Kalibr-style method;
   check Kalibr's licence before using its code).

## Rover test ladder (pass criteria before moving on)
| Test | Pass criterion |
|---|---|
| T1 Static, 10 min | Pose drift under 1 cm, yaw drift under 1 deg/min |
| T2 Straight line, 5 m, out and back | Position error under 5% of distance |
| T3 Square loop, 10 m per side, return to start | End-point error under 10% |
| T4 Loop in a room with loop closure enabled | End-point error under 3% |

Each test is recorded to disk first, so the estimator can be replayed offline and tuned without
the rover running.

## Generic output interface (flight-controller agnostic)
The estimator publishes a single internal message, `PoseEstimate`, containing a timestamp,
position (m), orientation (quaternion), velocity (m/s), an estimate-quality value, and a validity
flag. Transport adapters sit on top of it:
- UDP or serial binary frame with a fixed header and CRC (works with any controller or companion computer)
- ROS 2 `nav_msgs/Odometry` (on a compatible OS, see the board note in the design doc)
- MAVLink adapter, added only when a flight controller is chosen

Nothing above the adapter layer knows which controller is connected.
