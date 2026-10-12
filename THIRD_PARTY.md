# Third-party components

| Component | Licence | Used for | Status |
|---|---|---|---|
| NumPy | BSD-3-Clause | arrays and linear algebra | used |
| OpenCV | Apache-2.0 | optical flow, PnP, triangulation, image I/O | used (JetPack build on device) |
| smbus2 | MIT | I2C access to the MPU6050 | used on device only |
| MiDaS-Small | MIT | monocular relative depth | planned |
| NanoDet | Apache-2.0 (verify at source) | object detection | deferred (object mode only) |
| Ceres Solver | BSD-3-Clause | sliding-window optimisation | planned |
| DBoW2 | BSD-3-Clause | loop-closure place recognition | planned |
| Eigen | MPL-2.0 (file-level) | C++ linear algebra, if a native component is added | not used yet |

Excluded on licence grounds: OpenVINS, ORB-SLAM3, VINS-Mono (GPL-3.0), and Ultralytics YOLO (AGPL-3.0).
