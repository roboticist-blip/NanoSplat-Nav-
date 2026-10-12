# Setup guide: Jetson Nano (JetPack 4.6)

This guide targets the Jetson Nano (4 GB) with JetPack 4.6.x. JetPack 4.6 is the last release for the Nano.
It ships Ubuntu 18.04, Python 3.6.9, CUDA 10.2, TensorRT 8.2 and OpenCV 4.1.

Check the versions at each step. If a version differs, record it and report it, because the wheel
versions in this guide are pinned to the Python 3.6 toolchain.

## 1. Flash the device

1. On an x86 Ubuntu host, install NVIDIA SDK Manager.
2. Select the Jetson Nano, then flash JetPack 4.6.x (the latest 4.6 patch release).
3. On first boot, complete the Ubuntu setup and create a user.

Verify:
```bash
cat /etc/nv_tegra_release    # should mention R32 (JetPack 4.6)
python3 --version            # expect 3.6.9
```

## 2. Memory and swap

The Nano has 4 GB shared between CPU and GPU. Add swap before installing anything heavy.
Use `zram` or a swap file of at least 4 GB:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Set the power mode and clocks when benchmarking:
```bash
sudo nvpmodel -m 0          # MAXN power mode
sudo jetson_clocks          # fixed maximum clocks
```

## 3. Python packages

Install pip for Python 3.6. The last pip release that supports Python 3.6 is 21.3.1:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
python3 -m pip install --user "pip==21.3.1"
```

NumPy 1.19.5 is the last series for Python 3.6. Building it on aarch64 can take a long time and
needs swap:
```bash
python3 -m pip install --user "numpy==1.19.5"
python3 -c "import numpy; print(numpy.__version__)"
```

Install `smbus2` (device only):
```bash
python3 -m pip install --user "smbus2==0.4.1"
```

## 4. OpenCV (use the JetPack build)

Do not install OpenCV with pip on the Nano. The JetPack OpenCV build includes the right CUDA/GStreamer
support. Check it is present:
```bash
python3 -c "import cv2; print(cv2.__version__)"    # expect 4.1.x
```

If it is missing: `sudo apt-get install -y python3-opencv`.

The code uses these OpenCV calls. Confirm each one exists in your version:
`cv2.calcOpticalFlowFarneback`, `cv2.solvePnPRansac`, `cv2.solvePnP`, `cv2.triangulatePoints`,
`cv2.findEssentialMat`, `cv2.recoverPose`, `cv2.remap`. Run `python3 scripts/check_env.py` to check them all at once.

## 4a. Check the environment

```bash
python3 scripts/check_env.py
```
Fix anything it reports before running the tests.

## 5. I2C access for the MPU6050

Find the bus and grant access:
```bash
sudo usermod -aG i2c $USER        # log out and back in afterwards
sudo apt-get install -y i2c-tools
sudo i2cdetect -y -r 1            # MPU6050 should appear at 0x68
```

If 0x68 does not appear, check wiring (SDA on pin 3, SCL on pin 5, 3.3 V and GND) and the AD0 pin
(low gives 0x68).

Quick hardware smoke test (after step 3):
```bash
cd nanosplat-nav
PYTHONPATH=src python3 -c "
from nanosplat.sensors.imu_mpu6050 import MPU6050, open_smbus
rd, wr = open_smbus(bus=1)
imu = MPU6050(rd, wr, sample_rate_hz=200)
for _ in range(5):
    print(imu.read_sample())
"
```
Expect `accel` near (0, 0, 9.8) when the sensor is flat and still.

## 6. Get the code and run the tests

```bash
cd ~
# copy the nanosplat-nav folder here (git clone once a remote exists)
cd nanosplat-nav
./scripts/run_tests.sh
```

All tests should pass. Record the output with the versions from steps 1 to 4. That record is the
first entry in the device test log.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SyntaxError` on import | Running on Python 3.7+ | Use `python3` (3.6.9); check `which python3` |
| `ModuleNotFoundError: numpy` | Installed for another interpreter | Reinstall with `python3 -m pip install --user` |
| `ImportError: smbus2` | Device package missing | Install per step 3 |
| `PermissionError` on `/dev/i2c-1` | User not in `i2c` group | Step 5, then log out and back in |
| `MPU6050 not found` | Wiring or AD0 | Check with `i2cdetect` |
| Build killed during NumPy install | Out of memory | Add swap (step 2), then retry |

## 8. Run the navigation software

Simulation (no hardware), which checks the whole chain against ground truth:
```bash
PYTHONPATH=src python3 scripts/run_nav.py --sim --seconds 60
```
Live IMU (after the sensor passes its diagnostic). Output is IMU-only until the camera is integrated:
```bash
PYTHONPATH=src python3 scripts/run_nav.py --imu --rate 100 --seconds 60 --udp 127.0.0.1:14550
```

To watch the output, run the listener in a second terminal on the same machine:
```bash
PYTHONPATH=src python3 scripts/listen_pose.py --port 14550
```
It prints position and velocity, checks every frame's CRC, and reports dropped frames at the end.

## 9. Test the camera

Find the camera first, then run the test:
```bash
ls /dev/video*                      # e.g. /dev/video0
v4l2-ctl --list-devices             # sudo apt-get install -y v4l-utils if missing
PYTHONPATH=src python3 scripts/test_camera.py --device 0 --seconds 10 --snapshot cam.jpg
```
The test reports the resolution it actually gets, the frame rate, whether the image is black or
frozen, and saves `cam.jpg`. Copy that file to your PC to check the picture.
Exit code 0 means all checks passed.

## 10. What is not yet set up

No camera capture is implemented, so this guide does not cover a camera. The USB camera and its
calibration come with the capture module (see the roadmap).
