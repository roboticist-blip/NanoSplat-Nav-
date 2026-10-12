#!/usr/bin/env python3
"""
Environment check for NanoSplat Nav. Run before the tests on any new machine or device.

Verifies Python version, NumPy, OpenCV and every OpenCV function the code calls, and the
optional I2C package. Exits non-zero if a required item is missing.
"""
import sys

REQUIRED_CV2 = [
    "calcOpticalFlowFarneback", "remap", "findEssentialMat", "recoverPose",
    "triangulatePoints", "solvePnPRansac", "solvePnP", "Rodrigues",
]
REQUIRED_CONSTS = ["RANSAC", "SOLVEPNP_EPNP", "SOLVEPNP_ITERATIVE", "INTER_LINEAR"]

problems = []

print("Python  :", sys.version.split()[0])
if sys.version_info < (3, 6):
    problems.append("Python 3.6 or newer is required")

try:
    import numpy
    print("NumPy   :", numpy.__version__)
except ImportError:
    problems.append("numpy is not installed")
    numpy = None

try:
    import cv2
    print("OpenCV  :", cv2.__version__)
    for name in REQUIRED_CV2:
        if not hasattr(cv2, name):
            problems.append("cv2.%s is missing" % name)
    for name in REQUIRED_CONSTS:
        if not hasattr(cv2, name):
            problems.append("cv2.%s is missing" % name)
except ImportError:
    problems.append("OpenCV (cv2) is not installed")

try:
    import smbus2
    print("smbus2  : present (device I2C ok)")
except ImportError:
    print("smbus2  : not installed (needed only for the MPU6050 on the device)")

if problems:
    print("\nFAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nOK: environment supports every function NanoSplat Nav uses.")
