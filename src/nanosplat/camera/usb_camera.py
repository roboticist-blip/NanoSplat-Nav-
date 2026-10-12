"""
USB (UVC) camera capture with host timestamps.

Hardware status: NOT verified on the Jetson Nano. The code path is tested here with an injected fake
capture. The timestamp is the host time immediately after the frame is read; the true exposure time
is earlier by an unknown, variable latency. That offset must be estimated during camera-IMU
calibration (see docs/CALIBRATION.md) before visual-inertial fusion uses these times.
"""
import time

import cv2


class UsbCamera(object):
    def __init__(self, device=0, width=640, height=480, fps=30, capture_factory=None):
        self._cap = capture_factory() if capture_factory else cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError("camera %r could not be opened" % (device,))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        """Returns (t_monotonic, frame_bgr). Raises RuntimeError if a frame cannot be read."""
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok or frame is None:
            raise RuntimeError("camera read failed")
        return t, frame

    def close(self):
        self._cap.release()
