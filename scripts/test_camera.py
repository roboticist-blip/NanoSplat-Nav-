#!/usr/bin/env python3
"""
Camera test for the USB (UVC) camera on the Jetson Nano.

    PYTHONPATH=src python3 scripts/test_camera.py --device 0 --seconds 10 --snapshot cam.jpg

Checks, in order:
  1. The camera opens and reports the resolution it actually delivers.
  2. Frames arrive at a steady rate, and the timestamps keep increasing.
  3. The image is not black, and it changes over time (not a frozen feed).
  4. A snapshot is saved so you can look at the image.
Exit code 0 means every check passed.
"""
import argparse
import sys
import time

import numpy as np

from nanosplat.camera.usb_camera import UsbCamera


def frame_stats(frame):
    """Mean brightness (0..255) and the spread of values across the image."""
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    return float(gray.mean()), float(gray.std())


def check(name, ok, detail=""):
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="/dev/videoN index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--snapshot", default="camera_test.jpg")
    args = ap.parse_args()

    try:
        cam = UsbCamera(device=args.device, width=args.width, height=args.height, fps=args.fps)
    except RuntimeError as e:
        print("[FAIL] open camera: %s" % e)
        print("       Check: ls /dev/video*   and   v4l2-ctl --list-devices")
        return 1

    times, means = [], []
    last = None
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            t, frame = cam.read()
            times.append(t)
            means.append(frame_stats(frame)[0])
            last = frame
    except RuntimeError as e:
        print("[FAIL] read: %s" % e)
        cam.close()
        return 1
    cam.close()

    if not times:
        print("[FAIL] no frames read")
        return 1
    if len(times) < 2:
        print("[FAIL] only one frame read; cannot measure rate")
        return 1

    ok = True
    h, w = last.shape[:2]
    ok &= check("resolution", (w, h) == (args.width, args.height),
                "requested %dx%d, got %dx%d" % (args.width, args.height, w, h))

    elapsed = times[-1] - times[0]
    fps = (len(times) - 1) / elapsed if elapsed > 0 else 0.0
    ok &= check("frame rate", fps >= 0.8 * args.fps,
                "%.1f fps measured (requested %d, pass at 80%%)" % (fps, args.fps))

    dts = np.diff(times)
    ok &= check("timestamps increasing", bool(np.all(dts > 0)),
                "interval min %.1f ms, max %.1f ms" % (dts.min() * 1e3, dts.max() * 1e3))

    mean_b = float(np.mean(means))
    ok &= check("image not black", mean_b > 10.0, "mean brightness %.1f / 255" % mean_b)

    variation = float(np.std(means))
    ok &= check("image changes over time", variation > 0.05 or len(means) < 5,
                "brightness std across frames %.2f (0 would mean a frozen feed)" % variation)

    try:
        import cv2
        cv2.imwrite(args.snapshot, last)
        check("snapshot saved", True, "-> %s (copy it to your PC to look at it)" % args.snapshot)
    except Exception as e:  # noqa: BLE001
        ok &= check("snapshot saved", False, str(e))

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
