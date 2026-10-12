#!/usr/bin/env python3
"""
Print pose frames received over UDP. Use it to check the output of run_nav.py.

    PYTHONPATH=src python3 scripts/listen_pose.py --port 14550
Stop with Ctrl-C. Frames that fail the CRC or magic check are counted and reported, not printed.
"""
import argparse
import socket
import sys
import time

from nanosplat.output.pose_frame import decode, SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--every", type=int, default=50, help="print every Nth frame")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print("listening on %s:%d (Ctrl-C to stop)" % (args.host, args.port))
    good = bad = 0
    last_seq = None
    gaps = 0
    t_start = time.monotonic()
    try:
        while True:
            buf, _ = sock.recvfrom(256)
            if len(buf) != SIZE:
                bad += 1
                continue
            try:
                e = decode(buf)
            except ValueError:
                bad += 1
                continue
            good += 1
            if last_seq is not None and e.seq != (last_seq + 1) & 0xFFFFFFFF:
                gaps += 1
            last_seq = e.seq
            if good % args.every == 1:
                print("seq %6d  t %.2f  pos (%.3f, %.3f, %.3f) m  vel (%.2f, %.2f, %.2f) m/s  valid %d  quality %.2f" %
                      (e.seq, e.t, e.position[0], e.position[1], e.position[2],
                       e.velocity[0], e.velocity[1], e.velocity[2], int(e.valid), e.quality))
    except KeyboardInterrupt:
        pass
    elapsed = time.monotonic() - t_start
    print("\nframes ok: %d   rejected: %d   sequence gaps: %d   rate: %.1f Hz" %
          (good, bad, gaps, good / elapsed if elapsed > 0 else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
