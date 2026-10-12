"""UDP output: one pose frame per datagram. Controller-agnostic transport."""
import socket

from nanosplat.output.pose_frame import encode


class UdpPoseSink(object):
    def __init__(self, host="127.0.0.1", port=14550):
        self.addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, est):
        self._sock.sendto(encode(est), self.addr)

    def close(self):
        self._sock.close()
