# Pose frame protocol (version 1)

A fixed 74-byte little-endian packet. It is the contract between the estimator and any consumer:
flight controller bridge, ROS 2 adapter, logger, or a test client.

| Offset | Size | Type | Field | Notes |
|---|---|---|---|---|
| 0 | 2 | bytes | magic | `NS` |
| 2 | 1 | u8 | version | 1 |
| 3 | 1 | u8 | flags | bit 0 = valid |
| 4 | 4 | u32 | seq | frame counter, wraps at 2^32 |
| 8 | 8 | f64 | t | seconds, sender's monotonic clock |
| 16 | 24 | 3×f64 | position | x, y, z in metres, world frame |
| 40 | 16 | 4×f32 | quat | w, x, y, z, unit, world-from-body |
| 56 | 12 | 3×f32 | velocity | m/s, world frame |
| 68 | 4 | f32 | quality | 0 to 1, estimator's confidence |
| 72 | 2 | u16 | crc16 | CRC-16/CCITT-FALSE over bytes 0 to 71 |

Rules for consumers:
- Reject any frame whose magic, version, length or CRC is wrong. Do not resynchronise on partial data.
- Use `valid` (flag bit 0). When it is clear, treat the pose as unavailable regardless of `quality`.
- A jump in `seq` means frames were lost. A decrease means the sender restarted.
- Timestamps are the sender's monotonic clock. Consumers must not compare them with wall-clock time.

Transport is not defined here. Send over UDP (one frame per datagram), serial (with a length-prefixed
framing layer added by the adapter), or a log file.
