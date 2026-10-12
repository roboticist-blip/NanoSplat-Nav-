# NanoSplat v1 — Code and Paper Audit

Scope: uploaded `NanoSplat-main.zip` (18 files, ~2,000 lines) and `NanoSplat.pdf` (IEEE draft).
Method: read every source file, then executed the core modules with NumPy 2.4 and OpenCV 4.13
on synthetic data. Items marked **[tested]** were reproduced by running code. Items marked
**[read]** follow from the source and were not executed.

Reproduce the tests with `python3 audit_tests.py` from the `src/` directory of the repo.

---

## 1. Summary

The concept in the paper (depth-seeded Gaussians, confidence-weighted fusion, optical-flow mask
stabilisation, ORB pose) is sound and is a good basis for v2. The v1 code, however, does not yet
implement several of the paper's central mechanisms, so the paper's numbers cannot be reproduced
from this code path. The defects below must be fixed before any v2 benchmark is meaningful.

Priority order: correctness first (Section 2), then wiring (3.1), then performance and
licensing (Sections 3–4). The live viewer was checked and is correct; it is not a defect.

---

## 2. Defects that break the method

### 2.1 Temporal mask stabilisation is an identity map  **[tested]**
`tracker/extractor.py` → `TemporalStabilizer.update`

With α = 0.6 and τ = 0.5 on binary masks, the blend `α·raw + (1−α)·warped > τ` reduces to
`output = raw` for every pixel: if raw = 1 the score is ≥ 0.6 > 0.5; if raw = 0 the score is ≤ 0.4 < 0.5.
The warped history never affects the output. Over 20 frames of a moving disc with a shifted
background, output differed from raw in 0 pixels.

This contradicts Eq. (4) and the "TempStab" rows of Tables IV and V (IoU 0.58 → 0.72, CD 8.5 → 6.2 mm).
Those results cannot come from this code.

**Fix (v2):** keep a float belief map `b_t = α·raw_t + (1−α)·warp(b_{t−1})` with α ≈ 0.3, threshold
b_t at 0.5. A one-frame flip then reaches only 0.3 and is rejected; a pixel that persists for two
frames reaches 0.51 and is admitted. This matches the paper's stated intent. Re-measure IoU after the change.

### 2.2 Boundary-removal step is a no-op  **[tested]**
`tracker/extractor.py` → `ObjectSegmenter.segment`

`mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges_d)) | mask` simplifies to `mask`
(Boolean identity `(A & ~E) | A = A`). The Canny edge subtraction described in Section III.B.2 never runs.

**Fix (v2):** either implement the intended removal (`A & ~E_dilated`, then re-fill interior holes)
or drop the step and say so in the paper. Measure the IoU effect of each choice.

### 2.3 Pose chaining doubles the rotation error  **[tested]**
`pose.py` → `LightweightPoseEstimator.process_frame`

`cv2.recoverPose` returns the motion from the previous camera to the current one: `X_curr = R·X_prev + t`.
The code accumulates `R_cur = R_cur @ R_rel` and `t_cur += R_cur @ t_rel`, which uses the wrong
direction and the already-updated rotation. On a synthetic 40° orbit, rotation error after five
steps was **80°** with the code's update and **0°** with the inverse chain
(`R_world←cam ← R_prev · R_relᵀ`, `c ← c + R_prev · (−R_relᵀ · t)`).
Direction of the centre path still looked correct (cosine 0.99) only because the translation scale
was arbitrary; see 2.4.

**Fix (v2):** use the inverse chain above, and add a unit test with a known orbit (`audit_tests.py`).

### 2.4 Translation scale is never recovered  **[tested]**
`recoverPose` returns unit-length translations. Each keyframe step therefore adds length 1, regardless of
the true motion (0.14 m per step in the test). The trajectory length is meaningless, and the
claim in Section III.C that scale is "consistent within a session" does not hold for the pose path.

**Fix (v2):** build a small local landmark map. Triangulate matched ORB points between the last
two keyframes using the depth-aligned scale, then localise each new keyframe with PnP-RANSAC against
those landmarks. This gives a metric-consistent trajectory tied to the depth scale.

### 2.5 Fusion does not associate points spatially  **[tested]**
`gaussian.py` → `_insert_batch`

The "confidence-weighted merge" (Eqs. 10–12) is applied to whatever Gaussian sits in the next ring-buffer
slot, not to the nearest Gaussian in space.
- Observing the same surface in a second keyframe **doubled** the active count (1,000 → 2,000) instead of fusing.
- After the ring wraps, **1,000 of 1,000** slots were averaged with unrelated points several metres away.

This is the core contribution of the paper, so it needs a real implementation.

**Fix (v2):** voxel hash grid (cell ≈ pixel footprint at depth). Look up the cell and its 26
neighbours; fuse only within a radius. Keep the bounded-capacity guarantee by pruning the lowest
confidence voxels. Vectorise the update (no per-point Python loop).

### 2.6 Depth scale changes frame to frame  **[read]**
`depth.py` → `estimate` normalises each frame to [0, 1] by its own min and max. The same pixel
gets a different normalised value when the frame's extremes change (0.481 vs 0.489 in a test with two
extra extreme pixels). `estimate_metric` then computes `object_dist / (rel + ε)`, so the metric scale drifts
between keyframes. Also, MiDaS outputs affine-invariant inverse depth; the `d = dist / rel` form
assumes a particular scale and shift that the network does not provide.

**Fix (v2):** fit a per-frame affine (s, t) that aligns MiDaS output to a reference depth on
the object, anchored at the first keyframe, and reuse that alignment across the session.

---

## 3. Defects in the application code

### 3.1 Import paths are broken  **[tested]**
`reconstruction/orchestrator.py` → `start()` imports `..core.depth`, `..core.pose`, `..core.gaussian`
and `.hardware`. None exist at those paths; the files live at `nanosplat/depth.py`, `nanosplat/pose.py`,
`nanosplat/gaussian.py`, and `nanosplat/core/hardware.py`. `start()` raises `ModuleNotFoundError`.
The paper says runtime bugs were resolved, so this zip is probably a partial reorganisation of the
working code. Please confirm which tree produced the paper's results.

### 3.2 TensorRT path leaks memory and creates resources every frame  **[read]**
`depth.py` → `_infer_tensorrt` creates a new execution context, allocates device memory with
`cuda.mem_alloc`, and creates a new CUDA stream on every call. None of it is freed. It also depends on
`pycuda`, which is missing from `requirements.txt`.
In the paper's measurements, TensorRT was unavailable and MiDaS ran on ONNX CPU (Table II footnote),
so the 100 ms depth figure is the CPU path. Please say this explicitly in the paper.

### 3.3 Detection blocks the capture loop  **[read]**
`main.py` → `run` calls `extractor.process`, which runs YOLO synchronously. Detection runs every
15 frames, and on every frame until a target is found. On a Nano this stalls display and tracking.
Re-detection also re-initialises the tracker and calls `stabilizer.reset()`, which discards the
temporal history every 15 frames.

**Fix (v2):** run detection on a separate thread, and re-detect only when tracker confidence drops.

### 3.4 Advertised acceleration is not used  **[read]**
`gaussian.py` and the orchestrator use NumPy directly. `XP` (the CuPy/NumPy abstraction) and
Numba's `njit` are imported or defined but never applied to the cloud operations. Section III.G
describes CuPy keeping tensors on the GPU, which the code does not do.

### 3.5 Per-frame depth is computed for the full frame  **[read]**
`_process_keyframe` runs MiDaS on the full 640×480 frame, then uses only the object ROI. Running it
on an ROI with margin would cut the cost substantially.

---

## 4. Licensing (blocks commercial use)

| Component | License | Status |
|---|---|---|
| Ultralytics YOLOv8 (code and weights) | **AGPL-3.0** | **Not permissive.** Blocks closed commercial use unless an enterprise licence is bought. The code also downloads `yolov8n.pt` at runtime, which breaks the no-external-dependency goal. |
| MiDaS-Small (isl-org) | MIT | OK |
| OpenCV / opencv-contrib | Apache-2.0 | OK |
| NumPy, Numba, ONNX Runtime | BSD-3 / BSD-2 / MIT | OK |
| CUDA, cuDNN, TensorRT | NVIDIA proprietary | Permitted on Jetson; check terms before shipping |
| Original 3DGS reference code (INRIA) | Non-commercial research | Do not reuse. v1 does not use it. |

**Fix (v2):** replace YOLOv8n with a permissively licensed detector, for example NanoDet-Plus or
YOLOX-Nano (both Apache-2.0). Convert it to ONNX and then to a local TensorRT engine. Verify the licence
of each weight file at the source before shipping. Update the paper's Section III.B.1 and Table V if
the detector changes.

---

## 5. Paper and code inconsistencies (fix before submission)

- Jetson memory: the Introduction says 2 GB; Fig. 2 and Section III.H say 4 GB.
- Gaussian capacity: code default 20,000; Jetson path 40,000 (`hardware.py`, `max_gaussians`);
  README says 80,000 on Jetson.
- Depth input: paper and code use 256×192 for the Nano; `hardware.py` uses 384×288 at ≥3.5 GB VRAM.
- TensorRT: Section III.G and Table II describe it, but Table II says it was unavailable during measurement.
- "~1–5 cm accuracy" in the README has no supporting measurement.
- Table III FPS (5–7) is for the extraction thread only. The keyframe throughput is 0.71 KF/s.
- Eq. (4) claims temporal persistence that the code does not provide (see 2.1).
- The paper's ablation numbers should be regenerated from fixed code.

---

## 6. Proposed v2 architecture

Target: real-time on Jetson Nano 4 GB (JetPack 4.6.x, Ubuntu 18.04, CUDA 10.2, TensorRT 8.2),
no network calls at runtime, permissively licensed dependencies only.

Threads and stages:
1. **Capture** (V4L2 / GStreamer, 640×480, bounded queue of 2 frames).
2. **Track** (CSRT or KCF per frame, roughly 1 ms). Re-detect asynchronously when confidence drops.
3. **Segment** (HS K-means on the ROI, belief-map stabilisation per 2.1, boundary cleanup per 2.2).
4. **Keyframe** (every k frames or on sufficient baseline / parallax).
5. **Depth** (MiDaS-Small FP16 TensorRT on a margin-padded ROI, per-session affine alignment per 2.6).
6. **Pose** (ORB, local landmark map, PnP-RANSAC, scale from aligned depth per 2.4).
7. **Fuse** (voxel hash, confidence-weighted update per Eqs. 10–12, bounded capacity).
8. **Export** (PLY, `.splat`, and the existing live viewer, which was verified correct).

Each stage gets a latency budget and a benchmark in `bench/`. Targets are to be measured, not assumed.

Repository layout:

```
nanosplat/
├── README.md                  quick start, hardware matrix (measured numbers only)
├── LICENSE                    project licence (choose before publishing)
├── THIRD_PARTY_LICENSES.md    every dependency and model, with its licence
├── pyproject.toml
├── configs/                   default.yaml, jetson_nano_4gb.yaml, rpi4.yaml
├── scripts/                   fetch_models.py (checksummed), build_engine.py
├── src/nanosplat/
│   ├── core/                  config, types, hardware profile, timing
│   ├── capture/               camera sources
│   ├── track/                 tracker, detector (async worker)
│   ├── segment/               kmeans.py, belief_mask.py
│   ├── depth/                 midas.py, alignment.py
│   ├── pose/                  features.py, landmarks.py, pose.py
│   ├── map/                   voxel_hash.py, fusion.py, cloud.py
│   ├── export/                ply.py, splat.py, viewer/
│   └── app/                   pipeline.py, cli.py
├── tests/                     unit tests on synthetic data (orbit pose, fusion, belief mask)
├── bench/                     latency, memory, and accuracy harnesses
└── docs/                      architecture notes, paper source
```

---

## 7. Decisions needed before v2 coding starts

1. **JetPack version.** The Nano stops at JetPack 4.6.x (Python 3.6 by default, TensorRT 8.2). Confirm
   that this is the target, or whether a Nano 2 GB or other board is in scope.
2. **Detector replacement.** Approve NanoDet-Plus or YOLOX-Nano, or name another permissive detector.
3. **Scale source.** Keep the `object_dist` prior with session-consistent alignment, or add a
   stereo or ToF input for metric output.
4. **Primary metric.** Pick one target to optimise first: frame rate, peak memory, or Chamfer distance / F-score.
5. **Reference tree.** Confirm which code produced the paper's numbers, so we can reconcile the differences in Section 3.1.
