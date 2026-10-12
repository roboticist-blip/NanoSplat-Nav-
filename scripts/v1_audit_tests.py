#!/usr/bin/env python3
"""
NanoSplat v1 audit tests.

Run from the repo's src/ directory:   python3 audit_tests.py
Each check prints PASS if the behaviour is correct, FAIL if the v1 defect is present.
Requires numpy and opencv-python (or opencv-contrib-python).
"""
import sys
import re
import numpy as np
import cv2

sys.path.insert(0, ".")

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ── 2.1 Temporal stabiliser should change the mask over time ──────────────────
from nanosplat.tracker.extractor import TemporalStabilizer

rng = np.random.default_rng(0)
gray = rng.integers(0, 255, (60, 80), dtype=np.uint8)
st = TemporalStabilizer(alpha=0.6)
st.update(gray, np.zeros((60, 80), np.uint8))
diff = 0
for t in range(20):
    raw = np.zeros((60, 80), np.uint8)
    cv2.circle(raw, (30 + t, 30), 15, 255, -1)
    out = st.update(np.roll(gray, t, axis=1), raw)
    diff += int(np.count_nonzero(out != raw))
# A correct temporal filter must reject some single-frame flips, so output != raw somewhere.
check("2.1 stabiliser is not identity", diff > 0, f"(output differed from raw in {diff} px)")

# ── 2.2 Boundary-removal expression should not be a no-op ─────────────────────
A = (rng.integers(0, 2, (50, 50)) * 255).astype(np.uint8)
E = (rng.integers(0, 2, (50, 50)) * 255).astype(np.uint8)
noop = np.array_equal(cv2.bitwise_and(A, cv2.bitwise_not(E)) | A, A)
check("2.2 edge subtraction is not a no-op", not noop,
      "(v1 expression is literally `(A & ~E) | A`)")

# ── 2.3 / 2.4 Pose chain on a known orbit ─────────────────────────────────────
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
X = rng.uniform(-0.15, 0.15, (300, 3))


def make_pose(theta, radius=1.0):
    c = np.array([radius * np.sin(theta), 0.0, -radius * np.cos(theta)])
    z = -c / np.linalg.norm(c)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1), c


def project(R_wc, c, P):
    Xc = (R_wc.T @ (P - c).T).T
    p = (K @ Xc.T).T
    return (p[:, :2] / p[:, 2:3]).astype(np.float32)


def rot_err_deg(Ra, Rb):
    return np.rad2deg(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


thetas = np.linspace(0, np.deg2rad(40), 6)
gt = [make_pose(t) for t in thetas]

R_v1 = np.eye(3)
t_v1 = np.zeros((3, 1))         # pose.py accumulation, copied verbatim
R_ok = np.eye(3)
c_ok = np.zeros(3)              # inverse chain (the v2 fix)
for i in range(1, len(thetas)):
    p0 = project(*gt[i - 1], X)
    p1 = project(*gt[i], X)
    E, _ = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, p0, p1, K)
    R_v1 = R_v1 @ R
    t_v1 = t_v1 + R_v1 @ t
    c_ok = c_ok + R_ok @ (-(R.T @ t)).ravel()
    R_ok = R_ok @ R.T

err_v1 = rot_err_deg(R_v1, gt[-1][0])
err_ok = rot_err_deg(R_ok, gt[-1][0])
check("2.3 pose rotation chain is correct", err_v1 < 1.0,
      f"(v1 error {err_v1:.1f} deg, correct chain {err_ok:.2f} deg)")

gt_step = np.linalg.norm(gt[1][1] - gt[0][1])
rec_t = float(np.linalg.norm(t))
check("2.4 translation scale is recovered", abs(rec_t - gt_step) / gt_step < 0.05,
      f"(recoverPose step {rec_t:.3f} vs true {gt_step:.3f} m)")

# ── 2.5 Fusion must merge a re-observed surface, not duplicate it ─────────────
from nanosplat.gaussian import MicroGaussianCloud

cloud = MicroGaussianCloud(max_gaussians=5000)
pts = rng.uniform(-0.1, 0.1, (1000, 3)).astype(np.float32)
cols = rng.uniform(0, 1, (1000, 3)).astype(np.float32)
scl = np.full((1000, 3), 0.01, np.float32)
cloud._insert_batch(pts, cols, scl)
cloud._insert_batch(pts + 0.0005, cols, scl)
check("2.5 re-observed surface is fused, not duplicated", cloud.count <= 1100,
      f"(active={cloud.count}, expected about 1000)")

# ── 2.6 Depth normalisation must not depend on other pixels ───────────────────
def normalise(d):
    return (d - d.min()) / (d.max() - d.min())


rel = rng.uniform(0.2, 0.8, (96, 128))
rel2 = rel.copy()
rel2[0, 0], rel2[1, 1] = 0.0, 1.0
a, b = normalise(rel)[50, 60], normalise(rel2)[50, 60]
# This test documents the v1 per-frame min-max design. A fixed affine alignment passes.
check("2.6 depth scale is frame-independent (v1 fails by design)", abs(a - b) < 1e-6,
      f"(same pixel: {a:.3f} vs {b:.3f})")

# ── 3.1 Orchestrator start() import wiring ───────────────────────────────────
from nanosplat.core.hardware import HardwareProfile, Backend
from nanosplat.reconstruction.orchestrator import ReconstructionOrchestrator

orch = ReconstructionOrchestrator(HardwareProfile(backend=Backend.NUMPY, device_name="t"),
                                  output_dir="/tmp/nsaudit")
try:
    orch.start(640, 480)
    ok_start = True
    msg = ""
except Exception as e:  # noqa: BLE001
    ok_start = False
    msg = f"({type(e).__name__}: {e})"
check("3.1 orchestrator.start() imports resolve", ok_start, msg)

# ── 3.2 Viewer parser strings (regression guard; v1 is correct) ──────────────
import importlib.util
spec = importlib.util.spec_from_file_location("nsmain", "main.py")
nsmain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nsmain)
js = nsmain.VIEWER_HTML
regex_src = re.search(r"element vertex \(([^)]*)\)", js).group(1)
ply_header = "ply\nformat binary_little_endian 1.0\nelement vertex 92000\nend_header\n"
check("3.2 viewer vertex-count regex matches a PLY header",
      re.compile(regex_src).search(ply_header) is not None,
      f"(served regex: /{regex_src}/)")
check("3.2 viewer header search string matches a PLY header",
      "end_header\n" in ply_header and "indexOf('end_header\\n')" in js)

# ── Summary ───────────────────────────────────────────────────────────────────
n_fail = sum(1 for _, ok in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} checks pass. "
      f"{n_fail} v1 defect(s) reproduced.")
sys.exit(0)
