"""
Camera geometry for the v2 pose path.

Conventions
-----------
R_wc, c_w : world-from-camera rotation and camera centre in world coordinates.
recoverPose(E, p_prev, p_curr, K) returns (R_rel, t_rel) with
    X_curr = R_rel @ X_prev + t_rel
v1 used the opposite direction together with the already-updated rotation,
which gave an 80 deg error on a 40 deg orbit (audit 2.3).
"""
import cv2
import numpy as np


def chain_relative(R_wc: np.ndarray, c_w: np.ndarray,
                   R_rel: np.ndarray, t_rel: np.ndarray):
    """World-from-camera of the current frame, given the previous one and recoverPose output."""
    t = np.asarray(t_rel, dtype=np.float64).reshape(3)
    R_new = R_wc @ R_rel.T
    c_new = np.asarray(c_w, dtype=np.float64).reshape(3) + R_wc @ (-(R_rel.T @ t))
    return R_new, c_new


def projection_matrix(K: np.ndarray, R_wc: np.ndarray, c_w: np.ndarray) -> np.ndarray:
    R_cw = R_wc.T
    t_cw = -R_cw @ np.asarray(c_w, dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R_cw, t_cw])


def triangulate(K, R1, c1, R2, c2, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Triangulate matched pixels (N,2) from two posed views. Returns (N,3) world points."""
    P1 = projection_matrix(K, R1, c1)
    P2 = projection_matrix(K, R2, c2)
    Xh = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    return (Xh[:3] / Xh[3]).T


def pose_from_landmarks(K, X_world: np.ndarray, p_img: np.ndarray,
                        reproj_thresh: float = 2.0, iters: int = 200):
    """
    Localise a camera against known 3D landmarks with PnP-RANSAC, then refine.
    Returns (R_wc, c_w, inlier_mask) or None if it fails.
    """
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X_world.astype(np.float64), p_img.astype(np.float64), K, None,
        reprojectionError=reproj_thresh, iterationsCount=iters,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None
    idx = inliers.ravel()
    # Iterative refinement on the inliers, using solvePnP with an initial guess, which is
    # available in every OpenCV 3 and 4 build.
    _, rvec, tvec = cv2.solvePnP(
        X_world[idx].astype(np.float64), p_img[idx].astype(np.float64), K, None,
        rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    c_w = (-R_cw.T @ tvec).ravel()
    mask = np.zeros(len(X_world), bool)
    mask[idx] = True
    return R_wc, c_w, mask
