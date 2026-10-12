"""
Error-state EKF: IMU propagation with bias estimation, corrected by pose and velocity measurements.

State (nominal):  p (3), v (3), q (w,x,y,z, world-from-body), ba (3), bg (3)
Error state (15): dp, dv, dtheta, dba, dbg  (right perturbation: q_true = q_nom * exp(dtheta))

Frame convention: world is z-up; gravity g = [0, 0, -9.80665] m/s^2 in world frame.
Accelerometer returns specific force in the body frame; gyro returns body angular rate.

Noise inputs (continuous densities, as produced by calib.allan):
  accel white noise density    [m/s^2/sqrt(Hz)]
  gyro  white noise density    [rad/s/sqrt(Hz)]
  accel bias random walk       [m/s^3/sqrt(Hz)]  (process noise, chosen not measured)
  gyro  bias random walk       [rad/s^2/sqrt(Hz)]
"""
import numpy as np

from nanosplat.math3d import quat_mul, quat_conj, quat_normalize, quat_to_R, exp_map, log_map, skew

G_W = np.array([0.0, 0.0, -9.80665])
I3 = np.eye(3)


class ImuEkf(object):
    def __init__(self, p, v, q, sig_a, sig_g, sig_ba_rw=1e-3, sig_bg_rw=1e-5,
                 ba=None, bg=None, P0=None):
        self.p = np.asarray(p, np.float64).copy()
        self.v = np.asarray(v, np.float64).copy()
        self.q = quat_normalize(q)
        self.ba = np.zeros(3) if ba is None else np.asarray(ba, np.float64).copy()
        self.bg = np.zeros(3) if bg is None else np.asarray(bg, np.float64).copy()
        self.sig_a, self.sig_g = float(sig_a), float(sig_g)
        self.sig_ba_rw, self.sig_bg_rw = float(sig_ba_rw), float(sig_bg_rw)
        if P0 is None:
            P0 = np.diag([0.01] * 3 + [0.01] * 3 + [1e-3] * 3 + [1e-4] * 3 + [1e-5] * 3)
        self.P = np.asarray(P0, np.float64).copy()

    # ── prediction ──────────────────────────────────────────────────────────
    def propagate(self, acc_m, gyro_m, dt):
        if dt <= 0:
            return
        acc_m = np.asarray(acc_m, np.float64)
        gyro_m = np.asarray(gyro_m, np.float64)
        R = quat_to_R(self.q)
        a_b = acc_m - self.ba
        w_b = gyro_m - self.bg
        a_w = R.dot(a_b) + G_W

        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.q = quat_normalize(quat_mul(self.q, exp_map(w_b * dt)))

        F = np.eye(15)
        F[0:3, 3:6] = I3 * dt
        F[0:3, 6:9] = -0.5 * R.dot(skew(a_b)) * dt * dt
        F[0:3, 9:12] = -0.5 * R * dt * dt
        F[3:6, 6:9] = -R.dot(skew(a_b)) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = I3 - skew(w_b) * dt
        F[6:9, 12:15] = -I3 * dt

        # Noise: white noise enters velocity (accel) and attitude (gyro); biases random-walk.
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -R
        G[6:9, 3:6] = -I3
        G[9:12, 6:9] = I3
        G[12:15, 9:12] = I3
        var_a = self.sig_a ** 2 / (2.0 * dt)       # discrete variance of white noise at rate 1/dt
        var_g = self.sig_g ** 2 / (2.0 * dt)
        Qc = np.diag([var_a] * 3 + [var_g] * 3 + [self.sig_ba_rw ** 2 * dt] * 3 + [self.sig_bg_rw ** 2 * dt] * 3)
        self.P = F.dot(self.P).dot(F.T) + G.dot(Qc).dot(G.T)
        self.P = 0.5 * (self.P + self.P.T)

    # ── corrections ─────────────────────────────────────────────────────────
    def _apply(self, H, r, R_meas):
        S = H.dot(self.P).dot(H.T) + R_meas
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        dx = K.dot(r)
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_normalize(quat_mul(self.q, exp_map(dx[6:9])))
        self.ba = self.ba + dx[9:12]
        self.bg = self.bg + dx[12:15]
        IKH = np.eye(15) - K.dot(H)
        self.P = IKH.dot(self.P).dot(IKH.T) + K.dot(R_meas).dot(K.T)   # Joseph form
        self.P = 0.5 * (self.P + self.P.T)

    def update_pose(self, p_meas, q_meas, sig_p, sig_att):
        r_p = np.asarray(p_meas, np.float64) - self.p
        dq = quat_mul(quat_conj(self.q), quat_normalize(q_meas))
        r_a = log_map(dq)
        r = np.concatenate([r_p, r_a])
        H = np.zeros((6, 15))
        H[0:3, 0:3] = I3
        H[3:6, 6:9] = I3
        R_meas = np.diag([sig_p ** 2] * 3 + [sig_att ** 2] * 3)
        self._apply(H, r, R_meas)

    def update_velocity(self, v_meas, sig_v):
        r = np.asarray(v_meas, np.float64) - self.v
        H = np.zeros((3, 15))
        H[0:3, 3:6] = I3
        self._apply(H, r, np.eye(3) * sig_v ** 2)

    def position_std(self):
        return float(np.sqrt(np.trace(self.P[0:3, 0:3])))
