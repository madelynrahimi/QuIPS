"""
Core stochastic simulation engine for feedback cooling of a levitated
nanosphere, in dimensionless natural units (hbar = m = omega0 = 1).

All trajectories are generated from the conditional (Kalman-filtered)
equations of motion for a continuously-monitored harmonic oscillator
(Eqs. 4-5, 27-31 of the report). Four feedback laws are implemented,
corresponding to the two physical actuation methods (direct momentum
kick vs. beam-center modulation) crossed with the two possible feedback
signals (the raw, shot-noise-limited measurement record alpha vs. the
Kalman-filtered state estimate):

    simulate_direct_raw      -- direct momentum feedback, raw signal   (Sec 3.2.1, Fig 3)
    simulate_beam_mod_raw    -- beam-center feedback,    raw signal    (Sec 3.3.2)
    simulate_direct_kalman   -- direct momentum feedback, LQR/Kalman   (Sec 3.1, Eq 15-16)
    simulate_beam_mod_kalman -- beam-center feedback,    LQR/Kalman

The "raw" variants estimate velocity by finite-differencing the noisy
measurement record over a sliding window; because that record contains
shot noise, the resulting velocity estimate is itself noisy, and at large
feedback gain this reinjects noise into the system as force noise
(the reheating seen in Fig. 3). The "kalman" variants instead feed back the
already-optimally-filtered momentum estimate, which does not carry this
extra noise (Fig. 4).
"""
import numpy as np
from numba import njit


@njit
def _kalman_step(X, P, VX, VP, CXP, dW, dt, eta, kappa):
    """One Euler-Maruyama step of the conditional mean/covariance equations
    of motion (Eqs. 4-5, 29-31), with no feedback force applied."""
    root_term = np.sqrt(8.0 * eta * kappa)
    X += P * dt + root_term * VX * dW
    P += -X * dt + root_term * CXP * dW
    VX += (2.0 * CXP - 8.0 * eta * kappa * VX**2) * dt
    VP += (-2.0 * CXP + 2.0 * kappa - 8.0 * eta * kappa * CXP**2) * dt
    CXP += (VP - VX - 8.0 * eta * kappa * VX * CXP) * dt
    return X, P, VX, VP, CXP


@njit
def lqr_gains(g_fb):
    """LQR-optimal gain matrix elements K21, K22 evaluated at dimensionless
    feedback gain g_fb (Eq. 13, with omega0 = 1). K11 = 0 because the
    control force acts only on momentum, not position directly."""
    K21 = np.sqrt(1.0 + g_fb**2) - 1.0
    K22 = np.sqrt(g_fb**2 - 2.0 + 2.0 * np.sqrt(1.0 + g_fb**2))
    return K21, K22


@njit
def simulate_direct_raw(g_fb, dW_all, n_steps, dt, eta, kappa, window,
                         X0=20.0, P0=20.0, VX0=0.5, VP0=0.5, CXP0=0.0):
    """Direct momentum feedback using the *raw* measurement signal alpha
    (Eq. 6, Sec. 3.2.1). Velocity is estimated by finite-differencing the
    noisy measurement record over `window` steps, then fed back directly
    as a momentum kick. Reproduces Fig. 3: cooling at low/moderate gain,
    reheating at large gain as shot noise is amplified by the derivative
    and reinjected as force noise.
    """
    X, P, VX, VP, CXP = X0, P0, VX0, VP0, CXP0
    root_term = np.sqrt(8.0 * eta * kappa)

    X_traj = np.zeros(n_steps)
    VX_traj = np.zeros(n_steps)
    alpha_hist = np.zeros(n_steps + 1)

    for i in range(n_steps):
        dW = dW_all[i]
        X, P, VX, VP, CXP = _kalman_step(X, P, VX, VP, CXP, dW, dt, eta, kappa)

        X_traj[i] = X
        VX_traj[i] = VX

        alpha = X + dW / (root_term * dt)
        alpha_hist[i + 1] = alpha
        if i > window:
            adot = (alpha_hist[i] - alpha_hist[i - window]) / (window * dt)
            P -= g_fb * adot * dt

    return X_traj, VX_traj


@njit
def simulate_beam_mod_raw(g_fb, dW_all, n_steps, dt, eta, kappa, window,
                           X0=20.0, P0=20.0, VX0=0.5, VP0=0.5, CXP0=0.0):
    """Beam-center modulation feedback using the *raw* measurement signal
    (Sec. 3.3.2, Eq. 21-22). The trap center x_c is shifted in proportion
    to a finite-difference velocity estimate from the noisy measurement
    record, and the total force -(X - x_c) is applied. Physically
    equivalent to `simulate_direct_raw`; the trap-center formulation is
    what is actually implemented on the AOD in the experiment.
    """
    X, P, VX, VP, CXP = X0, P0, VX0, VP0, CXP0
    root_term = np.sqrt(8.0 * eta * kappa)

    X_traj = np.zeros(n_steps)
    VX_traj = np.zeros(n_steps)
    beamcenter = np.zeros(n_steps)
    X_est = np.zeros(n_steps + 1)
    x_c = 0.0

    for i in range(n_steps):
        dW = dW_all[i]

        X += P * dt + root_term * VX * dW
        X_est[i + 1] = X

        if i > window:
            adot = (X_est[i] - X_est[i - window]) / (window * dt)
            x_c = -g_fb * adot

        P += -(X - x_c) * dt + root_term * CXP * dW
        VX += (2.0 * CXP - 8.0 * eta * kappa * VX**2) * dt
        VP += (-2.0 * CXP + 2.0 * kappa - 8.0 * eta * kappa * CXP**2) * dt
        CXP += (VP - VX - 8.0 * eta * kappa * VX * CXP) * dt

        X_traj[i] = X
        VX_traj[i] = VX
        beamcenter[i] = x_c

    return X_traj, VX_traj, beamcenter


@njit
def simulate_direct_kalman(g_fb, dW_all, n_steps, dt, eta, kappa,
                            X0=20.0, P0=20.0, VX0=0.5, VP0=0.5, CXP0=0.0):
    """Direct momentum feedback using the LQR-optimal gain applied to the
    Kalman-filtered state estimate (Eq. 15-16, Sec. 3.1). Because the
    feedback acts on the filtered estimate rather than the raw record,
    measurement shot noise is not reinjected into the system and the
    temperature saturates at the ground state at large gain (Fig. 4)
    instead of reheating.
    """
    X, P, VX, VP, CXP = X0, P0, VX0, VP0, CXP0
    K21, K22 = lqr_gains(g_fb)

    X_traj = np.zeros(n_steps)
    VX_traj = np.zeros(n_steps)

    for i in range(n_steps):
        dW = dW_all[i]
        X, P, VX, VP, CXP = _kalman_step(X, P, VX, VP, CXP, dW, dt, eta, kappa)

        u = -(K21 * X + K22 * P)
        P += u * dt

        X_traj[i] = X
        VX_traj[i] = VX

    return X_traj, VX_traj


@njit
def simulate_beam_mod_kalman(g_fb, dW_all, n_steps, dt, eta, kappa,
                              X0=20.0, P0=20.0, VX0=0.5, VP0=0.5, CXP0=0.0):
    """Beam-center modulation feedback using the Kalman-filtered momentum
    estimate (rather than a finite-difference derivative of the raw
    signal) to set the trap-center displacement x_c."""
    X, P, VX, VP, CXP = X0, P0, VX0, VP0, CXP0
    root_term = np.sqrt(8.0 * eta * kappa)

    X_traj = np.zeros(n_steps)
    VX_traj = np.zeros(n_steps)
    beamcenter = np.zeros(n_steps)

    for i in range(n_steps):
        dW = dW_all[i]
        X += P * dt + root_term * VX * dW

        x_c = -g_fb * P

        P += -(X - x_c) * dt + root_term * CXP * dW
        VX += (2.0 * CXP - 8.0 * eta * kappa * VX**2) * dt
        VP += (-2.0 * CXP + 2.0 * kappa - 8.0 * eta * kappa * CXP**2) * dt
        CXP += (VP - VX - 8.0 * eta * kappa * VX * CXP) * dt

        X_traj[i] = X
        VX_traj[i] = VX
        beamcenter[i] = x_c

    return X_traj, VX_traj, beamcenter
