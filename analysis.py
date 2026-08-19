"""
Post-processing helpers shared by all figure scripts: extracting an
effective temperature or occupation number from a simulated trajectory,
and characterizing cooling dynamics (decay constant, cooling time).
"""
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

from .constants import T0


def _steady_state(X_traj, VX_traj, burn_in):
    half = int(len(X_traj) * burn_in)
    return X_traj[half:], VX_traj[half:]


def mean_X2(X_traj, VX_traj, dt, burn_in=0.5):
    """Full quantum second moment <X^2> = integral(PSD) + <VX>, computed
    from the steady-state (post burn-in) portion of the trajectory via
    Welch's method (matches the PSD-based estimator used throughout the
    report, e.g. Eq. in Sec. 3.2.2)."""
    X_ss, VX_ss = _steady_state(X_traj, VX_traj, burn_in)
    f, Pxx = welch(X_ss, fs=1.0 / dt, nperseg=len(X_ss) // 4)
    return simpson(Pxx, x=f) + np.mean(VX_ss)


def effective_temperature(X_traj, VX_traj, dt, burn_in=0.5):
    """Effective temperature in mK: T_eff = <X^2> * T0 (equipartition in
    natural units, valid because <X^2> = 0.5 at the ground state)."""
    return mean_X2(X_traj, VX_traj, dt, burn_in) * T0 * 1e3


def occupation_number(X_traj, VX_traj, dt, burn_in=0.5):
    """Steady-state phonon occupation number <n> = <X^2> / 2 (natural
    units, hbar = 1)."""
    return mean_X2(X_traj, VX_traj, dt, burn_in) / 2.0


def occupation_number_vs_time(X_traj, VX_traj, block_size=6000):
    """Time-resolved occupation number, block-averaged directly in the
    time domain (no PSD) so it can be evaluated during the transient
    cooling process, not just in steady state (Sec. 3.1.1, Fig. 2)."""
    n_blocks = len(X_traj) // block_size
    n_vs_t = np.zeros(n_blocks)
    for i in range(n_blocks):
        sl = slice(i * block_size, (i + 1) * block_size)
        n_vs_t[i] = np.mean(X_traj[sl] ** 2) / 2.0 + np.mean(VX_traj[sl]) / 2.0
    return n_vs_t


def decay_constant(n_vs_t, t_blocks):
    """1/e decay time tau: the first time at which <n>(t) has fallen to
    within 1/e of the gap between its initial and steady-state values
    (Sec. 3.1.1). Returns NaN if the trajectory never reaches that point.
    """
    n0, n_ss = n_vs_t[0], n_vs_t[-1]
    target = n_ss + (n0 - n_ss) / np.e
    below = np.nonzero(n_vs_t <= target)[0]
    if len(below) == 0:
        return np.nan
    return t_blocks[below[0]]


def cooling_time_to_target(n_vs_t, t_blocks, n_target=0.5):
    """First time at which <n>(t) reaches (or drops below) n_target,
    e.g. n_target=0.5 for the quantum ground state."""
    below = np.nonzero(n_vs_t <= n_target)[0]
    if len(below) == 0:
        return np.nan
    return t_blocks[below[0]]
