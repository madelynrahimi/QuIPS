"""
Reproduces Fig. 3 of the report: effective temperature T_eff vs.
dimensionless feedback gain g_fb, for direct momentum feedback driven by
the *raw* (shot-noise-limited) measurement signal alpha instead of its
Kalman-filtered estimate.

Expected result: T_eff decreases with g_fb until an optimal gain, then
increases again ("reheating") because the finite-difference velocity
estimate amplifies and reinjects measurement shot noise as force noise.
The minimum sits strictly above T_ground.
"""
import numpy as np
import matplotlib.pyplot as plt

from feedback_cooling.constants import (
    T_GROUND, ETA_DEFAULT, KAPPA_DEFAULT, DT_DEFAULT, WINDOW_DEFAULT,
    GAMMAS_DEFAULT,
)
from feedback_cooling.simulate import simulate_direct_raw
from feedback_cooling.analysis import effective_temperature


def main(n_steps=int(2.5e6), seed=0):
    rng = np.random.default_rng(seed)
    dt = DT_DEFAULT
    dW_all = rng.standard_normal(n_steps) * np.sqrt(dt)

    T_list = []
    for g_fb in GAMMAS_DEFAULT:
        X_traj, VX_traj = simulate_direct_raw(
            g_fb, dW_all, n_steps, dt, ETA_DEFAULT, KAPPA_DEFAULT, WINDOW_DEFAULT
        )
        T_list.append(effective_temperature(X_traj, VX_traj, dt))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(GAMMAS_DEFAULT, T_list, "o-")
    ax.axhline(
        T_GROUND * 1e3, color="k", linestyle="--",
        label=r"$T_{ground} = \hbar\omega/2k_B$ = %.2f $\mu$K" % (T_GROUND * 1e6),
    )
    ax.set_xlabel(r"Dimensionless feedback gain $g_{fb}$")
    ax.set_ylabel(r"$T_{eff}$ [mK]")
    ax.set_title("Effective Temperature vs Feedback Gain (raw measurement signal)")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    return fig, GAMMAS_DEFAULT, T_list


if __name__ == "__main__":
    fig, gammas, T_list = main()
    plt.show()
