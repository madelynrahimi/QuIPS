"""
Reproduces Fig. 4 of the report: effective temperature T_eff vs.
dimensionless feedback gain g_fb, when feedback acts on the Kalman-filtered
state estimate instead of the raw measurement signal, for both actuation
schemes (direct momentum update and beam-center modulation).

Expected result: T_eff decreases monotonically and saturates near
T_ground for both schemes, with no reheating at large gain (contrast
with fig3_temperature_raw_signal.py).
"""
import numpy as np
import matplotlib.pyplot as plt

from feedback_cooling.constants import (
    T_GROUND, ETA_DEFAULT, KAPPA_DEFAULT, DT_DEFAULT, GAMMAS_DEFAULT,
)
from feedback_cooling.simulate import simulate_direct_kalman, simulate_beam_mod_kalman
from feedback_cooling.analysis import effective_temperature


def main(n_steps=int(2.5e6), seed=0):
    rng = np.random.default_rng(seed)
    dt = DT_DEFAULT
    dW_all = rng.standard_normal(n_steps) * np.sqrt(dt)

    T_direct, T_beam = [], []
    for g_fb in GAMMAS_DEFAULT:
        X_traj, VX_traj = simulate_direct_kalman(
            g_fb, dW_all, n_steps, dt, ETA_DEFAULT, KAPPA_DEFAULT
        )
        T_direct.append(effective_temperature(X_traj, VX_traj, dt))

        X_traj_b, VX_traj_b, _ = simulate_beam_mod_kalman(
            g_fb, dW_all, n_steps, dt, ETA_DEFAULT, KAPPA_DEFAULT
        )
        T_beam.append(effective_temperature(X_traj_b, VX_traj_b, dt))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(GAMMAS_DEFAULT, T_direct, "o-", label="direct update (Kalman)")
    ax.loglog(GAMMAS_DEFAULT, T_beam, "o-", label="beam modulation (Kalman)")
    ax.axhline(
        T_GROUND * 1e3, color="k", linestyle="--",
        label=r"$T_{ground} = \hbar\omega/2k_B$ = %.2f $\mu$K" % (T_GROUND * 1e6),
    )
    ax.set_xlabel(r"Dimensionless feedback gain $g_{fb}$")
    ax.set_ylabel(r"$T_{eff}$ [mK]")
    ax.set_title("Effective Temperature vs Feedback Gain (Kalman-filtered estimate)")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    return fig, GAMMAS_DEFAULT, T_direct, T_beam


if __name__ == "__main__":
    fig, gammas, T_direct, T_beam = main()
    plt.show()
