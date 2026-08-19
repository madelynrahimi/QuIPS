"""
Reproduces Fig. 5 of the report: steady-state occupation number <n> vs.
dimensionless feedback gain g_fb, for beam-center modulation (raw signal),
starting from either a cold state (X0 = P0 = 0) or a thermal state
(X0, P0 drawn from a T=300K equilibrium distribution).

Expected result: both initial conditions converge to the same steady-state
<n> at a given g_fb, confirming the steady-state minimum occupation is set
by the feedback gain alone, independent of initial conditions.
"""
import numpy as np
import matplotlib.pyplot as plt

from feedback_cooling.constants import (
    ETA_DEFAULT, KAPPA_DEFAULT, DT_DEFAULT, WINDOW_DEFAULT, thermal_occupation,
)
from feedback_cooling.simulate import simulate_beam_mod_raw
from feedback_cooling.analysis import occupation_number

GAMMAS = np.logspace(-2, 0, 20)


def main(n_steps=int(2.5e6), seed=0):
    rng = np.random.default_rng(seed)
    dt = DT_DEFAULT
    dW_all = rng.standard_normal(n_steps) * np.sqrt(dt)

    n_th = thermal_occupation(300.0)
    X_init = rng.standard_normal() * np.sqrt(2 * n_th)
    P_init = rng.standard_normal() * np.sqrt(2 * n_th)

    n_cold, n_thermal = [], []
    for g_fb in GAMMAS:
        X_traj, VX_traj, _ = simulate_beam_mod_raw(
            g_fb, dW_all, n_steps, dt, ETA_DEFAULT, KAPPA_DEFAULT, WINDOW_DEFAULT,
            X0=0.0, P0=0.0, VX0=0.5, VP0=0.5,
        )
        n_cold.append(occupation_number(X_traj, VX_traj, dt))

        X_traj, VX_traj, _ = simulate_beam_mod_raw(
            g_fb, dW_all, n_steps, dt, ETA_DEFAULT, KAPPA_DEFAULT, WINDOW_DEFAULT,
            X0=X_init, P0=P_init, VX0=0.5, VP0=0.5,
        )
        n_thermal.append(occupation_number(X_traj, VX_traj, dt))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(GAMMAS, n_cold, "o-", label="$X_0=P_0=0$")
    ax.loglog(GAMMAS, n_thermal, "s-", label=r"$X_0,P_0 \sim \mathcal{N}(0,\sqrt{2n_{th}})$")
    ax.set_xlabel(r"Dimensionless feedback gain $g_{fb}$")
    ax.set_ylabel(r"Occupation number $\langle n \rangle$")
    ax.set_title("Occupation Number vs Feedback Gain")
    ax.legend()
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, GAMMAS, n_cold, n_thermal


if __name__ == "__main__":
    fig, gammas, n_cold, n_thermal = main()
    plt.show()
