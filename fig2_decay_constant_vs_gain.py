"""
Reproduces Fig. 2 of the report: 1/e decay constant tau vs. dimensionless
feedback gain g_fb, averaged over N_RUNS independent noise realizations,
for two initial conditions of the beam-center-modulation (raw signal)
scheme.

Expected result: both initial conditions follow the same tau ~ g_fb^-1
scaling and lie on top of each other, confirming the cooling *rate* is
set entirely by the feedback gain, independent of where the particle
starts.

NOTE: in the original notebook, the two initial-condition cases differ in
the initial *mean* position/momentum (X0, P0) -- one drawn from a thermal
distribution ~ N(0, sqrt(2*n_th)), the other fixed at X0=P0=5 -- while the
initial conditional *variance* (VX0, VP0) is held at 0.5 in both cases.
Double-check this against how you want to describe the two curves in the
figure caption before pushing (e.g. "initial position/momentum variance of
5" vs. "fixed initial mean of 5").
"""
import numpy as np
import matplotlib.pyplot as plt

from feedback_cooling.constants import (
    ETA_DEFAULT, KAPPA_DEFAULT, DT_DEFAULT, WINDOW_DEFAULT, thermal_occupation,
)
from feedback_cooling.simulate import simulate_beam_mod_raw
from feedback_cooling.analysis import occupation_number_vs_time, decay_constant

GAMMAS = np.logspace(-3, 0, 12)
N_RUNS = 5
BLOCK_SIZE = 6000


def _tau_for_ic(g_fb, n_steps, dt, block_size, X0_fn, rng):
    """Average the decay constant over N_RUNS independent noise draws for
    a given initial-condition generator X0_fn() -> (X0, P0)."""
    n_blocks = n_steps // block_size
    t_blocks = np.arange(n_blocks) * block_size * dt
    n_vs_t_runs = np.zeros((N_RUNS, n_blocks))

    for run in range(N_RUNS):
        dW = rng.standard_normal(n_steps) * np.sqrt(dt)
        X0, P0 = X0_fn()
        X_traj, VX_traj, _ = simulate_beam_mod_raw(
            g_fb, dW, n_steps, dt, ETA_DEFAULT, KAPPA_DEFAULT, WINDOW_DEFAULT,
            X0=X0, P0=P0, VX0=0.5, VP0=0.5,
        )
        n_vs_t_runs[run] = occupation_number_vs_time(X_traj, VX_traj, block_size)

    n_vs_t = n_vs_t_runs.mean(axis=0)
    return decay_constant(n_vs_t, t_blocks)


def main(n_steps=int(2.5e6), seed=0):
    rng = np.random.default_rng(seed)
    dt = DT_DEFAULT
    n_th = thermal_occupation(300.0)

    def thermal_ic():
        return (rng.standard_normal() * np.sqrt(2 * n_th),
                rng.standard_normal() * np.sqrt(2 * n_th))

    def fixed_ic():
        return 5.0, 5.0

    tau_thermal = [_tau_for_ic(g, n_steps, dt, BLOCK_SIZE, thermal_ic, rng) for g in GAMMAS]
    tau_fixed = [_tau_for_ic(g, n_steps, dt, BLOCK_SIZE, fixed_ic, rng) for g in GAMMAS]

    tau_thermal = np.array(tau_thermal)
    tau_fixed = np.array(tau_fixed)
    valid_th = ~np.isnan(tau_thermal)
    valid_fix = ~np.isnan(tau_fixed)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(GAMMAS[valid_th], tau_thermal[valid_th], "o-", color="steelblue",
              label=r"$X_0=P_0=\mathcal{N}(0,\sqrt{2n_{th}})$")
    ax.loglog(GAMMAS[valid_fix], tau_fixed[valid_fix], "s-", color="tomato",
              label=r"$X_0=P_0=5$")
    ax.set_xlabel(r"Dimensionless feedback gain $g_{fb}$", fontsize=13)
    ax.set_ylabel(r"Decay constant $\tau$ (dimensionless units)", fontsize=13)
    ax.set_title(f"Feedback Gain vs Decay Constant (1/e threshold, averaged over {N_RUNS} runs)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, GAMMAS, tau_thermal, tau_fixed


if __name__ == "__main__":
    fig, gammas, tau_thermal, tau_fixed = main()
    plt.show()
