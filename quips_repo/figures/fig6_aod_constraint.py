"""
Reproduces Fig. 6 of the report: peak tweezer displacement |dy| required
by the feedback signal (beam-center modulation, raw signal, cold initial
conditions) as a function of dimensionless feedback gain g_fb, compared
to the AOD bandwidth limit (Sec. 3.3.3).

Expected result: the required displacement stays several orders of
magnitude below the AOD's physical displacement ceiling across the full
range of feedback gains studied, i.e. beam-center modulation can reach
the ground state without ever exceeding the actuator's range.
"""
import numpy as np
import matplotlib.pyplot as plt

from feedback_cooling.constants import (
    ETA_DEFAULT, KAPPA_DEFAULT, DT_DEFAULT, WINDOW_DEFAULT, X_ZPF,
    GAMMAS_DEFAULT, aod_max_displacement_pm,
)
from feedback_cooling.simulate import simulate_beam_mod_raw
from feedback_cooling.analysis import occupation_number


def main(n_steps=int(2.5e6), seed=0):
    rng = np.random.default_rng(seed)
    dt = DT_DEFAULT
    dW_all = rng.standard_normal(n_steps) * np.sqrt(dt)
    half = n_steps // 2

    delta_y_max_pm = aod_max_displacement_pm()
    print(f"AOD max displacement: {delta_y_max_pm:.1f} pm")

    n_list, xc_peak_list, xc_rms_list = [], [], []
    for g_fb in GAMMAS_DEFAULT:
        X_traj, VX_traj, beamcenter = simulate_beam_mod_raw(
            g_fb, dW_all, n_steps, dt, ETA_DEFAULT, KAPPA_DEFAULT, WINDOW_DEFAULT,
            X0=0.0, P0=0.0, VX0=0.5, VP0=0.5,
        )
        n_list.append(occupation_number(X_traj, VX_traj, dt))

        xc_ss = beamcenter[half:]
        xc_peak_list.append(np.max(np.abs(xc_ss)) * X_ZPF * 1e12)  # pm
        xc_rms_list.append(np.sqrt(np.mean(xc_ss**2)) * X_ZPF * 1e12)  # pm

    xc_peak_list = np.array(xc_peak_list)
    clipped = xc_peak_list > delta_y_max_pm
    g_clip = GAMMAS_DEFAULT[np.argmax(clipped)] if clipped.any() else None
    if g_clip is not None:
        print(f"AOD clips at g_fb = {g_clip:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"$X_0 = P_0 = 0$", fontsize=14)

    ax = axes[0]
    ax.loglog(GAMMAS_DEFAULT, n_list, "o-", color="steelblue")
    if g_clip is not None:
        ax.axvline(g_clip, color="red", linestyle="--", label=f"AOD clip at g={g_clip:.3f}")
        ax.legend()
    ax.set_xlabel(r"$g_{fb}$", fontsize=13)
    ax.set_ylabel(r"$\langle n \rangle$", fontsize=13)
    ax.set_title("Occupation Number vs Feedback Gain", fontsize=13)
    ax.grid(True, which="both")

    ax = axes[1]
    ax.loglog(GAMMAS_DEFAULT, xc_peak_list, "s--", color="tomato", label=r"Peak $|\Delta y|$ required")
    ax.axhline(delta_y_max_pm, color="black", linewidth=2, linestyle=":",
               label=f"AOD limit = {delta_y_max_pm:.0f} pm")
    if g_clip is not None:
        ax.axvline(g_clip, color="red", linestyle="--", label=f"Clip at g={g_clip:.3f}")
    ax.set_xlabel(r"$g_{fb}$", fontsize=13)
    ax.set_ylabel(r"$\Delta y$ [pm]", fontsize=13)
    ax.set_title("Required vs Maximum Tweezer Displacement", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both")

    fig.tight_layout()
    return fig, GAMMAS_DEFAULT, n_list, xc_peak_list, xc_rms_list


if __name__ == "__main__":
    fig, gammas, n_list, xc_peak, xc_rms = main()
    plt.show()
