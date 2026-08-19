# Feedback Cooling of a Levitated Nanosphere

Simulation code accompanying "Feedback Cooling Notes" (Ro'aa Alkhawaja,
Quantum Measurements Group, LBNL, April 2026). Simulates continuous
measurement and feedback cooling of a silica nanosphere levitated in a
single-beam optical tweezer, comparing direct-momentum feedback vs.
beam-center modulation, and raw-measurement-signal feedback vs.
Kalman-filtered feedback.

## Structure

```
feedback_cooling/       core package
    constants.py         physical parameters & natural units (Sec. 2.2)
    simulate.py           stochastic trajectory integrators (Sec. 2.1, 3.1-3.3)
    analysis.py            temperature / occupation number / decay-time helpers
figures/                 one script per report figure, each runnable standalone
    fig2_decay_constant_vs_gain.py
    fig3_temperature_raw_signal.py
    fig4_temperature_kalman.py
    fig5_occupation_number.py
    fig6_aod_constraint.py
requirements.txt
```

## Physics summary

The particle's conditional state is evolved via the stochastic master
equation for a continuously homodyne-monitored harmonic oscillator
(Eqs. 1-5 of the report). Four feedback laws are implemented in
`feedback_cooling/simulate.py`:

| function                   | actuation            | feedback signal          |
|-----------------------------|-----------------------|----------------------------|
| `simulate_direct_raw`       | direct momentum kick  | raw measurement record `alpha` |
| `simulate_beam_mod_raw`     | trap-center shift     | raw measurement record `alpha` |
| `simulate_direct_kalman`    | direct momentum kick  | Kalman-filtered state estimate |
| `simulate_beam_mod_kalman`  | trap-center shift     | Kalman-filtered state estimate |

The "raw" feedback laws estimate velocity by finite-differencing the
noisy measurement record; because that record carries shot noise, large
feedback gain reinjects that noise as force noise, producing the
reheating seen in Fig. 3. The Kalman-filtered laws use the LQR-optimal
gain (Sec. 3.1, Eq. 13) applied to the already-filtered state, and
saturate near the ground state instead (Fig. 4).

## Running

```bash
pip install -r requirements.txt
python figures/fig3_temperature_raw_signal.py
python figures/fig4_temperature_kalman.py
python figures/fig5_occupation_number.py
python figures/fig6_aod_constraint.py
python figures/fig2_decay_constant_vs_gain.py
```

Each script's `main()` returns the figure and underlying data arrays, so
they can also be imported and called from a notebook, e.g.:

```python
from figures.fig3_temperature_raw_signal import main
fig, gammas, T_list = main(n_steps=int(2.5e6))
```

At the default `n_steps = 2.5e6` used throughout the report, each single
`g_fb` value takes a few seconds after Numba JIT warm-up; the full 20-point
scans in `fig3`-`fig6` take a few minutes each, and `fig2` (which averages
5 runs per gain, per initial condition) takes longer.

## Notes on cleanup from the original notebook

This is a reorganized version of the exploratory research notebook. A few
things worth knowing if you're comparing against old plots or the PDF:

- Only the final, working version of each simulation was kept. The
  original notebook has ~80 cells including dead-end approaches, dated
  scratch notes, and cells with incomplete/broken code from debugging
  sessions; none of that carried over.
- `fig2_decay_constant_vs_gain.py` reproduces the notebook's final
  decay-constant calculation, but its docstring flags a mismatch between
  what the code varies (initial *mean* position/momentum) and what the
  report's Fig. 2 caption currently describes (initial *variance*) —
  worth resolving before this goes out.
- Physical-unit conversions (Sec. 5.1 harmonic-limit checks, the
  Novotny-paper-style calibration between `kappa` and `NA`/laser
  power/efficiency mentioned in the report's future-work section) were
  exploratory in the notebook and are not yet included here as reusable
  functions.
