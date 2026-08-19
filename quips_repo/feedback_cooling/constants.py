"""
Physical parameters and natural units for the levitated-nanosphere
feedback-cooling simulation.

The simulation itself is carried out in dimensionless "natural" units where

    hbar = m = omega0 = 1.

Position, momentum, and time in the code are expressed in the corresponding
natural length, momentum, and time scales of a mechanical oscillator with
mass ``MASS`` and resonance frequency ``OMEGA0`` (see Sec. 2.1-2.2 of the
report). Converting a dimensionless simulation quantity back to physical
units means multiplying by the appropriate scale below (e.g. multiply a
dimensionless position ``X`` by ``X_ZPF`` to get meters).
"""
import numpy as np

# --------------------------------------------------------------------------
# Fundamental constants
# --------------------------------------------------------------------------
HBAR = 6.62607015e-34 / (2 * np.pi)   # J s
KB = 1.380649e-23                     # J / K

# --------------------------------------------------------------------------
# Nanosphere / trap parameters (Sec. 2.2 of the report)
# --------------------------------------------------------------------------
MASS = 4.8e-18            # kg, silica nanosphere mass (4.8 fg)
F0 = 55e3                 # Hz, mechanical resonance frequency (z-mode)
OMEGA0 = 2 * np.pi * F0   # rad/s

WAVELENGTH = 1064e-9      # m, trapping laser wavelength
P_OPT = 500e-3            # W, optical power
NA_TRAP = 0.77            # numerical aperture of trapping lens
NA_COLLECTION = 0.5       # numerical aperture of collection lens

PARTICLE_DENSITY = 2.0e3  # kg / m^3
PARTICLE_DIAMETER = 166e-9  # m

# --------------------------------------------------------------------------
# Natural (dimensionless) unit scales
# --------------------------------------------------------------------------
X_ZPF = np.sqrt(HBAR / (2 * MASS * OMEGA0))   # zero-point length scale, m
T0 = HBAR * OMEGA0 / KB                       # hbar*omega0 / kB, K
T_GROUND = T0 / 2                             # ground-state temperature, K

# --------------------------------------------------------------------------
# Default measurement / simulation parameters (Sec. 2.2)
# --------------------------------------------------------------------------
ETA_DEFAULT = 1.0          # detection efficiency (ideal / unity)
KAPPA_DEFAULT = 1e-3       # dimensionless measurement strength
DT_DEFAULT = 5e-4          # dimensionless integration time step
WINDOW_DEFAULT = 20        # finite-difference window (steps) for raw-signal
                           # velocity estimate used by the "raw" feedback laws

GAMMAS_DEFAULT = np.logspace(-3, 0, 20)   # standard g_fb scan used throughout

# --------------------------------------------------------------------------
# AOD (acousto-optic deflector) constraint parameters (Sec. 3.3.3),
# taken from the Moore-group / Novotny-style apparatus numbers.
# --------------------------------------------------------------------------
AOD_G = 2.5e-3          # rad / MHz, AOD tilt-angle-per-frequency gain
AOD_DELTA_OMEGA_MAX = 20e6   # Hz, maximum drive-frequency excursion
AOD_F_EFF = 0.5e-3      # m, effective focal length of the trapping lens
AOD_M = 0.6             # linear magnification of the trap-forming optics


def thermal_occupation(T=300.0):
    """Average thermal occupation number <n> ~ kB*T / (hbar*omega0) for a
    bath at temperature T (Kelvin). Used to set the initial state of the
    particle before feedback cooling begins (Sec. 2.2)."""
    return KB * T / (HBAR * OMEGA0)


def aod_max_displacement_pm():
    """Maximum tweezer displacement achievable within the AOD's bandwidth,
    in picometers (Sec. 3.3.3, Eq. 19-20)."""
    delta_y_max_m = (AOD_G * AOD_DELTA_OMEGA_MAX * AOD_F_EFF / AOD_M) * 1e-6
    return delta_y_max_m * 1e12
