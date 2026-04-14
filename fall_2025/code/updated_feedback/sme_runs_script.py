import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from sme_evolution_2d import *

plot_sme_traj_2D(del_t=0.0001,period_number=10000, eta=1, k=0.05, gamma=0.01, x0=[20,20], p0=[20,20], feedback='parametric', filt='kalman', M=None)