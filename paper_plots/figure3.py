"""Figure 3: Newtonian particle displacement near a wall, SAM vs FOM.

Three panels in one row, one per force angle theta = 0, 45, 90 degrees. Each
panel overlays the semi-analytical model (SAM, lines) on the full-order-model
data (FOM, markers) for wall gaps delta0/a = 0.1, 0.01, 0.001. The plotted
component is the one the model fits: X for theta = 0, 45 and Y for theta = 90.

Run:  python paper_plots/figure3.py
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.forward_models import ForwardModels
from packages.wall_corrections import WallCorrections

a = 1.0                 # particle radius
F = 12 * np.pi          # force magnitude
eta_s = 1.0             # solvent viscosity
t_final = 5.8           # final time
nsteps = 120            # time steps for the SAM curve
n_brenner_terms = 100   # terms in the Brenner (1961) series

thetas = [0, 45, 90]              # force angles (degrees), one panel each
deltas = [0.1, 0.01, 0.001]       # wall gaps delta0/a
colors = ["C0", "C1", "C2"]       # one colour per gap
data_dir = ROOT / "data" / "newtonian"

t = np.linspace(0, t_final, nsteps)


class SAM(ForwardModels, WallCorrections):
    """Minimal bounded-Newtonian forward model built from the package mixins."""

    def __init__(self, theta):
        self.theta = theta
        self.a = a
        self.boundary_model = "bounded"
        self.delta0 = None
        self.n_brenner_terms = n_brenner_terms
        self.bounds = {"delta0": (1e-4, 1e3)}
        self.rtol = 1e-7        # solve_ivp tolerances (match the package default)
        self.atol = 1e-9

    def is_perpendicular(self):
        return abs(self.theta - 90.0) < 1e-6


def load_fom(theta, delta0):
    """FOM time and fit-component displacement (X for 0/45, Y-Y0 for 90)."""
    raw = np.loadtxt(data_dir / f"angle-{theta}" / f"delta-{delta0}-{theta}.txt")
    thin = max(1, len(raw) // 30) 
    raw = raw[::thin]
    if abs(theta - 90) < 1e-6:
        return raw[:, 0], raw[:, 2] - raw[0, 2]
    return raw[:, 0], raw[:, 1]

def plot_panel(ax, theta):
    sam = SAM(theta)
    for delta0, color in zip(deltas, colors):
        disp_sam = sam.model_newtonian(eta_s, t, F, delta0=delta0)
        ax.plot(t, disp_sam, "-", color=color, label=rf"SAM $\delta_0$={delta0}")
        t_fom, disp_fom = load_fom(theta, delta0)
        ax.plot(t_fom, disp_fom, "o", mfc="none", color=color, label=rf"FOM $\delta_0$={delta0}")
    ylabel = "Y" if abs(theta - 90) < 1e-6 else "X"
    ax.set(xlabel="t", ylabel=ylabel, title=rf"$\theta = {theta}^\circ$")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for ax, theta in zip(axes, thetas):
    plot_panel(ax, theta)
plt.show()
