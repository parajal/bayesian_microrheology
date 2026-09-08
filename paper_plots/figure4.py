"""Figure 4: viscoelastic particle displacement near a wall, SAM vs FOM.

Three panels in one row, one per force angle theta = 0, 45, 90 degrees. Each
panel overlays the semi-analytical model (SAM, lines) on the full-order-model
data (FOM, markers) for wall gaps delta0/a = 0.1, 0.01, 0.005. The force is
applied until t_unload and then removed (creep followed by recovery). The
plotted component is X for theta = 0, 45 and Y for theta = 90.

Run:  python paper_plots/figure4.py
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
F = 8 * np.pi           # force magnitude
eta_s = 0.5             # solvent viscosity
eta_p = 0.9             # polymer viscosity
lam = 0.1               # relaxation time
t_unload = 0.2          # force removed here (creep -> recovery)
t_final = 0.5           # final time
nsteps = 200            # time steps for the SAM curve
n_brenner_terms = 100   # terms in the Brenner (1961) series

thetas = [0, 45, 90]              # force angles (degrees), one panel each
deltas = [0.1, 0.01, 0.005]       # wall gaps delta0/a
colors = ["C0", "C1", "C2"]       # one colour per gap
data_dir = ROOT / "data" / "linear_viscoelastic"

t = np.linspace(0, t_final, nsteps)


class SAM(ForwardModels, WallCorrections):
    """Minimal bounded-viscoelastic forward model built from the package mixins."""

    def __init__(self, theta, delta0):
        self.theta = theta
        self.a = a
        self.boundary_model = "bounded"
        self.delta0 = delta0
        self.n_brenner_terms = n_brenner_terms
        self.bounds = {"delta0": (1e-4, 1e3)}
        self.rtol = 1e-7        # solve_ivp tolerances (match the package default)
        self.atol = 1e-9

    def is_perpendicular(self):
        return abs(self.theta - 90.0) < 1e-6


def component_of(theta):
    return "y" if abs(theta - 90) < 1e-6 else "x"


def load_fom(theta, delta0):
    """FOM time and fit-component displacement (X for 0/45, Y-Y0 for 90)."""
    raw = np.loadtxt(data_dir / f"angle{theta}" / f"delta-{delta0}-{theta}.out")
    thin = max(1, len(raw) // 40)          # keep ~40 markers per curve
    raw = raw[::thin]
    if abs(theta - 90) < 1e-6:
        return raw[:, 0], raw[:, 2] - raw[0, 2]
    return raw[:, 0], raw[:, 1]


def plot_panel(ax, theta):
    comp = component_of(theta)
    for delta0, color in zip(deltas, colors):
        sam = SAM(theta, delta0)
        disp_sam = sam.model_viscoelastic(eta_s, eta_p, lam, t, F,
                                          t_unload=t_unload, component=comp)
        ax.plot(t, disp_sam, "-", color=color, label=rf"SAM $\delta_0$={delta0}")
        t_fom, disp_fom = load_fom(theta, delta0)
        ax.plot(t_fom, disp_fom, "o", mfc="none", color=color, label=rf"FOM $\delta_0$={delta0}")
    ax.axvline(t_unload, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set(xlabel="t", ylabel=comp.upper(), title=rf"$\theta = {theta}^\circ$")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()


fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for ax, theta in zip(axes, thetas):
    plot_panel(ax, theta)
plt.show()
