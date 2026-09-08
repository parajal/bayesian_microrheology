import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.forward_models import ForwardModels

a = 1.0                       # particle radius
F = 12 * np.pi                # force magnitude
eta_s = 1.0                   # solvent viscosity
seed = 42                     # noise seed (reproducible)
thin = 2                      # keep every 2nd FOM point
noise_pcts = [0.5, 2.0, 5.0]  # sigma as % of X_max, one per panel
data_file = ROOT / "data" / "newtonian" / "unbounded" / "delta-100-0.txt"

t, x_fom = np.loadtxt(data_file, unpack=True)   # columns: t, X(t)
t, x_fom = t[::thin], x_fom[::thin]             # thinning
x_max = np.max(np.abs(x_fom))


class SAM(ForwardModels):
    """Unbounded Newtonian forward model (analytic)."""

    def __init__(self):
        self.theta = 0
        self.a = a
        self.boundary_model = "unbounded"

    def is_perpendicular(self):
        return False


x_sam = SAM().model_newtonian(eta_s, t, F)

fig, axes = plt.subplots(1, 3, figsize=(24.0, 6.0), constrained_layout=True)
for ax, pct in zip(axes, noise_pcts):
    rng = np.random.default_rng(seed)
    noisy = x_fom + rng.normal(0, pct / 100 * x_max, size=x_fom.shape)
    ax.plot(t, x_sam, "-", color="red", lw=2, label="SAM")
    ax.plot(t, noisy, "o", color="black", ms=6, label="FOM + noise")
    ax.set(xlabel=r"$t$", title=rf"$\sigma_\mathrm{{exp}} = {pct:g}\%\,X_\mathrm{{max}}$")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0)
    ax.legend()
axes[0].set_ylabel(r"$X(t)$")
plt.show()
