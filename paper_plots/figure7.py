import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.forward_models import ForwardModels

a = 1.0                              # particle radius
F = 8 * np.pi                        # force magnitude
eta_s, eta_p, lam = 0.5, 0.9, 0.1    # solvent / polymer viscosity, relaxation time
t_unload = 0.2                       # force removed here (creep -> recovery)
seed = 42                            # noise seed (reproducible)
thin = 2                             # keep every 2nd FOM point
noise_pcts = [0.5, 2.0, 5.0]         # sigma as % of X_max, one per panel
data_file = ROOT / "data" / "linear_viscoelastic" / "unbounded" / "delta-100-0.out"

raw = np.loadtxt(data_file)
t, x_fom = raw[:, 0], raw[:, 1]      # columns: t, X(t), U_x(t)
t, x_fom = t[::thin], x_fom[::thin]  # thinning
x_max = np.max(np.abs(x_fom))


class SAM(ForwardModels):
    """Unbounded viscoelastic forward model (analytic creep/recovery)."""

    def __init__(self):
        self.theta = 0
        self.a = a
        self.boundary_model = "unbounded"

    def is_perpendicular(self):
        return False


x_sam = SAM().model_viscoelastic(eta_s, eta_p, lam, t, F, t_unload=t_unload, component="x")

fig, axes = plt.subplots(1, 3, figsize=(24.0, 6.0), constrained_layout=True)
for ax, pct in zip(axes, noise_pcts):
    rng = np.random.default_rng(seed)   # fresh per panel, matching data_io.load_data
    noisy = x_fom + rng.normal(0, pct / 100 * x_max, size=x_fom.shape)
    ax.plot(t, x_sam, "-", color="red", lw=2, label="SAM")
    ax.plot(t, noisy, "o", color="black", ms=6, label="FOM + noise")
    ax.set(xlabel=r"$t$", title=rf"$\sigma_\mathrm{{exp}} = {pct:g}\%\,X_\mathrm{{max}}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[0].set_ylabel(r"$X(t)$")

plt.show()
