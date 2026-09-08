import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.forward_models import ForwardModels
from packages.data_io import hasimoto_Q

a = 1.0                       # particle radius
F = 8 * np.pi                 # force magnitude
eta_s, eta_p, lam = 0.5, 0.9, 0.1   # solvent / polymer viscosity, relaxation time
t_unload = 0.2                # force removed here (creep -> recovery)
t_final = 0.5                 # final time
nsteps = 200                  # points on the SAM curve

Ls = [5, 10, 20, 40, 60]                                    # periodic box sizes
colors = ["tab:orange", "tab:blue", "tab:gray", "tab:red", "tab:cyan"]
data_dir = ROOT / "data" / "linear_viscoelastic" / "particle-particle"

t = np.linspace(0, t_final, nsteps)


class SAM(ForwardModels):
    """Unbounded viscoelastic forward model (analytic creep/recovery)."""

    def __init__(self):
        self.theta = 0
        self.a = a
        self.boundary_model = "unbounded"

    def is_perpendicular(self):
        return False

x_sam = SAM().model_viscoelastic(eta_s, eta_p, lam, t, F, t_unload=t_unload, delta0 = None, component="x")

def load_fom(L):
    raw = np.loadtxt(data_dir / f"lx-{L}.out")
    return raw[:, 0], raw[:, 1]   # columns: t, X(t)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.0, 6.0), constrained_layout=True)

for L, color in zip(Ls, colors):
    t_fom, x_fom = load_fom(L)
    axL.plot(t_fom, x_fom, ".", ms=7, color=color, label=f"L = {L}")
    axR.plot(t_fom, x_fom / hasimoto_Q(L), ".", ms=7, color=color, label=f"L = {L}")

for ax in (axL, axR):
    ax.plot(t, x_sam, "-", color="black", lw=1.5, label="unbounded SAM")
    ax.axvline(t_unload, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set(xlabel=r"$t$", xlim=(0, 0.5), ylim=(0, 0.30),
           xticks=[0, 0.25, 0.5], yticks=[0, 0.15, 0.30])
    ax.grid(True, alpha=0.3)
    ax.legend()
axL.set_ylabel(r"$X(t)$")
axR.set_ylabel(r"$X(t)/Q(L)$")
plt.show()
