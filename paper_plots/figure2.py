import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.wall_corrections import WallCorrections

a = 1.0                 # particle radius
n_brenner_terms = 100   # terms in the Brenner (1961) series
delta_min = 1e-2        # smallest delta/a on the analytical curve
delta_max = 1e2         # largest  delta/a on the analytical curve
n_curve = 400           # points on each analytical curve
data_dir = ROOT / "data" / "newtonian" / "wall_corr"
delta = np.geomspace(delta_min, delta_max, n_curve)

wc = WallCorrections()
wc.a = a
wc.n_brenner_terms = n_brenner_terms
wc.bounds = {"delta0": (delta_min, delta_max)}

# load the FOM data at theta = 0 (parallel) and theta = 90 (perpendicular)
fom_parallel = np.loadtxt(data_dir / "angle-0.txt")    # columns: delta/a, f_parallel
fom_perp = np.loadtxt(data_dir / "angle-90.txt")       # columns: delta/a, f_perp

def plot_panel(ax, fom, curve, label, ylabel, marker, color, theta, yscale):
    ax.plot(delta, curve(delta), "-", color=color, lw=2, label=label)
    ax.scatter(fom[:, 0], fom[:, 1], marker=marker, s=70, facecolors="none",
               edgecolors="black", linewidths=1.3, zorder=5,
               label=rf"FOM ($\theta={theta}^\circ$)")
    ax.set(xscale="log", yscale=yscale, xlabel=r"$\delta/a$", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


fig, (ax0, ax90) = plt.subplots(1, 2, figsize=(16, 6))
plot_panel(ax0, fom_parallel, wc.zeng_parallel, "Zeng et al. (2009)",
           r"$f_\parallel$", "o", "tab:blue", theta=0, yscale="linear")
plot_panel(ax90, fom_perp, wc.brenner_perpendicular, "Brenner (1961)",
           r"$f_\perp$", "s", "tab:red", theta=90, yscale="log")
ax0.set_ylim(1, 4)
plt.tight_layout()
plt.show()
