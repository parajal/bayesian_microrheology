"""Table 11 (visualized): newtonian bounded inference across wall separations.

Fit the bounded Newtonian model (theta = 45) to bounded-domain FOM data and infer
(eta_s, delta0) plus the experimental-noise scale sigma_exp, at several gaps.
True eta_s = 1; 2% measurement noise; no bias term.

NOTE: the paper's Table 11 lists delta0/a in {0.005, 0.01, 0.1}, but the newtonian
angle-45 data provides {0.1, 0.01, 0.001} (no 0.005 file), so the smallest gap here
is 0.001. Edit `deltas` if you add a 0.005 dataset.

Run:  python paper_plots/table11.py   (runs 3 MCMC fits)
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import InferenceProcedure

deltas = [0.1, 0.01, 0.001]
eta_s_true = 1.0
thin_fac = [1, 1, 4]                    # thin_factor per gap
params = ["eta_s", "delta0", "sigma_noise"]
plabels = {"eta_s": r"$\eta_s$", "delta0": r"$\delta_0$", "sigma_noise": r"$\sigma_{\mathrm{exp}}$"}

res = {p: {"mean": [], "std": []} for p in params}
sigma_true = []
for delta, thin in zip(deltas, thin_fac):
    print(f"\n===== delta0 = {delta} =====")
    model = InferenceProcedure(
        force=12 * np.pi,
        material_model="newtonian",
        boundary_model="bounded",
        theta=45,
        delta0=delta,
        sigma_bias=None,
        sigma_noise_percent=2.0,
        nsteps=10000,
        thin_factor=thin, theta_true=[eta_s_true, delta], seed=42)
    model.load_data(f"data/newtonian/angle-45/delta-{delta:g}-45.txt")
    model.run_mcmc(warmup=True)

    sigma_true.append(model.sigma_realized)
    labels = model._get_parameter_labels(latex=False)
    for p in params:
        c = model.samples[:, labels.index(p)]
        res[p]["mean"].append(float(c.mean()))
        res[p]["std"].append(float(c.std()))

x = np.arange(len(deltas))
truth = {"eta_s": [eta_s_true] * len(deltas), "delta0": deltas, "sigma_noise": sigma_true}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, p in zip(axes, params):
    ax.errorbar(x, res[p]["mean"], yerr=res[p]["std"], fmt="o", color="C0",
                capsize=5, ms=7, label=r"posterior mean $\pm$ std")
    ax.plot(x, truth[p], "k--o", lw=1.5, ms=5, mfc="none", label="truth")
    if p in ("delta0", "sigma_noise"):
        ax.set_yscale("log")
    ax.set(xticks=x, xlabel=r"$\delta_0/a$", title=plabels[p])
    ax.set_xticklabels([f"{d:g}" for d in deltas])
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()
