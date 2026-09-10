import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",

    "font.size": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.0,
})
l_biases = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 12.0]

gaps = {
    0.1:   {0: "data/newtonian/angle-0/delta-0.1-0.txt",
            90: "data/newtonian/angle-90/delta-0.1-90.txt"},
    0.001: {0: "data/newtonian/angle-0/delta-0.001-0.txt",
            90: "data/newtonian/angle-90/delta-0.001-90.txt"},
}
angle_style = {0: dict(color="blue", label=r"$0^\circ$"),
               90: dict(color="red", label=r"$90^\circ$")}
thin_by_gap = {0.1: 1, 0.001: 4}   # thin_factor per gap


def sweep(data_file, theta, thin):
    """Return sigma_bias/sigma_r posterior mean and 95% band over l_biases."""
    mean, lo, hi = [], [], []
    for lb in l_biases:
        print(f"\n===== delta {Path(data_file).stem}, l_bias = {lb} =====")
        model = InferenceProcedure(
            force=12 * np.pi,
            material_model="newtonian",
            boundary_model="unbounded",
            theta=theta,
            sigma_noise_percent=2.0,
            sigma_bias="infer",
            nsteps=10000,
            l_bias=lb,
            thin_factor=thin, seed=42)
        model.load_data(data_file)
        model.run_mcmc(warmup=True)

        labels = model._get_parameter_labels(latex=False)
        sb = model.samples[:, labels.index("sigma_bias")] / model.sigma_realized
        mean.append(sb.mean())
        lo.append(sb.mean() - 1.96 * sb.std())
        hi.append(sb.mean() + 1.96 * sb.std())
    return np.array(mean), np.clip(lo, 0, None), np.array(hi)


subcaps = ["(a)", "(b)"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
for idx, (ax, (gap, files)) in enumerate(zip(axes, gaps.items())):
    for theta, data_file in files.items():
        m, lo, hi = sweep(data_file, theta, thin_by_gap[gap])
        st = angle_style[theta]
        ax.plot(l_biases, m, "-o", color=st["color"], ms=4, label=st["label"])
        ax.fill_between(l_biases, lo, hi, color=st["color"], alpha=0.2)
    ax.set(
        xscale="log",
        xlabel=r"$\ell_{\mathrm{bias}}$",
        ylabel=r"$\sigma_{\mathrm{bias}}/\sigma_{\mathrm{r}}$",
        ylim=(0, 20))
    ax.set_xticks(l_biases)
    ax.set_xticklabels([f"{lb:g}" for lb in l_biases])
    ax.grid(True, alpha=0.3)
    ax.legend()
    # subfigure caption (a)/(b) with delta0/a, centered below each panel
    ax.text(0.5, -0.32, rf"{subcaps[idx]} $\delta_0/a = {gap:g}$",
            transform=ax.transAxes, ha="center", va="top")
plt.savefig('l_bias_sensitivity.pdf', bbox_inches='tight', dpi=300)
plt.show()
