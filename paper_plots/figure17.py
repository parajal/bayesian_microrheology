import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import InferenceProcedure

n_configs = 120
G_bulk = 19.08
het_dir = "data/linear_viscoelastic/particle-particle-het"

G_loc, G_std = [], []
for k in range(1, n_configs + 1):
    print(f"\n===== config{k} =====")
    model = InferenceProcedure(
        force=8 * np.pi,
        material_model="viscoelastic",
        boundary_model="unbounded",
        theta=0,
        L=10.0,
        hasimoto_corr=True,
        t_unload=0.2,
        sigma_bias=None,
        sigma_noise_percent=2.0,
        nsteps=10000,
        thin_factor=1, theta_true=[0.5, 0.9, 0.1], seed=42)
    model.load_data(f"{het_dir}/config{k}/particle_1_trajectory_full.out")
    model.run_mcmc(warmup=True)

    lab = model._get_parameter_labels(latex=False)
    G = model.samples[:, lab.index("eta_p")] / model.samples[:, lab.index("lambda_")]
    G_loc.append(float(G.mean()))
    G_std.append(float(G.std()))

G_loc = np.array(G_loc)
G_bar = G_loc.mean()
print(f"\nG_loc mean over {n_configs} configs = {G_bar:.2f}  (bulk {G_bulk})")

output_file = ROOT / "G_mean_std_80_configs.txt"

np.savetxt(
    output_file,
    np.column_stack((np.arange(1, n_configs + 1), G_loc, G_std)),
    header="config\tG_mean\tG_std",
    fmt=["%d", "%.8e", "%.8e"],
    delimiter="\t",
)

print(f"Saved results to {output_file}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(G_loc, bins=12, color="steelblue", alpha=0.75, edgecolor="black")
ax.axvline(G_bar, color="red", lw=2, label=rf"$\bar G_{{\mathrm{{loc}}}} = {G_bar:.2f}$")
ax.axvline(G_bulk, color="black", ls="--", lw=2, label=rf"$G_{{\mathrm{{eff}}}} = {G_bulk:.2f}$")
ax.set(xlabel=r"$G$")
ax.legend()
fig.savefig('G_histogram.pdf', dpi=300, bbox_inches='tight')
plt.show()
