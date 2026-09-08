"Produces Table 8 and figure 9 a, b and c."
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure

data_dir = ROOT / "data" / "nonlinear_viscoelastic"
force_ns = [8, 16, 32, 64, 128, 256, 512, 1024]
colors = plt.cm.viridis(np.linspace(0, 0.95, len(force_ns)))

#(a)
fig, ax = plt.subplots(figsize=(8, 6))
for n, color in zip(force_ns, colors):
    c = n // 8                                        
    t, x = np.loadtxt(data_dir / f"{n}pi.out")[:, :2].T 
    ax.plot(t, x / c, "-", color=color, lw=1.5, label=f"c = {c}")
ax.set(xlabel=r"$t$", ylabel=r"$X(t)/c$", xlim=(0, 0.5), ylim=(0, 0.3),
       xticks=[0, 0.25, 0.5], yticks=[0, 0.1, 0.2, 0.3])
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)
plt.show()

model = InferenceProcedure(
    force= 1024* np.pi,
    material_model="viscoelastic",
    boundary_model="unbounded",
    theta=0,
    t_unload=0.2,
    sigma_bias=None,
    sigma_noise_percent=2.0,
    nsteps=10000,
    thin_factor=10, theta_true=[0.5, 0.9, 0.1])

model.load_data("data/nonlinear_viscoelastic/1024pi.out")
model.plot_data()                    # (b)
model.run_mcmc( warmup = True)
model.plot_posterior_predictive()    # (c)
