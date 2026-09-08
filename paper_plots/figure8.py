import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure

model = InferenceProcedure(
    force=8*np.pi,
    a = 1,
    theta=0,
    t_unload=0.2,
    material_model="viscoelastic",
    boundary_model="unbounded",
    sigma_bias=None,
    sigma_noise_percent=2.0,
    nsteps=10000,
    thin_factor=2, theta_true=[0.5, 0.9, 0.1])

model.load_data("data/linear_viscoelastic/unbounded/delta-100-0.out")
model.run_mcmc( warmup = True)
model.plot_corner(physical_only = False)