import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure

model = InferenceProcedure(
    force=8 * np.pi,
    material_model="viscoelastic",
    boundary_model="bounded",
    theta=45,
    t_unload=0.2,
    delta0 = 0.1,
    sigma_bias=None,
    sigma_noise_percent=2.0,
    nsteps=10000,
    use_y = True,
    thin_factor=4, theta_true=[0.5, 0.9, 0.1, 0.1])

model.load_data("data/linear_viscoelastic/angle45/delta-0.1-45.out")
model.plot_data()
model.run_mcmc( warmup = True)
model.plot_corner(physical_only = False)