"also produces table 9 (bottom)"
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure

model = InferenceProcedure(
    force=12*np.pi,
    material_model="newtonian",
    boundary_model="bounded",
    theta=45,
    delta0 = 0.1,
    sigma_noise_percent=2.0,
    sigma_bias = None,
    nsteps=10000,
    thin_factor=1, theta_true = [1, 0.1], seed = 42)

model.load_data("data/newtonian/angle-45/delta-0.1-45.txt")
model.run_mcmc(warmup = True)
model.plot_corner(physical_only = False)