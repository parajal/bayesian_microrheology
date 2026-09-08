import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure

thetas = [0, 45, 90]
datasets = ["data/newtonian/angle-0/delta-0.1-0.txt", "data/newtonian/angle-45/delta-0.1-45.txt", "data/newtonian/angle-90/delta-0.1-90.txt"]
for theta, data in zip(thetas, datasets):
    model = InferenceProcedure(
        force=12*np.pi,
        material_model="newtonian",
        boundary_model="unbounded",
        theta=theta,
        sigma_noise_percent=2.0,
        sigma_bias = "infer",
        nsteps=10000,
        l_bias = 6,
        thin_factor=1, theta_true = [1, 0.1], seed = 42)

    model.load_data(data)
    model.run_mcmc(warmup = True)
    model.plot_posterior_predictive()