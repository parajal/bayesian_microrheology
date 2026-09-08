import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from packages.main import  InferenceProcedure
noise_percents = [0.5, 2, 5, 10, 20]
for noise in noise_percents: 
    print("noise added:", noise)
    model = InferenceProcedure(
        force=12*np.pi,
        theta=0, 
        a=1.0,
        eta_s_bounds = (0.01, 100), #prior
        material_model="newtonian",
        boundary_model="unbounded", 
        sigma_noise_percent= noise,
        nsteps=10000,
        thin_factor=2, 
        sigma_bias = None,

        theta_true = [1])

    model.load_data("data/newtonian/unbounded/delta-100-0.txt")
    model.run_mcmc()
