import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import InferenceProcedure

noise_percents = [0.5, 2, 5, 10, 20]
for noise in noise_percents: 
    print("noise added:", noise)
    model = InferenceProcedure(
        force=8*np.pi,
        theta=0,
        a = 1.0,
        t_unload=0.2,
        material_model="viscoelastic",
        boundary_model="unbounded",
        sigma_bias=None,
        sigma_noise_percent=noise,
        nsteps=10000,
        thin_factor=2, theta_true=[0.5, 0.9, 0.1])

    model.load_data("data/linear_viscoelastic/unbounded/delta-100-0.out")
    model.run_mcmc( warmup = True)

