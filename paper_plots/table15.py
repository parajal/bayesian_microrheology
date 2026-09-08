import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure

#uncorrected
L = [5, 10, 20, 40, 60]
for length in L:
    model = InferenceProcedure(
        force=8 * np.pi,
        material_model="viscoelastic",
        boundary_model="unbounded",
        theta=0,
        L = length, 
        hasimoto_corr = False,
        t_unload=0.2,
        sigma_bias=None,
        sigma_noise_percent=2.0,
        nsteps=10000,
        thin_factor=2, theta_true=[0.5, 0.9, 0.1])

    model.load_data(f"data/linear_viscoelastic/particle-particle/lx-{length:g}.out")
    model.run_mcmc( warmup = True)

#Hasimoto-corrected 
for length in L:
    model = InferenceProcedure(
        force=8 * np.pi,
        material_model="viscoelastic",
        boundary_model="unbounded",
        theta=0,
        L = length, 
        hasimoto_corr = True,
        t_unload=0.2,
        sigma_bias=None,
        sigma_noise_percent=2.0,
        nsteps=10000,
        thin_factor=2, theta_true=[0.5, 0.9, 0.1])

    model.load_data(f"data/linear_viscoelastic/particle-particle/lx-{length:g}.out")
    model.run_mcmc( warmup = True)