import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages.main import  InferenceProcedure

deltas = [0.1, 0.01, 0.005]
thin_factor = [2, 2, 4]
eta_s_true, eta_p_true, lam_true = 0.5, 0.9, 0.1

for delta, thin in zip(deltas, thin_factor):
    print(f"\n===== delta0 = {delta} =====")
    model = InferenceProcedure(
        force=8 * np.pi,
        material_model="viscoelastic",
        boundary_model="bounded",
        theta=45,
        t_unload=0.2,
        delta0=delta,
        sigma_noise_percent=2.0,
        nsteps=10000,
        use_y=True,
        thin_factor=thin, theta_true=[eta_s_true, eta_p_true, lam_true, delta], seed=42)

    model.load_data(f"data/linear_viscoelastic/angle45/delta-{delta:g}-45.out")
    model.run_mcmc(warmup=True) 
