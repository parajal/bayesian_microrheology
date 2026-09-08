import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages import  JointInferenceProcedure

datasets = [
    dict(filename="data/nonlinear_viscoelastic/16pi.out", force=16*np.pi, thin_factor=2, append = True),
    dict(filename="data/nonlinear_viscoelastic/32pi.out", force=32*np.pi,  thin_factor=2, append = True), 
    dict(filename="data/nonlinear_viscoelastic/64pi.out", force=64*np.pi,  thin_factor=2, append = True)]

model = JointInferenceProcedure(
    force=8*np.pi,
    a=1.0,
    theta=0,
    material_model="viscoelastic",
    boundary_model="unbounded",
    sigma_noise_percent=2.0,
    sigma_bias=None,
    nsteps=10000,
    theta_true = [0.5, 0.9, 0.1])

model.load_datasets(datasets)
model.plot_data()
samples = model.run_mcmc(warmup=True)
model.plot_posterior_predictive()