"also produces table 9 (top)"
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from packages import  JointInferenceProcedure

datasets = [
    dict(filename="data/nonlinear_viscoelastic/1024pi.out", force=1024*np.pi, thin_factor=10),
    dict(filename="data/nonlinear_viscoelastic/512pi.out", force=512*np.pi,  thin_factor=2), 
    dict(filename="data/nonlinear_viscoelastic/256pi.out", force=256*np.pi,  thin_factor=2)]

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