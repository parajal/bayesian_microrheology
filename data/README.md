# Data folder — column descriptions

This folder holds particle trajectories, wall-correction data, and spatial modulus data used by the paper plots.

Notation:

- `t` — time
- `X(t)`, `Y(t)`, `Z(t)` — particle position components
- `U_x(t)`, `U_y(t)`, `U_z(t)` — particle (translational) velocity components
- `Omega_x(t)`, `Omega_y(t)`, `Omega_z(t)` — particle angular velocity components

## Column layout by folder

| Folder | Files | Cols | Columns (in order) |
|---|---|:--:|---|
| `linear_viscoelastic/angle0`, `angle45`, `angle90` | `*.out` | 7 | `t`, `X(t)`, `Y(t)`, `Z(t)`, `U_x(t)`, `U_y(t)`, `U_z(t)` |
| `linear_viscoelastic/particle-particle` | `lx-*.out` | 10 | `t`, `X(t)`, `Y(t)`, `Z(t)`, `U_x(t)`, `U_y(t)`, `U_z(t)`, `Omega_x(t)`, `Omega_y(t)`, `Omega_z(t)` |
| `linear_viscoelastic/particle-particle-het/config1`–`config40` | `particle_1_trajectory_full.out` | 10 | `t`, `X(t)`, `Y(t)`, `Z(t)`, `U_x(t)`, `U_y(t)`, `U_z(t)`, `Omega_x(t)`, `Omega_y(t)`, `Omega_z(t)` |
| `linear_viscoelastic/unbounded` | `*.out` | 3 | `t`, `X(t)`, `U_x(t)` |
| `newtonian/angle-0`, `angle-45`, `angle-90` | `*.txt` | 7 | `t`, `X(t)`, `Y(t)`, `Z(t)`, `U_x(t)`, `U_y(t)`, `U_z(t)` |
| `newtonian/unbounded` | `*.txt` | 2 | `t`, `X(t)` |
| `newtonian/wall_corr` | `angle-0.txt`, `angle-90.txt` | 2 | `delta/a`, parallel or perpendicular drag correction factor, respectively |
| `nonlinear_viscoelastic` | `*.out` | 3 | `t`, `X(t)`, `U_x(t)` |
| `plot_G` | `bulk_stress.out` | 7 | `step`, `time`, `shear_rate`, `strain_increment`, `sigma_xy`, `delta_sigma_xy`, `macro_G` |

## Heterogeneous cases: `particle-particle-het`

`het` refers to a heterogeneous viscoelastic medium with spatially varying shear modulus. The folder contains 40 configurations (`config1`–`config40`), each storing the trajectory of particle 1. Each file has 51 rows spanning `t = 0` to `0.5` in steps of `0.01`, with the ten columns listed above.

[`figure17.py`](../paper_plots/figure17.py) fits each trajectory separately using the unbounded viscoelastic model with Hasimoto correction (`L = 10`), force `8*pi`, and unloading at `t = 0.2`. It adds 2% synthetic noise, computes the inferred local modulus `G = eta_p / lambda`, and plots the distribution of posterior mean moduli across configurations against the bulk reference `G_bulk = 17.14`.

The folder also includes the Fortran solver source `magnetic_viscoelastic_pardiso_full_8p.f90` and the Slurm launch script `run-1.sh`. The script defines a Gaussian modulus profile and currently specifies one particle at the origin. Configuration-specific inputs are not stored alongside the trajectories, so the exact changes between the 40 configurations are not recorded here.

## Modulus field: `plot_G`

- `flow0000.vtk` is a legacy binary VTK file containing mesh points and the scalar field `Gphi`. [`figure16.py`](../paper_plots/figure16.py) reads it to plot modulus slices at `z = -2.5, 0, 2.5`.
- `bulk_stress.out` contains the bulk shear response and macroscopic modulus, with a header identifying its seven columns.
