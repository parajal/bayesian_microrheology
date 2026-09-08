import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


HERE = Path(__file__).resolve().parent
VTK = HERE.parent / "data" / "plot_G" / "flow0000.vtk"
OUTPUT = Path(
    "C:/Users/paraj/Documents/mulit-particle/plot_G/G_slices_combined.pdf"
)

Z_SLICES = [-2.5, 0.0, 2.5]
LIM = 5.0
NG = 240


def read_vtk(filename):
    # Read points and Gphi from a legacy binary VTK file.
    data = filename.read_bytes()
    header = re.search(rb"POINTS\s+(\d+)\s+float[ \t]*\r?\n", data)
    if header is None:
        raise ValueError("Could not find a float POINTS header.")

    npts = int(header[1])
    start = header.end()
    pts = np.frombuffer(data, ">f4", npts * 3, start).reshape(-1, 3)

    header = re.search(
        rb"SCALARS[ \t]+Gphi[ \t]+float(?:[ \t]+1)?[ \t]*\r?\n"
        rb"LOOKUP_TABLE[^\n]*\n",
        data[start + npts * 12:],
    )
    if header is None:
        raise ValueError("Could not find a scalar float Gphi field.")

    start += npts * 12 + header.end()
    G = np.frombuffer(data, ">f4", npts, start)
    return pts.astype(float), G.astype(float)


# Build interpolators and a shared grid for all slices.
pts, G = read_vtk(VTK)
linear = LinearNDInterpolator(pts, G)
nearest = NearestNDInterpolator(pts, G)

x = np.linspace(-LIM, LIM, NG)
X, Y = np.meshgrid(x, x)
levels = np.linspace(G.min(), G.max(), 51)

fig, axes = plt.subplots(
    1, len(Z_SLICES), figsize=(18, 6), constrained_layout=True, squeeze=False
)

for ax, z in zip(axes.flat, Z_SLICES):
    # Fill points outside the linear interpolation domain with nearest values.
    Z = np.full_like(X, z)
    values = linear(X, Y, Z)
    mask = np.isnan(values)
    values[mask] = nearest(X[mask], Y[mask], Z[mask])

    contour = ax.contourf(X, Y, values, levels=levels, cmap="coolwarm")
    ax.set(title=f"z = {z}", xlabel="x", aspect="equal")

# Add a shared colorbar and save the figure.
axes[0, 0].set_ylabel("y")
fig.colorbar(contour, ax=axes.ravel().tolist(), orientation="horizontal", label="G")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=300)
plt.show()
