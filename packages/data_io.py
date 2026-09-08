import os
import numpy as np

def hasimoto_Q(L):
    return 1 - 2.8373 / L + (4 * np.pi / 3) / L**3

class DataIO:
    def _fit_components(self):
        if self.is_perpendicular():
            return ("y",)
        return ("x", "y") if (
            self.use_y
            and self.material_model == "viscoelastic"
            and self.boundary_model == "bounded") else ("x",)

    def load_data(self, filename, prior_fraction=0.1):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = np.atleast_2d(np.loadtxt(os.path.join(root, filename)))
        comps = self._fit_components()

        dataset = {
            "t": data[::self.thin_factor, 0],
            "x": data[::self.thin_factor, 1] if "x" in comps else None,
            "y": (data[:, 2] - data[0, 2])[::self.thin_factor] if "y" in comps else None,
            "fit_components": comps,
            "F": self.force,
            "Fx": self.force * np.cos(np.deg2rad(self.theta)),
            "Fy": self.force * np.sin(np.deg2rad(self.theta)),}

        if self.material_model == "viscoelastic":
            dataset["t_unload"] = self.t_unload

        if self.hasimoto_corr:
            q = hasimoto_Q(self.L)
            for c in comps:
                dataset[c] /= q
            dataset["hasimoto_Q"] = q

        rng = np.random.default_rng(self.seed)
        max_disp = {c: np.abs(dataset[c]).max() for c in comps}

        noises = {}
        for c in comps:
            sigma = self.sigma_noise_percent / 100 * max_disp[c]
            noises[c] = rng.normal(0, sigma, len(dataset[c]))
            dataset[c] += noises[c]

        self.beta = 1 / (prior_fraction * max(max_disp.values()))
        self.sigma_noise_prior = self.sigma_bias_prior = self.beta
        self.sigma_realized = np.std(np.concatenate(list(noises.values())), ddof=1)

        dataset["max_displacement"] = max_disp
        dataset["sigma_prior_beta"] = self.beta
        self.data = dataset

        print(f"Loaded {os.path.basename(filename)} "
              f"(angle={self.theta:.1f}, n={len(dataset['t'])}, "
              f"components={','.join(comps)})")
        print(f"realized noise = {self.sigma_realized:.5f}")