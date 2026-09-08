import numpy as np
from matplotlib import pyplot as plt
from .main import InferenceProcedure
_PER_DATASET = ("force", "theta", "boundary_model", "t_unload", "hasimoto_corr", "L")
class JointInferenceProcedure(InferenceProcedure):

    def load_datasets(self, items):
        defaults = {k: getattr(self, k) for k in _PER_DATASET}
        self.datasets = []

        for item in items:
            cfg = {**defaults, **{k: v for k, v in item.items() if k != "filename"}}
            thin = int(cfg.pop("thin_factor", self.thin_factor))
            for k, v in cfg.items():
                setattr(self, k, v)
            self.thin_factor = thin
            self.load_data(item["filename"])
            self.data["config"] = {**cfg, "thin_factor": thin}
            self.datasets.append(self.data)

        peak = max(d["max_displacement"][c] for d in self.datasets for c in d["fit_components"])
        self.beta = 1 / (0.1 * peak)
        self.sigma_noise_prior = self.sigma_bias_prior = self.beta
        self.select(0)

        print(f"\nJoint fit over {len(self.datasets)} datasets")
        return self.datasets

    def select(self, i):
        """Point the model at dataset i (its data and per-dataset config)."""
        self.data = self.datasets[i]
        for k in _PER_DATASET:
            setattr(self, k, self.data["config"][k])
        self._t_unload_eff = self.data["config"]["t_unload"]
        return self

    def log_likelihood(self, phi):
        total = 0.0
        for i in range(len(self.datasets)):
            self.select(i)
            value = super().log_likelihood(phi)
            if not np.isfinite(value):
                return -np.inf
            total += value
        return float(total)

    def _dataset_labels(self):
        configs = [d["config"] for d in self.datasets]
        keys = ("force", "theta", "t_unload", "L", "boundary_model")
        varying = [k for k in keys if len({c[k] for c in configs}) > 1]

        labels = []
        for i, c in enumerate(configs):
            parts = []
            if "force" in varying:
                parts.append(rf"$F={c['force']/np.pi:g}\pi$")
            if "theta" in varying:
                parts.append(rf"$\theta={c['theta']:g}^\circ$")
            if "t_unload" in varying:
                parts.append(rf"$t_0={c['t_unload']:g}$")
            if "L" in varying:
                parts.append(rf"$L={c['L']:g}$")
            if "boundary_model" in varying:
                parts.append(c["boundary_model"])
            labels.append(", ".join(parts) or f"dataset {i}")
        return labels

    def _joint_components(self):
        return sorted({c for d in self.datasets for c in d["fit_components"]})

    def plot_data(self, theta_true=None):
        theta_true = self._resolve_theta_true(theta_true)
        if theta_true is None:
            raise ValueError("plot_data needs theta_true (pass it or set it on the model).")
        labels = self._dataset_labels()
        for comp in self._joint_components():
            _, ax = plt.subplots(figsize=(8, 6))
            for i, label in enumerate(labels):
                self.select(i)
                d = self.data
                obs = d.get(comp)
                if obs is None:
                    continue
                model = self._model_component(theta_true, d["t"], d, component=comp)
                ax.scatter(d["t"], obs, color=f"C{i}", s=25, marker="o",
                           edgecolors="black", linewidths=0.5, zorder=3)
                ax.plot(d["t"], model, color=f"C{i}", lw=2, zorder=2, label=label)
            ax.set(xlabel=r"$t$", ylabel=rf"${comp.upper()}(t)$")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", framealpha=0.95)
            plt.show()
        self.select(0)

    def plot_posterior_predictive(self, nsamples_pred=5000, logx=False, logy=False):
        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        labels = self._dataset_labels()
        for comp in self._joint_components():
            _, ax = plt.subplots(figsize=(8, 6))
            for i, label in enumerate(labels):
                self.select(i)
                summary = self._predictive_summary(comp, nsamples_pred)
                if summary is None:
                    continue
                t, obs, mean_pred, pred_lo, pred_hi = summary
                ax.fill_between(t, pred_lo, pred_hi, color=f"C{i}", alpha=0.25)
                ax.plot(t, mean_pred, color=f"C{i}", lw=1.5, zorder=4, label=label)
                ax.scatter(t, obs, color=f"C{i}", s=10, marker="o", zorder=5,
                           edgecolors="black", linewidths=0.5, alpha=0.8)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set(xlabel=r"$t$", ylabel=rf"${comp.upper()}(t)$")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", framealpha=0.9)
            plt.show()
        self.select(0)

__all__ = ["JointInferenceProcedure"]
