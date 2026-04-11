import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

from unbiastap.models.custom.config import ClipSelectorConfig


class ClipSelector:
    def __init__(self, config: ClipSelectorConfig):
        self.config = config
        self.clip_value_: float | None = None
        self.clip_percentile_: float | None = None
        self.ess_curve_: np.ndarray | None = None

    @staticmethod
    def _compute_ess(weights: np.ndarray, c: float) -> float:
        clipped = np.minimum(weights, c)
        return float(clipped.sum() ** 2 / (clipped**2).sum())

    def fit(
        self, weights: np.ndarray, tau: float | None = None
    ) -> "ClipSelector":
        if tau is None and self.config.tau is None:
            raise ValueError(
                "tau must be provided in the config "
                "or as an argument to fit()."
            )
        if tau is None:
            tau = self.config.tau
        n = len(weights)

        grid = np.unique(
            np.percentile(
                weights, np.linspace(0, 100, self.config.n_grid_points)
            )
        )
        ess_ratios = np.array(
            [self._compute_ess(weights, c) / n for c in grid]
        )
        self.ess_curve_ = np.column_stack([grid, ess_ratios])

        valid_mask = ess_ratios >= tau
        if not valid_mask.any():
            raise ValueError(
                f"No clip value achieves ESS/n >= {tau}. "
                "The weight distribution may be too skewed for this tau. "
                "Try lowering tau or inspecting propensity calibration."
            )

        self.clip_value_ = float(grid[valid_mask][-1])
        self.clip_percentile_ = round(percentileofscore(weights, self.clip_value_), 2)
        return self

    def plot_ess_curve(self, tau: float | None = None):
        if self.ess_curve_ is None:
            raise RuntimeError("Call fit() before plot_ess_curve().")

        if tau is None:
            tau = self.config.tau
        c_values = self.ess_curve_[:, 0]
        ess_ratios = self.ess_curve_[:, 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(c_values, ess_ratios, color="steelblue")
        ax.axhline(
            y=tau,
            color="orange",
            linestyle="--",
            label=f"tau = {tau}",
        )
        ax.axvline(
            x=self.clip_value_,
            color="red",
            linestyle="--",
            label=f"c* = {self.clip_value_:.4f}",
        )
        ax.set_xlabel("Clip value (c)")
        ax.set_ylabel("ESS / n")
        ax.set_title("ESS ratio vs clip threshold")
        ax.legend()
        sns.despine()
        fig.tight_layout()
        return fig
