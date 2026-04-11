import numpy as np

from unbiastap.models.custom.config import ShrinkageSelectorConfig

class ShrinkageSelector:
    """
    Selects the optimal prior strength n0 for Bayesian propensity shrinkage.
    (Documentation was added given a more complex implementation than the
    ClipSelector, and to explain the motivation and mechanics of the method.)

    Notation
    --------
    - k: position index (1 = first item shown, 2 = second, ...)
    - n_k: sessions that could reach position k (exposure opportunity count)
    - p_emp_k: empirical P(seen | k), the raw reach rate from logs
    - p_param_k: parametric P(seen | k) = dropout_rate^(k-1)
    - lambda_k: per-position trust weight in [0, 1]
    - n0: prior strength — virtual observations the parametric model is worth

    The blended propensity at each position is:

        p_blend_k = lambda_k * p_emp_k + (1 - lambda_k) * p_param_k

    where lambda_k = n_k / (n_k + n0).

    When n_k >> n0, lambda_k approaches 1 and we trust the empirical estimate.
    When n_k << n0, lambda_k approaches 0 and we fall back to the parametric model.
    The tail positions (high k) always have fewer sessions, so they get shrunk more.

    n0 is selected by cross-validation:
    - Sessions at each position are split into fold A and fold B.
    - Fold A computes the blend (training half — what we fit the propensity on).
    - Fold B provides the held-out empirical propensity (evaluation half — ground truth).
    - The loss is position-weighted MSE, giving more importance to dense positions
      where propensity accuracy actually affects training signal.
    - n0* = argmin L(n0).

    Parameters
    ----------
    config : ShrinkageSelectorConfig
        Configuration for the n0 search grid and fold simulation seed.

    Attributes
    ----------
    n0_star_ : float
        Optimal prior strength. Set after calling fit().
    loss_curve_ : np.ndarray
        Array of shape (n, 2) with columns [n0, cv_loss]. Set after calling fit().
    """

    def __init__(self, config: ShrinkageSelectorConfig):
        self.config = config
        self.n0_star_: float | None = None
        self.loss_curve_: np.ndarray | None = None

    @staticmethod
    def _shrinkage_weight(session_counts: np.ndarray, n0: float) -> np.ndarray:
        return session_counts / (session_counts + n0)

    @staticmethod
    def _blend_propensity(
        empirical: np.ndarray,
        parametric: np.ndarray,
        lambda_k: np.ndarray,
    ) -> np.ndarray:
        return lambda_k * empirical + (1 - lambda_k) * parametric

    @staticmethod
    def _weighted_mse(
        predictions: np.ndarray,
        targets: np.ndarray,
        position_weights: np.ndarray,
    ) -> float:
        return float(np.sum(position_weights * (predictions - targets) ** 2))

    def _simulate_folds(
        self,
        empirical_propensity: np.ndarray,
        session_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sessions at each position into fold A (training) and fold B (holdout).

        Uses binomial draws so that the empirical propensity computed from fold A
        differs from fold B due to sampling noise. This is exactly where the
        cross-validation gets its signal: at sparse positions the two folds disagree
        more, and a well-chosen n0 reduces that disagreement by pulling toward
        the smooth parametric prior.

        Parameters
        ----------
        empirical_propensity : np.ndarray
            Empirical P(seen | k) per position.
        session_counts : np.ndarray
            Number of sessions per position.

        Returns
        -------
        tuple of (p_emp_fold_a, n_k_fold_a, p_emp_fold_b, n_k_fold_b)
        """
        rng = np.random.default_rng(self.config.random_seed)
        session_counts_int = session_counts.astype(int)
        reach_counts = np.round(empirical_propensity * session_counts).astype(int)

        n_k_a = np.maximum(rng.binomial(session_counts_int, 0.5), 1).astype(float)
        n_k_b = np.maximum(session_counts_int - n_k_a.astype(int), 1).astype(float)

        reach_a = rng.binomial(reach_counts, 0.5)
        reach_b = np.maximum(reach_counts - reach_a, 0)

        p_emp_fold_a = reach_a / n_k_a
        p_emp_fold_b = reach_b / n_k_b

        return p_emp_fold_a, n_k_a, p_emp_fold_b, n_k_b

    def fit(
        self,
        empirical_propensity: np.ndarray,
        session_counts: np.ndarray,
        parametric_propensity: np.ndarray,
    ) -> "ShrinkageSelector":
        """
        Find the n0 that minimizes cross-validated propensity error.

        Parameters
        ----------
        empirical_propensity : np.ndarray
            P(seen | k) from logged reach counts. Shape: (n_positions,).
        session_counts : np.ndarray
            Number of sessions that could reach each position. Shape: (n_positions,).
        parametric_propensity : np.ndarray
            P(seen | k) from the parametric dropout model. Shape: (n_positions,).

        Returns
        -------
        ShrinkageSelector
            self, for chaining.
        """
        p_emp_fold_a, n_k_fold_a, p_emp_fold_b, n_k_fold_b = self._simulate_folds(
            empirical_propensity, session_counts
        )
        n0_grid = np.logspace(
            np.log10(self.config.n0_min),
            np.log10(self.config.n0_max),
            self.config.n_grid_points,
        )
        losses = np.array([
            self._weighted_mse(
                self._blend_propensity(
                    p_emp_fold_a,
                    parametric_propensity,
                    self._shrinkage_weight(n_k_fold_a, n0),
                ),
                p_emp_fold_b,
                np.sqrt(n_k_fold_b),
            )
            for n0 in n0_grid
        ])
        self.n0_star_ = float(n0_grid[np.argmin(losses)])
        self.loss_curve_ = np.column_stack([n0_grid, losses])
        return self

    def blend(
        self,
        empirical_propensity: np.ndarray,
        session_counts: np.ndarray,
        parametric_propensity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute blended propensity using the fitted n0.

        Parameters
        ----------
        empirical_propensity : np.ndarray
            Empirical P(seen | k) per position.
        session_counts : np.ndarray
            Number of sessions per position.
        parametric_propensity : np.ndarray
            Parametric P(seen | k) per position.

        Returns
        -------
        np.ndarray
            Blended propensity for each position.
        """
        if self.n0_star_ is None:
            raise RuntimeError("Call fit() before blend().")
        lambda_k = self._shrinkage_weight(session_counts, self.n0_star_)
        return self._blend_propensity(empirical_propensity, parametric_propensity, lambda_k)

    def plot_loss_curve(self):
        """
        Plot cross-validation loss vs n0 with the selected prior strength marked.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.loss_curve_ is None:
            raise RuntimeError("Call fit() before plot_loss_curve().")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.loss_curve_[:, 0], self.loss_curve_[:, 1], color="steelblue")
        ax.axvline(
            x=self.n0_star_,
            color="red",
            linestyle="--",
            label=f"n0* = {self.n0_star_:.0f}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Prior strength (n0)")
        ax.set_ylabel("Cross-validation loss")
        ax.set_title("Shrinkage prior selection via propensity cross-validation")
        ax.legend()
        sns.despine()
        fig.tight_layout()
        return fig
