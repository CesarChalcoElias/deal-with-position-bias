import pytest

import numpy as np
from matplotlib.figure import Figure

from unbiastap.models.custom.config import ClipSelectorConfig
from unbiastap.models.custom.clip_selector import ClipSelector


def make_selector(tau: float = 0.3, n_grid_points: int = 100) -> ClipSelector:
    return ClipSelector(
        ClipSelectorConfig(tau=tau, n_grid_points=n_grid_points)
    )


def test_compute_ess_equal_weights_returns_n():
    "Equal weights have no variance, so ESS equals the sample size exactly."
    weights = np.ones(10) * 2.0
    ess = ClipSelector._compute_ess(weights, c=2.0)
    assert ess == pytest.approx(10.0)


def test_compute_ess_dominant_weight_returns_near_one():
    "One outlier weight dominates, collapsing ESS to approximately 1."
    weights = np.array([0.01, 0.01, 0.01, 10_000.0])
    ess = ClipSelector._compute_ess(weights, c=10_000.0)
    assert ess == pytest.approx(1.0, abs=0.05)


def test_fit_clip_value_is_largest_c_satisfying_tau():
    "clip_value_ must be the largest valid c where ESS/n >= tau."
    weights = np.array([1.0] * 9 + [50.0])
    selector = make_selector(tau=0.3)
    selector.fit(weights)

    n = len(weights)
    ess_at_clip = ClipSelector._compute_ess(weights, selector.clip_value_) / n
    assert ess_at_clip >= 0.3

    curve = selector.ess_curve_
    larger_c_values = curve[curve[:, 0] > selector.clip_value_, 1]
    assert not (larger_c_values >= 0.3).any()


def test_fit_raises_when_tau_is_unachievable():
    "Passing tau > 1.0 to fit() cannot be satisfied by any clip value."
    weights = np.array([1.0, 2.0, 3.0])
    selector = make_selector()
    with pytest.raises(ValueError, match="ESS/n"):
        selector.fit(weights, tau=1.01)


def test_plot_ess_curve_raises_runtime_error_before_fit():
    "Calling plot before fit() should raise RuntimeError."
    selector = make_selector()
    with pytest.raises(RuntimeError, match="fit()"):
        selector.plot_ess_curve()


def test_plot_ess_curve_returns_figure_after_fit():
    "After fit(), plot_ess_curve() returns a matplotlib Figure."
    weights = np.array([1.0] * 9 + [50.0])
    selector = make_selector()
    selector.fit(weights)
    fig = selector.plot_ess_curve()
    assert isinstance(fig, Figure)
