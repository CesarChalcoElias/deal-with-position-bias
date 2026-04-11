import numpy as np
import pytest
from matplotlib.figure import Figure

from unbiastap.models.custom.shrinkage_selector import ShrinkageSelector
from unbiastap.models.custom.config import ShrinkageSelectorConfig


def make_selector(**kwargs) -> ShrinkageSelector:
    return ShrinkageSelector(ShrinkageSelectorConfig(**kwargs))


def make_position_data(n_positions: int = 38):
    positions = np.arange(1, n_positions + 1)
    empirical = np.clip(0.999 * (0.95 ** (positions - 1)), 0.01, 1.0)
    session_counts = np.maximum(400_000 * (0.85 ** (positions - 1)), 100.0)
    parametric = 0.9 ** (positions - 1)
    return empirical, session_counts, parametric


def test_shrinkage_weight_at_zero_n0_returns_ones():
    "At n0=0 the empirical estimate gets full trust — lambda equals 1 everywhere."
    n_k = np.array([100.0, 1_000.0, 10_000.0])
    weights = ShrinkageSelector._shrinkage_weight(n_k, n0=0.0)
    np.testing.assert_array_equal(weights, 1.0)


def test_blend_at_lambda_one_returns_empirical():
    "At lambda=1 the blend collapses to the empirical propensity."
    empirical = np.array([0.9, 0.7, 0.4])
    parametric = np.array([1.0, 0.81, 0.49])
    result = ShrinkageSelector._blend_propensity(empirical, parametric, np.ones(3))
    np.testing.assert_array_equal(result, empirical)


def test_blend_at_lambda_zero_returns_parametric():
    "At lambda=0 the blend collapses to the parametric propensity."
    empirical = np.array([0.9, 0.7, 0.4])
    parametric = np.array([1.0, 0.81, 0.49])
    result = ShrinkageSelector._blend_propensity(empirical, parametric, np.zeros(3))
    np.testing.assert_array_equal(result, parametric)


def test_fit_n0_star_within_grid_bounds():
    "After fit, n0_star_ must lie within the configured search range."
    empirical, session_counts, parametric = make_position_data()
    selector = make_selector(n0_min=10.0, n0_max=100_000.0)
    selector.fit(empirical, session_counts, parametric)
    assert selector.config.n0_min <= selector.n0_star_ <= selector.config.n0_max


def test_blend_raises_before_fit():
    "Calling blend() before fit() should raise RuntimeError, not AttributeError."
    empirical, session_counts, parametric = make_position_data()
    selector = make_selector()
    with pytest.raises(RuntimeError, match="fit()"):
        selector.blend(empirical, session_counts, parametric)


def test_plot_loss_curve_returns_figure_after_fit():
    "After fit(), plot_loss_curve() returns a matplotlib Figure without errors."
    empirical, session_counts, parametric = make_position_data()
    selector = make_selector()
    selector.fit(empirical, session_counts, parametric)
    fig = selector.plot_loss_curve()
    assert isinstance(fig, Figure)
