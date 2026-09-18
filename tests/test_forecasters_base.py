import numpy as np
import pytest

from utrack.forecasters.base import resolve_seasonal_period_steps, simulate_lag_m_bootstrap_paths


def test_resolve_seasonal_period_steps_int_form() -> None:
    assert resolve_seasonal_period_steps(24, "1 hour") == 24
    assert resolve_seasonal_period_steps(1, "1 hour") is None  # <2 is not a usable season
    assert resolve_seasonal_period_steps(0, "1 hour") is None


def test_resolve_seasonal_period_steps_string_form_matching_frequency_is_unusable() -> None:
    # U0.4 finding: string seasonal_period re-encodes frequency itself (ratio ~1.0)
    assert resolve_seasonal_period_steps("1h", "1 hour") is None
    assert resolve_seasonal_period_steps("D", "1 day") is None


def test_resolve_seasonal_period_steps_string_form_genuine_multistep() -> None:
    # a string alias whose duration is a genuine multiple of the frequency resolves correctly
    assert resolve_seasonal_period_steps("1D", "1 hour") == 24
    assert resolve_seasonal_period_steps("7D", "1 day") == 7


def test_resolve_seasonal_period_steps_none_and_unparseable() -> None:
    assert resolve_seasonal_period_steps(None, "1 hour") is None
    assert resolve_seasonal_period_steps("not_a_real_alias", "1 hour") is None


def test_simulate_lag_1_bootstrap_flat_when_residuals_are_zero() -> None:
    history = np.array([5.0, 5.0, 5.0, 5.0])
    rng = np.random.default_rng(0)
    paths = simulate_lag_m_bootstrap_paths(history, m=1, horizon=5, n_samples=10, rng=rng)
    assert paths.shape == (10, 5)
    np.testing.assert_allclose(paths, 5.0)


def test_simulate_lag_m_bootstrap_deterministic_residual_pool() -> None:
    # history with a constant lag-2 residual of exactly 2, so the path is fully
    # deterministic regardless of the RNG draw (only one possible value to draw).
    history = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(123)
    paths = simulate_lag_m_bootstrap_paths(history, m=2, horizon=6, n_samples=3, rng=rng)
    expected = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    for row in paths:
        np.testing.assert_allclose(row, expected)


def test_simulate_lag_m_bootstrap_raises_on_insufficient_history() -> None:
    history = np.array([1.0])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_lag_m_bootstrap_paths(history, m=2, horizon=3, n_samples=5, rng=rng)


def test_simulate_lag_m_bootstrap_different_seeds_differ() -> None:
    history = np.random.default_rng(1).normal(size=50)
    paths_a = simulate_lag_m_bootstrap_paths(history, m=1, horizon=10, n_samples=20, rng=np.random.default_rng(1))
    paths_b = simulate_lag_m_bootstrap_paths(history, m=1, horizon=10, n_samples=20, rng=np.random.default_rng(2))
    assert not np.allclose(paths_a, paths_b)
