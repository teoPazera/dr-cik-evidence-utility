"""Unit tests 1-4, 7, 9 from plan_a.md U0.3."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from utrack.scoring.crps import (
    crps_energy_fair,
    crps_pwm,
    mae_of_median,
    mean_crps,
    rmse_of_mean,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CRPS_PATH = (
    REPO_ROOT / "external" / "context-is-key-forecasting" / "cik_benchmark" / "metrics" / "crps.py"
)


def _gaussian_crps_closed_form(mu: float, sigma: float, y: float) -> float:
    z = (y - mu) / sigma
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    big_phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * big_phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def _crps_energy_biased_reference(target: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """The common BIASED estimator using 1/(2n^2) instead of 1/(2n(n-1)).

    Deliberately kept local to this test file, not exported from production
    code: plan_a.md U0.3 explicitly says not to use it for real scoring.
    This function exists only to demonstrate, in
    test_fair_estimator_stable_across_n_biased_is_not, why the fair estimator
    was chosen instead (its expected value depends on n, distorting any
    comparison between cells with different valid-sample counts).
    """
    n = samples.shape[0]
    term1 = np.abs(samples - np.expand_dims(target, axis=0)).mean(axis=0)
    diffs = np.abs(np.expand_dims(samples, axis=1) - np.expand_dims(samples, axis=0))
    term2 = diffs.sum(axis=(0, 1)) / (2 * n * n)
    return term1 - term2


# --- Test 1: point-mass forecast gives CRPS equal to absolute error ---


@pytest.mark.parametrize("estimator", [crps_pwm, crps_energy_fair])
def test_point_mass_forecast_equals_absolute_error(estimator) -> None:
    target = np.array(7.5)
    samples = np.full((10,), 3.0)
    assert float(estimator(target, samples)) == pytest.approx(abs(7.5 - 3.0))


# --- Test 2: ported and independent estimator agree on random inputs ---


def test_ported_and_independent_estimator_agree_on_random_inputs() -> None:
    rng = np.random.default_rng(2)
    target = rng.normal(size=15)
    samples = rng.normal(size=(40, 15))
    a = crps_pwm(target, samples)
    b = crps_energy_fair(target, samples)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)


# --- Test 3: for Gaussian samples with large n, CRPS converges to the closed-form value ---


def test_crps_converges_to_gaussian_closed_form_for_large_n() -> None:
    rng = np.random.default_rng(1)
    mu, sigma, y = 10.0, 2.0, 11.5
    samples = rng.normal(mu, sigma, size=100_000)
    target = np.array(y)

    empirical = float(crps_pwm(target, samples))
    closed_form = _gaussian_crps_closed_form(mu, sigma, y)
    assert empirical == pytest.approx(closed_form, abs=0.02)


# --- Test 4: result is invariant to the order of samples ---


@pytest.mark.parametrize("estimator", [crps_pwm, crps_energy_fair])
def test_result_invariant_to_sample_order(estimator) -> None:
    rng = np.random.default_rng(3)
    target = rng.normal(size=8)
    samples = rng.normal(size=(25, 8))
    shuffled = samples[rng.permutation(25)]
    np.testing.assert_allclose(estimator(target, samples), estimator(target, shuffled))


# --- Test 7: fair CRPS is stable across n; the biased estimator is not (documentation test) ---


def test_fair_estimator_stable_across_n_biased_is_not() -> None:
    mu, sigma = 5.0, 1.5
    target = np.array(mu + 0.3)
    rng = np.random.default_rng(4)
    sample_counts = (10, 25, 100)
    reps = 600

    fair_means = {}
    biased_means = {}
    for n in sample_counts:
        fair_vals = []
        biased_vals = []
        for _ in range(reps):
            draws = rng.normal(mu, sigma, size=n)
            fair_vals.append(float(crps_energy_fair(target, draws)))
            biased_vals.append(float(_crps_energy_biased_reference(target, draws)))
        fair_means[n] = float(np.mean(fair_vals))
        biased_means[n] = float(np.mean(biased_vals))

    fair_spread = max(fair_means.values()) - min(fair_means.values())
    biased_spread = max(biased_means.values()) - min(biased_means.values())

    # The biased estimator trends down toward the true CRPS as n grows (bias ~ O(1/n));
    # the fair estimator's mean is already stable at every n. Its spread across n should
    # be visibly larger than the fair estimator's.
    assert biased_spread > 3 * fair_spread
    # And it should be monotonically decreasing as n grows, matching the O(1/n) bias story.
    assert biased_means[10] > biased_means[25] > biased_means[100]


# --- Test 9: if the CiK repository is available, the ported function reproduces the original ---


@pytest.mark.skipif(
    not ORIGINAL_CRPS_PATH.exists(),
    reason="external/context-is-key-forecasting not cloned (run `uv run utrack external sync`)",
)
def test_ported_crps_matches_original_cik_source() -> None:
    spec = importlib.util.spec_from_file_location("cik_original_crps", ORIGINAL_CRPS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    original_crps = module.crps

    rng = np.random.default_rng(0)
    target = rng.normal(size=20)
    samples = rng.normal(size=(30, 20))

    np.testing.assert_array_equal(original_crps(target, samples), crps_pwm(target, samples))


# --- Supporting metrics: mean_crps, mae_of_median, rmse_of_mean ---


def test_mean_crps_averages_over_horizon() -> None:
    target = np.array([1.0, 2.0])
    samples = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])  # perfect point mass at each step
    assert mean_crps(target, samples) == pytest.approx(0.0, abs=1e-9)


def test_mae_of_median_and_rmse_of_mean() -> None:
    target = np.array([0.0, 0.0])
    samples = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    # medians: [0.0, 0.0] -> MAE 0
    assert mae_of_median(target, samples) == pytest.approx(0.0)
    # means: [0.0, 2/3] -> RMSE = sqrt(mean([0, (2/3)^2]))
    expected_rmse = math.sqrt(((0.0) ** 2 + (2 / 3) ** 2) / 2)
    assert rmse_of_mean(target, samples) == pytest.approx(expected_rmse)
