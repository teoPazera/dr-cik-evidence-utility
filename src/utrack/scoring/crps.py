"""CRPS estimators for sample-based forecasts (plan_a.md U0.3).

`crps_pwm` is ported line-for-line from CiK's
`cik_benchmark/metrics/crps.py:crps` (Apache-2.0, ServiceNow; path and
license verified at U0.1 against commit 73f46016f8c5643bf6799ca31875eab3e8d0d075
of ServiceNow/context-is-key-forecasting) - the probability-weighted-moment
form, exact and without estimation bias, O(n log n) per variable.

`crps_energy_fair` is an independent implementation of the same quantity in
its "fair" energy-score form, written from the formula in plan_a.md U0.3
rather than from the CiK source, so that the agreement test in
tests/test_crps.py is a genuine cross-check rather than the same bug twice.

Do not use CiK's `crps_quantile` (a quantile-loss approximation) or the
common biased estimator with `1 / (2 n^2)`: its expected value depends on
sample count, which would distort comparisons between cells with different
valid-sample counts (plan_a.md U0.3).
"""

from __future__ import annotations

import numpy as np


def crps_pwm(target: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Copyright 2024 ServiceNow
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Ported from ServiceNow/context-is-key-forecasting,
    cik_benchmark/metrics/crps.py:crps.

    Compute the CRPS using the probability weighted moment form.
    See Eq ePWM from "Estimation of the Continuous Ranked Probability Score with
    Limited Information and Applications to Ensemble Weather Forecasts"
    https://link.springer.com/article/10.1007/s11004-017-9709-7

    This is a O(n log n) per variable exact implementation, without estimation bias.

    Parameters:
    -----------
    target: np.ndarray
        The target values. (variable dimensions)
    samples: np.ndarray
        The forecast values. (n_samples, variable dimensions)

    Returns:
    --------
    crps: np.ndarray
        The CRPS for each of the (variable dimensions)
    """
    assert (
        target.shape == samples.shape[1:]
    ), f"shapes mismatch between: {target.shape} and {samples.shape}"

    num_samples = samples.shape[0]
    num_dims = samples.ndim
    sorted_samples = np.sort(samples, axis=0)

    abs_diff = (
        np.abs(np.expand_dims(target, axis=0) - sorted_samples).sum(axis=0)
        / num_samples
    )

    beta0 = sorted_samples.sum(axis=0) / num_samples

    # An array from 0 to num_samples - 1, but expanded to allow broadcasting over the variable dimensions
    i_array = np.expand_dims(np.arange(num_samples), axis=tuple(range(1, num_dims)))
    beta1 = (i_array * sorted_samples).sum(axis=0) / (num_samples * (num_samples - 1))

    return abs_diff + beta0 - 2 * beta1


def crps_energy_fair(target: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Independent implementation of the unbiased ("fair") energy-score CRPS estimator:

        CRPS(x_1..n, y) = mean_i |x_i - y| - 1 / (2 n (n-1)) * sum_i sum_j |x_i - x_j|

    O(n^2) per variable; fine at U0/U1 sample counts (<=100).
    """
    n = samples.shape[0]
    if n < 2:
        raise ValueError("crps_energy_fair needs at least 2 samples per forecast")
    term1 = np.abs(samples - np.expand_dims(target, axis=0)).mean(axis=0)
    diffs = np.abs(np.expand_dims(samples, axis=1) - np.expand_dims(samples, axis=0))
    term2 = diffs.sum(axis=(0, 1)) / (2 * n * (n - 1))
    return term1 - term2


def mean_crps(target: np.ndarray, samples: np.ndarray, estimator=crps_pwm) -> float:
    """CRPS per time step, averaged over the horizon (plan_a.md U0.3: 'CRPS is computed
    per time step and averaged over the horizon'). target: (horizon,), samples: (n_samples, horizon)."""
    return float(np.mean(estimator(target, samples)))


def mae_of_median(target: np.ndarray, samples: np.ndarray) -> float:
    """MAE of the sample median against target, averaged over the horizon."""
    median = np.median(samples, axis=0)
    return float(np.mean(np.abs(median - target)))


def rmse_of_mean(target: np.ndarray, samples: np.ndarray) -> float:
    """RMSE of the sample mean against target, averaged over the horizon."""
    mean = np.mean(samples, axis=0)
    return float(np.sqrt(np.mean((mean - target) ** 2)))
