"""Distribution helpers and samplers for course demonstrations."""

import numpy as np
from scipy import stats


def sample_means(dist, n_samples: int = 1000, sample_size: int = 30, seed: int = 42):
    """Draw `n_samples` samples of size `sample_size` from `dist` and return their means.

    Useful for demonstrating the Central Limit Theorem.

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        e.g. stats.expon(scale=2) or stats.uniform(0, 1)
    n_samples : int
        Number of sample means to compute.
    sample_size : int
        Size of each individual sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    samples = dist.rvs(size=(n_samples, sample_size), random_state=rng)
    return samples.mean(axis=1)


def pdf_grid(dist, n_points: int = 300, margin: float = 0.01):
    """Return (x, y) arrays for plotting the PDF of a frozen scipy distribution.

    Parameters
    ----------
    dist : scipy.stats frozen distribution
    n_points : int
    margin : float
        Quantile margin to avoid infinite tails (0 < margin < 0.5).

    Returns
    -------
    (x, y) tuple of np.ndarrays
    """
    lo = dist.ppf(margin)
    hi = dist.ppf(1 - margin)
    x = np.linspace(lo, hi, n_points)
    y = dist.pdf(x)
    return x, y
