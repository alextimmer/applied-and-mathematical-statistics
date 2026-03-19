"""Smoke tests for the amstats package."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_import():
    import amstats
    assert amstats.__version__ == "0.1.0"


def test_plotting_palette():
    from amstats.plotting import PALETTE
    assert len(PALETTE) == 7


def test_distributions_sample_means():
    from scipy import stats
    from amstats.distributions import sample_means

    means = sample_means(stats.norm(0, 1), n_samples=100, sample_size=10)
    assert means.shape == (100,)
    # sample means of standard normal should be near 0
    assert abs(means.mean()) < 0.5


def test_distributions_pdf_grid():
    from scipy import stats
    from amstats.distributions import pdf_grid

    x, y = pdf_grid(stats.norm(0, 1))
    assert len(x) == 300
    assert len(y) == 300
    assert y.max() > 0
