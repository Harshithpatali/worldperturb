"""Bootstrap inference and lightweight statistical diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_samples: int


def bootstrap_mean(values: pd.Series | np.ndarray, *, confidence_level: float = 0.95,
                   samples: int = 2000, seed: int = 0) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for a sample mean."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("At least two finite observations are required.")
    if not 0 < confidence_level < 1 or samples < 100:
        raise ValueError("confidence_level must be in (0, 1) and samples must be >= 100.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    return ConfidenceInterval(float(array.mean()), float(np.quantile(means, alpha)),
                              float(np.quantile(means, 1 - alpha)), confidence_level, samples)


def bootstrap_paired_difference(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray, *,
                                confidence_level: float = 0.95, samples: int = 2000,
                                seed: int = 0) -> ConfidenceInterval:
    """Bootstrap paired mean difference (right minus left) using paired resampling."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Paired samples must be one-dimensional and have equal lengths.")
    valid = np.isfinite(a) & np.isfinite(b)
    return bootstrap_mean(b[valid] - a[valid], confidence_level=confidence_level,
                          samples=samples, seed=seed)


def statistical_diagnostics(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Return descriptive, normality and uncertainty diagnostics."""
    from scipy import stats
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("At least two finite observations are required.")
    normality = stats.shapiro(array[:5000]) if 3 <= array.size <= 5000 else None
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "std": float(array.std(ddof=1)), "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)), "q95": float(np.quantile(array, 0.95)),
        "skewness": float(stats.skew(array, bias=False)) if array.size > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(array, bias=False)) if array.size > 3 else float("nan"),
        "shapiro_statistic": float(normality.statistic) if normality else float("nan"),
        "shapiro_pvalue": float(normality.pvalue) if normality else float("nan"),
    }
