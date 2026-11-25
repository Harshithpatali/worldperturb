"""Deterministic, built-in datasets for reproducible examples and benchmarks."""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_retail_benchmark(*, periods: int = 180, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic daily retail demand benchmark with trend and calendar effects."""
    if periods < 30:
        raise ValueError("periods must be at least 30.")
    rng = np.random.default_rng(seed)
    date = pd.date_range("2024-01-01", periods=periods, freq="D")
    day = np.arange(periods)
    weekly = 1.0 + 0.18 * np.sin(2 * np.pi * day / 7)
    annual = 1.0 + 0.10 * np.sin(2 * np.pi * day / 90)
    trend = 30 + 0.04 * day
    demand = np.maximum(0, trend * weekly * annual + rng.normal(0, 2.5, periods))
    price = 10 + 0.3 * np.sin(2 * np.pi * day / 30) + rng.normal(0, 0.08, periods)
    return pd.DataFrame({"date": date, "day": day, "price": price,
                         "promotion": (day % 28 < 4).astype(int), "demand": demand})


def benchmark_dataset_metadata() -> dict[str, str | int]:
    return {"name": "synthetic_daily_retail", "version": 1, "seed_default": 42,
            "observations_minimum": 30, "target": "demand",
            "description": "Synthetic daily demand with weekly seasonality, slow trend and noise."}
