"""Baseline benchmarking and scenario sensitivity utilities."""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
import pandas as pd
from .scenarios import Scenario
from .core import World


def benchmark_predictions(actual: pd.Series, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """Compare named forecast baselines using MAE, RMSE and MAPE."""
    rows = []
    for name, predicted in predictions.items():
        aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
        if aligned.empty:
            raise ValueError(f"Benchmark {name!r} has no aligned observations.")
        error = aligned["predicted"] - aligned["actual"]
        nonzero = aligned["actual"].abs() > 1e-12
        mape = float((error[nonzero].abs() / aligned.loc[nonzero, "actual"].abs()).mean() * 100) if nonzero.any() else float("nan")
        rows.append({"model": name, "n": len(aligned), "mae": float(error.abs().mean()),
                     "rmse": float(np.sqrt((error ** 2).mean())), "mape": mape})
    return pd.DataFrame(rows).set_index("model").sort_values("rmse")


def sensitivity_analysis(world: World, scenario_factory: Callable[[float], Scenario], *,
                         parameter_values: list[float], metric: str,
                         aggregation: str = "mean") -> pd.DataFrame:
    """Run a one-factor-at-a-time sweep over a scenario parameter."""
    if not parameter_values:
        raise ValueError("parameter_values must not be empty.")
    baseline = world.run().data[metric]
    base_value = float(getattr(baseline, aggregation)())
    rows = []
    for value in parameter_values:
        observed = world.run(scenario_factory(value)).data[metric]
        result = float(getattr(observed, aggregation)())
        rows.append({"parameter": value, "value": result, "delta": result - base_value})
    return pd.DataFrame(rows)
