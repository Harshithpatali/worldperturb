"""Baseline benchmarking, sensitivity analysis and reproducible reports."""
from __future__ import annotations
from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path
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
    if aggregation not in {"mean", "sum", "median"}:
        raise ValueError("aggregation must be mean, sum or median.")
    baseline = world.run().data[metric]
    base_value = float(getattr(baseline, aggregation)())
    rows = []
    for value in parameter_values:
        observed = world.run(scenario_factory(value)).data[metric]
        result = float(getattr(observed, aggregation)())
        rows.append({"parameter": value, "value": result, "delta": result - base_value})
    return pd.DataFrame(rows)


def build_benchmark_report(actual: pd.Series, predictions: dict[str, pd.Series], *,
                           dataset_name: str, seed: int,
                           metadata: dict | None = None) -> dict:
    """Build a JSON-serializable benchmark report with a deterministic data fingerprint."""
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty.")
    scores = benchmark_predictions(actual, predictions)
    payload = pd.concat([actual.rename("actual")] + [
        value.rename(f"prediction:{name}") for name, value in predictions.items()
    ], axis=1)
    fingerprint = sha256(pd.util.hash_pandas_object(payload, index=True).values.tobytes()).hexdigest()
    metrics = scores.reset_index().replace({np.nan: None}).to_dict(orient="records")
    return {
        "report_version": 1, "dataset": dataset_name, "seed": int(seed),
        "data_fingerprint": fingerprint, "metadata": metadata or {},
        "ranking_metric": "rmse", "best_model": str(scores.index[0]),
        "models": metrics,
    }


def save_benchmark_report(report: dict, path: str | Path) -> tuple[Path, Path]:
    """Write stable JSON report and companion model-metrics CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    json_path = destination.with_suffix(".json")
    csv_path = destination.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    pd.DataFrame(report["models"]).to_csv(csv_path, index=False)
    return json_path, csv_path
