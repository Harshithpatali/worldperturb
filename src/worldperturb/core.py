"""Dataframe-backed worlds, scenario execution and Monte Carlo experiments."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any
import numpy as np
import pandas as pd
from .constraints import Constraint
from .scenarios import Scenario


@dataclass
class ScenarioResult:
    """Result of one baseline or scenario execution."""
    name: str
    data: pd.DataFrame
    fingerprint: str
    metadata: dict[str, Any]


@dataclass
class MonteCarloResult:
    """Replication-level scenario outputs and aggregate summaries."""
    samples: pd.DataFrame
    seed: int
    scenario_name: str

    def summary(self, metric: str) -> pd.DataFrame:
        if metric not in self.samples:
            raise KeyError(f"Metric {metric!r} not found.")
        values = self.samples[metric]
        return pd.DataFrame({
            "mean": [values.mean()], "std": [values.std(ddof=1)],
            "p05": [values.quantile(0.05)], "median": [values.median()],
            "p95": [values.quantile(0.95)],
        }, index=[metric])


@dataclass
class ExperimentResult:
    """Baseline plus scenario outputs."""
    baseline: ScenarioResult
    scenarios: dict[str, ScenarioResult]

    def compare(self, metric: str, aggregation: str = "mean") -> pd.DataFrame:
        allowed = {"mean", "sum", "median", "count"}
        if aggregation not in allowed:
            raise ValueError(f"aggregation must be one of {sorted(allowed)}")
        results = {"baseline": self.baseline, **self.scenarios}
        values: dict[str, float] = {}
        for name, result in results.items():
            if metric not in result.data.columns:
                raise KeyError(f"Metric {metric!r} not found in {name!r}.")
            series = result.data[metric]
            if aggregation != "count" and not pd.api.types.is_numeric_dtype(series):
                raise TypeError(f"Metric {metric!r} must be numeric for {aggregation}.")
            values[name] = float(getattr(series, aggregation)())
        base = values["baseline"]
        return pd.DataFrame([
            {"scenario": name, "value": value, "delta": value - base,
             "relative_delta": (value - base) / base if base != 0 else float("nan")}
            for name, value in values.items()
        ]).set_index("scenario")


@dataclass
class PairedMonteCarloResult:
    """Replication-aligned outcomes from scenarios using common random seeds."""
    samples: pd.DataFrame
    seed: int

    def compare(self, reference: str, candidate: str, *, confidence_level: float = 0.95,
                bootstrap_samples: int = 2000) -> dict[str, Any]:
        from .statistics import bootstrap_paired_difference
        if reference not in self.samples or candidate not in self.samples:
            raise KeyError("Both scenario names must exist in paired samples.")
        interval = bootstrap_paired_difference(
            self.samples[reference], self.samples[candidate],
            confidence_level=confidence_level, samples=bootstrap_samples, seed=self.seed,
        )
        return {
            "reference": reference, "candidate": candidate,
            "mean_difference": interval.estimate, "ci_lower": interval.lower,
            "ci_upper": interval.upper, "confidence_level": confidence_level,
            "probability_candidate_better": float(
                (self.samples[candidate] > self.samples[reference]).mean()
            ),
        }


class World:
    """A dataframe-backed world with constraints and reproducible interventions."""
    def __init__(self, data: pd.DataFrame, *, variables: dict[str, dict[str, Any]] | None = None,
                 constraints: list[Constraint] | None = None) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("data must contain at least one row.")
        if data.columns.has_duplicates:
            raise ValueError("data must not contain duplicate column names.")
        self._data = data.copy(deep=True)
        self.variables = variables or {}
        self.constraints = list(constraints or [])
        missing = set(self.variables) - set(self._data.columns)
        if missing:
            raise ValueError(f"Variable metadata references missing columns: {sorted(missing)}")
        self._validate(self._data)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, *, variables: dict[str, dict[str, Any]] | None = None,
                       constraints: list[Constraint] | None = None) -> "World":
        return cls(data, variables=variables, constraints=constraints)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy(deep=True)

    def add_constraint(self, constraint: Constraint) -> "World":
        constraint.validate(self._data)
        self.constraints.append(constraint)
        return self

    def scenario(self, name: str) -> Scenario:
        return Scenario(name)

    def _validate(self, frame: pd.DataFrame) -> None:
        for constraint in self.constraints:
            constraint.validate(frame)

    @staticmethod
    def _fingerprint(frame: pd.DataFrame) -> str:
        values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        schema = json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)],
                            separators=(",", ":")).encode()
        return sha256(schema + values).hexdigest()

    def run(self, scenario: Scenario | None = None, *, seed: int = 0) -> ScenarioResult:
        """Run baseline or scenario on a fresh copy; random operations use seed."""
        output = self._data.copy(deep=True)
        name = "baseline" if scenario is None else scenario.name
        if scenario is not None:
            for item in scenario.perturbations:
                if item.column not in output.columns:
                    raise KeyError(f"Scenario {name!r} references missing column {item.column!r}.")
                series = output[item.column]
                if item.operation == "scale":
                    if not pd.api.types.is_numeric_dtype(series):
                        raise TypeError(f"Cannot scale non-numeric column {item.column!r}.")
                    output[item.column] = series * item.value
                elif item.operation == "add":
                    if not pd.api.types.is_numeric_dtype(series):
                        raise TypeError(f"Cannot add to non-numeric column {item.column!r}.")
                    output[item.column] = series + item.value
                elif item.operation == "set":
                    output[item.column] = item.value
                elif item.operation == "clip":
                    if not pd.api.types.is_numeric_dtype(series):
                        raise TypeError(f"Cannot clip non-numeric column {item.column!r}.")
                    output[item.column] = series.clip(lower=item.lower, upper=item.upper)
            rng = np.random.default_rng(seed)
            for item in scenario.stochastic_perturbations:
                item.apply(output, rng)
        self._validate(output)
        return ScenarioResult(name, output, self._fingerprint(output),
                              {"rows": len(output), "columns": len(output.columns), "seed": seed})

    def monte_carlo(self, scenario: Scenario, *, repetitions: int = 100,
                    seed: int = 0) -> MonteCarloResult:
        """Run repeated scenario draws with deterministic independent seeds."""
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        rows = []
        for replication in range(repetitions):
            result = self.run(scenario, seed=seed + replication)
            row = result.data.select_dtypes(include="number").mean().to_dict()
            row["replication"] = replication
            rows.append(row)
        return MonteCarloResult(pd.DataFrame(rows).set_index("replication"), seed, scenario.name)

    def paired_monte_carlo(self, *, scenarios: list[Scenario], metric: str,
                           repetitions: int = 100, seed: int = 0,
                           aggregation: str = "mean") -> PairedMonteCarloResult:
        """Compare scenarios with common replication seeds (paired random numbers)."""
        allowed = {"mean", "sum", "median"}
        if aggregation not in allowed:
            raise ValueError(f"aggregation must be one of {sorted(allowed)}")
        if repetitions < 2:
            raise ValueError("repetitions must be at least 2.")
        names = ["baseline"] + [scenario.name for scenario in scenarios]
        if len(names) != len(set(names)):
            raise ValueError("Scenario names must be unique and cannot be 'baseline'.")
        rows = []
        for replication in range(repetitions):
            run_seed = seed + replication
            row = {}
            baseline = self.run(seed=run_seed).data
            if metric not in baseline:
                raise KeyError(f"Metric {metric!r} not found.")
            row["baseline"] = float(getattr(baseline[metric], aggregation)())
            for scenario in scenarios:
                frame = self.run(scenario, seed=run_seed).data
                if metric not in frame:
                    raise KeyError(f"Metric {metric!r} not found in {scenario.name!r}.")
                row[scenario.name] = float(getattr(frame[metric], aggregation)())
            row["replication"] = replication
            rows.append(row)
        return PairedMonteCarloResult(pd.DataFrame(rows).set_index("replication"), seed)

    def experiment(self, *, scenarios: list[Scenario], repetitions: int = 1,
                   seed: int = 0) -> ExperimentResult:
        """Run a baseline/scenario comparison (one draw per scenario)."""
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        names = [scenario.name for scenario in scenarios]
        if len(names) != len(set(names)) or "baseline" in names:
            raise ValueError("Scenario names must be unique and cannot be 'baseline'.")
        return ExperimentResult(self.run(), {
            scenario.name: self.run(scenario, seed=seed) for scenario in scenarios
        })
