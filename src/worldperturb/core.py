"""Dataframe-backed world and experiment execution."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

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
class ExperimentResult:
    """Baseline plus scenario outputs."""

    baseline: ScenarioResult
    scenarios: dict[str, ScenarioResult]

    def compare(self, metric: str, aggregation: str = "mean") -> pd.DataFrame:
        """Compare a numeric column's aggregate; delta is scenario minus baseline."""
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
        rows = [{"scenario": name, "value": value, "delta": value - base,
                 "relative_delta": (value - base) / base if base != 0 else float("nan")}
                for name, value in values.items()]
        return pd.DataFrame(rows).set_index("scenario")


class World:
    """A dataframe-backed world with declared metadata and row-wise constraints.

    v0.1 provides deterministic transformations only. It does not infer causal
    effects or implement hybrid simulation engines.
    """

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
        """Return a defensive copy of baseline data."""
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

    def run(self, scenario: Scenario | None = None) -> ScenarioResult:
        """Run the baseline or one scenario on a fresh copy of the data."""
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
        self._validate(output)
        return ScenarioResult(name, output, self._fingerprint(output),
                              {"rows": len(output), "columns": len(output.columns)})

    def experiment(self, *, scenarios: list[Scenario], repetitions: int = 1, seed: int = 0) -> ExperimentResult:
        """Run a deterministic baseline/scenario comparison.

        repetitions and seed are accepted for API planning; deterministic v0.1
        operators do not consume random numbers or repeat simulations.
        """
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        names = [s.name for s in scenarios]
        if len(names) != len(set(names)) or "baseline" in names:
            raise ValueError("Scenario names must be unique and cannot be 'baseline'.")
        return ExperimentResult(self.run(), {s.name: self.run(s) for s in scenarios})
