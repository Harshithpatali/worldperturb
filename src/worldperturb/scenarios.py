"""Scenario and deterministic/stochastic perturbation specifications."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
from .stochastic import CorrelatedShock, Distribution, StochasticPerturbation


@dataclass(frozen=True)
class Perturbation:
    """One ordered transformation of a dataframe column."""
    column: str
    operation: Literal["scale", "add", "set", "clip"]
    value: Any = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("Perturbation column must be non-empty.")
        if self.operation not in {"scale", "add", "set", "clip"}:
            raise ValueError(f"Unsupported operation: {self.operation}")
        if self.operation == "clip" and self.lower is None and self.upper is None:
            raise ValueError("clip requires lower or upper.")


@dataclass
class Scenario:
    """A named list of ordered deterministic and stochastic interventions."""
    name: str
    perturbations: list[Perturbation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    stochastic_perturbations: list[StochasticPerturbation | CorrelatedShock] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Scenario name must be non-empty.")

    def scale(self, column: str, factor: float) -> "Scenario":
        self.perturbations.append(Perturbation(column, "scale", factor))
        return self

    def add(self, column: str, amount: float) -> "Scenario":
        self.perturbations.append(Perturbation(column, "add", amount))
        return self

    def set(self, column: str, value: Any) -> "Scenario":
        self.perturbations.append(Perturbation(column, "set", value))
        return self

    def clip(self, column: str, lower: float | None = None, upper: float | None = None) -> "Scenario":
        self.perturbations.append(Perturbation(column, "clip", lower=lower, upper=upper))
        return self

    def sample(self, column: str, distribution: Distribution, operation: str = "add") -> "Scenario":
        self.stochastic_perturbations.append(StochasticPerturbation(column, distribution, operation))
        return self

    def correlated_shock(self, columns: tuple[str, ...], mean: tuple[float, ...],
                         covariance: tuple[tuple[float, ...], ...]) -> "Scenario":
        self.stochastic_perturbations.append(CorrelatedShock(columns, mean, covariance))
        return self
