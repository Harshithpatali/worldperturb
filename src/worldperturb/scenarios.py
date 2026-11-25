"""Scenario and deterministic perturbation specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
    """A named list of ordered perturbations."""

    name: str
    perturbations: list[Perturbation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

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
