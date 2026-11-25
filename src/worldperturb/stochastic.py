"""Seeded stochastic perturbations and multivariate correlated shocks."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Distribution:
    """A named random distribution with validated parameters."""
    kind: str
    params: dict[str, float]

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "normal":
            return rng.normal(self.params.get("mean", 0.0), self.params.get("sd", 1.0), size)
        if self.kind == "uniform":
            return rng.uniform(self.params.get("low", 0.0), self.params.get("high", 1.0), size)
        if self.kind == "bernoulli":
            return rng.binomial(1, self.params.get("p", 0.5), size).astype(float)
        if self.kind == "poisson":
            return rng.poisson(self.params.get("lam", 1.0), size).astype(float)
        raise ValueError(f"Unsupported distribution: {self.kind}")

    @classmethod
    def normal(cls, mean: float = 0.0, sd: float = 1.0) -> "Distribution":
        if sd < 0:
            raise ValueError("sd must be non-negative.")
        return cls("normal", {"mean": mean, "sd": sd})

    @classmethod
    def uniform(cls, low: float = 0.0, high: float = 1.0) -> "Distribution":
        if low > high:
            raise ValueError("low must not exceed high.")
        return cls("uniform", {"low": low, "high": high})

    @classmethod
    def bernoulli(cls, p: float = 0.5) -> "Distribution":
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1.")
        return cls("bernoulli", {"p": p})

    @classmethod
    def poisson(cls, lam: float = 1.0) -> "Distribution":
        if lam < 0:
            raise ValueError("lam must be non-negative.")
        return cls("poisson", {"lam": lam})


@dataclass(frozen=True)
class StochasticPerturbation:
    """Apply a sampled vector to one numeric column."""
    column: str
    distribution: Distribution
    operation: str = "add"

    def apply(self, frame: pd.DataFrame, rng: np.random.Generator) -> None:
        if self.column not in frame:
            raise KeyError(f"Missing stochastic target column {self.column!r}.")
        if not pd.api.types.is_numeric_dtype(frame[self.column]):
            raise TypeError(f"Stochastic target {self.column!r} must be numeric.")
        draws = self.distribution.sample(len(frame), rng)
        if self.operation == "add":
            frame[self.column] = frame[self.column] + draws
        elif self.operation == "multiply":
            frame[self.column] = frame[self.column] * draws
        elif self.operation == "set":
            frame[self.column] = draws
        else:
            raise ValueError("operation must be add, multiply or set.")


@dataclass(frozen=True)
class CorrelatedShock:
    """Add multivariate normal shocks with a specified covariance matrix."""
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (len(self.columns), len(self.columns)):
            raise ValueError("Covariance shape must match the number of columns.")
        if len(self.mean) != len(self.columns):
            raise ValueError("Mean length must match the number of columns.")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("Covariance matrix must be symmetric.")
        if np.linalg.eigvalsh(covariance).min() < -1e-9:
            raise ValueError("Covariance matrix must be positive semidefinite.")

    def apply(self, frame: pd.DataFrame, rng: np.random.Generator) -> None:
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise KeyError(f"Missing correlated shock columns: {sorted(missing)}")
        draws = rng.multivariate_normal(self.mean, self.covariance, size=len(frame))
        frame.loc[:, list(self.columns)] = frame.loc[:, list(self.columns)].to_numpy() + draws
