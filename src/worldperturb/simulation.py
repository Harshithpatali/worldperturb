"""Discrete-time entity simulation with transition arbitration and invariants."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import pandas as pd
from .constraints import ConstraintViolation


@dataclass(frozen=True)
class StateTransition:
    """A probabilistic transition between values in an entity state column."""
    source: Any
    target: Any
    probability: float = 1.0
    condition: Callable[[pd.DataFrame, int], pd.Series] | None = None
    updates: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Transition probability must be between 0 and 1.")


@dataclass(frozen=True)
class SimulationInvariant:
    """A named invariant; validator receives a state snapshot and timestep."""
    name: str
    validator: Callable[[pd.DataFrame, int], bool | pd.Series]
    description: str = ""


@dataclass
class SimulationResult:
    """Full entity trajectory and final state for a simulation run."""
    trajectory: pd.DataFrame
    final_state: pd.DataFrame
    transition_counts: pd.DataFrame
    seed: int


class SimulationEngine:
    """Run discrete-time transitions; first matching transition wins per entity."""
    def __init__(self, *, state_column: str, transitions: list[StateTransition],
                 invariants: list[SimulationInvariant] | None = None) -> None:
        if not state_column:
            raise ValueError("state_column must be non-empty.")
        self.state_column = state_column
        self.transitions = list(transitions)
        self.invariants = list(invariants or [])
        names = [item.name for item in self.invariants]
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError("Invariant names must be non-empty and unique.")

    def _validate(self, state: pd.DataFrame, step: int) -> None:
        for invariant in self.invariants:
            try:
                outcome = invariant.validator(state.copy(deep=True), step)
            except Exception as exc:
                raise ConstraintViolation(
                    f"Invariant {invariant.name!r} could not be evaluated at step {step}: {exc}"
                ) from exc
            if isinstance(outcome, pd.Series):
                passed = bool(outcome.fillna(False).astype(bool).all())
            elif isinstance(outcome, (bool, np.bool_)):
                passed = bool(outcome)
            else:
                raise ConstraintViolation(
                    f"Invariant {invariant.name!r} must return bool or pandas Series."
                )
            if not passed:
                raise ConstraintViolation(
                    f"Invariant {invariant.name!r} failed at step {step}: {invariant.description}"
                )

    def run(self, entities: pd.DataFrame, *, steps: int, seed: int = 0) -> SimulationResult:
        if not isinstance(entities, pd.DataFrame) or entities.empty:
            raise ValueError("entities must be a non-empty DataFrame.")
        if self.state_column not in entities:
            raise KeyError(f"Missing state column {self.state_column!r}.")
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        state = entities.copy(deep=True).reset_index(drop=True)
        rng = np.random.default_rng(seed)
        snapshots, counts = [], []
        self._validate(state, 0)
        for step in range(steps + 1):
            snapshot = state.copy(deep=True)
            snapshot.insert(0, "step", step)
            snapshots.append(snapshot)
            if step == steps:
                break
            start = state.copy(deep=True)
            claimed = pd.Series(False, index=state.index)
            next_state = state.copy(deep=True)
            step_counts = {"step": step}
            for index, transition in enumerate(self.transitions):
                eligible = start[self.state_column].eq(transition.source) & ~claimed
                if transition.condition is not None:
                    condition = transition.condition(start.copy(deep=True), step)
                    if not isinstance(condition, pd.Series) or len(condition) != len(start):
                        raise ValueError("Transition condition must return an aligned pandas Series.")
                    eligible &= condition.reindex(start.index).fillna(False).astype(bool)
                selected = eligible & (rng.random(len(start)) < transition.probability)
                for column, value in transition.updates.items():
                    if column not in next_state:
                        raise KeyError(f"Transition update references missing column {column!r}.")
                    update = value(start.copy(deep=True), step) if callable(value) else value
                    next_state.loc[selected, column] = update if np.isscalar(update) else update.reindex(start.index).loc[selected]
                next_state.loc[selected, self.state_column] = transition.target
                claimed |= selected
                step_counts[f"transition_{index}"] = int(selected.sum())
            state = next_state
            self._validate(state, step + 1)
            counts.append(step_counts)
        return SimulationResult(pd.concat(snapshots, ignore_index=True), state,
                                pd.DataFrame(counts).fillna(0), seed)
