"""Discrete-time entity simulation with explicit state transitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
import pandas as pd


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


@dataclass
class SimulationResult:
    """Full entity trajectory and final state for a simulation run."""
    trajectory: pd.DataFrame
    final_state: pd.DataFrame
    transition_counts: pd.DataFrame
    seed: int


class SimulationEngine:
    """Run entities through discrete time steps using ordered state transitions."""
    def __init__(self, *, state_column: str, transitions: list[StateTransition]) -> None:
        if not state_column:
            raise ValueError("state_column must be non-empty.")
        self.state_column = state_column
        self.transitions = list(transitions)

    def run(self, entities: pd.DataFrame, *, steps: int, seed: int = 0) -> SimulationResult:
        if entities.empty:
            raise ValueError("entities must contain at least one row.")
        if self.state_column not in entities:
            raise KeyError(f"Missing state column {self.state_column!r}.")
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        state = entities.copy(deep=True).reset_index(drop=True)
        rng = np.random.default_rng(seed)
        snapshots, counts = [], []
        for step in range(steps + 1):
            snapshot = state.copy(deep=True)
            snapshot.insert(0, "step", step)
            snapshots.append(snapshot)
            if step == steps:
                break
            step_counts = {"step": step}
            for index, transition in enumerate(self.transitions):
                eligible = state[self.state_column].eq(transition.source)
                if transition.condition is not None:
                    condition = transition.condition(state.copy(deep=True), step)
                    if not isinstance(condition, pd.Series) or len(condition) != len(state):
                        raise ValueError("Transition condition must return an aligned pandas Series.")
                    eligible &= condition.fillna(False).astype(bool)
                selected = eligible & (rng.random(len(state)) < transition.probability)
                for column, value in transition.updates.items():
                    if column not in state:
                        raise KeyError(f"Transition update references missing column {column!r}.")
                    update = value(state.copy(deep=True), step) if callable(value) else value
                    state.loc[selected, column] = update if np.isscalar(update) else update.loc[selected]
                state.loc[selected, self.state_column] = transition.target
                step_counts[f"transition_{index}"] = int(selected.sum())
            counts.append(step_counts)
        return SimulationResult(pd.concat(snapshots, ignore_index=True), state,
                                pd.DataFrame(counts).fillna(0), seed)
