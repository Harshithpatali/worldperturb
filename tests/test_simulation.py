import numpy as np
import pandas as pd
import pytest

from worldperturb import (CorrelatedShock, Distribution, SimulationEngine, StateTransition,
                          StochasticPerturbation, World)
from worldperturb.benchmarking import benchmark_predictions, sensitivity_analysis


def test_state_transitions_are_seeded_and_keep_trajectory():
    entities = pd.DataFrame({"id": [1, 2, 3], "state": ["new"] * 3})
    engine = SimulationEngine(state_column="state", transitions=[StateTransition("new", "active")])
    result = engine.run(entities, steps=3, seed=7)
    assert len(result.trajectory) == 12
    assert result.final_state.state.tolist() == ["active"] * 3
    assert result.transition_counts["transition_0"].tolist() == [3, 0, 0]


def test_transition_probability_is_reproducible():
    entities = pd.DataFrame({"state": ["a"] * 100})
    engine = SimulationEngine(state_column="state", transitions=[StateTransition("a", "b", 0.4)])
    a = engine.run(entities, steps=1, seed=42)
    b = engine.run(entities, steps=1, seed=42)
    assert a.final_state.equals(b.final_state)


def test_distributions_reproduce_seeded_draws():
    dist = Distribution.normal(2, 0.5)
    a = dist.sample(10, np.random.default_rng(9))
    b = dist.sample(10, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_stochastic_perturbation_changes_column():
    frame = pd.DataFrame({"x": [0.0] * 20})
    StochasticPerturbation("x", Distribution.uniform(1, 1)).apply(frame, np.random.default_rng(1))
    assert frame.x.tolist() == [1.0] * 20


def test_correlated_shock_preserves_requested_columns():
    frame = pd.DataFrame({"x": [0.0] * 100, "y": [0.0] * 100})
    CorrelatedShock(("x", "y"), (0, 0), ((1, 0.8), (0.8, 1))).apply(frame, np.random.default_rng(2))
    assert frame.corr().loc["x", "y"] > 0.5


def test_invalid_covariance_rejected():
    with pytest.raises(ValueError):
        CorrelatedShock(("x", "y"), (0, 0), ((1, 2), (2, 1)))


def test_monte_carlo_runs_are_reproducible():
    world = World(pd.DataFrame({"x": [1.0, 2.0]}))
    scenario = world.scenario("noise").sample("x", Distribution.normal())
    a = world.monte_carlo(scenario, repetitions=5, seed=11)
    b = world.monte_carlo(scenario, repetitions=5, seed=11)
    assert a.samples.equals(b.samples)


def test_sensitivity_and_benchmark():
    world = World(pd.DataFrame({"sales": [10.0, 20.0]}))
    result = sensitivity_analysis(world, lambda p: world.scenario("scale").scale("sales", p),
                                  parameter_values=[1, 2], metric="sales")
    assert result.delta.tolist() == [0.0, 15.0]
    scores = benchmark_predictions(pd.Series([1.0, 2.0]), {"perfect": pd.Series([1.0, 2.0])})
    assert scores.loc["perfect", "rmse"] == 0
