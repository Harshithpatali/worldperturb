import numpy as np
import pandas as pd
import pytest

from worldperturb import (
    ConstraintViolation, Distribution, SimulationEngine, SimulationInvariant,
    StateTransition, World,
)
from worldperturb.benchmarking import build_benchmark_report, save_benchmark_report
from worldperturb.datasets import make_retail_benchmark
from worldperturb.statistics import (
    bootstrap_mean, bootstrap_paired_difference, statistical_diagnostics,
)


def test_transition_arbitration_allows_only_one_transition_per_entity_per_step():
    entities = pd.DataFrame({"state": ["new"]})
    engine = SimulationEngine(state_column="state", transitions=[
        StateTransition("new", "active", 1.0),
        StateTransition("active", "churned", 1.0),
    ])
    result = engine.run(entities, steps=1, seed=1)
    assert result.final_state.loc[0, "state"] == "active"
    assert result.transition_counts.loc[0, "transition_0"] == 1
    assert result.transition_counts.loc[0, "transition_1"] == 0


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
    a, b = engine.run(entities, steps=1, seed=42), engine.run(entities, steps=1, seed=42)
    assert a.final_state.equals(b.final_state)


def test_simulation_invariant_validates_initial_and_each_step():
    entities = pd.DataFrame({"state": ["active"], "balance": [10.0]})
    invariant = SimulationInvariant("nonnegative_balance", lambda frame, step: frame.balance >= 0)
    engine = SimulationEngine(state_column="state", transitions=[
        StateTransition("active", "active", updates={"balance": -1.0})
    ], invariants=[invariant])
    with pytest.raises(ConstraintViolation, match="step 1"):
        engine.run(entities, steps=1)


def test_distributions_reproduce_seeded_draws():
    dist = Distribution.normal(2, 0.5)
    assert np.array_equal(dist.sample(10, np.random.default_rng(9)),
                          dist.sample(10, np.random.default_rng(9)))


def test_monte_carlo_runs_are_reproducible_and_paired():
    world = World(pd.DataFrame({"x": [1.0, 2.0]}))
    scenario = world.scenario("noise").sample("x", Distribution.normal())
    a = world.monte_carlo(scenario, repetitions=5, seed=11)
    b = world.monte_carlo(scenario, repetitions=5, seed=11)
    assert a.samples.equals(b.samples)
    paired = world.paired_monte_carlo(scenarios=[scenario], metric="x", repetitions=20, seed=11)
    result = paired.compare("baseline", "noise", bootstrap_samples=200)
    assert result["ci_lower"] <= result["mean_difference"] <= result["ci_upper"]


def test_bootstrap_and_diagnostics_are_reproducible():
    values = np.arange(1.0, 31.0)
    a = bootstrap_mean(values, samples=300, seed=8)
    b = bootstrap_mean(values, samples=300, seed=8)
    assert a == b
    paired = bootstrap_paired_difference(values, values + 2, samples=300, seed=8)
    assert paired.estimate == pytest.approx(2.0)
    assert statistical_diagnostics(values)["n"] == 30


def test_retail_dataset_and_report_are_reproducible(tmp_path):
    frame = make_retail_benchmark(periods=60, seed=7)
    assert frame.equals(make_retail_benchmark(periods=60, seed=7))
    actual = frame.demand.iloc[40:]
    report = build_benchmark_report(
        actual,
        {"naive": frame.demand.shift(1).iloc[40:], "perfect": actual},
        dataset_name="synthetic_daily_retail", seed=7,
    )
    assert report["best_model"] == "perfect"
    json_path, csv_path = save_benchmark_report(report, tmp_path / "retail_benchmark")
    assert json_path.exists() and csv_path.exists()
    assert json_path.read_text(encoding="utf-8").find(report["data_fingerprint"]) >= 0
