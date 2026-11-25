# WorldPerturb

**Constraint-aware simulation, stochastic scenario engineering and reproducible experiments.**

WorldPerturb is a Python framework for modelling tabular systems, simulating entity state transitions, applying deterministic or stochastic interventions, validating invariants and benchmarking outcomes.

## Installation

```bash
pip install worldperturb
```

## Entity simulation with invariants

```python
import pandas as pd
from worldperturb import SimulationEngine, SimulationInvariant, StateTransition

entities = pd.DataFrame({"customer_id": [1, 2, 3], "state": ["new"] * 3, "balance": [10., 20., 30.]})
engine = SimulationEngine(
    state_column="state",
    transitions=[StateTransition("new", "active", probability=0.7)],
    invariants=[SimulationInvariant("nonnegative_balance", lambda frame, step: frame.balance >= 0)],
)
result = engine.run(entities, steps=12, seed=42)
print(result.trajectory)
```

Transition rules use a start-of-step snapshot. The first successful matching rule claims an entity for that time step, preventing cascading transitions in one tick.

## Paired Monte Carlo experiments

```python
import pandas as pd
from worldperturb import Distribution, World

world = World(pd.DataFrame({"price": [500., 450.], "demand": [100., 120.]}))
promotion = world.scenario("promotion").sample("demand", Distribution.normal(5, 2))
price_cut = world.scenario("price_cut").scale("price", 0.95)
paired = world.paired_monte_carlo(
    scenarios=[promotion, price_cut], metric="demand", repetitions=500, seed=42,
)
print(paired.compare("baseline", "promotion"))
```

Each replication uses a common seed across scenarios, and paired bootstrap intervals are computed from replication-level differences.

## Reproducible benchmark dataset and report

```python
from worldperturb.datasets import make_retail_benchmark
from worldperturb.benchmarking import build_benchmark_report, save_benchmark_report

data = make_retail_benchmark(periods=180, seed=42)
actual = data.demand.iloc[30:]
predictions = {"naive": data.demand.shift(7).iloc[30:], "perfect_reference": actual}
report = build_benchmark_report(
    actual, predictions, dataset_name="synthetic_daily_retail", seed=42,
)
save_benchmark_report(report, "reports/retail_benchmark")
```

The report writes deterministic JSON metadata and a companion CSV with MAE, RMSE and MAPE rankings. The built-in synthetic retail dataset has a fixed generator, explicit seed and dataset version.

## Capabilities

- Discrete-time entity simulation with ordered probabilistic state transitions and per-transition updates.
- First-successful-rule transition arbitration: at most one state transition per entity per time step.
- Simulation invariants evaluated at initialization and after every transition step.
- Normal, uniform, Bernoulli and Poisson stochastic perturbations; multivariate correlated shocks.
- Reproducible Monte Carlo replications and paired scenario comparisons using common random seeds.
- Percentile bootstrap confidence intervals for means and paired differences.
- Statistical diagnostics: descriptive moments, quantiles, skewness, kurtosis and Shapiro-Wilk normality test.
- One-factor-at-a-time sensitivity analysis and forecast benchmarking with MAE, RMSE and MAPE.
- Built-in synthetic retail benchmark data and JSON/CSV benchmark reports.

## Interpretation

Randomness is controlled by NumPy generators and explicit seeds. Paired comparisons reduce noise from shared random variation but do not prove causal effects. Results depend on the specified transition rules, perturbations and data-generating assumptions; validate these assumptions before interpreting findings.

## Roadmap

- [x] Dataframe-backed deterministic scenarios and constraints
- [x] Entity state-transition simulation and invariant checks
- [x] Stochastic and correlated perturbations
- [x] Monte Carlo, paired comparisons and bootstrap uncertainty
- [x] Sensitivity analysis and reproducible benchmark reports
- [ ] Simulation adapters (Mesa / SimPy)
- [ ] Hybrid discrete-event and system-dynamics orchestration
- [ ] Advanced scenario dependency graphs and units-aware constraints

## License

MIT
