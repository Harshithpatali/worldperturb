# WorldPerturb

**Constraint-aware simulation, stochastic scenario engineering and reproducible experiments.**

WorldPerturb is a Python framework for modelling tabular systems, simulating entity state transitions, applying deterministic or stochastic interventions, validating invariants and benchmarking outcomes.

## Installation

```bash
pip install worldperturb
```

## Scenario experiments

```python
import pandas as pd
from worldperturb import Constraint, Distribution, World

world = World(pd.DataFrame({"price": [500.0, 450.0], "demand": [100.0, 120.0]}),
              constraints=[Constraint("nonnegative_demand", "demand >= 0")])
scenario = world.scenario("uncertain_price")
scenario.sample("price", Distribution.normal(mean=10, sd=3))
result = world.monte_carlo(scenario, repetitions=500, seed=42)
print(result.summary("price"))
```

## Entity simulation

```python
import pandas as pd
from worldperturb import SimulationEngine, StateTransition

entities = pd.DataFrame({"customer_id": [1, 2, 3], "state": ["new"] * 3})
engine = SimulationEngine(state_column="state", transitions=[
    StateTransition("new", "active", probability=0.7),
    StateTransition("active", "churned", probability=0.05),
])
result = engine.run(entities, steps=12, seed=42)
print(result.trajectory)
```

## Capabilities

- Discrete-time entity simulation with ordered probabilistic state transitions and per-transition updates.
- Normal, uniform, Bernoulli and Poisson stochastic perturbations.
- Multivariate normal shocks with validated covariance matrices.
- Reproducible Monte Carlo replications and percentile summaries.
- Scenario constraints and invariant validation.
- One-factor-at-a-time sensitivity analysis.
- Forecast baseline benchmarking with MAE, RMSE and MAPE.
- Deterministic dataframe transformations and stable data fingerprints.

## Reproducibility and interpretation

Randomness is controlled by NumPy generators and explicit seeds. A Monte Carlo run derives a distinct deterministic seed per replication. Results quantify outcomes under supplied transition rules and perturbations; they do not establish causal effects by themselves. Users must encode domain mechanisms and validate assumptions.

## Roadmap

- [x] Dataframe-backed deterministic scenarios and constraints
- [x] Entity state-transition simulation
- [x] Stochastic and correlated perturbations
- [x] Monte Carlo and sensitivity analysis
- [x] Forecast baseline metrics
- [ ] Simulation adapters (Mesa / SimPy)
- [ ] Hybrid discrete-event and system-dynamics orchestration
- [ ] Advanced scenario dependency graphs and units-aware constraints

## License

MIT
