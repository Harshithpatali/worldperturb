# WorldPerturb

**Constraint-aware scenario engineering and reproducible what-if experiments.**

WorldPerturb is an early-stage Python framework for defining, validating, executing and comparing controlled changes to tabular systems.

> **Status:** v0.1.0 foundation. The current release provides deterministic dataframe perturbations and row-wise constraints. Hybrid simulation adapters, causal mechanisms, stochastic replications and scenario compilation are planned; they are not yet implemented.

## Installation

```bash
pip install worldperturb
```

## Quick start

```python
import pandas as pd
from worldperturb import Constraint, World

df = pd.DataFrame({"price": [500.0, 450.0], "demand": [100.0, 120.0]})
world = World.from_dataframe(df, constraints=[Constraint("positive_demand", "demand >= 0")])
scenario = world.scenario("price_increase").scale("price", 1.10)
result = world.experiment(scenarios=[scenario])
print(result.compare("price"))
```

## v0.1 capabilities

- Defensive copying of baseline data.
- Deterministic scale, add, set and clip operators.
- Ordered named scenarios.
- Row-wise dataframe constraint validation.
- Reproducible dataframe fingerprints.
- Baseline-versus-scenario aggregate comparison.

## Limitations

- Constraints use pandas expressions evaluated row-wise; this is not a general symbolic solver.
- The current engine is dataframe-backed and deterministic.
- `repetitions` and `seed` are reserved API parameters; deterministic operators do not use randomness yet.
- The package does not infer causal effects. A real-world behavioural mechanism must be specified by the user.
- Hybrid ABM, system-dynamics and discrete-event orchestration are planned, not implemented.

## Roadmap

- [x] Initial dataframe-backed scenario API
- [x] Deterministic perturbations and constraints
- [x] Baseline comparison
- [ ] Scenario compiler and dependency graph
- [ ] Units-aware constraints
- [ ] Stochastic experiments and paired random streams
- [ ] Sensitivity and uncertainty analysis
- [ ] Simulation adapter protocol
- [ ] Mesa and SimPy integrations
- [ ] Hybrid orchestration
- [ ] Scenario integrity benchmark

## License

MIT