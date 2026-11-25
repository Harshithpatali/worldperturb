# Architecture

## v0.1 execution path

1. `World` defensively stores a baseline dataframe.
2. `Scenario` stores ordered deterministic perturbations.
3. `World.run` applies perturbations to a copy.
4. Declared constraints validate the result.
5. A SHA-256 fingerprint is calculated from schema and values.
6. `ExperimentResult.compare` compares numeric aggregates against baseline.

## Planned architecture

The next layers are scenario compilation, dependency graphs, units-aware constraints, stochastic experiment scheduling, simulation adapters and provenance manifests.

## Non-goals

WorldPerturb is not a replacement for Mesa, SimPy, PySD or BPTK-Py. These are candidate optional backends.
