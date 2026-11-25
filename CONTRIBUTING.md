# Contributing

Thanks for your interest in WorldPerturb.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests examples
pytest
python -m build
```

## Guidelines

- Add tests for new behaviour and regressions.
- Keep heavyweight simulation frameworks optional.
- Document assumptions and limitations.
- Do not describe mechanical perturbations as causal effects.
- Preserve deterministic results where no stochastic operator is involved.
