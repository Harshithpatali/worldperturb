"""Row-wise dataframe constraint definitions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ConstraintViolation(ValueError):
    """Raised when a declared scenario invariant fails."""


@dataclass(frozen=True)
class Constraint:
    """A named boolean pandas expression evaluated for every row."""

    name: str
    expression: str
    description: str = ""

    def validate(self, frame: pd.DataFrame) -> None:
        if not self.name.strip() or not self.expression.strip():
            raise ValueError("Constraint name and expression must be non-empty.")
        try:
            valid = frame.eval(self.expression, engine="python")
        except Exception as exc:
            raise ConstraintViolation(f"Constraint {self.name!r} could not be evaluated: {exc}") from exc
        if isinstance(valid, bool):
            passed = valid
        elif isinstance(valid, pd.Series):
            passed = bool(valid.fillna(False).all())
        else:
            raise ConstraintViolation(f"Constraint {self.name!r} must produce a boolean result.")
        if not passed:
            raise ConstraintViolation(f"Constraint {self.name!r} failed: {self.expression}")
