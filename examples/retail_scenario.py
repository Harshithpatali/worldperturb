"""Small deterministic retail what-if example."""

import pandas as pd

from worldperturb import Constraint, World


def main() -> None:
    data = pd.DataFrame({
        "price": [500.0, 450.0, 600.0],
        "demand": [100.0, 120.0, 80.0],
        "unit_cost": [300.0, 280.0, 350.0],
    })
    data["revenue"] = data.price * data.demand
    data["profit"] = (data.price - data.unit_cost) * data.demand
    world = World.from_dataframe(data, constraints=[
        Constraint("nonnegative_price", "price >= 0"),
        Constraint("nonnegative_demand", "demand >= 0"),
    ])
    scenario = world.scenario("price_increase").scale("price", 1.10)
    # Demand is intentionally held constant: this is a mechanical shock,
    # not an estimate of real customer response or a causal effect.
    results = world.experiment(scenarios=[scenario])
    print(results.compare("price"))
    print(results.compare("revenue"))


if __name__ == "__main__":
    main()
