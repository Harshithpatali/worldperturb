import pandas as pd
import pytest

from worldperturb import Constraint, ConstraintViolation, World


@pytest.fixture
def frame():
    return pd.DataFrame({"price": [10.0, 20.0, 30.0], "demand": [100.0, 80.0, 60.0]})


def test_world_defensively_copies_data(frame):
    world = World(frame)
    frame.loc[0, "price"] = -1
    assert world.data.loc[0, "price"] == 10
    copy = world.data
    copy.loc[0, "price"] = -2
    assert world.data.loc[0, "price"] == 10


def test_scale_and_ordered_perturbations(frame):
    result = World(frame).run(World(frame).scenario("x").scale("price", 2).add("price", 5))
    assert result.data.price.tolist() == pytest.approx([25, 45, 65])


def test_constraints_validate_baseline(frame):
    world = World(frame, constraints=[Constraint("positive", "price > 0")])
    assert world.run().name == "baseline"


def test_constraint_rejects_invalid_scenario(frame):
    world = World(frame, constraints=[Constraint("positive", "price > 0")])
    with pytest.raises(ConstraintViolation):
        world.run(world.scenario("bad").set("price", -1))


def test_missing_column_fails(frame):
    with pytest.raises(KeyError):
        World(frame).run(World(frame).scenario("bad").scale("missing", 2))


def test_non_numeric_operation_fails():
    world = World(pd.DataFrame({"segment": ["a", "b"]}))
    with pytest.raises(TypeError):
        world.run(world.scenario("bad").scale("segment", 2))


def test_compare_baseline_to_scenario(frame):
    world = World(frame)
    result = world.experiment(scenarios=[world.scenario("doubled").scale("price", 2)])
    comparison = result.compare("price")
    assert comparison.loc["baseline", "value"] == pytest.approx(20)
    assert comparison.loc["doubled", "value"] == pytest.approx(40)
    assert comparison.loc["doubled", "relative_delta"] == pytest.approx(1)


def test_duplicate_names_rejected(frame):
    with pytest.raises(ValueError):
        World(frame).experiment(scenarios=[World(frame).scenario("same"), World(frame).scenario("same")])


def test_invalid_repetitions_rejected(frame):
    with pytest.raises(ValueError):
        World(frame).experiment(scenarios=[], repetitions=0)


def test_clip_operation(frame):
    result = World(frame).run(World(frame).scenario("clip").clip("price", 15, 25))
    assert result.data.price.tolist() == [15, 20, 25]


def test_null_constraint_fails():
    with pytest.raises(ConstraintViolation):
        Constraint("non_null", "x > 0").validate(pd.DataFrame({"x": [1, None]}))


def test_fingerprint_is_repeatable(frame):
    world = World(frame)
    assert world.run().fingerprint == world.run().fingerprint
