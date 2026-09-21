"""Тесты A/B-модуля: пропорции, средние, power analysis."""
import numpy as np
from churn.ab_testing.ab_test import means_test, proportion_test
from churn.ab_testing.power_analysis import (
    achieved_power,
    mde_proportions,
    sample_size_proportions,
)
def test_proportion_test_detects_clear_effect():
    res = proportion_test(400, 5000, 600, 5000)
    assert res.p_value < 0.05
    assert res.significant is True
    assert res.winner == "treatment"
    assert res.relative_lift > 0
def test_proportion_test_no_effect():
    res = proportion_test(500, 5000, 505, 5000)
    assert res.significant is False
    assert res.winner == "no_difference"
def test_means_test_direction():
    rng = np.random.default_rng(0)
    control = rng.normal(100, 15, 2000)
    treatment = rng.normal(110, 15, 2000)
    res = means_test(control, treatment)
    assert res.significant is True
    assert res.winner == "treatment"
    assert res.absolute_lift > 0
def test_sample_size_increases_with_smaller_mde():
    big = sample_size_proportions(0.08, mde_relative=0.50)["sample_size_per_group"]
    small = sample_size_proportions(0.08, mde_relative=0.10)["sample_size_per_group"]
    assert small > big > 0
def test_mde_and_power_consistency():
    n = sample_size_proportions(0.08, mde_relative=0.25)["sample_size_per_group"]
    power = achieved_power(0.08, 0.25, n)
    assert power >= 0.75
    mde = mde_proportions(0.08, n)
    assert mde["mde_relative"] > 0
