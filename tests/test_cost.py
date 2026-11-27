import pytest

from rul_pm.evaluation.cost import asymmetric_cost, tail_share


def test_late_costs_three_times_early_at_default_ratio():
    assert asymmetric_cost([50], [55], c_late=3.0) == pytest.approx(15.0)  # late, d = 5
    assert asymmetric_cost([50], [45], c_late=3.0) == pytest.approx(5.0)   # early, d = -5
    assert asymmetric_cost([50], [50], c_late=3.0) == 0.0


def test_cost_is_mean_over_cuts():
    assert asymmetric_cost([50, 50], [55, 45], c_late=3.0) == pytest.approx(10.0)


def test_tail_share_counts_only_late_errors_above_threshold():
    # d = 25 (big late), d = 15 (late, not big), d = -30 (early), d = 0
    assert tail_share([50, 50, 50, 50], [75, 65, 20, 50]) == pytest.approx(0.25)