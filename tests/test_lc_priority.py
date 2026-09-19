"""LC priority allocator.

This module decides which shipments get USD and which get cancelled, so the
scoring arithmetic is checked against the formula in the class docstring rather
than against whatever the code currently produces.
"""
import pytest

from regime_risk import LCPriorityAllocator, LCStatus
from tests.conftest import lc


def test_perfect_scores_reach_one_hundred():
    """0.40*10*10 + 0.25*10*10 + 0.20*10*10 + 0.15*10*10 = 100, with
    days_until_expiry = 0 and the LC being the cheapest in the batch."""
    a = LCPriorityAllocator()
    # unit_cost 0 against a max of 100 gives cost efficiency 10.
    batch = [lc("GOOD", 100.0, urgency=10, strategic=10, days=0, unit_cost=0.0),
             lc("EXPENSIVE", 100.0, unit_cost=100.0)]
    plan = a.allocate(batch, 1_000.0)
    assert plan.to_dataframe().set_index("lc_id").loc["GOOD", "priority_score"] == 100.0


def test_expiry_urgency_decays_linearly_to_zero_at_thirty_days():
    a = LCPriorityAllocator()
    assert a._expiry_urgency(0) == 10.0
    assert a._expiry_urgency(15) == 5.0
    assert a._expiry_urgency(30) == 0.0
    assert a._expiry_urgency(90) == 0.0, "floored, never negative"


def test_cost_efficiency_is_ten_for_free_and_zero_for_the_dearest():
    a = LCPriorityAllocator()
    assert a._cost_efficiency(0.0, 100.0) == 10.0
    assert a._cost_efficiency(100.0, 100.0) == 0.0
    assert a._cost_efficiency(25.0, 100.0) == 7.5


def test_cost_efficiency_falls_back_when_all_costs_are_zero():
    assert LCPriorityAllocator()._cost_efficiency(0.0, 0.0) == 5.0


def test_weights_must_sum_to_one():
    with pytest.raises(AssertionError):
        LCPriorityAllocator(urgency_weight=0.9, strategic_weight=0.9)


def test_higher_priority_is_funded_first_when_capacity_binds():
    a = LCPriorityAllocator()
    batch = [lc("LOW", 1_000_000.0, urgency=1, strategic=1, days=90),
             lc("HIGH", 1_000_000.0, urgency=10, strategic=10, days=1)]
    plan = a.allocate(batch, 1_000_000.0)
    d = {x.lc_id: x.status for x in plan.decisions}
    assert d["HIGH"] is LCStatus.APPROVED
    assert d["LOW"] is not LCStatus.APPROVED


def test_decisions_come_back_in_priority_order():
    a = LCPriorityAllocator()
    batch = [lc("LOW", 100.0, urgency=1, strategic=1, days=90),
             lc("MID", 100.0, urgency=5, strategic=5, days=45),
             lc("HIGH", 100.0, urgency=10, strategic=10, days=1)]
    scores = [d.priority_score for d in a.allocate(batch, 1e9).decisions]
    assert scores == sorted(scores, reverse=True)


def test_ample_capacity_approves_everything_reasonable():
    a = LCPriorityAllocator()
    batch = [lc(f"L{i}", 100.0, urgency=8, strategic=8, days=10) for i in range(5)]
    plan = a.allocate(batch, 1e9)
    assert all(d.status is LCStatus.APPROVED for d in plan.decisions)
    assert plan.total_allocated_usd == 500.0


def test_allocated_amounts_never_exceed_capacity():
    a = LCPriorityAllocator()
    batch = [lc(f"L{i}", 400_000.0, urgency=8, strategic=8, days=5) for i in range(10)]
    plan = a.allocate(batch, 1_000_000.0)
    assert plan.total_allocated_usd <= 1_000_000.0


def test_every_lc_gets_exactly_one_decision():
    a = LCPriorityAllocator()
    batch = [lc(f"L{i}", 400_000.0) for i in range(10)]
    plan = a.allocate(batch, 1_000_000.0)
    assert len(plan.decisions) == 10
    assert len({d.lc_id for d in plan.decisions}) == 10


def test_amounts_are_conserved_across_the_three_outcomes():
    a = LCPriorityAllocator()
    batch = [lc(f"L{i}", 400_000.0, urgency=i % 10, strategic=i % 10) for i in range(12)]
    plan = a.allocate(batch, 1_000_000.0)
    total = plan.total_allocated_usd + plan.total_deferred_usd + plan.total_cancelled_usd
    assert total == pytest.approx(sum(x.amount_usd for x in batch))


def test_utilization_rate_is_allocated_over_capacity():
    a = LCPriorityAllocator()
    plan = a.allocate([lc("L1", 250_000.0, urgency=9, strategic=9, days=1)], 1_000_000.0)
    assert plan.utilization_rate == pytest.approx(0.25)


def test_empty_batch_returns_an_empty_plan():
    plan = LCPriorityAllocator().allocate([], 1_000_000.0)
    assert plan.decisions == []
    assert plan.total_allocated_usd == 0


def test_low_scoring_lc_is_cancelled_when_capacity_is_exhausted():
    a = LCPriorityAllocator(min_priority_threshold=20.0)
    batch = [lc("BIG", 1_000_000.0, urgency=10, strategic=10, days=1),
             lc("JUNK", 1_000_000.0, urgency=0, strategic=0, days=365)]
    plan = a.allocate(batch, 1_000_000.0)
    d = {x.lc_id: x for x in plan.decisions}
    assert d["JUNK"].status is LCStatus.CANCELLED
    assert d["JUNK"].priority_score < 20.0


def test_to_dataframe_has_a_row_per_decision():
    a = LCPriorityAllocator()
    df = a.allocate([lc("L1", 100.0), lc("L2", 100.0)], 1e9).to_dataframe()
    assert len(df) == 2 and "status" in df.columns


# ------------------------------------------------------------------- defects

def test_minimum_priority_threshold_is_ignored_when_capacity_is_ample():
    """DEFECT, HIGH: `min_priority_threshold` is only consulted in the `elif`
    branch, which is reached only after capacity has run out.

    An LC scoring far below the threshold is APPROVED whenever there happens to
    be room for it. The parameter is documented as "LCs below this score are
    cancelled rather than deferred", and it silently does nothing in the case
    that matters most — a well-funded cycle, where the control is supposed to
    stop money going to requests that do not justify it.

    Here an LC scoring 3.0 against a threshold of 20.0 is approved.
    See KNOWN_ISSUES.md #9.
    """
    a = LCPriorityAllocator(min_priority_threshold=20.0)
    junk = lc("JUNK", 1_000.0, urgency=0.0, strategic=0.0, days=365, unit_cost=100.0)
    plan = a.allocate([junk], 100_000_000.0)

    decision = plan.decisions[0]
    assert decision.priority_score < 20.0
    assert decision.status is LCStatus.APPROVED, (
        "approved despite scoring below the minimum threshold"
    )


def test_a_score_depends_on_which_other_lcs_are_in_the_batch():
    """DEFECT, medium: cost efficiency is normalised by the most expensive LC in
    the batch, so an LC's score changes when unrelated LCs are added or removed.

    The same request scores 85.0 in one batch and 70.0 in another without
    anything about it changing. That makes decisions non-reproducible across
    cycles and means an approval cannot be explained by reference to the LC
    alone. See KNOWN_ISSUES.md #10.
    """
    a = LCPriorityAllocator()
    target = lc("TARGET", 100.0, urgency=10, strategic=10, days=0, unit_cost=100.0)

    alone = a.allocate([target], 1e9).decisions[0].priority_score
    with_dearer = a.allocate(
        [target, lc("DEARER", 100.0, unit_cost=1000.0)], 1e9
    )
    target_score = {d.lc_id: d.priority_score for d in with_dearer.decisions}["TARGET"]

    assert alone == 85.0, "cheapest in batch -> cost efficiency 0"
    assert target_score == 98.5, "now 10x cheaper than the dearest -> efficiency 9"
    assert alone != target_score


def test_capacity_is_left_unused_because_allocation_is_strictly_greedy():
    """DEFECT, medium: allocation is first-fit down the priority ranking, so a
    large high-priority LC that does not fit is deferred and smaller LCs further
    down are then funded from what remains.

    That is a reasonable policy, but nothing reports the resulting slack, and
    `utilization_rate` is the only clue. Here 100k of a 1m cycle is left unspent
    with a deferred LC on the books that a different ordering could have partly
    served. Documented rather than called a bug, since the alternative is a
    knapsack solve. See KNOWN_ISSUES.md #11.
    """
    a = LCPriorityAllocator()
    batch = [lc("BIG", 900_000.0, urgency=10, strategic=10, days=1),
             lc("ALSO_BIG", 900_000.0, urgency=9, strategic=9, days=2)]
    plan = a.allocate(batch, 1_000_000.0)
    assert plan.total_allocated_usd == 900_000.0
    assert plan.total_deferred_usd == 900_000.0
    assert plan.utilization_rate == pytest.approx(0.9)


def test_weight_validation_uses_assert_and_vanishes_under_optimisation():
    """DEFECT, low: the weights-sum-to-one check is an `assert`, which Python
    removes entirely when run with -O.

    Input validation that disappears in an optimised interpreter is not input
    validation. With the check stripped, weights summing to 2.0 would silently
    produce scores up to 200 on a scale documented as 0-100.
    See KNOWN_ISSUES.md #12.
    """
    import subprocess
    import sys

    code = (
        "from regime_risk import LCPriorityAllocator as A;"
        "a = A(urgency_weight=1.0, strategic_weight=1.0,"
        "      expiry_weight=0.0, cost_weight=0.0);"
        "print('constructed with weights summing to 2.0')"
    )
    r = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "constructed" in r.stdout
