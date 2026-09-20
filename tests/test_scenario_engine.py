"""Scenario engine.

Every figure here is a small arithmetic expression, so the tests compute the
expected LCY P&L by hand from the position and the shock.
"""
import pytest

from regime_risk import DEFAULT_SCENARIOS, Position, ScenarioEngine, ScenarioShock
from tests.conftest import position

BASE_FX = 110.0
NOTIONAL = 1_000_000.0


def engine():
    return ScenarioEngine(base_fx_rate=BASE_FX)


def test_no_shock_produces_no_pnl(one_position):
    r = engine().run(one_position, ScenarioShock("flat", "nothing happens"))
    assert r.total_pnl_lcy == 0.0
    assert all(v == 0.0 for v in r.attribution.values())


def test_fx_shift_costs_notional_times_the_shift(one_position):
    """A weaker local currency makes the same USD invoice cost more LCY.
    1,000,000 USD x 5.0 LCY/USD = 5,000,000 LCY of extra cost."""
    r = engine().run(one_position, ScenarioShock("fx", "", fx_shift=5.0))
    assert r.attribution["fx"] == pytest.approx(-5_000_000.0)


def test_fx_strengthening_is_a_gain(one_position):
    r = engine().run(one_position, ScenarioShock("fx", "", fx_shift=-5.0))
    assert r.attribution["fx"] == pytest.approx(5_000_000.0)


def test_commodity_shock_is_converted_at_the_shocked_fx_rate(one_position):
    """10% on 1m USD is 100k USD, converted at the post-shock rate of 115:
    -100,000 x 115 = -11,500,000 LCY."""
    r = engine().run(one_position, ScenarioShock(
        "c", "", fx_shift=5.0, commodity_shocks={"copper": 0.10}))
    assert r.attribution["commodity"] == pytest.approx(-11_500_000.0)


def test_total_pnl_equals_the_exact_revaluation_including_the_cross_term():
    """Worth checking carefully, because this is the part that is easy to get
    wrong and this implementation gets it right.

    True LCY cost change = N*(1+p)*(F+s) - N*F
                         = N*(s + p*F + p*s)
    The engine books N*s under fx and N*p*(F+s) under commodity, which sums to
    exactly that, so the FX-commodity cross term is neither dropped nor
    double-counted.
    """
    p, s = 0.10, 5.0
    r = engine().run([position()], ScenarioShock(
        "x", "", fx_shift=s, commodity_shocks={"copper": p}))
    exact = -(NOTIONAL * (1 + p) * (BASE_FX + s) - NOTIONAL * BASE_FX)
    assert r.total_pnl_lcy == pytest.approx(exact)


def test_lc_fee_shift_is_charged_on_notional_at_the_shocked_rate(one_position):
    """150bps on 1m USD = 15,000 USD, at 110 = 1,650,000 LCY."""
    r = engine().run(one_position, ScenarioShock("lc", "", lc_fee_shift=0.015))
    assert r.attribution["lc_fees"] == pytest.approx(-1_650_000.0)


def test_tariff_shift_is_charged_on_notional_at_the_shocked_rate(one_position):
    r = engine().run(one_position, ScenarioShock("t", "", tariff_shift=0.10))
    assert r.attribution["tariffs"] == pytest.approx(-11_000_000.0)


def test_only_the_named_commodity_is_shocked():
    book = [position("P1", commodity="copper"), position("P2", commodity="PVC")]
    r = engine().run(book, ScenarioShock("c", "", commodity_shocks={"copper": 0.20}))
    df = r.position_pnl.set_index("position_id")
    assert df.loc["P1", "commodity_pnl_lcy"] != 0.0
    assert df.loc["P2", "commodity_pnl_lcy"] == 0.0


def test_attribution_sums_to_the_total(one_position):
    r = engine().run(one_position, DEFAULT_SCENARIOS[4])
    assert sum(r.attribution.values()) == pytest.approx(r.total_pnl_lcy, abs=0.05)


def test_pnl_scales_linearly_with_notional():
    small = engine().run([position(notional=1_000_000.0)], DEFAULT_SCENARIOS[1])
    big = engine().run([position(notional=3_000_000.0)], DEFAULT_SCENARIOS[1])
    assert big.total_pnl_lcy == pytest.approx(3 * small.total_pnl_lcy, rel=1e-6)


def test_book_pnl_is_the_sum_of_position_pnl():
    book = [position(f"P{i}", notional=500_000.0 * (i + 1)) for i in range(4)]
    r = engine().run(book, DEFAULT_SCENARIOS[1])
    assert r.total_pnl_lcy == pytest.approx(r.position_pnl["total_pnl_lcy"].sum(), abs=0.05)


def test_one_row_per_position():
    book = [position(f"P{i}") for i in range(5)]
    assert len(engine().run(book, DEFAULT_SCENARIOS[0]).position_pnl) == 5


def test_every_default_scenario_is_a_loss_for_an_importer(one_position):
    """All five shocks raise import costs, so an importer should lose on each."""
    for sc in DEFAULT_SCENARIOS:
        assert engine().run(one_position, sc).total_pnl_lcy < 0, sc.name


def test_severe_stress_is_the_worst_default_scenario(one_position):
    df = engine().run_all(one_position)
    assert df.iloc[0]["scenario"] == "Severe_Stress"


def test_run_all_is_sorted_worst_first(one_position):
    col = engine().run_all(one_position)["total_pnl_lcy"].tolist()
    assert col == sorted(col)


def test_run_all_covers_every_default_scenario(one_position):
    assert len(engine().run_all(one_position)) == len(DEFAULT_SCENARIOS)


def test_run_all_accepts_custom_scenarios(one_position):
    df = engine().run_all(one_position, [ScenarioShock("only", "", fx_shift=1.0)])
    assert list(df["scenario"]) == ["only"]


# ------------------------------------------------------------------- defects

def test_per_position_settlement_fx_is_used():
    """Regression #13: each settlement rate determines the commodity leg."""
    cheap = engine().run([position("A", settlement_fx=80.0)],
                         ScenarioShock("s", "", fx_shift=5.0, commodity_shocks={"copper": 0.1}))
    dear = engine().run([position("B", settlement_fx=200.0)],
                        ScenarioShock("s", "", fx_shift=5.0, commodity_shocks={"copper": 0.1}))
    assert cheap.total_pnl_lcy == pytest.approx(-5_000_000 - 100_000 * 85)
    assert dear.total_pnl_lcy == pytest.approx(-5_000_000 - 100_000 * 205)


def test_tariff_and_fee_reductions_stop_at_zero():
    """Regression #14: a reduction can eliminate charges, not create subsidies."""
    pos = position()
    assert pos.tariff_rate == 0.05
    r = engine().run([pos], ScenarioShock("t", "", tariff_shift=-0.20, lc_fee_shift=-0.2))
    assert r.attribution["tariffs"] == pytest.approx(5_500_000.0)
    assert r.attribution["lc_fees"] == pytest.approx(1_100_000.0)


def test_inconsistent_notional_is_rejected():
    """Regression #15: physical quantity and invoice notional must reconcile."""
    inconsistent = Position(
        position_id="BAD", commodity="copper", notional_usd=9_000_000.0,
        quantity=1_000.0, unit="tonne", entry_price_usd=1_000.0,
        lc_fee_pct=0.01, tariff_rate=0.05, settlement_fx=110.0,
    )
    with pytest.raises(ValueError, match="Notional must equal"):
        engine().run([inconsistent], ScenarioShock("fx", "", fx_shift=1.0))


def test_scenario_descriptions_use_absolute_fx_units():
    """Regression #16: descriptions match parameter units at any baseline."""
    political = DEFAULT_SCENARIOS[1]
    assert "LCY/USD +8.5" in political.description and political.fx_shift == 8.5
    assert "LCY/USD +15" in DEFAULT_SCENARIOS[4].description


def test_tariff_circular_description_uses_percentage_points():
    """Regression #16: additive rates are percentage-point changes."""
    tariff = DEFAULT_SCENARIOS[3]
    assert "10 percentage points" in tariff.description
    assert tariff.tariff_shift == 0.10
    assert position().tariff_rate == 0.05


def test_unset_settlement_fx_uses_engine_baseline():
    r = ScenarioEngine(80).run([position(settlement_fx=None)],
                              ScenarioShock("c", "", commodity_shocks={"copper": 0.1}))
    assert r.attribution["commodity"] == -8_000_000


def test_per_unit_output_reconciles_with_total(one_position):
    r = engine().run(one_position, ScenarioShock("fx", "", fx_shift=1))
    assert r.position_pnl.iloc[0]["pnl_per_unit_lcy"] == -1000
    assert r.position_pnl.iloc[0]["unit"] == "tonne"


@pytest.mark.parametrize("shift", [-110, -111, float("nan")])
def test_invalid_shocked_fx_is_rejected(one_position, shift):
    with pytest.raises(ValueError):
        engine().run(one_position, ScenarioShock("invalid", "", fx_shift=shift))


def test_empty_book_has_zero_impact():
    r = engine().run([], ScenarioShock("empty", ""))
    assert r.total_pnl_lcy == 0
    assert r.position_pnl.empty
