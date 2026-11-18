"""
Unit tests for the sample filters in `build_universe.py`.

Synthetic CRSP rows, so these need no WRDS credentials and run in CI.
"""

import pandas as pd
import pytest

from build_universe import (
    apply_common_stock_filter,
    apply_price_filter,
    attach_taq_symbols,
    build_universe,
    filter_attrition,
)

DAY = pd.Timestamp("2005-01-03")


def make_crsp(rows):
    """Build a synthetic CRSP daily frame, filling in sensible defaults."""
    defaults = {
        "dlycaldt": DAY,
        "permno": 10001,
        "ticker": "AAA",
        "primaryexch": "N",
        "sharetype": "NS",
        "securitytype": "EQTY",
        "securitysubtype": "COM",
        "issuertype": "CORP",
        "usincflg": "Y",
        "siccd": 3570,
        "dlyprc": 50.0,
        "dlyprevprc": 50.0,
        "dlyclose": 50.0,
        "dlyopen": 50.0,
        "dlyvol": 100000.0,
        "dlynumtrd": 500,
        "dlycap": 1e9,
        "shrout": 2e7,
        "dlyret": 0.0,
        "dlydelflg": "N",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def make_link(rows):
    """Build a synthetic CRSP-TAQ link frame."""
    defaults = {"date": DAY, "permno": 10001, "sym_root": "AAA", "sym_suffix": ""}
    return pd.DataFrame([{**defaults, **row} for row in rows])


# --------------------------------------------------------------------------
# Share-type screen
# --------------------------------------------------------------------------


def test_common_stock_filter_keeps_corporations_and_reits():
    """REITs are ordinary NYSE common shares and must survive."""
    crsp = make_crsp(
        [
            {"permno": 1, "issuertype": "CORP"},
            {"permno": 2, "issuertype": "REIT"},
        ]
    )
    assert sorted(apply_common_stock_filter(crsp)["permno"]) == [1, 2]


def test_common_stock_filter_drops_adrs_funds_and_foreign():
    crsp = make_crsp(
        [
            {"permno": 1},  # ordinary common stock
            {"permno": 2, "sharetype": "AD"},  # ADR
            {"permno": 3, "securitytype": "FUND"},  # closed-end fund
            {"permno": 4, "securitysubtype": "ETF"},  # ETF
            {"permno": 5, "usincflg": "N"},  # foreign incorporated
        ]
    )
    assert apply_common_stock_filter(crsp)["permno"].tolist() == [1]


# --------------------------------------------------------------------------
# The paper's $5 screen
# --------------------------------------------------------------------------


def test_price_filter_uses_the_previous_close_not_todays():
    """The paper screens on the previous day's close."""
    crsp = make_crsp(
        [
            {"permno": 1, "dlyprevprc": 5.01, "dlyprc": 1.00},
            {"permno": 2, "dlyprevprc": 4.99, "dlyprc": 100.00},
        ]
    )
    assert apply_price_filter(crsp)["permno"].tolist() == [1]


def test_price_filter_treats_negative_prices_as_magnitudes():
    """CRSP negates a price to flag a bid-ask midpoint; the sign is not a price."""
    crsp = make_crsp([{"permno": 1, "dlyprevprc": -42.0}])
    assert apply_price_filter(crsp)["permno"].tolist() == [1]


def test_price_filter_drops_rows_with_no_previous_price():
    crsp = make_crsp([{"permno": 1, "dlyprevprc": None}])
    assert apply_price_filter(crsp).empty


def test_price_filter_boundary_is_inclusive():
    crsp = make_crsp([{"permno": 1, "dlyprevprc": 5.0}])
    assert apply_price_filter(crsp)["permno"].tolist() == [1]


# --------------------------------------------------------------------------
# Matching CRSP to TAQ
# --------------------------------------------------------------------------


def test_unique_ticker_becomes_the_taq_symbol():
    crsp = make_crsp([{"permno": 1, "ticker": "IBM"}])
    out = attach_taq_symbols(crsp, make_link([]))
    assert out["symbol"].tolist() == ["IBM"]
    assert out["match_method"].tolist() == ["ticker"]


def test_dual_class_tickers_are_dropped_not_mismatched():
    """Two permnos sharing a ticker must not be guessed at.

    The WRDS link suffix can name a preferred issue rather than a share class
    (Constellation resolves to 'STZ PRA', which is preferred stock), so an
    ambiguous ticker is left unmatched instead of pointing at the wrong
    security's prices.
    """
    crsp = make_crsp(
        [
            {"permno": 1, "ticker": "STZ"},
            {"permno": 2, "ticker": "STZ"},
        ]
    )
    link = make_link(
        [
            {"permno": 1, "sym_root": "STZ", "sym_suffix": "PRA"},
            {"permno": 2, "sym_root": "STZ", "sym_suffix": "B"},
        ]
    )
    out = attach_taq_symbols(crsp, link)
    assert (out["symbol"] == "").all()
    assert (out["match_method"] == "unresolved").all()


def test_dual_class_can_be_resolved_when_explicitly_asked_for():
    crsp = make_crsp([{"permno": 1, "ticker": "BRK"}, {"permno": 2, "ticker": "BRK"}])
    link = make_link(
        [
            {"permno": 1, "sym_root": "BRK", "sym_suffix": "A"},
            {"permno": 2, "sym_root": "BRK", "sym_suffix": "B"},
        ]
    )
    out = attach_taq_symbols(crsp, link, resolve_dual_class=True)
    assert sorted(out["symbol"]) == ["BRK A", "BRK B"]


def test_same_ticker_on_different_days_is_not_ambiguous():
    """Ambiguity is per day; a ticker reused after a delisting is fine."""
    crsp = make_crsp(
        [
            {"permno": 1, "ticker": "XYZ", "dlycaldt": DAY},
            {"permno": 2, "ticker": "XYZ", "dlycaldt": DAY + pd.Timedelta(days=1)},
        ]
    )
    out = attach_taq_symbols(crsp, make_link([]))
    assert (out["match_method"] == "ticker").all()


# --------------------------------------------------------------------------
# Attrition accounting, which is a graded deliverable
# --------------------------------------------------------------------------


def test_attrition_counts_what_each_step_removed():
    a = make_crsp([{"permno": i} for i in range(1, 11)])
    b = a.head(7)
    c = b.head(4)
    att = filter_attrition([("all", a), ("step b", b), ("step c", c)])

    assert att["stock_days"].tolist() == [10, 7, 4]
    assert att["dropped"].tolist() == [0, 3, 3]
    assert att.loc[1, "pct_dropped"] == pytest.approx(30.0)


def test_build_universe_runs_end_to_end():
    crsp = make_crsp(
        [
            {"permno": 1, "ticker": "IBM", "dlyprevprc": 98.58},
            {"permno": 2, "ticker": "PENNY", "dlyprevprc": 2.0},
            {"permno": 3, "ticker": "ADRCO", "sharetype": "AD"},
        ]
    )
    universe, attrition = build_universe(crsp, make_link([]))

    assert universe["symbol"].tolist() == ["IBM"]
    assert universe["prev_close"].tolist() == [pytest.approx(98.58)]
    assert attrition["step"].iloc[0].startswith("NYSE-listed")
    assert attrition["stock_days"].iloc[-1] == 1
