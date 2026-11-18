"""
Unit tests for the minute-bar panel cleaning step.

These run on synthetic bars, so they need no WRDS credentials and execute in
CI. They check the conventions the panel depends on -- the shape of the session
grid, the bar-to-return alignment, and the stale-price rule -- because those
are the choices that would silently corrupt every downstream exhibit if they
were wrong.
"""

import numpy as np
import pandas as pd
import pytest

from build_minute_panel import (
    bars_to_price_matrix,
    build_day_panel,
    day_coverage,
    prices_to_returns,
    staleness_matrix,
)
from pull_taq_minute_bars import (
    BAR_COLUMNS,
    build_bar_query,
    tidy_bars,
    trades_table_name,
)
from trading_calendar import session_minute_grid

FULL_DAY = "2005-01-03"  # a regular Monday session
HALF_DAY = "2005-11-25"  # the day after Thanksgiving, 13:00 close
HALF_DAY_JULY = "2006-07-03"  # the day before July 4th, 13:00 close


def make_bars(date, symbol_prices):
    """Build a synthetic bar frame.

    Parameters
    ----------
    date : str
        Trading day.
    symbol_prices : dict
        Maps symbol -> {"HH:MM": price}.
    """
    rows = []
    for symbol, prices in symbol_prices.items():
        for hhmm, price in prices.items():
            rows.append(
                {
                    "symbol": symbol,
                    "sym_root": symbol,
                    "sym_suffix": "",
                    "bar_start": pd.Timestamp(f"{date} {hhmm}"),
                    "close": price,
                    "open": price,
                    "volume": 100.0,
                    "n_trades": np.int32(1),
                    "vwap": price,
                }
            )
    return pd.DataFrame(rows, columns=BAR_COLUMNS)


# --------------------------------------------------------------------------
# Session grid. The paper's 356 forecasts per stock-day follow arithmetically
# from a 390-bar session, so the grid is the one number everything else rests
# on.
# --------------------------------------------------------------------------


def test_full_session_has_390_bars():
    grid = session_minute_grid(FULL_DAY)
    assert len(grid) == 390
    assert grid[0] == pd.Timestamp(f"{FULL_DAY} 09:30")
    assert grid[-1] == pd.Timestamp(f"{FULL_DAY} 15:59")


def test_half_day_is_not_padded_to_a_full_session():
    """An early close must shorten the grid, not fill it with fake minutes."""
    grid = session_minute_grid(HALF_DAY)
    assert len(grid) == 210
    assert grid[-1] == pd.Timestamp(f"{HALF_DAY} 12:59")


def test_paper_forecast_window_follows_from_the_grid():
    """10:04 through 15:59 is 356 minutes, matching Section II.A."""
    grid = session_minute_grid(FULL_DAY)
    forecastable = grid[
        (grid >= pd.Timestamp(f"{FULL_DAY} 10:04"))
        & (grid <= pd.Timestamp(f"{FULL_DAY} 15:59"))
    ]
    assert len(forecastable) == 356


def test_non_trading_day_raises():
    with pytest.raises(ValueError, match="not a NYSE trading day"):
        session_minute_grid("2005-01-01")  # a Saturday, and New Year's Day


# --------------------------------------------------------------------------
# Bars to prices
# --------------------------------------------------------------------------


def test_price_matrix_is_aligned_to_the_grid_with_gaps_as_nan():
    bars = make_bars(FULL_DAY, {"AAA": {"09:30": 10.0, "09:32": 10.5}})
    grid = session_minute_grid(FULL_DAY)
    prices, n_off_grid = bars_to_price_matrix(bars, grid)

    assert n_off_grid == 0
    assert len(prices) == 390
    assert prices.loc[pd.Timestamp(f"{FULL_DAY} 09:30"), "AAA"] == 10.0
    assert np.isnan(prices.loc[pd.Timestamp(f"{FULL_DAY} 09:31"), "AAA"])
    assert prices.loc[pd.Timestamp(f"{FULL_DAY} 09:32"), "AAA"] == 10.5


def test_off_grid_bars_are_counted_not_silently_kept():
    """A bar at 16:05 means the trading-hours filter leaked."""
    bars = make_bars(FULL_DAY, {"AAA": {"09:30": 10.0, "16:05": 11.0}})
    grid = session_minute_grid(FULL_DAY)
    prices, n_off_grid = bars_to_price_matrix(bars, grid)

    assert n_off_grid == 1
    assert prices["AAA"].notna().sum() == 1


def test_build_day_panel_rejects_off_grid_bars():
    bars = make_bars(FULL_DAY, {"AAA": {"09:30": 10.0, "16:05": 11.0}})
    with pytest.raises(ValueError, match="outside the"):
        build_day_panel(bars, FULL_DAY)


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


def test_staleness_counts_minutes_since_last_trade():
    idx = pd.date_range(f"{FULL_DAY} 09:30", periods=5, freq="1min")
    prices = pd.DataFrame({"AAA": [np.nan, 10.0, np.nan, np.nan, 11.0]}, index=idx)
    stale = staleness_matrix(prices)
    assert stale["AAA"].tolist()[1:] == [0.0, 1.0, 2.0, 0.0]
    assert np.isnan(stale["AAA"].iloc[0])


# --------------------------------------------------------------------------
# Returns. These encode the assumptions recorded in the module docstring.
# --------------------------------------------------------------------------


def test_consecutive_trades_give_the_simple_return():
    idx = pd.date_range(f"{FULL_DAY} 09:30", periods=3, freq="1min")
    prices = pd.DataFrame({"AAA": [10.0, 11.0, 11.0]}, index=idx)
    returns = prices_to_returns(prices)

    assert np.isnan(returns["AAA"].iloc[0])  # no prior price within the day
    assert returns["AAA"].iloc[1] == pytest.approx(0.1, abs=1e-7)
    assert returns["AAA"].iloc[2] == pytest.approx(0.0, abs=1e-7)


def test_no_return_in_a_minute_without_a_trade():
    """A quiet minute must not be recorded as a zero return."""
    idx = pd.date_range(f"{FULL_DAY} 09:30", periods=3, freq="1min")
    prices = pd.DataFrame({"AAA": [10.0, np.nan, 11.0]}, index=idx)
    returns = prices_to_returns(prices)

    assert np.isnan(returns["AAA"].iloc[1])
    # The 09:32 trade does get a return, off the two-minute-old base price.
    assert returns["AAA"].iloc[2] == pytest.approx(0.1, abs=1e-7)


def test_base_price_older_than_max_staleness_is_dropped():
    """Past the staleness limit a 'one-minute' return spans too long."""
    idx = pd.date_range(f"{FULL_DAY} 09:30", periods=10, freq="1min")
    values = [10.0] + [np.nan] * 8 + [11.0]
    prices = pd.DataFrame({"AAA": values}, index=idx)

    lenient = prices_to_returns(prices, max_staleness=9)
    strict = prices_to_returns(prices, max_staleness=5)

    assert lenient["AAA"].iloc[9] == pytest.approx(0.1, abs=1e-7)
    assert np.isnan(strict["AAA"].iloc[9])


def test_prices_are_never_carried_backward():
    """A stock that starts trading late has no price before its first trade."""
    idx = pd.date_range(f"{FULL_DAY} 09:30", periods=4, freq="1min")
    prices = pd.DataFrame({"AAA": [np.nan, np.nan, 10.0, 10.5]}, index=idx)
    returns = prices_to_returns(prices)

    assert returns["AAA"].iloc[:3].isna().all()
    assert returns["AAA"].iloc[3] == pytest.approx(0.05, abs=1e-7)


def test_returns_are_float32_for_storage():
    bars = make_bars(FULL_DAY, {"AAA": {"09:30": 10.0, "09:31": 10.1}})
    returns, _ = build_day_panel(bars, FULL_DAY)
    assert returns["AAA"].dtype == np.float32


# --------------------------------------------------------------------------
# Coverage record, which the summary table and coverage figure are built from
# --------------------------------------------------------------------------


def test_coverage_counts_traded_filled_and_dropped_minutes():
    bars = make_bars(
        FULL_DAY,
        {"AAA": {"09:30": 10.0, "09:31": 10.1, "09:40": 10.2}},
    )
    grid = session_minute_grid(FULL_DAY)
    prices, _ = bars_to_price_matrix(bars, grid)
    returns = prices_to_returns(prices, max_staleness=5)
    coverage = day_coverage(bars, prices, returns, FULL_DAY).set_index("symbol")

    row = coverage.loc["AAA"]
    assert row["n_session_minutes"] == 390
    assert row["n_traded_minutes"] == 3
    # 09:32-09:39 are carried forward from the 09:31 trade; after 09:40 the
    # price is carried to the close.
    assert row["n_filled_minutes"] == 387
    # 09:31 gets a return. 09:40's base price is 9 minutes old, so it is
    # dropped. 09:30 is the first trade and can never have a return.
    assert row["n_returns"] == 1
    assert row["n_dropped_stale"] == 1
    assert row["first_bar"] == pd.Timestamp(f"{FULL_DAY} 09:30")
    assert row["last_bar"] == pd.Timestamp(f"{FULL_DAY} 09:40")
    assert row["volume"] == 300.0
    assert row["n_trades"] == 3


def test_coverage_has_one_row_per_symbol():
    bars = make_bars(
        FULL_DAY,
        {
            "AAA": {"09:30": 10.0, "09:31": 10.1},
            "BBB": {"09:30": 20.0, "09:31": 20.2},
        },
    )
    _, coverage = build_day_panel(bars, FULL_DAY)
    assert sorted(coverage["symbol"]) == ["AAA", "BBB"]
    assert len(coverage) == 2


def test_empty_day_does_not_crash():
    """A symbol list that matched nothing should give an empty panel, not an error."""
    bars = pd.DataFrame(columns=BAR_COLUMNS)
    returns, coverage = build_day_panel(bars, FULL_DAY)
    assert returns.empty
    assert coverage.empty


# --------------------------------------------------------------------------
# Query construction. The SQL itself cannot run in CI, but its shape can be
# checked. Each of these guards a mistake that was actually made and caught
# against the live database on 2026-11-17; see the module docstring of
# pull_taq_minute_bars.py.
# --------------------------------------------------------------------------


def test_table_name_matches_wrds_naming():
    """Trades live in per-year libraries. There is no `taq` library."""
    assert trades_table_name("2005-01-03") == "taqm_2005.ctm_20050103"
    assert trades_table_name("2012-06-01") == "taqm_2012.ctm_20120601"
    assert trades_table_name("2025-01-03") == "taqm_2025.ctm_20250103"


def test_query_aggregates_server_side():
    """The GROUP BY must be in the query; downloading ticks does not scale."""
    sql = build_bar_query("2005-01-03", symbols=["IBM"])
    assert "GROUP BY sym_root, sym_suffix, bar_start" in sql
    assert "'IBM'" in sql
    assert "taqm_2005.ctm_20050103" in sql


def test_query_buckets_on_a_timestamp_not_a_bare_time():
    """`time_m` is a `time`, which date_trunc rejects; it needs date + time_m."""
    sql = build_bar_query("2005-01-03")
    assert "date_trunc('minute', date + time_m)" in sql


def test_query_compares_correction_code_as_text():
    """`tr_corr` is zero-padded text, so `tr_corr = 0` is a type error."""
    sql = build_bar_query("2005-01-03")
    assert "tr_corr = '00'" in sql


def test_query_keeps_null_sale_conditions():
    """Most regular 2005 trades have tr_scond NULL.

    Filtering with a whitelist drops them all, which silently discards the
    majority of the tape. The filter must be by exclusion and must admit NULL.
    """
    sql = build_bar_query("2005-01-03")
    assert "tr_scond IS NULL OR" in sql
    assert "!~" in sql


def test_query_excludes_auction_and_irregular_conditions():
    sql = build_bar_query("2005-01-03", excluded_flags="BZ")
    assert "tr_scond !~ '[BZ]'" in sql


def test_query_applies_the_trade_level_filters():
    sql = build_bar_query("2005-01-03")
    assert "price > 0" in sql
    assert "size > 0" in sql
    assert "09:30:00" in sql and "16:00:00" in sql


def test_query_uses_the_calendar_session_not_a_fixed_window():
    """A hardcoded 16:00 close returns post-close trades on a half-day.

    2006-07-03 closes at 13:00. With a fixed window TAQ returned 1,815 bars
    after the close, which `build_day_panel` then rejected for falling outside
    the session grid, halting the build.
    """
    sql = build_bar_query(HALF_DAY_JULY)
    assert "13:00:00" in sql
    assert "16:00:00" not in sql


def test_query_uses_the_calendar_open_on_a_late_opening():
    """2006-12-27 opened two minutes late; the window must follow."""
    sql = build_bar_query("2006-12-27")
    assert "09:32:00" in sql


def test_tidy_bars_builds_the_share_class_symbol():
    """Berkshire B is root 'BRK' suffix 'B'; the CRSP link needs them joined."""
    raw = pd.DataFrame(
        {
            "sym_root": ["BRK", "IBM"],
            "sym_suffix": ["B", None],
            "bar_start": [pd.Timestamp(f"{FULL_DAY} 09:30")] * 2,
            "close": [2900.0, 98.97],
            "open": [2900.0, 98.97],
            "volume": [100.0, 199400.0],
            "n_trades": [1, 159],
            "vwap": [2900.0, 98.97],
        }
    )
    tidied = tidy_bars(raw).set_index("sym_root")
    assert tidied.loc["BRK", "symbol"] == "BRK B"
    assert tidied.loc["IBM", "symbol"] == "IBM"


def test_empty_symbol_list_is_rejected():
    """An empty list would silently pull nothing; None means 'everything'."""
    with pytest.raises(ValueError, match="pass None"):
        build_bar_query("2005-01-03", symbols=[])
