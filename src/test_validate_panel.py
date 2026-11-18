"""
Data validation for the minute panel: checks against sources we did not build.

These differ in kind from the unit tests. `test_build_minute_panel.py` checks
the code against inputs we made up; these check the *data* against CRSP, which
is compiled by different people from different source files. Code can be
perfectly correct and still be pointed at the wrong stock.

They need the pulled data on disk, so they skip rather than fail when it is
absent -- which is how they behave in CI, where there are no WRDS credentials.

Tolerances are stated with the mechanism that justifies them, not tuned to the
result. The volume bands follow from a known fact: our filters deliberately
exclude the opening and closing auction prints, extended-hours trading, and
average-price and out-of-sequence reports, which together are a single-digit
to low-teens percentage of a day's consolidated volume. So TAQ volume should
sit a little *below* CRSP's and never above it.
"""

from pathlib import Path

import pandas as pd
import pytest

from build_minute_panel import (
    available_panel_dates,
    coverage_path,
    load_coverage,
)
from build_universe import load_universe, universe_path
from pull_crsp_daily import crsp_daily_path, load_crsp_daily
from pull_taq_minute_bars import load_taq_minute_bars, taq_minute_bars_path
from trading_calendar import session_minute_grid

# Our filters are a strict subset of the consolidated tape, so the ratio of our
# volume to CRSP's must be at or below 1. The lower bound allows for the
# auction and irregular-condition volume we remove on purpose.
MIN_MEDIAN_VOLUME_RATIO = 0.85
MAX_MEDIAN_VOLUME_RATIO = 1.00
MIN_SHARE_WITHIN_15PCT = 0.85
MIN_SHARE_FOUND_IN_TAQ = 0.95

# A full NYSE session is 390 minutes; a scheduled early close is 210. Two days
# in the paper's sample are neither, and both are real: on 2005-06-01 the NYSE
# closed at 15:56 after a technical problem, and on 2006-12-27 it opened two
# minutes late. They are named here rather than absorbed into a loose tolerance,
# so that any *other* odd length still fails.
VALID_SESSION_LENGTHS = {390, 210}
KNOWN_SHORT_SESSIONS = {
    pd.Timestamp("2005-06-01"): 386,
    pd.Timestamp("2006-12-27"): 388,
}

REFERENCE_DAY = "2005-01-03"


def _require(path, what):
    """Skip the test when a pulled input is missing."""
    if not Path(path).exists():
        pytest.skip(f"{what} not on disk; run the pipeline first")


@pytest.fixture(scope="module")
def reference_day_bars():
    _require(taq_minute_bars_path(REFERENCE_DAY), "TAQ bars")
    return load_taq_minute_bars(REFERENCE_DAY)


@pytest.fixture(scope="module")
def reference_day_crsp():
    _require(crsp_daily_path(), "CRSP daily file")
    crsp = load_crsp_daily()
    return crsp[crsp["dlycaldt"] == pd.Timestamp(REFERENCE_DAY)]


@pytest.fixture(scope="module")
def reference_day_universe():
    _require(universe_path(), "universe")
    universe = load_universe()
    return universe[universe["date"] == pd.Timestamp(REFERENCE_DAY)]


def _volume_comparison(bars, universe):
    """Join per-stock TAQ volume to CRSP volume for one day."""
    taq_volume = bars.groupby("symbol")["volume"].sum().rename("taq_volume")
    joined = universe.merge(taq_volume, left_on="symbol", right_index=True, how="left")
    both = joined[joined["taq_volume"].notna() & joined["volume"].gt(0)].copy()
    both["ratio"] = both["taq_volume"] / both["volume"]
    return joined, both


# --------------------------------------------------------------------------
# Reconciliation against CRSP
# --------------------------------------------------------------------------


def test_taq_volume_never_exceeds_crsp(reference_day_bars, reference_day_universe):
    """An identity, not a tolerance.

    Our trade filters keep a subset of the consolidated tape, so per-stock TAQ
    volume cannot exceed CRSP's. Exceeding it would mean double counting --
    most likely the same trade reported by two venues.
    """
    _, both = _volume_comparison(reference_day_bars, reference_day_universe)
    assert len(both) > 100, "too few stocks to validate against"
    offenders = both[both["ratio"] > 1.001]
    assert offenders.empty, (
        f"{len(offenders)} stocks have more TAQ volume than CRSP, "
        f"worst {offenders['ratio'].max():.3f}"
    )


def test_median_volume_ratio_is_in_the_expected_band(
    reference_day_bars, reference_day_universe
):
    """The bulk of CRSP's volume should be accounted for."""
    _, both = _volume_comparison(reference_day_bars, reference_day_universe)
    median = both["ratio"].median()
    assert MIN_MEDIAN_VOLUME_RATIO <= median <= MAX_MEDIAN_VOLUME_RATIO, (
        f"median TAQ/CRSP volume {median:.3f} outside "
        f"[{MIN_MEDIAN_VOLUME_RATIO}, {MAX_MEDIAN_VOLUME_RATIO}]"
    )


def test_most_stocks_reconcile_individually(reference_day_bars, reference_day_universe):
    """A good median can hide a badly matched tail, so check the spread too."""
    _, both = _volume_comparison(reference_day_bars, reference_day_universe)
    share = both["ratio"].between(0.85, 1.15).mean()
    assert share >= MIN_SHARE_WITHIN_15PCT, (
        f"only {share:.1%} of stocks within 15% of CRSP volume"
    )


def test_universe_stocks_are_present_in_taq(reference_day_bars, reference_day_universe):
    """A symbol we cannot find in TAQ is a failed match, not a quiet stock."""
    joined, _ = _volume_comparison(reference_day_bars, reference_day_universe)
    # Stocks CRSP says did not trade are legitimately absent from the tape.
    traded = joined[joined["volume"] > 0]
    found = traded["taq_volume"].notna().mean()
    assert found >= MIN_SHARE_FOUND_IN_TAQ, (
        f"only {found:.1%} of stocks CRSP says traded were found in TAQ"
    )


def test_opening_price_matches_crsp(reference_day_bars, reference_day_crsp):
    """The first bar's open should be CRSP's official open.

    The *close* deliberately will not match, because we exclude the closing
    auction print that sets CRSP's official close.
    """
    ibm_bars = reference_day_bars[reference_day_bars["symbol"] == "IBM"]
    if ibm_bars.empty:
        pytest.skip("IBM not in this day's bars")
    taq_open = ibm_bars.sort_values("bar_start")["open"].iloc[0]
    crsp_open = reference_day_crsp.loc[
        reference_day_crsp["ticker"] == "IBM", "dlyopen"
    ].iloc[0]
    assert taq_open == pytest.approx(crsp_open, abs=0.01)


# --------------------------------------------------------------------------
# Internal consistency of the built panel
# --------------------------------------------------------------------------


def test_every_panel_day_has_a_valid_session_length():
    """Catches a day padded out to 390 minutes when the NYSE closed early."""
    dates = available_panel_dates()
    if not dates:
        pytest.skip("no panel days built")
    for date in dates:
        n_minutes = len(session_minute_grid(date))
        expected = KNOWN_SHORT_SESSIONS.get(pd.Timestamp(date).normalize())
        if expected is not None:
            assert n_minutes == expected, (
                f"{date.date()} is a known short session of {expected} minutes "
                f"but the grid has {n_minutes}"
            )
        else:
            assert n_minutes in VALID_SESSION_LENGTHS, (
                f"{date.date()} has an unexpected session length of {n_minutes}; "
                "if this is a real exchange anomaly, add it to KNOWN_SHORT_SESSIONS"
            )


def test_coverage_records_are_internally_consistent():
    """Traded plus carried-forward minutes cannot exceed the session."""
    dates = available_panel_dates()
    if not dates:
        pytest.skip("no panel days built")
    coverage = load_coverage(dates)
    total = coverage["n_traded_minutes"] + coverage["n_filled_minutes"]
    assert (total <= coverage["n_session_minutes"]).all()
    assert (coverage["n_returns"] <= coverage["n_traded_minutes"]).all()


def test_coverage_has_no_unexplained_break():
    """A day losing most of its stocks means a pull failed, not a quiet market."""
    dates = available_panel_dates()
    if len(dates) < 12:
        pytest.skip("too few panel days to judge a break in coverage")
    coverage = load_coverage(dates)
    per_day = coverage.groupby("date").size()
    assert (per_day > 0.5 * per_day.median()).all(), (
        "a trading day has less than half the usual number of stocks: "
        f"{per_day[per_day <= 0.5 * per_day.median()].to_dict()}"
    )


def test_coverage_exists_for_every_built_day():
    """Returns and coverage are written together; one without the other is a bug."""
    for date in available_panel_dates():
        assert Path(coverage_path(date)).exists(), f"no coverage for {date.date()}"
