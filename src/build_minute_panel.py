"""
The cleaning step for the minute-bar panel in Chinco, Clark-Joseph and
Ye (2019).

Highlevel Overview:
    Takes prices from TAQ and produces returns required for the paper.

1) It takes the one minute bars written by `pull_taq_minute_bars.py`

2) for each trading day, it produces a tidy matrix of one-minute returns plus a per-stock
coverage record.

**No external service needed - every function here is unit testable in CI without WRDS credentials.**



Storage layout
--------------
One file per trading day, rows indexed by minute and columns named by symbol,
stored as float32:

    _data/minute_panel/returns_YYYYMMDD.parquet

The LASSO stage reads exactly one day at a time -- it needs a (390 x N) matrix
of returns and nothing else -- so partitioning by day matches its access
pattern. The wide float32 layout also matters for size: at roughly 2,000 stocks
and 390 minutes a day costs about 3 MB, so the paper's 2,013 trading days come
to about 6 GB. The same panel in long format with symbol and minute keys on
every row is an order of magnitude larger and slower to read.

A second, much smaller file records coverage per stock-day:

    _data/minute_panel/coverage_YYYYMMDD.parquet

The summary table and coverage figure are built from these coverage records
rather than from the panel itself, so producing them does not mean re-reading
the whole panel.





Bar and return convention
-------------------------
Bar `t` is the last trade price in `[t, t+1)`, labelled by its starting minute
(see `pull_taq_minute_bars.py`). For a full session the bar grid runs 09:30
through 15:59 (390 bars) and returns are

    r_t = close_t / close_{t-1} - 1,    t = 09:31 ... 15:59

which is 389 returns. Returns are simple, not log. Nothing is carried across
days: there is no return at 09:30, so the overnight return never enters.

Half-days (the NYSE closes at 13:00 on 16 days in the paper's sample) get a
shorter grid from the exchange calendar rather than being padded out to 390
minutes, which would otherwise inject several hours of fake zero returns.





ASSUMPTIONS RECORDED HERE (see README)
-------------------------------------
Minutes in which a stock did not trade have no price. The paper does not say
how it handles them. Our choice:

 - Prices are carried forward within the day, never backward. A stock whose
   first trade is at 10:15 has no price before 10:15, not its opening price.
 - A return at minute `t` is kept only if the stock actually traded in bar `t`.
   If nobody traded, the price did not move because there was no trading, not
   because the price was unchanged, and recording a zero return would be a
   fabrication.
 - The carried-forward base price may be at most MAX_STALENESS_MINUTES old.
   Beyond that the "one-minute" return is really a multi-minute return and is
   dropped instead.

Each of these is measured rather than assumed away: the coverage record counts
traded, filled and dropped minutes per stock-day, so the summary table and
coverage figure show exactly how much of the panel rests on this choice.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from settings import config
from trading_calendar import (
    EXCHANGE_CALENDAR,
    session_minute_grid,
)

DATA_DIR = Path(config("DATA_DIR"))

# A carried-forward price older than this many minutes is not used as the base
# of a return. The paper is silent on stale prices; see the module docstring.
MAX_STALENESS_MINUTES = 5

COVERAGE_COLUMNS = [
    "date",
    "symbol",
    "n_session_minutes",
    "n_traded_minutes",
    "n_filled_minutes",
    "n_returns",
    "n_dropped_stale",
    "first_bar",
    "last_bar",
    "volume",
    "n_trades",
    "mean_return_bps",
    "sd_return_bps",
]


# how many minutes in the session, and what their labels are, for a given trading day
def bars_to_price_matrix(bars, grid):
    """Pivot bars into a wide price matrix aligned to the session grid.

    Bars whose `bar_start` falls outside the session grid are dropped; the
    count of dropped bars is returned so the caller can assert it is zero or
    explain it.

    Parameters
    ----------
    bars : pandas.DataFrame
        Columns `symbol`, `bar_start`, `close` at minimum.
    grid : pandas.DatetimeIndex
        Session minute grid from `session_minute_grid`.

    Returns
    -------
    prices : pandas.DataFrame
        Indexed by `grid`, one column per symbol. NaN where the stock did not
        trade in that minute.
    n_off_grid : int
        Number of input bars whose minute was not on the grid.
    """
    if bars.empty:
        return pd.DataFrame(index=grid, dtype="float64"), 0

    on_grid = bars["bar_start"].isin(grid)
    n_off_grid = int((~on_grid).sum())
    kept = bars.loc[on_grid]

    prices = kept.pivot_table(
        index="bar_start", columns="symbol", values="close", aggfunc="last"
    )
    prices = prices.reindex(grid)
    prices.index.name = "minute"
    prices.columns.name = "symbol"
    return prices, n_off_grid


def staleness_matrix(prices):
    """Minutes elapsed since each stock's most recent actual trade.

    Zero in a minute the stock traded, NaN before its first trade of the day.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2005-01-03 09:30", periods=4, freq="1min")
    >>> p = pd.DataFrame({"A": [np.nan, 10.0, np.nan, np.nan]}, index=idx)
    >>> staleness_matrix(p)["A"].tolist()
    [nan, 0.0, 1.0, 2.0]
    """
    traded = prices.notna()
    # Index of the most recent traded minute, carried forward.
    positions = pd.DataFrame(
        np.where(traded, np.arange(len(prices))[:, None], np.nan),
        index=prices.index,
        columns=prices.columns,
    ).ffill()
    return pd.DataFrame(
        np.arange(len(prices))[:, None] - positions.to_numpy(),
        index=prices.index,
        columns=prices.columns,
    )


def prices_to_returns(prices, max_staleness=MAX_STALENESS_MINUTES):
    """Difference a price matrix into one-minute simple returns.

    A return at minute `t` survives only if the stock traded in bar `t` and its
    previous price is at most `max_staleness` minutes old.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price matrix from `bars_to_price_matrix`, NaN where no trade.
    max_staleness : int
        Maximum age in minutes of the carried-forward base price.

    Returns
    -------
    pandas.DataFrame
        Simple returns, float32, same shape as `prices`. The first row is
        always NaN because there is no prior price within the day.
    """
    if prices.empty:
        return prices.astype("float32")

    stale = staleness_matrix(prices)
    filled = prices.ffill()

    returns = filled / filled.shift(1) - 1.0

    traded_now = stale.eq(0)
    base_fresh_enough = stale.shift(1).le(max_staleness)
    returns = returns.where(traded_now & base_fresh_enough)

    return returns.astype("float32")


def day_coverage(bars, prices, returns, date):
    """Build the per-stock coverage record for one trading day.

    One row per symbol, counting how much of the session the stock actually
    traded in and how many returns survived the staleness rule. This is what
    the summary table and coverage figure are built from.
    """
    day = pd.Timestamp(date).normalize()
    if prices.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)

    traded = prices.notna()
    filled = prices.ffill().notna()

    if bars.empty:
        volume = pd.Series(0.0, index=prices.columns)
        n_trades = pd.Series(0, index=prices.columns)
    else:
        grouped = bars.groupby("symbol")
        volume = grouped["volume"].sum().reindex(prices.columns).fillna(0.0)
        n_trades = grouped["n_trades"].sum().reindex(prices.columns).fillna(0)

    # Minutes where the stock traded but the return was dropped for staleness.
    # The first traded minute of the day can never have a return, so it is not
    # counted as a staleness drop.
    droppable = traded & traded.cumsum().gt(1)
    n_dropped_stale = (droppable & returns.isna()).sum()

    coverage = pd.DataFrame(
        {
            "date": day,
            "symbol": prices.columns,
            "n_session_minutes": len(prices),
            "n_traded_minutes": traded.sum().to_numpy(),
            "n_filled_minutes": (filled & ~traded).sum().to_numpy(),
            "n_returns": returns.notna().sum().to_numpy(),
            "n_dropped_stale": n_dropped_stale.to_numpy(),
            "first_bar": _first_valid(prices),
            "last_bar": _last_valid(prices),
            "volume": volume.to_numpy(),
            "n_trades": n_trades.to_numpy(),
            "mean_return_bps": (returns.mean() * 1e4).to_numpy(),
            "sd_return_bps": (returns.std() * 1e4).to_numpy(),
        }
    )
    return coverage[COVERAGE_COLUMNS].reset_index(drop=True)


def _first_valid(prices):
    """Timestamp of each column's first non-missing price."""
    return [prices[col].first_valid_index() for col in prices.columns]


def _last_valid(prices):
    """Timestamp of each column's last non-missing price."""
    return [prices[col].last_valid_index() for col in prices.columns]


def build_day_panel(
    bars,
    date,
    max_staleness=MAX_STALENESS_MINUTES,
    calendar_name=EXCHANGE_CALENDAR,
):
    """Build one trading day's return panel and coverage record from bars.

    This is the whole cleaning step for a day, and it is a pure function of its
    inputs so it can be tested on synthetic bars.

    Parameters
    ----------
    bars : pandas.DataFrame
        One day of bars as written by `pull_taq_minute_bars.py`.
    date : str or datetime-like
        The trading day the bars belong to.
    max_staleness : int
        Maximum age in minutes of a carried-forward base price.
    calendar_name : str
        Exchange calendar name.

    Returns
    -------
    returns : pandas.DataFrame
        Wide float32 returns, indexed by minute, one column per symbol.
    coverage : pandas.DataFrame
        One row per symbol, columns COVERAGE_COLUMNS.
    """
    grid = session_minute_grid(date, calendar_name=calendar_name)
    prices, n_off_grid = bars_to_price_matrix(bars, grid)
    if n_off_grid:
        raise ValueError(
            f"{n_off_grid} bars on {pd.Timestamp(date).date()} fall outside the "
            f"{grid[0].time()}-{grid[-1].time()} session grid; check the "
            "trading-hours filter in pull_taq_minute_bars.py"
        )
    returns = prices_to_returns(prices, max_staleness=max_staleness)
    coverage = day_coverage(bars, prices, returns, date)
    return returns, coverage


def minute_panel_path(date, data_dir=DATA_DIR):
    """Path the day's return panel is written to."""
    stamp = pd.Timestamp(date)
    return (
        Path(data_dir) / "minute_panel" / f"returns_{stamp.strftime('%Y%m%d')}.parquet"
    )


def coverage_path(date, data_dir=DATA_DIR):
    """Path the day's coverage record is written to."""
    stamp = pd.Timestamp(date)
    return (
        Path(data_dir) / "minute_panel" / f"coverage_{stamp.strftime('%Y%m%d')}.parquet"
    )


def write_day_panel(returns, coverage, date, data_dir=DATA_DIR):
    """Write a day's returns and coverage record to their canonical paths."""
    returns_file = minute_panel_path(date, data_dir=data_dir)
    coverage_file = coverage_path(date, data_dir=data_dir)
    returns_file.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(returns_file)
    coverage.to_parquet(coverage_file, index=False)
    return returns_file, coverage_file


def load_day_panel(date, data_dir=DATA_DIR):
    """Read back a day's return panel."""
    return pd.read_parquet(minute_panel_path(date, data_dir=data_dir))


def load_coverage(dates, data_dir=DATA_DIR):
    """Read and concatenate coverage records for a sequence of dates."""
    frames = [pd.read_parquet(coverage_path(date, data_dir=data_dir)) for date in dates]
    return pd.concat(frames, ignore_index=True)


def available_panel_dates(data_dir=DATA_DIR):
    """Trading days that already have a return panel on disk."""
    folder = Path(data_dir) / "minute_panel"
    if not folder.exists():
        return []
    stems = sorted(
        p.stem.removeprefix("returns_") for p in folder.glob("returns_*.parquet")
    )
    return [pd.Timestamp(s) for s in stems]
