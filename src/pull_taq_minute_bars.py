"""
Pull one trading day of one-minute trade bars from WRDS TAQ.

Raw-data step for the minute-bar panel in Chinco, Clark-Joseph and Ye (2019).
Returns are built from these bars in `build_minute_panel.py`.

The aggregation to minute bars runs inside the database. The consolidated tape
carries 10-30 million trades a day over the paper's sample, so pulling raw
trades for 2,013 trading days is not feasible; pulling bars is about 780,000
rows a day.

Schema notes, verified against the live database on 2026-11-17:
 - Trades live in `taqm_<year>.ctm_<yyyymmdd>`. There is no `taq` library.
 - `time_m` is a `time`, so bucketing needs `date + time_m` to make a timestamp.
 - `tr_corr` is text, zero padded: good trades are '00', not 0.
 - `tr_scond` is NULL for most regular trades in 2005, and holds multi-character
   flag sets ('F I', '4 I') in recent years. See EXCLUDED_CONDITION_FLAGS.
 - Share classes are split across `sym_root` and `sym_suffix` (Berkshire B is
   root 'BRK', suffix 'B'). Both are returned for the CRSP link.
"""

import argparse
from pathlib import Path

import pandas as pd
import wrds

from settings import config
from trading_calendar import session_bounds

DATA_DIR = Path(config("DATA_DIR"))
# Defaulted so this module imports, and its query-building functions stay
# testable, on a machine with no WRDS credentials (which is how CI runs).
WRDS_USERNAME = config("WRDS_USERNAME", default="")

# Sale conditions are filtered by exclusion, not by whitelist, because modern
# TAQ packs several flags into one string. A trade is dropped if its condition
# contains any of these:
#   4 derivatively priced     B/W average price        C cash
#   G bunched sold            H price variation        L sold last
#   M/6/9 closing prints      N next day               O/Q opening prints
#   P prior reference price   R seller                 T/U extended hours
#   V contingent trade        Z sold out of sequence   5 reopening print
#   7 qualified contingent
# What survives is the regular continuous-session tape: NULL/blank, '@'
# (regular), 'E' (automatic execution), 'F' (intermarket sweep), 'I' (odd lot).
#
# Excluding the opening and closing prints is also how the paper's instruction
# to "drop the last minute of each trading day" is implemented: bar 15:59 holds
# continuous-session trades only.
EXCLUDED_CONDITION_FLAGS = "456789BCGHLMNOPQRTUVWZ"

GOOD_CORRECTION_CODE = "00"

BAR_COLUMNS = [
    "symbol",
    "sym_root",
    "sym_suffix",
    "bar_start",
    "close",
    "open",
    "volume",
    "n_trades",
    "vwap",
]


def trades_table_name(date):
    """Return the TAQ trades table for a date.

    Examples
    --------
    >>> trades_table_name("2005-01-03")
    'taqm_2005.ctm_20050103'
    >>> trades_table_name("2025-01-03")
    'taqm_2025.ctm_20250103'
    """
    stamp = pd.Timestamp(date)
    return f"taqm_{stamp.year}.ctm_{stamp.strftime('%Y%m%d')}"


def build_bar_query(date, symbols=None, excluded_flags=EXCLUDED_CONDITION_FLAGS):
    """Build the SQL that aggregates one day of trades into one-minute bars.

    `array_agg` ordered by time picks the last and first trade in each minute;
    Postgres has no built-in last/first aggregate.

    The trading-hours window comes from the exchange calendar, not a hardcoded
    09:30-16:00. On the 16 half-days in the paper's sample the NYSE closes at
    13:00, and a fixed window returns trades stamped after the close that the
    panel build then rejects for falling outside the session grid.

    Parameters
    ----------
    date : str or datetime-like
        Trading day.
    symbols : sequence of str, optional
        Restrict to these `sym_root` values. None pulls the whole cross-section.
    excluded_flags : str
        Sale condition characters that disqualify a trade.

    Returns
    -------
    str
        SQL returning one row per symbol-minute.
    """
    table = trades_table_name(date)
    open_time, close_time = session_bounds(date)
    market_open = open_time.strftime("%H:%M:%S")
    market_close = close_time.strftime("%H:%M:%S")

    symbol_clause = ""
    if symbols is not None:
        symbols = list(symbols)
        if not symbols:
            raise ValueError("symbols was empty; pass None to pull the whole tape")
        quoted = ", ".join(f"'{s}'" for s in symbols)
        symbol_clause = f"AND sym_root IN ({quoted})"

    return f"""
        WITH filtered AS (
            SELECT sym_root,
                   sym_suffix,
                   date_trunc('minute', date + time_m) AS bar_start,
                   date + time_m AS trade_time,
                   price,
                   size
            FROM {table}
            WHERE tr_corr = '{GOOD_CORRECTION_CODE}'
              AND price > 0
              AND size > 0
              AND (tr_scond IS NULL OR tr_scond !~ '[{excluded_flags}]')
              AND time_m >= '{market_open}'
              AND time_m < '{market_close}'
              {symbol_clause}
        )
        SELECT sym_root,
               sym_suffix,
               bar_start,
               (array_agg(price ORDER BY trade_time DESC))[1] AS close,
               (array_agg(price ORDER BY trade_time ASC))[1] AS open,
               SUM(size) AS volume,
               COUNT(*) AS n_trades,
               SUM(price * size) / NULLIF(SUM(size), 0) AS vwap
        FROM filtered
        GROUP BY sym_root, sym_suffix, bar_start
        ORDER BY sym_root, sym_suffix, bar_start
    """


def build_condition_audit_query(date, symbols=None):
    """Build SQL counting trades by sale condition, for the filter audit.

    Used by the data-validation step to report how many trades each excluded
    flag removes, rather than asserting the filter is harmless.
    """
    table = trades_table_name(date)
    open_time, close_time = session_bounds(date)
    market_open = open_time.strftime("%H:%M:%S")
    market_close = close_time.strftime("%H:%M:%S")
    symbol_clause = ""
    if symbols is not None:
        quoted = ", ".join(f"'{s}'" for s in symbols)
        symbol_clause = f"AND sym_root IN ({quoted})"
    return f"""
        SELECT COALESCE(tr_scond, '<null>') AS tr_scond,
               tr_corr,
               COUNT(*) AS n_trades,
               SUM(size) AS volume
        FROM {table}
        WHERE price > 0 AND size > 0
          AND time_m >= '{market_open}' AND time_m < '{market_close}'
          {symbol_clause}
        GROUP BY 1, 2
        ORDER BY n_trades DESC
    """


def tidy_bars(df):
    """Put the raw SQL result into the canonical bar format.

    Split out from the pull so it is testable without WRDS. Builds the combined
    `symbol` the CRSP link matches on: 'BRK B' for root 'BRK', suffix 'B'.
    """
    df = df.copy()
    df["sym_root"] = df["sym_root"].astype(str).str.strip()
    df["sym_suffix"] = df["sym_suffix"].fillna("").astype(str).str.strip()
    df["symbol"] = (df["sym_root"] + " " + df["sym_suffix"]).str.strip()
    df["bar_start"] = pd.to_datetime(df["bar_start"])
    df["close"] = df["close"].astype("float64")
    df["open"] = df["open"].astype("float64")
    df["vwap"] = df["vwap"].astype("float64")
    df["volume"] = df["volume"].astype("float64")
    df["n_trades"] = df["n_trades"].astype("int32")
    return df[BAR_COLUMNS].sort_values(["symbol", "bar_start"]).reset_index(drop=True)


def pull_taq_minute_bars(
    date,
    symbols=None,
    wrds_username=WRDS_USERNAME,
    excluded_flags=EXCLUDED_CONDITION_FLAGS,
    db=None,
):
    """Pull one trading day of one-minute trade bars from TAQ.

    Parameters
    ----------
    date : str or datetime-like
        Trading day. Weekends and holidays have no table and will raise.
    symbols : sequence of str, optional
        Restrict to these `sym_root` values. None pulls the whole cross-section.
    wrds_username : str
        WRDS login, read from .env by default.
    excluded_flags : str
        Sale condition characters that disqualify a trade.
    db : wrds.Connection, optional
        Reuse an open connection. A job array should open one and pass it in
        rather than reconnecting per day.

    Returns
    -------
    pandas.DataFrame
        One row per symbol-minute, columns BAR_COLUMNS.
    """
    query = build_bar_query(date, symbols=symbols, excluded_flags=excluded_flags)

    close_when_done = db is None
    if db is None:
        if not wrds_username:
            raise ValueError(
                "No WRDS username. Set WRDS_USERNAME in .env (see .env.example) "
                "or pass wrds_username=."
            )
        db = wrds.Connection(wrds_username=wrds_username)
    try:
        df = db.raw_sql(query)
    finally:
        if close_when_done:
            db.close()

    return tidy_bars(df)


def taq_minute_bars_path(date, data_dir=DATA_DIR):
    """Path this module writes a day's bars to."""
    stamp = pd.Timestamp(date)
    return Path(data_dir) / "taq_bars" / f"bars_{stamp.strftime('%Y%m%d')}.parquet"


def load_taq_minute_bars(date, data_dir=DATA_DIR):
    """Read back a day of bars written by this module."""
    return pd.read_parquet(taq_minute_bars_path(date, data_dir=data_dir))


def write_taq_minute_bars(df, date, data_dir=DATA_DIR):
    """Write a day's bars to its canonical path, creating the directory."""
    path = taq_minute_bars_path(date, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def main():
    """Pull one day of bars from the command line, for use in a job array."""
    parser = argparse.ArgumentParser(
        description="Pull one trading day of one-minute TAQ bars."
    )
    parser.add_argument("date", help="trading day, e.g. 2005-01-03")
    parser.add_argument(
        "--symbols",
        default=None,
        help="comma-separated sym_root values; omit for the whole cross-section",
    )
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    df = pull_taq_minute_bars(args.date, symbols=symbols)
    path = write_taq_minute_bars(df, args.date)
    print(
        f"{len(df):,} bars for {args.date} "
        f"({df['symbol'].nunique():,} symbols) written to {path}"
    )


if __name__ == "__main__":
    main()
