"""
Build the one-minute return panel for given trading days.

The driver that ties the pieces together: pull a day's bars from TAQ, screen
them to the stocks eligible that day, difference them into returns, and write
both the panel and its coverage record.

This is the only part of the panel build that touches WRDS, which keeps the
return math in `build_minute_panel.py` unit testable without credentials.

Run one day, or many:

    python src/run_minute_panel.py 2005-01-03
    python src/run_minute_panel.py 2005-01-03 2005-02-01 2005-03-01

`dodo.py` runs one task per trading day, so a failed or newly added day rebuilds
on its own and the same tasks can be farmed out to a cluster job array.
"""

import argparse

import pandas as pd

from build_minute_panel import (
    MAX_STALENESS_MINUTES,
    build_day_panel,
    write_day_panel,
)
from pull_taq_minute_bars import (
    load_taq_minute_bars,
    taq_minute_bars_path,
    write_taq_minute_bars,
)
from settings import config


def restrict_to_universe(bars, universe, date):
    """Keep only bars for symbols eligible on `date`.

    The pull asks TAQ for the eligible `sym_root` values, but a root alone does
    not pin down a share class, so the screen is applied again here against the
    full symbol.
    """
    day = pd.Timestamp(date).normalize()
    eligible = set(universe.loc[universe["date"] == day, "symbol"])
    return bars[bars["symbol"].isin(eligible)]


def day_bars(date, universe=None, db=None, use_cache=True):
    """Return one day's minute bars, from the cache if we already have them.

    Bars are kept on disk because they are expensive to fetch and cheap to
    store: roughly 4 MB a day against 45 seconds of WRDS time. Any later change
    to the return convention -- the staleness rule, the bar labelling, simple
    against log returns -- can then be re-derived in seconds instead of
    re-pulling the whole sample.
    """
    if use_cache and taq_minute_bars_path(date).exists():
        return load_taq_minute_bars(date)

    from pull_taq_minute_bars import pull_taq_minute_bars

    # Asking TAQ only for the eligible symbols roughly halves the query time
    # (49s against 90s on 2012-06-01) and cuts the bars returned threefold.
    symbols = None
    if universe is not None:
        day = pd.Timestamp(date).normalize()
        symbols = sorted(universe.loc[universe["date"] == day, "sym_root"].unique())
        if not symbols:
            raise ValueError(f"no eligible stocks on {day.date()}")

    bars = pull_taq_minute_bars(date, symbols=symbols, db=db)
    write_taq_minute_bars(bars, date)
    return bars


def build_and_write_day(
    date,
    universe=None,
    db=None,
    max_staleness=MAX_STALENESS_MINUTES,
    use_cache=True,
):
    """Pull, clean and write one trading day of the panel.

    Parameters
    ----------
    date : str or datetime-like
        Trading day to build.
    universe : pandas.DataFrame, optional
        Eligible stock-days from `build_universe`. None keeps every symbol on
        the tape.
    db : wrds.Connection, optional
        Reuse an open connection across days.
    max_staleness : int
        Maximum age in minutes of a carried-forward base price.
    use_cache : bool
        Reuse bars already on disk instead of re-querying WRDS.

    Returns
    -------
    dict
        Row counts for a progress log.
    """
    bars = day_bars(date, universe=universe, db=db, use_cache=use_cache)
    n_all = bars["symbol"].nunique()
    if universe is not None:
        bars = restrict_to_universe(bars, universe, date)

    returns, coverage = build_day_panel(bars, date, max_staleness=max_staleness)
    write_day_panel(returns, coverage, date)
    return {
        "date": pd.Timestamp(date).date(),
        "symbols_on_tape": n_all,
        "symbols_in_universe": returns.shape[1],
        "returns": int(returns.notna().sum().sum()),
    }


def main():
    """Build the panel for one or more trading days."""
    parser = argparse.ArgumentParser(
        description="Build the one-minute return panel for given trading days."
    )
    parser.add_argument("dates", nargs="+", help="trading days, e.g. 2005-01-03")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="skip the universe screen and keep every symbol on the tape",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="re-query WRDS even when cached bars exist",
    )
    args = parser.parse_args()

    import wrds

    from build_universe import load_universe

    universe = None if args.all_symbols else load_universe()
    db = wrds.Connection(wrds_username=config("WRDS_USERNAME"))
    failures = []
    try:
        for date in args.dates:
            try:
                info = build_and_write_day(
                    date, universe=universe, db=db, use_cache=not args.no_cache
                )
            except Exception as exc:  # noqa: BLE001 - report and keep going
                # One bad day should not abandon the rest of a batch. The day is
                # reported and the process exits non-zero, so doit still treats
                # the run as failed and will retry just that day.
                failures.append((date, exc))
                print(f"{date}: FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            print(
                f"{info['date']}: {info['symbols_in_universe']:,} stocks, "
                f"{info['returns']:,} returns",
                flush=True,
            )
    finally:
        db.close()

    if failures:
        raise SystemExit(f"{len(failures)} of {len(args.dates)} days failed")


if __name__ == "__main__":
    main()
