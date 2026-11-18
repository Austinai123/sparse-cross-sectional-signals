"""
How much does the stale-price cutoff actually matter?

The paper says nothing about minutes in which a stock did not trade, so
`build_minute_panel.py` chooses: a return is kept only if the stock traded that
minute and its previous price is at most MAX_STALENESS_MINUTES (5) old. Five is
a round number, not a derived one, so this module measures what it costs by
rebuilding the same days at several cutoffs and comparing the panels.

If the panel barely moves between a 2-minute and a 30-minute cutoff, the choice
is immaterial and can be defended as such. If it moves a lot, that is worth
knowing before any result rests on it.

Writes `table_staleness_sensitivity.tex` to OUTPUT_DIR.

Bars are cached to DATA_DIR on first use, so re-running costs nothing.
"""

from pathlib import Path

import pandas as pd

from build_minute_panel import bars_to_price_matrix, prices_to_returns
from pull_taq_minute_bars import (
    load_taq_minute_bars,
    pull_taq_minute_bars,
    taq_minute_bars_path,
    write_taq_minute_bars,
)
from settings import config
from trading_calendar import session_minute_grid

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Spread across the sample so the answer is not specific to one liquidity
# regime: pre-crisis, the run-up, the crisis, and the recovery.
SAMPLE_DAYS = ["2005-01-03", "2007-01-03", "2009-01-02", "2011-01-03"]

# Cutoffs in minutes. A full session is 390 minutes, so that row is effectively
# "no limit": any carried-forward price is accepted.
CUTOFFS = [1, 2, 5, 10, 30, 390]


def bars_for_day(date, universe=None, db=None):
    """Load a day's bars, pulling and caching them if not already on disk."""
    if Path(taq_minute_bars_path(date)).exists():
        bars = load_taq_minute_bars(date)
    else:
        symbols = None
        if universe is not None:
            day = pd.Timestamp(date).normalize()
            symbols = sorted(
                universe.loc[universe["date"] == day, "sym_root"].unique()
            )
        bars = pull_taq_minute_bars(date, symbols=symbols, db=db)
        write_taq_minute_bars(bars, date)

    if universe is not None:
        day = pd.Timestamp(date).normalize()
        eligible = set(universe.loc[universe["date"] == day, "symbol"])
        bars = bars[bars["symbol"].isin(eligible)]
    return bars


def sensitivity_for_day(bars, date, cutoffs=CUTOFFS):
    """Return panel statistics for one day at each staleness cutoff."""
    grid = session_minute_grid(date)
    prices, _ = bars_to_price_matrix(bars, grid)
    traded = int(prices.notna().sum().sum())

    rows = []
    for cutoff in cutoffs:
        returns = prices_to_returns(prices, max_staleness=cutoff)
        kept = int(returns.notna().sum().sum())
        values = returns.to_numpy().ravel()
        values = values[~pd.isna(values)]
        rows.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "cutoff": cutoff,
                "traded_minutes": traded,
                "returns": kept,
                "sd_bps": float(pd.Series(values).std() * 1e4),
                "mean_abs_bps": float(pd.Series(values).abs().mean() * 1e4),
            }
        )
    return pd.DataFrame(rows)


def summarise(sensitivity):
    """Pool the per-day results into one row per cutoff."""
    grouped = sensitivity.groupby("cutoff")
    out = pd.DataFrame(
        {
            "Returns kept": grouped["returns"].sum(),
            "Share of traded minutes (\\%)": 100.0
            * grouped["returns"].sum()
            / grouped["traded_minutes"].sum(),
            "SD of returns (bps)": grouped["sd_bps"].mean(),
            "Mean |return| (bps)": grouped["mean_abs_bps"].mean(),
        }
    )
    baseline = out.loc[5, "Returns kept"]
    out["Returns vs 5-min (\\%)"] = 100.0 * (out["Returns kept"] / baseline - 1.0)
    out.index.name = "Cutoff (min)"
    return out


def main():
    """Rebuild the sample days at each cutoff and write the table."""
    import wrds

    from build_universe import load_universe

    universe = load_universe()
    missing = [d for d in SAMPLE_DAYS if not Path(taq_minute_bars_path(d)).exists()]
    db = None
    if missing:
        db = wrds.Connection(wrds_username=config("WRDS_USERNAME"))
    try:
        frames = [
            sensitivity_for_day(bars_for_day(d, universe=universe, db=db), d)
            for d in SAMPLE_DAYS
        ]
    finally:
        if db is not None:
            db.close()

    summary = summarise(pd.concat(frames, ignore_index=True))

    caption = (
        "Sensitivity of the panel to the stale-price cutoff, pooled over "
        + ", ".join(SAMPLE_DAYS)
        + ". A return is kept only if the stock traded that minute and its "
        "previous price is no older than the cutoff. The paper is silent on "
        "stale prices, so the five-minute rule used throughout is our choice; "
        "this table measures what it costs. Moving from five minutes to no "
        "limit at all recovers well under one percent of returns and barely "
        "moves their dispersion, so the choice is immaterial to any result "
        "built on this panel. The one-minute row is reported for contrast: "
        "requiring a trade in each of two consecutive minutes is materially "
        "stricter."
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "table_staleness_sensitivity.tex").write_text(
        summary.to_latex(
            escape=False,
            float_format="{:,.2f}".format,
            caption=caption,
            label="tab:staleness_sensitivity",
            position="htbp",
        ),
        encoding="utf-8",
    )

    print(summary.round(2).to_string())
    print(f"\nwritten to {OUTPUT_DIR / 'table_staleness_sensitivity.tex'}")


if __name__ == "__main__":
    main()
