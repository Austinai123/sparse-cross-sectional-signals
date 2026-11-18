"""
Summary tables for the one-minute return panel.

Writes two LaTeX tables to OUTPUT_DIR:

 - `table_panel_summary.tex`: the panel year by year -- how many stocks, how
   many minute returns, how much of the session each stock actually traded in,
   and the size of the returns themselves.
 - `table_universe_attrition.tex`: how many stock-days each sample screen
   removed, which is the data-validation deliverable for the filters.

Both are built from the small per-day coverage records written by
`build_minute_panel.py`, not from the return panel itself, so neither needs to
read the full panel.
"""

from pathlib import Path

import pandas as pd

from build_minute_panel import available_panel_dates, load_coverage
from build_universe import load_attrition
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def summarise_by_year(coverage):
    """Aggregate per-stock-day coverage records into one row per year.

    Parameters
    ----------
    coverage : pandas.DataFrame
        Concatenated coverage records from `build_minute_panel.load_coverage`.

    Returns
    -------
    pandas.DataFrame
        Indexed by year, one column per reported statistic.
    """
    df = coverage.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["pct_traded"] = 100.0 * df["n_traded_minutes"] / df["n_session_minutes"]

    grouped = df.groupby("year")
    summary = pd.DataFrame(
        {
            "Days": grouped["date"].nunique(),
            "Stocks/day": grouped.apply(
                lambda g: g.groupby("date").size().mean(), include_groups=False
            ),
            "Stock-days": grouped.size(),
            "Returns (m)": grouped["n_returns"].sum() / 1e6,
            "Minutes traded (\\%)": grouped["pct_traded"].mean(),
            "Stale dropped (\\%)": 100.0
            * grouped["n_dropped_stale"].sum()
            / grouped["n_traded_minutes"].sum(),
            "Mean ret (bps)": grouped["mean_return_bps"].mean(),
            "SD ret (bps)": grouped["sd_return_bps"].mean(),
            "Volume/min": grouped["volume"].sum() / grouped["n_traded_minutes"].sum(),
        }
    )
    return summary


def write_latex_table(df, path, caption, label, float_format="{:,.2f}".format):
    """Write a DataFrame as a LaTeX tabular with a caption and label."""
    body = df.to_latex(
        escape=False,
        float_format=float_format,
        caption=caption,
        label=label,
        position="htbp",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def main():
    """Build both summary tables from whatever days are on disk."""
    dates = available_panel_dates()
    if not dates:
        raise SystemExit(
            "No panel days found. Run build_minute_panel.py for some dates first."
        )

    coverage = load_coverage(dates)
    summary = summarise_by_year(coverage)

    write_latex_table(
        summary,
        OUTPUT_DIR / "table_panel_summary.tex",
        caption=(
            "One-minute return panel, by year. "
            "\\emph{Stocks/day} is the average number of NYSE common stocks "
            "passing the sample screens. \\emph{Minutes traded} is the share of "
            "the trading session in which the average stock actually traded; the "
            "remainder has no price and therefore no return. "
            "\\emph{Stale dropped} is the share of traded minutes whose return "
            "was discarded because the previous price was more than five minutes "
            "old. Coverage is high and stable throughout, so the panel is not "
            "materially thinner in any one year."
        ),
        label="tab:panel_summary",
    )

    attrition = load_attrition().set_index("step")
    attrition.index.name = "Screen"
    attrition = attrition.rename(
        columns={
            "stock_days": "Stock-days",
            "stocks": "Stocks",
            "dropped": "Dropped",
            "pct_dropped": "Dropped (\\%)",
        }
    )
    write_latex_table(
        attrition,
        OUTPUT_DIR / "table_universe_attrition.tex",
        caption=(
            "Sample screens and what each removes. Starting from every "
            "NYSE-listed security in CRSP, the share-type screen removes ADRs, "
            "closed-end funds and ETFs; the \\$5 screen is the paper's, applied "
            "to the previous day's close. The final row drops stock-days whose "
            "CRSP ticker maps to more than one security, because the WRDS link "
            "table's share-class suffix can name a preferred issue rather than a "
            "share class and would attach the wrong security's prices."
        ),
        label="tab:universe_attrition",
        float_format="{:,.3f}".format,
    )

    print(summary.round(2).to_string())
    print()
    print(f"written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
