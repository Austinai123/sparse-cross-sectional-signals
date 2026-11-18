"""
Coverage figure for the one-minute return panel.

Writes `figure_coverage.png` to OUTPUT_DIR: two stacked panels sharing a time
axis, showing how many stocks are in the sample on each day and how much of the
trading session those stocks actually traded in.

The point of the figure is to show that coverage has no *unexplained* breaks. A
gap means a day pulled badly or a screen misfiring, and that is easier to see in
a time series than in any table.

Both panels do move, and both movements are economic. The cross-section thins
through 2008 to a trough of 1,057 stocks in early 2009 because the paper's $5
screen removed 23.6% of NYSE common stocks on 2 March 2009 against 2.9% in
January 2005, and recovers with the market afterwards. The share of session
minutes traded rises from about 79% to 97% as algorithmic trading fills in
minutes that used to be quiet. Neither is a pipeline failure; a pipeline
failure would be a single day falling off a cliff while its neighbours did not.

The two panels are deliberately separate rather than one chart with two y-axes:
a stock count and a percentage share no common scale, and overlaying them on
twin axes invites false readings of where the lines cross.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from build_minute_panel import available_panel_dates, load_coverage
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Chart chrome, kept recessive so the data carries the figure.
SERIES = "#2a78d6"
BAND = "#9ec5f4"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def daily_coverage_stats(coverage):
    """Collapse per-stock coverage records to one row per trading day.

    Returns
    -------
    pandas.DataFrame
        Indexed by date, with the stock count and the 25th, 50th and 75th
        percentiles of the share of session minutes each stock traded in.
    """
    df = coverage.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["pct_traded"] = 100.0 * df["n_traded_minutes"] / df["n_session_minutes"]

    grouped = df.groupby("date")
    stats = pd.DataFrame(
        {
            "n_stocks": grouped.size(),
            "p25": grouped["pct_traded"].quantile(0.25),
            "p50": grouped["pct_traded"].quantile(0.50),
            "p75": grouped["pct_traded"].quantile(0.75),
        }
    )
    return stats.sort_index()


def _style_axis(ax):
    """Apply the recessive grid and axis treatment."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.6, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED_INK, labelsize=9, length=3, width=0.8)


def plot_coverage(stats, path):
    """Draw and save the two-panel coverage figure."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8.0, 6.0), sharex=True, height_ratios=[1, 1]
    )
    fig.patch.set_facecolor(SURFACE)

    # Panel A: how many stocks are in the sample.
    ax_top.plot(stats.index, stats["n_stocks"], color=SERIES, linewidth=1.6)
    ax_top.set_ylabel("Stocks in sample", color=SECONDARY_INK, fontsize=10)
    ax_top.set_title(
        "Stocks passing the sample screens on each trading day",
        color=PRIMARY_INK,
        fontsize=11,
        loc="left",
        pad=8,
    )
    ax_top.set_ylim(0, stats["n_stocks"].max() * 1.12)
    _style_axis(ax_top)

    # Panel B: how much of the session those stocks actually traded in.
    ax_bottom.fill_between(
        stats.index,
        stats["p25"],
        stats["p75"],
        color=BAND,
        linewidth=0,
        label="25th–75th percentile",
    )
    ax_bottom.plot(
        stats.index,
        stats["p50"],
        color=SERIES,
        linewidth=1.6,
        label="Median stock",
    )
    ax_bottom.set_ylabel("Session minutes traded (%)", color=SECONDARY_INK, fontsize=10)
    ax_bottom.set_title(
        "Share of the trading session in which a stock actually traded",
        color=PRIMARY_INK,
        fontsize=11,
        loc="left",
        pad=8,
    )
    ax_bottom.set_ylim(0, 102)
    _style_axis(ax_bottom)

    legend = ax_bottom.legend(
        loc="lower left", frameon=False, fontsize=9, handlelength=1.6
    )
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)

    ax_bottom.xaxis.set_major_locator(mdates.YearLocator())
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    """Build the coverage figure from whatever days are on disk."""
    dates = available_panel_dates()
    if not dates:
        raise SystemExit(
            "No panel days found. Run build_minute_panel.py for some dates first."
        )

    stats = daily_coverage_stats(load_coverage(dates))
    path = plot_coverage(stats, OUTPUT_DIR / "figure_coverage.png")

    print(f"{len(stats)} trading days plotted")
    print(stats.describe().round(1).to_string())
    print(f"written to {path}")


if __name__ == "__main__":
    main()
