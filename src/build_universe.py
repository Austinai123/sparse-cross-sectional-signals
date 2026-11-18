"""
Apply the paper's sample filters to decide which stocks are in on each day.

Section II.A of Chinco, Clark-Joseph and Ye (2019):
    "We obtain data on the one-minute returns of NYSE-listed stocks from the
     Trade and Quote (TAQ) database. Our sample includes every trading day from
     January 2005 to December 2012. We remove stocks with prices below $5 at
     the previous day's close."

So three screens: NYSE-listed, common stock, and a previous close of at least
$5. A fourth is ours rather than the paper's: a stock is only usable if CRSP
can be matched to a TAQ symbol, because otherwise there are no prices to use.

Every screen is applied in sequence and counted, so `filter_attrition` reports
how many stock-days each one removes. That table is a deliverable: a screen
that removes far more or far less than expected is how you find out a filter is
wrong.
"""

from pathlib import Path

import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

MIN_PREVIOUS_CLOSE = 5.0

# Dual-class stocks share one CRSP ticker across two permnos, and TAQ separates
# them by sym_suffix. The obvious fix -- take the suffix from the WRDS link
# table -- is wrong: that suffix can name a *preferred* issue rather than a
# share class. Constellation Brands resolves to 'STZ PRA', whose TAQ volume on
# 2005-01-03 is 5,400 shares against CRSP's 535,200, because PR means preferred.
# Rather than silently attach the wrong security's prices, these stock-days are
# dropped. They are 1.7% of the sample, and the attrition table reports them.
# Set True only with a volume reconciliation in place to verify each match.
RESOLVE_DUAL_CLASS = False

# CRSP CIZ codes for ordinary US common stock, the equivalent of share codes
# 10 and 11 in the older SIZ format. REIT is included alongside CORP because
# REITs are ordinary NYSE-listed common shares.
COMMON_STOCK_FILTERS = {
    "sharetype": ["NS"],
    "securitytype": ["EQTY"],
    "securitysubtype": ["COM"],
    "usincflg": ["Y"],
    "issuertype": ["CORP", "REIT"],
}

UNIVERSE_COLUMNS = [
    "date",
    "permno",
    "symbol",
    "sym_root",
    "sym_suffix",
    "ticker",
    "prev_close",
    "market_cap",
    "volume",
    "siccd",
    "match_method",
]


def apply_common_stock_filter(crsp, filters=COMMON_STOCK_FILTERS):
    """Keep ordinary common stock, dropping ADRs, funds and foreign issuers."""
    mask = pd.Series(True, index=crsp.index)
    for column, allowed in filters.items():
        mask &= crsp[column].isin(allowed)
    return crsp[mask]


def apply_price_filter(crsp, min_previous_close=MIN_PREVIOUS_CLOSE):
    """Keep stocks whose previous close was at least `min_previous_close`.

    CRSP reports a negative price when the value is a bid-ask midpoint rather
    than a trade. The sign is a flag, not a price, so compare on magnitude.
    Rows with no previous price at all (a stock's first day) are dropped.
    """
    prev = pd.to_numeric(crsp["dlyprevprc"], errors="coerce").abs()
    return crsp[prev.notna() & (prev >= min_previous_close)]


def attach_taq_symbols(crsp, link, resolve_dual_class=RESOLVE_DUAL_CLASS):
    """Map each CRSP stock-day to its TAQ symbol.

    The primary key is the CRSP ticker, which equals TAQ's `sym_root` for
    99.9% of eligible NYSE stock-days. WRDS also ships a permno-to-symbol link
    table, but its daily version is missing whole categories of stock -- every
    REIT, including names as large as Simon Property and AvalonBay -- so using
    it as the primary match silently drops about 9% of the sample.

    The ticker is not unique for dual-class stocks: Berkshire, Brown-Forman,
    Molson Coors and about a dozen others have two permnos sharing one ticker,
    which TAQ separates by `sym_suffix`. Those are 1.7% of stock-days and are
    dropped by default; see RESOLVE_DUAL_CLASS.

    Adds `match_method` so the attrition table can report how each stock-day
    was matched rather than treating all matches as equally reliable.
    """
    crsp = crsp.copy()
    crsp["ticker"] = crsp["ticker"].fillna("").astype(str).str.strip()

    # A ticker is ambiguous on a day if more than one permno claims it.
    per_ticker = (
        crsp[crsp["ticker"] != ""]
        .groupby(["dlycaldt", "ticker"])["permno"]
        .transform("nunique")
    )
    crsp["ticker_is_unique"] = per_ticker.reindex(crsp.index).fillna(0).eq(1)

    # Share class from the link table, for the ambiguous ones. The link is only
    # a tie-breaker, so an empty or absent one must still leave ticker matching
    # working rather than raising.
    if link is None or link.empty or "sym_suffix" not in link.columns:
        crsp["link_suffix"] = ""
    else:
        suffix = link[["date", "permno", "sym_suffix"]].rename(
            columns={"date": "dlycaldt", "sym_suffix": "link_suffix"}
        )
        crsp = crsp.merge(suffix, on=["dlycaldt", "permno"], how="left")
    crsp["link_suffix"] = crsp["link_suffix"].fillna("").astype(str).str.strip()

    unique = crsp["ticker_is_unique"]
    resolved = (
        ~unique & (crsp["link_suffix"] != "")
        if resolve_dual_class
        else pd.Series(False, index=crsp.index)
    )

    crsp["sym_root"] = crsp["ticker"]
    crsp["sym_suffix"] = ""
    crsp.loc[resolved, "sym_suffix"] = crsp.loc[resolved, "link_suffix"]

    crsp["match_method"] = "unresolved"
    crsp.loc[unique, "match_method"] = "ticker"
    crsp.loc[resolved, "match_method"] = "ticker+share_class"

    crsp["symbol"] = (crsp["sym_root"] + " " + crsp["sym_suffix"]).str.strip()
    crsp.loc[crsp["match_method"] == "unresolved", "symbol"] = ""
    return crsp.drop(columns=["ticker_is_unique", "link_suffix"])


def build_universe(
    crsp,
    link,
    min_previous_close=MIN_PREVIOUS_CLOSE,
    common_stock_filters=COMMON_STOCK_FILTERS,
):
    """Build the eligible universe and the filter attrition record.

    Parameters
    ----------
    crsp : pandas.DataFrame
        CRSP daily stock file from `pull_crsp_daily`, already restricted to
        NYSE.
    link : pandas.DataFrame
        CRSP-TAQ link from `pull_taq_crsp_link`.
    min_previous_close : float
        The paper's $5 screen.
    common_stock_filters : dict
        CRSP CIZ codes defining ordinary common stock.

    Returns
    -------
    universe : pandas.DataFrame
        One row per eligible stock-day, columns UNIVERSE_COLUMNS.
    attrition : pandas.DataFrame
        One row per screen, with the stock-days and stocks surviving it.
    """
    steps = [("NYSE-listed stock-days (CRSP)", crsp)]

    common = apply_common_stock_filter(crsp, filters=common_stock_filters)
    steps.append(("Ordinary common stock", common))

    priced = apply_price_filter(common, min_previous_close=min_previous_close)
    steps.append((f"Previous close >= ${min_previous_close:,.0f}", priced))

    has_ticker = priced[priced["ticker"].notna() & (priced["ticker"] != "")]
    steps.append(("Has a CRSP ticker", has_ticker))

    linked = attach_taq_symbols(has_ticker, link)
    matched = linked[linked["symbol"] != ""]
    steps.append(("Matched to a TAQ symbol", matched))

    attrition = filter_attrition(steps)

    universe = matched.rename(
        columns={
            "dlycaldt": "date",
            "dlyprevprc": "prev_close",
            "dlycap": "market_cap",
            "dlyvol": "volume",
        }
    )
    universe["prev_close"] = universe["prev_close"].abs()
    universe = universe[UNIVERSE_COLUMNS].sort_values(["date", "permno"])
    return universe.reset_index(drop=True), attrition


def filter_attrition(steps):
    """Count what each screen removed.

    Parameters
    ----------
    steps : list of (str, DataFrame)
        Screen name and the data surviving it, in order.

    Returns
    -------
    pandas.DataFrame
        Columns: step, stock_days, stocks, dropped, pct_dropped.
    """
    rows = []
    previous = None
    for name, frame in steps:
        n = len(frame)
        rows.append(
            {
                "step": name,
                "stock_days": n,
                "stocks": frame["permno"].nunique(),
                "dropped": 0 if previous is None else previous - n,
                "pct_dropped": 0.0
                if previous in (None, 0)
                else 100.0 * (previous - n) / previous,
            }
        )
        previous = n
    return pd.DataFrame(rows)


def universe_path(data_dir=DATA_DIR):
    """Path the universe is written to."""
    return Path(data_dir) / "universe.parquet"


def attrition_path(data_dir=DATA_DIR):
    """Path the filter attrition record is written to."""
    return Path(data_dir) / "universe_attrition.parquet"


def load_universe(data_dir=DATA_DIR):
    """Read back the universe."""
    return pd.read_parquet(universe_path(data_dir))


def load_attrition(data_dir=DATA_DIR):
    """Read back the filter attrition record."""
    return pd.read_parquet(attrition_path(data_dir))


def symbols_on(date, universe):
    """TAQ symbols eligible on one trading day, for restricting a TAQ pull."""
    day = pd.Timestamp(date).normalize()
    return sorted(universe.loc[universe["date"] == day, "sym_root"].unique())


def main():
    """Build the universe from the cached CRSP pulls and write it to DATA_DIR."""
    from pull_crsp_daily import load_crsp_daily, load_taq_crsp_link

    crsp = load_crsp_daily()
    link = load_taq_crsp_link()
    universe, attrition = build_universe(crsp, link)

    universe_path().parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(universe_path(), index=False)
    attrition.to_parquet(attrition_path(), index=False)

    print(attrition.to_string(index=False))
    print()
    per_day = universe.groupby("date")["permno"].size()
    print(
        f"universe: {len(universe):,} stock-days, "
        f"{universe['permno'].nunique():,} stocks, "
        f"{per_day.mean():.0f} stocks per day on average"
    )
    print(f"written to {universe_path()}")


if __name__ == "__main__":
    main()
