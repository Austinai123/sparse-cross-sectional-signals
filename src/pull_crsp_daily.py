"""
Pull the CRSP daily stock file and the CRSP-TAQ symbol link.

CRSP defines *which* stocks are in the sample; TAQ supplies their prices. This
module pulls both pieces and writes them to disk. The filters that turn them
into the paper's universe live in `build_universe.py`.

Two tables:
 - `crsp.dsf_v2` is the CRSP daily stock file in CIZ format. `dlyprevprc` is
   the previous trading day's price, which is exactly the paper's $5 screen.
 - `wrdsapps_link_crsp_taqm.taqmclink` maps permno to a TAQ sym_root/sym_suffix
   for each day, which is how a CRSP stock is matched to its TAQ prices. It
   starts 2003-09-10.

Only `primaryexch = 'N'` (NYSE) is filtered in SQL, because that is the paper's
stated scope. Share-type filters are applied in `build_universe.py` so they
show up in the filter attrition table rather than disappearing into a WHERE
clause.
"""

from pathlib import Path

import pandas as pd
import wrds

from settings import END_DATE, START_DATE, config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME", default="")

CRSP_DAILY_COLUMNS = """
    permno, dlycaldt, ticker, primaryexch, sharetype, securitytype,
    securitysubtype, issuertype, usincflg, siccd,
    dlyprc, dlyprevprc, dlyclose, dlyopen, dlyvol, dlynumtrd,
    dlycap, shrout, dlyret, dlydelflg
"""


def _connect(wrds_username, db):
    """Open a WRDS connection unless one was passed in."""
    if db is not None:
        return db, False
    if not wrds_username:
        raise ValueError(
            "No WRDS username. Set WRDS_USERNAME in .env (see .env.example) "
            "or pass wrds_username=."
        )
    return wrds.Connection(wrds_username=wrds_username), True


def pull_crsp_daily(
    start_date=START_DATE,
    end_date=END_DATE,
    exchange="N",
    wrds_username=WRDS_USERNAME,
    db=None,
):
    """Pull NYSE rows of the CRSP daily stock file.

    Parameters
    ----------
    start_date, end_date : datetime-like
        Inclusive date range. Defaults are the committed constants in
        settings.py.
    exchange : str
        CRSP `primaryexch` code. 'N' is NYSE.
    wrds_username : str
        WRDS login, read from .env by default.
    db : wrds.Connection, optional
        Reuse an open connection.

    Returns
    -------
    pandas.DataFrame
        One row per stock-day.
    """
    query = f"""
        SELECT {CRSP_DAILY_COLUMNS}
        FROM crsp.dsf_v2
        WHERE dlycaldt BETWEEN '{pd.Timestamp(start_date).date()}'
                           AND '{pd.Timestamp(end_date).date()}'
          AND primaryexch = '{exchange}'
        ORDER BY dlycaldt, permno
    """
    db, close_when_done = _connect(wrds_username, db)
    try:
        df = db.raw_sql(query, date_cols=["dlycaldt"])
    finally:
        if close_when_done:
            db.close()

    df["permno"] = df["permno"].astype("int64")
    return df


def pull_taq_crsp_link(
    start_date=START_DATE,
    end_date=END_DATE,
    wrds_username=WRDS_USERNAME,
    db=None,
):
    """Pull the daily CRSP-to-TAQ symbol link.

    `match_lvl` records how the match was made; level 1 is a direct CUSIP
    match. The column is kept so `build_universe.py` can report it rather than
    trusting every match equally.

    Returns
    -------
    pandas.DataFrame
        Columns date, sym_root, sym_suffix, permno, ticker, match_lvl.
    """
    query = f"""
        SELECT date, sym_root, sym_suffix, permno, ticker, match_lvl
        FROM wrdsapps_link_crsp_taqm.taqmclink
        WHERE date BETWEEN '{pd.Timestamp(start_date).date()}'
                       AND '{pd.Timestamp(end_date).date()}'
        ORDER BY date, permno
    """
    db, close_when_done = _connect(wrds_username, db)
    try:
        df = db.raw_sql(query, date_cols=["date"])
    finally:
        if close_when_done:
            db.close()

    df["permno"] = df["permno"].astype("int64")
    df["sym_root"] = df["sym_root"].fillna("").astype(str).str.strip()
    df["sym_suffix"] = df["sym_suffix"].fillna("").astype(str).str.strip()
    df["symbol"] = (df["sym_root"] + " " + df["sym_suffix"]).str.strip()
    return df


def crsp_daily_path(data_dir=DATA_DIR):
    """Path the CRSP daily file is written to."""
    return Path(data_dir) / "crsp_daily.parquet"


def taq_crsp_link_path(data_dir=DATA_DIR):
    """Path the CRSP-TAQ link is written to."""
    return Path(data_dir) / "taq_crsp_link.parquet"


def load_crsp_daily(data_dir=DATA_DIR):
    """Read back the CRSP daily file."""
    return pd.read_parquet(crsp_daily_path(data_dir))


def load_taq_crsp_link(data_dir=DATA_DIR):
    """Read back the CRSP-TAQ link."""
    return pd.read_parquet(taq_crsp_link_path(data_dir))


def main():
    """Pull both tables and write them to DATA_DIR."""
    db, _ = _connect(WRDS_USERNAME, None)
    try:
        for name, puller, path in [
            ("CRSP daily", pull_crsp_daily, crsp_daily_path()),
            ("CRSP-TAQ link", pull_taq_crsp_link, taq_crsp_link_path()),
        ]:
            df = puller(db=db)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            print(f"{name}: {len(df):,} rows written to {path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
