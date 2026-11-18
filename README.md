Sparse Cross Sectional Signals
==============================

Replication and extension of Chinco, Clark-Joseph and Ye (2019), "Sparse Signals
in the Cross-Section of Returns," *Journal of Finance* 74(1), 449-492.

## What the paper claims

Hand the LASSO the entire cross-section of lagged one-minute returns, with no
economic prior about which stocks should matter, and ask it to forecast each
stock's return one minute ahead. It works: out-of-sample fit and
forecast-implied Sharpe ratios both rise, and the predictors it picks are not
noise -- they tend to be stocks with fresh news about fundamentals.

## Requirements

- Python 3.12
- A WRDS account with access to TAQ (`taqm_*`) and CRSP (`crsp.dsf_v2`)
- A LaTeX distribution on your PATH, for the report
  ([Windows](https://tug.org/texlive/windows.html#install),
  [Mac](https://tug.org/mactex/mactex-download.html))

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `WRDS_USERNAME` to your WRDS login.
Then create the WRDS password file once, interactively:

```bash
python -c "import wrds; wrds.Connection(wrds_username='YOUR_USERNAME')"
```

Enter your password and answer `y` when it offers to create `.pgpass`. The rest
of the pipeline runs unattended after this.

Note that `START_DATE`, `END_DATE` and `PANEL_FREQUENCY` are committed constants
in `src/settings.py`, deliberately *not* read from `.env`, so that two people
with different `.env` files who build the same commit get the same output.

## Running it

```bash
doit
```

That runs, in order:

| Task | What it does | Roughly |
|---|---|---|
| `pull:crsp_daily` | CRSP daily stock file and the CRSP-TAQ symbol link | 3 min |
| `universe` | Applies the paper's sample screens, day by day | 1 min |
| `minute_panel` | TAQ trades to one-minute returns, one task per trading day | 45 s per day |
| `panel_exhibits` | Summary tables and the coverage figure | seconds |
| `run_pytest` | Unit tests and data validation | seconds |

`PANEL_FREQUENCY = "monthly"` in `src/settings.py` builds one randomly chosen
trading day from each month, 96 days, which finishes in about an hour. The day
is drawn at random rather than taken as the first of the month, because
sampling the first trading day would select only turn-of-the-month days, which
carry a documented return and volume anomaly. The draw is seeded by
`PANEL_SAMPLE_SEED`, also a committed constant, so everyone gets the same days. Set it to `"all"` for
the paper's full 2,013-day sample; that is roughly 25 hours single-threaded and
is meant for a cluster job array, one task per day.

## Data sources

| Source | Table | Used for |
|---|---|---|
| TAQ (WRDS) | `taqm_<year>.ctm_<yyyymmdd>` | Trades, aggregated to one-minute bars |
| CRSP (WRDS) | `crsp.dsf_v2` | Sample screens, and the volume reconciliation |
| CRSP-TAQ link (WRDS) | `wrdsapps_link_crsp_taqm.taqmclink` | Share class for dual-class tickers |

## Sample definition

Section II.A of the paper: NYSE-listed stocks, every trading day from January
2005 to December 2012, dropping stocks priced below $5 at the previous day's
close. Our screens and what each removes are in
`_output/table_universe_attrition.tex`.

## Choices the paper does not make for us

The paper is silent on several things that materially affect the panel. Each is
recorded here, implemented in one place, and measured rather than assumed.

**Minute bars.** Bar `t` is the last trade price in `[t, t+1)`, labelled by its
starting minute. A full session is 390 bars (09:30-15:59) and 389 returns.
Labelling bars by their closing minute instead would shift every return by one
minute. Returns are simple, not log, and nothing is carried across days, so the
overnight return never enters.

**Minutes with no trade.** Prices are carried forward within the day, never
backward. A return is kept only if the stock actually traded in that minute --
a quiet minute is recorded as missing, not as a zero return -- and only if the
previous price is at most `MAX_STALENESS_MINUTES` (5) old. Past that the
"one-minute" return really spans several minutes. Five is a round number rather
than a derived one, so `table_staleness_sensitivity.tex` measures what it costs:
dropping the cutoff entirely recovers 0.67% of returns and moves their standard
deviation by 1.6%, so the choice is immaterial. Tightening it to one minute is
materially stricter, costing 4% of returns.

**Trade filters.** The paper does not state its trade filters. We keep
correction code `00` and drop about twenty sale conditions: auction prints,
average-price trades, next-day settlement, out-of-sequence reports and
extended-hours trading. Excluding the closing auction print is how the paper's
instruction to drop the last minute of each day is implemented.

**Matching CRSP to TAQ.** The primary key is the CRSP ticker, which equals TAQ's
`sym_root` for 99.9% of eligible stock-days. WRDS's daily link table is *not*
used as the primary match: it omits whole categories of stock, including every
REIT, and using it drops about 9% of the sample. Dual-class stocks, where two
permnos share one ticker, are dropped (1.7% of stock-days) rather than guessed
at, because the link table's share-class suffix can name a preferred issue --
Constellation Brands resolves to `STZ PRA`, whose volume is 1% of CRSP's.

**Universe breadth.** We restrict to ordinary common stock, which gives about
1,350 stocks a day. The paper says "roughly N = 2,000 NYSE-listed stocks,"
suggesting it also includes ADRs and closed-end funds. The screens are a single
dict at the top of `src/build_universe.py` if you want to widen them.

## Caching

Minute bars are kept in `_data/taq_bars/` as they are pulled, about 4 MB a day
against 45 seconds of WRDS time. Changing the return convention -- the
staleness rule, the bar labelling, simple against log returns -- then re-derives
the panel from cached bars in seconds rather than re-pulling the sample. Pass
`--no-cache` to `run_minute_panel.py` to force a fresh query.

## Validation

`src/test_validate_panel.py` checks the data against CRSP rather than against
our own code. The strongest check is an identity: because our trade filters keep
a strict subset of the consolidated tape, per-stock TAQ volume can never exceed
CRSP's. For IBM on 2005-01-03 the whole tape less corrected trades reconciles to
CRSP's daily volume exactly, 5,301,400 shares. Across the sample the median
TAQ/CRSP volume ratio is 0.94, the shortfall being the auction and
irregular-condition volume we remove on purpose.

These tests skip, rather than fail, when the pulled data is absent, which is how
they behave in CI where there are no WRDS credentials.

## Code layout

| File | Purpose |
|---|---|
| `src/settings.py` | Paths, credentials, and the committed analysis constants |
| `src/pull_crsp_daily.py` | CRSP daily file and the CRSP-TAQ link |
| `src/build_universe.py` | Sample screens and the filter attrition record |
| `src/trading_calendar.py` | NYSE session times and the sampled trading days |
| `src/pull_taq_minute_bars.py` | One day of TAQ trades, aggregated to minute bars in SQL |
| `src/build_minute_panel.py` | Bars to one-minute returns, plus per-day coverage |
| `src/run_minute_panel.py` | Driver: pull, screen, clean and write one day |
| `src/table_staleness_sensitivity.py` | What the stale-price cutoff costs |
| `src/table_panel_summary.py` | Summary and attrition tables |
| `src/figure_coverage.py` | Coverage figure |
| `src/test_*.py` | Unit tests and data validation |

Data cleaning (`pull_*`, `build_*`) is kept separate from analysis
(`table_*`, `figure_*`).

### General Directory Structure

 - The `assets` folder is used for things like hand-drawn figures or other
   pictures that were not generated from code. These things cannot be easily
   recreated if they are deleted.

 - The `_output` folder, on the other hand, contains dataframes and figures that are
   generated from code. The entire folder should be able to be deleted, because
   the code can be run again, which would again generate all of the contents.

 - The `data_manual` is for data that cannot be easily recreated. This data
   should be version controlled. Anything in the `_data` folder or in
   the `_output` folder should be able to be recreated by running the code
   and can safely be deleted.

 - I'm using the `doit` Python module as a task runner. It works like `make` and
   the associated `Makefile`s. To rerun the code, install `doit`
   (https://pydoit.org/) and execute the command `doit` from the `src`
   directory. Note that doit is very flexible and can be used to run code
   commands from the command prompt, thus making it suitable for projects that
   use scripts written in multiple different programming languages.

 - I'm using the `.env` file as a container for absolute paths that are private
   to each collaborator in the project. You can also use it for private
   credentials, if needed. It should not be tracked in Git.

### Data and Output Storage

I'll often use a separate folder for storing data. Any data in the data folder
can be deleted and recreated by rerunning the PyDoit command (the pulls are in
the dodo.py file). Any data that cannot be automatically recreated should be
stored in the "data_manual" folder. Because of the risk of manually-created data
getting changed or lost, I prefer to keep it under version control if I can.
Thus, data in the "_data" folder is excluded from Git (see the .gitignore file),
while the "data_manual" folder is tracked by Git.

Output is stored in the "_output" directory. This includes dataframes, charts, and
rendered notebooks. When the output is small enough, I'll keep this under
version control. I like this because I can keep track of how dataframes change as my
analysis progresses, for example.

Of course, the _data directory and _output directory can be kept elsewhere on the
machine. To make this easy, I always include the ability to customize these
locations by defining the path to these directories in environment variables,
which I intend to be defined in the `.env` file, though they can also simply be
defined on the command line or elsewhere. The `settings.py` is responsible for
loading these environment variables and doing some preprocessing on them.
The `settings.py` file is the entry point for all other scripts to these
definitions. That is, all code that references these variables and others are
loaded by importing `config`.

### Naming Conventions

 - **`pull_` vs `load_`**: Files or functions that pull data from an external
 data source are prepended with "pull_", as in "pull_fred.py". Functions that
 load data that has been cached in the "_data" folder are prepended with "load_".
 For example, inside of the `pull_CRSP_Compustat.py` file there is both a
 `pull_compustat` function and a `load_compustat` function. The first pulls from
 the web, whereas the other loads cached data from the "_data" directory.


### Dependencies and Virtual Environments

#### Working with `pip` requirements

This project uses `pip` with a virtual environment. Install requirements with:
```bash
pip install -r requirements.txt
```

To update the requirements file after adding new packages:
```bash
pip freeze > requirements.txt
```

