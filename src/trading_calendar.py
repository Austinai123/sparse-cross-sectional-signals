"""
NYSE session times, in one place.

Both the TAQ pull and the panel build need to know when a trading session
starts and ends, and they have to agree. Hardcoding 09:30-16:00 in the pull is
wrong on the 16 half-days in the paper's sample, when the NYSE closes at 13:00:
TAQ then returns trades stamped after the close, which the panel build
correctly refuses because they fall outside the session grid.

Two days in the paper's sample are neither a full session nor a scheduled half
day, and both are real: on 2005-06-01 the NYSE closed at 15:56 after a technical
problem, and on 2006-12-27 it opened two minutes late. Reading the session from
the exchange calendar handles all of these without special cases.
"""

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from settings import (
    PANEL_FREQUENCY,
    PANEL_SAMPLE_SEED,
    PAPER_END_DATE,
    PAPER_START_DATE,
)

EXCHANGE_CALENDAR = "NYSE"
EXCHANGE_TIMEZONE = "America/New_York"


def session_bounds(date, calendar_name=EXCHANGE_CALENDAR):
    """Return (open, close) for one trading day as naive local timestamps.

    Raises
    ------
    ValueError
        If `date` is not a trading day on this calendar.

    Examples
    --------
    >>> open_, close = session_bounds("2005-01-03")
    >>> close.strftime("%H:%M")
    '16:00'
    >>> session_bounds("2006-07-03")[1].strftime("%H:%M")
    '13:00'
    """
    day = pd.Timestamp(date).normalize()
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(start_date=day, end_date=day)
    if schedule.empty:
        raise ValueError(f"{day.date()} is not a {calendar_name} trading day")

    row = schedule.iloc[0]
    bounds = []
    for field in ("market_open", "market_close"):
        stamp = pd.Timestamp(row[field])
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(EXCHANGE_TIMEZONE).tz_localize(None)
        bounds.append(stamp)
    return bounds[0], bounds[1]


def session_minute_grid(date, calendar_name=EXCHANGE_CALENDAR):
    """Return the one-minute bar grid for one trading day.

    Bars are labelled by their starting minute and run from the open through
    the minute before the close, so a full session gives 390 bars and a
    scheduled early close gives 210.

    Examples
    --------
    >>> len(session_minute_grid("2005-01-03"))
    390
    >>> len(session_minute_grid("2006-07-03"))
    210
    """
    open_time, close_time = session_bounds(date, calendar_name=calendar_name)
    return pd.date_range(open_time, close_time - pd.Timedelta(minutes=1), freq="1min")


def trading_days(start_date, end_date, calendar_name=EXCHANGE_CALENDAR):
    """Every trading day in a date range."""
    calendar = mcal.get_calendar(calendar_name)
    schedule = calendar.schedule(
        start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date)
    )
    return pd.Series(pd.DatetimeIndex(schedule.index).normalize())


def panel_dates(frequency=None, start_date=None, end_date=None, seed=None):
    """Trading days the minute panel is built for.

    Derived from the exchange calendar rather than from any pulled data, so the
    list is identical for everyone building this commit and `dodo.py` can read
    it before the data exists.

    Parameters
    ----------
    frequency : {'monthly', 'all'}
        'monthly' takes one randomly chosen trading day from each month; 'all'
        takes every trading day. Defaults to PANEL_FREQUENCY in settings.py.
    start_date, end_date : datetime-like, optional
        Default to the paper's sample window.
    seed : int, optional
        Seed for the within-month draw. Defaults to PANEL_SAMPLE_SEED.

    Returns
    -------
    list of pandas.Timestamp

    Notes
    -----
    The within-month day is drawn at random rather than taken as the first
    trading day. Sampling the first of each month would select only
    turn-of-the-month days, which carry a well-documented return and volume
    anomaly, and any statistic computed on the sample would inherit it.
    """
    frequency = PANEL_FREQUENCY if frequency is None else frequency
    start_date = PAPER_START_DATE if start_date is None else start_date
    end_date = PAPER_END_DATE if end_date is None else end_date

    days = trading_days(start_date, end_date)
    if frequency == "all":
        return list(days)
    if frequency == "monthly":
        seed = PANEL_SAMPLE_SEED if seed is None else seed
        rng = np.random.default_rng(seed)
        picks = []
        # sort=True so the draw order, and therefore the sample, is identical
        # on every machine.
        for _, group in days.groupby([days.dt.year, days.dt.month], sort=True):
            picks.append(group.iloc[int(rng.integers(len(group)))])
        return picks
    raise ValueError(f"unknown frequency {frequency!r}; use 'monthly' or 'all'")
