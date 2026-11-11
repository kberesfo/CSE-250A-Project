"""data/load.py: calendar columns added by load_air_quality (#19 prerequisite)."""

import pandas as pd
import pytest

from cse250a.modules.config import config
from cse250a.modules.core import TIME_CARDS, Time
from cse250a.modules.data import load_air_quality


@pytest.fixture(scope="module")
def df():
    return load_air_quality(config.paths.air_quality_csv)


def test_calendar_columns_present_and_complete(df):
    for col in (Time.HOUR, Time.DOW, Time.IS_WEEKDAY):
        assert col in df.columns
        assert df[col].notna().all()


def test_dow_matches_timestamp(df):
    assert (df[Time.DOW] == df.index.dayofweek).all()
    assert set(df[Time.DOW].unique()) == set(range(TIME_CARDS[Time.DOW]))
    # 10 March 2004, the first day in the file, was a Wednesday
    assert df.loc[pd.Timestamp("2004-03-10 18:00"), Time.DOW] == 2


def test_is_weekday_is_monday_to_friday(df):
    assert set(df[Time.IS_WEEKDAY].unique()) == {0, 1}
    assert ((df[Time.DOW] < 5).astype(int) == df[Time.IS_WEEKDAY]).all()
    assert df.loc[pd.Timestamp("2004-03-13 12:00"), Time.IS_WEEKDAY] == 0  # Saturday
    assert df.loc[pd.Timestamp("2004-03-15 12:00"), Time.IS_WEEKDAY] == 1  # Monday


def test_calendar_values_fit_their_cards(df):
    for col in (Time.HOUR, Time.DOW, Time.IS_WEEKDAY):
        assert df[col].between(0, TIME_CARDS[col] - 1).all()
