"""
test_feature_engineering.py

Unit tests for FeatureEngineer.

Tests:
- Lag Features
- Rolling Features
- Calendar Features
- Master Feature Pipeline

Author: Nani
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.preprocessing.feature_engineering import (
    FeatureEngineer
)


@pytest.fixture
def sample_data():
    """
    Create realistic weekly sales data.
    """

    np.random.seed(42)

    dates = pd.date_range(
        start="2020-01-05",
        periods=120,
        freq="W"
    )

    sales = (
        1000
        + np.arange(120) * 5
        + 100 * np.sin(
            2 * np.pi * np.arange(120) / 52
        )
        + np.random.normal(
            0,
            20,
            120
        )
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Weekly_Sales": sales
        }
    )


def test_lag_features(sample_data):
    """
    Verify lag features are created.
    """

    engineer = FeatureEngineer()

    df = engineer.create_lag_features(
        sample_data
    )

    expected_columns = [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "lag_12",
        "lag_26",
        "lag_52"
    ]

    for col in expected_columns:
        assert col in df.columns


def test_rolling_features(sample_data):
    """
    Verify rolling features are created.
    """

    engineer = FeatureEngineer()

    df = engineer.create_rolling_features(
        sample_data
    )

    expected_columns = [
        "rolling_mean_4",
        "rolling_mean_12",
        "rolling_mean_26",
        "rolling_std_4",
        "rolling_std_12"
    ]

    for col in expected_columns:
        assert col in df.columns


def test_calendar_features(sample_data):
    """
    Verify calendar features are created.
    """

    engineer = FeatureEngineer()

    df = engineer.create_calendar_features(
        sample_data
    )

    expected_columns = [
        "year",
        "month",
        "quarter",
        "week_of_year"
    ]

    for col in expected_columns:
        assert col in df.columns


def test_create_features(sample_data):
    """
    Verify master pipeline creates
    all expected features.
    """

    engineer = FeatureEngineer()

    df = engineer.create_features(
        sample_data
    )

    expected_columns = [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "lag_12",
        "lag_26",
        "lag_52",
        "rolling_mean_4",
        "rolling_mean_12",
        "rolling_mean_26",
        "rolling_std_4",
        "rolling_std_12",
        "year",
        "month",
        "quarter",
        "week_of_year"
    ]

    for col in expected_columns:
        assert col in df.columns


def test_no_nulls_after_create_features(
    sample_data
):
    """
    Verify create_features()
    removes NaNs.
    """

    engineer = FeatureEngineer()

    df = engineer.create_features(
        sample_data
    )

    assert df.isnull().sum().sum() == 0


def test_output_not_empty(sample_data):
    """
    Verify output contains rows.
    """

    engineer = FeatureEngineer()

    df = engineer.create_features(
        sample_data
    )

    assert len(df) > 0


def test_output_smaller_than_input(
    sample_data
):
    """
    Verify rows are dropped because
    of lag/rolling calculations.
    """

    engineer = FeatureEngineer()

    df = engineer.create_features(
        sample_data
    )

    assert len(df) < len(sample_data)

def test_lag_1_correctness(sample_data):
    """
    Verify lag_1 uses previous row value.
    """

    engineer = FeatureEngineer()

    df = engineer.create_lag_features(
        sample_data
    )

    assert (
        df["lag_1"].iloc[1]
        ==
        df["Weekly_Sales"].iloc[0]
    )