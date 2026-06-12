"""
test_prophet.py

Unit tests for ProphetForecaster.

Tests:
- Training
- Prediction
- Parameter extraction
- Model persistence

Author: Nani
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.forecasting.prophet import ( ProphetForecaster)


@pytest.fixture(scope="module")
def sample_data():
    """
    Generate realistic weekly sales data.

    Includes:
    - Trend
    - Seasonality
    - Noise
    """

    np.random.seed(42)

    dates = pd.date_range(  start="2020-01-05",  periods=120, freq="W" )

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
        { "Date": dates,"Weekly_Sales": sales } )


@pytest.fixture(
    scope="module"
)
def trained_model(sample_data):

    model = ProphetForecaster()

    model.fit(sample_data)

    return model


def test_fit(trained_model):

    assert trained_model.is_trained is True
    assert trained_model.model is not None


def test_predict(trained_model):

    horizon = 12

    predictions = trained_model.predict( horizon=horizon )

    assert len(predictions) == horizon


def test_get_params(trained_model):

    params = trained_model.get_params()

    required_keys = [
        "model_type",
        "yearly_seasonality",
        "seasonality_mode"
    ]

    for key in required_keys:
        assert key in params


def test_save_model(trained_model):

    with tempfile.TemporaryDirectory() as temp_dir:

        model_path = (
            Path(temp_dir)
            / "prophet.pkl"
        )

        trained_model.save_model(
            str(model_path)
        )

        assert model_path.exists()


def test_load_model(trained_model):

    with tempfile.TemporaryDirectory() as temp_dir:

        model_path = (
            Path(temp_dir)
            / "prophet.pkl"
        )

        trained_model.save_model(
            str(model_path)
        )

        loaded_model = ( ProphetForecaster() )

        loaded_model.load_model(   str(model_path) )

        assert (
            loaded_model.is_trained
            is True
        )

        preds = loaded_model.predict( horizon=5 )

        assert len(preds) == 5


def test_predict_before_fit():

    model = ProphetForecaster()

    with pytest.raises(  RuntimeError  ):
        model.predict(5)


def test_save_before_fit():

    model = ProphetForecaster()

    with pytest.raises(RuntimeError ):
        model.save_model(
            "dummy.pkl"
        )
