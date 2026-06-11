import pytest

from pipeline.training.model_factory import (
    ModelFactory
)

from pipeline.forecasting.sarima import (
    SARIMAForecaster
)

from pipeline.forecasting.prophet import (
    ProphetForecaster
)

from pipeline.forecasting.xgboost import (
    XGBoostForecaster
)


def test_create_sarima():

    model = (
        ModelFactory.create_model(
            "SARIMA"
        )
    )

    assert isinstance(
        model,
        SARIMAForecaster
    )


def test_create_prophet():

    model = (
        ModelFactory.create_model(
            "Prophet"
        )
    )

    assert isinstance(
        model,
        ProphetForecaster
    )


def test_create_xgboost():

    model = (
        ModelFactory.create_model(
            "XGBoost"
        )
    )

    assert isinstance(
        model,
        XGBoostForecaster
    )


def test_invalid_model():

    with pytest.raises(
        ValueError
    ):

        ModelFactory.create_model(
            "INVALID"
        )


def test_list_models():

    models = (
        ModelFactory.list_models()
    )

    assert "SARIMA" in models
    assert "Prophet" in models
    assert "XGBoost" in models