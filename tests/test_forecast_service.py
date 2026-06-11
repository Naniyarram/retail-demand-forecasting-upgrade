"""
test_forecast_service.py

Unit tests for the forecast serving service.
"""

from pathlib import Path

import joblib
import pytest

from pipeline.api.forecast_service import ForecastService


class DummyForecaster:
    """
    Minimal serializable forecaster for serving tests.
    """

    def predict(
        self,
        horizon
    ):

        return [100.0 + index for index in range(horizon)]

    def get_model_name(self):

        return "DummyForecaster"


def test_forecast_service_loads_and_predicts(
    tmp_path
):

    artifact_path = (
        tmp_path
        / "champion_model.pkl"
    )

    joblib.dump(
        DummyForecaster(),
        artifact_path
    )

    service = ForecastService(
        artifact_path=artifact_path,
        metadata_path=tmp_path / "metadata.json"
    )

    forecast = service.forecast(
        forecast_horizon=3
    )

    assert forecast == [
        100.0,
        101.0,
        102.0
    ]

    assert service.get_model_name() == "DummyForecaster"


def test_forecast_service_missing_artifact(
    tmp_path
):

    service = ForecastService(
        artifact_path=tmp_path / "missing.pkl",
        metadata_path=tmp_path / "metadata.json"
    )

    with pytest.raises(
        FileNotFoundError
    ):

        service.load_model()


def test_forecast_service_rejects_invalid_horizon(
    tmp_path
):

    artifact_path = Path(
        tmp_path
        / "champion_model.pkl"
    )

    joblib.dump(
        DummyForecaster(),
        artifact_path
    )

    service = ForecastService(
        artifact_path=artifact_path,
        metadata_path=tmp_path / "metadata.json"
    )

    with pytest.raises(
        ValueError
    ):

        service.forecast(
            forecast_horizon=0
        )
