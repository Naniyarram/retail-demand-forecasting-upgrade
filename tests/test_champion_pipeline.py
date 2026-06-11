"""
test_champion_pipeline.py

Unit tests for ChampionPipeline orchestration.

Author: Nani
"""

import tempfile
import pandas as pd
import numpy as np
import pytest
import mlflow
from pathlib import Path

from pipeline.training.champion_pipeline import ChampionPipeline

class DummyForecaster:
    """
    Serializable test double representing a model wrapper.
    """

    def __init__(self):
        self.is_trained = False
        self.model_name = "DummyForecaster"

    def fit(self, df: pd.DataFrame) -> None:
        self.is_trained = True

    def predict(self, horizon: int):
        return np.ones(horizon) * 100.0

    def get_params(self):
        return {"param_key": "param_value"}

    def get_model_name(self):
        return self.model_name


@pytest.fixture
def dummy_dataset():
    """
    Create simple timeseries for testing.
    """
    dates = pd.date_range(start="2022-01-01", periods=10, freq="W")
    sales = np.random.uniform(500, 1500, size=10)
    return pd.DataFrame({
        "Date": dates,
        "Weekly_Sales": sales,
        "Store": [1] * 10,
        "Dept": [1] * 10
    })


@pytest.fixture
def mock_champion_result(dummy_dataset):
    """
    Build structured champion model payload.
    """
    model = DummyForecaster()
    return {
        "model_name": "DummyForecaster",
        "RMSE": 12.5,
        "MAE": 10.0,
        "MAPE": 1.2,
        "parameters": model.get_params(),
        "model_object": model
    }


def test_champion_pipeline_run(tmp_path, dummy_dataset, mock_champion_result, monkeypatch):
    """
    Verify that ChampionPipeline executes retraining, serialization, 
    and returns correct paths.
    """
    # Redirect MLflow tracking for tests to avoid writing to mlflow.db
    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment("test_champion_pipeline_run")

    pipeline = ChampionPipeline()

    # Override packaging directories to use pytest temp directories
    monkeypatch.setattr(pipeline.packager, "artifact_dir", tmp_path / "models")

    # Run the pipeline
    results = pipeline.run(
        champion_result=mock_champion_result,
        full_dataset=dummy_dataset
    )

    # Validate output dictionary schema
    assert results["model_name"] == "DummyForecaster"
    assert results["production_model"].is_trained is True
    assert results["champion_summary"]["RMSE"] == 12.5
    assert "runs:/" in results["model_uri"]
    
    # Check that model artifact exists
    assert Path(results["artifact_path"]).exists()
    assert Path(results["metadata_path"]).exists()
