"""
test_experiment_runner.py

Fast tests for experiment orchestration.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation.metrics import ForecastMetrics
from pipeline.training.experiment_runner import (
    ExperimentRunner
)


class DummyForecaster:
    """
    Lightweight forecasting model for runner tests.
    """

    def __init__( self,model_name,adjustment ):

        self.model_name = model_name

        self.adjustment = adjustment

        self.is_trained = False

        self.last_value = 0.0

        self.metrics = {}

    def fit(  self,  train_df ):

        self.last_value = float( train_df["Weekly_Sales"].iloc[-1] )

        self.is_trained = True

    def predict( self, horizon ):

        return np.full(horizon, self.last_value + self.adjustment   )

    def evaluate(self,y_true,y_pred ):

        self.metrics = ForecastMetrics.evaluate( y_true, y_pred )

        return self.metrics

    def get_model_name(self):

        return self.model_name

    def get_params(self):

        return {
            "adjustment": self.adjustment
        }


@pytest.fixture
def sample_data():

    dates = pd.date_range(start="2020-01-05", periods=90, freq="W" )

    sales = np.linspace( 1000, 1300,   90)

    return pd.DataFrame(
        { "Date": dates,"Weekly_Sales": sales })


@pytest.fixture
def models():

    return [
        DummyForecaster(
            "Baseline",
            0
        ),
        DummyForecaster(
            "Optimistic",
            25
        ),
        DummyForecaster(
            "Conservative",
            -25
        )
    ]


def test_run_experiments( sample_data,  models):

    runner = ExperimentRunner(   models=models )

    leaderboard = runner.run(sample_data )

    assert len(leaderboard) == 3


def test_leaderboard_columns(sample_data,models):

    runner = ExperimentRunner( models=models  )

    leaderboard = runner.run( sample_data)

    expected_columns = [
        "Rank",
        "Model",
        "RMSE",
        "MAE",
        "MAPE"
    ]

    for col in expected_columns:

        assert col in leaderboard.columns


def test_champion_selection( sample_data, models):

    runner = ExperimentRunner( models=models )

    runner.run(   sample_data)

    champion = runner.get_champion()

    assert "model_name" in champion


def test_full_results(sample_data, models):

    runner = ExperimentRunner( models=models )

    runner.run( sample_data)

    results = runner.get_full_results()

    assert len(results) == 3
