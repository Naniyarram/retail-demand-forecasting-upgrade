"""
test_champion_trainer.py

Unit tests for ChampionTrainer.

Author: Nani
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.training.champion_trainer import (ChampionTrainer)

from pipeline.forecasting.sarima import ( SARIMAForecaster)


@pytest.fixture(
    scope="module"
)
def sample_data():
    """
    Generate realistic weekly sales data.
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

    return pd.DataFrame( {"Date": dates,"Weekly_Sales": sales})


@pytest.fixture(scope="module")
def champion_result(sample_data):
    """
    Create mock champion result.
    """

    model = SARIMAForecaster()

    model.fit(sample_data)

    return {
        "model_name": "SARIMA",
        "RMSE": 100.0,
        "MAE": 80.0,
        "MAPE": 5.0,
        "parameters": model.get_params(),
        "model_object": model
    }


def test_retrain_champion(champion_result,sample_data):
    """
    Verify retraining works.
    """

    trainer = ChampionTrainer()

    retrained_model = (
        trainer.retrain_champion(
            champion_result=champion_result,
            full_dataset=sample_data
        )
    )

    assert retrained_model is not None
    assert retrained_model.is_trained is True


def test_get_champion_summary(champion_result):
    """
    Verify summary generation.
    """

    trainer = ChampionTrainer()

    summary = (trainer.get_champion_summary( champion_result  ))

    expected_keys = [
        "model_name",
        "RMSE",
        "MAE",
        "MAPE",
        "parameters"
    ]

    for key in expected_keys:

        assert key in summary


def test_summary_values(
    champion_result
):
    """
    Verify summary values.
    """

    trainer = ChampionTrainer()

    summary = (
        trainer.get_champion_summary( champion_result ) )

    assert (
        summary["model_name"]
        == "SARIMA"
    )

    assert (
        summary["RMSE"]
        == 100.0
    )

    assert (
        summary["MAE"]
        == 80.0
    )

    assert (
        summary["MAPE"]
        == 5.0
    )


def test_print_champion_summary(champion_result, capsys):
    """
    Verify summary printing.
    """

    trainer = ChampionTrainer()

    trainer.print_champion_summary(champion_result)

    captured = (
        capsys.readouterr()
    )

    assert (
        "Champion Summary"
        in captured.out
    )

    assert (
        "SARIMA"
        in captured.out
    )
