import pandas as pd
import pytest

from pipeline.training.forecasting_pyfunc import (
    ForecastingPyFuncModel
)


class DummyForecaster:

    def predict(
        self,
        horizon
    ):
        return [100] * horizon


def test_predict():

    model = ForecastingPyFuncModel(
        DummyForecaster()
    )

    result = model.predict(
        context=None,
        model_input=pd.DataFrame(
            {
                "horizon": [5]
            }
        )
    )

    assert len(result) == 5

    assert "forecast" in result.columns


def test_invalid_input():

    model = ForecastingPyFuncModel(
        DummyForecaster()
    )

    with pytest.raises(
        ValueError
    ):

        model.predict(
            context=None,
            model_input=pd.DataFrame()
        )