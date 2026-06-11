import mlflow

from pipeline.training.model_logger import (
    ModelLogger
)


class DummyModel:

    def predict(
        self,
        horizon
    ):
        return [100] * horizon


def test_log_model(
    tmp_path
):

    mlflow.set_tracking_uri(
        tmp_path.as_uri()
    )

    mlflow.set_experiment(
        "test_model_logger"
    )

    logger = ModelLogger()

    with mlflow.start_run():

        model_uri = logger.log_model(
            model_name="Dummy",
            model_object=DummyModel()
        )

    assert "runs:/" in model_uri
