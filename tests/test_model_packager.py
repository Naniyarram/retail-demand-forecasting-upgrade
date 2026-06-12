"""
test_model_packager.py

Unit tests for ModelPackager.

"""

from pathlib import Path

import pytest

from pipeline.training.model_packager import (
    ModelPackager
)


class DummyModel:
    """
    Simple serializable model
    for testing.
    """

    def __init__(self):
        self.name = "dummy_model"


@pytest.fixture
def packager(
    tmp_path
):
    """
    Create test packager.
    """

    return ModelPackager(
        artifact_dir=tmp_path / "models"
    )


@pytest.fixture
def dummy_model():
    """
    Create dummy model.
    """

    return DummyModel()


def test_package_model(
    packager,
    dummy_model
):
    """
    Verify model packaging.
    """

    artifact_path = (
        packager.package_model(
            model_name="DummyModel",
            model_object=dummy_model
        )
    )

    assert Path(
        artifact_path
    ).exists()


def test_artifact_exists(
    packager,
    dummy_model
):
    """
    Verify artifact detection.
    """

    artifact_path = (
        packager.package_model(
            model_name="DummyModel",
            model_object=dummy_model
        )
    )

    assert (
        packager.artifact_exists(
            artifact_path
        )
        is True
    )


def test_load_artifact(
    packager,
    dummy_model
):
    """
    Verify artifact loading.
    """

    artifact_path = (
        packager.package_model(
            model_name="DummyModel",
            model_object=dummy_model
        )
    )

    loaded_model = (
        packager.load_artifact(
            artifact_path
        )
    )

    assert (
        loaded_model.name
        == "dummy_model"
    )


def test_invalid_artifact():
    """
    Verify invalid path handling.
    """

    packager = ModelPackager()

    with pytest.raises(
        FileNotFoundError
    ):
        packager.load_artifact(
            "invalid.pkl"
        )
