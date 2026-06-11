"""
test_model_registry.py

Unit tests for ModelRegistryManager using mocks to avoid 
external MLflow backend database requirements.

Author: Nani
"""

import pytest
from unittest.mock import MagicMock, patch
from pipeline.training.model_registry import ModelRegistryManager


@pytest.fixture
def mock_client():
    """
    Create a mock MlflowClient instance.
    """
    client = MagicMock()
    return client


@pytest.fixture
def registry_manager(mock_client):
    """
    Create ModelRegistryManager with a mocked client.
    """
    with patch("pipeline.training.model_registry.MlflowClient", return_value=mock_client):
        manager = ModelRegistryManager()
        return manager


def test_register_model(registry_manager):
    """
    Verify model registration parses and returns the version correctly.
    """
    mock_version = MagicMock()
    mock_version.version = "3"

    with patch("mlflow.register_model", return_value=mock_version) as mock_register:
        version = registry_manager.register_model(
            model_uri="runs:/test_run/model",
            registered_model_name="RetailForecaster"
        )
        
        mock_register.assert_called_once_with(
            model_uri="runs:/test_run/model",
            name="RetailForecaster"
        )
        assert version == 3


def test_set_alias(registry_manager, mock_client):
    """
    Verify client method is called to set a model alias.
    """
    registry_manager.set_alias(
        model_name="RetailForecaster",
        version=2,
        alias="champion"
    )

    mock_client.set_registered_model_alias.assert_called_once_with(
        name="RetailForecaster",
        alias="champion",
        version="2"
    )


def test_get_champion_version(registry_manager, mock_client):
    """
    Verify retrieval of model version by alias.
    """
    mock_version = MagicMock()
    mock_version.version = "5"
    mock_client.get_model_version_by_alias.return_value = mock_version

    version = registry_manager.get_champion_version("RetailForecaster")

    mock_client.get_model_version_by_alias.assert_called_once_with(
        name="RetailForecaster",
        alias="champion"
    )
    assert version == 5


def test_get_champion_version_failure(registry_manager, mock_client):
    """
    Verify None is returned when model alias is not found.
    """
    mock_client.get_model_version_by_alias.side_effect = Exception("Alias not found")

    version = registry_manager.get_champion_version("RetailForecaster")
    
    assert version is None


def test_list_versions(registry_manager, mock_client):
    """
    Verify model versions search formatting.
    """
    mock_v1 = MagicMock()
    mock_v1.version = "1"
    mock_v1.status = "READY"
    mock_v1.run_id = "run_1"

    mock_v2 = MagicMock()
    mock_v2.version = "2"
    mock_v2.status = "READY"
    mock_v2.run_id = "run_2"

    mock_client.search_model_versions.return_value = [mock_v1, mock_v2]

    versions = registry_manager.list_versions("RetailForecaster")

    mock_client.search_model_versions.assert_called_once_with(
        "name='RetailForecaster'"
    )
    assert len(versions) == 2
    assert versions[0] == {"version": "1", "status": "READY", "run_id": "run_1"}
    assert versions[1] == {"version": "2", "status": "READY", "run_id": "run_2"}


def test_promote_and_rollback_aliases(registry_manager, mock_client):
    """
    Verify high-level promotion wrappers call client.
    """
    registry_manager.promote_to_champion("RetailForecaster", 4)
    mock_client.set_registered_model_alias.assert_any_call(
        name="RetailForecaster",
        alias="champion",
        version="4"
    )

    registry_manager.promote_to_challenger("RetailForecaster", 5)
    mock_client.set_registered_model_alias.assert_any_call(
        name="RetailForecaster",
        alias="challenger",
        version="5"
    )

    registry_manager.rollback_champion("RetailForecaster", 3)
    mock_client.set_registered_model_alias.assert_any_call(
        name="RetailForecaster",
        alias="champion",
        version="3"
    )
