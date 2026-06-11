"""
test_llm_client.py

Unit tests for the Hugging Face serverless LLM client.

Author: Nani
"""

import pytest
from unittest.mock import MagicMock, patch
from pipeline.utils.llm_client import HFLLMClient


@pytest.fixture
def mock_forecast_data():
    """
    Standard test payload representing forecast metrics.
    """
    return {
        "store_id": 1,
        "department_id": 1,
        "horizon": 12,
        "average_historical": 12000.0,
        "average_forecast": 13000.0,
        "total_forecast": 156000.0,
        "trend_direction": "Increase of +8.3%",
        "change_pct": 8.3
    }


def test_build_prompt(mock_forecast_data):
    """
    Verify that the prompt is constructed with all required statistics.
    """
    client = HFLLMClient()
    prompt = client._build_prompt(mock_forecast_data)
    
    assert "Store 1" in prompt
    assert "Department 1" in prompt
    assert "12 weeks" in prompt
    assert "$12,000.00" in prompt
    assert "$13,000.00" in prompt
    assert "$156,000.00" in prompt
    assert "Increase of +8.3%" in prompt


def test_verify_output_success(mock_forecast_data):
    """
    Verify output validation passes on clean structured text referencing sales.
    """
    client = HFLLMClient()
    valid_text = (
        "#### 1. Demand & Trend Analysis\n"
        "The sales are projected to see a solid expansion over the coming weeks.\n"
        "#### 2. Inventory & Stocking Recommendations\n"
        "We recommend raising stock levels to support demand.\n"
        "#### 3. Operational & Marketing Actions\n"
        "Run marketing promotions to boost sales."
    )
    assert client._verify_output(valid_text, mock_forecast_data) is True


def test_verify_output_failure_too_short(mock_forecast_data):
    """
    Verify validation fails when text is too short.
    """
    client = HFLLMClient()
    short_text = "Sales forecast looks good."
    assert client._verify_output(short_text, mock_forecast_data) is False


def test_verify_output_failure_missing_relevance(mock_forecast_data):
    """
    Verify validation fails if it doesn't mention retail-related terms.
    """
    client = HFLLMClient()
    irrelevant_text = (
        "The weather in Seattle is rainy today. We expect clouds and temperature "
        "drops. The local coffee shops are open and people are walking around with "
        "umbrellas. This has nothing to do with sales or inventory planning."
    )
    assert client._verify_output(irrelevant_text, mock_forecast_data) is False


@patch("requests.post")
def test_generate_insights_success(mock_post, mock_forecast_data):
    """
    Verify successful API response triggers parsing and validation.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": (
                    "#### 1. Demand & Trend Analysis\n"
                    "Sales are projected to grow nicely over the next few weeks due to seasonality.\n"
                    "#### 2. Inventory & Stocking Recommendations\n"
                    "We must raise inventory stock buffer parameters by 10% to prevent stockouts.\n"
                    "#### 3. Operational & Marketing Actions\n"
                    "Run a small advertising promotion to capture this traffic."
                )
            }
        }]
    }
    mock_post.return_value = mock_response

    client = HFLLMClient()
    result = client.generate_retail_insights(mock_forecast_data)

    assert result["verified"] is True
    assert "Demand & Trend" in result["raw_insights"]
    assert result["model_used"] == client.DEFAULT_MODEL


@patch("requests.post")
def test_generate_insights_api_failure_fallback(mock_post, mock_forecast_data):
    """
    Verify that an API error automatically triggers rule-based fallback generation.
    """
    mock_post.side_effect = Exception("Connection Timeout")

    client = HFLLMClient()
    result = client.generate_retail_insights(mock_forecast_data)

    # Verified is False because it failed the raw API call, but returned fallback text
    assert result["verified"] is False
    assert "Rule-Based Analytical Fallback" in result["model_used"]
    assert "#### 1. Demand & Trend Analysis" in result["raw_insights"]
    assert "safety stock" in result["raw_insights"].lower()
