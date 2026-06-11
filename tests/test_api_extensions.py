"""
test_api_extensions.py

Unit tests for new FastAPI serving endpoints.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from pipeline.api.app import app

client = TestClient(app)


def test_inventory_optimize_endpoint():
    payload = {
        "forecast_demands": [100.0, 120.0, 110.0, 130.0],
        "historical_sales_std": 15.0,
        "lead_time_weeks": 2.0,
        "service_level": 0.95,
        "holding_cost_unit_year": 1.5,
        "setup_cost_order": 50.0
    }
    response = client.post("/inventory/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["average_forecasted_demand"] == 115.0
    assert "safety_stock" in data
    assert "reorder_point" in data
    assert "economic_order_quantity" in data


def test_inventory_risk_endpoint():
    payload = {
        "current_inventory": 50.0,
        "reorder_point": 100.0,
        "safety_stock": 40.0,
        "total_forecasted_demand": 500.0
    }
    response = client.post("/inventory/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "stockout_risk" in data
    assert "overstock_risk" in data
    assert data["stockout_risk"]["level"] == "High"
    assert data["overstock_risk"]["level"] == "Low"


@patch("pipeline.utils.llm_client.requests.post")
def test_decision_recommendations_endpoint(mock_post):
    # Mock LLM API response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [{
            "message": {
                "content": (
                    "#### 1. Demand & Trend Analysis\n"
                    "Demand is rising for Store 1, Department 1.\n"
                    "#### 2. Inventory & Stocking Recommendations\n"
                    "Increase safety stock buffer levels by 15% immediately.\n"
                    "#### 3. Operational & Marketing Actions\n"
                    "Ensure appropriate staffing during the target period."
                )
            }
        }]
    }

    payload = {
        "store_id": 1,
        "department_id": 1,
        "horizon": 12,
        "average_historical": 12000.0,
        "average_forecast": 13000.0,
        "total_forecast": 156000.0,
        "trend_direction": "Increase of +8.3%",
        "change_pct": 8.3
    }
    
    response = client.post("/decision/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["verified"] is True
    assert "Llama-3.1" in data["model_used"] or "llama" in data["model_used"].lower()
    assert "Demand & Trend Analysis" in data["raw_insights"]


def test_monitoring_metrics_endpoint():
    # Make a request to health to increment metric counter
    client.get("/health")
    
    response = client.get("/monitoring/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert data["total_requests"] >= 1
    assert "/health" in data["requests_by_endpoint"]
    assert data["requests_by_endpoint"]["/health"] >= 1
