"""
test_risk_classification.py

Unit tests for inventory risk classification engine.
"""

from pipeline.inventory.risk import classify_risk


def test_classify_risk_critical_stockout():
    # safety stock is 100, current stock is 50 (< safety stock)
    res = classify_risk(
        current_inventory=50.0,
        reorder_point=150.0,
        safety_stock=100.0,
        total_forecasted_demand=1000.0
    )
    assert res["stockout_risk"]["level"] == "Critical"
    assert "safety stock" in res["stockout_risk"]["description"].lower()


def test_classify_risk_high_stockout():
    # safety stock is 100, ROP is 150, current stock is 120 (between 100 and 150)
    res = classify_risk(
        current_inventory=120.0,
        reorder_point=150.0,
        safety_stock=100.0,
        total_forecasted_demand=1000.0
    )
    assert res["stockout_risk"]["level"] == "High"
    assert "reorder point" in res["stockout_risk"]["description"].lower()


def test_classify_risk_overstock_high():
    # total demand + safety stock = 1000 + 100 = 1100
    # current stock is 1500 (> 1100 * 1.3 = 1430)
    res = classify_risk(
        current_inventory=1500.0,
        reorder_point=150.0,
        safety_stock=100.0,
        total_forecasted_demand=1000.0
    )
    assert res["overstock_risk"]["level"] == "High"
    assert "exceed" in res["overstock_risk"]["description"].lower()


def test_classify_risk_normal():
    res = classify_risk(
        current_inventory=300.0,
        reorder_point=150.0,
        safety_stock=100.0,
        total_forecasted_demand=1000.0
    )
    assert res["stockout_risk"]["level"] == "Low"
    assert res["overstock_risk"]["level"] == "Low"
