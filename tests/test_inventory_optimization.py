"""
test_inventory_optimization.py

Unit tests for inventory optimization engine.
"""

import pytest
from pipeline.inventory.optimization import (
    get_z_score,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    optimize_inventory
)


def test_get_z_score():
    assert get_z_score(0.99) == 2.33
    assert get_z_score(0.95) == 1.645
    assert get_z_score(0.90) == 1.28
    assert get_z_score(0.50) == 1.0


def test_calculate_safety_stock():
    # Z = 1.645 for 0.95 service level
    # demand_std = 100, lead_time_weeks = 4
    # SS = 1.645 * 100 * sqrt(4) = 1.645 * 100 * 2 = 329.0
    ss = calculate_safety_stock(demand_std=100.0, lead_time_weeks=4.0, service_level=0.95)
    assert ss == 329.0


def test_calculate_reorder_point():
    # ROP = (avg_demand * lead_time) + safety_stock
    # ROP = (50 * 2) + 20 = 120
    rop = calculate_reorder_point(average_demand=50.0, lead_time_weeks=2.0, safety_stock=20.0)
    assert rop == 120.0


def test_calculate_eoq():
    # D = 10 * 52 = 520
    # S = 50
    # H = 2
    # EOQ = sqrt((2 * 520 * 50) / 2) = sqrt(26000) = 161.25
    eoq = calculate_eoq(average_weekly_demand=10.0, holding_cost_per_unit_year=2.0, setup_cost_per_order=50.0)
    assert pytest.approx(eoq, 0.01) == 161.25


def test_optimize_inventory():
    forecasts = [100.0, 110.0, 90.0, 100.0]  # mean = 100.0
    res = optimize_inventory(
        forecast_demands=forecasts,
        historical_sales_std=20.0,
        lead_time_weeks=2.0,
        service_level=0.95,
        holding_cost_unit_year=1.5,
        setup_cost_order=50.0
    )
    assert res["average_forecasted_demand"] == 100.0
    assert res["safety_stock"] > 0
    assert res["reorder_point"] > res["average_forecasted_demand"] * 2.0
    assert res["economic_order_quantity"] > 0
    assert res["parameters"]["lead_time_weeks"] == 2.0
