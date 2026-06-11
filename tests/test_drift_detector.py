"""
test_drift_detector.py

Unit tests for Kolmogorov-Smirnov drift detection.
"""

import pandas as pd
import numpy as np
from pipeline.monitoring.drift_detector import DataDriftDetector
from pipeline.training.retrain import AutomatedRetrainer


def test_drift_detection_no_drift():
    # Identical distributions
    np.random.seed(42)
    baseline = pd.DataFrame({"sales": np.random.normal(100, 10, 1000)})
    current = pd.DataFrame({"sales": np.random.normal(100, 10, 1000)})

    detector = DataDriftDetector(alpha=0.05)
    report = detector.detect_drift(baseline, current, ["sales"])
    
    assert report["drift_detected"] is False
    assert report["metrics"]["sales"]["drift_detected"] is False


def test_drift_detection_with_drift():
    # Statistically shifted distributions
    np.random.seed(42)
    baseline = pd.DataFrame({"sales": np.random.normal(100, 10, 1000)})
    current = pd.DataFrame({"sales": np.random.normal(130, 15, 1000)})  # Shifted mean and std

    detector = DataDriftDetector(alpha=0.05)
    report = detector.detect_drift(baseline, current, ["sales"])
    
    assert report["drift_detected"] is True
    assert report["metrics"]["sales"]["drift_detected"] is True
    assert report["metrics"]["sales"]["p_value"] < 0.05


def test_automated_retrainer_no_trigger():
    np.random.seed(42)
    baseline = pd.DataFrame({"Weekly_Sales": np.random.normal(100, 10, 100)})
    current = pd.DataFrame({"Weekly_Sales": np.random.normal(100, 10, 100)})

    retrainer = AutomatedRetrainer(significance_level=0.05)
    res = retrainer.run_retraining_check(
        baseline_df=baseline,
        current_df=current,
        drift_columns=["Weekly_Sales"],
        champion_result={"model_name": "XGBoost"},
        force=False
    )
    assert res["drift_detected"] is False
    assert res["retraining_triggered"] is False
    assert res["pipeline_output"] is None
