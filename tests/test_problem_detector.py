"""
Unit tests for ProblemDetector and transparent decision rules.
"""
import pytest
import pandas as pd
import numpy as np
from core.profiler import DataProfiler
from core.problem_detector import ProblemDetector, ProblemType


@pytest.fixture
def detector():
    return ProblemDetector()


@pytest.fixture
def profiler():
    return DataProfiler()


def test_detect_regression(detector, profiler):
    np.random.seed(42)
    df = pd.DataFrame({
        "Spend": np.random.uniform(10, 500, size=100),
        "Clicks": np.random.randint(5, 100, size=100),
        "Revenue": np.random.uniform(100, 5000, size=100)
    })
    prof = profiler.profile(df)
    res = detector.detect(df, prof)
    
    assert res["detected_problem"] == ProblemType.REGRESSION.value
    assert res["target_column"] == "Revenue"
    assert "scores" in res
    assert len(res["reasoning"]) > 0


def test_detect_classification(detector, profiler):
    np.random.seed(42)
    df = pd.DataFrame({
        "Tenure": np.random.randint(1, 60, size=100),
        "Monthly_Charges": np.random.uniform(20, 150, size=100),
        "Churn": np.random.choice([0, 1], size=100)
    })
    prof = profiler.profile(df)
    res = detector.detect(df, prof)
    
    assert res["detected_problem"] == ProblemType.CLASSIFICATION.value
    assert res["target_column"] == "Churn"
    assert res["scores"][ProblemType.CLASSIFICATION.value] > 50.0


def test_detect_forecasting(detector, profiler):
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Demand_Units": np.random.randint(50, 200, size=100)
    })
    prof = profiler.profile(df)
    res = detector.detect(df, prof)
    
    assert res["detected_problem"] == ProblemType.FORECASTING.value
    assert res["datetime_column"] == "Date"
    assert res["scores"][ProblemType.FORECASTING.value] > 50.0
