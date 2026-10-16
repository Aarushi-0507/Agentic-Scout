"""
Unit tests for Machine Learning tournament pipelines, preprocessing, and evaluation.
"""
import pytest
import numpy as np
import pandas as pd
from ml.preprocessor import DataPreprocessor
from ml.regression import RegressionPipeline
from ml.classification import ClassificationPipeline
from ml.clustering import ClusteringPipeline
from ml.forecasting import ForecastingPipeline
from ml.evaluation import Evaluator


def test_zero_leakage_preprocessor():
    df = pd.DataFrame({
        "Num1": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0],
        "Cat1": ["A", "B", "A", np.nan, "B", "A"],
        "Target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    })
    prep = DataPreprocessor(test_size=0.33, random_state=42)
    res = prep.prepare_data(df, target_col="Target")
    
    assert "X_train" in res
    assert "X_test" in res
    assert res["X_train"].shape[1] == len(res["feature_names"])
    assert not np.isnan(res["X_train"]).any()
    assert not np.isnan(res["X_test"]).any()


def test_regression_pipeline():
    np.random.seed(42)
    X_train = np.random.randn(80, 4)
    y_train = X_train[:, 0] * 3.0 + X_train[:, 1] * -1.5 + np.random.randn(80) * 0.1
    X_test = np.random.randn(20, 4)
    y_test = X_test[:, 0] * 3.0 + X_test[:, 1] * -1.5 + np.random.randn(20) * 0.1
    
    pipe = RegressionPipeline(random_state=42)
    res = pipe.train_and_evaluate(X_train, X_test, y_train, y_test, ["F1", "F2", "F3", "F4"])
    
    assert "champion_model_name" in res
    assert res["leaderboard"] is not None
    assert "MAE" in res["test_metrics"]
    assert "RMSE" in res["test_metrics"]
    assert "R2" in res["test_metrics"]
    assert res["test_metrics"]["R2"] > 0.80  # Good linear fit


def test_classification_pipeline():
    np.random.seed(42)
    X_train = np.random.randn(80, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(20, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    
    pipe = ClassificationPipeline(random_state=42)
    res = pipe.train_and_evaluate(X_train, X_test, y_train, y_test, ["F1", "F2", "F3", "F4"])
    
    assert "champion_model_name" in res
    assert "Accuracy" in res["test_metrics"]
    assert "F1" in res["test_metrics"]
    assert "confusion_matrix" in res


def test_clustering_pipeline():
    np.random.seed(42)
    X = np.random.randn(60, 3)
    raw_df = pd.DataFrame(X, columns=["A", "B", "C"])
    
    pipe = ClusteringPipeline(random_state=42)
    res = pipe.fit_and_profile(X, raw_df, ["A", "B", "C"], min_k=2, max_k=4)
    
    assert "optimal_k" in res
    assert "Silhouette" in res["test_metrics"]
    assert len(res["personas"]) > 0


def test_forecasting_pipeline():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    vals = 100 + np.sin(np.linspace(0, 10, 60)) * 20 + np.random.normal(0, 2, 60)
    df = pd.DataFrame({"Date": dates, "Demand": vals})
    
    pipe = ForecastingPipeline(forecast_horizon=7, random_state=42)
    res = pipe.train_and_forecast(df, date_col="Date", target_col="Demand")
    
    assert "champion_model_name" in res
    assert "future_forecast" in res
    assert len(res["future_forecast"]["dates"]) == 7
    assert len(res["future_forecast"]["predictions"]) == 7
