"""
Regression Pipeline for AGENTIC SCOUT.
Trains and evaluates Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
Includes graceful fallback if XGBoost is unavailable.
"""
from typing import Dict, Any, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

from ml.evaluation import Evaluator


class RegressionPipeline:
    """Trains multiple competitive regression architectures and selects the champion model."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list
    ) -> Dict[str, Any]:
        """Runs model tournament across candidate regression algorithms."""
        models: Dict[str, Any] = {}
        
        # 1. Linear Regression
        models["Linear Regression"] = LinearRegression()
        
        # 2. Random Forest
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
        )
        
        # 3. Gradient Boosting
        models["Gradient Boosting"] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.08, max_depth=4, random_state=self.random_state
        )
        
        # 4. XGBoost (with fallback)
        try:
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=self.random_state, n_jobs=-1
            )
        except Exception:
            models["Hist Gradient Boosting"] = HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.08, random_state=self.random_state
            )
            
        results: Dict[str, Any] = {}
        best_r2 = -float("inf")
        champion_name = ""
        
        for name, estimator in models.items():
            start_t = time.time()
            try:
                estimator.fit(X_train, y_train)
                fit_ms = (time.time() - start_t) * 1000.0
                
                y_train_pred = estimator.predict(X_train)
                y_test_pred = estimator.predict(X_test)
                
                train_metrics = Evaluator.evaluate_regression(y_train, y_train_pred)
                test_metrics = Evaluator.evaluate_regression(y_test, y_test_pred)
                
                results[name] = {
                    "estimator": estimator,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "fit_time_ms": round(fit_ms, 2),
                    "y_pred": y_test_pred,
                    "residuals": (y_test - y_test_pred).tolist()
                }
                
                if test_metrics["R2"] > best_r2:
                    best_r2 = test_metrics["R2"]
                    champion_name = name
            except Exception as e:
                # Graceful handling if any individual estimator encounters an issue
                results[name] = {
                    "error": str(e),
                    "fit_time_ms": 0.0,
                    "test_metrics": {"MAE": 999.0, "RMSE": 999.0, "R2": -99.0, "MAPE": 999.0}
                }
                
        # Build leaderboard
        leaderboard = Evaluator.build_leaderboard(results, "Regression")
        if not champion_name and len(results) > 0:
            champion_name = list(results.keys())[0]
            
        champion_data = results.get(champion_name, {})
        
        return {
            "models": results,
            "leaderboard": leaderboard,
            "champion_model_name": champion_name,
            "champion_estimator": champion_data.get("estimator"),
            "test_metrics": champion_data.get("test_metrics", {}),
            "train_metrics": champion_data.get("train_metrics", {}),
            "y_pred": champion_data.get("y_pred"),
            "y_test": y_test,
            "feature_names": feature_names
        }
