"""
Classification Pipeline for AGENTIC SCOUT.
Trains and evaluates Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, and XGBoost with graceful fallback and confusion matrix calculation.
"""
from typing import Dict, Any
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from ml.evaluation import Evaluator


class ClassificationPipeline:
    """Trains multiple competitive classification architectures and selects the champion model."""
    
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
        """Runs model tournament across candidate classification algorithms."""
        models: Dict[str, Any] = {}
        
        # 1. Logistic Regression
        models["Logistic Regression"] = LogisticRegression(
            max_iter=1000, random_state=self.random_state
        )
        
        # 2. Decision Tree
        models["Decision Tree"] = DecisionTreeClassifier(
            max_depth=6, random_state=self.random_state
        )
        
        # 3. Random Forest
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
        )
        
        # 4. Gradient Boosting
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=4, random_state=self.random_state
        )
        
        # 5. XGBoost (with fallback)
        try:
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=self.random_state, eval_metric="logloss", n_jobs=-1
            )
        except Exception:
            models["Hist Gradient Boosting"] = HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.08, random_state=self.random_state
            )

        results: Dict[str, Any] = {}
        best_f1 = -float("inf")
        champion_name = ""
        
        for name, estimator in models.items():
            start_t = time.time()
            try:
                estimator.fit(X_train, y_train)
                fit_ms = (time.time() - start_t) * 1000.0
                
                y_train_pred = estimator.predict(X_train)
                y_test_pred = estimator.predict(X_test)
                
                y_test_prob = None
                if hasattr(estimator, "predict_proba"):
                    y_test_prob = estimator.predict_proba(X_test)
                elif hasattr(estimator, "decision_function"):
                    dfn = estimator.decision_function(X_test)
                    y_test_prob = 1.0 / (1.0 + np.exp(-dfn))
                    
                train_metrics = Evaluator.evaluate_classification(y_train, y_train_pred)
                test_metrics = Evaluator.evaluate_classification(y_test, y_test_pred, y_test_prob)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred).tolist()
                
                results[name] = {
                    "estimator": estimator,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "fit_time_ms": round(fit_ms, 2),
                    "y_pred": y_test_pred,
                    "y_prob": y_test_prob,
                    "confusion_matrix": cm,
                    "classes": np.unique(y_train).tolist()
                }
                
                if test_metrics["F1"] > best_f1:
                    best_f1 = test_metrics["F1"]
                    champion_name = name
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "fit_time_ms": 0.0,
                    "test_metrics": {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "ROC_AUC": 0.0}
                }

        # Build leaderboard
        leaderboard = Evaluator.build_leaderboard(results, "Classification")
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
            "y_prob": champion_data.get("y_prob"),
            "y_test": y_test,
            "confusion_matrix": champion_data.get("confusion_matrix"),
            "classes": champion_data.get("classes", []),
            "feature_names": feature_names
        }
