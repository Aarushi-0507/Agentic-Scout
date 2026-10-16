"""
Model Explainability Agent for AGENTIC SCOUT.
Calculates SHAP values, feature importance attributions, and local decision breakdowns.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from core.state import AnalysisState
from explainability.shap_explainer import ShapExplainer
from explainability.feature_importance import FallbackExplainer


class ExplanationAgent(BaseAgent):
    """Explains model predictions using SHAP values and intrinsic feature importance."""

    def __init__(self, max_samples: int = 100):
        super().__init__(name="Explanation Agent", role="SHAP & Feature Attribution")
        self.shap_explainer = ShapExplainer(max_samples=max_samples)

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        ml_res = state.ml_results
        champion_estimator = ml_res.get("champion_estimator")
        
        # Check if we have X_train and X_test from predictive agent
        if champion_estimator is not None and "X_train" in ml_res and "X_test" in ml_res:
            X_train = ml_res["X_train"]
            X_test = ml_res["X_test"]
            y_test = ml_res.get("y_test", np.zeros(len(X_test)))
            feature_names = ml_res.get("feature_names", [f"Feat_{i}" for i in range(X_test.shape[1])])
            
            exp_data = self.shap_explainer.explain(
                estimator=champion_estimator,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names
            )
        else:
            # When supervised splits or champion estimator are not available (e.g. ARIMA, unsupervised clustering)
            champ_name = ml_res.get("champion_model_name", "the active model")
            exp_data = {
                "available": False,
                "method": "Attribution Unavailable",
                "importance_df": pd.DataFrame(),
                "top_features": [],
                "base_value": 0.0,
                "local_explanation": [],
                "reason": f"Feature attribution is unavailable for {champ_name} (unsupervised clustering or statistical time series forecasting without supervised feature matrices)."
            }

        state.explainability = exp_data
        if exp_data.get("available", True) and exp_data.get("top_features"):
            top_3 = ", ".join(exp_data.get("top_features", [])[:3])
            summary = f"Computed {exp_data.get('method')} attributions. Top predictive drivers: {top_3}."
        else:
            summary = f"Feature attribution unavailable for {ml_res.get('champion_model_name', 'current configuration')}."
        return {"summary": summary, "explainability": exp_data}
