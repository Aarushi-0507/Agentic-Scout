"""
Predictive Modeling Agent for AGENTIC SCOUT.
Orchestrates machine learning tournaments across Regression, Classification, and Forecasting.
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from core.state import AnalysisState
from ml.preprocessor import DataPreprocessor
from ml.regression import RegressionPipeline
from ml.classification import ClassificationPipeline
from ml.forecasting import ForecastingPipeline


class PredictiveAgent(BaseAgent):
    """Trains competitive supervised ML models, enforces zero-leakage splits, and compiles leaderboards."""

    def __init__(self):
        super().__init__(name="Predictive Agent", role="Predictive ML Modeling")
        self.regression_pipe = RegressionPipeline()
        self.classification_pipe = ClassificationPipeline()
        self.forecasting_pipe = ForecastingPipeline()

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        df = state.df
        problem_type = state.selected_problem or state.problem_detection.get("detected_problem", "Regression")
        target_col = state.selected_target or state.problem_detection.get("target_column")
        date_col = state.selected_datetime_col or state.problem_detection.get("datetime_column")

        # 1. Forecasting Branch
        if "Forecasting" in problem_type and date_col and target_col:
            results = self.forecasting_pipe.train_and_forecast(
                df=df, date_col=date_col, target_col=target_col
            )
            state.ml_results = results
            champ = results["champion_model_name"]
            rmse = results["test_metrics"].get("RMSE", 0.0)
            summary = f"Completed forecasting tournament. Champion: {champ} with Test RMSE: {rmse}."
            return {"summary": summary, "ml_results": results}

        # 2. Supervised Tabular Branch (Regression or Classification)
        if not target_col or target_col not in df.columns:
            # Pick a fallback numeric column if none set
            num_cols = state.profile.get("column_types", {}).get("numeric", [])
            if num_cols:
                target_col = num_cols[-1]
            else:
                raise ValueError("No viable target column identified for supervised predictive modeling.")

        preprocessor = DataPreprocessor(test_size=0.20, random_state=42)
        stratify_flag = "Classification" in problem_type
        
        prep_data = preprocessor.prepare_data(
            df=df, target_col=target_col, stratify=stratify_flag
        )
        
        X_train = prep_data["X_train"]
        X_test = prep_data["X_test"]
        y_train = prep_data["y_train"].values
        y_test = prep_data["y_test"].values
        feature_names = prep_data["feature_names"]

        if "Classification" in problem_type:
            results = self.classification_pipe.train_and_evaluate(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=feature_names
            )
            metric_str = f"F1: {results['test_metrics'].get('F1', 0.0)}, Accuracy: {results['test_metrics'].get('Accuracy', 0.0)}"
        else:
            # Default to Regression
            results = self.regression_pipe.train_and_evaluate(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=feature_names
            )
            metric_str = f"R²: {results['test_metrics'].get('R2', 0.0)}, RMSE: {results['test_metrics'].get('RMSE', 0.0)}"

        # Attach preprocessor and raw splits to results for explainability & scenarios
        results["preprocessor"] = preprocessor
        results["X_train"] = X_train
        results["X_test"] = X_test
        results["prep_data"] = prep_data
        
        state.ml_results = results
        champ = results["champion_model_name"]
        summary = f"Completed {problem_type} tournament. Champion: {champ} ({metric_str})."
        return {"summary": summary, "ml_results": results}
