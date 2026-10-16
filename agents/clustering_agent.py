"""
Clustering and Customer Segmentation Agent for AGENTIC SCOUT.
Discovers behavioral cohorts, segment personas, and multivariate anomalies.
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from core.state import AnalysisState
from ml.preprocessor import DataPreprocessor
from ml.clustering import ClusteringPipeline


class ClusteringAgent(BaseAgent):
    """Executes unsupervised clustering, optimal k search, and persona synthesis."""

    def __init__(self):
        super().__init__(name="Clustering Agent", role="Unsupervised Segmentation & Anomaly Detection")
        self.pipeline = ClusteringPipeline()

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        df = state.df
        problem_type = state.selected_problem or state.problem_detection.get("detected_problem", "")
        
        preprocessor = DataPreprocessor(random_state=42)
        prep_data = preprocessor.prepare_data(df=df, target_col=None)
        
        X = prep_data["X"]
        X_raw = prep_data["X_raw"]
        feature_names = prep_data["feature_names"]
        
        clustering_results = self.pipeline.fit_and_profile(
            X=X, X_raw=X_raw, feature_names=feature_names
        )
        clustering_results["preprocessor"] = preprocessor
        
        # If clustering is the primary task, store directly in ml_results
        if "Clustering" in problem_type or "Anomaly" in problem_type:
            state.ml_results = clustering_results
        else:
            # Store alongside supervised results
            state.ml_results["clustering_auxiliary"] = clustering_results

        k = clustering_results["optimal_k"]
        sil = clustering_results["test_metrics"].get("Silhouette", 0.0)
        anomalies = clustering_results.get("anomaly_count", 0)
        
        summary = (
            f"Segmented data into {k} optimal behavioral cohorts (Silhouette: {sil}). "
            f"Detected {anomalies} statistical anomalies via Isolation Forest."
        )
        return {"summary": summary, "clustering_results": clustering_results}
