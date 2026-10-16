"""
Prescriptive Analytics Agent for AGENTIC SCOUT.
Transforms model insights and SHAP drivers into prioritized, evidence-based business initiatives.
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from core.state import AnalysisState
from recommendations.engine import RecommendationEngine
from recommendations.scenarios import ScenarioSimulator


class PrescriptiveAgent(BaseAgent):
    """Generates actionable recommendations, impact/effort rankings, and what-if simulation hooks."""

    def __init__(self):
        super().__init__(name="Prescriptive Agent", role="Prescriptive Strategy & Optimization")

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        problem_type = state.selected_problem or state.problem_detection.get("detected_problem", "Regression")
        target_col = state.selected_target or state.problem_detection.get("target_column")
        
        top_features = state.explainability.get("top_features", [])
        importance_df = state.explainability.get("importance_df")
        test_metrics = state.ml_results.get("test_metrics", {})
        
        # 1. Synthesize prioritized recommendations
        rec_data = RecommendationEngine.generate_recommendations(
            problem_type=problem_type,
            target_col=target_col,
            top_features=top_features,
            importance_df=importance_df,
            test_metrics=test_metrics
        )
        
        # 2. Attach Scenario Simulator if supervised model is available
        champ_estimator = state.ml_results.get("champion_estimator")
        preprocessor = state.ml_results.get("preprocessor")
        
        simulator = None
        if champ_estimator is not None and preprocessor is not None and state.df is not None:
            simulator = ScenarioSimulator(
                estimator=champ_estimator,
                preprocessor=preprocessor,
                raw_df=state.df,
                target_col=target_col
            )
            
        rec_data["simulator"] = simulator
        state.recommendations = rec_data
        
        n_recs = len(rec_data.get("action_items", []))
        n_qw = rec_data.get("quick_wins_count", 0)
        summary = f"Synthesized {n_recs} evidence-based business initiatives ({n_qw} Quick Wins) with rigorous statistical caveats."
        return {"summary": summary, "recommendations": rec_data}
