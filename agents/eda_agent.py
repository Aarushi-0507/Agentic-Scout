"""
Exploratory Data Analysis (EDA) Agent for AGENTIC SCOUT.
Performs descriptive analytics, distribution reviews, and feature interaction assessments.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent
from core.state import AnalysisState


class EDAAgent(BaseAgent):
    """Executes exploratory analysis, identifies distributions, and visual interactions."""

    def __init__(self):
        super().__init__(name="EDA Agent", role="Descriptive & Exploratory Analysis")

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        df = state.df
        profile = state.profile
        
        num_stats = profile.get("numerical_statistics", {})
        cat_stats = profile.get("categorical_statistics", {})
        correlations = profile.get("correlations", {})
        
        # Identify top skewed features
        highly_skewed = [
            {"feature": col, "skewness": data["skewness"]}
            for col, data in num_stats.items()
            if abs(data.get("skewness", 0)) > 1.0
        ]
        
        # Dominant categories
        dominant_cats = []
        for col, data in cat_stats.items():
            top_pcts = data.get("top_categories_pct", {})
            if top_pcts:
                top_cat, top_val = next(iter(top_pcts.items()))
                if top_val > 60.0:
                    dominant_cats.append({
                        "column": col,
                        "dominant_category": top_cat,
                        "share_pct": top_val
                    })
                    
        eda_insights = {
            "highly_skewed_features": highly_skewed,
            "dominant_categories": dominant_cats,
            "top_positive_correlations": correlations.get("top_positive", [])[:5],
            "top_negative_correlations": correlations.get("top_negative", [])[:5],
            "collinear_features": correlations.get("collinear_pairs", [])
        }
        state.eda_insights = eda_insights
        
        summary = (
            f"Analyzed {len(num_stats)} numerical & {len(cat_stats)} categorical features. "
            f"Identified {len(highly_skewed)} skewed distributions and {len(correlations.get('top_positive', []))} notable correlation pairs."
        )
        return {"summary": summary, "insights": eda_insights}
