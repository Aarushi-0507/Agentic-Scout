"""
Diagnostic Analytics Agent for AGENTIC SCOUT.
Answers 'Why did it happen?': isolates root causes, variance drivers, and subgroup divergences.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent
from core.state import AnalysisState


class DiagnosticAgent(BaseAgent):
    """Diagnoses data anomalies, segment divergences, and statistical variance drivers."""

    def __init__(self):
        super().__init__(name="Diagnostic Agent", role="Diagnostic & Root Cause Analysis")

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        df = state.df
        profile = state.profile
        quality = state.quality_report
        
        findings: List[Dict[str, Any]] = []
        
        # 1. Inspect Outlier drivers
        outliers = profile.get("outliers", {})
        high_outlier_cols = [
            (col, stat["percentage"], stat["count"])
            for col, stat in outliers.items()
            if stat.get("percentage", 0) > 3.0
        ]
        if high_outlier_cols:
            findings.append({
                "category": "Outlier Concentration",
                "finding": f"Significant tail-risk in {len(high_outlier_cols)} features: " +
                           ", ".join([f"{c} ({pct}% outliers)" for c, pct, _ in high_outlier_cols[:3]]),
                "diagnostic_implication": "May distort linear metrics; non-parametric and tree ensembles will yield more stable decision boundaries."
            })

        # 2. Inspect Target Variance / Imbalance if target is identified
        target_col = state.selected_target or state.problem_detection.get("target_column")
        if target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            
            # Binary class imbalance check
            if target_series.nunique() == 2:
                vc = target_series.value_counts(normalize=True)
                minority_pct = round(float(vc.min()) * 100.0, 1)
                majority_pct = round(float(vc.max()) * 100.0, 1)
                if minority_pct < 20.0:
                    findings.append({
                        "category": "Class Imbalance",
                        "finding": f"Target '{target_col}' displays severe class imbalance: {minority_pct}% positive vs. {majority_pct}% negative.",
                        "diagnostic_implication": "Accuracy will be misleading. Model evaluation must prioritize Precision, Recall, and F1 score."
                    })
            elif pd.api.types.is_numeric_dtype(target_series):
                cv = float(target_series.std() / target_series.mean()) if target_series.mean() != 0 else 0
                if cv > 1.0:
                    findings.append({
                        "category": "High Target Dispersion",
                        "finding": f"Target '{target_col}' has high coefficient of variation (CV = {round(cv, 2)}).",
                        "diagnostic_implication": "Heterogeneous business outcomes; segment-specific trees or regularization will improve generalization."
                    })

        # 3. Subgroup Divergence Check
        cat_cols = profile.get("column_types", {}).get("categorical", [])
        num_cols = profile.get("column_types", {}).get("numeric", [])
        
        subgroup_divergences = []
        if target_col and target_col in num_cols and cat_cols:
            for cat in cat_cols[:2]:
                try:
                    grp_means = df.groupby(cat)[target_col].mean().dropna()
                    if len(grp_means) > 1:
                        max_g = grp_means.idxmax()
                        min_g = grp_means.idxmin()
                        spread = round(float(grp_means.max() - grp_means.min()), 2)
                        subgroup_divergences.append(
                            f"Across '{cat}', {max_g} achieves highest average {target_col} while {min_g} is lowest (spread: {spread})."
                        )
                except Exception:
                    pass

        if subgroup_divergences:
            findings.append({
                "category": "Cohort Divergence",
                "finding": " | ".join(subgroup_divergences),
                "diagnostic_implication": "Subgroup variance explains top-line volatility; personalization or cohort-based pricing recommended."
            })

        diagnostic_insights = {
            "findings": findings,
            "root_cause_factors": [f["finding"] for f in findings]
        }
        state.diagnostic_insights = diagnostic_insights
        
        summary = f"Identified {len(findings)} key diagnostic drivers explaining dataset variance and structural shifts."
        return {"summary": summary, "diagnostic_insights": diagnostic_insights}
