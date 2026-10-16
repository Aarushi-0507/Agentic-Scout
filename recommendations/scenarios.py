"""
Interactive What-If Scenario Simulation Engine for AGENTIC SCOUT.
Quantifies projected impact of hypothetical policy levers using fitted ML models.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class ScenarioSimulator:
    """Executes sensitivity tests and counterfactual simulations across controllable levers."""

    def __init__(self, estimator: Any, preprocessor: Any, raw_df: pd.DataFrame, target_col: Optional[str] = None):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.raw_df = raw_df
        self.target_col = target_col

    def get_baseline_row(self) -> pd.DataFrame:
        """Constructs a representative median/mode baseline profile."""
        baseline = {}
        for col in self.raw_df.columns:
            if col == self.target_col:
                continue
            if pd.api.types.is_numeric_dtype(self.raw_df[col]):
                baseline[col] = float(self.raw_df[col].median())
            elif pd.api.types.is_datetime64_any_dtype(self.raw_df[col]):
                baseline[col] = self.raw_df[col].iloc[-1]
            else:
                mode_vals = self.raw_df[col].mode()
                baseline[col] = str(mode_vals.iloc[0]) if not mode_vals.empty else "Unknown"
                
        return pd.DataFrame([baseline])

    def simulate(self, feature_adjustments: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulates the effect of altering feature values on the predicted outcome.
        feature_adjustments: e.g. {"Marketing_Spend": 1.15 (multiplier), "Discount_Pct": -5.0 (additive)}
        """
        base_row = self.get_baseline_row()
        sim_row = base_row.copy()
        
        applied_changes = []
        
        for feat, val in feature_adjustments.items():
            # Check match
            matched_col = None
            for col in base_row.columns:
                if col.lower() == feat.lower():
                    matched_col = col
                    break
                    
            if matched_col and pd.api.types.is_numeric_dtype(base_row[matched_col]):
                orig_val = float(base_row[matched_col].iloc[0])
                new_val = float(val)
                sim_row[matched_col] = new_val
                diff = new_val - orig_val
                pct_diff = (diff / orig_val) * 100.0 if orig_val != 0 else 0.0
                applied_changes.append({
                    "feature": matched_col,
                    "baseline_value": round(orig_val, 2),
                    "simulated_value": round(new_val, 2),
                    "change_pct": round(pct_diff, 1)
                })

        # Preprocess and predict
        try:
            X_base = self.preprocessor.transform_single(base_row)
            X_sim = self.preprocessor.transform_single(sim_row)
            
            # Predict
            if hasattr(self.estimator, "predict_proba"):
                # Classification probability of class 1
                base_prob = self.estimator.predict_proba(X_base)[0]
                sim_prob = self.estimator.predict_proba(X_sim)[0]
                base_pred = float(base_prob[1]) if len(base_prob) > 1 else float(base_prob[0])
                sim_pred = float(sim_prob[1]) if len(sim_prob) > 1 else float(sim_prob[0])
                is_prob = True
            else:
                base_pred = float(self.estimator.predict(X_base)[0])
                sim_pred = float(self.estimator.predict(X_sim)[0])
                is_prob = False

            delta = sim_pred - base_pred
            pct_change = (delta / base_pred * 100.0) if base_pred != 0 else 0.0

            assumptions = [
                "Ceteris Paribus: Non-adjusted features remain fixed at representative dataset baseline values (medians/modes).",
                "Non-Causal Statistical Association: Predictions quantify model response under hypothetical inputs; they do not prove direct causation or guarantee real-world outcomes.",
                "Distributional Validity: Assumes future operating conditions and unobserved confounders remain consistent with training distributions."
            ]

            return {
                "is_model_scenario": True,
                "is_causal": False,
                "is_probability": is_prob,
                "baseline_prediction": round(base_pred, 4),
                "simulated_prediction": round(sim_pred, 4),
                "delta": round(delta, 4),
                "percentage_change": round(pct_change, 2),
                "applied_changes": applied_changes,
                "assumptions": assumptions,
                "disclaimer": "Model-based scenario projection. Statistical correlation does not establish causal efficacy."
            }
        except Exception as e:
            return {
                "is_model_scenario": True,
                "is_causal": False,
                "error": str(e),
                "baseline_prediction": 0.0,
                "simulated_prediction": 0.0,
                "delta": 0.0,
                "percentage_change": 0.0,
                "applied_changes": [],
                "assumptions": [],
                "disclaimer": "Simulation error occurred."
            }
