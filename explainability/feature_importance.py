"""Reliable model-derived feature importance (not SHAP attribution)."""
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class FallbackExplainer:
    """Compute global importance with explicit labels and safe input handling."""

    @staticmethod
    def explain(estimator: Any, X_train: Any, X_test: Any, y_test: Any,
                feature_names: List[str]) -> Dict[str, Any]:
        names = list(feature_names)
        X = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
        y = np.asarray(y_test).reshape(-1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X_test must be a non-empty 2D matrix")
        if X.shape[0] != len(y):
            raise ValueError("X_test and y_test row counts do not match")
        if X.shape[1] != len(names):
            raise ValueError("feature_names length does not match X_test columns")
        if not np.isfinite(X.astype(float)).all():
            raise ValueError("X_test contains non-finite or non-numeric values")

        method = ""
        if hasattr(estimator, "feature_importances_"):
            raw = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
            method = "Tree Impurity Importance"
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            raw = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef).reshape(-1)
            method = "Absolute Model Coefficients"
        else:
            result = permutation_importance(estimator, X, y, n_repeats=5, random_state=42)
            raw = np.maximum(result.importances_mean, 0.0)
            method = "Permutation Importance"

        if len(raw) != len(names):
            raise ValueError(f"Estimator returned {len(raw)} importance values for {len(names)} features")
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.maximum(raw, 0.0)
        total = float(raw.sum())
        pct = raw / total * 100 if total > 0 else np.zeros_like(raw)

        # Correlation with predictions is descriptive association, not causal direction.
        try:
            pred = np.asarray(estimator.predict(X)).reshape(-1)
            directions = []
            for i in range(X.shape[1]):
                col = X[:, i].astype(float)
                corr = np.corrcoef(col, pred)[0, 1] if np.std(col) > 1e-12 and np.std(pred) > 1e-12 else np.nan
                directions.append("Positive association with prediction" if corr > .05 else
                                  "Negative association with prediction" if corr < -.05 else
                                  "No clear linear association")
        except Exception:
            directions = ["Unknown"] * len(names)

        frame = pd.DataFrame({"Feature": names, "Importance_Score": raw,
                              "Importance_Pct": np.round(pct, 2), "Direction": directions})
        frame = frame.sort_values("Importance_Score", ascending=False).reset_index(drop=True)
        return {"available": bool(total > 0), "method": method, "importance_df": frame,
                "top_features": frame["Feature"].head(6).tolist() if total > 0 else [],
                "summary_dict": dict(zip(frame["Feature"], frame["Importance_Pct"])),
                "reason": None if total > 0 else "The model produced no positive importance scores."}
