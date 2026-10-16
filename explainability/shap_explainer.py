
"""
SHAP Explainability Engine for AGENTIC SCOUT.

Computes SHAP-based global feature importance and local prediction
breakdowns, with a clearly labelled model-derived importance fallback.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from explainability.feature_importance import FallbackExplainer


class ShapExplainer:
    """Computes global feature importance and local SHAP explanations."""

    def __init__(self, max_samples: int = 100, random_state: int = 42):
        self.max_samples = max(1, int(max_samples))
        self.random_state = random_state

    @staticmethod
    def _as_2d_array(values: Any) -> np.ndarray:
        """Convert input features to a 2D NumPy array."""
        if isinstance(values, pd.DataFrame):
            values = values.to_numpy()

        values = np.asarray(values)

        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if values.ndim != 2:
            raise ValueError("Expected a 2D feature matrix.")

        return values

    @staticmethod
    def _normalize_shap_values(
        values: Any,
        n_samples: int,
        n_features: int,
    ) -> np.ndarray:
        """
        Normalize SHAP output to (samples, features).

        For multiclass output, use the final class as the selected
        output, and document that this is class-specific attribution.
        """
        if hasattr(values, "values"):
            values = values.values

        # Some SHAP versions return a list of arrays, one per class.
        if isinstance(values, list):
            if not values:
                raise ValueError("SHAP returned an empty output list.")
            values = values[-1]

        values = np.asarray(values)

        if values.ndim == 1:
            if n_samples == 1 and values.size == n_features:
                values = values.reshape(1, n_features)
            else:
                raise ValueError("Unexpected one-dimensional SHAP output.")

        elif values.ndim == 3:
            # Common layout: (samples, features, outputs)
            if values.shape[0] == n_samples and values.shape[1] == n_features:
                values = values[:, :, -1]

            # Alternative layout: (outputs, samples, features)
            elif (
                values.shape[1] == n_samples
                and values.shape[2] == n_features
            ):
                values = values[-1, :, :]

            else:
                raise ValueError(
                    f"Unsupported 3D SHAP shape: {values.shape}"
                )

        if values.ndim != 2:
            raise ValueError(
                f"Unsupported SHAP output shape: {values.shape}"
            )

        if values.shape[0] != n_samples:
            raise ValueError(
                "SHAP sample count does not match the input sample count."
            )

        if values.shape[1] != n_features:
            raise ValueError(
                "SHAP feature count does not match the input feature count."
            )

        if not np.isfinite(values).all():
            raise ValueError("SHAP output contains non-finite values.")

        return values.astype(float)

    @staticmethod
    def _get_expected_value(explainer: Any, y_test: np.ndarray) -> float:
        """Return a scalar expected value when available."""
        expected = getattr(explainer, "expected_value", None)

        if expected is not None:
            expected = np.asarray(expected, dtype=float).reshape(-1)

            if expected.size:
                return float(expected[-1])

        y_test = np.asarray(y_test, dtype=float).reshape(-1)
        y_test = y_test[np.isfinite(y_test)]

        return float(np.mean(y_test)) if y_test.size else 0.0

    @staticmethod
    def _feature_direction(
        feature_values: np.ndarray,
        shap_values: np.ndarray,
    ) -> str:
        """
        Describe association between feature values and SHAP values.

        This is descriptive, not causal.
        """
        feature_values = np.asarray(feature_values, dtype=float)
        shap_values = np.asarray(shap_values, dtype=float)

        valid = np.isfinite(feature_values) & np.isfinite(shap_values)
        feature_values = feature_values[valid]
        shap_values = shap_values[valid]

        if len(feature_values) < 2:
            return "Insufficient variation"

        if (
            np.std(feature_values) < 1e-12
            or np.std(shap_values) < 1e-12
        ):
            return "No clear association"

        corr = np.corrcoef(feature_values, shap_values)[0, 1]

        if not np.isfinite(corr) or abs(corr) < 0.10:
            return "No clear linear association"

        if corr > 0:
            return "Positive association with SHAP value"

        return "Negative association with SHAP value"

    def _fallback(
        self,
        estimator: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        reason: str,
    ) -> Dict[str, Any]:
        """Return model-derived importance without mislabelling it as SHAP."""
        try:
            result = FallbackExplainer.explain(
                estimator,
                X_train,
                X_test,
                y_test,
                feature_names.copy(),
            )
        except Exception as fallback_error:
            result = None
            fallback_reason = str(fallback_error)
        else:
            fallback_reason = None

        if (
            not isinstance(result, dict)
            or result.get("importance_df") is None
            or result["importance_df"].empty
        ):
            return {
                "available": False,
                "method": "Attribution Unavailable",
                "importance_df": pd.DataFrame(),
                "top_features": [],
                "base_value": self._get_expected_value(
                    object(), y_test
                ),
                "local_explanation": [],
                "reason": reason,
                "fallback_reason": fallback_reason,
            }

        importance_df = result["importance_df"].copy()

        # Do not invent per-row effects from global importance.
        return {
            **result,
            "available": True,
            "method": (
                "Model-Derived Feature Importance "
                f"({result.get('method', 'Fallback')})"
            ),
            "importance_df": importance_df,
            "top_features": (
                importance_df["Feature"].head(6).tolist()
                if "Feature" in importance_df.columns
                else []
            ),
            "local_explanation": [],
            "reason": (
                "SHAP was unavailable. This is global model-derived "
                "importance, not a local prediction explanation."
            ),
            "shap_available": False,
            "shap_error": reason,
        }

    def explain(
        self,
        estimator: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Calculate SHAP explanations, using a labelled fallback if needed."""

        feature_names = list(feature_names)

        try:
            X_train = self._as_2d_array(X_train)
            X_test = self._as_2d_array(X_test)
            y_test = np.asarray(y_test).reshape(-1)

            if X_train.shape[0] == 0:
                raise ValueError("Training data is empty.")

            if X_test.shape[0] == 0:
                raise ValueError("Test data is empty.")

            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(
                    "Training and test feature counts do not match."
                )

            if X_test.shape[1] != len(feature_names):
                raise ValueError(
                    "Feature names do not match the feature matrix."
                )

            if not np.isfinite(X_train).all():
                raise ValueError(
                    "Training features contain non-finite values."
                )

            if not np.isfinite(X_test).all():
                raise ValueError(
                    "Test features contain non-finite values."
                )

            import shap

            rng = np.random.default_rng(self.random_state)
            n_samples = min(self.max_samples, len(X_test))

            sample_indices = rng.choice(
                len(X_test),
                size=n_samples,
                replace=False,
            )

            X_sample = X_test[sample_indices]

            estimator_type = str(type(estimator)).lower()
            explainer = None
            method_name = None

            # Tree-based models
            if any(
                name in estimator_type
                for name in (
                    "forest",
                    "gradientboosting",
                    "decisiontree",
                    "xgb",
                    "histgradientboosting",
                    "extratrees",
                )
            ):
                try:
                    explainer = shap.TreeExplainer(estimator)
                    method_name = "SHAP (TreeExplainer)"
                except Exception:
                    explainer = None

            # Linear models
            if explainer is None and any(
                name in estimator_type
                for name in ("linear", "logistic", "ridge", "lasso")
            ):
                try:
                    background = X_train[
                        :min(50, len(X_train))
                    ]

                    explainer = shap.LinearExplainer(
                        estimator,
                        background,
                    )
                    method_name = "SHAP (LinearExplainer)"
                except Exception:
                    explainer = None

            # Model-agnostic fallback
            if explainer is None:
                background_size = min(30, len(X_train))

                background = shap.sample(
                    X_train,
                    background_size,
                    random_state=self.random_state,
                )

                explainer = shap.Explainer(
                    estimator.predict,
                    background,
                )

                method_name = "SHAP (Model-Agnostic Explainer)"

            shap_output = explainer(X_sample)

            values = self._normalize_shap_values(
                shap_output,
                n_samples=len(X_sample),
                n_features=X_sample.shape[1],
            )

            mean_abs_shap = np.mean(np.abs(values), axis=0)
            total_importance = float(np.sum(mean_abs_shap))

            importance_pct = (
                mean_abs_shap / total_importance * 100.0
                if total_importance > 0
                else np.zeros_like(mean_abs_shap)
            )

            directions = [
                self._feature_direction(
                    X_sample[:, i],
                    values[:, i],
                )
                for i in range(len(feature_names))
            ]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean_Abs_SHAP": np.round(mean_abs_shap, 6),
                "Importance_Pct": np.round(importance_pct, 2),
                "Direction": directions,
            }).sort_values(
                "Mean_Abs_SHAP",
                ascending=False,
            ).reset_index(drop=True)

            base_value = self._get_expected_value(
                explainer,
                y_test,
            )

            # Local explanation for the first sampled observation.
            local_pairs = sorted(
                zip(feature_names, values[0]),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )[:5]

            local_explanation = [
                f"{feature}: {value:+.4f} SHAP value"
                for feature, value in local_pairs
            ]

            return {
                "available": True,
                "shap_available": True,
                "method": method_name,
                "importance_df": importance_df,
                "top_features": (
                    importance_df["Feature"].head(6).tolist()
                ),
                "base_value": round(base_value, 6),
                "local_explanation": local_explanation,
                "raw_shap_sample": values[:10].tolist(),
                "shap_sample_matrix": values,
                "X_sample": X_sample,
                "reason": None,
            }

        except Exception as error:
            return self._fallback(
                estimator=estimator,
                X_train=np.asarray(X_train),
                X_test=np.asarray(X_test),
                y_test=np.asarray(y_test),
                feature_names=feature_names,
                reason=f"{type(error).__name__}: {error}",
            )