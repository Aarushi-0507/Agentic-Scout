"""
Model Evaluation and Leaderboard Ranking Engine for AGENTIC SCOUT.
Calculates MAE, RMSE, R² for Regression;
Accuracy, Precision, Recall, F1, ROC-AUC for Classification;
Silhouette Score, Davies-Bouldin for Clustering;
MAE, RMSE, MAPE for Forecasting.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


class Evaluator:
    """Computes rigorous performance metrics across problem types and compiles model leaderboards."""

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates MAE, RMSE, R²."""
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        # Mean Absolute Percentage Error (MAPE)
        mask = y_true != 0
        if np.sum(mask) > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        else:
            mape = 0.0
            
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2)
        }

    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculates Accuracy, Precision, Recall, F1, ROC-AUC."""
        acc = float(accuracy_score(y_true, y_pred))
        
        # Determine average type
        n_classes = len(np.unique(y_true))
        avg = "binary" if n_classes <= 2 else "macro"
        
        prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        rec = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
        
        # ROC AUC
        roc_auc = 0.0
        if y_prob is not None:
            try:
                if n_classes <= 2:
                    # binary
                    prob_pos = y_prob[:, 1] if y_prob.ndim > 1 and y_prob.shape[1] > 1 else y_prob
                    roc_auc = float(roc_auc_score(y_true, prob_pos))
                else:
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                roc_auc = 0.0
                
        return {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4)
        }

    @staticmethod
    def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculates Silhouette Score, Davies-Bouldin, Calinski-Harabasz."""
        unique_labels = np.unique(labels)
        # Exclude noise label (-1 from DBSCAN) if multiple clusters exist
        valid_mask = labels != -1
        n_clusters = len(np.unique(labels[valid_mask]))
        
        if n_clusters < 2 or len(labels) < 4:
            return {
                "Silhouette": -1.0,
                "Davies_Bouldin": 99.0,
                "Calinski_Harabasz": 0.0,
                "Clusters_Found": int(n_clusters)
            }
            
        try:
            sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
            db = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
            ch = float(calinski_harabasz_score(X[valid_mask], labels[valid_mask]))
        except Exception:
            sil, db, ch = 0.0, 0.0, 0.0
            
        return {
            "Silhouette": round(sil, 4),
            "Davies_Bouldin": round(db, 4),
            "Calinski_Harabasz": round(ch, 2),
            "Clusters_Found": int(n_clusters)
        }

    @staticmethod
    def build_leaderboard(models_dict: Dict[str, Dict[str, Any]], problem_type: str) -> pd.DataFrame:
        """
        Builds a ranked leaderboard DataFrame and marks the champion model.
        """
        rows = []
        for name, data in models_dict.items():
            metrics = data.get("test_metrics", {})
            row = {"Model": name}
            row.update(metrics)
            row["Fit_Time_ms"] = data.get("fit_time_ms", 0.0)
            rows.append(row)
            
        if not rows:
            return pd.DataFrame()
            
        df_lead = pd.DataFrame(rows)
        
        # Rank models
        if "Regression" in problem_type or "Forecasting" in problem_type:
            # Rank by R2 descending or RMSE ascending
            if "R2" in df_lead.columns:
                df_lead = df_lead.sort_values(by=["R2", "RMSE"], ascending=[False, True])
            else:
                df_lead = df_lead.sort_values(by=["RMSE"], ascending=[True])
        elif "Classification" in problem_type:
            # Rank by F1 descending, then Accuracy
            if "F1" in df_lead.columns:
                df_lead = df_lead.sort_values(by=["F1", "Accuracy"], ascending=[False, False])
            else:
                df_lead = df_lead.sort_values(by=["Accuracy"], ascending=[False])
        elif "Clustering" in problem_type:
            # Rank by Silhouette descending
            if "Silhouette" in df_lead.columns:
                df_lead = df_lead.sort_values(by=["Silhouette"], ascending=[False])
                
        df_lead.reset_index(drop=True, inplace=True)
        df_lead["Rank"] = [f"#{i+1}" for i in range(len(df_lead))]
        # Mark champion
        df_lead["Champion"] = ["★ Champion" if i == 0 else "" for i in range(len(df_lead))]
        
        # Put Rank and Champion in front
        cols = ["Rank", "Champion", "Model"] + [c for c in df_lead.columns if c not in ["Rank", "Champion", "Model"]]
        return df_lead[cols]
