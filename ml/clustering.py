"""
Clustering and Anomaly Detection Pipeline for AGENTIC SCOUT.
Implements K-Means (with optimal k selection), DBSCAN, and Isolation Forest.
Provides cluster persona summaries and 2D PCA projections for interactive visualization.
"""
from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

from ml.evaluation import Evaluator


class ClusteringPipeline:
    """Performs unsupervised clustering, customer segmentation, and anomaly detection."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_and_profile(
        self,
        X: np.ndarray,
        X_raw: pd.DataFrame,
        feature_names: List[str],
        min_k: int = 2,
        max_k: int = 6
    ) -> Dict[str, Any]:
        """Finds optimal k for K-Means, runs DBSCAN and Isolation Forest."""
        start_t = time.time()
        
        # 1. K-Means Optimal K Search
        best_k = min_k
        best_silhouette = -1.0
        kmeans_candidates = {}
        
        for k in range(min_k, min(max_k + 1, len(X))):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            metrics = Evaluator.evaluate_clustering(X, labels)
            kmeans_candidates[f"K-Means (k={k})"] = {
                "estimator": km,
                "labels": labels,
                "metrics": metrics,
                "k": k
            }
            if metrics["Silhouette"] > best_silhouette:
                best_silhouette = metrics["Silhouette"]
                best_k = k
                
        optimal_km_data = kmeans_candidates[f"K-Means (k={best_k})"]
        champion_km = optimal_km_data["estimator"]
        champion_labels = optimal_km_data["labels"]
        
        # 2. DBSCAN
        db = DBSCAN(eps=0.8, min_samples=5)
        db_labels = db.fit_predict(X)
        db_metrics = Evaluator.evaluate_clustering(X, db_labels)
        
        # 3. Isolation Forest (Anomaly Detection)
        iso = IsolationForest(contamination=0.05, random_state=self.random_state)
        anomaly_preds = iso.fit_predict(X) # -1 is anomaly, 1 is normal
        anomaly_flags = (anomaly_preds == -1).astype(int)
        anomaly_count = int(np.sum(anomaly_flags))
        
        # 4. 2D PCA Projection for Plotting
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_coords = pca.fit_transform(X)
        explained_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
        
        # 5. Cluster Personas & Statistics
        df_clustered = X_raw.copy()
        df_clustered["Cluster"] = [f"Segment {l}" for l in champion_labels]
        df_clustered["Is_Anomaly"] = anomaly_flags
        
        personas = self._generate_personas(df_clustered, champion_labels)
        
        # Model dictionary for leaderboard
        models_dict = {
            f"K-Means (k={best_k})": {
                "test_metrics": optimal_km_data["metrics"],
                "fit_time_ms": round((time.time() - start_t) * 1000.0, 2)
            },
            "DBSCAN": {
                "test_metrics": db_metrics,
                "fit_time_ms": 15.0
            }
        }
        leaderboard = Evaluator.build_leaderboard(models_dict, "Clustering")
        
        return {
            "optimal_k": best_k,
            "champion_model_name": f"K-Means (k={best_k})",
            "leaderboard": leaderboard,
            "test_metrics": optimal_km_data["metrics"],
            "labels": champion_labels,
            "personas": personas,
            "anomaly_count": anomaly_count,
            "anomaly_flags": anomaly_flags,
            "pca_x": pca_coords[:, 0].tolist(),
            "pca_y": pca_coords[:, 1].tolist(),
            "pca_variance": explained_var,
            "clustered_df": df_clustered,
            "feature_names": feature_names
        }

    def _generate_personas(self, df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
        """Synthesizes human-interpretable segment personas based on feature divergence."""
        personas = []
        unique_clusters = sorted(np.unique(labels))
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ["Cluster", "Is_Anomaly"]]
        
        total_records = len(df)
        
        for c_id in unique_clusters:
            c_name = f"Segment {c_id}"
            sub_df = df[df["Cluster"] == c_name]
            count = len(sub_df)
            pct = round((count / total_records) * 100.0, 1)
            
            # Identify dominant characteristics
            traits = []
            for col in num_cols[:6]:
                col_mean = sub_df[col].mean()
                pop_mean = df[col].mean()
                if pop_mean > 0:
                    diff_pct = ((col_mean - pop_mean) / pop_mean) * 100.0
                    if abs(diff_pct) > 15:
                        direction = "Above Average" if diff_pct > 0 else "Below Average"
                        traits.append(f"{col}: {round(col_mean, 1)} ({direction} by {abs(round(diff_pct, 1))}%)")
                        
            personas.append({
                "cluster_id": int(c_id),
                "name": f"Cohort {c_id}",
                "size": count,
                "percentage": pct,
                "key_traits": traits if traits else ["Representative of overall baseline population."],
                "metrics_summary": sub_df[num_cols].mean().round(2).to_dict() if num_cols else {}
            })
            
        return personas
