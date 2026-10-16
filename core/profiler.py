"""
Comprehensive Data Profiling Engine for AGENTIC SCOUT.
Calculates rows, columns, types, missingness, row completeness, duplicates,
unique values, high cardinality, mean, median, variance, std dev, skewness,
outliers, and correlation matrices.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class DataProfiler:
    """Profiles tabular datasets across statistical, structural, and distribution dimensions."""
    
    def __init__(self, high_cardinality_threshold: int = 50, outlier_iqr_factor: float = 1.5):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.outlier_iqr_factor = outlier_iqr_factor

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs full automated profiling across the dataset."""
        if df is None or df.empty:
            raise ValueError("Cannot profile an empty DataFrame.")
            
        n_rows, n_cols = df.shape
        
        # 1. Column classifications
        col_types = self._classify_columns(df)
        numeric_cols = col_types["numeric"]
        categorical_cols = col_types["categorical"]
        datetime_cols = col_types["datetime"]
        boolean_cols = col_types["boolean"]
        id_cols = col_types["id_like"]
        
        # 2. Missingness & Completeness
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / n_rows) * 100.0
        total_missing_cells = int(missing_counts.sum())
        total_cells = n_rows * n_cols
        overall_missing_pct = (total_missing_cells / total_cells) * 100.0 if total_cells > 0 else 0.0
        
        complete_rows_count = int(df.dropna().shape[0])
        row_completeness_pct = (complete_rows_count / n_rows) * 100.0 if n_rows > 0 else 0.0
        
        cols_with_missing = {
            col: {
                "count": int(missing_counts[col]),
                "percentage": round(float(missing_pcts[col]), 2)
            }
            for col in df.columns if missing_counts[col] > 0
        }
        
        # 3. Duplicate rows
        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_rows_pct = (duplicate_rows_count / n_rows) * 100.0 if n_rows > 0 else 0.0
        
        # 4. Uniqueness & High Cardinality
        uniqueness = {}
        high_cardinality_cols = []
        for col in df.columns:
            n_unique = int(df[col].nunique(dropna=True))
            ratio = n_unique / n_rows if n_rows > 0 else 0.0
            uniqueness[col] = {
                "unique_count": n_unique,
                "unique_ratio": round(float(ratio), 4)
            }
            if col in categorical_cols:
                if n_unique > self.high_cardinality_threshold or (n_rows > 100 and ratio > 0.4):
                    high_cardinality_cols.append({
                        "column": col,
                        "unique_count": n_unique,
                        "ratio": round(float(ratio), 2)
                    })
                    
        # 5. Numerical Statistics (mean, median, variance, std, skewness)
        num_stats = {}
        outlier_summary = {}
        total_outlier_instances = 0
        
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            var_val = float(series.var(ddof=1)) if len(series) > 1 else 0.0
            min_val = float(series.min())
            max_val = float(series.max())
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            
            # Skewness
            try:
                skew_val = float(series.skew())
                if abs(skew_val) < 0.5:
                    skew_label = "Fairly Symmetric"
                elif 0.5 <= abs(skew_val) < 1.0:
                    skew_label = "Moderately Skewed"
                else:
                    skew_label = "Highly Skewed"
            except Exception:
                skew_val = 0.0
                skew_label = "Unknown"
                
            # Outlier detection via IQR
            lower_bound = q25 - (self.outlier_iqr_factor * iqr)
            upper_bound = q75 + (self.outlier_iqr_factor * iqr)
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_count = int(len(outliers))
            outlier_pct = (outlier_count / len(series)) * 100.0 if len(series) > 0 else 0.0
            total_outlier_instances += outlier_count
            
            num_stats[col] = {
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "std": round(std_val, 4),
                "variance": round(var_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr, 4),
                "skewness": round(skew_val, 4),
                "skew_interpretation": skew_label,
            }
            
            outlier_summary[col] = {
                "count": outlier_count,
                "percentage": round(outlier_pct, 2),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4)
            }
            
        # 6. Categorical Distributions
        cat_stats = {}
        for col in categorical_cols:
            series = df[col].dropna()
            top_vals = series.value_counts(normalize=True).head(5).to_dict()
            cat_stats[col] = {
                "unique_count": int(df[col].nunique()),
                "top_categories_pct": {str(k): round(float(v) * 100, 2) for k, v in top_vals.items()}
            }
            
        # 7. Correlation Analysis (Numeric)
        correlations = self._calculate_correlations(df, numeric_cols)
        
        return {
            "summary": {
                "rows": n_rows,
                "columns": n_cols,
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "datetime_columns_count": len(datetime_cols),
                "boolean_columns_count": len(boolean_cols),
                "id_columns_count": len(id_cols),
                "total_cells": total_cells,
                "missing_cells": total_missing_cells,
                "missing_rate_pct": round(overall_missing_pct, 2),
                "complete_rows_pct": round(row_completeness_pct, 2),
                "duplicate_rows_count": duplicate_rows_count,
                "duplicate_rows_pct": round(duplicate_rows_pct, 2),
                "total_outliers": total_outlier_instances
            },
            "column_types": col_types,
            "missing_values": {
                "columns_with_missing": cols_with_missing,
                "overall_missing_pct": round(overall_missing_pct, 2),
                "complete_rows_pct": round(row_completeness_pct, 2)
            },
            "uniqueness": uniqueness,
            "high_cardinality_columns": high_cardinality_cols,
            "numerical_statistics": num_stats,
            "categorical_statistics": cat_stats,
            "outliers": outlier_summary,
            "correlations": correlations
        }

    def _classify_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize each column into semantic types."""
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        boolean_cols = []
        id_cols = []
        
        n_rows = len(df)
        
        for col in df.columns:
            series = df[col]
            col_name_lower = str(col).lower()
            
            # Check boolean
            if pd.api.types.is_bool_dtype(series):
                boolean_cols.append(col)
                continue
                
            # Check datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
                continue
                
            # Check ID-like
            n_unique = series.nunique(dropna=True)
            is_id_name = any(k in col_name_lower for k in ["id", "uuid", "key", "identifier", "code", "hash"])
            if is_id_name and (n_unique / n_rows > 0.85 if n_rows > 10 else False):
                id_cols.append(col)
                continue
                
            # Numeric
            if pd.api.types.is_numeric_dtype(series):
                # If numeric but only 2 unique values and name sounds boolean
                if n_unique <= 2 and series.dropna().isin([0, 1]).all():
                    boolean_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                # Text/Categorical
                categorical_cols.append(col)
                
        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "boolean": boolean_cols,
            "id_like": id_cols
        }

    def _calculate_correlations(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        """Calculates Pearson & Spearman correlation matrices and flags collinear pairs."""
        if len(numeric_cols) < 2:
            return {
                "pearson": {},
                "spearman": {},
                "top_positive": [],
                "top_negative": [],
                "collinear_pairs": []
            }
            
        num_df = df[numeric_cols].dropna()
        if len(num_df) < 3:
            return {
                "pearson": {},
                "spearman": {},
                "top_positive": [],
                "top_negative": [],
                "collinear_pairs": []
            }
            
        try:
            pearson_corr = num_df.corr(method="pearson").round(4)
            spearman_corr = num_df.corr(method="spearman").round(4)
            
            pairs = []
            cols = pearson_corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1, c2 = cols[i], cols[j]
                    val = float(pearson_corr.loc[c1, c2])
                    if not np.isnan(val):
                        pairs.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "correlation": round(val, 4),
                            "abs_correlation": round(abs(val), 4)
                        })
                        
            pairs.sort(key=lambda x: x["correlation"], reverse=True)
            top_pos = [p for p in pairs if p["correlation"] > 0][:8]
            top_neg = [p for p in sorted(pairs, key=lambda x: x["correlation"]) if p["correlation"] < 0][:8]
            collinear = [p for p in pairs if p["abs_correlation"] >= 0.85]
            
            return {
                "pearson": pearson_corr.to_dict(),
                "spearman": spearman_corr.to_dict(),
                "top_positive": top_pos,
                "top_negative": top_neg,
                "collinear_pairs": collinear
            }
        except Exception:
            return {
                "pearson": {},
                "spearman": {},
                "top_positive": [],
                "top_negative": [],
                "collinear_pairs": []
            }
