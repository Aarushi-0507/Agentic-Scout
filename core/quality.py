"""
Data Quality Auditor and Health Score calculation engine for AGENTIC SCOUT.
Evaluates data completeness, uniqueness, anomalies, and analytical readiness.
"""
from typing import Dict, Any, List
import pandas as pd


class QualityAuditor:
    """Audits data health and flags structural, distribution, and integrity risks."""
    
    def __init__(self, missing_warn_pct: float = 5.0, missing_crit_pct: float = 25.0, dup_warn_pct: float = 2.0):
        self.missing_warn_pct = missing_warn_pct
        self.missing_crit_pct = missing_crit_pct
        self.dup_warn_pct = dup_warn_pct

    def audit(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates profile results to compute health score, issues, and readiness."""
        summary = profile.get("summary", {})
        missing_vals = profile.get("missing_values", {})
        outliers = profile.get("outliers", {})
        high_card = profile.get("high_cardinality_columns", [])
        col_types = profile.get("column_types", {})
        
        n_rows = summary.get("rows", 0)
        n_cols = summary.get("columns", 0)
        missing_rate = summary.get("missing_rate_pct", 0.0)
        dup_pct = summary.get("duplicate_rows_pct", 0.0)
        
        # 1. Base Score calculation (Start at 100)
        score = 100.0
        issues: List[Dict[str, Any]] = []
        
        # Missing values penalty
        if missing_rate > self.missing_crit_pct:
            penalty = min(35.0, missing_rate * 0.8)
            score -= penalty
            issues.append({
                "severity": "CRITICAL",
                "category": "Missing Data",
                "message": f"Severe missingness detected: {missing_rate}% of all cells are missing.",
                "remediation": "Impute numerical columns with median/KNN and categorical with mode, or consider dropping features with >50% missingness."
            })
        elif missing_rate > self.missing_warn_pct:
            penalty = min(15.0, missing_rate * 0.5)
            score -= penalty
            issues.append({
                "severity": "WARNING",
                "category": "Missing Data",
                "message": f"Moderate missingness detected: {missing_rate}% of cells are missing.",
                "remediation": "Apply standard median/mode imputation prior to modeling."
            })
        else:
            issues.append({
                "severity": "INFO",
                "category": "Missing Data",
                "message": f"Missing rate is healthy ({missing_rate}%).",
                "remediation": "Standard pipeline imputation will handle residual missing entries smoothly."
            })
            
        # Duplicate rows penalty
        if dup_pct > self.dup_warn_pct:
            dup_penalty = min(20.0, dup_pct * 1.5)
            score -= dup_penalty
            issues.append({
                "severity": "WARNING" if dup_pct < 10.0 else "CRITICAL",
                "category": "Duplicates",
                "message": f"{dup_pct}% duplicate rows found ({summary.get('duplicate_rows_count', 0)} rows).",
                "remediation": "Deduplicate identical records before model training to prevent data leakage and sample bias."
            })
            
        # Outlier evaluation
        total_outliers = summary.get("total_outliers", 0)
        outlier_cols = [c for c, stat in outliers.items() if stat.get("percentage", 0) > 5.0]
        if outlier_cols:
            score -= min(15.0, len(outlier_cols) * 3.0)
            issues.append({
                "severity": "WARNING",
                "category": "Outliers",
                "message": f"{len(outlier_cols)} numerical columns show >5% extreme values via IQR bounds ({', '.join(outlier_cols[:4])}).",
                "remediation": "Use robust scaling or tree-based algorithms (Random Forest, Gradient Boosting) that are resilient to heavy tails."
            })
            
        # High cardinality / ID columns
        if high_card:
            score -= min(10.0, len(high_card) * 2.0)
            issues.append({
                "severity": "WARNING",
                "category": "High Cardinality",
                "message": f"{len(high_card)} categorical columns have unusually high cardinality.",
                "remediation": "Exclude unique identifiers from predictors and target-encode or frequency-encode high-cardinality nominal features."
            })
            
        # Sample size audit
        if n_rows < 50:
            score -= 20.0
            issues.append({
                "severity": "CRITICAL",
                "category": "Sample Size",
                "message": f"Small dataset: only {n_rows} rows. High risk of model overfitting.",
                "remediation": "Prioritize simple regularized models (Ridge/Lasso/Logistic) with strict cross-validation."
            })
        elif n_rows < 200:
            score -= 5.0
            issues.append({
                "severity": "WARNING",
                "category": "Sample Size",
                "message": f"Modest dataset size ({n_rows} rows).",
                "remediation": "Use cross-validation with at least 3-5 folds."
            })

        final_score = max(5.0, min(100.0, round(score, 1)))
        
        # Determine Health Label
        if final_score >= 85.0:
            health_label = "Excellent"
            health_badge = "success"
        elif final_score >= 70.0:
            health_label = "Good"
            health_badge = "primary"
        elif final_score >= 50.0:
            health_label = "Fair / Moderate Attention"
            health_badge = "warning"
        else:
            health_label = "Poor / High Risk"
            health_badge = "danger"
            
        # Readiness Checklist
        readiness = {
            "tabular_structure": n_rows >= 10 and n_cols >= 2,
            "has_numeric_features": len(col_types.get("numeric", [])) >= 1,
            "missing_acceptable": missing_rate < 50.0,
            "regression_ready": len(col_types.get("numeric", [])) >= 2,
            "classification_ready": (len(col_types.get("categorical", [])) + len(col_types.get("boolean", [])) >= 1) and len(col_types.get("numeric", [])) >= 1,
            "forecasting_ready": len(col_types.get("datetime", [])) >= 1 and len(col_types.get("numeric", [])) >= 1,
            "clustering_ready": len(col_types.get("numeric", [])) >= 2,
        }
        
        return {
            "health_score": final_score,
            "health_label": health_label,
            "health_badge": health_badge,
            "issues": issues,
            "issues_count_by_severity": {
                "CRITICAL": sum(1 for i in issues if i["severity"] == "CRITICAL"),
                "WARNING": sum(1 for i in issues if i["severity"] == "WARNING"),
                "INFO": sum(1 for i in issues if i["severity"] == "INFO"),
            },
            "readiness": readiness
        }
