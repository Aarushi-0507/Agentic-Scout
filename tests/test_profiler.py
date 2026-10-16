"""
Unit tests for DataProfiler and QualityAuditor.
"""
import pytest
import numpy as np
import pandas as pd
from core.profiler import DataProfiler
from core.quality import QualityAuditor


@pytest.fixture
def profiler():
    return DataProfiler()


@pytest.fixture
def auditor():
    return QualityAuditor()


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "ID": [f"ID_{i}" for i in range(n)],
        "Age": np.random.randint(20, 65, size=n),
        "Salary": np.random.normal(60000, 15000, size=n).round(2),
        "Department": np.random.choice(["Sales", "Engineering", "Marketing"], size=n),
        "Is_Active": np.random.choice([True, False], size=n)
    })
    # Inject missing values & outliers
    df.loc[0:4, "Salary"] = np.nan
    df.loc[95, "Salary"] = 500000.0  # Outlier
    return df


def test_profiler_metrics(profiler, sample_dataframe):
    res = profiler.profile(sample_dataframe)
    summary = res["summary"]
    
    assert summary["rows"] == 100
    assert summary["columns"] == 5
    assert summary["missing_cells"] == 5
    assert summary["missing_rate_pct"] == 1.0
    assert summary["complete_rows_pct"] == 95.0
    assert "numeric" in res["column_types"]
    assert "Salary" in res["column_types"]["numeric"]
    assert "Department" in res["column_types"]["categorical"]
    assert "ID" in res["column_types"]["id_like"]
    
    # Statistical measures
    num_stats = res["numerical_statistics"]
    assert "Salary" in num_stats
    assert "mean" in num_stats["Salary"]
    assert "median" in num_stats["Salary"]
    assert "variance" in num_stats["Salary"]
    assert "std" in num_stats["Salary"]
    assert "skewness" in num_stats["Salary"]
    
    # Outliers
    outliers = res["outliers"]
    assert "Salary" in outliers
    assert outliers["Salary"]["count"] >= 1


def test_quality_auditor(profiler, auditor, sample_dataframe):
    prof = profiler.profile(sample_dataframe)
    report = auditor.audit(prof)
    
    assert "health_score" in report
    assert 0.0 <= report["health_score"] <= 100.0
    assert report["health_label"] in ["Excellent", "Good", "Fair / Moderate Attention", "Poor / High Risk"]
    assert len(report["issues"]) > 0
    assert "readiness" in report
    assert report["readiness"]["tabular_structure"] is True
