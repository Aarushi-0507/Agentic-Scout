"""
Global settings and configuration for AGENTIC SCOUT.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass(frozen=True)
class AppConfig:
    APP_NAME: str = "AGENTIC SCOUT"
    APP_SUBTITLE: str = "Autonomous AI Framework for Automated Business Data Analysis & Decision Support"
    VERSION: str = "1.0.0"
    SUPPORTED_EXTENSIONS: List[str] = field(default_factory=lambda: [".csv", ".xlsx", ".xls"])
    MAX_FILE_SIZE_MB: int = 200
    DEFAULT_SAMPLE_SIZE: int = 10000


@dataclass(frozen=True)
class MLConfig:
    RANDOM_SEED: int = 42
    TEST_SPLIT_RATIO: float = 0.20
    CV_FOLDS: int = 3
    
    # Clustering
    CLUSTERING_MIN_K: int = 2
    CLUSTERING_MAX_K: int = 7
    DBSCAN_EPS: float = 0.5
    DBSCAN_MIN_SAMPLES: int = 5
    
    # Forecasting
    FORECAST_HORIZON: int = 14
    MAX_LAGS: int = 7
    
    # Explainability
    SHAP_MAX_SAMPLES: int = 100
    TOP_FEATURES_COUNT: int = 8


@dataclass(frozen=True)
class AgentConfig:
    AGENT_NAMES: List[str] = field(default_factory=lambda: [
        "Data Profiling Agent",
        "EDA Agent",
        "Diagnostic Agent",
        "Predictive Agent",
        "Clustering Agent",
        "Prescriptive Agent",
        "Explanation Agent",
        "Orchestrator Agent",
    ])
    
    # Quality thresholds
    MISSING_WARNING_PCT: float = 5.0
    MISSING_CRITICAL_PCT: float = 25.0
    DUPLICATE_WARNING_PCT: float = 2.0
    HIGH_CARDINALITY_THRESHOLD: int = 50
    OUTLIER_IQR_FACTOR: float = 1.5


# Theme and palette for visualization
THEME_COLORS = {
    "primary": "#2563EB",       # Royal Blue
    "secondary": "#7C3AED",     # Violet
    "accent": "#06B6D4",        # Cyan
    "success": "#10B981",       # Emerald Green
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Red
    "background_dark": "#0F172A",
    "card_dark": "#1E293B",
    "text_dark": "#F8FAFC",
    "text_muted": "#94A3B8",
}
