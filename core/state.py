"""
AnalysisState: Central container holding all analysis artifacts, datasets, metrics, and agent logs.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import time


@dataclass
class AnalysisState:
    dataset_name: str = "dataset"
    raw_df: Optional[pd.DataFrame] = None
    df: Optional[pd.DataFrame] = None
    
    # Core artifacts
    profile: Dict[str, Any] = field(default_factory=dict)
    quality_report: Dict[str, Any] = field(default_factory=dict)
    eda_insights: Dict[str, Any] = field(default_factory=dict)
    diagnostic_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Problem detection & config
    problem_detection: Dict[str, Any] = field(default_factory=dict)
    selected_problem: Optional[str] = None
    selected_target: Optional[str] = None
    selected_datetime_col: Optional[str] = None
    
    # Modeling & results
    ml_results: Dict[str, Any] = field(default_factory=dict)
    explainability: Dict[str, Any] = field(default_factory=dict)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    final_report: Dict[str, Any] = field(default_factory=dict)
    
    data_signature: str = ""
    
    # Execution status & Partial Success reporting
    execution_summary: Dict[str, Any] = field(default_factory=lambda: {
        "status": "NOT_STARTED",
        "successful_agents": [],
        "failed_agents": [],
        "warnings": []
    })
    
    # Agent execution audit
    agent_logs: List[Dict[str, Any]] = field(default_factory=list)
    agent_statuses: Dict[str, str] = field(default_factory=lambda: {
        "Data Profiling Agent": "IDLE",
        "EDA Agent": "IDLE",
        "Diagnostic Agent": "IDLE",
        "Predictive Agent": "IDLE",
        "Clustering Agent": "IDLE",
        "Prescriptive Agent": "IDLE",
        "Explanation Agent": "IDLE",
        "Orchestrator Agent": "IDLE",
    })

    def log_agent_action(self, agent_name: str, status: str, message: str, details: Optional[Dict[str, Any]] = None, duration_ms: float = 0.0):
        self.agent_statuses[agent_name] = status
        entry = {
            "agent": agent_name,
            "status": status,
            "message": message,
            "details": details or {},
            "duration_ms": round(duration_ms, 2),
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.agent_logs.append(entry)

    def is_pipeline_ready(self) -> bool:
        return self.df is not None and not self.df.empty

    def has_ml_results(self) -> bool:
        return bool(self.ml_results and "leaderboard" in self.ml_results)

    def reset_downstream_artifacts(self, preserve_profile: bool = True):
        """Clears stale downstream ML, explainability, recommendations, and report artifacts."""
        self.ml_results = {}
        self.explainability = {}
        self.recommendations = {}
        self.final_report = {}
        self.agent_logs = []
        self.agent_statuses = {k: "IDLE" for k in self.agent_statuses}
        self.execution_summary = {
            "status": "NOT_STARTED",
            "successful_agents": [],
            "failed_agents": [],
            "warnings": []
        }
        if not preserve_profile:
            self.profile = {}
            self.quality_report = {}
            self.eda_insights = {}
            self.diagnostic_insights = {}
            self.problem_detection = {}
