"""
Data Profiling Agent for AGENTIC SCOUT.
Executes statistical profiling, data quality auditing, and schema inference.
"""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from core.state import AnalysisState
from core.profiler import DataProfiler
from core.quality import QualityAuditor


class ProfilingAgent(BaseAgent):
    """Understands and profiles tabular data, missing values, duplicates, and health score."""

    def __init__(self):
        super().__init__(name="Data Profiling Agent", role="Structural & Statistical Profiling")
        self.profiler = DataProfiler()
        self.auditor = QualityAuditor()

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        if state.df is None or state.df.empty:
            raise ValueError("No active dataset in AnalysisState to profile.")

        # Run profiling
        profile = self.profiler.profile(state.df)
        state.profile = profile

        # Run quality audit
        quality_report = self.auditor.audit(profile)
        state.quality_report = quality_report

        n_rows = profile["summary"]["rows"]
        n_cols = profile["summary"]["columns"]
        health = quality_report["health_score"]

        summary = f"Profiled {n_rows:,} records across {n_cols} columns. Data Health Score: {health}/100 ({quality_report['health_label']})."
        return {
            "summary": summary,
            "profile": profile,
            "quality_report": quality_report
        }
