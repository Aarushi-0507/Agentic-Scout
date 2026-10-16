"""
Orchestrator Agent for AGENTIC SCOUT.
Manages end-to-end pipeline execution, coordinates specialized autonomous agents,
and synthesizes a deterministic executive report.
"""
from typing import Dict, Any
import time
import pandas as pd
from agents.base_agent import BaseAgent
from agents.profiling_agent import ProfilingAgent
from agents.eda_agent import EDAAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.predictive_agent import PredictiveAgent
from agents.clustering_agent import ClusteringAgent
from agents.explanation_agent import ExplanationAgent
from agents.prescriptive_agent import PrescriptiveAgent
from core.state import AnalysisState
from core.problem_detector import ProblemDetector


class OrchestratorAgent(BaseAgent):
    """Coordinates agent execution lifecycle and produces the final executive briefing."""

    def __init__(self):
        super().__init__(name="Orchestrator Agent", role="Pipeline Orchestration & Executive Reporting")
        self.profiling_agent = ProfilingAgent()
        self.eda_agent = EDAAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.problem_detector = ProblemDetector()
        self.predictive_agent = PredictiveAgent()
        self.clustering_agent = ClusteringAgent()
        self.explanation_agent = ExplanationAgent()
        self.prescriptive_agent = PrescriptiveAgent()

    def execute(self, state: AnalysisState) -> Dict[str, Any]:
        """Runs the 8-agent end-to-end workflow with agent-level error tracking and partial-success reporting."""
        if not state.is_pipeline_ready():
            raise ValueError("No valid dataset loaded in AnalysisState.")

        successful_agents = []
        failed_agents = []
        warnings = []

        # 1. Data Profiling & Quality Audit
        try:
            self.profiling_agent.run(state)
            successful_agents.append(self.profiling_agent.name)
        except Exception as e:
            failed_agents.append({"agent": self.profiling_agent.name, "error": str(e)})
            state.log_agent_action(self.profiling_agent.name, "FAILED", f"Data profiling failed: {str(e)}")

        # 2. Exploratory Data Analysis
        try:
            self.eda_agent.run(state)
            successful_agents.append(self.eda_agent.name)
        except Exception as e:
            failed_agents.append({"agent": self.eda_agent.name, "error": str(e)})
            state.log_agent_action(self.eda_agent.name, "FAILED", f"EDA failed: {str(e)}")

        # 3. Problem Detection & Decision Matrix
        try:
            det_result = self.problem_detector.detect(
                df=state.df,
                profile=state.profile,
                user_target=state.selected_target
            )
            state.problem_detection = det_result
            if not state.selected_problem:
                state.selected_problem = det_result["detected_problem"]
            if not state.selected_target:
                state.selected_target = det_result["target_column"]
            if not state.selected_datetime_col:
                state.selected_datetime_col = det_result["datetime_column"]
            successful_agents.append("Problem Detector")
        except Exception as e:
            failed_agents.append({"agent": "Problem Detector", "error": str(e)})
            state.log_agent_action("Problem Detector", "FAILED", f"Problem detection failed: {str(e)}")

        # 4. Diagnostic Agent
        try:
            self.diagnostic_agent.run(state)
            successful_agents.append(self.diagnostic_agent.name)
        except Exception as e:
            failed_agents.append({"agent": self.diagnostic_agent.name, "error": str(e)})
            state.log_agent_action(self.diagnostic_agent.name, "FAILED", f"Diagnostic analysis failed: {str(e)}")

        # 5. Machine Learning Modeling
        active_problem = state.selected_problem or state.problem_detection.get("detected_problem", "Regression")
        if "Clustering" in active_problem or "Anomaly" in active_problem:
            try:
                self.clustering_agent.run(state)
                successful_agents.append(self.clustering_agent.name)
            except Exception as e:
                failed_agents.append({"agent": self.clustering_agent.name, "error": str(e)})
                state.log_agent_action(self.clustering_agent.name, "FAILED", f"Clustering failed: {str(e)}")
        else:
            try:
                self.predictive_agent.run(state)
                successful_agents.append(self.predictive_agent.name)
            except Exception as e:
                failed_agents.append({"agent": self.predictive_agent.name, "error": str(e)})
                state.log_agent_action(self.predictive_agent.name, "FAILED", f"Predictive modeling failed: {str(e)}")

            # Auxiliary clustering for customer/sales segmentation insight if tabular
            try:
                self.clustering_agent.run(state)
                successful_agents.append(f"{self.clustering_agent.name} (Auxiliary)")
            except Exception as e:
                warn_msg = f"Auxiliary clustering skipped/failed: {str(e)}"
                warnings.append(warn_msg)
                state.log_agent_action(self.clustering_agent.name, "WARNING", warn_msg, details={"warning": str(e)})

        # 6. Explainability Agent
        try:
            self.explanation_agent.run(state)
            successful_agents.append(self.explanation_agent.name)
        except Exception as e:
            failed_agents.append({"agent": self.explanation_agent.name, "error": str(e)})
            state.log_agent_action(self.explanation_agent.name, "FAILED", f"Explainability failed: {str(e)}")

        # 7. Prescriptive Agent
        try:
            self.prescriptive_agent.run(state)
            successful_agents.append(self.prescriptive_agent.name)
        except Exception as e:
            failed_agents.append({"agent": self.prescriptive_agent.name, "error": str(e)})
            state.log_agent_action(self.prescriptive_agent.name, "FAILED", f"Prescriptive strategy failed: {str(e)}")

        # Track execution summary in state
        if not failed_agents:
            pipeline_status = "SUCCESS"
        elif len(successful_agents) > 0:
            pipeline_status = "PARTIAL_SUCCESS"
        else:
            pipeline_status = "FAILED"

        state.execution_summary = {
            "status": pipeline_status,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "warnings": warnings
        }

        # 8. Executive Report Synthesis
        try:
            report = self.synthesize_report(state)
            state.final_report = report
            successful_agents.append("Executive Report Synthesizer")
        except Exception as e:
            failed_agents.append({"agent": "Executive Report Synthesizer", "error": str(e)})
            state.log_agent_action("Executive Report Synthesizer", "FAILED", f"Report synthesis failed: {str(e)}")
            report = {"markdown_text": f"# Analysis Incomplete\nError generating report: {str(e)}"}
            state.final_report = report

        summary = (
            f"Pipeline execution finished with status [{pipeline_status}]. "
            f"{len(successful_agents)} agent steps succeeded, {len(failed_agents)} failed."
        )
        return {
            "summary": summary,
            "pipeline_status": pipeline_status,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "final_report": report
        }

    def synthesize_report(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Synthesizes a publication-grade business intelligence report.
        Uses a rich deterministic executive narrative engine (guaranteed 100% offline).
        """
        # Collect context
        p = state.profile.get("summary", {})
        q = state.quality_report
        pd_info = state.problem_detection
        problem_type = state.selected_problem or pd_info.get("detected_problem", "Regression")
        target_col = state.selected_target or pd_info.get("target_column", "N/A")
        
        ml_data = state.ml_results
        champ_name = ml_data.get("champion_model_name", "N/A")
        test_metrics = ml_data.get("test_metrics", {})
        
        exp_data = state.explainability
        attribution_available = exp_data.get("available", True) and bool(exp_data.get("top_features"))
        
        recs = state.recommendations.get("action_items", [])
        caveats = state.recommendations.get("caveats", [])

        # Build Markdown Document
        metrics_bullet_points = (
            "\n".join([f"- **{k}**: `{v}`" for k, v in test_metrics.items()])
            if test_metrics else "- No test metrics available for current configuration."
        )
        recommendation_bullets = (
            "\n".join([
                f"### {r['id']}: {r['title']} ({r['quadrant']})\n"
                f"- **Strategic Lever**: {r['lever']} ({r['category']})\n"
                f"- **Priority & Effort**: Priority {r['priority_rank']} | Impact: **{r['impact']}** | Effort: **{r['effort']}**\n"
                f"- **Action Guidance**: {r['strategic_guidance']}\n"
                f"- **Evidence**: {r['evidence']}\n"
                for r in recs
            ])
            if recs else "- Prescriptive action recommendations could not be derived."
        )
        
        caveat_bullets = "\n".join([f"- {c}" for c in caveats]) if caveats else "- Standard statistical assumptions apply."

        # Explainability section
        if attribution_available and exp_data.get("importance_df") is not None and not exp_data.get("importance_df").empty:
            driver_lines = []
            for _, row in exp_data["importance_df"].head(6).iterrows():
                driver_lines.append(f"- **{row['Feature']}**: `{row.get('Importance_Pct', 0)}%` attribution — *{row.get('Direction', 'Neutral')}*")
            drivers_md = "\n".join(driver_lines)
            exp_section = f"""The system utilized `{exp_data.get('method', 'Feature Attribution')}` to isolate structural drivers behind model predictions.

### Primary Drivers Ranked by Relative Attribution
{drivers_md}"""
        else:
            reason = exp_data.get("reason", "Feature attribution is not available for this model type.")
            exp_section = f"Attribution Notice: {reason}"

        # Status note
        exec_summary = state.execution_summary
        pipeline_status = exec_summary.get("status", "SUCCESS")
        status_banner = ""
        if pipeline_status == "PARTIAL_SUCCESS":
            failed_str = ", ".join([f["agent"] for f in exec_summary.get("failed_agents", [])])
            status_banner = f"> [!WARNING]\n> **Partial Pipeline Execution**: Some downstream components experienced issues ({failed_str}). Available findings are reported below.\n\n"

        markdown_content = f"""# AGENTIC SCOUT EXECUTIVE REPORT
**Autonomous AI Framework for Business Data Analysis & Decision Support**

---

{status_banner}## 1. Executive Briefing
- **Dataset Evaluated**: `{state.dataset_name}`
- **Analyzed Records**: {p.get('rows', 0):,} rows across {p.get('columns', 0)} attributes
- **Data Health Score**: **{q.get('health_score', 0)}/100** (`{q.get('health_label', 'Unknown')}`)
- **Detected Problem Type**: **{problem_type}** (Confidence: {pd_info.get('confidence', 0)}%)
- **Target Variable**: `{target_col}`
- **Champion Model**: **{champ_name}**
- **Pipeline Execution**: `{pipeline_status}`


---

## 2. Data Health & Integrity Audit
- **Completeness**: {p.get('complete_rows_pct', 0)}% of rows are fully populated ({p.get('missing_rate_pct', 0)}% overall missingness).
- **Duplicates**: {p.get('duplicate_rows_count', 0)} duplicate records ({p.get('duplicate_rows_pct', 0)}%).
- **Outliers**: Identified {p.get('total_outliers', 0)} statistical outliers via IQR dispersion bounds.

### Priority Quality Audit Findings
"""
        issues = q.get("issues", [])
        if issues:
            for issue in issues[:4]:
                markdown_content += f"- **[{issue['severity']}] {issue['category']}**: {issue['message']} *Remediation: {issue['remediation']}*\n"
        else:
            markdown_content += "- No critical data quality anomalies detected.\n"

        markdown_content += f"""
---

## 3. Machine Learning Performance Benchmark
Tournament evaluation completed across candidate algorithms. Zero-leakage data partitioning applied.

**Selected Champion Architecture**: `{champ_name}`

### Test Partition Metrics
{metrics_bullet_points}

---

## 4. Explainability & Key Business Drivers
{exp_section}

---

## 5. Evidence-Based Prescriptive Strategy
Prioritized action plan derived from empirical leverage metrics and domain playbooks:

{recommendation_bullets}

---

## 6. Methodological Governance & Statistical Guardrails
{caveat_bullets}

---
*Report autonomously generated by AGENTIC SCOUT v1.0.0 on {time.strftime('%Y-%m-%d %H:%M:%S')}.*
"""

        return {
            "markdown_text": markdown_content.strip(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": state.dataset_name,
            "champion_model": champ_name,
            "health_score": q.get("health_score", 0),
            "problem_type": problem_type,
            "pipeline_status": pipeline_status
        }
