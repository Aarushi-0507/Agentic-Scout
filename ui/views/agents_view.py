"""
Specialized Autonomous Agents Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
from core.state import AnalysisState
from ui.components import render_agent_status_pill


def render_agents_view(state: AnalysisState):
    """Renders agent workflow architecture, live agent statuses, and audit execution traces."""
    st.subheader("🤖 Autonomous Agent Orchestration Hub")

    st.markdown("""
    AGENTIC SCOUT coordinates 8 specialized autonomous agents using a structured reactive blackboard architecture.
    Each agent possesses specialized domain intelligence and operates with strict methodological guardrails.
    """)

    # Architecture Overview Graph
    with st.expander("🗺️ View Multi-Agent Architecture Workflow", expanded=True):
        st.markdown("""
        ```mermaid
        graph TD
            Data[Dataset Upload/CSV/Excel] --> Prof[1. Data Profiling Agent]
            Prof --> EDA[2. EDA Agent]
            Prof --> Det[3. Problem Detection Engine]
            EDA --> Diag[4. Diagnostic Agent]
            Det --> Orch[5. Orchestrator Agent]
            Diag --> Orch
            Orch --> Pred[6. Predictive Agent]
            Orch --> Clust[7. Clustering Agent]
            Pred --> Expl[8. Explanation Agent (SHAP)]
            Clust --> Expl
            Expl --> Presc[9. Prescriptive Agent]
            Presc --> Rep[10. Executive Report Synthesizer]
        ```
        """)

    st.markdown("### ⚡ Live Specialized Agent Status")
    c1, c2 = st.columns(2)
    
    # Render status pills for agents
    agent_list = [
        ("Data Profiling Agent", "Structural & statistical profiling"),
        ("EDA Agent", "Descriptive & interaction analytics"),
        ("Diagnostic Agent", "Root cause & variance driver analysis"),
        ("Predictive Agent", "Competitive ML tournament & zero-leakage validation"),
        ("Clustering Agent", "Behavioral segmentation & anomaly detection"),
        ("Explanation Agent", "SHAP attribution & local explanations"),
        ("Prescriptive Agent", "What-if scenarios & prioritized business initiatives"),
        ("Orchestrator Agent", "Pipeline coordination & executive report synthesis")
    ]
    
    # Calculate durations from logs
    durations = {}
    last_msgs = {}
    for log in state.agent_logs:
        durations[log["agent"]] = log.get("duration_ms", 0.0)
        last_msgs[log["agent"]] = log.get("message", "")

    for i, (name, role) in enumerate(agent_list):
        col = c1 if i % 2 == 0 else c2
        status = state.agent_statuses.get(name, "IDLE")
        dur = durations.get(name, 0.0)
        msg = last_msgs.get(name, role)
        with col:
            render_agent_status_pill(name, status, dur, msg)

    st.markdown("---")

    # Execution Trace Table
    st.markdown("### 📜 Real-Time Agent Execution Audit Trace")
    if not state.agent_logs:
        st.info("No execution trace logs recorded yet. Run the pipeline to view real-time agent messages.")
    else:
        log_df = pd.DataFrame(state.agent_logs)
        # Select clean columns
        cols = ["timestamp", "agent", "status", "duration_ms", "message"]
        display_df = log_df[[c for c in cols if c in log_df.columns]]
        display_df.columns = [c.capitalize() for c in display_df.columns]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
