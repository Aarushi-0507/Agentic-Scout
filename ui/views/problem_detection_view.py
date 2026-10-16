"""
Problem Detection and Decision Matrix Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from core.state import AnalysisState


def render_problem_detection_view(state: AnalysisState):
    """Renders the transparent rule-based problem detector and decision matrix."""
    st.subheader("🎯 Problem Detection & Decision Matrix")

    if not state.problem_detection:
        st.info("Execute the pipeline to view transparent problem detection scoring.")
        return

    det = state.problem_detection
    detected_problem = det.get("detected_problem", "Unknown")
    confidence = det.get("confidence", 0.0)
    scores = det.get("scores", {})
    reasoning = det.get("reasoning", [])
    target = det.get("target_column")
    datetime_col = det.get("datetime_column")

    # Banner Callout
    st.markdown(f"""
    <div class="scout-card" style="border-left: 5px solid #2563EB;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">
            Primary Recommended Problem: <span style="color: #2563EB;">{detected_problem}</span>
        </div>
        <div style="font-size: 0.9rem; color: #64748B; margin-top: 4px;">
            Heuristic Rule Confidence: <b>{confidence}%</b> | Target: <b>{target or 'N/A'}</b> | Temporal Column: <b>{datetime_col or 'None'}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### ⚖️ Decision Matrix Score Comparison")
        score_df = pd.DataFrame([
            {"Problem Type": k, "Rule Match Score (%)": v}
            for k, v in scores.items()
        ]).sort_values(by="Rule Match Score (%)", ascending=False)
        
        fig = px.bar(
            score_df, x="Rule Match Score (%)", y="Problem Type",
            orientation="h",
            color="Rule Match Score (%)",
            color_continuous_scale="Blues",
            text_auto=".1f"
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🔍 Rule-Engine Scoring Explanations")
        st.caption("Deterministic structural and statistical rules are used for task selection:")
        for idx, rule in enumerate(reasoning):
            st.markdown(f"**{idx + 1}.** {rule}")

    st.markdown("---")

    # Candidate Targets
    candidates = det.get("candidate_targets", [])
    if candidates:
        st.markdown("#### 🎯 Identified Target Candidates")
        cand_df = pd.DataFrame(candidates)
        cand_df.columns = ["Column Name", "Recommended Task", "Unique Values", "Relevance Score"]
        st.dataframe(cand_df, use_container_width=True, hide_index=True)

    # Active User Configuration Confirmation & Validation
    st.markdown("---")
    st.markdown("### 🛠️ Active Task & Target Confirmation")
    active_prob = state.selected_problem or detected_problem
    active_target = state.selected_target or target
    active_dt = state.selected_datetime_col or datetime_col
    
    from core.problem_detector import ProblemDetector
    detector = ProblemDetector()
    val_res = detector.validate_configuration(
        df=state.df,
        problem_type=active_prob,
        target_col=active_target,
        datetime_col=active_dt
    )

    vc1, vc2 = st.columns([1, 1])
    with vc1:
        status_color = "#10B981" if val_res["is_valid"] else "#EF4444"
        status_badge = "✓ Validated & Ready" if val_res["is_valid"] else "⚠️ Needs Review"
        st.markdown(f"""
        <div class="scout-card" style="border-left: 5px solid {status_color};">
            <div style="font-weight: 700; color: #1E293B;">Active Analytical Configuration</div>
            <div style="font-size: 0.95rem; margin-top: 4px;">
                • Analytical Task: <b>{active_prob}</b><br/>
                • Selected Target: <b>{active_target or 'None (Unsupervised)'}</b><br/>
                • Temporal Column: <b>{active_dt or 'None'}</b>
            </div>
            <div style="margin-top: 8px;">
                <span class="badge {'badge-success' if val_res['is_valid'] else 'badge-danger'}">{status_badge}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with vc2:
        if not val_res["is_valid"]:
            st.warning("⚠️ **Configuration Compatibility Issues:**")
            for iss in val_res["issues"]:
                st.markdown(f"- {iss}")
            for sug in val_res["suggestions"]:
                st.info(f"💡 *Suggestion:* {sug}")
        else:
            st.success("✓ **Task and target variables are confirmed compatible.**")
            st.caption("You can adjust the analytical task or target column anytime from the sidebar controls.")
