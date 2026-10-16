"""
Overview Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
from core.state import AnalysisState
from ui.components import render_metric_card


def render_overview_view(state: AnalysisState):
    """Renders the executive dataset overview and status card."""
    st.subheader("📊 Executive Overview & Dataset Summary")

    if state.df is None or state.df.empty:
        st.info("👋 Welcome to AGENTIC SCOUT! Please select a sample dataset or upload your own CSV/Excel file from the sidebar to begin.")
        return

    profile_summary = state.profile.get("summary", {})
    quality = state.quality_report
    problem_type = state.selected_problem or state.problem_detection.get("detected_problem", "Pending Detection")
    target_col = state.selected_target or state.problem_detection.get("target_column", "Auto-Detecting")
    
    # KPI Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            label="Total Rows",
            value=f"{profile_summary.get('rows', len(state.df)):,}",
            subtitle="Analyzed observation count",
            badge_text="Dataset Size",
            badge_type="primary"
        )
    with c2:
        render_metric_card(
            label="Features / Columns",
            value=f"{profile_summary.get('columns', len(state.df.columns))}",
            subtitle=f"{profile_summary.get('numeric_columns_count', 0)} numeric, {profile_summary.get('categorical_columns_count', 0)} categorical",
            badge_text="Dimensions",
            badge_type="neutral"
        )
    with c3:
        health_score = quality.get("health_score", 0)
        badge_type = "success" if health_score >= 80 else ("warning" if health_score >= 60 else "danger")
        render_metric_card(
            label="Data Health Score",
            value=f"{health_score}/100",
            subtitle=quality.get("health_label", "Evaluated"),
            badge_text="Integrity",
            badge_type=badge_type
        )
    with c4:
        render_metric_card(
            label="Detected Problem",
            value=problem_type.split("/")[0].strip(),
            subtitle=f"Target: {target_col}",
            badge_text="Analytical Task",
            badge_type="primary"
        )

    st.markdown("---")

    # Interactive Data Explorer Preview
    st.markdown("### 🗃️ Raw Data Preview")
    st.dataframe(state.df.head(15), use_container_width=True)

    # Column Schema Summary
    with st.expander("🔍 View Column Metadata & Data Types", expanded=False):
        col_types = state.profile.get("column_types", {})
        schema_rows = []
        for col in state.df.columns:
            dtype_str = str(state.df[col].dtype)
            sem_type = "Unknown"
            for t_name, t_cols in col_types.items():
                if col in t_cols:
                    sem_type = t_name.capitalize()
                    break
            schema_rows.append({
                "Column": col,
                "Pandas Dtype": dtype_str,
                "Inferred Semantic Type": sem_type,
                "Non-Null Count": f"{state.df[col].count():,} / {len(state.df):,}",
                "Unique Values": state.df[col].nunique()
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)
