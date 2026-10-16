"""
Data Quality Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from core.state import AnalysisState
from ui.components import render_health_gauge


def render_data_quality_view(state: AnalysisState):
    """Renders the comprehensive data health, missingness, and outlier audit."""
    st.subheader("🛡️ Data Quality & Integrity Audit")

    if not state.quality_report:
        st.warning("Please execute the profiling pipeline to view the Data Quality audit.")
        return

    q = state.quality_report
    profile = state.profile
    summary = profile.get("summary", {})
    
    # Top Row: Health Score & Readiness
    c1, c2 = st.columns([1, 2])
    with c1:
        render_health_gauge(q.get("health_score", 0), q.get("health_label", "Good"))
        
    with c2:
        st.markdown("#### 📋 Analytical Readiness Checklist")
        readiness = q.get("readiness", {})
        rc1, rc2 = st.columns(2)
        with rc1:
            st.checkbox("Tabular Dimensions Valid", value=readiness.get("tabular_structure", False), disabled=True)
            st.checkbox("Missing Rate Acceptable (<50%)", value=readiness.get("missing_acceptable", False), disabled=True)
            st.checkbox("Numeric Features Present", value=readiness.get("has_numeric_features", False), disabled=True)
        with rc2:
            st.checkbox("Regression Ready", value=readiness.get("regression_ready", False), disabled=True)
            st.checkbox("Classification Ready", value=readiness.get("classification_ready", False), disabled=True)
            st.checkbox("Time Series Forecasting Ready", value=readiness.get("forecasting_ready", False), disabled=True)

    st.markdown("---")

    # Issues Breakdown
    st.markdown("### ⚠️ Quality Issues & Remediation Guidance")
    issues = q.get("issues", [])
    if not issues:
        st.success("No critical data quality issues identified. The dataset is in pristine condition!")
    else:
        for issue in issues:
            sev = issue["severity"]
            badge_color = "🔴" if sev == "CRITICAL" else ("🟡" if sev == "WARNING" else "🔵")
            with st.container():
                st.markdown(f"""
                **{badge_color} [{sev}] {issue['category']}**: {issue['message']}  
                *Recommended Remediation:* `{issue['remediation']}`
                """)

    st.markdown("---")

    # Missingness and Outliers Deep-Dive
    tab1, tab2, tab3 = st.tabs(["Missing Values Analysis", "Outliers Analysis", "Duplicates & Uniqueness"])
    
    with tab1:
        missing_dict = profile.get("missing_values", {}).get("columns_with_missing", {})
        if not missing_dict:
            st.success("Zero missing values across all columns! Row completeness is 100%.")
        else:
            miss_df = pd.DataFrame([
                {"Column": col, "Missing Count": data["count"], "Missing %": data["percentage"]}
                for col, data in missing_dict.items()
            ]).sort_values(by="Missing %", ascending=False)
            
            fig = px.bar(
                miss_df, x="Column", y="Missing %",
                title="Missing Rate by Column (%)",
                color="Missing %",
                color_continuous_scale="Reds",
                text_auto=".1f"
            )
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss_df, use_container_width=True)

    with tab2:
        outliers_dict = profile.get("outliers", {})
        if not outliers_dict:
            st.info("No numerical features to evaluate for outliers.")
        else:
            out_rows = [
                {
                    "Feature": col,
                    "Outlier Count": data["count"],
                    "Outlier %": data["percentage"],
                    "Lower Bound (IQR)": data["lower_bound"],
                    "Upper Bound (IQR)": data["upper_bound"]
                }
                for col, data in outliers_dict.items()
                if data["count"] > 0
            ]
            if not out_rows:
                st.success("No statistical outliers detected using 1.5 * IQR boundaries.")
            else:
                out_df = pd.DataFrame(out_rows).sort_values(by="Outlier Count", ascending=False)
                st.dataframe(out_df, use_container_width=True)

    with tab3:
        st.markdown(f"- **Duplicate Rows**: `{summary.get('duplicate_rows_count', 0)}` rows ({summary.get('duplicate_rows_pct', 0)}% of dataset)")
        high_cards = profile.get("high_cardinality_columns", [])
        if high_cards:
            st.markdown("#### High Cardinality Warnings")
            st.dataframe(pd.DataFrame(high_cards), use_container_width=True)
        else:
            st.info("No excessive high-cardinality categorical features identified.")
