"""
Explainability and SHAP Attribution Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from core.state import AnalysisState


def render_explainability_view(state: AnalysisState):
    """Renders SHAP global feature attribution, local sample breakdowns, and directional impacts."""
    st.subheader("💡 Model Explainability & SHAP Drivers")

    if not state.explainability:
        st.info("Execute the pipeline to view SHAP values and model feature attributions.")
        return

    exp = state.explainability
    method = exp.get("method", "Feature Attribution")
    importance_df = exp.get("importance_df")
    local_exp = exp.get("local_explanation", [])
    base_val = exp.get("base_value", 0.0)

    # Explainer method banner
    st.markdown(f"""
    <div class="scout-card" style="border-left: 5px solid #7C3AED;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">
            Explainability Engine: <span style="color: #7C3AED;">{method}</span>
        </div>
        <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">
            Baseline Expected Value (μ): <b>{round(base_val, 3)}</b> | Decomposes marginal contribution of each business feature.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # If attribution is unavailable
    if not exp.get("available", True) or importance_df is None or (isinstance(importance_df, pd.DataFrame) and importance_df.empty):
        reason = exp.get("reason", "Feature attribution is not available for this model architecture.")
        st.info(f"ℹ️ **Attribution Notice:** {reason}")
        st.markdown("---")
        with st.expander("📚 Explainability Methodology Note", expanded=False):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** and Tree/Permutation Attributions require supervised feature matrices. 
            For unsupervised algorithms (such as standalone Clustering) or univariate statistical models (such as ARIMA), 
            feature attribution is not applicable and is therefore omitted to prevent misleading interpretations.
            """)
        return

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("#### 🌟 Global Feature Importance Attribution")
        if importance_df is not None and not importance_df.empty:
            top_df = importance_df.head(10).sort_values(by="Importance_Pct", ascending=True)
            fig = px.bar(
                top_df,
                x="Importance_Pct",
                y="Feature",
                orientation="h",
                color="Importance_Pct",
                color_continuous_scale="Viridis",
                title="Top Predictive Drivers (% Attribution)",
                text_auto=".1f"
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 🧭 Directional Business Impact")
        if importance_df is not None and not importance_df.empty:
            disp_df = importance_df[["Feature", "Importance_Pct", "Direction"]].head(8)
            disp_df.columns = ["Driver Feature", "Impact Share (%)", "Observed Direction"]
            st.dataframe(disp_df, use_container_width=True, hide_index=True)

        st.markdown("#### 🔬 Sample Case Local Attribution")
        if local_exp:
            st.caption("Marginal push (+ / -) on a representative observation:")
            for item in local_exp[:5]:
                st.markdown(f"- `{item}`")

    st.markdown("---")
    with st.expander("📚 Explainability Methodology Note", expanded=False):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** is grounded in cooperative game theory. 
        It allocates the optimal credit for a model prediction fairly among all contributing features. 
        A positive SHAP value indicates that a feature increases the model's prediction relative to the baseline expectation, 
        whereas a negative value pushes the prediction lower.
        """)
