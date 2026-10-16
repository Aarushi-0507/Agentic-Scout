"""
Exploratory Data Analysis (EDA) Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from core.state import AnalysisState


def render_eda_view(state: AnalysisState):
    """Renders interactive descriptive statistics, distributions, and correlation matrices."""
    st.subheader("📈 Exploratory Data Analysis (EDA)")

    if state.df is None or state.df.empty:
        st.warning("Please load a dataset to view Exploratory Data Analysis.")
        return

    profile = state.profile
    df = state.df
    col_types = profile.get("column_types", {})
    numeric_cols = col_types.get("numeric", [])
    categorical_cols = col_types.get("categorical", [])
    
    eda_tabs = st.tabs(["Distributions & Summary", "Correlation Matrix", "Categorical Analysis", "Bivariate Relationships"])

    # 1. Distributions
    with eda_tabs[0]:
        st.markdown("#### Numerical Feature Distributions")
        if numeric_cols:
            selected_num = st.selectbox("Select Numerical Feature to Inspect", options=numeric_cols, key="eda_num_select")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.histogram(
                    df, x=selected_num, nbins=30, marginal="box",
                    title=f"Distribution & Boxplot of {selected_num}",
                    color_discrete_sequence=["#2563EB"]
                )
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                stats_dict = profile.get("numerical_statistics", {}).get(selected_num, {})
                if stats_dict:
                    st.markdown(f"**Statistical Profile: {selected_num}**")
                    stat_df = pd.DataFrame([
                        {"Metric": str(k).replace("_", " ").title(), "Value": str(v)}
                        for k, v in stats_dict.items()
                    ])
                    st.dataframe(stat_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numerical features available for distribution analysis.")

    # 2. Correlation Matrix
    with eda_tabs[1]:
        st.markdown("#### Feature Correlation Heatmap")
        corrs = profile.get("correlations", {})
        pearson = corrs.get("pearson", {})
        
        if pearson and len(numeric_cols) >= 2:
            corr_df = pd.DataFrame(pearson)
            fig = px.imshow(
                corr_df,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1.0, zmax=1.0,
                title="Pearson Correlation Heatmap"
            )
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top Positive Correlations**")
                pos_pairs = corrs.get("top_positive", [])
                if pos_pairs:
                    st.dataframe(pd.DataFrame(pos_pairs), use_container_width=True, hide_index=True)
                else:
                    st.caption("No strong positive correlations found.")
            with c2:
                st.markdown("**Top Negative Correlations**")
                neg_pairs = corrs.get("top_negative", [])
                if neg_pairs:
                    st.dataframe(pd.DataFrame(neg_pairs), use_container_width=True, hide_index=True)
                else:
                    st.caption("No strong negative correlations found.")
        else:
            st.info("At least two numerical columns are required to generate correlation heatmaps.")

    # 3. Categorical Analysis
    with eda_tabs[2]:
        st.markdown("#### Categorical Breakdown")
        if categorical_cols:
            selected_cat = st.selectbox("Select Categorical Feature", options=categorical_cols, key="eda_cat_select")
            vc = df[selected_cat].value_counts().reset_index()
            vc.columns = [selected_cat, "Count"]
            
            fig = px.bar(
                vc.head(10), x=selected_cat, y="Count",
                title=f"Frequency Breakdown of {selected_cat}",
                color="Count", color_continuous_scale="Blues",
                text_auto=True
            )
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical features found in the current dataset.")

    # 4. Bivariate Relationships
    with eda_tabs[3]:
        st.markdown("#### Bivariate Feature Interaction")
        if len(numeric_cols) >= 2:
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                x_axis = st.selectbox("X-Axis Feature", options=numeric_cols, index=0)
            with bc2:
                y_axis = st.selectbox("Y-Axis Feature", options=numeric_cols, index=min(1, len(numeric_cols) - 1))
            with bc3:
                color_opt = st.selectbox("Color By (Optional)", options=["None"] + categorical_cols)
                
            color_arg = None if color_opt == "None" else color_opt
            fig = px.scatter(
                df, x=x_axis, y=y_axis, color=color_arg,
                title=f"{y_axis} vs. {x_axis}",
                trendline="ols" if color_arg is None else None,
                opacity=0.7
            )
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numerical features are needed for bivariate scatter plots.")
