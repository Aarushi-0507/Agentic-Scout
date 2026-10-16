"""
Model Results and Performance Benchmark Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.state import AnalysisState


def render_model_results_view(state: AnalysisState):
    """Renders the model leaderboard, champion architecture details, and diagnostic visual plots."""
    st.subheader("🏆 Model Tournament & Performance Results")

    if not state.ml_results:
        st.info("Execute the pipeline to view Machine Learning tournament benchmarks.")
        return

    ml_res = state.ml_results
    champ = ml_res.get("champion_model_name", "N/A")
    test_metrics = ml_res.get("test_metrics", {})
    problem_type = state.selected_problem or state.problem_detection.get("detected_problem", "Regression")
    leaderboard = ml_res.get("leaderboard")

    # Champion Callout Banner
    st.markdown(f"""
    <div class="scout-card" style="border-left: 5px solid #10B981;">
        <div style="font-size: 1.2rem; font-weight: 700; color: #1E293B;">
            ★ Selected Champion: <span style="color: #10B981;">{champ}</span>
        </div>
        <div style="font-size: 0.9rem; color: #64748B; margin-top: 4px;">
            Optimized for {problem_type} with zero data leakage validation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Leaderboard Table
    st.markdown("### 📊 Competitive Algorithm Leaderboard")
    if leaderboard is not None and not leaderboard.empty:
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    else:
        st.caption("Leaderboard data currently processing...")

    st.markdown("---")

    # Task-Specific Diagnostic Visualizations
    st.markdown("### 📈 Model Diagnostic Visualizations")
    
    # 1. Forecasting Visuals
    if "Forecasting" in problem_type and "future_forecast" in ml_res:
        _render_forecasting_plots(ml_res, state)
        
    # 2. Classification Visuals
    elif "Classification" in problem_type:
        _render_classification_plots(ml_res)
        
    # 3. Clustering Visuals
    elif "Clustering" in problem_type:
        _render_clustering_plots(ml_res)
        
    # 4. Regression Visuals (Default)
    else:
        _render_regression_plots(ml_res)

    # Check if auxiliary clustering exists to display segment insights as well
    if "clustering_auxiliary" in ml_res:
        with st.expander("👥 Auxiliary Customer / Behavioral Clustering & Segments", expanded=False):
            _render_clustering_plots(ml_res["clustering_auxiliary"])


def _render_regression_plots(ml_res: dict):
    """Actual vs Predicted and Residuals plot for Regression."""
    y_test = ml_res.get("y_test")
    y_pred = ml_res.get("y_pred")
    
    if y_test is not None and y_pred is not None:
        c1, c2 = st.columns(2)
        with c1:
            # Actual vs Predicted
            df_plot = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            min_val = min(df_plot["Actual"].min(), df_plot["Predicted"].min())
            max_val = max(df_plot["Actual"].max(), df_plot["Predicted"].max())
            
            fig = px.scatter(
                df_plot, x="Actual", y="Predicted",
                title="Actual vs. Predicted Target Values",
                opacity=0.65, color_discrete_sequence=["#2563EB"]
            )
            # Add ideal line
            fig.add_shape(
                type="line", line=dict(dash="dash", color="#EF4444", width=2),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Residuals Distribution
            residuals = y_test - y_pred
            res_df = pd.DataFrame({"Residuals": residuals})
            fig = px.histogram(
                res_df, x="Residuals", nbins=30,
                title="Residuals (Error) Distribution",
                color_discrete_sequence=["#7C3AED"]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#0F172A")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)


def _render_classification_plots(ml_res: dict):
    """Confusion Matrix and Probability Calibration."""
    cm = ml_res.get("confusion_matrix")
    classes = ml_res.get("classes", [0, 1])
    
    c1, c2 = st.columns(2)
    with c1:
        if cm is not None:
            cm_arr = np.array(cm)
            str_classes = [str(c) for c in classes]
            fig = px.imshow(
                cm_arr,
                x=str_classes,
                y=str_classes,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix (Test Fold)",
                labels=dict(x="Predicted Class", y="Actual True Class")
            )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        y_prob = ml_res.get("y_prob")
        if y_prob is not None:
            # Prediction confidence histogram
            prob_vals = y_prob[:, 1] if y_prob.ndim > 1 and y_prob.shape[1] > 1 else y_prob.flatten()
            prob_df = pd.DataFrame({"Confidence_Score": prob_vals})
            fig = px.histogram(
                prob_df, x="Confidence_Score", nbins=25,
                title="Prediction Probability / Confidence Distribution",
                color_discrete_sequence=["#10B981"]
            )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)


def _render_clustering_plots(clust_res: dict):
    """2D PCA Projection and Segment Personas."""
    pca_x = clust_res.get("pca_x", [])
    pca_y = clust_res.get("pca_y", [])
    clust_df = clust_res.get("clustered_df")
    
    if pca_x and pca_y and clust_df is not None:
        plot_df = pd.DataFrame({
            "PCA Component 1": pca_x,
            "PCA Component 2": pca_y,
            "Cluster": clust_df["Cluster"],
            "Is_Anomaly": ["Anomaly" if a == 1 else "Normal" for a in clust_df.get("Is_Anomaly", [0]*len(clust_df))]
        })
        fig = px.scatter(
            plot_df, x="PCA Component 1", y="PCA Component 2",
            color="Cluster", symbol="Is_Anomaly",
            title=f"2D PCA Projection: Identified {clust_res.get('optimal_k', 3)} Behavioral Cohorts",
            opacity=0.75
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    # Personas Cards
    personas = clust_res.get("personas", [])
    if personas:
        st.markdown("#### 👥 Discovered Segment Personas")
        cols = st.columns(min(3, len(personas)))
        for idx, persona in enumerate(personas):
            col = cols[idx % len(cols)]
            with col:
                st.markdown(f"""
                <div class="scout-card">
                    <div style="font-weight: 700; color: #2563EB;">{persona['name']}</div>
                    <div style="font-size: 0.85rem; color: #64748B;">Size: <b>{persona['size']}</b> records ({persona['percentage']}%)</div>
                    <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #E2E8F0;"/>
                    <div style="font-size: 0.82rem; color: #334155;">
                        {'<br/>• '.join([''] + persona['key_traits'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def _render_forecasting_plots(ml_res: dict, state: AnalysisState):
    """Historical test actual vs predicted + 14-day future forecast ribbon."""
    future = ml_res.get("future_forecast", {})
    test_dates = ml_res.get("test_dates", [])
    y_test = ml_res.get("y_test", [])
    y_pred = ml_res.get("y_pred", [])
    
    fig = go.Figure()
    
    # 1. Historical test actual
    if test_dates and y_test:
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_test, mode="lines+markers",
            name="Actual (Test Set)", line=dict(color="#0F172A", width=2)
        ))
        
    # 2. Historical test predicted
    if test_dates and y_pred:
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_pred, mode="lines",
            name="Model Prediction (Test)", line=dict(color="#2563EB", width=2, dash="dot")
        ))
        
    # 3. Future Forecast line
    fut_dates = future.get("dates", [])
    fut_preds = future.get("predictions", [])
    lower_b = future.get("lower_bounds", [])
    upper_b = future.get("upper_bounds", [])
    
    if fut_dates and fut_preds:
        # Confidence interval ribbon
        fig.add_trace(go.Scatter(
            x=fut_dates + fut_dates[::-1],
            y=upper_b + lower_b[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name="95% Forecast Confidence Band",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=fut_dates, y=fut_preds, mode="lines+markers",
            name="Future Forecast", line=dict(color="#10B981", width=3)
        ))
        
    fig.update_layout(
        title="Time Series Trajectory: Historic Actuals, Predictions & 14-Day Future Forecast",
        xaxis_title="Date", yaxis_title="Target Demand / Value",
        height=480, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
