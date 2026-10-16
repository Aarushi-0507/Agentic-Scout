"""
Prescriptive Recommendations and Scenario Simulator Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from core.state import AnalysisState


def render_recommendations_view(state: AnalysisState):
    """Renders evidence-based business recommendations, Impact vs Effort matrix, and What-If simulator."""
    st.subheader("🚀 Prescriptive Business Strategy & What-If Simulator")

    if not state.recommendations:
        st.info("Execute the pipeline to view evidence-based prescriptive recommendations.")
        return

    rec_data = state.recommendations
    action_items = rec_data.get("action_items", [])
    caveats = rec_data.get("caveats", [])
    simulator = rec_data.get("simulator")

    tab_recs, tab_matrix, tab_sim = st.tabs(["Prioritized Initiatives", "Impact vs. Effort Matrix", "What-If Scenario Simulator"])

    # Tab 1: Action Items
    with tab_recs:
        st.markdown("### 📋 Formulated Strategic Initiatives")
        for rec in action_items:
            quadrant = rec["quadrant"]
            border_color = "#10B981" if quadrant == "Quick Win" else ("#2563EB" if quadrant == "Strategic Initiative" else "#F59E0B")
            
            st.markdown(f"""
            <div class="rec-card" style="border-left-color: {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 700; color: {border_color};">{rec['id']} • {quadrant.upper()}</span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-top: 2px;">{rec['title']}</div>
                    </div>
                    <div>
                        <span class="badge badge-primary">Impact: {rec['impact']}</span>
                        <span class="badge badge-neutral">Effort: {rec['effort']}</span>
                    </div>
                </div>
                <div style="font-size: 0.9rem; color: #334155; margin-top: 8px;">
                    <b>Strategic Action:</b> {rec['strategic_guidance']}
                </div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 6px; font-style: italic;">
                    <b>Evidence Grounding:</b> {rec['evidence']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tab 2: Impact vs Effort Matrix
    with tab_matrix:
        st.markdown("### 🎯 Impact vs. Effort Prioritization Grid")
        matrix_rows = []
        for r in action_items:
            # Map Impact and Effort to numerical coordinates
            imp_val = 3 if r["impact"] == "High" else (2 if r["impact"] == "Medium" else 1)
            eff_val = 1 if r["effort"] == "Low" else (2 if r["effort"] == "Medium" else 3)
            matrix_rows.append({
                "Initiative": r["title"],
                "ID": r["id"],
                "Effort (Lower is easier)": eff_val,
                "Projected Impact": imp_val,
                "Quadrant": r["quadrant"],
                "Category": r["category"]
            })
            
        if matrix_rows:
            mat_df = pd.DataFrame(matrix_rows)
            fig = px.scatter(
                mat_df, x="Effort (Lower is easier)", y="Projected Impact",
                text="ID", color="Quadrant",
                size=[25]*len(mat_df),
                title="Strategic Initiative Positioning",
                color_discrete_map={
                    "Quick Win": "#10B981",
                    "Strategic Initiative": "#2563EB",
                    "Operational Efficiency": "#F59E0B",
                    "Long-Term Re-evaluation": "#64748B"
                }
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis=dict(tickvals=[1, 2, 3], ticktext=["Low Effort", "Medium Effort", "High Effort"], range=[0.5, 3.5]),
                yaxis=dict(tickvals=[1, 2, 3], ticktext=["Low Impact", "Medium Impact", "High Impact"], range=[0.5, 3.5]),
                height=420, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: What-If Simulator
    with tab_sim:
        st.markdown("""
        <div class="scout-card" style="border-left: 5px solid #2563EB; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: #1E293B;">🎛️ Model-Based What-If Scenario Projection</div>
            <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">
                <b>Methodology Note:</b> Simulations evaluate model response under hypothetical parameter adjustments (ceteris paribus). 
                Outputs reflect statistical associations from observational data and do <b>not</b> constitute proven causal claims.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Adjust policy levers below to simulate projected model response in real time:")

        if simulator is None:
            st.info("Scenario simulation is active for supervised regression and classification models.")
        else:
            base_row = simulator.get_baseline_row()
            num_cols = base_row.select_dtypes(include=["number"]).columns.tolist()
            
            # Select up to 3 top numerical levers
            levers_to_adjust = num_cols[:3]
            adjustments = {}
            
            sim_cols = st.columns(len(levers_to_adjust))
            for i, col_name in enumerate(levers_to_adjust):
                base_val = float(base_row[col_name].iloc[0])
                with sim_cols[i]:
                    min_val = round(base_val * 0.5, 2)
                    max_val = round(base_val * 1.5, 2)
                    if min_val == max_val:
                        max_val = min_val + 10.0
                    step_val = max(0.1, round((max_val - min_val) / 20.0, 2))
                    
                    new_val = st.slider(
                        f"Adjust {col_name}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(base_val),
                        step=float(step_val),
                        key=f"sim_slider_{col_name}"
                    )
                    adjustments[col_name] = new_val

            # Run Simulation
            sim_result = simulator.simulate(adjustments)
            
            if "error" in sim_result:
                st.error(f"Simulation error: {sim_result['error']}")
            else:
                st.markdown("#### Estimated Model Response under Scenario")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric("Baseline Model Expectation", f"{sim_result['baseline_prediction']:,}")
                with r2:
                    st.metric("Simulated Scenario Output", f"{sim_result['simulated_prediction']:,}")
                with r3:
                    delta_prefix = "+" if sim_result['delta'] > 0 else ""
                    st.metric("Model Shift (Δ)", f"{delta_prefix}{sim_result['delta']:,}")
                with r4:
                    pct_prefix = "+" if sim_result['percentage_change'] > 0 else ""
                    st.metric("Estimated Shift %", f"{pct_prefix}{sim_result['percentage_change']}%")

                # Display explicit assumptions
                assumptions = sim_result.get("assumptions", [])
                if assumptions:
                    with st.expander("🔍 Scenario Assumptions & Boundary Limits", expanded=False):
                        for asm in assumptions:
                            st.markdown(f"- 📌 *{asm}*")

    st.markdown("---")

    # Methodological Caveats (Never claim correlation proves causation)
    st.markdown("### ⚠️ Methodological Guardrails & Statistical Caveats")
    for c in caveats:
        st.markdown(f"- ⚖️ *{c}*")
