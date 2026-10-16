"""
Reusable UI Components for AGENTIC SCOUT Streamlit application.
"""
from typing import Optional, List, Dict, Any
import streamlit as st
import plotly.graph_objects as go


def render_header():
    """Renders the top title and subtitle banner."""
    st.markdown("""
        <div style="margin-bottom: 1.2rem;">
            <div class="brand-title">AGENTIC SCOUT</div>
            <div class="brand-subtitle">An Autonomous AI Framework for Automated Business Data Analysis & Decision Support</div>
        </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: Any, subtitle: str = "", badge_text: str = "", badge_type: str = "primary"):
    """Renders an executive KPI metric box."""
    badge_html = f'<span class="badge badge-{badge_type}">{badge_text}</span>' if badge_text else ""
    st.markdown(f"""
        <div class="scout-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="metric-label">{label}</div>
                {badge_html}
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


def render_health_gauge(score: float, label: str):
    """Renders an interactive radial gauge for Data Health."""
    color = "#10B981" if score >= 80 else ("#F59E0B" if score >= 60 else "#EF4444")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Data Health Score: <b>{label}</b>", 'font': {'size': 16, 'color': '#0F172A'}},
        number={'suffix': "/100", 'font': {'size': 32, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "#F1F5F9",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 50], 'color': '#FEE2E2'},
                {'range': [50, 75], 'color': '#FEF3C7'},
                {'range': [75, 100], 'color': '#D1FAE5'}
            ],
            'threshold': {
                'line': {'color': "#0F172A", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)


def render_agent_status_pill(agent_name: str, status: str, duration_ms: float = 0.0, msg: str = ""):
    """Renders an individual agent execution pill."""
    badge_map = {
        "SUCCESS": ("badge-success", "✓ Done"),
        "RUNNING": ("badge-primary", "⏳ Running"),
        "FAILED": ("badge-danger", "✕ Failed"),
        "IDLE": ("badge-neutral", "○ Idle")
    }
    b_class, b_label = badge_map.get(status, ("badge-neutral", status))
    
    st.markdown(f"""
        <div class="agent-pill">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #1E293B;">{agent_name}</span>
                <div>
                    <span class="badge {b_class}">{b_label}</span>
                    <span style="font-size: 0.75rem; color: #64748B; margin-left: 6px;">{round(duration_ms)} ms</span>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #475569; margin-top: 4px;">{msg}</div>
        </div>
    """, unsafe_allow_html=True)
