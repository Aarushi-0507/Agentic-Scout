"""
Final Human-Readable Executive Report Tab View for AGENTIC SCOUT Streamlit application.
"""
import streamlit as st
from core.state import AnalysisState


def render_final_report_view(state: AnalysisState):
    """Renders the comprehensive executive report with download capabilities."""
    st.subheader("📄 Autonomous Executive Report")

    if not state.final_report:
        st.info("Execute the pipeline to generate the full human-readable executive report.")
        return

    report = state.final_report
    markdown_text = report.get("markdown_text", "")
    generated_at = report.get("generated_at", "")
    dataset_name = report.get("dataset_name", "dataset")

    # Download Bar
    d1, d2, d3 = st.columns([1, 1, 2])
    with d1:
        st.download_button(
            label="📥 Download Markdown (.md)",
            data=markdown_text,
            file_name=f"agentic_scout_report_{dataset_name}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with d2:
        # Generate clean HTML wrapper for export
        html_wrapper = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AGENTIC SCOUT Report - {dataset_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1E293B; }}
h1, h2, h3 {{ color: #0F172A; }}
code {{ background: #F1F5F9; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
hr {{ border: 0; border-top: 1px solid #E2E8F0; margin: 24px 0; }}
</style>
</head>
<body>
<pre style="white-space: pre-wrap; font-family: inherit;">
{markdown_text}
</pre>
</body>
</html>"""
        st.download_button(
            label="📥 Download HTML (.html)",
            data=html_wrapper,
            file_name=f"agentic_scout_report_{dataset_name}.html",
            mime="text/html",
            use_container_width=True
        )

    st.markdown("---")

    # Render Report Content
    st.markdown(markdown_text)
