"""Modular views for the 9 dashboard tabs."""
from ui.views.overview_view import render_overview_view
from ui.views.data_quality_view import render_data_quality_view
from ui.views.eda_view import render_eda_view
from ui.views.problem_detection_view import render_problem_detection_view
from ui.views.agents_view import render_agents_view
from ui.views.model_results_view import render_model_results_view
from ui.views.explainability_view import render_explainability_view
from ui.views.recommendations_view import render_recommendations_view
from ui.views.final_report_view import render_final_report_view

__all__ = [
    "render_overview_view",
    "render_data_quality_view",
    "render_eda_view",
    "render_problem_detection_view",
    "render_agents_view",
    "render_model_results_view",
    "render_explainability_view",
    "render_recommendations_view",
    "render_final_report_view",
]
