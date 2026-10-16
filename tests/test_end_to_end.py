"""
End-to-end integration tests verifying the full 8-agent pipeline across all 3 synthetic datasets.
"""
import os
import pytest
from core.state import AnalysisState
from core.loader import DataLoader
from agents.orchestrator import OrchestratorAgent


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def orchestrator():
    return OrchestratorAgent()


def test_end_to_end_sample_sales(loader, orchestrator):
    path = "data/sample_sales.csv"
    assert os.path.exists(path), f"Sample file missing: {path}"
    
    df, meta = loader.load(path)
    state = AnalysisState(dataset_name="sample_sales.csv", raw_df=df, df=df)
    
    # Run complete autonomous workflow
    orchestrator.run(state)
    
    # Verifications
    assert state.profile is not None
    assert state.quality_report is not None
    assert state.selected_problem in ["Regression", "Forecasting"]
    assert state.ml_results.get("champion_model_name") is not None
    assert state.explainability.get("importance_df") is not None
    assert len(state.recommendations.get("action_items", [])) > 0
    assert state.final_report.get("markdown_text") is not None
    assert len(state.agent_logs) >= 7

    # Verify What-If Simulator
    sim = state.recommendations.get("simulator")
    if sim is not None:
        base_row = sim.get_baseline_row()
        num_cols = base_row.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            sim_res = sim.simulate({num_cols[0]: float(base_row[num_cols[0]].iloc[0]) * 1.1})
            assert "simulated_prediction" in sim_res


def test_end_to_end_sample_customer(loader, orchestrator):
    path = "data/sample_customer.csv"
    assert os.path.exists(path), f"Sample file missing: {path}"
    
    df, meta = loader.load(path)
    state = AnalysisState(dataset_name="sample_customer.csv", raw_df=df, df=df)
    
    orchestrator.run(state)
    
    assert state.profile is not None
    assert state.selected_problem == "Classification"
    assert state.ml_results.get("champion_model_name") is not None
    assert "Accuracy" in state.ml_results.get("test_metrics", {})
    assert state.explainability.get("importance_df") is not None
    assert len(state.recommendations.get("action_items", [])) > 0
    assert state.final_report.get("markdown_text") is not None


def test_end_to_end_sample_forecasting(loader, orchestrator):
    path = "data/sample_forecasting.csv"
    assert os.path.exists(path), f"Sample file missing: {path}"
    
    df, meta = loader.load(path)
    state = AnalysisState(dataset_name="sample_forecasting.csv", raw_df=df, df=df)
    
    orchestrator.run(state)
    
    assert state.profile is not None
    assert state.selected_problem in ["Forecasting", "Regression"]
    assert state.ml_results.get("champion_model_name") is not None
    assert state.final_report.get("markdown_text") is not None
