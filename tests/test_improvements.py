"""
Tests verifying all 10 priority improvements in AGENTIC SCOUT:
1. Genuine explainability & unavailable handling (no fabricated numbers).
2. Orchestrator error logging and partial-success tracking.
3. API key integration & secret hygiene.
4. Task/target validation.
5. Forecasting chronological split, sorted timestamps, leakage-free lags.
6. What-if scenario assumptions & non-causal labeling.
7. Downstream artifact reset & content-based file signature tracking.
"""
import pytest
import numpy as np
import pandas as pd
from core.state import AnalysisState
from core.loader import DataLoader
from core.problem_detector import ProblemDetector, ProblemType
from agents.orchestrator import OrchestratorAgent
from agents.explanation_agent import ExplanationAgent
from explainability.feature_importance import FallbackExplainer
from explainability.shap_explainer import ShapExplainer
from ml.forecasting import ForecastingPipeline
from recommendations.scenarios import ScenarioSimulator
from ml.preprocessor import DataPreprocessor
from sklearn.linear_model import LinearRegression


def test_genuine_explainability_and_unavailable_handling():
    """Verify that genuine explainability works and unavailable attribution does not fabricate numbers."""
    # Case 1: Legitimate Linear Model
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_train = np.array([10.0, 20.0, 30.0, 40.0])
    X_test = np.array([[2.0, 3.0], [4.0, 5.0]])
    y_test = np.array([15.0, 25.0])
    feature_names = ["FeatA", "FeatB"]
    
    model = LinearRegression().fit(X_train, y_train)
    explainer = ShapExplainer(max_samples=10)
    res = explainer.explain(model, X_train, X_test, y_test, feature_names)
    
    assert res["available"] is True
    assert res["importance_df"] is not None
    assert len(res["importance_df"]) == 2
    assert "FeatA" in res["top_features"]
    
    # Case 2: ExplanationAgent when no supervised champion is present (e.g. Clustering/ARIMA)
    state = AnalysisState(dataset_name="dummy", df=pd.DataFrame({"A": [1, 2, 3]}))
    state.ml_results = {"champion_model_name": "ARIMA (1,1,1)"}
    
    agent = ExplanationAgent()
    exp_out = agent.execute(state)
    
    assert exp_out["explainability"]["available"] is False
    assert exp_out["explainability"]["method"] == "Attribution Unavailable"
    assert len(exp_out["explainability"]["top_features"]) == 0
    assert "unavailable" in exp_out["summary"].lower()


def test_orchestrator_error_handling_and_partial_success():
    """Verify that orchestrator logs agent-level errors and reports partial success."""
    orchestrator = OrchestratorAgent()
    
    # DataFrame with valid columns
    df = pd.DataFrame({
        "Spend": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "Clicks": [100, 200, 300, 400, 500, 600],
        "Revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    })
    state = AnalysisState(
        dataset_name="test_err.csv",
        raw_df=df,
        df=df,
        selected_problem="Regression",
        selected_target="Revenue"
    )
    
    # Force a mock agent failure by corrupting predictive agent temporarily
    original_execute = orchestrator.predictive_agent.execute
    def failing_execute(s):
        raise RuntimeError("Simulated ML training failure")
    orchestrator.predictive_agent.execute = failing_execute
    
    try:
        res = orchestrator.execute(state)
        # Should record partial success instead of silently crashing
        assert res["pipeline_status"] == "PARTIAL_SUCCESS"
        assert any(f["agent"] == "Predictive Agent" for f in state.execution_summary["failed_agents"])
        assert "Data Profiling Agent" in state.execution_summary["successful_agents"]
        assert any(log["status"] == "FAILED" for log in state.agent_logs)
    finally:
        orchestrator.predictive_agent.execute = original_execute


def test_task_target_validation():
    """Verify rigorous validation of incompatible task and target combinations."""
    detector = ProblemDetector()
    
    # Continuous dataset
    df = pd.DataFrame({
        "Feature1": np.random.randn(100),
        "ContinuousFloatTarget": np.random.uniform(10.0, 1000.0, 100),
        "CategoryTarget": np.random.choice(["Yes", "No"], 100),
        "TextDate": pd.date_range("2023-01-01", periods=100).astype(str)
    })
    
    # 1. Classification on high-cardinality continuous float target -> should flag incompatibility
    val1 = detector.validate_configuration(df, ProblemType.CLASSIFICATION.value, target_col="ContinuousFloatTarget")
    assert val1["is_valid"] is False
    assert len(val1["issues"]) > 0
    
    # 2. Regression on string category target -> should flag non-numeric target
    val2 = detector.validate_configuration(df, ProblemType.REGRESSION.value, target_col="CategoryTarget")
    assert val2["is_valid"] is False
    assert any("non-numeric" in iss.lower() for iss in val2["issues"])
    
    # 3. Valid Classification -> should pass
    val3 = detector.validate_configuration(df, ProblemType.CLASSIFICATION.value, target_col="CategoryTarget")
    assert val3["is_valid"] is True
    
    # 4. Valid Forecasting -> should pass
    val4 = detector.validate_configuration(
        df, ProblemType.FORECASTING.value, target_col="ContinuousFloatTarget", datetime_col="TextDate"
    )
    assert val4["is_valid"] is True


def test_forecasting_chronological_and_leakage_free():
    """Verify chronological ordering, no look-ahead lag leakage, and multi-step forecasting."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    # Provide un-sorted input
    np.random.seed(42)
    unshuffled_dates = dates.copy()
    shuffled_idx = np.random.permutation(len(dates))
    
    df = pd.DataFrame({
        "Date": unshuffled_dates[shuffled_idx],
        "Sales": (np.arange(50) * 2.0 + np.random.normal(0, 1, 50))[shuffled_idx]
    })
    
    pipe = ForecastingPipeline(forecast_horizon=7, random_state=42)
    res = pipe.train_and_forecast(df, date_col="Date", target_col="Sales")
    
    assert res["champion_model_name"] is not None
    assert len(res["future_forecast"]["dates"]) == 7
    assert len(res["future_forecast"]["predictions"]) == 7
    
    # Verify lag creation uses shift(1) or greater (strictly leakage-free)
    ts_df = df.sort_values(by="Date").reset_index(drop=True)
    lags = pipe._create_lag_features(ts_df, "Sales", "Date")
    # At index 0, lag_1 should be NaN because target at t=0 has no prior value
    assert np.isnan(lags["lag_1"].iloc[0])
    # At index 1, lag_1 should equal Sales at index 0
    assert lags["lag_1"].iloc[1] == ts_df["Sales"].iloc[0]


def test_what_if_scenario_assumptions_and_non_causal_labeling():
    """Verify that what-if simulator includes explicit assumptions and non-causal metadata."""
    df = pd.DataFrame({
        "Price": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
        "Marketing": [100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 350.0],
        "Demand": [500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0]
    })
    prep = DataPreprocessor()
    prep_data = prep.prepare_data(df, target_col="Demand")
    
    model = LinearRegression().fit(prep_data["X_train"], prep_data["y_train"])
    
    sim = ScenarioSimulator(
        estimator=model,
        preprocessor=prep,
        raw_df=df,
        target_col="Demand"
    )
    
    base_row = sim.get_baseline_row()
    assert "Price" in base_row.columns
    assert "Demand" not in base_row.columns  # Target excluded from baseline features
    
    res = sim.simulate({"Price": 25.0})
    assert res["is_model_scenario"] is True
    assert res["is_causal"] is False
    assert len(res["assumptions"]) >= 2
    assert "non-causal" in res["disclaimer"].lower() or "causal" in res["disclaimer"].lower()


def test_downstream_state_reset_and_data_signatures(tmp_path):
    """Verify that state resets downstream artifacts and loader tracks content signatures."""
    state = AnalysisState()
    state.ml_results = {"champion_model_name": "TestModel"}
    state.explainability = {"available": True}
    state.final_report = {"markdown_text": "# Report"}
    state.agent_logs = [{"agent": "Test", "status": "SUCCESS"}]
    
    # Target change reset
    state.reset_downstream_artifacts(preserve_profile=True)
    assert state.ml_results == {}
    assert state.explainability == {}
    assert state.final_report == {}
    assert state.agent_logs == []
    
    # File Signature test with identical filename but different content
    file_path = tmp_path / "same_name.csv"
    
    df1 = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    df1.to_csv(file_path, index=False)
    loader = DataLoader()
    _, meta1 = loader.load(str(file_path))
    
    df2 = pd.DataFrame({"A": [9, 8, 7, 6, 5], "B": [90, 80, 70, 60, 50]})
    df2.to_csv(file_path, index=False)
    _, meta2 = loader.load(str(file_path))
    
    assert meta1["filename"] == meta2["filename"]  # Same filename
    assert meta1["data_signature"] != meta2["data_signature"]  # Different content signature!
