"""AGENTIC SCOUT - simple five-operation dashboard."""

import os
import streamlit as st
import pandas as pd

from config.settings import AppConfig
from core.state import AnalysisState
from core.loader import DataLoader
from core.profiler import DataProfiler
from core.quality import QualityAuditor
from core.problem_detector import ProblemDetector, ProblemType
from agents.orchestrator import OrchestratorAgent
from ui.styles import apply_custom_styles
from ui.components import render_header


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title=AppConfig.APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "state" not in st.session_state:
    st.session_state.state = AnalysisState()

if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = None

if "last_signature" not in st.session_state:
    st.session_state.last_signature = ""


state: AnalysisState = st.session_state.state


# ---------------------------------------------------------
# COMPONENTS
# ---------------------------------------------------------

loader = DataLoader()
profiler = DataProfiler()
auditor = QualityAuditor()
rule_detector = ProblemDetector()
orchestrator = OrchestratorAgent()

render_header()


# ---------------------------------------------------------
# AUTOMATIC PROBLEM DETECTION
# ---------------------------------------------------------

def detect_problem():
    """
    Automatically understand the current dataset and select
    the most suitable analytical problem using deterministic
    rule-based detection.
    """

    # Profile the CURRENT dataset
    profile = profiler.profile(state.df)

    state.profile = profile
    state.quality_report = auditor.audit(profile)

    # -----------------------------------------------------
    # IMPORTANT:
    # Never reuse a target from a previous dataset.
    # Only keep a target if it actually exists in the
    # CURRENT dataset.
    # -----------------------------------------------------

    valid_target = None

    if (
        state.selected_target is not None
        and state.selected_target in state.df.columns
    ):
        valid_target = state.selected_target

    # Run the rule-based detector
    result = rule_detector.detect(
        df=state.df,
        profile=profile,
        user_target=valid_target,
    )

    # Store the NEW decision
    state.problem_detection = result

    state.selected_problem = result.get(
        "detected_problem"
    )

    state.selected_target = result.get(
        "target_column"
    )

    state.selected_datetime_col = result.get(
        "datetime_column"
    )


# ---------------------------------------------------------
# RUN COMPLETE ANALYSIS
# ---------------------------------------------------------

def run_analysis():
    """Run the complete offline analysis pipeline."""

    with st.spinner("Analyzing the dataset..."):
        orchestrator.run(state)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## Get started")

    data_source = st.radio(
        "Data source",
        [
            "Sample dataset",
            "Upload CSV / Excel"
        ],
        label_visibility="collapsed",
    )

    df_loaded = None
    loaded_name = ""
    loaded_sig = ""

    # -----------------------------------------------------
    # SAMPLE DATASETS
    # -----------------------------------------------------

    if data_source == "Sample dataset":

        sample_choice = st.selectbox(
            "Choose a sample",
            [
                "Sales data",
                "Customer churn",
                "Time-series demand",
                "Customer segmentation",
                "Transaction anomalies",
            ],
        )

        file_map = {
            "Sales data":
                "data/sample_sales.csv",

            "Customer churn":
                "data/sample_customer.csv",

            "Time-series demand":
                "data/sample_forecasting.csv",

            "Customer segmentation":
                "data/customer_segmentation_1000.csv",

            "Transaction anomalies":
                "data/transaction_anomaly_1000.csv",
        }

        target_path = file_map[sample_choice]

        loaded_name = os.path.basename(target_path)

        if os.path.exists(target_path):

            df_loaded, meta = loader.load(target_path)

            loaded_sig = meta.get(
                "data_signature",
                ""
            )

        else:
            st.error(
                "Sample dataset not found."
            )

    # -----------------------------------------------------
    # USER UPLOAD
    # -----------------------------------------------------

    else:

        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=[
                "csv",
                "xlsx",
                "xls"
            ],
        )

        if uploaded_file is not None:

            loaded_name = uploaded_file.name

            df_loaded, meta = loader.load(
                uploaded_file,
                filename=loaded_name
            )

            loaded_sig = meta.get(
                "data_signature",
                ""
            )

    # -----------------------------------------------------
    # DETECT WHETHER THIS IS A NEW DATASET
    # -----------------------------------------------------

    dataset_is_new = (
        df_loaded is not None
        and (
            st.session_state.last_dataset != loaded_name
            or
            st.session_state.last_signature != loaded_sig
            or
            state.df is None
        )
    )

    # -----------------------------------------------------
    # NEW DATASET
    # -----------------------------------------------------

    if dataset_is_new:

        # Remember the new dataset
        st.session_state.last_dataset = loaded_name
        st.session_state.last_signature = loaded_sig

        # -------------------------------------------------
        # VERY IMPORTANT:
        # Clear ALL decisions from the previous dataset.
        # This prevents things like "Revenue" from being
        # reused when switching to Customer Churn.
        # -------------------------------------------------

        state.selected_problem = None
        state.selected_target = None
        state.selected_datetime_col = None

        # Clear old analysis results
        state.reset_downstream_artifacts(
            preserve_profile=False
        )

        # Store the new dataset
        state.dataset_name = loaded_name
        state.data_signature = loaded_sig
        state.raw_df = df_loaded
        state.df = df_loaded

        # Automatically understand the new dataset
        with st.spinner(
            "Understanding the dataset..."
        ):
            detect_problem()

    # =====================================================
    # AUTOMATIC DECISION DISPLAY
    # =====================================================

    if state.df is not None:

        st.markdown("---")

        st.markdown(
            "### Automatic decision"
        )

        st.caption(
            "The rule-based engine selects the most suitable "
            "task from the dataset structure."
        )

        st.write(
            f"**Task:** "
            f"{state.selected_problem or 'Not detected'}"
        )

        st.write(
            f"**Target:** "
            f"{state.selected_target or 'None'}"
        )

        # -------------------------------------------------
        # MANUAL OVERRIDE
        # -------------------------------------------------

        with st.expander(
            "Change manually",
            expanded=False
        ):

            task_options = [
                ProblemType.REGRESSION.value,
                ProblemType.CLASSIFICATION.value,
                ProblemType.FORECASTING.value,
                ProblemType.CLUSTERING.value,
                ProblemType.ANOMALY_DETECTION.value,
            ]

            current_task = (
                state.selected_problem
                if state.selected_problem in task_options
                else task_options[0]
            )

            chosen_task = st.selectbox(
                "Task",
                task_options,
                index=task_options.index(
                    current_task
                ),
            )

            target_options = [
                "None"
            ] + list(state.df.columns)

            current_target = (
                state.selected_target
                if state.selected_target
                in state.df.columns
                else "None"
            )

            chosen_target = st.selectbox(
                "Target",
                target_options,
                index=target_options.index(
                    current_target
                ),
            )

            # -------------------------------------------------
            # Detect manual changes
            # -------------------------------------------------

            task_changed = (
                chosen_task
                != state.selected_problem
            )

            target_changed = (
                (
                    chosen_target == "None"
                    and state.selected_target is not None
                )
                or
                (
                    chosen_target != "None"
                    and chosen_target
                    != state.selected_target
                )
            )

            if task_changed or target_changed:

                # Clear old model results
                state.reset_downstream_artifacts(
                    preserve_profile=True
                )

                state.selected_problem = (
                    chosen_task
                )

                state.selected_target = (
                    None
                    if chosen_target == "None"
                    else chosen_target
                )

            # -------------------------------------------------
            # Forecasting date column
            # -------------------------------------------------

            if "Forecasting" in chosen_task:

                date_options = list(
                    state.df.columns
                )

                current_date = (
                    state.selected_datetime_col
                    if state.selected_datetime_col
                    in date_options
                    else date_options[0]
                )

                state.selected_datetime_col = (
                    st.selectbox(
                        "Date column",
                        date_options,
                        index=date_options.index(
                            current_date
                        ),
                    )
                )

        # -------------------------------------------------
        # ANALYZE BUTTON
        # -------------------------------------------------

        if st.button(
            "Analyze data",
            type="primary",
            use_container_width=True,
        ):

            run_analysis()

            st.rerun()

    # =====================================================
    # RESET
    # =====================================================

    st.markdown("---")

    if st.button(
        "Reset",
        use_container_width=True
    ):

        st.session_state.state = (
            AnalysisState()
        )

        st.session_state.last_dataset = None
        st.session_state.last_signature = ""

        st.rerun()


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.markdown("# Agentic Scout")

st.caption(
    "A simple path from business data to useful decisions."
)


# ---------------------------------------------------------
# NO DATA
# ---------------------------------------------------------

if state.df is None:

    st.info(
        "Choose a sample dataset or upload a CSV / Excel "
        "file from the sidebar."
    )

    st.stop()


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

summary = state.profile.get(
    "summary",
    {}
)

quality = (
    state.quality_report
    or {}
)

problem = (
    state.selected_problem
    or "Not detected"
)


st.markdown(
    f"### {state.dataset_name}"
)

st.caption(
    f"Detected task: **{problem}**"
)


# =========================================================
# FIVE OPERATIONS
# =========================================================

t1, t2, t3, t4, t5 = st.tabs(
    [
        "1. Understand",
        "2. Identify",
        "3. Analyze",
        "4. Explain",
        "5. Recommend",
    ]
)


# =========================================================
# 1. UNDERSTAND
# =========================================================

with t1:

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        f"{summary.get('rows', len(state.df)):,}"
    )

    c2.metric(
        "Columns",
        f"{summary.get('columns', len(state.df.columns))}"
    )

    c3.metric(
        "Data quality",
        f"{quality.get('health_score', 0)}/100"
    )

    issues = quality.get(
        "issues",
        []
    )

    if issues:

        st.markdown(
            "#### Main data-quality issues"
        )

        for issue in issues[:3]:

            st.write(
                f"- {issue.get('message', 'Data quality issue found.')}"
            )

    else:

        st.success(
            "No major data-quality issues were detected."
        )

    st.markdown(
        "#### Data preview"
    )

    st.dataframe(
        state.df.head(8),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 2. IDENTIFY
# =========================================================

with t2:

    c1, c2 = st.columns(2)

    c1.metric(
        "Detected task",
        problem
    )

    c2.metric(
        "Target",
        state.selected_target or "None"
    )

    reasoning = state.problem_detection.get(
        "reasoning",
        []
    )

    st.markdown(
        "#### Why was this task selected?"
    )

    if reasoning:

        for reason in reasoning[:3]:

            st.write(
                f"- {reason}"
            )

    else:

        st.write(
            "The task was selected from the dataset "
            "structure and column characteristics."
        )

    st.caption(
        "Decision method: deterministic rule-based detection"
    )


# =========================================================
# 3. ANALYZE
# =========================================================

with t3:

    if not state.ml_results:

        st.info(
            "Click **Analyze data** in the sidebar "
            "to run the analysis."
        )

    else:

        ml = state.ml_results

        metrics = ml.get(
            "test_metrics",
            {}
        )

        # -------------------------------------------------
        # CLUSTERING
        # -------------------------------------------------

        if "Clustering" in problem:

            c1, c2 = st.columns(2)

            c1.metric(
                "Number of groups",
                ml.get(
                    "optimal_k",
                    "N/A"
                )
            )

            c2.metric(
                "Silhouette score",
                round(
                    float(
                        metrics.get(
                            "Silhouette",
                            0
                        )
                    ),
                    3
                )
            )

            st.write(
                "The system grouped similar records "
                "into data segments."
            )

        # -------------------------------------------------
        # ANOMALY DETECTION
        # -------------------------------------------------

        elif "Anomaly" in problem:

            c1, c2 = st.columns(2)

            c1.metric(
                "Anomalies found",
                ml.get(
                    "anomaly_count",
                    "N/A"
                )
            )

            c2.metric(
                "Detection",
                "Isolation Forest"
            )

            st.write(
                "The system identified observations "
                "that are unusual compared with the "
                "rest of the data."
            )

        # -------------------------------------------------
        # SUPERVISED / FORECASTING
        # -------------------------------------------------

        else:

            c1, c2 = st.columns(2)

            c1.metric(
                "Best model",
                ml.get(
                    "champion_model_name",
                    "N/A"
                )
            )

            if "Classification" in problem:

                value = metrics.get(
                    "F1",
                    0
                )

                label = "F1 score"

            elif "Forecasting" in problem:

                value = metrics.get(
                    "RMSE",
                    0
                )

                label = "RMSE"

            else:

                value = metrics.get(
                    "R2",
                    0
                )

                label = "R² score"

            c2.metric(
                label,
                round(
                    float(value),
                    3
                )
            )

            st.write(
                "Suitable models were compared and "
                "the best-performing model was selected."
            )


# =========================================================
# 4. EXPLAIN
# =========================================================

with t4:

    exp = (
        state.explainability
        or {}
    )

    importance_df = exp.get(
        "importance_df"
    )

    if (
        importance_df is not None
        and not importance_df.empty
    ):

        st.markdown(
            "#### Most important factors"
        )

        show_cols = [
            c
            for c in [
                "Feature",
                "Importance_Pct",
                "Direction"
            ]
            if c in importance_df.columns
        ]

        st.dataframe(
            importance_df[
                show_cols
            ].head(5),
            use_container_width=True,
            hide_index=True,
        )

    elif exp.get(
        "top_features"
    ):

        st.markdown(
            "#### Most important factors"
        )

        for feature in exp.get(
            "top_features",
            []
        )[:5]:

            st.write(
                f"- {feature}"
            )

    else:

        st.info(
            "Feature-level explanation is not "
            "available for this analysis type."
        )


# =========================================================
# 5. RECOMMEND
# =========================================================

with t5:

    recs = (
        state.recommendations.get(
            "action_items",
            []
        )
        if state.recommendations
        else []
    )

    if recs:

        st.markdown(
            "#### Suggested actions"
        )

        for i, rec in enumerate(
            recs[:3],
            start=1
        ):

            st.markdown(
                f"**{i}. "
                f"{rec.get('title', 'Recommendation')}**"
            )

            st.write(
                rec.get(
                    "strategic_guidance",
                    ""
                )
            )

    else:

        st.info(
            "Run the analysis to generate recommendations."
        )


# =========================================================
# DETAILED REPORT
# =========================================================

if state.final_report.get(
    "markdown_text"
):

    with st.expander(
        "View detailed report"
    ):

        st.markdown(
            state.final_report[
                "markdown_text"
            ]
        )