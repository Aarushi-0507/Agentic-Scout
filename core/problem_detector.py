"""
Transparent Rule-Based Problem Detection Engine for AGENTIC SCOUT.

Detects:
- Regression
- Classification
- Forecasting
- Clustering / Segmentation
- Anomaly Detection

The detector does NOT depend on dataset filenames.

It examines:
- data types
- number of unique values
- column names
- date/time information
- number of numeric columns
- customer/entity structure
- statistical outliers

The decision is deterministic and rule-based.
"""

from typing import Dict, Any, List, Optional
from enum import Enum

import pandas as pd
import numpy as np


class ProblemType(str, Enum):

    REGRESSION = "Regression"

    CLASSIFICATION = "Classification"

    FORECASTING = "Forecasting"

    CLUSTERING = "Clustering / Segmentation"

    ANOMALY_DETECTION = "Anomaly Detection"


class ProblemDetector:

    """
    Automatically identifies the most suitable analytical
    problem from the structure of a dataset.
    """

    # -----------------------------------------------------
    # KEYWORDS
    # -----------------------------------------------------

    CLASSIFICATION_KEYWORDS = [
        "churn",
        "target",
        "label",
        "class",
        "status",
        "converted",
        "buy",
        "fraud",
        "default",
        "success",
        "outcome",
        "passed",
        "approved",
        "response",
    ]

    REGRESSION_KEYWORDS = [
        "revenue",
        "profit",
        "sales",
        "price",
        "cost",
        "demand",
        "amount",
        "spend",
        "charges",
        "income",
        "value",
        "quantity",
        "units",
        "score",
        "salary",
        "margin",
    ]

    DATETIME_KEYWORDS = [
        "date",
        "time",
        "timestamp",
        "day",
        "month",
        "year",
        "period",
        "datetime",
    ]

    # -----------------------------------------------------
    # DETECT
    # -----------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        user_target: Optional[str] = None,
    ) -> Dict[str, Any]:

        """
        Analyze the CURRENT dataset and determine the
        most suitable analytical problem.
        """

        # -------------------------------------------------
        # BASIC PROFILE INFORMATION
        # -------------------------------------------------

        col_types = profile.get(
            "column_types",
            {}
        )

        numeric_cols = col_types.get(
            "numeric",
            []
        )

        categorical_cols = col_types.get(
            "categorical",
            []
        )

        datetime_cols = col_types.get(
            "datetime",
            []
        )

        boolean_cols = col_types.get(
            "boolean",
            []
        )

        id_cols = col_types.get(
            "id_like",
            []
        )

        n_rows = len(df)

        # -------------------------------------------------
        # IMPORTANT SAFETY CHECK
        # -------------------------------------------------
        #
        # If a target came from another dataset and does
        # not exist in this dataset, completely ignore it.
        #
        # Example:
        #
        # Previous dataset:
        # Revenue
        #
        # New dataset:
        # Churn
        #
        # Revenue must NOT be used.
        # -------------------------------------------------

        if (
            user_target is not None
            and user_target not in df.columns
        ):
            user_target = None

        # -------------------------------------------------
        # 1. FIND DATETIME COLUMN
        # -------------------------------------------------

        detected_dt_col = None

        if datetime_cols:

            detected_dt_col = datetime_cols[0]

        else:

            for col in df.columns:

                col_name = str(col).lower()

                if any(
                    keyword in col_name
                    for keyword in self.DATETIME_KEYWORDS
                ):

                    try:

                        parsed = pd.to_datetime(
                            df[col],
                            errors="coerce"
                        )

                        if (
                            parsed.notnull().mean()
                            > 0.7
                        ):

                            detected_dt_col = col
                            break

                    except Exception:
                        pass

        # -------------------------------------------------
        # 2. FIND POSSIBLE TARGET COLUMNS
        # -------------------------------------------------

        candidate_targets: List[
            Dict[str, Any]
        ] = []

        for col in df.columns:

            # Never use IDs or dates as targets
            if (
                col in id_cols
                or col == detected_dt_col
            ):
                continue

            n_unique = int(
                df[col].nunique(
                    dropna=True
                )
            )

            col_lower = str(
                col
            ).lower()

            is_num = (
                col in numeric_cols
            )

            is_bool = (
                col in boolean_cols
            )

            is_cat = (
                col in categorical_cols
            )

            # ---------------------------------------------
            # CLASSIFICATION CANDIDATE
            # ---------------------------------------------

            if (
                is_bool
                or (
                    is_cat
                    and 2 <= n_unique <= 10
                )
                or (
                    is_num
                    and 2 <= n_unique <= 5
                )
            ):

                score_clf = 35

                if any(
                    keyword in col_lower
                    for keyword in
                    self.CLASSIFICATION_KEYWORDS
                ):

                    score_clf += 55

                if n_unique == 2:

                    score_clf += 15

                candidate_targets.append(
                    {
                        "column": col,
                        "likely_type":
                            ProblemType.CLASSIFICATION.value,
                        "unique_values":
                            n_unique,
                        "score":
                            score_clf,
                    }
                )

            # ---------------------------------------------
            # REGRESSION CANDIDATE
            # ---------------------------------------------

            if (
                is_num
                and n_unique > 10
            ):

                score_reg = 30

                primary_reg_keywords = [
                    "revenue",
                    "profit",
                    "sales",
                    "target",
                    "margin",
                    "income",
                    "demand",
                    "price",
                ]

                secondary_reg_keywords = [
                    "spend",
                    "cost",
                    "charges",
                    "amount",
                    "quantity",
                    "units",
                    "score",
                    "salary",
                    "value",
                ]

                if any(
                    keyword in col_lower
                    for keyword in
                    primary_reg_keywords
                ):

                    score_reg += 55

                elif any(
                    keyword in col_lower
                    for keyword in
                    secondary_reg_keywords
                ):

                    score_reg += 35

                try:

                    if df[col].std() > 0:
                        score_reg += 10

                except Exception:
                    pass

                candidate_targets.append(
                    {
                        "column": col,
                        "likely_type":
                            ProblemType.REGRESSION.value,
                        "unique_values":
                            n_unique,
                        "score":
                            score_reg,
                    }
                )

        # -------------------------------------------------
        # SORT TARGET CANDIDATES
        # -------------------------------------------------

        candidate_targets.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        # -------------------------------------------------
        # DETERMINE PRIMARY TARGET
        # -------------------------------------------------

        primary_target = user_target

        if (
            not primary_target
            and candidate_targets
        ):

            primary_target = (
                candidate_targets[0][
                    "column"
                ]
            )

        # -------------------------------------------------
        # DECISION SCORES
        # -------------------------------------------------

        scores: Dict[str, float] = {

            ProblemType.REGRESSION.value:
                20.0,

            ProblemType.CLASSIFICATION.value:
                20.0,

            ProblemType.FORECASTING.value:
                20.0,

            ProblemType.CLUSTERING.value:
                30.0,

            ProblemType.ANOMALY_DETECTION.value:
                25.0,
        }

        reasoning: List[str] = []

        # =================================================
        # RULE A: FORECASTING
        # =================================================

        is_monotonic_time = False

        if detected_dt_col is not None:

            scores[
                ProblemType.FORECASTING.value
            ] += 40.0

            reasoning.append(
                f"Temporal column identified: "
                f"'{detected_dt_col}' indicates "
                f"time-series structure."
            )

            # ---------------------------------------------
            # Check whether time is ordered
            # ---------------------------------------------

            try:

                dt_series = pd.to_datetime(
                    df[detected_dt_col],
                    errors="coerce",
                ).dropna()

                if dt_series.is_monotonic_increasing:

                    is_monotonic_time = True

                    scores[
                        ProblemType.FORECASTING.value
                    ] += 35.0

                    reasoning.append(
                        f"'{detected_dt_col}' is "
                        f"monotonically increasing."
                    )

            except Exception:
                pass

            # ---------------------------------------------
            # Time-series related keywords
            # ---------------------------------------------

            all_column_names = str(
                df.columns
            ).lower()

            if any(
                keyword in all_column_names
                for keyword in [
                    "demand",
                    "traffic",
                    "daily",
                    "trend",
                    "revenue",
                    "sales",
                ]
            ):

                scores[
                    ProblemType.FORECASTING.value
                ] += 15.0

        # =================================================
        # RULE B: TARGET ANALYSIS
        # =================================================

        if primary_target:

            # ---------------------------------------------
            # SAFETY CHECK AGAIN
            # ---------------------------------------------

            if primary_target not in df.columns:

                primary_target = None

            else:

                target_series = (
                    df[primary_target]
                    .dropna()
                )

                n_target_unique = (
                    target_series.nunique()
                )

                target_is_num = (
                    pd.api.types
                    .is_numeric_dtype(
                        target_series
                    )
                )

                # -----------------------------------------
                # CLASSIFICATION
                # -----------------------------------------

                if (
                    n_target_unique <= 10
                    or
                    pd.api.types
                    .is_bool_dtype(
                        target_series
                    )
                ):

                    scores[
                        ProblemType.CLASSIFICATION.value
                    ] += 60.0

                    reasoning.append(
                        f"Target '{primary_target}' "
                        f"has {n_target_unique} "
                        f"discrete categories."
                    )

                    if any(
                        keyword in
                        str(primary_target).lower()
                        for keyword in
                        self.CLASSIFICATION_KEYWORDS
                    ):

                        scores[
                            ProblemType.CLASSIFICATION.value
                        ] += 15.0

                        reasoning.append(
                            f"Target name "
                            f"'{primary_target}' "
                            f"matches classification "
                            f"keywords."
                        )

                # -----------------------------------------
                # REGRESSION
                # -----------------------------------------

                elif (
                    target_is_num
                    and n_target_unique > 10
                ):

                    scores[
                        ProblemType.REGRESSION.value
                    ] += 60.0

                    reasoning.append(
                        f"Target '{primary_target}' "
                        f"is continuous numeric "
                        f"with {n_target_unique} "
                        f"unique values."
                    )

                    if any(
                        keyword in
                        str(primary_target).lower()
                        for keyword in
                        self.REGRESSION_KEYWORDS
                    ):

                        scores[
                            ProblemType.REGRESSION.value
                        ] += 15.0

                        reasoning.append(
                            f"Target name "
                            f"'{primary_target}' "
                            f"matches regression "
                            f"keywords."
                        )

        # =================================================
        # RULE C: CLUSTERING
        # =================================================

        if len(numeric_cols) >= 3:

            scores[
                ProblemType.CLUSTERING.value
            ] += 30.0

            reasoning.append(
                f"Multiple numerical attributes "
                f"({len(numeric_cols)} numeric columns) "
                f"are suitable for clustering."
            )

        all_column_names = str(
            df.columns
        ).lower()

        if (
            id_cols
            or any(
                keyword in all_column_names
                for keyword in [
                    "customer",
                    "client",
                    "user",
                    "segment",
                    "income",
                    "age",
                ]
            )
        ):

            scores[
                ProblemType.CLUSTERING.value
            ] += 20.0

            reasoning.append(
                "Customer/entity attributes "
                "are present, supporting "
                "segmentation."
            )

        # =================================================
        # RULE D: ANOMALY DETECTION
        # =================================================

        total_outliers = (
            profile
            .get("summary", {})
            .get("total_outliers", 0)
        )

        if total_outliers > 10:

            scores[
                ProblemType.ANOMALY_DETECTION.value
            ] += 30.0

            reasoning.append(
                f"{total_outliers} statistical "
                f"outliers were identified."
            )

            if (
                n_rows > 0
                and
                (
                    total_outliers
                    / n_rows
                ) >= 0.15
            ):

                scores[
                    ProblemType.ANOMALY_DETECTION.value
                ] += 20.0

                reasoning.append(
                    "A relatively high proportion "
                    "of observations are statistical "
                    "outliers."
                )

        # =================================================
        # NORMALIZE SCORES
        # =================================================

        for key in scores:

            scores[key] = min(
                100.0,
                max(
                    5.0,
                    round(
                        scores[key],
                        1
                    ),
                ),
            )

        # =================================================
        # FINAL PROBLEM SELECTION
        # =================================================

        if user_target:

            # ---------------------------------------------
            # User explicitly selected target
            # ---------------------------------------------

            target_series = (
                df[user_target]
                .dropna()
            )

            target_unique = (
                target_series.nunique()
            )

            if (
                target_unique <= 10
                or
                pd.api.types
                .is_bool_dtype(
                    target_series
                )
            ):

                detected_problem = (
                    ProblemType.CLASSIFICATION.value
                )

            elif (
                detected_dt_col is not None
                and
                "forecast"
                in str(
                    df.columns
                ).lower()
            ):

                detected_problem = (
                    ProblemType.FORECASTING.value
                )

            else:

                detected_problem = (
                    ProblemType.REGRESSION.value
                )

        else:

            # ---------------------------------------------
            # Automatic decision
            # ---------------------------------------------

            outlier_ratio = (
                total_outliers
                / max(
                    1,
                    n_rows
                )
            )

            strong_classification = any(
                item["likely_type"]
                == ProblemType.CLASSIFICATION.value
                and
                item["score"] >= 70
                for item in candidate_targets
            )

            all_names = str(
                df.columns
            ).lower()

            behavior_terms = [
                "spend",
                "purchase",
                "order",
                "visit",
                "discount",
                "frequency",
                "tenure",
                "usage",
                "score",
            ]

            behavior_count = sum(
                term in all_names
                for term in behavior_terms
            )

            customer_structure = (
                bool(id_cols)
                and
                len(numeric_cols) >= 4
                and
                (
                    "customer"
                    in all_names
                    or
                    "client"
                    in all_names
                    or
                    behavior_count >= 3
                )
            )

            anomaly_structure = (
                bool(id_cols)
                and
                outlier_ratio >= 0.10
            )

            # ---------------------------------------------
            # Priority 1: Forecasting
            # ---------------------------------------------

            if (
                is_monotonic_time
                and
                scores[
                    ProblemType.FORECASTING.value
                ] >= 85.0
            ):

                detected_problem = (
                    ProblemType.FORECASTING.value
                )

            # ---------------------------------------------
            # Priority 2: Classification
            # ---------------------------------------------

            elif strong_classification:

                detected_problem = (
                    ProblemType.CLASSIFICATION.value
                )

            # ---------------------------------------------
            # Priority 3: Anomaly detection
            # ---------------------------------------------

            elif anomaly_structure:

                detected_problem = (
                    ProblemType.ANOMALY_DETECTION.value
                )

            # ---------------------------------------------
            # Priority 4: Clustering
            # ---------------------------------------------

            elif customer_structure:

                detected_problem = (
                    ProblemType.CLUSTERING.value
                )

            # ---------------------------------------------
            # Otherwise choose highest score
            # ---------------------------------------------

            else:

                sorted_scores = sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                detected_problem = (
                    sorted_scores[0][0]
                )

        # =================================================
        # UNSUPERVISED TASKS HAVE NO TARGET
        # =================================================

        if detected_problem in {
            ProblemType.CLUSTERING.value,
            ProblemType.ANOMALY_DETECTION.value,
        }:

            primary_target = None

        # =================================================
        # CONFIDENCE
        # =================================================

        confidence = scores[
            detected_problem
        ]

        # =================================================
        # FINAL RESULT
        # =================================================

        return {

            "detected_problem":
                detected_problem,

            "confidence":
                confidence,

            "target_column":
                primary_target,

            "datetime_column":
                detected_dt_col,

            "scores":
                scores,

            "reasoning":
                reasoning,

            "candidate_targets":
                candidate_targets,
        }

    # =====================================================
    # CONFIGURATION VALIDATION
    # =====================================================

    def validate_configuration(
        self,
        df: pd.DataFrame,
        problem_type: str,
        target_col: Optional[str] = None,
        datetime_col: Optional[str] = None,
    ) -> Dict[str, Any]:

        """
        Validate a selected analytical problem,
        target column and datetime column.
        """

        if (
            df is None
            or df.empty
        ):

            return {
                "is_valid": False,
                "status": "INVALID",
                "message":
                    "Dataset is empty.",
                "issues": [
                    "No records loaded in DataFrame."
                ],
            }

        issues: List[str] = []

        suggestions: List[str] = []

        # =================================================
        # CLASSIFICATION / REGRESSION
        # =================================================

        if (
            "Classification"
            in problem_type
            or
            "Regression"
            in problem_type
        ):

            if not target_col:

                issues.append(
                    f"A valid target column is "
                    f"required for {problem_type}."
                )

            elif target_col not in df.columns:

                issues.append(
                    f"Target column "
                    f"'{target_col}' "
                    f"not found in dataset."
                )

            else:

                series = (
                    df[target_col]
                    .dropna()
                )

                n_unique = int(
                    series.nunique()
                )

                if n_unique < 2:

                    issues.append(
                        f"Target column "
                        f"'{target_col}' "
                        f"contains only "
                        f"{n_unique} unique value."
                    )

                # -----------------------------------------
                # CLASSIFICATION VALIDATION
                # -----------------------------------------

                if "Classification" in problem_type:

                    if (
                        pd.api.types
                        .is_float_dtype(
                            series
                        )
                        and
                        n_unique > 30
                        and
                        (
                            n_unique
                            / len(series)
                        ) > 0.3
                    ):

                        issues.append(
                            f"Target '{target_col}' "
                            f"has {n_unique} "
                            f"distinct continuous "
                            f"float values."
                        )

                        suggestions.append(
                            "Consider Regression "
                            "or binning the target."
                        )

                # -----------------------------------------
                # REGRESSION VALIDATION
                # -----------------------------------------

                if "Regression" in problem_type:

                    if not (
                        pd.api.types
                        .is_numeric_dtype(
                            series
                        )
                    ):

                        issues.append(
                            f"Target '{target_col}' "
                            f"is non-numeric."
                        )

                        suggestions.append(
                            "Consider Classification "
                            "or encode the column numerically."
                        )

                    elif n_unique < 3:

                        suggestions.append(
                            f"Target '{target_col}' "
                            f"has only {n_unique} "
                            f"unique numeric values. "
                            f"Classification may be "
                            f"more appropriate."
                        )

        # =================================================
        # FORECASTING
        # =================================================

        elif "Forecasting" in problem_type:

            if (
                not datetime_col
                or
                datetime_col
                not in df.columns
            ):

                issues.append(
                    "A valid date/time column "
                    "is required for forecasting."
                )

            else:

                try:

                    parsed_dt = pd.to_datetime(
                        df[datetime_col],
                        errors="coerce",
                    )

                    valid_pct = (
                        parsed_dt
                        .notnull()
                        .mean()
                    )

                    if valid_pct < 0.7:

                        issues.append(
                            f"Column "
                            f"'{datetime_col}' "
                            f"could not be parsed "
                            f"as valid dates."
                        )

                except Exception as e:

                    issues.append(
                        f"Error parsing temporal "
                        f"column: {str(e)}"
                    )

            if (
                not target_col
                or
                target_col
                not in df.columns
            ):

                issues.append(
                    "A numeric target metric "
                    "is required for forecasting."
                )

            elif not (
                pd.api.types
                .is_numeric_dtype(
                    df[target_col]
                )
            ):

                issues.append(
                    f"Target metric "
                    f"'{target_col}' "
                    f"must be numeric."
                )

        # =================================================
        # CLUSTERING / ANOMALY
        # =================================================

        elif (
            "Clustering"
            in problem_type
            or
            "Anomaly"
            in problem_type
        ):

            num_cols = (
                df.select_dtypes(
                    include=["number"]
                )
                .columns
                .tolist()
            )

            if len(num_cols) < 2:

                issues.append(
                    "At least two numerical "
                    "features are recommended "
                    "for this analysis."
                )

        # =================================================
        # RESULT
        # =================================================

        is_valid = (
            len(issues) == 0
        )

        status = (
            "VALID"
            if is_valid
            else
            "INVALID"
        )

        return {

            "is_valid":
                is_valid,

            "status":
                status,

            "message":
                (
                    "Configuration confirmed valid."
                    if is_valid
                    else
                    "Configuration contains incompatibilities."
                ),

            "issues":
                issues,

            "suggestions":
                suggestions,
        }