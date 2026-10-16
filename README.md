# AGENTIC SCOUT

Agentic Scout is an offline framework for automated business data analysis and decision support.

## Simple working

**Upload data -> Understand -> Identify -> Analyze -> Explain -> Recommend**

1. **Understand Data** - profiles the dataset and checks basic data quality.
2. **Identify Problem** - the rule-based `ProblemDetector` selects Regression, Classification, Forecasting, Clustering / Segmentation, or Anomaly Detection.
3. **Analyze** - runs the suitable machine-learning workflow and compares appropriate models where applicable.
4. **Explain Results** - shows the most important factors behind the result when available.
5. **Recommend Actions** - converts the analysis into a few business-oriented recommendations.

## Task detection

Task selection is deterministic and uses dataset structure, data types, unique values, column names, date/time information, and detected outliers. No LLM or external API is required.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application includes sample sales, customer, and forecasting datasets and also accepts CSV/Excel uploads.
