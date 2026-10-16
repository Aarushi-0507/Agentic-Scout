"""
Synthetic dataset generator for AGENTIC SCOUT sample data.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_datasets(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. sample_sales.csv (Regression / Revenue / Profit Analysis)
    # -------------------------------------------------------------
    n_sales = 1000
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=int(i * 0.73)) for i in range(n_sales)]
    
    regions = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"], size=n_sales, p=[0.4, 0.3, 0.2, 0.1])
    categories = np.random.choice(["Electronics", "Home & Office", "Industrial Supplies", "Apparel"], size=n_sales, p=[0.35, 0.25, 0.25, 0.15])
    segments = np.random.choice(["Enterprise", "Mid-Market", "Small Business"], size=n_sales, p=[0.3, 0.45, 0.25])
    
    marketing_spend = np.random.gamma(shape=5.0, scale=300.0, size=n_sales).round(2)
    discount_pct = np.random.choice([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], size=n_sales, p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05])
    unit_price = np.random.choice([45.0, 95.0, 150.0, 299.0, 599.0], size=n_sales, p=[0.25, 0.3, 0.25, 0.15, 0.05])
    
    base_demand = {
        "Electronics": 18,
        "Home & Office": 25,
        "Industrial Supplies": 12,
        "Apparel": 35
    }
    
    units_sold = []
    for i in range(n_sales):
        cat = categories[i]
        mkt = marketing_spend[i]
        disc = discount_pct[i]
        seg = segments[i]
        seg_boost = 1.4 if seg == "Enterprise" else (1.1 if seg == "Mid-Market" else 0.8)
        
        expected = (base_demand[cat] * seg_boost) + (mkt / 120.0) + (disc * 0.8) + np.random.normal(0, 3)
        units = max(1, int(np.round(expected)))
        units_sold.append(units)
    
    units_sold = np.array(units_sold)
    gross_revenue = units_sold * unit_price
    revenue = np.round(gross_revenue * (1.0 - discount_pct / 100.0), 2)
    shipping_cost = np.round(15.0 + units_sold * 1.8 + np.random.normal(0, 2, size=n_sales), 2)
    shipping_cost = np.maximum(5.0, shipping_cost)
    
    cogs_ratio = np.random.uniform(0.40, 0.55, size=n_sales)
    cogs = gross_revenue * cogs_ratio
    profit = np.round(revenue - cogs - (marketing_spend * 0.25) - shipping_cost, 2)
    
    sales_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Region": regions,
        "Product_Category": categories,
        "Customer_Segment": segments,
        "Marketing_Spend": marketing_spend,
        "Discount_Pct": discount_pct,
        "Unit_Price": unit_price,
        "Units_Sold": units_sold,
        "Shipping_Cost": shipping_cost,
        "Revenue": revenue,
        "Profit": profit
    })
    
    # Introduce small realistic missingness (~1.5%) in Marketing_Spend and Shipping_Cost
    mask_mkt = np.random.rand(n_sales) < 0.015
    sales_df.loc[mask_mkt, "Marketing_Spend"] = np.nan
    mask_ship = np.random.rand(n_sales) < 0.012
    sales_df.loc[mask_ship, "Shipping_Cost"] = np.nan
    
    sales_path = os.path.join(output_dir, "sample_sales.csv")
    sales_df.to_csv(sales_path, index=False)
    print(f"Generated {sales_path} with shape {sales_df.shape}")

    # -------------------------------------------------------------
    # 2. sample_customer.csv (Classification / Churn & Clustering)
    # -------------------------------------------------------------
    n_cust = 1000
    cust_ids = [f"CUST-{10001 + i}" for i in range(n_cust)]
    ages = np.random.randint(19, 72, size=n_cust)
    
    # Cluster-like income & spending score
    cluster_pick = np.random.choice([0, 1, 2, 3], size=n_cust, p=[0.3, 0.3, 0.2, 0.2])
    income_means = [32000, 75000, 115000, 55000]
    spend_means = [80, 70, 30, 45]
    
    annual_income = [int(max(15000, np.random.normal(income_means[k], 8000))) for k in cluster_pick]
    spending_score = [int(np.clip(np.random.normal(spend_means[k], 10), 1, 100)) for k in cluster_pick]
    
    sub_tiers = np.random.choice(["Basic", "Professional", "Enterprise"], size=n_cust, p=[0.45, 0.35, 0.20])
    contract_months = np.random.choice([1, 12, 24], size=n_cust, p=[0.55, 0.30, 0.15])
    
    base_charges = {"Basic": 39.0, "Professional": 89.0, "Enterprise": 189.0}
    monthly_charges = [round(base_charges[t] + np.random.uniform(-5, 15), 2) for t in sub_tiers]
    tenure_months = np.random.randint(1, 48, size=n_cust)
    total_charges = [round(m * t + np.random.normal(0, 10), 2) for m, t in zip(monthly_charges, tenure_months)]
    
    support_tickets = np.random.poisson(lam=1.8, size=n_cust)
    usage_hours = np.round(np.random.gamma(shape=4.0, scale=12.0, size=n_cust), 1)
    nps_scores = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n_cust, p=[0.05, 0.05, 0.08, 0.08, 0.1, 0.12, 0.18, 0.16, 0.10, 0.08])
    
    # Churn probability logit
    churn_list = []
    for i in range(n_cust):
        logit = (
            -1.8
            + (1.5 if contract_months[i] == 1 else -0.8)
            + (support_tickets[i] * 0.45)
            - (nps_scores[i] * 0.25)
            + (monthly_charges[i] * 0.012)
            - (tenure_months[i] * 0.035)
            - (usage_hours[i] * 0.015)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        churn_list.append(1 if np.random.rand() < prob else 0)
        
    cust_df = pd.DataFrame({
        "Customer_ID": cust_ids,
        "Age": ages,
        "Annual_Income": annual_income,
        "Spending_Score": spending_score,
        "Subscription_Tier": sub_tiers,
        "Contract_Months": contract_months,
        "Tenure_Months": tenure_months,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Support_Tickets": support_tickets,
        "Usage_Hours": usage_hours,
        "NPS_Score": nps_scores,
        "Churn": churn_list
    })
    
    # Introduce small missing values (~1%) in Total_Charges and Usage_Hours
    mask_tot = np.random.rand(n_cust) < 0.01
    cust_df.loc[mask_tot, "Total_Charges"] = np.nan
    mask_use = np.random.rand(n_cust) < 0.01
    cust_df.loc[mask_use, "Usage_Hours"] = np.nan
    
    cust_path = os.path.join(output_dir, "sample_customer.csv")
    cust_df.to_csv(cust_path, index=False)
    print(f"Generated {cust_path} with shape {cust_df.shape}")

    # -------------------------------------------------------------
    # 3. sample_forecasting.csv (Time Series Forecasting)
    # -------------------------------------------------------------
    n_days = 730  # 2 years daily
    start_ts = datetime(2023, 1, 1)
    dates_ts = [start_ts + timedelta(days=i) for i in range(n_days)]
    
    t = np.arange(n_days)
    trend = 0.12 * t
    yearly_seasonality = 15.0 * np.sin(2 * np.pi * t / 365.25)
    weekly_seasonality = 12.0 * np.sin(2 * np.pi * t / 7.0)
    
    promotions = (np.random.rand(n_days) < 0.15).astype(int)
    holidays = (np.random.rand(n_days) < 0.04).astype(int)
    competitor_price = np.round(np.random.normal(55.0, 3.5, size=n_days), 2)
    temperature = np.round(18.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.25) + np.random.normal(0, 3, size=n_days), 1)
    
    foot_traffic = np.maximum(50, 350 + (weekly_seasonality * 8) + (promotions * 120) + (holidays * 80) + np.random.normal(0, 35, size=n_days)).astype(int)
    
    base_units = 180 + trend + yearly_seasonality + weekly_seasonality + (promotions * 45) + (holidays * 30) - (competitor_price - 55.0) * 2.5
    demand_units = np.maximum(20, base_units + np.random.normal(0, 15, size=n_days)).astype(int)
    
    unit_retail_price = 42.50
    sales_revenue = [
        round(float(u * unit_retail_price * (0.90 if p else 1.0)), 2)
        for p, u in zip(promotions, demand_units)
    ]
    
    forecast_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates_ts],
        "Store_ID": ["STORE-01"] * n_days,
        "Demand_Units": demand_units,
        "Foot_Traffic": foot_traffic,
        "Promotion_Active": promotions,
        "Competitor_Price": competitor_price,
        "Holiday": holidays,
        "Temperature": temperature,
        "Sales_Revenue": list(sales_revenue)
    })
    
    forecast_path = os.path.join(output_dir, "sample_forecasting.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Generated {forecast_path} with shape {forecast_df.shape}")

if __name__ == "__main__":
    generate_datasets("/Users/aarushibashisth/Downloads/dynamic-pricing-game-theory/game/agentic_scout/data")
