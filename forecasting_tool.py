"""
E-Commerce Inventory Forecasting Tool
======================================
Author  : Felix Opitz
Method  : Linear Regression (scikit-learn)
Input   : inventory_data.csv
Output  : forecast_results.csv  +  console summary

Required packages:
    pip install pandas scikit-learn matplotlib openpyxl
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load inventory CSV and parse dates."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    print(f"✔  Loaded {len(df)} records | {df['sku'].nunique()} SKUs | "
          f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}")
    return df


# ─────────────────────────────────────────────
# 2. FORECAST DEMAND (Linear Regression)
# ─────────────────────────────────────────────

def forecast_demand(df: pd.DataFrame, periods_ahead: int = 3) -> pd.DataFrame:
    """
    For each SKU, fit a linear regression on historical units_sold
    and predict demand for the next N months.

    X = month index (0, 1, 2, ...)
    y = units_sold
    """
    results = []

    for sku, group in df.groupby("sku"):
        group = group.sort_values("date").reset_index(drop=True)

        # Encode time as integer index
        X = np.arange(len(group)).reshape(-1, 1)
        y = group["units_sold"].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Predict future months
        future_indices = np.arange(len(group), len(group) + periods_ahead).reshape(-1, 1)
        future_demand  = model.predict(future_indices)
        future_demand  = np.maximum(future_demand, 0)  # demand cannot be negative

        # Generate future dates
        last_date = group["date"].max()
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods_ahead)]

        # Current stock (latest record)
        current_stock = group["stock_level"].iloc[-1]
        unit_price    = group["unit_price"].iloc[-1]
        product_name  = group["product_name"].iloc[-1]
        category      = group["category"].iloc[-1]

        # Total forecasted demand over the forecast window
        total_forecast = future_demand.sum()

        # Overstock = stock that exceeds total forecasted demand
        overstock_units = max(0, current_stock - total_forecast)
        overstock_value = round(overstock_units * unit_price, 2)

        # Recommended reorder = max(0, total_forecast - current_stock)
        reorder_units = max(0, total_forecast - current_stock)

        # Days of supply = current stock / avg monthly demand
        avg_monthly_demand = y.mean()
        days_of_supply = round((current_stock / avg_monthly_demand) * 30) if avg_monthly_demand > 0 else 999

        # Risk classification
        if days_of_supply < 7:
            risk = "Critical"
        elif days_of_supply < 14:
            risk = "Warning"
        elif days_of_supply < 30:
            risk = "Watch"
        elif overstock_units > 0 and overstock_value > 3000:
            risk = "Overstock"
        elif overstock_units > 0:
            risk = "Moderate"
        else:
            risk = "OK"

        # R² score to show model fit quality
        r2 = round(model.score(X, y), 3)

        results.append({
            "sku":               sku,
            "product_name":      product_name,
            "category":          category,
            "current_stock":     int(current_stock),
            "avg_monthly_demand": round(avg_monthly_demand, 1),
            "forecast_3m_total": round(total_forecast, 0),
            "overstock_units":   round(overstock_units, 0),
            "overstock_value_eur": overstock_value,
            "reorder_units":     round(reorder_units, 0),
            "days_of_supply":    days_of_supply,
            "risk":              risk,
            "unit_price":        unit_price,
            "model_r2":          r2,
            "forecast_month_1":  round(future_demand[0], 0),
            "forecast_month_2":  round(future_demand[1], 0),
            "forecast_month_3":  round(future_demand[2], 0),
        })

    return pd.DataFrame(results).sort_values("overstock_value_eur", ascending=False)


# ─────────────────────────────────────────────
# 3. CALCULATE SAVINGS POTENTIAL
# ─────────────────────────────────────────────

def calculate_savings(forecast_df: pd.DataFrame) -> dict:
    """
    Break down total savings into three categories:
      - Overstock reduction  : value of excess stock that can be freed up
      - Reorder optimisation : savings from not placing unnecessary reorders
      - Markdown prevention  : estimated markdown cost avoided (15% of overstock value)
    """
    overstock_reduction  = forecast_df["overstock_value_eur"].sum()
    reorder_optimisation = forecast_df.loc[
        forecast_df["overstock_units"] > 0, "overstock_value_eur"
    ].sum() * 0.25  # 25% of overstock value as reorder saving estimate

    markdown_prevention  = overstock_reduction * 0.15  # 15% markdown risk on excess stock

    total_savings = overstock_reduction + reorder_optimisation + markdown_prevention

    return {
        "overstock_reduction":  round(overstock_reduction, 2),
        "reorder_optimisation": round(reorder_optimisation, 2),
        "markdown_prevention":  round(markdown_prevention, 2),
        "total_savings":        round(total_savings, 2),
    }


# ─────────────────────────────────────────────
# 4. PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame, savings: dict) -> None:
    """Print a readable dashboard summary to the console."""
    print("\n" + "=" * 60)
    print("  INVENTORY FORECASTING SUMMARY")
    print("=" * 60)

    print(f"\n  SKUs tracked          : {len(df)}")
    print(f"  Stockout risk (30d)   : {len(df[df['risk'].isin(['Critical','Warning','Watch'])])}")
    print(f"    └ Critical (<7d)    : {len(df[df['risk'] == 'Critical'])}")
    print(f"    └ Warning (7-14d)   : {len(df[df['risk'] == 'Warning'])}")
    print(f"  Overstock value       : €{df['overstock_value_eur'].sum():,.2f}")
    print(f"  Projected savings     : €{savings['total_savings']:,.2f}")

    print("\n" + "-" * 60)
    print("  SAVINGS BREAKDOWN")
    print("-" * 60)
    print(f"  Overstock reduction   : €{savings['overstock_reduction']:,.2f}")
    print(f"  Reorder optimisation  : €{savings['reorder_optimisation']:,.2f}")
    print(f"  Markdown prevention   : €{savings['markdown_prevention']:,.2f}")

    print("\n" + "-" * 60)
    print("  TOP 5 PRODUCTS BY SAVINGS POTENTIAL")
    print("-" * 60)
    top5 = df.head(5)[["product_name", "overstock_units", "overstock_value_eur", "risk"]]
    for _, row in top5.iterrows():
        print(f"  {row['product_name']:<25} {int(row['overstock_units']):>5} units  "
              f"€{row['overstock_value_eur']:>8,.2f}  [{row['risk']}]")

    print("\n" + "-" * 60)
    print("  REORDER RECOMMENDATIONS")
    print("-" * 60)
    reorder = df[df["reorder_units"] > 0][["product_name", "reorder_units", "days_of_supply", "risk"]]
    if len(reorder) > 0:
        for _, row in reorder.iterrows():
            print(f"  {row['product_name']:<25} order {int(row['reorder_units']):>4} units  "
                  f"(only {row['days_of_supply']}d of supply left)  [{row['risk']}]")
    else:
        print("  No immediate reorders required.")

    print("\n" + "=" * 60 + "\n")


# ─────────────────────────────────────────────
# 5. EXPORT RESULTS
# ─────────────────────────────────────────────

def export_results(forecast_df: pd.DataFrame, output_path: str = "forecast_results.csv") -> None:
    """Save forecast results to CSV for use in Power BI or Excel."""
    forecast_df.to_csv(output_path, index=False)
    print(f"✔  Results exported to: {output_path}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # --- Config ---
    INPUT_FILE  = "inventory_data.csv"   # path to your CSV
    OUTPUT_FILE = "forecast_results.csv" # output file
    FORECAST_MONTHS = 3                  # how many months ahead to forecast

    # --- Run pipeline ---
    df           = load_data(INPUT_FILE)
    forecast_df  = forecast_demand(df, periods_ahead=FORECAST_MONTHS)
    savings      = calculate_savings(forecast_df)

    print_summary(forecast_df, savings)
    export_results(forecast_df, OUTPUT_FILE)
