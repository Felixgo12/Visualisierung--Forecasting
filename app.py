"""
E-Commerce Inventory Forecasting Dashboard
===========================================
Author  : Felix Opitz
Stack   : Streamlit + Pandas + Scikit-learn + Plotly
Run     : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Forecasting Dashboard",
    page_icon="📦",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (Teal / Green theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0F7F4; }

    /* Top header bar */
    .main-header {
        background: #028090;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: white;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
    }
    .main-header p {
        color: #B5D4F4;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #C8E6D8;
        text-align: center;
        box-shadow: 0 2px 8px rgba(2,128,144,0.07);
    }
    .kpi-label  { font-size: 0.78rem; color: #64748B; margin-bottom: 4px; }
    .kpi-value  { font-size: 1.8rem; font-weight: 700; color: #028090; line-height: 1.1; }
    .kpi-delta-pos { font-size: 0.75rem; color: #3B6D11; margin-top: 4px; }
    .kpi-delta-neg { font-size: 0.75rem; color: #A32D2D; margin-top: 4px; }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #C8E6D8;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(2,128,144,0.07);
    }

    /* Upload area */
    .upload-box {
        background: white;
        border: 2px dashed #028090;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }

    /* Risk pills */
    .pill-critical { background:#FCEBEB; color:#A32D2D; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .pill-overstock { background:#FAEEDA; color:#854F0B; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .pill-moderate  { background:#CCFBF1; color:#0F766E; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .pill-low       { background:#D1FAE5; color:#065F46; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
    .pill-ok        { background:#EFF6FF; color:#1D4ED8; padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FORECASTING LOGIC
# ─────────────────────────────────────────────

def forecast_demand(df: pd.DataFrame, periods_ahead: int = 3) -> pd.DataFrame:
    results = []
    for sku, group in df.groupby("sku"):
        group = group.sort_values("date").reset_index(drop=True)
        X = np.arange(len(group)).reshape(-1, 1)
        y = group["units_sold"].values

        model = LinearRegression()
        model.fit(X, y)

        future_indices = np.arange(len(group), len(group) + periods_ahead).reshape(-1, 1)
        future_demand  = np.maximum(model.predict(future_indices), 0)

        current_stock    = group["stock_level"].iloc[-1]
        unit_price       = group["unit_price"].iloc[-1]
        product_name     = group["product_name"].iloc[-1]
        category         = group["category"].iloc[-1]
        total_forecast   = future_demand.sum()
        avg_monthly      = y.mean()

        overstock_units  = max(0, current_stock - total_forecast)
        overstock_value  = round(overstock_units * unit_price, 2)
        reorder_units    = max(0, total_forecast - current_stock)
        days_of_supply   = round((current_stock / avg_monthly) * 30) if avg_monthly > 0 else 999

        if days_of_supply < 7:       risk = "Critical"
        elif days_of_supply < 14:    risk = "Warning"
        elif days_of_supply < 30:    risk = "Watch"
        elif overstock_value > 3000: risk = "Overstock"
        elif overstock_units > 0:    risk = "Moderate"
        else:                        risk = "OK"

        # Full history for chart
        history_dates  = group["date"].tolist()
        history_actual = group["units_sold"].tolist()
        last_date      = group["date"].max()
        future_dates   = [last_date + pd.DateOffset(months=i+1) for i in range(periods_ahead)]
        history_forecast = model.predict(X).tolist()

        results.append({
            "sku": sku, "product_name": product_name, "category": category,
            "current_stock": int(current_stock), "avg_monthly_demand": round(avg_monthly, 1),
            "forecast_3m_total": round(total_forecast, 0),
            "overstock_units": round(overstock_units, 0),
            "overstock_value_eur": overstock_value,
            "reorder_units": round(reorder_units, 0),
            "days_of_supply": days_of_supply, "risk": risk,
            "unit_price": unit_price,
            "history_dates": history_dates, "history_actual": history_actual,
            "history_forecast": history_forecast,
            "future_dates": future_dates, "future_demand": future_demand.tolist(),
        })

    return pd.DataFrame(results).sort_values("overstock_value_eur", ascending=False).reset_index(drop=True)


def calculate_savings(forecast_df: pd.DataFrame) -> dict:
    overstock_reduction  = forecast_df["overstock_value_eur"].sum()
    reorder_optimisation = forecast_df.loc[forecast_df["overstock_units"] > 0, "overstock_value_eur"].sum() * 0.25
    markdown_prevention  = overstock_reduction * 0.15
    return {
        "overstock_reduction":  round(overstock_reduction, 2),
        "reorder_optimisation": round(reorder_optimisation, 2),
        "markdown_prevention":  round(markdown_prevention, 2),
        "total_savings":        round(overstock_reduction + reorder_optimisation + markdown_prevention, 2),
    }


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📦 Inventory Forecasting Dashboard</h1>
    <p>E-Commerce Forecasting Tool &nbsp;·&nbsp; Felix Opitz &nbsp;·&nbsp; Linear Regression Model</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
with st.expander("📂 Upload your inventory CSV", expanded=True):
    st.markdown("Upload a CSV file with columns: `date, sku, product_name, category, units_sold, stock_level, unit_price`")
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    st.caption("💡 No file? The demo data below will be used automatically.")

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.success(f"✔ Loaded {len(df)} records from your file.")
else:
    df = pd.read_csv("inventory_data.csv", parse_dates=["date"])
    st.info("📊 Showing demo data (inventory_data.csv). Upload your own file above.")

df = df.sort_values(["sku", "date"]).reset_index(drop=True)

# Run forecast
forecast_df = forecast_demand(df, periods_ahead=3)
savings     = calculate_savings(forecast_df)

stockout_risk = len(forecast_df[forecast_df["risk"].isin(["Critical", "Warning", "Watch"])])
critical      = len(forecast_df[forecast_df["risk"] == "Critical"])


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">SKUs tracked</div>
        <div class="kpi-value">{len(forecast_df)}</div>
        <div class="kpi-delta-pos">▲ Active products</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Stockout risk (30d)</div>
        <div class="kpi-value">{stockout_risk}</div>
        <div class="kpi-delta-neg">⚠ {critical} critical</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Overstock value</div>
        <div class="kpi-value">€{forecast_df['overstock_value_eur'].sum():,.0f}</div>
        <div class="kpi-delta-neg">▲ Tied-up capital</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Projected savings</div>
        <div class="kpi-value">€{savings['total_savings']:,.0f}</div>
        <div class="kpi-delta-pos">▲ With reorder optimisation</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Demand forecast vs actual (units)**")

    sku_options = forecast_df["sku"].tolist()
    selected_sku = st.selectbox("Select SKU", sku_options, label_visibility="collapsed")
    row = forecast_df[forecast_df["sku"] == selected_sku].iloc[0]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=row["history_dates"], y=row["history_actual"],
        name="Actual", line=dict(color="#02C39A", width=2.5),
        mode="lines+markers"
    ))
    fig_line.add_trace(go.Scatter(
        x=row["history_dates"], y=row["history_forecast"],
        name="Forecast (fit)", line=dict(color="#028090", width=2, dash="dot"),
        mode="lines"
    ))
    fig_line.add_trace(go.Scatter(
        x=row["future_dates"], y=row["future_demand"],
        name="Projected", line=dict(color="#94D2BD", width=2, dash="dash"),
        mode="lines+markers"
    ))
    fig_line.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F0F7F4")
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Stock level by category**")

    cat_df = df.groupby("category").agg(
        current_stock=("stock_level", "last"),
    ).reset_index()

    reorder_map = forecast_df.groupby("category")["avg_monthly_demand"].sum().reset_index()
    reorder_map.columns = ["category", "reorder_point"]
    cat_df = cat_df.merge(reorder_map, on="category", how="left")

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Current stock", x=cat_df["category"], y=cat_df["current_stock"], marker_color="#028090", marker_line_width=0))
    fig_bar.add_trace(go.Bar(name="Reorder point", x=cat_df["category"], y=cat_df["reorder_point"], marker_color="#E24B4A", marker_line_width=0))
    fig_bar.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        barmode="group", legend=dict(orientation="h", y=-0.2),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F0F7F4")
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SAVINGS BREAKDOWN
# ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns([3,1,1,1])
with s1: st.markdown("**Savings potential by product**")
with s4: st.markdown(f"<span style='color:#0F766E;font-weight:700;font-size:1rem'>€{savings['total_savings']:,.0f} total</span>", unsafe_allow_html=True)

pill_map = {
    "Critical": "pill-critical", "Warning": "pill-critical",
    "Overstock": "pill-overstock", "Moderate": "pill-moderate",
    "Watch": "pill-moderate", "OK": "pill-ok", "Low risk": "pill-low"
}

cols = st.columns(len(forecast_df))
for i, (_, row) in enumerate(forecast_df.iterrows()):
    pct = min(int(row["overstock_units"] / max(row["current_stock"], 1) * 100), 100)
    bar_color = "#E24B4A" if pct > 70 else "#EF9F27" if pct > 40 else "#02C39A"
    pill_class = pill_map.get(row["risk"], "pill-ok")
    with cols[i]:
        st.markdown(f"""
        <div style="border:1px solid #C8E6D8;border-radius:10px;padding:12px;">
            <div style="font-size:12px;font-weight:600;color:#1E293B">{row['product_name']}</div>
            <div style="font-size:10px;color:#94A3B8;margin-bottom:8px">{row['sku']}</div>
            <div style="background:#F0F7F4;border-radius:4px;height:6px;margin-bottom:6px;overflow:hidden">
                <div style="width:{pct}%;background:{bar_color};height:100%;border-radius:4px"></div>
            </div>
            <div style="font-size:10px;color:#64748B">{int(row['overstock_units'])} units overstocked</div>
            <div style="font-size:14px;font-weight:700;color:#0F766E;margin-top:4px">€{row['overstock_value_eur']:,.0f}</div>
            <span class="{pill_class}">{row['risk']}</span>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SAVINGS BREAKDOWN CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("**Savings breakdown**")

fig_savings = go.Figure(go.Bar(
    x=["Overstock reduction", "Reorder optimisation", "Markdown prevention"],
    y=[savings["overstock_reduction"], savings["reorder_optimisation"], savings["markdown_prevention"]],
    marker_color=["#028090", "#02C39A", "#94D2BD"],
    text=[f"€{v:,.0f}" for v in [savings["overstock_reduction"], savings["reorder_optimisation"], savings["markdown_prevention"]]],
    textposition="outside"
))
fig_savings.update_layout(
    height=220, margin=dict(l=0, r=0, t=20, b=0),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F0F7F4", showticklabels=False)
)
st.plotly_chart(fig_savings, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DOWNLOAD RESULTS
# ─────────────────────────────────────────────
export_df = forecast_df[[
    "sku", "product_name", "category", "current_stock",
    "avg_monthly_demand", "forecast_3m_total", "overstock_units",
    "overstock_value_eur", "reorder_units", "days_of_supply", "risk"
]]
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download forecast results as CSV",
    data=csv,
    file_name="forecast_results.csv",
    mime="text/csv"
)
