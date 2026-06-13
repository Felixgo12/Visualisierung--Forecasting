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
# DARK MODE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark background */
    .stApp, .stApp > div { background-color: #0D1117 !important; }
    section[data-testid="stSidebar"] { background-color: #161B22 !important; }

    /* All text */
    html, body, [class*="css"], p, span, div, label {
        color: #E6EDF3 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }

    /* Header */
    .dash-header {
        background: linear-gradient(135deg, #0D2137 0%, #0A3D4A 100%);
        border-bottom: 1px solid #02C39A;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .dash-header h1 { color: #E6EDF3 !important; font-size: 1.4rem; font-weight: 700; margin: 0; }
    .dash-header span { color: #02C39A; font-size: 0.8rem; }

    /* KPI cards */
    .kpi-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #02C39A, #028090);
    }
    .kpi-label  { font-size: 0.72rem !important; color: #8B949E !important; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .kpi-value  { font-size: 2rem !important; font-weight: 700 !important; color: #02C39A !important; line-height: 1.1; }
    .kpi-delta-pos { font-size: 0.72rem !important; color: #3FB950 !important; margin-top: 4px; }
    .kpi-delta-neg { font-size: 0.72rem !important; color: #F85149 !important; margin-top: 4px; }
    .kpi-delta-warn { font-size: 0.72rem !important; color: #D29922 !important; margin-top: 4px; }

    /* Section cards */
    .section-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #E6EDF3 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #21262D;
        padding-bottom: 0.5rem;
    }

    /* Upload */
    .stFileUploader > div {
        background: #161B22 !important;
        border: 2px dashed #02C39A !important;
        border-radius: 10px !important;
    }
    .stFileUploader label { color: #E6EDF3 !important; }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #21262D !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        color: #E6EDF3 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 8px !important;
        color: #E6EDF3 !important;
    }
    .streamlit-expanderContent {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
    }

    /* Code block */
    .stCodeBlock { background: #0D1117 !important; border: 1px solid #21262D !important; }
    code { color: #02C39A !important; }

    /* Info/success boxes */
    .stAlert { background: #161B22 !important; border: 1px solid #21262D !important; }

    /* Download button */
    .stDownloadButton > button {
        background: #028090 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover { background: #02C39A !important; }

    /* Pills */
    .pill-critical { background:#2D1B1B; color:#F85149; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; border:1px solid #F85149; }
    .pill-overstock { background:#2D2419; color:#D29922; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; border:1px solid #D29922; }
    .pill-moderate  { background:#152B25; color:#02C39A; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; border:1px solid #02C39A; }
    .pill-low       { background:#1A2D1A; color:#3FB950; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; border:1px solid #3FB950; }
    .pill-ok        { background:#1A1F2D; color:#58A6FF; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600; border:1px solid #58A6FF; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHART THEME (dark)
# ─────────────────────────────────────────────
DARK_BG   = "#161B22"
GRID_COL  = "#21262D"
TICK_COL  = "#8B949E"
TEXT_COL  = "#E6EDF3"
TEAL      = "#02C39A"
TEAL2     = "#028090"
GREEN     = "#3FB950"
RED       = "#F85149"
AMBER     = "#D29922"

def dark_axis(show_grid=True):
    return dict(
        showgrid=show_grid, gridcolor=GRID_COL, gridwidth=1,
        tickfont=dict(color=TICK_COL, size=11),
        linecolor=GRID_COL, linewidth=1, showline=True,
        zeroline=False
    )

# ─────────────────────────────────────────────
# FORECASTING LOGIC
# ─────────────────────────────────────────────
def forecast_demand(df, periods_ahead=3):
    results = []
    for sku, group in df.groupby("sku"):
        group = group.sort_values("date").reset_index(drop=True)
        X = np.arange(len(group)).reshape(-1, 1)
        y = group["units_sold"].values
        model = LinearRegression()
        model.fit(X, y)
        future_indices = np.arange(len(group), len(group) + periods_ahead).reshape(-1, 1)
        future_demand  = np.maximum(model.predict(future_indices), 0)
        current_stock  = group["stock_level"].iloc[-1]
        unit_price     = group["unit_price"].iloc[-1]
        product_name   = group["product_name"].iloc[-1]
        category       = group["category"].iloc[-1]
        total_forecast = future_demand.sum()
        avg_monthly    = y.mean()
        overstock_units = max(0, current_stock - total_forecast)
        overstock_value = round(overstock_units * unit_price, 2)
        reorder_units   = max(0, total_forecast - current_stock)
        days_of_supply  = round((current_stock / avg_monthly) * 30) if avg_monthly > 0 else 999
        if days_of_supply < 7:       risk = "Critical"
        elif days_of_supply < 14:    risk = "Warning"
        elif days_of_supply < 30:    risk = "Watch"
        elif overstock_value > 3000: risk = "Overstock"
        elif overstock_units > 0:    risk = "Moderate"
        else:                        risk = "OK"
        results.append({
            "sku": sku, "product_name": product_name, "category": category,
            "current_stock": int(current_stock), "avg_monthly_demand": round(avg_monthly, 1),
            "forecast_3m_total": round(total_forecast, 0),
            "overstock_units": round(overstock_units, 0),
            "overstock_value_eur": overstock_value,
            "reorder_units": round(reorder_units, 0),
            "days_of_supply": days_of_supply, "risk": risk, "unit_price": unit_price,
            "history_dates": group["date"].tolist(),
            "history_actual": group["units_sold"].tolist(),
            "history_forecast": model.predict(X).tolist(),
            "future_dates": [group["date"].max() + pd.DateOffset(months=i+1) for i in range(periods_ahead)],
            "future_demand": future_demand.tolist(),
        })
    return pd.DataFrame(results).sort_values("overstock_value_eur", ascending=False).reset_index(drop=True)

def calculate_savings(forecast_df):
    ov = forecast_df["overstock_value_eur"].sum()
    ro = forecast_df.loc[forecast_df["overstock_units"] > 0, "overstock_value_eur"].sum() * 0.25
    mp = ov * 0.15
    return {"overstock_reduction": round(ov,2), "reorder_optimisation": round(ro,2),
            "markdown_prevention": round(mp,2), "total_savings": round(ov+ro+mp,2)}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <h1>📦 Inventory Forecasting Dashboard</h1>
    <span>E-Commerce Forecasting Tool &nbsp;·&nbsp; Felix Opitz &nbsp;·&nbsp; Linear Regression Model</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
with st.expander("📂 Upload your inventory CSV", expanded=True):
    st.markdown("Upload a CSV file with the following columns – the order and spelling must match exactly:")
    st.code("date, sku, product_name, category, units_sold, stock_level, unit_price", language="text")
    st.markdown("""
**Column format:**
- `date` → Format: YYYY-MM-DD (e.g. 2026-01-01)
- `sku` → Unique product ID (e.g. SKU-0041)
- `product_name` → Product name (e.g. Laptop stand A3)
- `category` → Product category (e.g. Electronics)
- `units_sold` → Units sold that month (number)
- `stock_level` → Current stock on hand (number)
- `unit_price` → Price per unit in € (e.g. 6.00)
    """)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    st.caption("💡 No file? Demo data is used automatically.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.success(f"✔ Loaded {len(df)} records from your file.")
else:
    df = pd.read_csv("inventory_data.csv", parse_dates=["date"])
    st.info("📊 Showing demo data. Upload your own CSV above.")

df = df.sort_values(["sku", "date"]).reset_index(drop=True)
forecast_df = forecast_demand(df, periods_ahead=3)
savings     = calculate_savings(forecast_df)
stockout_risk = len(forecast_df[forecast_df["risk"].isin(["Critical","Warning","Watch"])])
critical      = len(forecast_df[forecast_df["risk"] == "Critical"])

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">SKUs tracked</div>
        <div class="kpi-value">{len(forecast_df)}</div>
        <div class="kpi-delta-pos">▲ Active products</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Stockout risk (30d)</div>
        <div class="kpi-value">{stockout_risk}</div>
        <div class="kpi-delta-neg">⚠ {critical} critical</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Overstock value</div>
        <div class="kpi-value">€{forecast_df['overstock_value_eur'].sum():,.0f}</div>
        <div class="kpi-delta-neg">▲ Tied-up capital</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Projected savings</div>
        <div class="kpi-value">€{savings['total_savings']:,.0f}</div>
        <div class="kpi-delta-pos">▲ With reorder opt.</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-card"><div class="section-title">Demand Forecast vs Actual</div>', unsafe_allow_html=True)
    selected_sku = st.selectbox("Select SKU", forecast_df["sku"].tolist(), label_visibility="collapsed")
    row = forecast_df[forecast_df["sku"] == selected_sku].iloc[0]
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=row["history_dates"], y=row["history_actual"],
        name="Actual", line=dict(color=TEAL, width=2.5), mode="lines+markers",
        marker=dict(size=5, color=TEAL)))
    fig_line.add_trace(go.Scatter(x=row["history_dates"], y=row["history_forecast"],
        name="Forecast (fit)", line=dict(color=TEAL2, width=2, dash="dot"), mode="lines"))
    fig_line.add_trace(go.Scatter(x=row["future_dates"], y=row["future_demand"],
        name="Projected", line=dict(color=AMBER, width=2, dash="dash"),
        mode="lines+markers", marker=dict(size=5, color=AMBER)))
    fig_line.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COL),
        legend=dict(orientation="h", y=-0.25, font=dict(color=TICK_COL, size=11),
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis={**dark_axis(False)},
        yaxis=dark_axis(True),
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-card"><div class="section-title">Stock Level by Category</div>', unsafe_allow_html=True)
    cat_df = df.groupby("category").agg(current_stock=("stock_level","last")).reset_index()
    reorder_map = forecast_df.groupby("category")["avg_monthly_demand"].sum().reset_index()
    reorder_map.columns = ["category","reorder_point"]
    cat_df = cat_df.merge(reorder_map, on="category", how="left")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Current stock", x=cat_df["category"], y=cat_df["current_stock"],
        marker_color=TEAL2, marker_line_width=0, opacity=0.9))
    fig_bar.add_trace(go.Bar(name="Reorder point", x=cat_df["category"], y=cat_df["reorder_point"],
        marker_color=RED, marker_line_width=0, opacity=0.9))
    fig_bar.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COL), barmode="group",
        legend=dict(orientation="h", y=-0.25, font=dict(color=TICK_COL, size=11),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dark_axis(False), yaxis=dark_axis(True),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAVINGS BY PRODUCT
# ─────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
s1, _, s4 = st.columns([3,2,1])
with s1: st.markdown('<div class="section-title">Savings Potential by Product</div>', unsafe_allow_html=True)
with s4: st.markdown(f"<span style='color:#02C39A;font-weight:700;font-size:1rem'>€{savings['total_savings']:,.0f} total</span>", unsafe_allow_html=True)

pill_map = {"Critical":"pill-critical","Warning":"pill-critical","Overstock":"pill-overstock",
            "Moderate":"pill-moderate","Watch":"pill-moderate","OK":"pill-ok","Low risk":"pill-low"}

cols = st.columns(len(forecast_df))
for i, (_, row) in enumerate(forecast_df.iterrows()):
    pct = min(int(row["overstock_units"] / max(row["current_stock"],1) * 100), 100)
    bar_color = RED if pct > 70 else AMBER if pct > 40 else TEAL
    pill_class = pill_map.get(row["risk"],"pill-ok")
    with cols[i]:
        st.markdown(f"""
        <div style="background:#0D1117;border:1px solid #21262D;border-radius:10px;padding:12px;">
            <div style="font-size:12px;font-weight:600;color:#E6EDF3">{row['product_name']}</div>
            <div style="font-size:10px;color:#8B949E;margin-bottom:8px">{row['sku']}</div>
            <div style="background:#21262D;border-radius:4px;height:6px;margin-bottom:6px;overflow:hidden">
                <div style="width:{pct}%;background:{bar_color};height:100%;border-radius:4px"></div>
            </div>
            <div style="font-size:10px;color:#8B949E">{int(row['overstock_units'])} units overstocked</div>
            <div style="font-size:15px;font-weight:700;color:#02C39A;margin-top:4px">€{row['overstock_value_eur']:,.0f}</div>
            <span class="{pill_class}">{row['risk']}</span>
        </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAVINGS BREAKDOWN CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">Savings Breakdown</div>', unsafe_allow_html=True)
fig_savings = go.Figure(go.Bar(
    x=["Overstock reduction","Reorder optimisation","Markdown prevention"],
    y=[savings["overstock_reduction"], savings["reorder_optimisation"], savings["markdown_prevention"]],
    marker_color=[TEAL2, TEAL, "#94D2BD"],
    text=[f"€{v:,.0f}" for v in [savings["overstock_reduction"], savings["reorder_optimisation"], savings["markdown_prevention"]]],
    textposition="outside", textfont=dict(color=TEXT_COL, size=13)
))
fig_savings.update_layout(
    height=260, margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
    font=dict(color=TEXT_COL),
    xaxis=dark_axis(False), yaxis={**dark_axis(True), "showticklabels": True},
)
st.plotly_chart(fig_savings, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
export_df = forecast_df[["sku","product_name","category","current_stock",
    "avg_monthly_demand","forecast_3m_total","overstock_units",
    "overstock_value_eur","reorder_units","days_of_supply","risk"]]
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download forecast results as CSV",
    data=csv, file_name="forecast_results.csv", mime="text/csv"
)
