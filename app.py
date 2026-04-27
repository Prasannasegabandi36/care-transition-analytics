import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt

st.set_page_config(
    page_title="Care Transition Analytics",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #1f4e79;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.metric-card {
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #0d47a1;
}
.metric-label {
    font-size: 15px;
    color: #444;
}
.insight-box {
    background-color: #f8fbff;
    border-left: 6px solid #1f77b4;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Care Transition Efficiency & Placement Outcome Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A Data Science dashboard to analyze CBP → HHS → Discharge pipeline efficiency</div>', unsafe_allow_html=True)
st.write("")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_uac_pipeline_analytics.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all")
    df = df.sort_values("Date").reset_index(drop=True)

    df["Transfer Efficiency"] = df["Children transferred out of CBP custody"] / df["Children in CBP custody"]
    df["Discharge Effectiveness"] = df["Children discharged from HHS Care"] / df["Children in HHS Care"]
    df["Pipeline Throughput"] = df["Children discharged from HHS Care"] / df["Children apprehended and placed in CBP custody*"]

    df["CBP Backlog"] = df["Children in CBP custody"] - df["Children transferred out of CBP custody"]
    df["HHS Backlog"] = df["Children in HHS Care"] - df["Children discharged from HHS Care"]
    df["Net Pressure"] = df["Children apprehended and placed in CBP custody*"] - df["Children discharged from HHS Care"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Day Name"] = df["Date"].dt.day_name()

    return df

df = load_data()

st.sidebar.header("Dashboard Filters")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["Date"].dt.date >= start_date) &
        (df["Date"].dt.date <= end_date)
    ].copy()
else:
    filtered_df = df.copy()

st.sidebar.markdown("---")
st.sidebar.info("Use filters to analyze different time periods.")

# KPI CARDS
st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_df['Transfer Efficiency'].mean():.3f}</div>
        <div class="metric-label">Avg Transfer Efficiency</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_df['Discharge Effectiveness'].mean():.3f}</div>
        <div class="metric-label">Avg Discharge Effectiveness</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_df['Pipeline Throughput'].mean():.3f}</div>
        <div class="metric-label">Pipeline Throughput</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{filtered_df['Net Pressure'].mean():,.0f}</div>
        <div class="metric-label">Avg Net Pressure</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# CHART 1
st.subheader("Pipeline Inflow vs Outflow")

flow_df = filtered_df[[
    "Date",
    "Children apprehended and placed in CBP custody*",
    "Children discharged from HHS Care"
]].rename(columns={
    "Children apprehended and placed in CBP custody*": "Apprehended / Inflow",
    "Children discharged from HHS Care": "Discharged / Outflow"
})

flow_long = flow_df.melt(id_vars="Date", var_name="Metric", value_name="Count")

fig1 = px.line(
    flow_long,
    x="Date",
    y="Count",
    color="Metric",
    markers=True,
    title="Daily Inflow vs Outflow Trend"
)
st.plotly_chart(fig1, use_container_width=True)

# CHARTS SIDE BY SIDE
c1, c2 = st.columns(2)

with c1:
    st.subheader("Efficiency Trend")
    eff_df = filtered_df[["Date", "Transfer Efficiency", "Discharge Effectiveness"]]
    eff_long = eff_df.melt(id_vars="Date", var_name="Metric", value_name="Ratio")

    fig2 = px.line(
        eff_long,
        x="Date",
        y="Ratio",
        color="Metric",
        markers=True,
        title="Transfer vs Discharge Efficiency"
    )
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.subheader("Backlog Trend")
    backlog_df = filtered_df[["Date", "CBP Backlog", "HHS Backlog"]]
    backlog_long = backlog_df.melt(id_vars="Date", var_name="Backlog Type", value_name="Count")

    fig3 = px.line(
        backlog_long,
        x="Date",
        y="Count",
        color="Backlog Type",
        markers=True,
        title="CBP and HHS Backlog Over Time"
    )
    st.plotly_chart(fig3, use_container_width=True)

# MONTHLY ANALYSIS
st.subheader("Month-over-Month Analysis")

monthly_df = filtered_df.groupby("Month")[[
    "Children apprehended and placed in CBP custody*",
    "Children discharged from HHS Care",
    "CBP Backlog",
    "HHS Backlog",
    "Net Pressure"
]].mean().reset_index()

fig4 = px.bar(
    monthly_df,
    x="Month",
    y=["CBP Backlog", "HHS Backlog"],
    barmode="group",
    title="Monthly Average Backlog Comparison"
)
st.plotly_chart(fig4, use_container_width=True)

st.dataframe(monthly_df, use_container_width=True)

# INSIGHTS
st.subheader("Key Insights")

avg_inflow = filtered_df["Children apprehended and placed in CBP custody*"].mean()
avg_outflow = filtered_df["Children discharged from HHS Care"].mean()
avg_pressure = filtered_df["Net Pressure"].mean()

st.markdown(f"""
<div class="insight-box">
<b>Insight 1:</b> Average daily inflow is <b>{avg_inflow:.0f}</b>, while average daily outflow is <b>{avg_outflow:.0f}</b>.<br><br>
<b>Insight 2:</b> Average net pressure is <b>{avg_pressure:.0f}</b>. Positive pressure means more children are entering than exiting.<br><br>
<b>Insight 3:</b> Backlog charts help identify where delays are accumulating in the pipeline.<br><br>
<b>Insight 4:</b> Transfer Efficiency and Discharge Effectiveness help measure operational performance over time.
</div>
""", unsafe_allow_html=True)

# RAW DATA
with st.expander("View Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

# DOWNLOAD
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Filtered Data",
    csv,
    "filtered_care_transition_data.csv",
    "text/csv"
)

   
        

    



    
