import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Care Transition Efficiency & Placement Outcome Analytics",
    layout="wide"
)

st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.markdown("### Data Analytics Dashboard for Optimizing the UAC Care Pipeline")

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

    df["Transfer Efficiency Ratio"] = (
        df["Children transferred out of CBP custody"] /
        df["Children in CBP custody"]
    )

    df["Discharge Effectiveness Index"] = (
        df["Children discharged from HHS Care"] /
        df["Children in HHS Care"]
    )

    df["Pipeline Throughput Rate"] = (
        df["Children discharged from HHS Care"] /
        df["Children apprehended and placed in CBP custody*"]
    )

    df["CBP Backlog"] = (
        df["Children in CBP custody"] -
        df["Children transferred out of CBP custody"]
    )

    df["HHS Backlog"] = (
        df["Children in HHS Care"] -
        df["Children discharged from HHS Care"]
    )

    df["Net Daily Pressure"] = (
        df["Children apprehended and placed in CBP custody*"] -
        df["Children discharged from HHS Care"]
    )

    df["Outcome Stability Score"] = (
        df["Discharge Effectiveness Index"]
        .rolling(window=7, min_periods=3)
        .std()
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df["Day Name"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df

df = load_data()

st.sidebar.header("Filters")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
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

transfer_threshold = st.sidebar.slider(
    "Transfer bottleneck threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)

discharge_threshold = st.sidebar.slider(
    "Discharge bottleneck threshold",
    min_value=0.0,
    max_value=0.10,
    value=0.01,
    step=0.001
)

filtered_df["Transfer Bottleneck Flag"] = np.where(
    filtered_df["Transfer Efficiency Ratio"] < transfer_threshold, 1, 0
)

filtered_df["Discharge Bottleneck Flag"] = np.where(
    filtered_df["Discharge Effectiveness Index"] < discharge_threshold, 1, 0
)

st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Avg Transfer Efficiency", f"{filtered_df['Transfer Efficiency Ratio'].mean():.3f}")
k2.metric("Avg Discharge Effectiveness", f"{filtered_df['Discharge Effectiveness Index'].mean():.3f}")
k3.metric("Avg Pipeline Throughput", f"{filtered_df['Pipeline Throughput Rate'].mean():.3f}")
k4.metric("Avg Net Daily Pressure", f"{filtered_df['Net Daily Pressure'].mean():,.0f}")

k5, k6 = st.columns(2)
k5.metric("Avg CBP Backlog", f"{filtered_df['CBP Backlog'].mean():,.0f}")
k6.metric("Avg HHS Backlog", f"{filtered_df['HHS Backlog'].mean():,.0f}")

st.subheader("Threshold-Based Alerts")

transfer_alert_count = int(filtered_df["Transfer Bottleneck Flag"].sum())
discharge_alert_count = int(filtered_df["Discharge Bottleneck Flag"].sum())

c1, c2 = st.columns(2)

if transfer_alert_count > 0:
    c1.warning(f"Transfer bottleneck days detected: {transfer_alert_count}")
else:
    c1.success("No transfer bottleneck days detected.")

if discharge_alert_count > 0:
    c2.warning(f"Discharge bottleneck days detected: {discharge_alert_count}")
else:
    c2.success("No discharge bottleneck days detected.")

st.subheader("Care Pipeline Flow Visualization")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(filtered_df["Date"], filtered_df["Children in CBP custody"], label="CBP Custody")
ax1.plot(filtered_df["Date"], filtered_df["Children in HHS Care"], label="HHS Care")
ax1.plot(filtered_df["Date"], filtered_df["Children discharged from HHS Care"], label="Discharged")
ax1.set_title("Pipeline Stages Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Children Count")
ax1.legend()
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Transfer & Discharge Efficiency")

col1, col2 = st.columns(2)

with col1:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(filtered_df["Date"], filtered_df["Transfer Efficiency Ratio"])
    ax2.axhline(transfer_threshold, linestyle="--")
    ax2.set_title("Transfer Efficiency Ratio")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Ratio")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

with col2:
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(filtered_df["Date"], filtered_df["Discharge Effectiveness Index"])
    ax3.axhline(discharge_threshold, linestyle="--")
    ax3.set_title("Discharge Effectiveness Index")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Ratio")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

st.subheader("Bottleneck Detection Charts")

col3, col4 = st.columns(2)

with col3:
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(filtered_df["Date"], filtered_df["CBP Backlog"])
    ax4.set_title("CBP Backlog Over Time")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Backlog")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

with col4:
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(filtered_df["Date"], filtered_df["HHS Backlog"])
    ax5.set_title("HHS Backlog Over Time")
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Backlog")
    plt.xticks(rotation=45)
    st.pyplot(fig5)

st.subheader("Outcome Trend Analysis")

fig6, ax6 = plt.subplots(figsize=(12, 5))
ax6.plot(filtered_df["Date"], filtered_df["Outcome Stability Score"])
ax6.set_title("Outcome Stability Score")
ax6.set_xlabel("Date")
ax6.set_ylabel("Rolling Std Dev")
plt.xticks(rotation=45)
st.pyplot(fig6)

st.subheader("Weekday Analysis")

weekday_analysis = filtered_df.groupby("Day Name")[[
    "Transfer Efficiency Ratio",
    "Discharge Effectiveness Index",
    "Pipeline Throughput Rate",
    "CBP Backlog",
    "HHS Backlog"
]].mean()

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_analysis = weekday_analysis.reindex(weekday_order)
st.dataframe(weekday_analysis)

st.subheader("Month-over-Month Trends")

monthly_analysis = filtered_df.groupby("Month")[[
    "Children apprehended and placed in CBP custody*",
    "Children discharged from HHS Care",
    "CBP Backlog",
    "HHS Backlog",
    "Net Daily Pressure"
]].mean()

st.dataframe(monthly_analysis)

st.subheader("Download Filtered Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered dataset as CSV",
    data=csv_data,
    file_name="filtered_uac_pipeline_analytics.csv",
    mime="text/csv"
)