import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Care Transition Analytics", layout="wide")

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #1f4e79;
}
.subtitle {
    font-size: 16px;
    color: #555;
}
.metric-box {
    background: #f0f7ff;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #0d47a1;
}
.metric-label {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Care Transition Efficiency Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze CBP → HHS → Discharge pipeline efficiency</div>', unsafe_allow_html=True)

# -------------------------------
# Load Data
# -------------------------------
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

    df = df.dropna()
    df = df.sort_values("Date")

    df["Transfer Efficiency"] = df["Children transferred out of CBP custody"] / df["Children in CBP custody"]
    df["Discharge Effectiveness"] = df["Children discharged from HHS Care"] / df["Children in HHS Care"]
    df["Net Pressure"] = df["Children apprehended and placed in CBP custody*"] - df["Children discharged from HHS Care"]

    return df

df = load_data()

# -------------------------------
# Sidebar Filter
# -------------------------------
st.sidebar.header("Filters")

start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

filtered_df = df[(df["Date"] >= pd.to_datetime(start_date)) &
                 (df["Date"] <= pd.to_datetime(end_date))]

# -------------------------------
# KPI Section
# -------------------------------
st.subheader("Key Performance Indicators")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{filtered_df['Transfer Efficiency'].mean():.3f}</div>
        <div class="metric-label">Transfer Efficiency</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{filtered_df['Discharge Effectiveness'].mean():.3f}</div>
        <div class="metric-label">Discharge Effectiveness</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{filtered_df['Net Pressure'].mean():.0f}</div>
        <div class="metric-label">Net Pressure</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Chart 1
# -------------------------------
st.subheader("Inflow vs Outflow")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(filtered_df["Date"], filtered_df["Children apprehended and placed in CBP custody*"], label="Inflow", color="blue")
ax.plot(filtered_df["Date"], filtered_df["Children discharged from HHS Care"], label="Outflow", color="green")
ax.set_title("Inflow vs Outflow Trend")
ax.legend()
st.pyplot(fig)

# -------------------------------
# Chart 2
# -------------------------------
st.subheader("Efficiency Trends")

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(filtered_df["Date"], filtered_df["Transfer Efficiency"], label="Transfer Efficiency", color="orange")
ax2.plot(filtered_df["Date"], filtered_df["Discharge Effectiveness"], label="Discharge Effectiveness", color="red")
ax2.set_title("Efficiency Over Time")
ax2.legend()
st.pyplot(fig2)

# -------------------------------
# Insights
# -------------------------------
st.subheader("Insights")

st.info(f"""
- Average Transfer Efficiency: {filtered_df['Transfer Efficiency'].mean():.3f}
- Average Discharge Effectiveness: {filtered_df['Discharge Effectiveness'].mean():.3f}
- Net Pressure indicates system load (positive means backlog risk)
""")

# -------------------------------
# Raw Data
# -------------------------------
with st.expander("View Data"):
    st.dataframe(filtered_df)
