import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Care Transition Analytics Dashboard")

# Load data
df = pd.read_csv("cleaned_uac_pipeline_analytics.csv")
df["Date"] = pd.to_datetime(df["Date"])

# KPIs
df["Transfer Efficiency"] = df["Children transferred out of CBP custody"] / df["Children in CBP custody"]
df["Discharge Effectiveness"] = df["Children discharged from HHS Care"] / df["Children in HHS Care"]

st.subheader("KPIs")

col1, col2 = st.columns(2)

col1.metric("Avg Transfer Efficiency", round(df["Transfer Efficiency"].mean(), 3))
col2.metric("Avg Discharge Effectiveness", round(df["Discharge Effectiveness"].mean(), 3))

# Chart
st.subheader("Inflow vs Outflow")

fig, ax = plt.subplots()
ax.plot(df["Date"], df["Children apprehended and placed in CBP custody*"], label="Apprehended")
ax.plot(df["Date"], df["Children discharged from HHS Care"], label="Discharged")
ax.legend()

st.pyplot(fig)
