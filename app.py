# AI Sustainability for Water Management in Urban Areas
# Full Streamlit app with sidebar, charts, explanations, and AI forecast

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet

# ----------------------
# Generate a large realistic dataset
# ----------------------
np.random.seed(42)
dates = pd.date_range(start="2018-01-01", end="2025-08-01", freq="D")
cities = ["City A", "City B", "City C", "City D"]
data = []

for date in dates:
    for city in cities:
        consumption = np.random.normal(500000, 80000)  # liters
        rainfall = np.random.uniform(0, 200)  # mm
        population = np.random.randint(100000, 500000)
        temperature = np.random.uniform(15, 40)
        data.append([date, city, max(consumption, 0), rainfall, population, temperature])

df = pd.DataFrame(data, columns=["Date", "City", "Consumption", "Rainfall", "Population", "Temperature"])

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="AI Sustainability - Water Management", layout="wide")
st.title("💧 AI Sustainability for Water Management in Urban Areas")

# Sidebar Filters
st.sidebar.header("Filters")
selected_cities = st.sidebar.multiselect("Select Cities", options=df["City"].unique(), default=df["City"].unique())
start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

# Filter Data
filtered_df = df[
    (df["City"].isin(selected_cities)) &
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

# ----------------------
# Chart 1: Time Series
# ----------------------
st.subheader("📈 Daily Water Consumption Over Time")
fig_ts = px.line(filtered_df, x="Date", y="Consumption", color="City", title="Water Consumption Over Time")
st.plotly_chart(fig_ts, use_container_width=True)
st.markdown("**Explanation:** This time series shows daily water consumption per city, helping detect seasonal trends and sudden spikes in usage.")

# ----------------------
# Chart 2: Monthly Average Consumption
# ----------------------
st.subheader("📊 Average Monthly Water Consumption")
df_monthly = filtered_df.copy()
df_monthly["Month"] = df_monthly["Date"].dt.to_period("M").astype(str)
monthly_avg = df_monthly.groupby(["Month", "City"])["Consumption"].mean().reset_index()
fig_bar = px.bar(monthly_avg, x="Month", y="Consumption", color="City", barmode="group",
                 title="Average Monthly Consumption")
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("**Explanation:** This chart shows average monthly consumption, allowing comparisons between cities and months.")

# ----------------------
# Chart 3: Consumption Distribution
# ----------------------
st.subheader("📦 Distribution of Daily Water Consumption")
fig_hist = px.histogram(filtered_df, x="Consumption", nbins=40, color="City", marginal="box")
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("**Explanation:** The histogram reveals the distribution of daily consumption values, highlighting typical and extreme usage levels.")

# ----------------------
# Chart 4: Correlation Heatmap
# ----------------------
st.subheader("🔍 Correlation Between Variables")
corr = filtered_df[["Consumption", "Rainfall", "Population", "Temperature"]].corr()
fig_corr, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="Blues", square=True, ax=ax)
st.pyplot(fig_corr)
st.markdown("**Explanation:** The heatmap shows correlations between factors affecting water consumption, such as rainfall and temperature.")

# ----------------------
# AI Forecast with Prophet
# ----------------------
st.subheader("🤖 AI Forecast of Water Consumption (Next 30 Days)")

city_for_forecast = st.selectbox("Select City for Forecast", options=filtered_df["City"].unique())
forecast_df = filtered_df[filtered_df["City"] == city_for_forecast][["Date", "Consumption"]].rename(
    columns={"Date": "ds", "Consumption": "y"}
)

model = Prophet()
model.fit(forecast_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig_forecast = px.line(forecast, x="ds", y="yhat", title=f"Predicted Water Consumption for {city_for_forecast}")
fig_forecast.add_scatter(x=forecast_df["ds"], y=forecast_df["y"], mode="lines", name="Actual")
st.plotly_chart(fig_forecast, use_container_width=True)
st.markdown(f"**Explanation:** This forecast predicts water consumption for the next 30 days in {city_for_forecast}, based on historical patterns.")

# ----------------------
# Dataset Download
# ----------------------
st.subheader("📥 Download Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="filtered_water_data.csv", mime="text/csv")
