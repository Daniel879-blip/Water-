# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import io

st.set_page_config(page_title='AI Water Sustainability Dashboard', layout='wide')

# --- Sidebar ---
st.sidebar.title('Controls')
use_sample = st.sidebar.checkbox('Use sample dataset', value=True)
uploaded_file = st.sidebar.file_uploader('Upload CSV (should contain Date, Consumption)', type=['csv'])
forecast_horizon = st.sidebar.slider('Forecast horizon (days)', 1, 90, 14)
train_pct = st.sidebar.slider('Train split (%)', 50, 95, 80)
n_estimators = st.sidebar.slider('RF n_estimators', 10, 500, 100)
max_depth = st.sidebar.slider('RF max_depth (0 means None)', 0, 50, 10)
detect_anomalies = st.sidebar.checkbox('Enable anomaly detection', value=True)
show_ai_suggestions = st.sidebar.checkbox('Show AI suggestions', value=True)
download_model = st.sidebar.button('Export model (placeholder)')

st.title('AI for Sustainable Water Management — Urban Areas')
st.markdown('''This interactive dashboard demonstrates AI-driven approaches to forecast water demand, detect anomalies, '
              'and provide sustainability metrics for urban water management. Use the sidebar to upload your data or use the sample dataset.''')

# --- Data loading ---
def make_sample_data(days=365*2):
    rng = pd.date_range(end=datetime.today(), periods=days, freq='D')
    base = 100 + 10*np.sin(np.linspace(0, 12*np.pi, days))  # seasonal
    trend = np.linspace(0, 20, days)
    noise = np.random.normal(scale=5, size=days)
    consumption = base + trend + noise + 20*(np.random.rand(days) < 0.01)  # occasional jumps
    df = pd.DataFrame({'Date': rng, 'Consumption': consumption})
    return df

if use_sample or uploaded_file is None:
    df = make_sample_data(365*2)
else:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    if 'Consumption' not in df.columns:
        st.error('CSV must contain a "Consumption" column.')
        st.stop()

df = df.sort_values('Date').reset_index(drop=True)
df['Date'] = pd.to_datetime(df['Date'])

st.subheader('Dataset preview')
st.dataframe(df.tail(10))

# --- Feature engineering ---
def create_lag_features(df, lags=14):
    data = df.copy()
    for lag in range(1, lags+1):
        data[f'lag_{lag}'] = data['Consumption'].shift(lag)
    data['dayofyear'] = data['Date'].dt.dayofyear
    data = data.dropna().reset_index(drop=True)
    return data

lags = 14
data = create_lag_features(df, lags=lags)

# Split train/test
split_idx = int(len(data) * train_pct / 100)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

X_train = train.drop(['Date', 'Consumption'], axis=1)
y_train = train['Consumption']
X_test = test.drop(['Date', 'Consumption'], axis=1)
y_test = test['Consumption']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model training ---
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth==0 else max_depth), random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test
preds = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, preds, squared=False)

st.subheader('Forecasting results')
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('**Consumption time-series and forecast (test period)**')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['Date'], df['Consumption'], label='Observed')
    ax.plot(test['Date'], preds, label='Forecast (RF)', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Water Consumption (units)')
    ax.legend()
    st.pyplot(fig)
    st.markdown('**Explanation:** The chart above shows observed daily water consumption for the whole dataset and the model forecast on the hold-out (test) period. The model uses lag features and day-of-year to capture seasonality and trends. Root Mean Squared Error (RMSE) on test set: **{:.2f}**'.format(rmse))

with col2:
    st.markdown('**Model diagnostics & importance**')
    importances = model.feature_importances_
    feat_names = X_train.columns.tolist()
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
    st.dataframe(imp_df)

    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.bar(imp_df['feature'], imp_df['importance'])
    ax2.set_xticklabels(imp_df['feature'], rotation=45, ha='right')
    ax2.set_ylabel('Importance')
    st.pyplot(fig2)
    st.markdown('**Explanation:** Feature importances indicate which lag-days and engineered features the Random Forest relied on most. High importance of recent lags implies short-term persistence in consumption patterns.')

# --- Forecast horizon rolling prediction ---
st.subheader('Multi-day forecast (rolling)')
last_row = data.iloc[-1:].copy()
future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, forecast_horizon+1)]
future_preds = []
current_row = last_row.copy()

for day in range(forecast_horizon):
    X_curr = current_row.drop(['Date', 'Consumption'], axis=1)
    X_curr_scaled = scaler.transform(X_curr)
    pred = model.predict(X_curr_scaled)[0]
    future_preds.append(pred)
    # shift lags
    new_row = current_row.copy()
    for lag in range(lags,1,-1):
        new_row[f'lag_{lag}'] = new_row[f'lag_{lag-1}']
    new_row['lag_1'] = pred
    new_row['Date'] = df['Date'].max() + timedelta(days=day+1)
    current_row = new_row

future_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_preds})
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(df['Date'].tail(365), df['Consumption'].tail(365), label='Recent observed')
ax3.plot(future_df['Date'], future_df['Forecast'], label='Forecast (rolling)', linestyle='--')
ax3.set_xlabel('Date')
ax3.set_ylabel('Daily Consumption')
ax3.legend()
st.pyplot(fig3)
st.markdown('**Explanation:** Rolling multi-day forecasts simulate how the model would predict day-by-day when previous predictions are used as inputs for future days. This helps planners estimate near-term demand for resource allocation.')

# --- Anomaly detection (simple) ---
if detect_anomalies:
    st.subheader('Anomaly detection (statistical method)')
    resid = y_test.values - preds
    mu = resid.mean()
    sigma = resid.std()
    anomaly_idx = np.where(np.abs(resid - mu) > 2*sigma)[0]
    anomalies = test.iloc[anomaly_idx]
    fig4, ax4 = plt.subplots(figsize=(10,3))
    ax4.plot(test['Date'], y_test.values, label='Observed (test)')
    ax4.plot(test['Date'], preds, label='Predicted')
    if len(anomalies)>0:
        ax4.scatter(anomalies['Date'], anomalies['Consumption'], marker='x', s=60)
    ax4.set_xlabel('Date')
    ax4.legend()
    st.pyplot(fig4)
    st.markdown('**Explanation:** Points marked with an "x" are anomalies where observed consumption deviated significantly from model prediction (beyond 2 standard deviations). These may indicate leaks, data issues, or exceptional events requiring investigation.')
    if len(anomalies)==0:
        st.write('No anomalies detected in the test period.')

# --- Sustainability metrics ---
st.subheader('Sustainability metrics & KPIs')
total_consumption = df['Consumption'].sum()
avg_daily = df['Consumption'].mean()
peak_daily = df['Consumption'].max()
water_saved_estimate = max(0, 0.05 * total_consumption)  # hypothetical 5% saving scenario

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric('Total consumption (historic)', f'{total_consumption:,.0f} units')
kpi2.metric('Average daily', f'{avg_daily:.1f} units')
kpi3.metric('Peak daily', f'{peak_daily:.0f} units')
kpi4.metric('Estimated saved (5%)', f'{water_saved_estimate:,.0f} units')

st.markdown('**Explanation:** These KPIs give a quick snapshot of historical demand and conservative estimates of potential savings from efficiency programs (here assumed 5%). Planners can combine forecasts with policy levers to set targets.')

# --- Simple policy simulator ---
st.subheader('Policy simulator: Demand reduction scenarios')
reduction = st.slider('Apply % reduction across demand (simulated conservation)', 0, 50, 10)
sim_forecast = future_df.copy()
sim_forecast['Forecast_reduced'] = sim_forecast['Forecast'] * (1 - reduction/100)
fig5, ax5 = plt.subplots(figsize=(10,4))
ax5.plot(future_df['Date'], future_df['Forecast'], label='Baseline forecast')
ax5.plot(sim_forecast['Date'], sim_forecast['Forecast_reduced'], label=f'Forecast with {reduction}% reduction', linestyle='--')
ax5.set_xlabel('Date')
ax5.set_ylabel('Forecast consumption')
ax5.legend()
st.pyplot(fig5)
st.markdown('**Explanation:** This scenario tool quickly estimates how different conservation measures (e.g., tariffs, incentives, leak repairs) translate into reduced demand over the forecast horizon. Use it to compare interventions.')

# --- Export results ---
st.subheader('Export')
csv_buf = io.StringIO()
future_df.to_csv(csv_buf, index=False)
st.download_button('Download forecast CSV', data=csv_buf.getvalue(), file_name='forecast.csv', mime='text/csv')

if download_model:
    st.info('Model export is a placeholder in this demo. In a real deployment we would serialize the trained model and provide it for download.')
