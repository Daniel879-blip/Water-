# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Prophet optional (may not be installed in every environment)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title='AI Water Sustainability Dashboard', layout='wide')
st.title('AI for Sustainable Water Management — Urban Areas')

# -------------------------
# SIDEBAR - all features
# -------------------------
st.sidebar.header('Data source & controls')

use_sample = st.sidebar.checkbox('Use built-in sample dataset', value=True)
uploaded_file = st.sidebar.file_uploader('Upload CSV (should contain Date, Consumption, optional Area)', type=['csv'])

# Urban areas control - for sample we'll generate several areas; for uploaded dataset we allow mapping
default_areas = ["Central", "North", "South", "East", "West", "Industrial", "Suburb"]
selected_areas = st.sidebar.multiselect('Select urban areas (for analysis)', options=default_areas, default=default_areas)

# Date range - placeholders; will be adjusted after loading data
st.sidebar.markdown("### Model & Forecast")
forecast_horizon = st.sidebar.slider('Forecast horizon (days)', 1, 365, 30)
train_pct = st.sidebar.slider('Train split (%)', 50, 95, 80)

model_options = ['Random Forest', 'Naive Persistence']
if PROPHET_AVAILABLE:
    model_options.insert(1, 'Prophet')
model_choice = st.sidebar.selectbox('Model choice', model_options)

n_estimators = st.sidebar.slider('RF n_estimators', 10, 500, 100)
max_depth = st.sidebar.slider('RF max_depth (0 means None)', 0, 50, 10)

st.sidebar.markdown("### Anomaly detection")
anomaly_method = st.sidebar.selectbox('Method', ['Statistical residuals', 'Isolation Forest'])
anomaly_sigma = st.sidebar.slider('Statistical threshold (σ)', 1, 5, 2)

st.sidebar.markdown("### Aggregation & features")
agg_freq = st.sidebar.selectbox('Aggregation frequency', ['Daily', 'Weekly', 'Monthly'])
aggregate_fn = st.sidebar.selectbox('Aggregate function for multiple areas', ['Sum', 'Mean'])

st.sidebar.markdown("### Policy & exports")
reduction_default = st.sidebar.slider('Simulated conservation reduction (%)', 0, 50, 10)
show_feature_importances = st.sidebar.checkbox('Show feature importances', value=True)
show_ai_suggestions = st.sidebar.checkbox('Show AI suggestions', value=True)
download_model = st.sidebar.button('Export trained model (pickle)')

st.sidebar.markdown("### Advanced / smart")
smart_mode = st.sidebar.checkbox('Enable smart preprocessing & suggestions', value=True)
ad_hoc_refresh = st.sidebar.button('Refresh analysis (recompute)')

# -------------------------
# DATA LOADING
# -------------------------
def generate_sample_data(years=5, areas=default_areas, start_date='2020-01-01'):
    # Create multi-area realistic consumption time series
    start = pd.to_datetime(start_date)
    days = years * 365
    dates = pd.date_range(start=start, periods=days, freq='D')
    rows = []
    rng = np.random.RandomState(42)
    for area in areas:
        base_scale = rng.uniform(0.6, 1.6)  # different base per area
        seasonal_amp = rng.uniform(0.10, 0.35)
        trend = rng.uniform(0.0, 0.05)  # mild upward trend
        for i, dt in enumerate(dates):
            # seasonal weekly + yearly
            day_of_year = dt.timetuple().tm_yday
            seasonal = seasonal_amp * np.sin(2 * np.pi * day_of_year / 365)
            weekly = 0.05 * np.sin(2 * np.pi * dt.weekday() / 7)
            noise = rng.normal(scale=0.05)
            # occasional spikes: leaks/events
            spike = 0.0
            if rng.rand() < 0.01:
                spike = rng.uniform(0.2, 0.6)
            # Compose consumption (normalized then scaled)
            consumption = 1000 * base_scale * (1 + seasonal + weekly + trend * (i/days) + noise + spike)
            rain = max(0, rng.normal(5, 15))
            temp = rng.normal(25, 7)
            population = int(rng.uniform(50000, 500000))
            rows.append([dt, area, round(consumption, 2), round(rain,2), population, round(temp,1)])
    df = pd.DataFrame(rows, columns=['Date','Area','Consumption','Rainfall','Population','Temperature'])
    return df

# Load or generate
if use_sample or uploaded_file is None:
    df = generate_sample_data(years=5, areas=default_areas, start_date='2020-01-01')
    st.sidebar.info('Using built-in realistic sample dataset (5 years, multiple urban areas).')
else:
    # read uploaded CSV and let user map columns
    try:
        raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f'Unable to read uploaded CSV: {e}')
        st.stop()
    st.sidebar.success('CSV uploaded. Map the columns below if needed.')
    cols = raw.columns.tolist()
    date_col = st.sidebar.selectbox('Date column', options=cols, index=0)
    consumption_col = st.sidebar.selectbox('Consumption column', options=cols, index= min(1, len(cols)-1))
    area_col = st.sidebar.selectbox('Area column (optional)', options=[None] + cols, index=0)
    # parse dates
    try:
        raw[date_col] = pd.to_datetime(raw[date_col])
    except Exception:
        st.error('Could not parse the chosen date column. Ensure dates are in a parseable format.')
        st.stop()
    # If no area column, ask for default area assignment
    if area_col is None:
        default_area_name = st.sidebar.text_input('Assign a single Area name to this CSV (if it is single-area data)', value='Uploaded Area')
        raw['Area'] = default_area_name
    else:
        raw.rename(columns={area_col: 'Area'}, inplace=True)
    raw.rename(columns={date_col: 'Date', consumption_col: 'Consumption'}, inplace=True)
    # keep only expected columns, try to preserve Rainfall/Temp/Population if present
    expected_cols = ['Date','Area','Consumption','Rainfall','Population','Temperature']
    df = raw[[c for c in expected_cols if c in raw.columns]]
    # Fill missing optional columns
    if 'Rainfall' not in df.columns:
        df['Rainfall'] = np.nan
    if 'Population' not in df.columns:
        df['Population'] = np.nan
    if 'Temperature' not in df.columns:
        df['Temperature'] = np.nan

# Ensure types and sorting
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# If user changed selected areas via sidebar, use them (fallback to all if selection empty)
available_areas = sorted(df['Area'].unique().tolist())
if len(selected_areas) == 0:
    selected_areas = available_areas
else:
    # ensure selected are valid; if not, intersect
    selected_areas = [a for a in selected_areas if a in available_areas]
    if len(selected_areas) == 0:
        selected_areas = available_areas

# Date range defaults (reflect dataset range)
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
start_date = st.sidebar.date_input('Start date', min_date)
end_date = st.sidebar.date_input('End date', max_date)
if start_date > end_date:
    st.sidebar.error('Start date must be before end date.')
    st.stop()

# Filter dataset by user choices
mask = (
    (df['Area'].isin(selected_areas)) &
    (df['Date'].dt.date >= start_date) &
    (df['Date'].dt.date <= end_date)
)
df_filtered = df.loc[mask].copy()
if df_filtered.empty:
    st.warning('No data in the selected range / areas. Try widening the date range or selecting other areas.')
    st.stop()

# Aggregation
freq_map = {'Daily':'D','Weekly':'W','Monthly':'M'}
resample_freq = freq_map.get(agg_freq, 'D')

if aggregate_fn == 'Sum':
    agg_series = (
    df_filtered
    .set_index('Date')  # Make Date the index
    .resample(resample_freq)['Consumption']  # Resample directly on datetime index
    .sum()
    .reset_index()
)

# Show small preview and KPIs
st.subheader('Dataset preview and KPIs')
colA, colB = st.columns([2,1])
with colA:
    st.dataframe(df_filtered.head(10))
with colB:
    total_consumption = agg_series['Consumption'].sum()
    avg_daily = agg_series['Consumption'].mean()
    peak = agg_series['Consumption'].max()
    est_saved = 0.05 * total_consumption
    st.metric('Total consumption (selected range)', f'{total_consumption:,.0f} units')
    st.metric('Average (resampled)', f'{avg_daily:,.1f} units')
    st.metric('Peak (resampled)', f'{peak:,.0f} units')
    st.metric('Conservative 5% savings', f'{est_saved:,.0f} units')

st.markdown('---')

# -------------------------
# FEATURE ENGINEERING & MODEL PREP
# -------------------------
def create_lag_features_single(series_df, lags=14):
    data = series_df.copy()
    data = data.set_index('Date').asfreq(resample_freq).interpolate()
    for lag in range(1, lags+1):
        data[f'lag_{lag}'] = data['Consumption'].shift(lag)
    data['dayofyear'] = data.index.dayofyear
    data = data.dropna().reset_index()
    return data

lags = 14
series_for_model = agg_series.copy()
data_model = create_lag_features_single(series_for_model, lags=lags)

# Chronological split
split_idx = int(len(data_model) * train_pct / 100)
train = data_model.iloc[:split_idx].copy()
test = data_model.iloc[split_idx:].copy()

X_train = train.drop(['Date','Consumption'], axis=1)
y_train = train['Consumption']
X_test = test.drop(['Date','Consumption'], axis=1)
y_test = test['Consumption']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# MODEL TRAIN + FORECAST
# -------------------------
st.subheader('Forecasting results')

model = None
preds = None
rmse = None
train_msg = ""

if model_choice == 'Random Forest':
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth==0 else max_depth), random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, preds))
    train_msg = f'Random Forest trained. Test RMSE: {rmse:.2f}'
elif model_choice == 'Prophet' and PROPHET_AVAILABLE:
    # Prophet wants columns ds/y and continuous daily data (we will use the aggregated daily series)
    prophet_df = series_for_model.rename(columns={'Date':'ds','Consumption':'y'}).dropna()
    p = Prophet(daily_seasonality=True, yearly_seasonality=True)
    p.fit(prophet_df)
    future = p.make_future_dataframe(periods=forecast_horizon, freq=resample_freq)
    forecast_prophet = p.predict(future)
    # separate historical fit vs forecast for plotting
    preds = None  # we'll handle prophet plot separately
    model = p
    train_msg = 'Prophet model trained.'
else:
    # Naive persistence model (y_t = last observed)
    last_value = train['Consumption'].iloc[-1]
    preds = np.repeat(last_value, len(y_test))
    rmse = mean_squared_error(y_test, preds, squared=False)
    train_msg = f'Naive persistence model used. Test RMSE: {rmse:.2f}'

st.write(train_msg)

# Plot test-period observed vs predicted (if we have preds)
col1, col2 = st.columns([2,1])
with col1:
    st.markdown('**Observed vs Forecast (test period)**')
    fig, ax = plt.subplots(figsize=(10,4))
    # plot full historical resampled series
    ax.plot(series_for_model['Date'], series_for_model['Consumption'], label='Observed (historical)', alpha=0.6)
    if model_choice == 'Prophet' and PROPHET_AVAILABLE:
        # show prophet predictions
        ax.plot(forecast_prophet['ds'], forecast_prophet['yhat'], linestyle='--', label='Prophet forecast')
    else:
        # align preds with test['Date']
        ax.plot(test['Date'], preds, linestyle='--', label='Model forecast (test period)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Consumption')
    ax.legend()
    st.pyplot(fig)
    st.markdown('**Explanation:** Observed resampled consumption (solid) and model predictions (dashed). Use RMSE and residuals to evaluate fit.')
with col2:
    st.markdown('**Model diagnostics & feature importances**')
    if hasattr(model, 'feature_importances_') and show_feature_importances:
        importances = model.feature_importances_
        feat_names = X_train.columns.tolist()
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
        st.dataframe(imp_df)
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.barh(imp_df['feature'], imp_df['importance'])
        ax2.set_xlabel('Importance')
        ax2.invert_yaxis()
        st.pyplot(fig2)
        st.markdown('**Explanation:** Feature importances show which lag-days and day-of-year influence predictions most.')
    else:
        st.write('Feature importances not available for the chosen model or disabled.')

# -------------------------
# Multi-day rolling forecast (for RandomForest / Naive)
# -------------------------
st.subheader('Multi-day rolling forecast')
if model_choice == 'Prophet' and PROPHET_AVAILABLE:
    # Show Prophet future forecast already computed
    forecast_to_show = forecast_prophet[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_horizon)
    figf = px.line(forecast_prophet, x='ds', y='yhat', title=f'Prophet forecast (next {forecast_horizon} {resample_freq} steps)')
    figf.add_scatter(x=series_for_model['Date'], y=series_for_model['Consumption'], mode='lines', name='Actual')
    st.plotly_chart(figf, use_container_width=True)
    st.markdown('**Explanation:** Prophet produces probabilistic forecasts (wide bands may indicate uncertainty).')
else:
    # Rolling predict using last available row
    last_row = data_model.iloc[-1:].copy()
    current_row = last_row.copy()
    future_dates = []
    future_preds = []
    # prepare column order
    cols_X = X_train.columns.tolist()
    for day in range(forecast_horizon):
        # build X_curr
        X_curr = current_row.drop(['Date','Consumption'], axis=1)[cols_X]
        X_curr_scaled = scaler.transform(X_curr)
        p_val = model.predict(X_curr_scaled)[0]
        future_preds.append(p_val)
        # shift lags forward
        new_row = current_row.copy()
        for lag in range(lags, 1, -1):
            new_row[f'lag_{lag}'] = new_row[f'lag_{lag-1}']
        new_row['lag_1'] = p_val
        new_row['Date'] = series_for_model['Date'].max() + pd.to_timedelta(day+1, unit=resample_freq.lower()[0])
        current_row = new_row
        future_dates.append(new_row['Date'].iloc[0])
    future_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_preds})
    figf, axf = plt.subplots(figsize=(10,4))
    axf.plot(series_for_model['Date'].tail(365), series_for_model['Consumption'].tail(365), label='Recent observed')
    axf.plot(future_df['Date'], future_df['Forecast'], linestyle='--', label='Rolling forecast')
    axf.set_xlabel('Date')
    axf.set_ylabel('Consumption')
    axf.legend()
    st.pyplot(figf)
    st.markdown('**Explanation:** Rolling multi-step forecast where model outputs feed into future inputs (typical for non-sequence models).')

# -------------------------
# Anomaly detection
# -------------------------
st.subheader('Anomaly detection')
anomaly_results = pd.DataFrame()
if anomaly_method == 'Statistical residuals':
    if preds is not None:
        resid = y_test.values - preds
        mu = resid.mean()
        sigma = resid.std()
        anomaly_idx = np.where(np.abs(resid - mu) > anomaly_sigma * sigma)[0]
        anomalies = test.iloc[anomaly_idx][['Date','Consumption']].copy()
        anomalies['residual'] = resid[anomaly_idx]
        st.write(f'Found {len(anomalies)} anomalies (statistical threshold = {anomaly_sigma}σ).')
        # Plot
        fig4, ax4 = plt.subplots(figsize=(10,3))
        ax4.plot(test['Date'], y_test.values, label='Observed (test)')
        ax4.plot(test['Date'], preds, label='Predicted')
        if len(anomalies) > 0:
            ax4.scatter(anomalies['Date'], anomalies['Consumption'], marker='x', s=80, color='red', label='Anomalies')
        ax4.set_xlabel('Date')
        ax4.legend()
        st.pyplot(fig4)
        st.markdown('**Explanation:** Points where observed consumption deviates strongly from predicted — these could indicate leaks, meter errors, or special events.')
        anomaly_results = anomalies
    else:
        st.write('No residuals available for statistical anomaly detection (model may be Prophet).')
elif anomaly_method == 'Isolation Forest':
    # Use lag features on the full prepared data to detect anomalies
    iso = IsolationForest(random_state=42, contamination=0.01)
    X_full = scaler.transform(np.vstack([X_train, X_test]))
    iso.fit(X_full)
    preds_iso = iso.predict(X_full)  # -1 anomaly
    idx = np.where(preds_iso == -1)[0]
    combined = pd.concat([train, test], ignore_index=True)
    anam = combined.iloc[idx][['Date','Consumption']].copy()
    st.write(f'IsolationForest found {len(anam)} anomaly points.')
    fig_iso, ax_iso = plt.subplots(figsize=(10,3))
    ax_iso.plot(combined['Date'], combined['Consumption'], label='Observed (train+test)')
    if len(anam) > 0:
        ax_iso.scatter(anam['Date'], anam['Consumption'], marker='x', s=80, color='red', label='Anomalies')
    ax_iso.set_xlabel('Date')
    ax_iso.legend()
    st.pyplot(fig_iso)
    st.markdown('**Explanation:** IsolationForest learns normal patterns from lag features and flags unusual points.')

# -------------------------
# Policy simulator & scenario comparison
# -------------------------
st.subheader('Policy simulator: demand reduction scenarios')
reduction = st.slider('Apply % reduction across forecast (simulated conservation)', 0, 50, reduction_default)
if model_choice == 'Prophet' and PROPHET_AVAILABLE:
    # apply reduction to the prophet forecast portion
    sim = forecast_prophet[['ds','yhat']].tail(forecast_horizon).copy()
    sim['reduced'] = sim['yhat'] * (1 - reduction/100)
    fig_sim = px.line(sim, x='ds', y=['yhat','reduced'], labels={'value':'Consumption','ds':'Date'}, title='Policy scenario: reduction applied')
    st.plotly_chart(fig_sim, use_container_width=True)
else:
    if 'future_df' in locals():
        sim_df = future_df.copy()
        sim_df['Forecast_reduced'] = sim_df['Forecast'] * (1 - reduction/100)
        fig_sim, ax_sim = plt.subplots(figsize=(10,4))
        ax_sim.plot(future_df['Date'], future_df['Forecast'], label='Baseline forecast')
        ax_sim.plot(sim_df['Date'], sim_df['Forecast_reduced'], linestyle='--', label=f'{reduction}% reduction')
        ax_sim.set_xlabel('Date')
        ax_sim.set_ylabel('Consumption')
        ax_sim.legend()
        st.pyplot(fig_sim)
        st.markdown('**Explanation:** Quick scenario tool to compare baseline forecasts vs conservation measures.')

# -------------------------
# Smart suggestions (heuristics)
# -------------------------
if show_ai_suggestions:
    st.subheader('AI suggestions & smart insights')
    suggestions = []
    # simple heuristic: high volatility -> recommend sensor densification
    volatility = agg_series['Consumption'].pct_change().std()
    if volatility > 0.05:
        suggestions.append('High consumption volatility detected — consider adding sensors and short-term monitoring to detect leaks quickly.')
    else:
        suggestions.append('Consumption is relatively stable — focus on targeted demand management.')
    # anomaly-driven suggestion
    if not anomaly_results.empty:
        suggestions.append(f'{len(anomaly_results)} anomalies detected — schedule immediate inspections in the selected areas/dates.')
    # feature-driven recommendations
    if show_feature_importances and hasattr(model, 'feature_importances_'):
        top_feat = X_train.columns[np.argsort(model.feature_importances_)[-3:]].tolist()
        suggestions.append(f'Important features: {top_feat}. Consider monitoring the most recent days and day-of-year effects.')
    for s in suggestions:
        st.write('- ' + s)

# -------------------------
# Export: forecast CSV and model
# -------------------------
st.subheader('Export results')
if model_choice == 'Prophet' and PROPHET_AVAILABLE:
    out_csv = forecast_prophet[['ds','yhat']].tail(forecast_horizon).rename(columns={'ds':'Date','yhat':'Forecast'})
    buf = io.StringIO()
    out_csv.to_csv(buf, index=False)
    st.download_button('Download Prophet forecast CSV', data=buf.getvalue(), file_name='prophet_forecast.csv', mime='text/csv')
else:
    if 'future_df' in locals():
        buf = io.StringIO()
        future_df.to_csv(buf, index=False)
        st.download_button('Download forecast CSV', data=buf.getvalue(), file_name='forecast.csv', mime='text/csv')

if download_model:
    if model is None:
        st.warning('No serializable model available to export.')
    else:
        try:
            model_bytes = pickle.dumps(model)
            st.download_button('Download model (.pkl)', data=model_bytes, file_name='trained_model.pkl', mime='application/octet-stream')
        except Exception as e:
            st.error(f'Failed to pickle model for download: {e}')

st.markdown('---')
st.markdown('**Notes:**\n- If you plan to deploy to Streamlit Cloud / GitHub, ensure `seaborn` and (optionally) `prophet` are in your `requirements.txt` so deployment installs them automatically.')
