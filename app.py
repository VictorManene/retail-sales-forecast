import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Page configuration
st.set_page_config(page_title='Retail Sales Forecast', layout='wide')

# 1) Load data function with caching
@st.cache_data
def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=['DATE'],
        index_col='DATE'
    )
    return df

# 2) Load pre-trained models with caching
@st.cache_resource
def load_models(sarima_path, hw_path):
    sarima = joblib.load(sarima_path)
    hw = joblib.load(hw_path)
    return sarima, hw

# Load dataset and models at top level
DATA_PATH = 'data/processed/retail_sales_RSXFS_processed.csv'
SARIMA_MODEL_PATH = 'models/sarima_model_final.pkl'
HW_MODEL_PATH = 'models/holtwinters_model.pkl'

df = load_data(DATA_PATH)
sarima_model, hw_model = load_models(SARIMA_MODEL_PATH, HW_MODEL_PATH)

# Sidebar controls
st.sidebar.title('🛍️ Retail Sales Forecast')
forecast_horizon = st.sidebar.slider(
    'Forecast horizon (months)',
    min_value=1, max_value=24, value=12
)

# Button to trigger forecasting
if st.sidebar.button('Generate Forecast'):
    # Generate SARIMA forecast
    sarima_forecast = sarima_model.get_forecast(steps=forecast_horizon)
    sarima_pred = sarima_forecast.predicted_mean
    pred_index = pd.date_range(
        start=df.index[-1],
        periods=forecast_horizon + 1,
        freq='MS'
    )[1:]
    sarima_df = pd.DataFrame({'Date': pred_index, 'SARIMA': sarima_pred})

    # Generate Holt-Winters forecast
    hw_forecast = hw_model.forecast(steps=forecast_horizon)
    hw_df = pd.DataFrame({'Date': pred_index, 'HoltWinters': hw_forecast})

    # Ensemble forecast
    ensemble_pred = sarima_pred * 0.7 + hw_forecast * 0.3
    ensemble_df = pd.DataFrame({'Date': pred_index, 'Ensemble': ensemble_pred})

    # Combine forecasts
    forecast_combined = (
        sarima_df
        .merge(hw_df, on='Date')
        .merge(ensemble_df, on='Date')
        .set_index('Date')
    )

    # Display
    st.subheader(f'🔮 {forecast_horizon}-Month Forecast')
    st.dataframe(forecast_combined)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Retail_Sales_Real'][-24:], label='Historical')
    ax.plot(forecast_combined['SARIMA'], '--', label='SARIMA')
    ax.plot(forecast_combined['HoltWinters'], ':', label='Holt-Winters')
    ax.plot(forecast_combined['Ensemble'], linewidth=2, label='Ensemble')
    ax.set_title('Retail Sales Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales (Real USD Millions)')
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

    # Download button
    csv = forecast_combined.to_csv(index=True)
    st.download_button(
        '📥 Download Forecast',
        data=csv,
        file_name='retail_sales_forecast.csv',
        mime='text/csv'
    )
else:
    st.markdown('Select a forecast horizon and click **Generate Forecast** on the sidebar.')