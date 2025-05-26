#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib

# Load data
df = pd.read_csv('data/processed/retail_sales_RSXFS_processed.csv',
                 parse_dates=['DATE'],
                 index_col='DATE')


# Load models
sarima_model = joblib.load('models/sarima_model_final.pkl')
hw_model = joblib.load('models/holtwinters_model.pkl')  

# Sidebar Controls
st.sidebar.title("🛍️ Retail Sales Forecast")
forecast_horizon = st.sidebar.slider("Select forecast horizon (months)", 1, 24, 12)

if st.sidebar.button("Generate Forecast"):
    # Generate SARIMA forecast
    sarima_forecast = sarima_model.get_forecast(steps=forecast_horizon)
    sarima_pred = sarima_forecast.predicted_mean
    pred_index = pd.date_range(start=df.index[-1], periods=forecast_horizon + 1, freq='MS')[1:]
    sarima_df = pd.DataFrame({'Date': pred_index, 'SARIMA': sarima_pred})

    # Generate Holt-Winters forecast
    hw_forecast = hw_model.forecast(steps=forecast_horizon)
    hw_df = pd.DataFrame({'Date': pred_index, 'HoltWinters': hw_forecast})

    # Generate Ensemble forecast
    ensemble_pred = (
        sarima_pred * 0.7 +
        hw_forecast * 0.3
    )
    ensemble_df = pd.DataFrame({'Date': pred_index, 'Ensemble': ensemble_pred})

    # Combine
    forecast_combined = sarima_df.merge(hw_df, on='Date').merge(ensemble_df, on='Date')
    forecast_combined.set_index('Date', inplace=True)

    # Show forecast table
    st.subheader(f"🔮 {forecast_horizon}-Month Forecast")
    st.write(forecast_combined)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Retail_Sales_Real'][-24:], label='Historical')
    ax.plot(forecast_combined['SARIMA'], label='SARIMA Forecast', linestyle='--')
    ax.plot(forecast_combined['HoltWinters'], label='Holt-Winters Forecast', linestyle=':')
    ax.plot(forecast_combined['Ensemble'], label='Ensemble Forecast', linewidth=2, color='black')
    ax.set_title("Retail Sales Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales (Millions USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Download button
    st.download_button(
        "Download Forecast",
        data=forecast_combined.to_csv(index=True),
        file_name="retail_sales_forecast.csv"
    )