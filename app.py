#import  libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Load your data at the top-level, before any button-guards.
@st.cache_data
def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=['DATE'],
        index_col='DATE'
    )
    return df

df = load_data('data/processed/retail_sales_RSXFS_processed.csv')

# 2. Load your pre-trained models at the top-level, too.
sarima_model = joblib.load('models/sarima_model_final.pkl')
hw_model     = joblib.load('models/holtwinters_model.pkl')  

# 3. Sidebar controls
st.sidebar.title("🛍️ Retail Sales Forecast")
forecast_horizon = st.sidebar.slider(
    "Select forecast horizon (months)", 
    min_value=1, max_value=24, value=12
)

# 4. Only once the user clicks do we reference df …
if st.sidebar.button("Generate Forecast"):
    # … SARIMA …
    sarima_forecast = sarima_model.get_forecast(steps=forecast_horizon)
    sarima_pred     = sarima_forecast.predicted_mean
    pred_index = pd.date_range(
        start = df.index[-1],                  
        periods = forecast_horizon + 1,         
        freq = 'MS'                             
    )[1:]                                       
    sarima_df = pd.DataFrame({
        'Date': pred_index, 
        'SARIMA': sarima_pred
    })

    # … Holt-Winters …
    hw_forecast = hw_model.forecast(steps=forecast_horizon)
    hw_df = pd.DataFrame({
        'Date': pred_index, 
        'HoltWinters': hw_forecast
    })

    # … Ensemble …
    ensemble_pred = sarima_pred * 0.7 + hw_forecast * 0.3
    ensemble_df = pd.DataFrame({
        'Date': pred_index, 
        'Ensemble': ensemble_pred
    })

    # 5. Combine, display, and download
    forecast_combined = (
        sarima_df
        .merge(hw_df, on='Date')
        .merge(ensemble_df, on='Date')
        .set_index('Date')
    )

    st.subheader(f"🔮 {forecast_horizon}-Month Forecast")
    st.write(forecast_combined)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Retail_Sales_Real'][-24:], label='Historical')
    ax.plot(forecast_combined['SARIMA'],           linestyle='--', label='SARIMA')
    ax.plot(forecast_combined['HoltWinters'],      linestyle=':',  label='Holt-Winters')
    ax.plot(forecast_combined['Ensemble'], linewidth=2,   label='Ensemble')
    ax.set_title("Retail Sales Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales (Millions USD)")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

    st.download_button(
        "Download Forecast",
        data=forecast_combined.to_csv(),
        file_name="retail_sales_forecast.csv"
    )
