📊 Time Series Forecasting Project
📘 Objective: Forecast Advance Real Retail Sales (RSXFS)
This project develops a robust forecasting model for the Advance Real Retail Sales (RSXFS) — a key U.S. economic indicator — using time series modeling techniques, and presents findings through an interactive Streamlit dashboard .

🎯 Economic Indicator Chosen
Indicator Name: Advance Real Retail Sales (RSXFS)
Description: Monthly inflation-adjusted retail sales, providing early insight into consumer spending behavior.
Source: FRED (Federal Reserve Economic Data)
Frequency: Monthly
Time Range: 2001–2024 (or your dataset's range)
Relevance: Reflects short-term economic health and business performance

🧹 Key Steps in the Pipeline
1.Data Acquisition
  Fetched RSXFS from FRED
  Added external regressors like:
    a)Consumer Confidence Index
    b)Unemployment Rate
2.Data Preprocessing
  Cleaned missing values and outliers
  Resampled to monthly frequency (MS)
  Created lag features and rolling statistics
  Added holiday flags and event-based regressors
3.Exploratory Data Analysis (EDA)
  Decomposed trend, seasonality, and residuals
  Visualized historical patterns and anomalies
  Checked stationarity using ADF test
  Analyzed ACF/PACF for SARIMA parameter selection
4.Modeling
  Trained several forecasting models:
  Naïve & Seasonal Naïve
  Holt–Winters Triple Exponential Smoothing
  SARIMA / ARIMA
  Prophet (with regressors)
  LSTM Neural Networks
  Ensemble (SARIMA + Holt–Winters)
5.Model Evaluation
  Evaluated using:
  MAE (Mean Absolute Error)
  RMSE (Root Mean Squared Error)
  MAPE (Mean Absolute Percentage Error)
  Ensemble Forecasting
  Combined forecasts using weighted averaging
  Improved accuracy and stability over single models
6.Streamlit Dashboard Development
  Built interactive plots of actual vs forecasted values
  Included user inputs for forecast horizon
  Enabled CSV download of forecasts
  Visualized confidence bounds where applicable
📊 Model Comparison
MODEL
MAPE (%)
Holt–Winters (Multiplicative)
3.85% ← best performer
Ensemble (Weighted)
4.26%
SARIMA
4.53%

✅ The Holt–Winters model outperformed all others due to strong annual seasonality in the data.
