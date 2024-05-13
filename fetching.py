import h2o
import numpy as np
import pandas as pd
import requests
from h2o.automl import H2OAutoML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

# API Key and Endpoint Setup
api_key = "hBKygPpuWeevL0ZxOl5ah8hsaZrCj0YV"  # Replace with your actual FMP API key
symbol = "TSLA"  # Example stock symbol
start_date = "2023-01-01"  # Adjust start date as needed
end_date = "2024-05-01"  # Latest complete data as of your request

# Construct API URL
url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"

# Fetching the data
response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data["historical"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Normalize the 'close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
df["normalized_close"] = scaler.fit_transform(df[["close"]])

# Additional feature engineering
df["return"] = df["close"].pct_change()
df["moving_average_5"] = df["normalized_close"].rolling(window=5).mean()
df["moving_average_10"] = df["normalized_close"].rolling(window=10).mean()

# Drop NaN values after feature creation
df.dropna(inplace=True)

features = df[["return", "moving_average_5", "moving_average_10"]]
target = df["normalized_close"]

# Correct way to split time series data
split_point = int(len(df) * 0.8)
X_train, X_test = features.iloc[:split_point], features.iloc[split_point:]
y_train, y_test = target.iloc[:split_point], target.iloc[split_point:]

# H2O
h2o.init()

# Prepare data for H2O
X_train_h2o = X_train.copy()
X_train_h2o["target"] = y_train  # Rename target column to avoid overlap

hf_train = h2o.H2OFrame(X_train_h2o)
hf_test = h2o.H2OFrame(X_test)

# H2O AutoML
automl = H2OAutoML(max_models=10, seed=1, max_runtime_secs=600)
automl.train(y="target", training_frame=hf_train)

# ARIMA
model = ARIMA(y_train, order=(5, 1, 0))
model_fit = model.fit()

# Linear Regression and Random Forest
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Calculate the last known moving averages and return
last_ma_5 = df["moving_average_5"].iloc[-1]
last_ma_10 = df["moving_average_10"].iloc[-1]
last_return = df["return"].iloc[-1]

# Create a DataFrame for future dates
future_dates = pd.date_range(start="2024-05-01", periods=5, freq="D")
future_data = {
    "return": [last_return] * 5,  # Assuming return remains constant; adjust as needed
    "moving_average_5": np.linspace(
        last_ma_5, last_ma_5 * 1.02, 5
    ),  # Small increase assumed
    "moving_average_10": np.linspace(
        last_ma_10, last_ma_10 * 1.02, 5
    ),  # Small increase assumed
}
future_df = pd.DataFrame(future_data, index=future_dates)
future_df.index.name = "date"

# Convert future_df for H2O prediction
hf_future = h2o.H2OFrame(future_df)

# Predict with H2O AutoML
future_automl_pred = automl.predict(hf_future).as_data_frame().values.flatten()

# ARIMA model forecast
forecast_result_ARIMA = model_fit.get_forecast(steps=5)
future_arima_pred = forecast_result_ARIMA.predicted_mean

# Predict with Linear Regression and Random Forest
future_lr_pred = lr.predict(future_df)
future_rf_pred = rf.predict(future_df)

# Combine these predictions to create an ensemble prediction for the future
ensemble_future_predictions = np.mean(
    np.column_stack(
        (future_automl_pred, future_arima_pred, future_lr_pred, future_rf_pred)
    ),
    axis=1,
)

# Add predictions to the future_df for review or visualization
future_df["predicted_normalized_close"] = ensemble_future_predictions

# Assuming 'date' columns exist and are not set as index yet, and are in a standard date format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Reverting historical normalized close values
df["original_close"] = scaler.inverse_transform(df[["normalized_close"]])

# Reverting forecasted normalized close values
future_df["original_close"] = scaler.inverse_transform(
    future_df[["predicted_normalized_close"]]
)

import mplfinance as mpf

# Historical data
df["Open"] = df["original_close"] * 0.99
df["High"] = df["original_close"] * 1.01
df["Low"] = df["original_close"] * 0.98
df["Close"] = df["original_close"]

# Future data
future_df["Open"] = future_df["original_close"] * 0.99
future_df["High"] = future_df["original_close"] * 1.01
future_df["Low"] = future_df["original_close"] * 0.98
future_df["Close"] = future_df["original_close"]

# Ensure the indices are datetime objects and set them as index
df.index = pd.to_datetime(df.index)
future_df.index = pd.to_datetime(future_df.index)

# Combine historical and future data
full_df = pd.concat([df, future_df])

# Prepare the data for mplfinance
full_df.index.name = "Date"  # mplfinance expects the index name to be 'Date'

# Define a style
mc = mpf.make_marketcolors(up="green", down="red", inherit=True)
s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc)

# Create a plot
mpf.plot(
    full_df,
    type="candle",
    style=s,
    title="Historical and Forecasted Price Movement",
    ylabel="Price ($)",
    volume=False,
    figsize=(12, 6),
    show_nontrading=True,
)

# If needed, highlight the forecast period
apd = mpf.make_addplot(full_df["Close"], type="line", color="orange")
mpf.plot(
    full_df,
    type="candle",
    style=s,
    addplot=apd,
    title="Historical and Forecasted Price Movement",
    ylabel="Price ($)",
    volume=False,
    figsize=(12, 6),
    show_nontrading=True,
    hlines=dict(hlines=[full_df["Close"].iloc[-6]], colors=["blue"], linestyle="-."),
)
