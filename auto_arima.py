import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# === Load preprocessed weekly data ===
weekly_data = pd.read_csv('weekly_sales_walmart.csv', parse_dates=['week'])
weekly_data.set_index('week', inplace=True)

# === Define target and exogenous variables ===
target = 'sales_usd'
exog_cols = [col for col in weekly_data.columns if col not in ['sales_usd', 'units_sold']]
X = weekly_data[exog_cols]
y = weekly_data[target]

# === Split data ===
train_y, test_y = y[:-52], y[-52:]
train_X, test_X = X.iloc[:-52], X.iloc[-52:]

# === Auto ARIMA model selection ===
model = auto_arima(
    train_y,
    exogenous=train_X,
    seasonal=True,
    m=52,  # weekly data with yearly seasonality
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True
)

# === Forecasting ===
forecast = model.predict(n_periods=52, exogenous=test_X)

# === Evaluate ===
rmse = np.sqrt(mean_squared_error(test_y, forecast))
print(f'Auto ARIMA RMSE: {rmse:.2f}')

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(train_y[-52:], label='Train (last year)')
plt.plot(test_y, label='Actual')
plt.plot(test_y.index, forecast, label='Forecast', linestyle='--')
plt.title(f'Auto ARIMA Forecast — RMSE: {rmse:.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# Optional: print model summary
print(model.summary())
