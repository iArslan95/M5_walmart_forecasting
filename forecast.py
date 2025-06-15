import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the weekly sales data
print("Loading data...")
data = pd.read_csv('weekly_sales_walmart.csv')
data['week'] = pd.to_datetime(data['week'])
data.set_index('week', inplace=True)

# Identify columns that represent holidays (binary exogenous variables)
sales_columns = ['sales_usd', 'units_sold']
exog_columns = [col for col in data.columns if col not in sales_columns]

print(f"Found {len(exog_columns)} exogenous variables (holidays): {exog_columns}")

# Split data: keep last 52 weeks (1 year) for testing
train_data = data.iloc[:-52]
test_data = data.iloc[-52:]

print(f"Training data: {len(train_data)} weeks from {train_data.index.min()} to {train_data.index.max()}")
print(f"Test data: {len(test_data)} weeks from {test_data.index.min()} to {test_data.index.max()}")


# Function to fit SARIMAX model
def fit_sarimax(train_data, order, seasonal_order, exog_vars=None, target='sales_usd'):
    if exog_vars is not None:
        exog = train_data[exog_vars]
        model = SARIMAX(train_data[target], exog=exog, order=order, seasonal_order=seasonal_order)
    else:
        model = SARIMAX(train_data[target], order=order, seasonal_order=seasonal_order)

    results = model.fit(disp=False)
    return results


# Function to evaluate the model
def evaluate_model(model, test_data, exog_vars=None, target='sales_usd'):
    if exog_vars is not None:
        exog_test = test_data[exog_vars]
        pred_results = model.get_forecast(steps=len(test_data), exog=exog_test)
    else:
        pred_results = model.get_forecast(steps=len(test_data))

    forecasts = pred_results.predicted_mean
    conf_int = pred_results.conf_int(alpha=0.05)  # 95% confidence interval

    mse = mean_squared_error(test_data[target], forecasts)
    mape = mean_absolute_percentage_error(test_data[target], forecasts) * 100

    return forecasts, conf_int, mse, mape


# Define logical parameter combinations for SARIMAX
# For Walmart sales, considering weekly seasonality (s=52)
# Based on common patterns for retail data
parameter_combinations = [
    # (p,d,q)(P,D,Q,s)
    ((1, 1, 1), (1, 0, 1, 52)),  # Simple model with seasonality
    ((2, 1, 1), (1, 0, 1, 52)),  # Increasing AR component
    ((1, 1, 2), (1, 0, 1, 52)),  # Increasing MA component
    ((2, 1, 2), (1, 0, 1, 52)),  # Balanced higher-order model
    ((1, 1, 1), (2, 0, 1, 52)),  # Increasing seasonal AR
]

# Target column to forecast
target_col = 'sales_usd'
results = []

print("\nTraining and evaluating SARIMAX models...")
for i, (order, seasonal_order) in enumerate(parameter_combinations):
    model_name = f"SARIMAX{order}x{seasonal_order}"
    print(f"\nTraining model {i + 1}/{len(parameter_combinations)}: {model_name}")

    try:
        # Fit the model
        model = fit_sarimax(train_data, order, seasonal_order, exog_vars=exog_columns, target=target_col)

        # Evaluate on test data
        forecasts, conf_int, mse, mape = evaluate_model(
            model, test_data, exog_vars=exog_columns, target=target_col
        )

        print(f"AIC: {model.aic:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Store results
        results.append({
            'Model': model_name,
            'AIC': model.aic,
            'MSE': mse,
            'MAPE': mape,
            'Order': order,
            'Seasonal_Order': seasonal_order,
            'Forecasts': forecasts,
            'Conf_Int': conf_int
        })

    except Exception as e:
        print(f"Error fitting model: {e}")

# Convert results to DataFrame and sort by MAPE
results_df = pd.DataFrame([{
    'Model': r['Model'],
    'AIC': r['AIC'],
    'MSE': r['MSE'],
    'MAPE': r['MAPE'],
} for r in results])
results_df.sort_values(by='MAPE', inplace=True)

print("\nModel Comparison:")
print(results_df)

# Find the best model based on MAPE
best_model_idx = results_df['MAPE'].idxmin()
best_model = results[best_model_idx]
print(f"\nBest model by MAPE: {best_model['Model']}")
print(f"MAPE: {best_model['MAPE']:.2f}%")

# Plot the best model's forecasts
plt.figure(figsize=(15, 8))

# Plot actual values
plt.plot(train_data.index, train_data[target_col], color='blue', label='Training Data')
plt.plot(test_data.index, test_data[target_col], color='green', label='Test Data')

# Plot forecasts
forecasts = best_model['Forecasts']
conf_int = best_model['Conf_Int']

plt.plot(test_data.index, forecasts, color='red', label='Forecasts')
plt.fill_between(
    test_data.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='pink', alpha=0.3,
    label='95% Confidence Interval'
)

plt.title(f"Best Model: {best_model['Model']} - MAPE: {best_model['MAPE']:.2f}%")
plt.xlabel('Date')
plt.ylabel(f'{target_col}')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sarimax_walmart_forecast.png', dpi=300)
plt.show()

# Create a zoomed version around the test period
plt.figure(figsize=(15, 8))

# Define the zoom range (12 weeks before test data and all test data)
zoom_start = train_data.index[-12]
zoom_data = data.loc[zoom_start:]

# Plot zoomed actual values
plt.plot(zoom_data.index[:len(zoom_data) - len(test_data)],
         zoom_data[target_col][:len(zoom_data) - len(test_data)],
         color='blue', label='Training Data')
plt.plot(test_data.index, test_data[target_col], color='green', label='Test Data')

# Plot forecasts
plt.plot(test_data.index, forecasts, color='red', label='Forecasts')
plt.fill_between(
    test_data.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='pink', alpha=0.3,
    label='95% Confidence Interval'
)

plt.title(f"Best Model: {best_model['Model']} - MAPE: {best_model['MAPE']:.2f}% (Zoomed)")
plt.xlabel('Date')
plt.ylabel(f'{target_col}')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sarimax_walmart_forecast_zoomed.png', dpi=300)
plt.show()

# Save best model results for further analysis
output_df = pd.DataFrame({
    'Date': test_data.index,
    'Actual': test_data[target_col],
    'Forecast': forecasts,
    'Lower_CI': conf_int.iloc[:, 0],
    'Upper_CI': conf_int.iloc[:, 1],
    'Error': test_data[target_col] - forecasts,
    'Abs_Pct_Error': np.abs((test_data[target_col] - forecasts) / test_data[target_col]) * 100
})

output_df.to_csv('walmart_forecast_results.csv')
print("\nBest model forecast results saved to 'walmart_forecast_results.csv'")
print("Plots saved as 'sarimax_walmart_forecast.png' and 'sarimax_walmart_forecast_zoomed.png'")