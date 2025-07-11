# Walmart Sales Forecasting

This project is based on the Walmart data from the M5 competition. The raw sales data is transformed into weekly total sales (in USD) and volumes (units sold) data. Time series forecasting models are applied on these weekly aggregates to predict future sales.

## Data Structure

The dataset contains weekly sales information:
- Date column formatted as week start (Monday)
- Sales amount in USD
- Number of units sold
- Binary indicators for 10+ US holidays (1 = holiday occurred that week)

## Data Processing

The project processes Walmart's hierarchical raw data by:
1. Converting from wide to long format
2. Aggregating to weekly totals for sales and units sold
3. Adding holiday indicators as exogenous variables
4. Calculating sales in USD based on units sold and pricing

## Models

### SARIMAX
- Tests multiple parameter combinations for optimal forecasting
- Incorporates holiday indicators as exogenous variables
- Evaluates models using AIC, MSE, and MAPE metrics
- Creates visualizations of forecasts with confidence intervals

### Auto ARIMA
- Automatically selects optimal ARIMA parameters
- Incorporates holiday information as exogenous variables
- Evaluates using RMSE
- Provides forecast visualization

## Project Structure

- `preprocessing.py`: Data cleaning and preparation
- `forecast.py`: SARIMAX model implementation and evaluation
- `auto_arima.py`: Auto ARIMA model implementation
- `weekly_sales_walmart.csv`: Processed weekly sales data with holiday indicators

## Results

Model forecasts are evaluated on a 52-week test period, with performance metrics saved to CSV and visualizations saved as PNG files.