import pandas as pd
import matplotlib.pyplot as plt
import holidays
from datetime import timedelta

# === Load the CSV files ===
sales_eval = pd.read_csv('data/sales_train_evaluation.csv')
calendar = pd.read_csv('data/calendar.csv')
sell_prices = pd.read_csv('data/sell_prices.csv')

# === Define holidays to keep (clean, no 'observed') ===
highlight_holidays = [
    "New Year's Day",
    "Martin Luther King Jr. Day",
    "Washington's Birthday",
    "Memorial Day",
    "Independence Day",
    "Labor Day",
    "Columbus Day",
    "Veterans Day",
    "Thanksgiving Day",
    "Christmas Day",
    "Juneteenth National Independence Day"
]

# === Reshape sales data to long format ===
sales_long = sales_eval.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    var_name='d',
    value_name='units_sold'
)

# === Merge with calendar to get actual dates and week info ===
sales_long = sales_long.merge(calendar[['d', 'date', 'wm_yr_wk']], on='d', how='left')
sales_long['date'] = pd.to_datetime(sales_long['date'])

# === Merge with sell prices ===
sales_long = sales_long.merge(
    sell_prices,
    how='left',
    on=['store_id', 'item_id', 'wm_yr_wk']
)

# === Calculate sales in USD ===
sales_long['sales_usd'] = sales_long['units_sold'] * sales_long['sell_price']

# === Assign week starting Monday ===
sales_long['week'] = sales_long['date'] - pd.to_timedelta(sales_long['date'].dt.weekday, unit='d')

# === Aggregate to weekly totals ===
weekly_data = sales_long.groupby('week').agg({
    'sales_usd': 'sum',
    'units_sold': 'sum'
}).reset_index()

# === Remove first week if partial ===
weekly_data = weekly_data.iloc[1:]

# === Create holiday exogenous variables ===
us_holidays = holidays.US(years=range(2011, 2022))
holiday_df = pd.DataFrame([
    {'date': pd.to_datetime(date), 'holiday': name}
    for date, name in us_holidays.items()
])
holiday_df['week'] = holiday_df['date'] - pd.to_timedelta(holiday_df['date'].dt.weekday, unit='d')

# One-hot encode and aggregate per week
holiday_flags = pd.get_dummies(holiday_df['holiday'])
holiday_df = pd.concat([holiday_df[['week']], holiday_flags], axis=1)
holiday_weekly = holiday_df.groupby('week').max().reset_index()

# Merge with sales data
weekly_data_exogs = weekly_data.merge(holiday_weekly, on='week', how='left')
weekly_data_exogs.fillna(0, inplace=True)

# Keep only selected holidays
existing_holidays = [col for col in highlight_holidays if col in weekly_data_exogs.columns]
weekly_data_exogs = weekly_data_exogs[['week', 'sales_usd', 'units_sold'] + existing_holidays]

# Ensure binary type
weekly_data_exogs[existing_holidays] = weekly_data_exogs[existing_holidays].astype(int)

# === Export final dataset ===
weekly_data_exogs.to_csv('weekly_sales_walmart.csv', index=False)

# === Plotting with clean holiday annotations ===
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot weekly sales USD
axes[0].plot(weekly_data_exogs['week'], weekly_data_exogs['sales_usd'], marker='o')
axes[0].set_title('Weekly Sales in USD')
axes[0].set_ylabel('Sales (USD)')
axes[0].grid(True)

# Plot weekly sales volume
axes[1].plot(weekly_data_exogs['week'], weekly_data_exogs['units_sold'], marker='o', color='orange')
axes[1].set_title('Weekly Sales Volume (Units Sold)')
axes[1].set_xlabel('Week')
axes[1].set_ylabel('Units Sold')
axes[1].grid(True)

# === Mark selected holidays ===
labeled_years = set()

for holiday in existing_holidays:
    holiday_weeks = weekly_data_exogs.loc[weekly_data_exogs[holiday] == 1, 'week']
    for week in holiday_weeks:
        for ax in axes:
            ax.axvline(x=week, color='red', linestyle='--', alpha=0.3)
        year = pd.to_datetime(week).year
        label_key = f"{holiday}_{year}"
        if label_key not in labeled_years:
            axes[0].text(week, axes[0].get_ylim()[1]*0.98, holiday, rotation=90,
                         fontsize=8, color='red', ha='right')
            labeled_years.add(label_key)

plt.tight_layout()
plt.show()
