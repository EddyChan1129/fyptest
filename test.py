import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error
import ta
import numpy as np
import matplotlib.pyplot as plt
# Download stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Calculate similarity between two time series
def calculate_similarity(current_data, past_data):
    min_length = min(len(current_data), len(past_data))
    current_data = current_data[:min_length]
    past_data = past_data[:min_length]
    
    # Normalize the data
    current_data = (current_data - np.mean(current_data)) / np.std(current_data)
    past_data = (past_data - np.mean(past_data)) / np.std(past_data)
    
    return mean_squared_error(current_data, past_data)



# Calibrate indicators
def calibrate_indicators(data, rsi_window, sma_window):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], rsi_window).rsi()
    data['SMA'] = ta.trend.sma_indicator(data['Close'], sma_window)




def find_similar_periods(ticker, current_start_date, current_end_date, past_start_year, past_end_year, step_months, top_n=5):
    current_data = get_stock_data(ticker, current_start_date, current_end_date)
    
    similar_periods = {}
    
    for year in range(past_start_year, past_end_year + 1):
        for month in range(1, 13, step_months):
            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.DateOffset(months=3) - pd.DateOffset(days=1)
            
            if end_date > current_end_date:
                continue
            
            try:
                past_data = get_stock_data(ticker, start_date, end_date)
                similarity_score = calculate_similarity(current_data['Close'], past_data['Close'])
                similar_periods[(start_date, end_date)] = similarity_score
            except Exception as e:
                print(f"Failed to get data for period {start_date} to {end_date}: {e}")
    
    ranked_periods = sorted(similar_periods.items(), key=lambda x: x[1])
    
    return ranked_periods[:top_n]



def calculate_profit(actual_data, predicted_data):
    # Assume that we buy when the prediction is higher than the current price and sell when it's lower
    profit = 0
    for actual, predicted in zip(actual_data, predicted_data):
        if predicted > actual:  # Buy
            profit -= actual
        elif predicted < actual:  # Sell
            profit += actual
    return profit

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    # Calculate the excess returns
    excess_returns = returns - risk_free_rate

    # Calculate the average of the excess returns
    avg_excess_return = excess_returns.mean()

    # Calculate the standard deviation of the excess returns
    std_dev_excess_return = excess_returns.std()

    # Calculate the Sharpe Ratio
    sharpe_ratio = avg_excess_return / std_dev_excess_return

    return sharpe_ratio

# Predict prices using the single-period SMA
def predict_prices(data, sma_window):
    data['SMA'] = ta.trend.sma_indicator(data['Close'], sma_window)
    predicted_prices = data['SMA']  # Use the single-period SMA directly
    return predicted_prices


# Example usage
ticker_symbol = '0001.hk'
current_start = pd.Timestamp(year=2023, month=12, day=1)
current_end = pd.Timestamp(year=2024, month=3, day=21)
past_start_year = 2010
past_end_year = 2022
step_months = 3

similar_periods = find_similar_periods(ticker_symbol, current_start, current_end, past_start_year, past_end_year, step_months)
for i, period in enumerate(similar_periods, 1):
    print(f"The {i}{'' if i == 0 else 'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} most similar year period is:", period[0])

    
    # The most simliar period is the first period
    if ( i == 1):
         #Fetch the data for the two periods
        data_current = get_stock_data('0001.hk', current_start, current_end)
        # period[0][0] is the start date and period[0][1] is the end date
        data_similar = get_stock_data('0001.hk', period[0][0], period[0][1])

        # Plot the closing prices for the current period
        plt.figure(figsize=(14, 7))
        plt.plot(data_current.index, data_current['Close'], label='Current Period')

        # Plot the closing prices for the most similar period
        plt.plot(data_similar.index, data_similar['Close'], label='Most Similar Period')

        plt.title('0001.hk Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

# Select the top similar period for calibration
top_period = similar_periods[0][0]
top_data = get_stock_data(ticker_symbol, top_period[0], top_period[1])

# Calibrate indicators
rsi_window = 7
sma_window = 20  # Define a single SMA window
calibrate_indicators(top_data, rsi_window, sma_window)

# Evaluate performance with past future periods
# After 1 year of the top similar period is the future period we want to predict
future_start = top_period[1] + pd.DateOffset(days=1)
future_end = future_start + pd.DateOffset(years=1) - pd.DateOffset(days=1)
future_data = get_stock_data(ticker_symbol, future_start, future_end)

predicted_prices = predict_prices(future_data, sma_window)
actual_prices = future_data['Close']

# Evaluate performance for SMA and RSI
sma_profit = calculate_profit(actual_prices, predicted_prices)
future_data['RSI'] = ta.momentum.RSIIndicator(future_data['Close'], rsi_window).rsi()
rsi_profit = calculate_profit(actual_prices, future_data['RSI'])

# Calculate the Sharpe Ratio for SMA and RSI
sma_shape = calculate_sharpe_ratio(predicted_prices.pct_change())
rsi_shape = calculate_sharpe_ratio(future_data['RSI'].pct_change())

print(f"Profit with calibrated SMA: {sma_profit}")
print(f"sharpe_ratio with calibrated SMA: {sma_shape}")
print(f"Profit with calibrated RSI: {rsi_profit}")
print(f"sharpe_ratio with calibrated RSI: {rsi_shape}")


