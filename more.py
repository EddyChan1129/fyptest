import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error
import ta

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def find_similar_periods(ticker, current_start_date, current_end_date, past_start_year, past_end_year, step_months, top_n=5):
    current_data = get_stock_data(ticker, current_start_date, current_end_date)
    
    similar_periods = {}
    
    for year in range(past_start_year, past_end_year + 1):
        for month in range(3, 13, step_months):
            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.DateOffset(years=2) - pd.DateOffset(days=1)
            
            if end_date > current_end_date:
                continue
            
            past_data = get_stock_data(ticker, start_date, end_date)
            similarity_score = calculate_similarity(current_data['Close'], past_data['Close'])
            
            similar_periods[(start_date, end_date)] = similarity_score
    
    ranked_periods = sorted(similar_periods.items(), key=lambda x: x[1])
    
    return ranked_periods[:top_n]

def calculate_similarity(current_data, past_data):
    min_length = min(len(current_data), len(past_data))
    current_data = current_data[:min_length]
    past_data = past_data[:min_length]
    
    return mean_squared_error(current_data, past_data)

def calibrate_indicators(data, rsi_window, sma_short_window, sma_long_window):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], rsi_window).rsi()
    data['SMA_short'] = ta.trend.sma_indicator(data['Close'], sma_short_window)
    data['SMA_long'] = ta.trend.sma_indicator(data['Close'], sma_long_window)
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=20)  # Adding EMA
    data['MACD'] = ta.trend.macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9)  # Adding MACD




def evaluate_performance(actual_data, predicted_data):
    # Remove NaN values from both actual and predicted data
    actual_data = actual_data.dropna()
    predicted_data = predicted_data.dropna()

    # Align the lengths of actual and predicted data after removing NaNs
    min_length = min(len(actual_data), len(predicted_data))
    actual_data = actual_data[:min_length]
    predicted_data = predicted_data[:min_length]

    return mean_squared_error(actual_data, predicted_data)

def predict_prices(data, sma_short_window, sma_long_window):
    data['SMA_short'] = ta.trend.sma_indicator(data['Close'], sma_short_window)
    data['SMA_long'] = ta.trend.sma_indicator(data['Close'], sma_long_window)
    predicted_prices = data['SMA_short'] * 0.5 + data['SMA_long'] * 0.5  # Adjust this based on your prediction logic
    return predicted_prices

# Example usage parameters
ticker_symbol = 'TSLA'
current_start = pd.Timestamp(year=2023, month=6, day=1)
current_end = pd.Timestamp(year=2023, month=6, day=30)
past_start_year = 2010
past_end_year = 2023
step_months = 3

# Find similar periods
similar_periods = find_similar_periods(ticker_symbol, current_start, current_end, past_start_year, past_end_year, step_months)
top_period = similar_periods[0][0]

# Get stock data for the top similar period
top_data = get_stock_data(ticker_symbol, top_period[0], top_period[1])

# Calibrate indicators for the top similar period
rsi_window = 14
sma_short_window = 20
sma_long_window = 50
calibrate_indicators(top_data, rsi_window, sma_short_window, sma_long_window)

def calibrate_indicators(data, rsi_window, sma_short_window, sma_long_window):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], rsi_window).rsi()
    data['SMA_short'] = ta.trend.sma_indicator(data['Close'], sma_short_window)
    data['SMA_long'] = ta.trend.sma_indicator(data['Close'], sma_long_window)
    data['MACD'] = ta.trend.macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9)  # Adding MACD


# Print calibrated indicators
print(top_data[['RSI', 'SMA_short', 'SMA_long', 'MACD']])

# Get stock data for future period
future_start = top_period[1] + pd.DateOffset(days=1)
future_end = future_start + pd.DateOffset(years=2) - pd.DateOffset(days=1)
future_data = get_stock_data(ticker_symbol, future_start, future_end)

# Example: Assuming a simple prediction using SMA as a placeholder
sma_short_window = 20
sma_long_window = 50
predicted_prices_sma = predict_prices(future_data, sma_short_window, sma_long_window)
actual_prices = future_data['Close']

# Evaluate performance for SMA and RSI
sma_performance = evaluate_performance(actual_prices, predicted_prices_sma)
future_data['RSI'] = ta.momentum.RSIIndicator(future_data['Close'], rsi_window).rsi()
rsi_performance = evaluate_performance(actual_prices, future_data['RSI'])

# Evaluate performance for EMA and MACD

predicted_prices_macd = future_data['MACD']
macd_performance = evaluate_performance(actual_prices, predicted_prices_macd)

print(f"Performance with calibrated SMA: {sma_performance}")
print(f"Performance with calibrated RSI: {rsi_performance}")
print(f"Performance with calibrated MACD: {macd_performance}")
