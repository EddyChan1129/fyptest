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
def calibrate_indicators(data, rsi_window, sma_window, ema_window,bollinger_window, stoch_window):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], rsi_window).rsi()
    data['SMA'] = ta.trend.sma_indicator(data['Close'], sma_window)
    data['EMA'] = ta.trend.EMAIndicator(data['Close'], ema_window).ema_indicator()
    data['Bollinger'] = ta.volatility.BollingerBands(data['Close'], bollinger_window).bollinger_mavg()
    data['Stochastic'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], stoch_window).stoch()



def find_similar_periods(ticker, current_start_date, current_end_date, past_start_year, past_end_year, step_months, top_n=5):
    current_data = get_stock_data(ticker, current_start_date, current_end_date)
    
    similar_periods = {}
    
    for year in range(past_start_year, past_end_year + 1):
        for month in range(1, 13, step_months):
            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.DateOffset(months=3)  # Do not subtract 1 day
            
            if end_date > current_end_date:
                continue
            
            try:
                past_data = get_stock_data(ticker, start_date, end_date)
                similarity_score = calculate_similarity(current_data['Close'], past_data['Close'])
                similar_periods[(start_date, end_date)] = (similarity_score, (start_date, end_date))
            except Exception as e:
                print(f"Failed to get data for period {start_date} to {end_date}: {e}")
    
    ranked_periods = sorted(similar_periods.items(), key=lambda x: x[1][0])
    
    return [x[1][1] for x in ranked_periods[:top_n]]


def calculate_profit(actual_data, predicted_data, indicator, init_price):
    # use plot to show the actual and predicted data
    # Find the first non-NaN value in predicted_data
    start_index = predicted_data.first_valid_index()

    # Synchronize actual_data with predicted_data
    actual_data = actual_data.loc[start_index:]
    # Now plot the actual and predicted data
    # The Y should 5 not 10

    plt.title(indicator)
    plt.plot(actual_data, label='Actual')
    plt.plot(predicted_data, label='Predicted')
    plt.legend()
    plt.show()

    # Calculate the profit
    profit = 0
    bought = False
    for i in range(1, len(actual_data)):
        if predicted_data.iloc[i] > actual_data.iloc[i] and not bought:
            bought = True
            profit -= actual_data.iloc[i]

        elif predicted_data.iloc[i] < actual_data.iloc[i] and bought:
            bought = False
            profit += actual_data.iloc[-1]
    if bought:
        profit += actual_data.iloc[-1]
    
    # Calculate the final price
    final_price = init_price + profit

    return profit, final_price



# Predict prices using the single-period SMA
def predict_prices(data, sma_window):
    data['SMA'] = ta.trend.sma_indicator(data['Close'], sma_window)
    predicted_prices = data['SMA']  # Use the single-period SMA directly
    return predicted_prices

# Example usage
ticker_symbol = input("Enter the ticker symbol: ")
# The current_start and current_end should be user input
current_start = pd.Timestamp(year=2023, month=12, day=1)
current_end = pd.Timestamp(year=2024, month=3, day=21)
past_start_year = 2014
past_end_year = 2024
step_months = 3

# Calibrate indicators
rsi_window = 14
sma_window = 20
ema_window = 12
bollinger_window = 20
stoch_window = 14
init_price = 1000

sma_profits = []
rsi_profits = []
ema_profits = []
bollinger_profits = []
stoch_profits = []

sma_final_prices = []
rsi_final_prices = []
ema_final_prices = []
bollinger_final_prices = []
stoch_final_prices = []

similar_periods = find_similar_periods(ticker_symbol, current_start, current_end, past_start_year, past_end_year, step_months)
for i, period in enumerate(similar_periods, 1):
    # Fetch the data for the two periods
    data_current = get_stock_data(ticker_symbol, current_start, current_end)
    data_similar = get_stock_data(ticker_symbol, period[0], period[1])
    # Calibrate indicators
    calibrate_indicators(data_similar, rsi_window, sma_window, ema_window,bollinger_window, stoch_window)

    # Define future_start and future_end (3 month)
    future_start = period[1]
    future_end = future_start + pd.DateOffset(months=3)

    # Evaluate performance with past future periods
    future_data = get_stock_data(ticker_symbol, future_start, future_end)
    calibrate_indicators(future_data, rsi_window, sma_window, ema_window, bollinger_window, stoch_window)

    predicted_prices = predict_prices(future_data, sma_window)
    actual_prices = future_data['Close']

    # Calculate profit and final price for each indicator
    sma_profit, sma_final_price = calculate_profit(actual_prices, predicted_prices, "SMA", init_price)
    rsi_profit, rsi_final_price = calculate_profit(actual_prices, future_data['RSI'], "RSI", init_price)
    ema_profit, ema_final_price = calculate_profit(actual_prices, future_data['EMA'], "EMA", init_price)
    bollinger_profit, bollinger_final_price = calculate_profit(actual_prices, future_data['Bollinger'], "Bollinger", init_price)
    stoch_profit, stoch_final_price = calculate_profit(actual_prices, future_data['Stochastic'], "Stochastic", init_price)
    # Append profits and final prices to the respective lists
    sma_profits.append(sma_profit)
    rsi_profits.append(rsi_profit)
    ema_profits.append(ema_profit)
    bollinger_profits.append(bollinger_profit)
    stoch_profits.append(stoch_profit)

    sma_final_prices.append(sma_final_price)
    rsi_final_prices.append(rsi_final_price)
    ema_final_prices.append(ema_final_price)
    bollinger_final_prices.append(bollinger_final_price)
    stoch_final_prices.append(stoch_final_price)

# Calculate average profit and final price for each indicator
average_sma_profit = sum(sma_profits) / len(sma_profits)
average_rsi_profit = sum(rsi_profits) / len(rsi_profits)
average_ema_profit = sum(ema_profits) / len(ema_profits)
average_bollinger_profit = sum(bollinger_profits) / len(bollinger_profits)
average_stoch_profit = sum(stoch_profits) / len(stoch_profits)

average_sma_final_price = sum(sma_final_prices) / len(sma_final_prices)
average_rsi_final_price = sum(rsi_final_prices) / len(rsi_final_prices)
average_ema_final_price = sum(ema_final_prices) / len(ema_final_prices)
average_bollinger_final_price = sum(bollinger_final_prices) / len(bollinger_final_prices)
average_stoch_final_price = sum(stoch_final_prices) / len(stoch_final_prices)

print(f"Average SMA profit: {average_sma_profit}")
print(f"Average RSI profit: {average_rsi_profit}")
print(f"Average EMA profit: {average_ema_profit}")
print(f"Average Bollinger Bands profit: {average_bollinger_profit}")
print(f"Average Stochastic Oscillator profit: {average_stoch_profit}")

print(f"Average SMA final price: {average_sma_final_price}")
print(f"Average RSI final price: {average_rsi_final_price}")
print(f"Average EMA final price: {average_ema_final_price}")
print(f"Average Bollinger Bands final price: {average_bollinger_final_price}")
print(f"Average Stochastic Oscillator final price: {average_stoch_final_price}")


