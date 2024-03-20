import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

def compute_sma(data, window):
    return data.rolling(window=window).mean()

def evaluate_performance(data, short_window, long_window):
    short_sma = compute_sma(data, short_window)
    long_sma = compute_sma(data, long_window)

    # Generate trading signals
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['short_sma'] = short_sma
    signals['long_sma'] = long_sma
    signals['signal'][short_window:] = np.where(signals['short_sma'][short_window:] > signals['long_sma'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    # Calculate returns
    portfolio = signals.copy()
    portfolio['market_returns'] = data.pct_change()
    portfolio['strategy_returns'] = portfolio['market_returns'] * portfolio['positions'].shift(1)

    total_return = portfolio['strategy_returns'].cumsum().iloc[-1]
    return total_return

def grid_search(data, param1_range, param2_range, step_size):
    best_performance = float('-inf')
    best_params = (None, None)

    for param1 in range(param1_range[0], param1_range[1], step_size):
        for param2 in range(param2_range[0], param2_range[1], step_size):
            if param1 >= param2:  # Ensure short_window < long_window
                continue
            performance = evaluate_performance(data, param1, param2)
            if performance > best_performance:
                best_performance = performance
                best_params = (param1, param2)

    return best_params

# Fetch historical data
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-01-01'
data = get_stock_data(ticker, start_date, end_date)

# Define the range for SMA windows and the step size
param1_range = (10, 50)  # Short window range
param2_range = (20, 100)  # Long window range
step_size = 5  # Step size for the grid search

# Perform the grid search
best_params = grid_search(data, param1_range, param2_range, step_size)
print(f"Best SMA parameters: Short window = {best_params[0]}, Long window = {best_params[1]}")

# You could refine the search around the best parameters found using a smaller step size if necessary
