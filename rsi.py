import yfinance as yf
import pandas as pd
import pandas_ta as ta

def evaluate_rsi(ticker, data, calibration_start, calibration_end, evaluation_start, evaluation_end, rsi_periods=[14]):
    best_performance = float('-inf')
    best_rsi_period = 14
    for period in rsi_periods:
        # Calculate RSI for the given period
        rsi = ta.rsi(data['Close'], length=period)

        # Select the calibration period for analysis
        rsi_calibration = rsi[calibration_start:calibration_end]
        if rsi_calibration.empty:
            continue

        # Evaluate performance (e.g., total return) of the RSI for this period
        total_return, _ = evaluate_rsi_profitability(data[calibration_start:calibration_end], period)
        if total_return > best_performance:
            best_performance = total_return
            best_rsi_period = period

    # Use the best RSI period for evaluation
    rsi_evaluation = ta.rsi(data['Close'], length=best_rsi_period)[evaluation_start:evaluation_end]
    performance_evaluation = analyze_performance(rsi_evaluation)

    return best_rsi_period, performance_evaluation

def evaluate_rsi_profitability(data, rsi_period=14, overbought_threshold=70, oversold_threshold=30):
    rsi = ta.rsi(data['Close'], length=rsi_period)

    buy_signals = (rsi < oversold_threshold)
    sell_signals = (rsi > overbought_threshold)

    buy_prices = data['Close'][buy_signals]
    sell_prices = data['Close'][sell_signals]

    if not buy_prices.empty and not sell_prices.empty:
        if buy_prices.index[0] > sell_prices.index[0]:
            sell_prices = sell_prices.iloc[1:]

        num_trades = min(len(buy_prices), len(sell_prices))
        buy_prices = buy_prices.iloc[:num_trades]
        sell_prices = sell_prices.iloc[:num_trades]

        trade_returns = sell_prices.values / buy_prices.values - 1
        total_return = trade_returns.sum()

        return total_return, trade_returns

    return 0, pd.Series()

def analyze_performance(rsi_series):
    if rsi_series.empty:
        return 0
    return rsi_series.median()

ticker_symbol = 'AAPL'
calibration_period_1 = pd.Timestamp('2023-01-01')
evaluation_period_1 = pd.Timestamp('2024-01-01')
calibration_period_2 = pd.Timestamp('2008-01-01')
evaluation_period_2 = pd.Timestamp('2009-01-01')

start_date = min(calibration_period_1, calibration_period_2) - pd.DateOffset(years=1)
end_date = max(evaluation_period_1, evaluation_period_2)
data = yf.download(ticker_symbol, start=start_date, end=end_date)

rsi_periods_to_test = range(5, 21)  # Testing RSI periods from 5 to 20
best_rsi_period_1, performance_1_evaluation = evaluate_rsi(ticker_symbol, data, calibration_period_1, calibration_period_1 + pd.DateOffset(years=1), evaluation_period_1, evaluation_period_1 + pd.DateOffset(years=1), rsi_periods=rsi_periods_to_test)
best_rsi_period_2, performance_2_evaluation = evaluate_rsi(ticker_symbol, data, calibration_period_2, calibration_period_2 + pd.DateOffset(years=1), evaluation_period_2, evaluation_period_2 + pd.DateOffset(years=1), rsi_periods=rsi_periods_to_test)

average_evaluation_performance = (performance_1_evaluation + performance_2_evaluation) / 2
print("Average evaluation performance of RSI:", average_evaluation_performance)
print(f"Best RSI period for first calibration: {best_rsi_period_1}")
