import pandas as pd
from pandas.tseries.offsets import DateOffset
import yfinance as yf
from sklearn.metrics import mean_squared_error
import ta
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

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
    print("data[RSI].33",data['RSI'])
    data['SMA'] = ta.trend.sma_indicator(data['Close'], sma_window)
    print("data[SMA].35",data['SMA'])
    data['EMA'] = ta.trend.EMAIndicator(data['Close'], ema_window).ema_indicator()
    print("data[EMA].37",data['EMA'])
    data['Bollinger'] = ta.volatility.BollingerBands(data['Close'], bollinger_window).bollinger_mavg()
    print("data[Bollinger].39",data['Bollinger'])
    data['Stochastic'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], stoch_window).stoch()
    print("data[Stochastic].41",data['Stochastic'])



def find_similar_periods(ticker, current_start_date, current_end_date, past_start_year, past_end_year, step_months, top_n=5):
    current_data = get_stock_data(ticker, current_start_date, current_end_date)
    
    similar_periods = {}
    
    for year in range(past_start_year, past_end_year + 1):
        for month in range(1, 13, step_months):
            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.DateOffset(months=step_months)  


            print(f"Checking period {start_date} to {end_date}")
            
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

has_been_called = False

def calculate_profit(actual_data, predicted_data, indicator, init_price):
    # use plot to show the actual and predicted data
    start_index = predicted_data.first_valid_index()

    # Synchronize actual_data with predicted_data
    actual_data = actual_data.loc[start_index:]

    """global has_been_called

    if not has_been_called:
        plt.title(indicator)
        plt.plot(actual_data, label='Actual')
        plt.plot(predicted_data, label='Predicted')
        plt.legend()
        #plt.show()
        has_been_called = True
    """
    # Calculate the profit
    profit = 0
    bought = False
    #int init_price
    money_left = int(init_price)

    # Moving Average
    short_term_ma = predicted_data.rolling(window=5).mean()
    long_term_ma = predicted_data.rolling(window=20).mean()

    # RSI
    delta = predicted_data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=14).mean()
    average_loss = abs(down.rolling(window=14).mean())
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma = predicted_data.rolling(window=20).mean()
    std = predicted_data.rolling(window=20).std()
    upper_bb = sma + (2 * std)
    lower_bb = sma - (2 * std)

    # Stochastic Oscillator
    low_min  = predicted_data.rolling( window = 14 ).min()
    high_max = predicted_data.rolling( window = 14 ).max()
    k = 100 * (predicted_data - low_min) / (high_max - low_min)
    d = k.rolling(window = 3).mean()

    for i in range(1, len(actual_data)):
        # Buy signal
        if (short_term_ma.iloc[i] > long_term_ma.iloc[i] or rsi.iloc[i] < 30 or predicted_data.iloc[i] < lower_bb.iloc[i] or k.iloc[i] < 20) and money_left >= actual_data.iloc[i]:
            stocks_to_buy = money_left // actual_data.iloc[i]  # Calculate how many stocks to buy
            bought += stocks_to_buy  # Buy the stocks
            money_spent = stocks_to_buy * actual_data.iloc[i]
            profit -= money_spent
            money_left -= money_spent  # Deduct the money spent from the money left
        # Sell signal
        elif (short_term_ma.iloc[i] < long_term_ma.iloc[i] or rsi.iloc[i] > 70 or predicted_data.iloc[i] > upper_bb.iloc[i] or k.iloc[i] > 80) and bought > 0:
            money_earned = bought * actual_data.iloc[i]  # Sell all the stocks
            profit += money_earned
            money_left += money_earned  # Add the money earned to the money left
            bought = 0  # Reset the number of stocks bought
    if bought > 0:
        money_earned = bought * actual_data.iloc[-1]  # Sell all the stocks
        profit += money_earned
        money_left += money_earned  # Add the money earned to the money left

    # Calculate the final price
    final_price = init_price + profit

    # Calculate return
    return_rate = profit / init_price

    # Calculate Sharpe ratio
    excess_returns = predicted_data.pct_change().dropna()

    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    return profit, final_price, return_rate, sharpe_ratio



# Predict prices using the single-period SMA
def predict_prices(data, sma_window):
    data['SMA'] = ta.trend.sma_indicator(data['Close'], sma_window)
    predicted_prices = data['SMA']  # Use the single-period SMA directly
    return predicted_prices

def calculate_profit_with_signals(prices, buy_signals, sell_signals, initial_cash):
    cash = initial_cash
    stock = 0
    buy_price = 0

    for i in range(len(prices)):
        # If we have a buy signal and we don't own the stock yet, we buy
        if buy_signals[i] and stock == 0:
            stock = cash / prices[i]
            cash = 0
            buy_price = prices[i]

        # If we have a sell signal and we own the stock, we sell
        elif sell_signals[i] and stock > 0:
            cash = stock * prices[i]
            stock = 0

    # If we own the stock at the end of the period, we sell it
    if stock > 0:
        cash = stock * prices.iloc[-1]

    # The profit is the final amount of cash minus the initial amount
    profit = cash - initial_cash

    return profit

def calibrate_rsi(data, rsi_window):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], rsi_window).rsi()
    return data

def backtest_rsi(ticker_symbol):
    # Get the current date
    current_date = datetime.now()

    # Calculate the start date for 3 months ago
    start_date = current_date - timedelta(days=90)

    # Get the stock data for the last 3 months
    rsi_stock_data = get_stock_data(ticker_symbol, start_date, current_date)

    # Calibrate the RSI indicator
    rsi_stock_data = calibrate_rsi(rsi_stock_data, rsi_window=14)

    # Generate buy and sell signals
    buy_signals = rsi_stock_data['RSI'] < 30
    sell_signals = rsi_stock_data['RSI'] > 70

    # Calculate the profit
    profit = calculate_profit_with_signals(rsi_stock_data['Close'], buy_signals, sell_signals, 1000)

    final_price = 1000 + profit

    # calculate the return rate
    return_rate = profit / 1000

    # calculate the sharpe ratio
    excess_returns = rsi_stock_data['Close'].pct_change().dropna()
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    return profit, final_price, return_rate, sharpe_ratio

def calibrate_sma(data, sma_window):
    data['SMA'] = ta.trend.sma_indicator(data['Close'], sma_window)
    return data

def backtest_sma(ticker_symbol):
    start_date = datetime.now() - timedelta(days=90)
    current_date = datetime.now()
    sam_stock_data = get_stock_data(ticker_symbol, start_date, current_date)
    sam_stock_data = calibrate_sma(sam_stock_data, sma_window=20)
    buy_signals = sam_stock_data['SMA'] < sam_stock_data['Close']
    sell_signals = sam_stock_data['SMA'] > sam_stock_data['Close']
    profit = calculate_profit_with_signals(sam_stock_data['Close'], buy_signals, sell_signals, 1000)

    final_price = 1000 + profit

    # calculate the return rate
    return_rate = profit / 1000

    # calculate the sharpe ratio
    excess_returns = sam_stock_data['Close'].pct_change().dropna()
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    return profit, final_price, return_rate, sharpe_ratio

def calibrate_ema(data, ema_window):
    data['EMA'] = ta.trend.EMAIndicator(data['Close'], ema_window).ema_indicator()
    return data

def backtest_ema(ticker_symbol):
    start_date = datetime.now() - timedelta(days=90)
    current_date = datetime.now()
    ema_stock_data = get_stock_data(ticker_symbol, start_date, current_date)
    ema_stock_data = calibrate_ema(ema_stock_data, ema_window=12)
    buy_signals = ema_stock_data['EMA'] < ema_stock_data['Close']
    sell_signals = ema_stock_data['EMA'] > ema_stock_data['Close']
    profit = calculate_profit_with_signals(ema_stock_data['Close'], buy_signals, sell_signals, 1000)

    # calulate the final price
    final_price = 1000 + profit

    # calculate the return rate
    return_rate = profit / 1000

    # calculate the sharpe ratio
    excess_returns = ema_stock_data['Close'].pct_change().dropna()
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    return profit, final_price, return_rate, sharpe_ratio

def calibrate_stoch(data, stoch_window):
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], stoch_window)
    data['Stoch'] = stoch.stoch()
    # Set pandas options
    """ pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None) """

    # Your code
    data['Stoch'] = stoch.stoch()
    print("data[Stoch]", data['Stoch'])
    return data

def backtest_stoch(ticker_symbol):
    start_date = datetime.now() - timedelta(days=90)
    current_date = datetime.now()
    stoch_stock_data = get_stock_data(ticker_symbol, start_date, current_date)
    stoch_stock_data = calibrate_stoch(stoch_stock_data, stoch_window=14)
    buy_signals = stoch_stock_data['Stoch'] < 20
    sell_signals = stoch_stock_data['Stoch'] > 80
    profit = calculate_profit_with_signals(stoch_stock_data['Close'], buy_signals, sell_signals, 1000)
    final_price = 1000 + profit

    # calculate the return rate
    return_rate = profit / 1000

    # calculate the sharpe ratio
    excess_returns = stoch_stock_data['Close'].pct_change().dropna()
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    return profit, final_price, return_rate, sharpe_ratio

def calibrate_bollinger(data, bollinger_window):
    bollinger = ta.volatility.BollingerBands(data['Close'], bollinger_window)
    data['Bollinger'] = bollinger.bollinger_mavg()
    """pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None) """
    print("data[Bollinger]", data['Bollinger'])
    return data

def backtest_bollinger(ticker_symbol):
    start_date = datetime.now() - timedelta(days=90)
    current_date = datetime.now()
    bollinger_stock_data = get_stock_data(ticker_symbol, start_date, current_date)
    bollinger_stock_data = calibrate_bollinger(bollinger_stock_data, bollinger_window=20)

    print("bollinger_stock_data['Close']", bollinger_stock_data['Close'])
    buy_signals = bollinger_stock_data['Close'] < bollinger_stock_data['Bollinger']
    sell_signals = bollinger_stock_data['Close'] > bollinger_stock_data['Bollinger']
    profit = calculate_profit_with_signals(bollinger_stock_data['Close'], buy_signals, sell_signals, 1000)
    final_price = 1000 + profit

    # calculate the return rate
    return_rate = profit / 1000

    # calculate the sharpe ratio
    excess_returns = bollinger_stock_data['Close'].pct_change().dropna()
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    return profit, final_price, return_rate, sharpe_ratio

def analyze_stock(ticker_symbol,init_price):
    endYear = datetime.now().year
    endMonth = datetime.now().month
    endDay = datetime.now().day

    current_start = pd.Timestamp(year=2023, month=12, day=1)
    current_end = current_start + DateOffset(months=3)

    past_start_year = 2014
    past_end_year = 2024
    step_months = 3

    # Calibrate indicators
    rsi_window = 14
    sma_window = 20
    ema_window = 12
    bollinger_window = 20
    stoch_window = 14

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

    # Initialize lists for return rates
    sma_return_rates = []
    rsi_return_rates = []
    ema_return_rates = []
    bollinger_return_rates = []
    stoch_return_rates = []

    # Initialize list for sharpe ratios
    sma_sharpe_ratios = []
    rsi_sharpe_ratios = []
    ema_sharpe_ratios = []
    bollinger_sharpe_ratios = []
    stoch_sharpe_ratios = []



    similar_periods = find_similar_periods(ticker_symbol, current_start, current_end, past_start_year, past_end_year, step_months)
    print(f"Similar periods: {similar_periods}\n")
    for i, period in enumerate(similar_periods, 1):
        simplars_start_date, simplars_end_date = period
        data = get_stock_data(ticker_symbol, simplars_start_date, simplars_end_date)  # Assuming you have a function to fetch the data

        plt.figure(figsize=(12,7))
        # Plot the fetched stock data
        plt.plot(data['Close'], label=f'Period {i}')
        plt.title(f'{ticker_symbol} Stock Price for Similar Periods {i}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.savefig(f'./static/sim{i}.png') 

        # Fetch the data for the two periods
        data_current = get_stock_data(ticker_symbol, current_start, current_end)
        data_similar = get_stock_data(ticker_symbol, period[0], period[1])
        # Calibrate indicators
        calibrate_indicators(data_similar, rsi_window, sma_window, ema_window,bollinger_window, stoch_window)

        # Define future_start and future_end (3 month)
        future_start = period[1]
        future_end = future_start + pd.DateOffset(months=step_months)

        # Evaluate performance with past future periods
        future_data = get_stock_data(ticker_symbol, future_start, future_end)
        calibrate_indicators(future_data, rsi_window, sma_window, ema_window, bollinger_window, stoch_window)

        predicted_prices = predict_prices(future_data, sma_window)
        actual_prices = future_data['Close']

        #    return profit,return_rate, sharpe_ratio, final_price

        
        # Inside the loop
        sma_profit, sma_final_price,sma_return_rate,sma_sharpe_ratio = calculate_profit(actual_prices, predicted_prices, "SMA", init_price)
        rsi_profit, rsi_final_price, rsi_return_rate,rsi_sharpe_ratio = calculate_profit(actual_prices, future_data['RSI'], "RSI", init_price)
        ema_profit, ema_final_price, ema_return_rate, ema_sharpe_ratio = calculate_profit(actual_prices, future_data['EMA'], "EMA", init_price)
        bollinger_profit, bollinger_final_price, bollinger_return_rate, bollinger_sharpe_ratio = calculate_profit(actual_prices, future_data['Bollinger'], "Bollinger", init_price)
        stoch_profit, stoch_final_price, stoch_return_rate, stoch_sharpe_ratio = calculate_profit(actual_prices, future_data['Stochastic'], "Stochastic", init_price)
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

        # Append return rates to the respective lists
        sma_return_rates.append(sma_return_rate)
        rsi_return_rates.append(rsi_return_rate)
        ema_return_rates.append(ema_return_rate)
        bollinger_return_rates.append(bollinger_return_rate)
        stoch_return_rates.append(stoch_return_rate)

        sma_sharpe_ratios.append(sma_sharpe_ratio)
        rsi_sharpe_ratios.append(rsi_sharpe_ratio)
        ema_sharpe_ratios.append(ema_sharpe_ratio)
        bollinger_sharpe_ratios.append(bollinger_sharpe_ratio)
        stoch_sharpe_ratios.append(stoch_sharpe_ratio)

 
    # Calculate average profit and final price for each indicator
    average_sma_profit = sum(sma_profits) / len(sma_profits) if sma_profits else 0
    average_rsi_profit = sum(rsi_profits) / len(rsi_profits) if rsi_profits else 0
    average_ema_profit = sum(ema_profits) / len(ema_profits) if ema_profits else 0
    average_bollinger_profit = sum(bollinger_profits) / len(bollinger_profits) if bollinger_profits else 0
    average_stoch_profit = sum(stoch_profits) / len(stoch_profits) if stoch_profits else 0

    average_sma_final_price = sum(sma_final_prices) / len(sma_final_prices) if sma_final_prices else 0
    average_rsi_final_price = sum(rsi_final_prices) / len(rsi_final_prices) if rsi_final_prices else 0
    average_ema_final_price = sum(ema_final_prices) / len(ema_final_prices) if ema_final_prices else 0
    average_bollinger_final_price = sum(bollinger_final_prices) / len(bollinger_final_prices)   if bollinger_final_prices else 0
    average_stoch_final_price = sum(stoch_final_prices) / len(stoch_final_prices) if stoch_final_prices else 0

    average_sma_return_rate = sum(sma_return_rates) / len(sma_return_rates) if sma_return_rates else 0
    average_rsi_return_rate = sum(rsi_return_rates) / len(rsi_return_rates) if rsi_return_rates else 0
    average_ema_return_rate = sum(ema_return_rates) / len(ema_return_rates) if ema_return_rates else 0
    average_bollinger_return_rate = sum(bollinger_return_rates) / len(bollinger_return_rates) if bollinger_return_rates else 0
    average_stoch_return_rate = sum(stoch_return_rates) / len(stoch_return_rates) if stoch_return_rates else 0

    average_sma_sharpe_ratio = np.nanmean(sma_sharpe_ratios)
    average_rsi_sharpe_ratio = np.nanmean(rsi_sharpe_ratios)
    average_ema_sharpe_ratio = np.nanmean(ema_sharpe_ratios)
    average_bollinger_sharpe_ratio = np.nanmean(bollinger_sharpe_ratios)
    average_stoch_sharpe_ratio = np.nanmean(stoch_sharpe_ratios)

    print(f'{"Average SMA profit":50s}| {average_sma_profit:10.2f}')
    print(f'{"Average RSI profit":50s}| {average_rsi_profit:10.2f}')
    print(f'{"Average EMA profit":50s}| {average_ema_profit:10.2f}')
    print(f'{"Average Bollinger Bands profit":50s}| {average_bollinger_profit:10.2f}')
    print(f'{"Average Stochastic Oscillator profit":50s}| {average_stoch_profit:10.2f}\n')

    print(f'{"Average SMA final price":50s}| {average_sma_final_price:10.2f}')
    print(f'{"Average RSI final price":50s}| {average_rsi_final_price:10.2f}')
    print(f'{"Average EMA final price":50s}| {average_ema_final_price:10.2f}')
    print(f'{"Average Bollinger Bands final price":50s}| {average_bollinger_final_price:10.2f}')
    print(f'{"Average Stochastic Oscillator final price":50s}| {average_stoch_final_price:10.2f}\n')

    print(f'{"Average SMA return rate":50s}| {average_sma_return_rate:10.5f}')
    print(f'{"Average RSI return rate":50s}| {average_rsi_return_rate:10.5f}')
    print(f'{"Average EMA return rate":50s}| {average_ema_return_rate:10.5f}')
    print(f'{"Average Bollinger Bands return rate":50s}| {average_bollinger_return_rate:10.5f}')
    print(f'{"Average Stochastic Oscillator return rate":50s}| {average_stoch_return_rate:10.5f}\n')

    print(f'{"Average SMA Sharpe ratio":50s}| {average_sma_sharpe_ratio:10.2f}')
    print(f'{"Average RSI Sharpe ratio":50s}| {average_rsi_sharpe_ratio:10.2f}')
    print(f'{"Average EMA Sharpe ratio":50s}| {average_ema_sharpe_ratio:10.2f}')
    print(f'{"Average Bollinger Bands Sharpe ratio":50s}| {average_bollinger_sharpe_ratio:10.2f}')
    print(f'{"Average Stochastic Oscillator Sharpe ratio":50s}| {average_stoch_sharpe_ratio:10.2f}')

    # Python
    profits = [
        ("SMA", average_sma_profit),
        ("RSI", average_rsi_profit),
        ("EMA", average_ema_profit),
        ("Bollinger", average_bollinger_profit),
        ("Stochastic", average_stoch_profit)
    ]


    # Sort the list in descending order based on the profit, if the value < 0 remove it
    """ profits = [x for x in profits if x[1] > 0]
    profits.sort(key=lambda x: x[1], reverse=True) """

    # The first element in the sorted list is the best indicator
    # If profits.length > 0
    # profits [('SMA', -6.240677452087402), ('Bollinger', -6.240677452087402), ('RSI', -3.3538946151733398), ('Stochastic', 1.7565530776977538), ('EMA', 4.102328872680664)]

    # SORT BY PROFIT
    profits = sorted(profits, key=lambda x: x[1], reverse=True)
    if len(profits) > 0:
        best_indicator = profits[0]
    else:
        best_indicator = ("No suitable indicator found", 0)

    print("profits",profits)
    print("best_indicator",best_indicator)

    # write a for loop print i check the profits
    def backtest_profit(ticker_symbol,profits):
        backtest_profit_value =0
        backtest_final_price=0
        backtest_return=0
        backtest_shape_ratio=0
        backtest_indicator = ''
        for index, profit in enumerate(profits):
            if profit[0] == 'RSI':
                backtest_profit_value,backtest_final_price,backtest_return,backtest_shape_ratio = backtest_rsi(ticker_symbol)
                backtest_indicator = 'RSI'
                return backtest_profit_value,backtest_indicator
            elif profit[0] == 'SMA':
                backtest_profit_value,backtest_final_price,backtest_return,backtest_shape_ratioesult = backtest_sma(ticker_symbol)
                backtest_indicator = 'SMA'
                return backtest_profit_value,backtest_indicator
            elif profit[0] == 'EMA':
                backtest_profit_value,backtest_final_price,backtest_return,backtest_shape_ratio = backtest_ema(ticker_symbol)
                backtest_indicator = 'EMA'
                return backtest_profit_value,backtest_indicator
            elif profit[0] == 'Bollinger':
                backtest_profit_value,backtest_final_price,backtest_return,backtest_shape_ratio = backtest_bollinger(ticker_symbol)
                backtest_indicator = 'Bollinger'
                return backtest_profit_value,backtest_indicator
            elif profit[0] == 'Stochastic':
                backtest_profit_value,backtest_final_price,backtest_return,backtest_shape_ratio = backtest_stoch(ticker_symbol)
                backtest_indicator = 'Stochastic'
                return backtest_profit_value,backtest_indicator
        return backtest_profit_value,backtest_indicator
                          

    print(f"The best indicator is {best_indicator[0]} with a profit of {best_indicator[1]}")
    bactest_profit, backtest_indicator = backtest_profit(ticker_symbol,profits)
    print("bacltestprofit",bactest_profit) 

    backtest_stoch_profit,backtest_stoch_final_price,backtest_stoch_return,backtest_stoch_sharpe_ratio = backtest_stoch(ticker_symbol)
    print("backtest_stoch_profit",backtest_stoch_profit)
    print("backtest_stoch_final_price",backtest_stoch_final_price)
    print("backtest_stoch_return",backtest_stoch_return)
    print("backtest_stoch_sharpe_ratio",backtest_stoch_sharpe_ratio)

    backtest_sma_profit,backtest_sma_final_price,backtest_sma_return,backtest_sma_sharpe_ratio = backtest_sma(ticker_symbol)
    print("backtest_sma_profit",backtest_sma_profit)
    print("backtest_sma_final_price",backtest_sma_final_price)
    print("backtest_sma_return",backtest_sma_return)
    print("backtest_sma_sharpe_ratio",backtest_sma_sharpe_ratio)

    backtest_rsi_profit,backtest_rsi_final_price,backtest_rsi_return,backtest_rsi_sharpe_ratio = backtest_rsi(ticker_symbol)
    print("backtest_rsi_profit",backtest_rsi_profit)
    print("backtest_rsi_final_price",backtest_rsi_final_price)
    print("backtest_rsi_return",backtest_rsi_return)
    print("backtest_rsi_sharpe_ratio",backtest_rsi_sharpe_ratio)

    backtest_ema_profit,backtest_ema_final_price,backtest_ema_return,backtest_ema_sharpe_ratio = backtest_ema(ticker_symbol)
    print("backtest_ema_profit",backtest_ema_profit)
    print("backtest_ema_final_price",backtest_ema_final_price)
    print("backtest_ema_return",backtest_ema_return)
    print("backtest_ema_sharpe_ratio",backtest_ema_sharpe_ratio)

    backtest_bollinger_profit,backtest_bollinger_final_price,backtest_bollinger_return,backtest_bollinger_sharpe_ratio = backtest_bollinger(ticker_symbol)
    print("backtest_bollinger_profit",backtest_bollinger_profit)
    print("backtest_bollinger_final_price",backtest_bollinger_final_price)
    print("backtest_bollinger_return",backtest_bollinger_return)
    print("backtest_bollinger_sharpe_ratio",backtest_bollinger_sharpe_ratio)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Get data for the past year
    data = get_stock_data(ticker_symbol, start_date, end_date)

    rsi_window = 14
    sma_window = 20
    ema_window = 12
    bollinger_window = 20
    stoch_window = 14
    calibrate_indicators(data, rsi_window, sma_window, ema_window, bollinger_window, stoch_window)

    if backtest_indicator == 'Stochastic':
        buy_signals = data['Stochastic'] < 20
        sell_signals = data['Stochastic'] > 80
    elif backtest_indicator == 'SMA':
        short_window = min(5, len(data))  # Define short-term window size
        long_window = min(20, len(data))  # Define long-term window size

        # Calculate short-term and long-term moving averages
        data['Short_SMA'] = data['Close'].rolling(window=short_window).mean()
        data['Long_SMA'] = data['Close'].rolling(window=long_window).mean()

        # Print data to debug

        # Generate signals based on crossover strategy
        buy_signals = data['Short_SMA'] > data['Long_SMA']
        sell_signals = data['Short_SMA'] < data['Long_SMA']  
    elif backtest_indicator == 'RSI':
        buy_signals = data['RSI'] < 30
        sell_signals = data['RSI'] > 70
    elif backtest_indicator == 'EMA':
        short_window = 5  # Define  short-term window size
        long_window = 20  # Define  long-term window size

        # Calculate short-term and long-term EMAs
        data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()

        # Generate signals based on crossover strategy
        buy_signals = data['Short_EMA'] > data['Long_EMA']
        sell_signals = data['Short_EMA'] < data['Long_EMA']
    elif backtest_indicator == 'Bollinger':
        buy_signals = data['Close'] < data['Bollinger']  # Adjust these conditions as needed
        sell_signals = data['Close'] > data['Bollinger']

    buy_prices = data['Close'].where(buy_signals, None)
    sell_prices = data['Close'].where(sell_signals, None)

    plt.figure(figsize=(12,7))
    plt.plot(data['Close'], label='Close Price', color='blue', alpha=0.35)

    plt.scatter(data.index, buy_prices, color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(data.index, sell_prices, color='red', label='Sell Signal', marker='v', alpha=1)

    # Check if today is a buy or sell signal
    if buy_signals.iloc[-1]:
        plt.scatter(data.index[-1], data['Close'].iloc[-1], color='blue', label='Predict stock increase', marker='o', alpha=1)
        today_signal = 'increase'
    elif sell_signals.iloc[-1]:
        plt.scatter(data.index[-1], data['Close'].iloc[-1], color='purple', label='Predict stock decrease', marker='o', alpha=1)
        today_signal = 'decrease'
    else:
        plt.scatter(data.index[-1], data['Close'].iloc[-1], color='orange', label='Predict stock decrease', marker='o', alpha=1)
        today_signal = 'decrease'

    plt.title(f'{backtest_indicator}\n{ticker_symbol} Stock Price with Buy & Sell Signals\nToday ({end_date.strftime("%Y-%m-%d")}): Predict {today_signal} ')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.savefig('./static/predict.png')  # Save the plot as a PNG file
    #plt.show()
    results = {}
    results["SMA"] = {
        "Average SMA profit": average_sma_profit, 
        "Average SMA final price": average_sma_final_price, 
        "Average SMA return rate": average_sma_return_rate, 
        "Average SMA Sharpe ratio": average_sma_sharpe_ratio 
    }
    results["RSI"] = {
        "Average RSI profit": average_rsi_profit, 
        "Average RSI final price": average_rsi_final_price, 
        "Average RSI return rate": average_rsi_return_rate, 
        "Average RSI Sharpe ratio": average_rsi_sharpe_ratio 
    }
    
    results["EMA"] = {
        "Average EMA profit": average_ema_profit, 
        "Average EMA final price": average_ema_final_price, 
        "Average EMA return rate": average_ema_return_rate, 
        "Average EMA Sharpe ratio": average_ema_sharpe_ratio
    }

    results["Bollinger"] = {
        "Average Bollinger profit": average_bollinger_profit, 
        "Average Bollinger final price": average_bollinger_final_price, 
        "Average Bollinger return rate": average_bollinger_return_rate, 
        "Average Bollinger Sharpe ratio": average_bollinger_sharpe_ratio
    }

    results["Stochastic"] = {
        "Average Stochastic profit": average_stoch_profit, 
        "Average Stochastic final price": average_stoch_final_price, 
        "Average Stochastic return rate": average_stoch_return_rate, 
        "Average Stochastic Sharpe ratio": average_stoch_sharpe_ratio
    }

    results["backtest_SMA"] = {   
        "Backtest backtest_SMA profit": backtest_sma_profit,
        "Backtest backtest_SMA final price": backtest_sma_final_price,
        "Backtest backtest_SMA return rate": backtest_sma_return,
        "Backtest backtest_SMA Sharpe ratio": backtest_sma_sharpe_ratio
    }
   
    results["backtest_EMA"] = {   
        "Backtest backtest_EMA profit": backtest_ema_profit,
        "Backtest backtest_EMA final price": backtest_ema_final_price,
        "Backtest backtest_EMA return rate": backtest_ema_return,
        "Backtest backtest_EMA Sharpe ratio": backtest_ema_sharpe_ratio
    }

    results["backtest_RSI"] = {
        "Backtest backtest_RSI profit": backtest_rsi_profit,
        "Backtest backtest_RSI final price": backtest_rsi_final_price,
        "Backtest backtest_RSI return rate": backtest_rsi_return,
        "Backtest backtest_RSI Sharpe ratio": backtest_rsi_sharpe_ratio
    }
    
    results["backtest_Bollinger"] = {
        "Backtest backtest_Bollinger profit": backtest_bollinger_profit,
        "Backtest backtest_Bollinger final price": backtest_bollinger_final_price,
        "Backtest backtest_Bollinger return rate": backtest_bollinger_return,
        "Backtest backtest_Bollinger Sharpe ratio": backtest_bollinger_sharpe_ratio
    }

    results["backtest_Stochastic"] = {
        "Backtest backtest_Stochastic profit": backtest_stoch_profit,
        "Backtest backtest_Stochastic final price": backtest_stoch_final_price,
        "Backtest backtest_Stochastic return rate": backtest_stoch_return,
        "Backtest backtest_Stochastic Sharpe ratio": backtest_stoch_sharpe_ratio
    }

    backtest_profit_avg = (backtest_sma_profit + backtest_ema_profit + backtest_rsi_profit + backtest_bollinger_profit + backtest_stoch_profit) / 5
    peredicted_best_indicator = best_indicator[0]
    predicted_best_profit = best_indicator[1]
    
    results["backtest_performance"] = {
        "Backtest profit average": backtest_profit_avg,
        "Peredicted best indicator": peredicted_best_indicator,
        "Predicted best profit": predicted_best_profit
    }

    results["Image"] = "predict.png"
    
    
    return results
