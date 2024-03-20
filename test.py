import yfinance as yf
import pandas as pd
from sklearn.metrics import mean_squared_error

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

def calculate_similarity(current_data, past_data):
    min_length = min(len(current_data), len(past_data))
    current_data = current_data[:min_length]
    past_data = past_data[:min_length]
    
    return mean_squared_error(current_data, past_data)

def find_similar_periods(ticker, current_start_date, current_end_date, step_months):
    current_data = get_stock_data(ticker, current_start_date, current_end_date)
    
    similar_periods = {}
    
    for year in range(current_start_date.year - 10, current_end_date.year):
        for month in range(1, 13 - step_months + 1, step_months):
            start_date = pd.Timestamp(year=year, month=month, day=1)
            end_date = start_date + pd.DateOffset(months=step_months) - pd.DateOffset(days=1)
            
            if end_date > current_end_date:
                break
            
            past_data = get_stock_data(ticker, start_date, end_date)
            similarity_score = calculate_similarity(current_data, past_data)
            
            similar_periods[(start_date, end_date)] = similarity_score
    
    ranked_periods = sorted(similar_periods.items(), key=lambda x: x[1])
    
    most_similar_period = ranked_periods[0][0]
    
    return most_similar_period

# Example usage
ticker_symbol = '0002.hk'
current_start = pd.Timestamp(year=2024, month=1, day=1)
current_end = pd.Timestamp(year=2024, month=3, day=31)
step_movement_months = 3

most_similar_period = find_similar_periods(ticker_symbol, current_start, current_end, step_movement_months)
print("The most similar 3-month period is:", most_similar_period)

