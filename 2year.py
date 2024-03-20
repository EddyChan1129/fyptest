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

def find_similar_periods(ticker, current_start_date, current_end_date, top_n=5):
    current_data = get_stock_data(ticker, current_start_date, current_end_date)
    
    similar_periods = {}
    
    for year in range(current_start_date.year - 20, current_end_date.year):
        start_date = pd.Timestamp(year=year, month=1, day=1)
        end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)
        
        if end_date > current_end_date:
            continue
        
        past_data = get_stock_data(ticker, start_date, end_date)
        similarity_score = calculate_similarity(current_data, past_data)
        
        similar_periods[(start_date, end_date)] = similarity_score
    
    ranked_periods = sorted(similar_periods.items(), key=lambda x: x[1])

    return ranked_periods[:top_n]

# Example usage
ticker_symbol = '0001.hk'
current_start = pd.Timestamp(year=2023, month=3, day=19)
current_end = pd.Timestamp(year=2024, month=3, day=19)

similar_periods = find_similar_periods(ticker_symbol, current_start, current_end)
for i, period in enumerate(similar_periods, 1):
    print(f"The {i}{'' if i == 0 else 'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} similar year period is:", period[0])
