import yfinance as yf
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

def find_similar_3month_period(ticker):
    # Download 10 years + 3 months of daily stock data
    data = yf.download(ticker, period="123m")  # 120 months for 10 years and 3 months for comparison
    
    # Ensure data is sorted by date
    data.sort_index(inplace=True)
    
    # Normalize the data to make the comparison scale-invariant
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1)).flatten()
    
    # Get the most recent 3 months of data for comparison
    recent_3_months = data_scaled[-63:]  # Assuming 21 trading days per month
    
    min_distance = float('inf')
    most_similar_period_start = None
    
    # Iterate over 10-year period in 3-month windows
    for i in range(len(data_scaled) - 63):
        historical_3_months = data_scaled[i:i+63]
        distance = euclidean(historical_3_months, recent_3_months)
        
        if distance < min_distance:
            min_distance = distance
            most_similar_period_start = data.index[i]
            
    most_similar_period_end = most_similar_period_start + pd.Timedelta(days=63)
    
    return most_similar_period_start, most_similar_period_end, min_distance

# Example usage
# Example usage
ticker = "AAPL"
most_similar_start, most_similar_end, similarity_score = find_similar_3month_period(ticker)

if most_similar_start is not None and most_similar_end is not None:
    print(f"Most similar 3-month period starts on {most_similar_start.date()} and ends on {most_similar_end.date()}, with a similarity score of {similarity_score:.2f}")
else:
    print("Could not find a similar 3-month period.")

print(f"Most similar 3-month period starts on {most_similar_start.date()} and ends on {most_similar_end.date()}, with a similarity score of {similarity_score:.2f}")
