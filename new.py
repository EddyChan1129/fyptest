import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

def download_stock_data(ticker, start='2010-01-01', end='2023-03-01'):
    """
    Download stock data for the specified ticker between start and end dates.
    """
    data = yf.download(ticker, start=start, end=end)
    return data['Adj Close']  # Focus on adjusted close prices

def compute_similarity_scores(current_data, past_data_segments):
    """
    Compute similarity scores between the current data and each past data segment.
    Ensures comparison is made between segments of equal length.
    """
    scaler = MinMaxScaler()
    scores = []
    
    # Find the minimum length to ensure all comparisons are of the same length
    min_length = min(len(current_data), min(len(segment) for segment in past_data_segments))
    
    # If current data is longer than the minimum length, trim it
    if len(current_data) > min_length:
        current_data = current_data[:min_length]
    
    current_data_scaled = scaler.fit_transform(current_data.reshape(-1, 1)).flatten()

    for segment in past_data_segments:
        # Trim or extend the segment to match the minimum length
        segment = segment[:min_length]
        segment_scaled = scaler.transform(segment.reshape(-1, 1)).flatten()
        score = euclidean(current_data_scaled, segment_scaled)
        scores.append(score)
    
    return scores

def segment_past_data(data, segment_length_years=2, step_months=3):
    """
    Segment the past data into 2-year periods with a step of 3 months.
    """
    segments = []
    start_dates = []

    for i in range(0, len(data) - segment_length_years * 365, step_months * 30):  # Approximate days
        start_date = data.index[i]
        end_date = start_date + pd.DateOffset(years=segment_length_years)
        if end_date > data.index[-1]:
            break
        segment = data.loc[start_date:end_date]
        segments.append(segment.values)
        start_dates.append(start_date)
        
    return segments, start_dates

# Main analysis function
def find_similar_periods(ticker):
    data = download_stock_data(ticker)
    current_period_data = data[-2*365:].values  # Last 2 years for the current period

    past_data = data[:-2*365]  # Exclude the last 2 years to avoid overlap
    past_data_segments, start_dates = segment_past_data(past_data)

    scores = compute_similarity_scores(current_period_data, past_data_segments)
    sorted_indices = np.argsort(scores)

    print("Top 5 most similar past periods (start date):")
    for index in sorted_indices[:5]:
        print(f"{start_dates[index].date()} to {(start_dates[index] + pd.DateOffset(years=2)).date()}, Similarity Score: {scores[index]}")

# Example usage
ticker = 'AAPL'  # Example ticker
find_similar_periods(ticker)
