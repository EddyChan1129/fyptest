from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import ta
import numpy as np

pd.set_option('display.max_rows', None)


app = Flask(__name__)
CORS(app)

def get_stock_data(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    return data



def calculate_technical_indicators(data):
    """
    Calculates technical indicators for the given stock data.
    """
    # Calculate Moving Averages
    data['MA50'] = ta.trend.sma_indicator(data['Close'], window=50)
    print("data['MA50']\n", data['MA50'])

    data['MA200'] = ta.trend.sma_indicator(data['Close'], window=200)
    print("data['MA200']\n", data['MA200'])
    # Calculate RSI
    data['RSI'] = ta.momentum.rsi(data['Close'])
    print("data['RSI']\n", data['RSI'])

    # Calculate MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()

    return data


def rank_indicators(data):
    """
    Ranks indicators based on their win rate.
    """
    # Define thresholds for RSI
    rsi_buy_threshold = 30
    rsi_sell_threshold = 70
    # Create signal columns and determine if the signal was successful
    data['MA_Success'] = ((data['MA50'] > data['MA200']) == (
        data['Close'].shift(-1) > data['Close'])).astype(int)

    data['RSI_Success'] = (((data['RSI'] < rsi_buy_threshold) == (data['Close'].shift(-1) > data['Close'])) |
                           ((data['RSI'] > rsi_sell_threshold) == (data['Close'].shift(-1) < data['Close']))).astype(int)
                           
    data['MACD_Success'] = ((data['MACD'] > data['MACD_Signal']) == (
        data['Close'].shift(-1) > data['Close'])).astype(int)
    # Calculate win rate for each indicator
    performance = {
        'MA': data['MA_Success'].mean(),
        'RSI': data['RSI_Success'].mean(),
        'MACD': data['MACD_Success'].mean(),

    }

    # Rank indicators by win rate
    ranked_indicators = sorted(
        performance.items(), key=lambda x: x[1], reverse=True)

    return ranked_indicators


@app.route('/analyze', methods=['GET'])
def analyze_stock():
    ticker = request.args.get('ticker')
    data = get_stock_data(ticker)

    
    data_with_indicators = calculate_technical_indicators(data)
    ranked = rank_indicators(data_with_indicators) 
    return jsonify(ranked)


if __name__ == '__main__':
    app.run(debug=True,port=5500)
