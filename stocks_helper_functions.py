import pandas as pd
import numpy as np

class stocks_helper_class:
    def __init__(self):
        pass
    def classify_candlestick(self,row):
        if row['Close'] > row['Open']:
            if row['High'] == row['Close'] and row['Low'] == row['Open']:
                return 'Hammer'
            elif row['High'] == row['Close'] and row['Low'] < row['Open']:
                return 'Bullish Engulfing'
            elif row['High'] > row['Close'] and row['Low'] == row['Open']:
                return 'Bullish Harami'
            elif row['High'] > row['Close'] and row['Low'] < row['Open']:
                return 'Bullish Marubozu'
            else:
                return 'Bullish Candlestick'
        elif row['Close'] < row['Open']:
            if row['High'] == row['Open'] and row['Low'] == row['Close']:
                return 'Inverted Hammer'
            elif row['High'] == row['Open'] and row['Low'] > row['Close']:
                return 'Bearish Engulfing'
            elif row['High'] < row['Open'] and row['Low'] == row['Close']:
                return 'Bearish Harami'
            elif row['High'] < row['Open'] and row['Low'] > row['Close']:
                return 'Bearish Marubozu'
            else:
                return 'Bearish Candlestick'
        else:
            return 'Doji'
