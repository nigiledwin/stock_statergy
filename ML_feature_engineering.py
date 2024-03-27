import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class feature_engg_class:
    def __init__(self):
        pass

    def feature_engg_func(self,df):

        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        #adding new colum with next day close value
        df['Close_Next_Day'] = df['Close'].shift(-1)
        # Calculate some common technical indicators

        # Moving Averages
        df['MA10'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
        df['MA50'] = df['Close'].rolling(window=50).mean()  # 50-day moving average

        # Exponential Moving Averages
        df['EMA10'] = df['Close'].ewm(span=10, min_periods=0, adjust=False).mean()  # 10-day EMA
        df['EMA50'] = df['Close'].ewm(span=50, min_periods=0, adjust=False).mean()  # 50-day EMA

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        df['EMA12'] = df['Close'].ewm(span=12, min_periods=0, adjust=False).mean()  # 12-day EMA
        df['EMA26'] = df['Close'].ewm(span=26, min_periods=0, adjust=False).mean()  # 26-day EMA
        df['MACD'] = df['EMA12'] - df['EMA26']

        # Bollinger Bands
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['20STD'] = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['20MA'] + (df['20STD'] * 2)
        df['LowerBand'] = df['20MA'] - (df['20STD'] * 2)

        # Drop rows with missing values (NaN)
        df.dropna(inplace=True)

        return df
df_nse = pd.read_csv('stock_data_NSE_1d.csv')
df_ADANIPORTS=df_nse[df_nse['Symbol']=='ADANIPORTS.NS']

feature_df=feature_engg_class()
df_ADANIPORTS=feature_df.feature_engg_func(df_ADANIPORTS)
print(df_ADANIPORTS)




