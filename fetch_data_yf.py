import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
import calendar

#define fetch_data class

class fetch_data_yf:
    def __init__(self):
        pass
# define function to fetch data from yfinance and return a dtaframe df_full 

    def fetch_stock_data(self,symbols, start_date, end_date, time_frame):
        df_full = []

        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, interval=time_frame)
                data['Symbol'] = symbol
                data.reset_index(inplace=True)
                #data['Year'] = data['Date'].dt.year
                #data['Month'] = data['Date'].dt.month
                #data['Day'] = data['Date'].dt.day
                #data['Day_of_Week'] = data['Date'].dt.day_name()
                #Determine if the date corresponds to an expiry day (monthly or weekly)
                #data['Expiry_Day'] = data.apply(lambda row: is_expiry_day(row['Date']), axis=1)
                df_full.append(data)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")

        return df_full