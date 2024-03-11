import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fetch_data_yf import fetch_data_yf

NSE = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
    'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GRASIM.NS',
    'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'ICICIBANK.NS', 'IBULHSGFIN.NS', 'IOC.NS',
    'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
    'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'VEDL.NS',
    'WIPRO.NS', 'YESBANK.NS', 'ZEEL.NS']  # Add more if needed

BSE = ['ADANIPORTS.BO', 'ASIANPAINT.BO', 'AXISBANK.BO', 'BAJAJ-AUTO.BO',
    'BAJFINANCE.BO', 'BAJAJFINSV.BO', 'BPCL.BO', 'BHARTIARTL.BO',]  # Add more if needed


start_date = datetime.now() - timedelta(days=5000)
end_date = datetime.now()
interval = '1d'
stock_lists = {'NSE': NSE, 'BSE': BSE}

for market, stocks in stock_lists.items():

    # Initialize fetch_data_yf object
    fetch_data = fetch_data_yf()

    # Fetch data for each stock and concatenate into one DataFrame
    dfs = []
    for stock in stocks:
        df = fetch_data.fetch_stock_data([stock], start_date, end_date, interval)
        dfs.append(pd.concat(df))  # Concatenate each DataFrame and append it to the list

    df= pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames

    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Write DataFrame to CSV with current date and time in the file name
    csv_file_name = f"stock_data_{market}_{interval}.csv"
    df.to_csv(csv_file_name, index=False)
