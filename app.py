import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Function to get stock data
def get_stock_data(symbols, start_date, end_date, time_frame):
    df_full = []

    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=time_frame)
            data['Symbol'] = symbol
            data.reset_index(inplace=True)
            df_full.append(data)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

    return df_full

# Function for moving average crossover strategy
def moving_avg_crossover(data, short_window, long_window):
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0)
    data['Position'] = data['Signal'].diff()
    # Calculate MACD
    data['ema_12'] = data['Close'].ewm(span=12, min_periods=0, adjust=True).mean()
    data['ema_26'] = data['Close'].ewm(span=26, min_periods=0, adjust=True).mean()
    data['MACD'] = data['ema_12'] - data['ema_26']
    data['Signal_line'] = data['MACD'].ewm(span=9, min_periods=0, adjust=True).mean()
    data['Histogram'] = data['MACD'] - data['Signal_line']

# Streamlit app
st.title('Stock Candlestick Chart with Moving Averages')

# Sidebar layout
st.sidebar.title('Input Parameters')

# User input for interval
interval = st.sidebar.selectbox('Select interval:', ['1m', '5m', '15m', '30m', '1h'])

# Specify start and end dates for historical data
no_backdays = st.sidebar.slider('Select number of past days:', 1, 30, 3)
start_date = datetime.now() - timedelta(days=no_backdays)
end_date = datetime.now()

# User input for short and long EMA windows
short_window = st.sidebar.selectbox('Select short EMA window:',[10, 20, 50])
long_window = st.sidebar.selectbox('Select long EMA window:',[50, 200])

# Get stock data
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

# User input for Stock Indices
selected_index = st.sidebar.selectbox('Select Stock Index:', ['NSE', 'BSE']) # add more if required

if selected_index == 'NSE':
    stock_index = NSE
else:
    stock_index = BSE

# Submit button
if st.sidebar.button('Submit'):
    df_full = get_stock_data(stock_index, start_date, end_date, interval)

    # Apply moving average strategy
    for symbol_df in df_full:
        moving_avg_crossover(symbol_df, short_window, long_window)

    # Dataframe to store buy signals
    df_buy = pd.DataFrame()

    # Find buy signals
    for i, symbol_df in enumerate(df_full):
        buy_position = symbol_df.tail(5)['Position'].values
        if buy_position.sum() < 0:
            df_buy[i] = symbol_df['Symbol'].mode()
    df_buy = df_buy.T

    # Display buy signals dataframe
    st.write('Buy Signals:')
    st.write(df_buy)

    # Candlestick chart with SMA
    for i in df_buy.index:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_full[i].index, open=df_full[i]['Open'], high=df_full[i]['High'],
                                      low=df_full[i]['Low'], close=df_full[i]['Close']))
        fig.add_trace(go.Scatter(x=df_full[i].index, y=df_full[i]['SMA_short'], mode='lines', name='Short EMA',
                                 line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=df_full[i].index, y=df_full[i]['SMA_long'], mode='lines', name='Long EMA',
                                 line=dict(color='red', width=2)))
        buy_signals = df_full[i][df_full[i]['Position'] == 1]
        sell_signals = df_full[i][df_full[i]['Position'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['SMA_short'], mode='markers',
                                 marker=dict(color='green', size=10), name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['SMA_short'], mode='markers',
                                 marker=dict(color='yellow', size=10), name='Sell Signal'))
        fig.update_layout(title=f'Candlestick Chart with SMA for {df_full[i]["Symbol"].iloc[0]}',
                          xaxis_title='Date', yaxis_title='Price')


        # Calculate y-axis range based on latest 100 close prices
        latest_100_close_prices = df_full[i]['Close'].tail(100)
        y_range = [latest_100_close_prices.min() - 10, latest_100_close_prices.max() + 10]
        fig.update_layout(yaxis=dict(range=y_range))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        fig.update_layout(width=700, height=700)

        st.plotly_chart(fig)