#df = pd.read_csv("/content/sample_data/EURUSD_Candlestick_1_D_BID_04.05.2003-21.01.2023.csv")
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

def get_stock_data(symbols, start_date, end_date,time_frame):
    df_full =[]

    for symbol in symbols:
        try:
            #stock = yf.Ticker(symbol)
            data = yf.download(symbol,start=start_date, end=end_date, interval=time_frame)

            # Add a 'Symbol' column to the DataFrame
            data['Symbol'] = symbol
            data.reset_index(inplace=True)

            # Add data to list
            df_full.append(data)

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

    return df_full

symbol_list = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
               'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
               'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GRASIM.NS',
               'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS',
               'HINDUNILVR.NS', 'ITC.NS', 'ICICIBANK.NS', 'IBULHSGFIN.NS', 'IOC.NS',
               'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
               'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS',
               'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'VEDL.NS',
               'WIPRO.NS', 'YESBANK.NS', 'ZEEL.NS']

# Specify start and end dates for historical data
no_backdays=3
start_date = datetime.now() - timedelta(days=no_backdays)
end_date = datetime.now()
time_frame='5m'
df_full = get_stock_data(symbol_list, start_date, end_date,time_frame)

#moving average statergy
def moving_avg_crossover(data,short_window,long_window):
  data['SMA_short']=data['Close'].rolling(window=short_window).mean()
  data['SMA_long']=data['Close'].rolling(window=long_window).mean()
  data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0)
  data['Position'] = data['Signal'].diff()

  #applying moving average statergy to all dataframes
short_window=9
long_window=21
for symbol in range(0,len(symbol_list)):
  moving_avg_crossover(df_full[symbol],short_window,long_window)

df_buy=pd.DataFrame()
for symbol in range(0,len(symbol_list)):
  buy_position=df_full[symbol].tail(5)['Position'].values
  if df_full[symbol].tail(5)['Position'].values.sum()<0:
  #array_name = f'array{i+1}'
    print(symbol)
    df=pd.DataFrame({symbol:[df_full[symbol]['Symbol'].mode()]})
    df_buy = pd.concat([df_buy, df], axis=1)



#candlestick chart with SMA
for i in df_buy.index:
  fig = go.Figure()
  fig.add_trace(go.Candlestick(x=df_full[i].index,
                open=df_full[i]['Open'],
                high=df_full[i]['High'],
                low=df_full[i]['Low'],
                close=df_full[i]['Close']))
  fig.add_trace(go.Scatter(x=df_full[i].index,y=df_full[i]['SMA_short'],mode='lines', name='9-EMA', line=dict(color='blue', width=2)))
  fig.add_trace(go.Scatter(x=df_full[i].index,y=df_full[i]['SMA_long'],mode='lines', name='20-EMA', line=dict(color='red', width=2)))
  #fig.update_layout(title=f'{ticker} Candlestick Chart with 20-day EMA',xaxis_title='Date',yaxis_title='Price')
  buy_signals = df_full[i][df_full[i]['Position'] == 1]
  sell_signals = df_full[i][df_full[i]['Position'] == -1]
  fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['SMA_short'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
  fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['SMA_short'], mode='markers', marker=dict(color='yellow', size=10), name='sell Signal'))
  fig.update_layout(title=f'Candlestick Chart with SMA for {i}',
                      xaxis_title='Date',
                      yaxis_title='Price')
  fig.show()