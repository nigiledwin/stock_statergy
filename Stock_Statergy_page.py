import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import json
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
from fetch_data_yf import fetch_data_yf
from options_functions import options_functions_class

class Stocks_statergy_page_class:
    def __init__(self):
        pass

        # Function to display Nifty and Bank Nifty analysis
    def ema_func(self):         
        interval = st.sidebar.selectbox('Select interval:', ['15m', '30m', '1h','1d'])
        # Specify start and end dates for historical data
        no_backdays = st.sidebar.slider('Select number of past days:', 1, 300, 50)
        start_date = datetime.now() - timedelta(days=no_backdays)
        end_date = datetime.now()

        # User input for short and long EMA windows
        short_window = st.sidebar.selectbox('Select short EMA window:',[20, 50])
        long_window = st.sidebar.selectbox('Select long EMA window:',[50, 200])
        no_backcandle=st.sidebar.selectbox('Select no candles to check for signal:',[5,10, 50])

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
            #importing fetch_stock_data function  
            from fetch_data_yf import fetch_data_yf
            from stocks_helper_functions import stocks_helper_class
            fetch_data=fetch_data_yf()
            candle_stick_func=stocks_helper_class()

            df_full = fetch_data.fetch_stock_data(stock_index, start_date, end_date, interval)
            
            
            # Apply moving average strategy
            for tickers in df_full:
                tickers['SMA_short'] = tickers['Close'].rolling(window=short_window).mean()
                tickers['SMA_long'] = tickers['Close'].rolling(window=long_window).mean()
                tickers['previous_SMA_Long']=tickers['SMA_long'].shift(1)
                tickers.dropna(inplace=True)
                # Calculate MACD
                tickers['ema_12'] = tickers['Close'].ewm(span=12, min_periods=0, adjust=True).mean()
                tickers['ema_26'] = tickers['Close'].ewm(span=26, min_periods=0, adjust=True).mean()
                tickers['MACD'] = tickers['ema_12'] - tickers['ema_26']
                tickers['Signal_line'] = tickers['MACD'].ewm(span=9, min_periods=0, adjust=True).mean()
                tickers['Histogram'] = tickers['MACD'] - tickers['Signal_line']
                tickers['previous_MACD'] = tickers['MACD'].shift(1)
# create candlestick pattern to dataframe
                tickers['candlestick_pattern_type'] = tickers.apply(lambda row: candle_stick_func.classify_candlestick(row), axis=1)

            # Apply buy_sell_signal to dataframe based on EMA
            def buy_sell_signals_ema(SMA_short, SMA_long, previous_SMA_Long):
                if SMA_long > SMA_short and previous_SMA_Long < SMA_short:
                    return 'Bullish Crossover'
                elif SMA_long < SMA_short and previous_SMA_Long > SMA_short:
                    return 'Bearish Crossover'
                else:
                    return 'None'
            def MACD_buy_sell(MACD,Signal_line,previous_MACD):
                if previous_MACD < Signal_line and MACD > Signal_line:
                    return 'Bullish MACD Crossover'
                if previous_MACD > Signal_line and MACD < Signal_line:
                    return 'Bearish MACD Crossover'

            # Apply buy_sell_signal to dataframe based on EMA
            for symbol_df in df_full:
                if not symbol_df.empty:
                    symbol_df['Signal_Type'] = np.vectorize(buy_sell_signals_ema)(symbol_df['SMA_short'], symbol_df['SMA_long'], symbol_df['previous_SMA_Long'])
                else:
                    print("DataFrame is empty or does not contain valid data.")
            # Apply MACD_buy_sell to dataframe based on MACD
            '''for symbol_df in df_full:
                if not symbol_df.empty:
                    symbol_df['Signal_Type_MACD'] = np.vectorize(MACD_buy_sell)(symbol_df['MACD'], symbol_df['Signal_line'], symbol_df['previous_MACD'])
                else:
                    print("DataFrame is empty or does not contain valid data.")'''
            # Dataframe to store buy signals
            df_buy=pd.DataFrame()
            for symbol in range(0,len(df_full)):
            #buy_position=df_full[symbol].tail(5)['Position'].values
                if 'Bullish Crossover' in df_full[symbol].tail(no_backcandle)['Signal_Type'].values:
                    df=pd.DataFrame({symbol:[df_full[symbol]['Symbol'].mode()]})
                    df_buy = pd.concat([df_buy, df], axis=1)   

            # Dataframe to store sell signals
            df_sell=pd.DataFrame()
            for symbol in range(0,len(df_full)):
            #buy_position=df_full[symbol].tail(50)['Position'].values
                if 'Bearish Crossover' in df_full[symbol].tail(no_backcandle)['Signal_Type'].values:                   
                    df=pd.DataFrame({symbol:[df_full[symbol]['Symbol'].mode()]})
                    df_sell = pd.concat([df_sell, df], axis=1)
            #st.write(df_full[0][df_full[0]['Signal_Type_MACD']=='Bullish MACD Crossover'])            
            df_buy=df_buy.T 
            df_sell=df_sell.T 


            # Display buy signals dataframe
            st.write(df_full[1])
            st.write(df_buy)
            st.write(df_sell)


            # Candlestick chart with SMA
            for i in df_buy.index:
                fig = go.Figure()
                no_candle_plot=100
                fig.add_trace(go.Candlestick(x=df_full[i].tail(no_candle_plot).index, open=df_full[i].tail(no_candle_plot)['Open'], high=df_full[i].tail(no_candle_plot)['High'],
                                            low=df_full[i].tail(no_candle_plot)['Low'], close=df_full[i].tail(no_candle_plot)['Close']))
                fig.add_trace(go.Scatter(x=df_full[i].tail(no_candle_plot).index, y=df_full[i].tail(no_candle_plot)['SMA_short'], mode='lines', name='Short EMA',
                                        line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=df_full[i].tail(no_candle_plot).tail(no_candle_plot).index, y=df_full[i].tail(no_candle_plot)['SMA_long'], mode='lines', name='Long EMA',
                                        line=dict(color='red', width=2)))
                buy_signals = df_full[i].tail(no_candle_plot)[df_full[i]['Signal_Type'] == 'Bullish Crossover']
                sell_signals = df_full[i].tail(no_candle_plot)[df_full[i]['Signal_Type'] == 'Bearish Crossover']
                fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['SMA_short'], mode='markers',
                                        marker=dict(color='green', size=10), name='Buy Signal'))
                fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['SMA_short'], mode='markers',
                                        marker=dict(color='yellow', size=10), name='Sell Signal'))
                fig.update_layout(title=f'Candlestick Chart with SMA for {df_full[i]["Symbol"].iloc[0]}',
                                xaxis_title='Date', yaxis_title='Price')

                st.plotly_chart(fig)

                '''fig1 = go.Figure()

                fig1.add_trace(go.Scatter(x=df_full[i].index, y=df_full[i]['MACD'], mode='lines', name='MACD',
                                        line=dict(color='black', width=2)))
                fig1.add_trace(go.Scatter(x=df_full[i].index, y=df_full[i]['Signal_line'], mode='lines', name='Signal_line',
                                        line=dict(color='blue', width=2)))
                positive_histogram = df_full[i]['Histogram'].apply(lambda x: x if x > 0 else 0)
                negative_histogram = df_full[i]['Histogram'].apply(lambda x: x if x < 0 else 0)

                fig1.add_trace(go.Bar(x=df_full[i].index, y=positive_histogram, name='Signal_line',
                                        marker=dict(color='green'), width=2))       
                fig1.add_trace(go.Bar(x=df_full[i].index, y=negative_histogram, name='Signal_line',
                                        marker=dict(color='red'), width=2))           

                buy_signals_MACD = df_full[i][df_full[i]['Signal_Type_MACD'] == 'Bullish MACD Crossover']
                sell_signals_MACD = df_full[i][df_full[i]['Signal_Type_MACD'] == 'Bearish MACD Crossover']
                fig1.add_trace(go.Scatter(x=buy_signals_MACD.index, y=buy_signals_MACD['MACD'], mode='markers',
                                        marker=dict(color='green', size=5), name='Buy Signal'))
                fig1.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['MACD'], mode='markers',
                                        marker=dict(color='yellow', size=5), name='Sell Signal'))
                fig1.update_layout(title=f'MACD Chart with SMA for {df_full[i]["Symbol"].iloc[0]}',
                                xaxis_title='Date', yaxis_title='Price')


                # Calculate y-axis range based on latest 100 close prices
                latest_100_close_prices = df_full[i]['MACD'].tail(500)
                y_range = [latest_100_close_prices.min() - 10, latest_100_close_prices.max() + 10]
                fig1.update_layout(yaxis=dict(range=y_range))
                fig1.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                fig1.update_layout(width=700, height=500)
                st.write(df_full[i][df_full[i]['Signal_Type_MACD']=='Bullish MACD Crossover'])   

                st.plotly_chart(fig1)'''
                
        '''elif Statergy == 'Super Trend':
            def get_stock_data(symbols, start_date, end_date, time_frame):
                df_full = []

                for symbol in symbols:
                    try:
                        data = yf.download(symbol, start=start_date, end=end_date, interval=time_frame)
                        data['Symbol'] = symbol
                        data.reset_index(inplace=True)
                        ##data['Year'] = data['Date'].dt.year
                        #data['Month'] = data['Date'].dt.month
                        #data['Day'] = data['Date'].dt.day
                        #data['Day_of_Week'] = data['Date'].dt.day_name()
                        # Determine if the date corresponds to an expiry day (monthly or weekly)
                        #data['Expiry_Day'] = data.apply(lambda row: is_expiry_day(row['Date']), axis=1)
                        df_full.append(data)
                    except Exception as e:
                        print(f"Error fetching data for {symbol}: {str(e)}")

                return df_full
            interval = st.sidebar.selectbox('Select interval:', ['1m', '5m', '15m', '30m', '1h','1d'])
            df_full = get_stock_data(['ADANIPORTS.BO', 'ASIANPAINT.BO'] , start_date = datetime.now() - timedelta(days=5), end_date = datetime.now(),time_frame=interval)
            st.write('df_full')
            st.write(df_full)
        elif Statergy == 'Breakout_signal':
            None'''
    def breakout_func(self):         
        interval = st.sidebar.selectbox('Select interval:', ['15m', '30m', '1h','1d'])
        # Specify start and end dates for historical data
        no_backdays = st.sidebar.slider('Select number of past days:', 1, 300, 50)
        start_date = datetime.now() - timedelta(days=no_backdays)
        end_date = datetime.now()
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
            #importing fetch_stock_data function  
            from fetch_data_yf import fetch_data_yf
            from stocks_helper_functions import stocks_helper_class
            fetch_data=fetch_data_yf()
            candle_stick_func=stocks_helper_class()

            df_full = fetch_data.fetch_stock_data(stock_index, start_date, end_date, interval)
            return df_full[1]