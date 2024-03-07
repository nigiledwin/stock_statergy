import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import calendar

# Function to determine if a given date is an expiry day
def is_expiry_day(date):
    # Check if the date is the last Thursday of the month (monthly expiry)
    if date.weekday() == 2 and date.day > (calendar.monthrange(date.year, date.month)[1] - 7):
        return 'Monthly Expiry'
    # Check if the date is a Friday (weekly expiry for indices)
    elif date.weekday() == 2:
        return 'Weekly Expiry'
    else:
        return 'Not Expiry Day'
# Function determine the gap up basket based on its values    
def gap_per(gap):
    if gap>1000:
        return '1000 an more'
    elif gap<1000 and gap >750 :
        return '750-1000'
    elif gap<750 and gap >500 :
        return '500-750'
    elif gap<500 and gap >3000 :
        return '300-500'
    elif gap<300 and gap >100 :
        return '100-300'
    elif gap<100 and gap >50 :
        return '50-100'
    elif gap<50 and gap >-50 :
        return '-50-50'
    elif gap<-50 and gap >-100 :
        return 'Negative 50-100'
    elif gap<-100 and gap >-300 :
        return 'Negative 100-300'
    elif gap<-300 and gap >-500 :
        return 'Negative -300-500'
    elif gap<-500 and gap >-750 :
        return 'Negative 500-750'
    elif gap<-750 and gap >-1000:
        return 'Negative -750-1000'
    elif gap<-1000:
        return 'More than -1000'
    
# Function for moving average crossover strategy
def moving_avg_crossover(data, short_window, long_window):
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    data['previous_SMA_Long']=data['SMA_long'].shift(1)
    def buy_sell_signals(row):
        if row['SMA_long'] > row['SMA_short'] and row['previous_SMA_Long'] < row['SMA_short']:
            return 'Bullish Crossover'
        elif row['SMA_long'] < row['SMA_short'] and row['previous_SMA_Long'] > row['SMA_short']:
            return 'Bearish Crossover' 
        else:
            return None  
    
    data['Signal_Type'] = data.apply(buy_sell_signals, axis=1)
    # Calculate MACD
    data['ema_12'] = data['Close'].ewm(span=12, min_periods=0, adjust=True).mean()
    data['ema_26'] = data['Close'].ewm(span=26, min_periods=0, adjust=True).mean()
    data['MACD'] = data['ema_12'] - data['ema_26']
    data['Signal_line'] = data['MACD'].ewm(span=9, min_periods=0, adjust=True).mean()
    data['Histogram'] = data['MACD'] - data['Signal_line']
    data['Date_only'] = pd.to_datetime(data['Date']).dt.date
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Day_of_Week'] = data['Date'].dt.day_name()
    #Determine if the date corresponds to an expiry day (monthly or weekly)
    data['Expiry_Day'] = data.apply(lambda row: is_expiry_day(row['Date']), axis=1)


# Function to display Nifty and Bank Nifty analysis
def option_analysis():
    Statergy = st.sidebar.selectbox('Select the Analysis:', ['Gap_Analysis', 'OI_Analysis','Option_Greeks_Analysis'])
    if Statergy =='Gap_Analysis':
        st.title('Gap_Up & Gap_down Analysis')

        # Sidebar layout
        st.sidebar.title('Input Parameters')

        # User input for interval
        interval = st.sidebar.selectbox('Select interval:', ['1d','1mo'])

        # Specify start and end dates for historical data
        no_backdays = st.sidebar.slider('Select number of past days:', 1, 3000, 50)
        symbols_select = st.selectbox('Select Symbol',['^NSEBANK', '^NSEI','^INDIAVIX','ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
            'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
            'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GRASIM.NS',
            'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'ICICIBANK.NS', 'IBULHSGFIN.NS', 'IOC.NS',
            'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
            'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS',
            'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'VEDL.NS',
            'WIPRO.NS', 'YESBANK.NS', 'ZEEL.NS'])
        start_date = datetime.now() - timedelta(days=no_backdays)
        end_date = datetime.now()

        # User input for short and long EMA windows
        #short_window = st.sidebar.selectbox('Select short EMA window:', [10, 20, 50])
        #long_window = st.sidebar.selectbox('Select long EMA window:', [50, 200])

        # Get stock data
        #symbols = ['^NSEBANK', '^NSEI','^INDIAVIX']
        from fetch_data_yf import fetch_data_yf
        fetch_data=fetch_data_yf()

        df_full = fetch_data.fetch_stock_data([symbols_select], start_date, end_date, interval)

        # Calculate gap up and gap down values
        for i in range(len(df_full)):
            df_full[i]['PrevClose'] = df_full[i]['Close'].shift(1)
            df_full[i]['Gap'] = df_full[i]['Open'] - df_full[i]['PrevClose']
            df_full[i]['Gap_Up'] = np.where(df_full[i]['Gap'] > 0, df_full[i]['Gap'], np.nan)
            df_full[i]['Gap_Down'] = np.where(df_full[i]['Gap'] < 0, df_full[i]['Gap'], np.nan)
            df_full[i]['Gap_categeory'] = df_full[i].apply(lambda row: gap_per(row['Gap']), axis=1)

        # Apply moving average strategy
        for symbol_df in df_full:
            moving_avg_crossover(symbol_df, 20, 50)


        


        # Add your visualization code here
        # Example:
        for i in range(len(df_full)):
            fig = go.Figure()
            
            fig.add_trace(go.Bar(x=df_full[i]['Date_only'], y=df_full[i]['Gap'], 
                            marker_color=np.where(df_full[i]['Gap'] > 0, 'green', 'red')))
            fig.update_layout(title=f'Gap up Gap Down Analysis for      {df_full[i]["Symbol"].iloc[0] } Spot Price={round(df_full[i]["Close"].tail(1).iloc[0],2)} ')
            #fig.update_layout(template="ggplot2")
            fig.update_layout(xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False))
            st.plotly_chart(fig)
            #Adding dataframe
            col1, col2, col3=st.columns(3)
            with col1:
                st.write('Gap Statistics')
                st.write(df_full[i]['Gap'].describe().round(1))

            with col2:
                st.write('Gap up')
                st.write(df_full[i].sort_values(by='Gap', ascending=False)[['Date_only','Gap']].head(5))

            with col3:
                st.write('Gap down')
                st.write(df_full[i].sort_values(by='Gap', ascending=True)[['Date_only','Gap']].head(5))
            
            fig_gap_dist = go.Figure()
            # Splitting the data into positive and negative gaps
            positive_gaps = df_full[i]['Gap'][df_full[i]['Gap'] > 0]
            negative_gaps = df_full[i]['Gap'][df_full[i]['Gap'] < 0]

            # Define a color palette for years
            year_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # You can use any other color palette

            # Adding violin traces for positive and negative gaps
            #fig_gap_dist.add_trace(go.Violin(x=positive_gaps, line_color='green', side='positive', name='Positive Gap Distribution', hoverinfo='y'))
            #fig_gap_dist.add_trace(go.Violin(x=negative_gaps, line_color='red', side='negative', name='Negative Gap Distribution', hoverinfo='y'))
            # Display violin plot for gap data grouped by month
        for i in range(len(df_full)):
            fig_gap_dist = go.Figure()

            # Iterate over unique months
            for day in df_full[i]['Day_of_Week'].unique():
                # Filter data for the current month
                day_data = df_full[i][df_full[i]['Day_of_Week'] == day]
                # Add violin trace for the current year
                fig_gap_dist.add_trace(go.Violin(x=day_data['Day_of_Week'],
                                                y=day_data['Gap'],
                                                legendgroup=str(day),
                                                scalegroup=str(day),
                                                name=f' {day}',
                                                side='positive',
                                                line_color='blue',
                                                showlegend=False,
                                                width=0.8,
                                                fillcolor=year_palette[df_full[i]['Day_of_Week'].unique().tolist().index(day)]))
            y_range = [df_full[i]['Gap'].min(), df_full[i]['Gap'].max()]  # Adjust this range as needed
        
            fig_gap_dist.update_layout(title=f'Gap Distribution by Year for {df_full[i]["Symbol"].iloc[0]}',
                                    xaxis_title='Year',
                                    yaxis_title='Gap',
                                    violinmode='group',
                                    yaxis=dict(range=y_range))
            fig_gap_dist.update_layout(xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False))
            st.plotly_chart(fig_gap_dist)

                    
            #fig_gap_dist.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', 
                                                    #line=dict(color='red', width=2), name='Normal Distribution'))
        st.write(df_full[i])
        st.write(df_full[i]['Gap_categeory'].value_counts(normalize=True) * 100)
    elif Statergy =='OI_Analysis':
        #import optionchainAnalyzer class from option_chain_analyzer
        from option_chain_analyzer import OptionsChainAnalyzer
        analyzer = OptionsChainAnalyzer()
        st.sidebar.title("Options Chain Analysis")
        symbol = st.sidebar.selectbox('Select Symbol:',['BANKNIFTY','NIFTY'])
        # call fetch_nse_data function to fetch oprtions data 
        data=analyzer.fetch_nse_data(symbol)
         # Fetch list of expiry dates from options data
        expiry_dates = data['records']['expiryDates']
        expiry_date = st.sidebar.selectbox("Select expiry date:", expiry_dates)
        #df_final=analyzer.modify_data(data,expiry_date)
        #st.write(df_final)
        
        # Submit button
        
        if st.sidebar.button("Submit"):
            df_final, sorted_ce, sorted_pe, spot_price, R1, R2, R3,S1, S2, S3, current_datetime=analyzer.modify_data(data,expiry_date)
            st.write(sorted_ce)            
            if data:
            #st.write(data)
            #call modify_data function from optionAnalyser to create options dataframe
            #df_final=analyzer.modify_data(data,expiry_date)
                analyzer.display_charts(df_final, sorted_ce, sorted_pe, spot_price, R1, R2, R3, current_datetime)
            #st.write(data)

#Option Greeks Page   
    elif Statergy == 'Option_Greeks_Analysis':
        from option_chain_analyzer import OptionsChainAnalyzer
        symbol = st.sidebar.selectbox('Select Symbol:', ['BANKNIFTY', 'NIFTY'],key='symbol_select')
        analyzer = OptionsChainAnalyzer()
        data = analyzer.fetch_nse_data(symbol)
        if data:
            expiry_dates = data['records']['expiryDates']
            expiry_date = st.sidebar.selectbox("Select expiry date:", expiry_dates,key='expiry_date_select')
            df_full = analyzer.fetch_data(symbol, expiry_date)
            st.write(expiry_dates)


        from option_greeks import option_greeks
        greek_analyser=option_greeks()
        st.sidebar.title("Option_Greeks_Analysis")
        symbol = st.sidebar.selectbox('Select Symbol:', ['BANKNIFTY','NIFTY'])
        expiry_date = st.sidebar.selectbox("Select expiry date:", [])
        st.write(df_full)

                    # Submit button
        if st.sidebar.button("Submit"):
            
            st.write(greek_analyser.option_greeks_analyser(120,17500,17560))
            st.write('option_greeks_analysis')  
            

# Add navigation to different pages
page = st.sidebar.radio("Select Page", ['Stock_Statergies', 'Stock_Option_Analysis'])

if page == 'Stock_Statergies':
    # Sidebar layout
    st.sidebar.title('Input Parameters')

    # User input for interval
    
    Statergy = st.sidebar.selectbox('Select the statergy:', ['EMA', 'Super Trend'])
    if Statergy =='EMA':
            interval = st.sidebar.selectbox('Select interval:', ['5m', '15m', '30m', '1h','1d'])
            # Specify start and end dates for historical data
            no_backdays = st.sidebar.slider('Select number of past days:', 1, 300, 20)
            start_date = datetime.now() - timedelta(days=no_backdays)
            end_date = datetime.now()

            # User input for short and long EMA windows
            short_window = st.sidebar.selectbox('Select short EMA window:',[10, 20, 50])
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
                fetch_data=fetch_data_yf()

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
 
                # Apply buy_sell_signal to dataframe based on EMA
                def buy_sell_signals(SMA_short, SMA_long, previous_SMA_Long):
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
                        symbol_df['Signal_Type'] = np.vectorize(buy_sell_signals)(symbol_df['SMA_short'], symbol_df['SMA_long'], symbol_df['previous_SMA_Long'])
                    else:
                        print("DataFrame is empty or does not contain valid data.")
                # Apply MACD_buy_sell to dataframe based on MACD
                for symbol_df in df_full:
                    if not symbol_df.empty:
                        symbol_df['Signal_Type_MACD'] = np.vectorize(MACD_buy_sell)(symbol_df['MACD'], symbol_df['Signal_line'], symbol_df['previous_MACD'])
                    else:
                        print("DataFrame is empty or does not contain valid data.")
                # Dataframe to store buy signals
                df_buy=pd.DataFrame()
                for symbol in range(0,len(df_full)):
                #buy_position=df_full[symbol].tail(5)['Position'].values
                    if 'Bullish Crossover' in df_full[symbol].tail(no_backcandle)['Signal_Type'].values and 'Bullish MACD Crossover' in df_full[symbol].tail(no_backcandle)['Signal_Type_MACD'].values:
                        df=pd.DataFrame({symbol:[df_full[symbol]['Symbol'].mode()]})
                        df_buy = pd.concat([df_buy, df], axis=1)   

                # Dataframe to store sell signals
                df_sell=pd.DataFrame()
                for symbol in range(0,len(df_full)):
                #buy_position=df_full[symbol].tail(50)['Position'].values
                    if 'Bearish Crossover' in df_full[symbol].tail(no_backcandle)['Signal_Type'].values and 'Bearish MACD Crossover' in df_full[symbol].tail(no_backcandle)['Signal_Type_MACD'].values:
                    #array_name = f'array{i+1}'

                        df=pd.DataFrame({symbol:[df_full[symbol]['Symbol'].mode()]})
                        df_sell = pd.concat([df_sell, df], axis=1)
                #st.write(df_full[0][df_full[0]['Signal_Type_MACD']=='Bullish MACD Crossover'])            
                df_buy=df_buy.T 
                df_sell=df_sell.T 


                # Display buy signals dataframe
                st.write(df_buy)
                st.write(df_sell)


                # Candlestick chart with SMA
                for i in df_buy.index:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df_full[i].index, open=df_full[i]['Open'], high=df_full[i]['High'],
                                                low=df_full[i]['Low'], close=df_full[i]['Close']))
                    fig.add_trace(go.Scatter(x=df_full[i].index, y=df_full[i]['SMA_short'], mode='lines', name='Short EMA',
                                            line=dict(color='blue', width=2)))
                    fig.add_trace(go.Scatter(x=df_full[i].index, y=df_full[i]['SMA_long'], mode='lines', name='Long EMA',
                                            line=dict(color='red', width=2)))
                    buy_signals = df_full[i][df_full[i]['Signal_Type'] == 'Bullish Crossover']
                    sell_signals = df_full[i][df_full[i]['Signal_Type'] == 'Bearish Crossover']
                    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['SMA_short'], mode='markers',
                                            marker=dict(color='green', size=10), name='Buy Signal'))
                    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['SMA_short'], mode='markers',
                                            marker=dict(color='yellow', size=10), name='Sell Signal'))
                    fig.update_layout(title=f'Candlestick Chart with SMA for {df_full[i]["Symbol"].iloc[0]}',
                                    xaxis_title='Date', yaxis_title='Price')


                    # Calculate y-axis range based on latest 100 close prices
                    latest_100_close_prices = df_full[i]['Close'].tail(500)
                    y_range = [latest_100_close_prices.min() - 10, latest_100_close_prices.max() + 10]
                    fig.update_layout(yaxis=dict(range=y_range))
                    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                    fig.update_layout(width=700, height=700)

                    st.plotly_chart(fig)

                    fig1 = go.Figure()

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

                    st.plotly_chart(fig1)
                
    elif Statergy == 'Super Trend':
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
    elif Statergy == '3':
        None
elif page == 'Stock_Option_Analysis':
    option_analysis()
