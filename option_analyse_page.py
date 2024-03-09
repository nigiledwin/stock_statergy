import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
from fetch_data_yf import fetch_data_yf
from options_functions import options_functions_class

class OptionsChainAnalyzer_class:
    def __init__(self):
        pass

        # Function to display Nifty and Bank Nifty analysis
    def option_analysis_page_func(self,Statergy): 
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


            from fetch_data_yf import fetch_data_yf
            fetch_data=fetch_data_yf()

            df_full = fetch_data.fetch_stock_data([symbols_select], start_date, end_date, interval)



            # Apply all modification methods from options_functions _class
            for df in df_full:
                from options_functions import options_functions_class
                options_func=options_functions_class()
                df=options_func.add_date(df)
                df=options_func.gapup_gapdown(df)
                df['Expiry_Day'] = df.apply(lambda row: options_func.is_expiry_day(row['Date']), axis=1)
                df['Gap_categeory'] = df.apply(lambda row: options_func.gap_per_cat(row['Gap']), axis=1)


            # Adding visualization
            
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
            from option_chain_analyzer import OptionsChainAnalyzer_class
            analyzer = OptionsChainAnalyzer_class()
            st.sidebar.title("Options Chain Analysis")
            symbol = st.sidebar.selectbox('Select Symbol:',['NIFTY'])
            # call fetch_nse_data function to fetch oprtions data 
            data=analyzer.fetch_nse_data(symbol)
            expiry_dates = data['records']['expiryDates']
            expiry_date = st.sidebar.selectbox("Select expiry date:", expiry_dates)
    
    
            # Submit button
            
            if st.sidebar.button("Submit"):
                df_final, sorted_ce, sorted_pe, spot_price, R1, R2, R3,S1, S2, S3, current_datetime=analyzer.modify_data(data,expiry_date)
                st.write(sorted_ce)            
                if data:
                #call modify_data function from optionAnalyser to create options dataframe
                #df_final=analyzer.modify_data(data,expiry_date)
                    analyzer.display_charts(df_final, sorted_ce, sorted_pe, spot_price, R1, R2, R3, current_datetime)
                #st.write(data)

    #Option Greeks Page   
        elif Statergy == 'Option_Greeks_Analysis':
            from option_chain_analyzer import OptionsChainAnalyzer_class
            symbol = st.sidebar.selectbox('Select Symbol:', ['NIFTY'],key='symbol_select')
            analyzer = OptionsChainAnalyzer_class()
            data = analyzer.fetch_nse_data(symbol)
            expiry_dates = data['records']['expiryDates']
            expiry_date = st.sidebar.selectbox("Select expiry date:", expiry_dates)
            df_final, sorted_ce, sorted_pe, spot_price, R1, R2, R3,S1, S2, S3, current_datetime=analyzer.modify_data(data,expiry_date)
            df_final.reset_index(inplace=True)
            df_final.dropna(inplace=True)
            st.write(spot_price)
            st.write(df_final)
            
            #Submit button
            if st.sidebar.button("Submit"):  
                from option_greeks import option_greeks_class
                greek_analyser=option_greeks_class()                       
                df_greeks=greek_analyser.option_greeks_calc(df_final, spot_price)
                st.write(df_greeks)
                
