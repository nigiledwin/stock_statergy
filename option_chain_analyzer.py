import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objs as go
import datetime

class OptionsChainAnalyzer:
    def __init__(self):
        pass

    # Function to fetch data from NSE website
    def fetch_nse_data(self, symbol):
        url = f'https://www.nseindia.com/api/option-chain-indices?symbol={symbol}'
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/option-chain',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200 and response.text.strip():
            data = response.json()
            return data
        else:
            st.error("Failed to fetch valid data from the NSE website.")
            return None

    # Fetch data
    def modify_data(self,data,expiry_date):
        # Fetch data based on symbol and expiry date
        
        if data:
            exp_list = data['records']['expiryDates']
            exp_date = expiry_date

            ce = {}
            pe = {}
            for i in data['records']['data']:
                if i['expiryDate'] == exp_date:
                    try:
                        ce[i['strikePrice']] = i['CE']
                    except:
                        pass
                    try:
                        pe[i['strikePrice']] = i['PE']
                    except:
                        pass

            # Convert column names to strings before concatenating with '_CE' and '_PE'
            df_ce = pd.DataFrame.from_dict(ce).transpose()
            df_ce.columns = df_ce.columns.astype(str) + '_CE'
            df_pe = pd.DataFrame.from_dict(pe).transpose()
            df_pe.columns = df_pe.columns.astype(str) + '_PE'

            # Spot price
            spot_price = data['records']['underlyingValue']

            # Sort CE and PE dataframes based on open interest
            sorted_ce = df_ce.sort_values(by='openInterest_CE', ascending=False)
            sorted_pe = df_pe.sort_values(by='openInterest_PE', ascending=False)

            # Extract the top 3 strike prices for CE and PE
            R1, R2, R3 = sorted_ce.index[:3]
            S1, S2, S3 = sorted_pe.index[:3]

            df_strike_prices = pd.DataFrame({
                'Resistance': [R1, R2, R3],
                'Support': [S1, S2, S3]
            }, index=['L1', 'L2', 'L3'])

            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Calculate ATM, ITM, or OTM for CE
            def calculate_option_status_CE(strike_price, spot_price):
                if strike_price == round(spot_price / 100) * 100:
                    return 'ATM'
                elif strike_price > spot_price:
                    return 'OTM'
                else:
                    return 'ITM'

            # Calculate ATM, ITM, or OTM for PE
            def calculate_option_status_PE(strike_price, spot_price):
                if strike_price == round(spot_price / 100) * 100:
                    return 'ATM'
                elif strike_price < spot_price:
                    return 'OTM'
                else:
                    return 'ITM'

            rounded_spot_price = round(spot_price / 100) * 100
            # Calculate the difference between each strike price and the ATM strike price
            df_ce['strike_price_difference'] = (df_ce['strikePrice_CE'] - rounded_spot_price) / 100

            # Define a function to assign labels based on the difference
            def assign_label(difference):
                if difference < 0:
                    return int(difference) - 1
                elif difference > 0:
                    return int(difference) + 1
                else:
                    return 0
            #Function to Calculate option greeks
            # Apply the function to create the new column
            df_ce['label'] = df_ce['strike_price_difference'].apply(assign_label)
            # Add a column to the DataFrame indicating the status of the option
            df_ce['StrikePriceStatus_CE'] = df_ce['strikePrice_CE'].astype(float).apply(lambda x: calculate_option_status_CE(x, spot_price))
            df_pe['StrikePriceStatus_PE'] = df_pe['strikePrice_PE'].astype(float).apply(lambda x: calculate_option_status_PE(x, spot_price))

            df = pd.concat([df_ce, df_pe], axis=1)

            # Filtering only strike price in the range of Spot price +-1000
            filtervalue = 2000
            df_filtered = df[(df['strikePrice_CE'] > (round(spot_price / 100) * 100) - filtervalue) & (
                        df['strikePrice_CE'] < (round(spot_price / 100) * 100) + filtervalue)]
            columns_to_keep = ['underlying_CE', 'expiryDate_CE', 'strikePrice_CE', 'StrikePriceStatus_CE', 'label', 'openInterest_CE',
                               'changeinOpenInterest_CE', 'totalTradedVolume_CE', 'lastPrice_CE', 'change_CE', 'openInterest_PE',
                               'changeinOpenInterest_PE', 'totalTradedVolume_PE', 'lastPrice_PE', 'change_PE', 'StrikePriceStatus_PE']
            df_final = df_filtered[columns_to_keep].copy()

            # Define conditions and corresponding values for the new columns for CE
            conditions_CE = [
                (df_final['openInterest_CE'] > 0) & (df_final['change_CE'] > 0),
                (df_final['openInterest_CE'] < 0) & (df_final['change_CE'] > 0),
                (df_final['openInterest_CE'] > 0) & (df_final['change_CE'] < 0),
                (df_final['openInterest_CE'] < 0) & (df_final['change_CE'] < 0)]

            # Define conditions and corresponding values for the new columns for PE
            conditions_PE = [
                (df_final['openInterest_PE'] > 0) & (df_final['change_PE'] > 0),
                (df_final['openInterest_PE'] < 0) & (df_final['change_PE'] > 0),
                (df_final['openInterest_PE'] > 0) & (df_final['change_PE'] < 0),
                (df_final['openInterest_PE'] < 0) & (df_final['change_PE'] < 0) ]

            # Define values corresponding to conditions
            values = ['Bullish Buying', 'Bearish Call Unwinding', 'Bearish Writing', 'Bearish Short Covering']

            # Create a new column for 'Market View' for CE
            df_final['Market View_CE'] = np.select(conditions_CE, values, default='')

            # Create a new column for 'Market View' for PE
            df_final['Market View_PE'] = np.select(conditions_PE, values, default='')

            df_statergy = df_final[
                ['Market View_CE', 'StrikePriceStatus_CE', 'lastPrice_CE', 'change_CE', 'strikePrice_CE', 'StrikePriceStatus_PE',
                 'lastPrice_PE', 'change_PE', 'Market View_PE',
                 'openInterest_CE', 'changeinOpenInterest_CE', 'openInterest_PE', 'changeinOpenInterest_PE']].copy()
            return df_final,sorted_ce,sorted_pe,R1,R2,R3,spot_price
            # Plot the chart
    def display_charts(self, df_final, sorted_ce, sorted_pe, spot_price, R1, R2, R3, current_datetime):
            fig = go.Figure()

            # Iterate over unique values of Market View_CE
            for market_view in df_final['Market View_CE'].unique():
                df_market_view = df_final[df_final['Market View_CE'] == market_view]
                hover_text_CE = market_view + '<br>Change CE: ' + df_market_view['change_CE'].astype(str) + '<br>Change_OI_CE: ' + df_market_view['changeinOpenInterest_CE'].astype(str) + '<br>OI_CE: ' + df_market_view['openInterest_CE'].astype(str)
                fig.add_trace(go.Scatter(y=df_market_view['strikePrice_CE'], x=df_market_view['lastPrice_CE'], mode='markers',
                                        name='Last Price CE (' + market_view + ')', hoverinfo='x+y+text', text=hover_text_CE))

            # Similarly, add traces for PE
            for market_view in df_final['Market View_PE'].unique():
                df_market_view = df_final[df_final['Market View_PE'] == market_view]
                hover_text_PE = market_view + '<br>Change PE: ' + df_market_view['change_PE'].astype(str) + '<br>Change_OI_PE: ' + df_market_view['changeinOpenInterest_PE'].astype(str) + '<br>OI_PE: ' + df_market_view['openInterest_PE'].astype(str)
                fig.add_trace(go.Scatter(y=df_market_view['strikePrice_CE'], x=df_market_view['lastPrice_PE'], mode='markers',
                                        name='Last Price PE (' + market_view + ')', hoverinfo='x+y+text', text=hover_text_PE))

            fig.update_layout(title=f'Last Prices for CE and PE Across Strike Prices for {spot_price} at {current_datetime} and 1st Resistance {R1}, 2nd Resistance {R2}, 3rd Resistance {R3} -Support {S1}{S2}{S3}',
                            xaxis_title='Strike Price',
                            yaxis_title='Last Price',
                            yaxis=dict(tickformat=",.0f"))

            st.plotly_chart(fig)

            # Create bar chart for open interest
            ce_open_interest = sorted_ce['openInterest_CE'].head()
            pe_open_interest = sorted_pe['openInterest_PE'].head()

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=ce_open_interest.index, y=ce_open_interest, name='CE Open Interest'))
            fig_bar.add_trace(go.Bar(x=pe_open_interest.index, y=pe_open_interest, name='PE Open Interest'))
            fig_bar.update_layout(title='Open Interest for Call and Put Options',
                                    xaxis_title='Strike Price',
                                    yaxis_title='Open Interest',
                                    yaxis=dict(tickformat=",.0f"),
                                    xaxis=dict(tickformat=",.0f"))

            # Display DataFrame and bar chart
            st.write("DataFrame with Strike Prices for Call and Put Options:")
            st.write(df_strike_prices)
            st.plotly_chart(fig_bar)    
            st.write(df_strike_prices)


if __name__ == "__main__":
    # Example usage
    analyzer = OptionsChainAnalyzer()
    st.sidebar.title("Options Chain Analysis")
    symbol = st.sidebar.text_input("Enter symbol (e.g., BANKNIFTY):", "BANKNIFTY")
    expiry_date = st.sidebar.selectbox("Select expiry date:", [])

    # Fetch expiry dates
    data = analyzer.fetch_nse_data(symbol)
    if data:
        expiry_dates = data['records']['expiryDates']
        expiry_date = st.sidebar.selectbox("Select expiry date:", expiry_dates)

    # Submit button
    if st.sidebar.button("Submit"):
        analyzer.fetch_and_display_data(symbol, expiry_date)
