import pandas as pd
import numpy as np
import yfinance as yf

class stocks_helper_class:
    def __init__(self):
        pass

    def classify_candlestick(self, row):
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

    def create_fundamental_features(self, tickers):
        fundamental_features = []

        for ticker in tickers:
            stock_data = yf.Ticker(ticker)
            history = stock_data.history(period="max")

            if 'Total Equity' in history.columns:
                net_income = history['Net Income']
                shareholders_equity = history['Total Equity']
                roe = (net_income / shareholders_equity).mean() * 100
            else:
                # Handle missing data
                roe = np.nan
                shareholders_equity = np.nan  # Assigning a default value when data is missing

            # Calculate market capitalization
            market_cap = stock_data.info.get('marketCap', np.nan)

            # Calculate price to earnings ratio (P/E ratio)
            pe_ratio = stock_data.info.get('trailingPE', np.nan)

            # Calculate price to book ratio (P/B ratio)
            pb_ratio = stock_data.info.get('priceToBook', np.nan)

            # Calculate dividend yield
            dividend_yield = stock_data.info.get('dividendYield', np.nan)

            # Calculate earnings per share (EPS)
            eps = stock_data.info.get('trailingEps', np.nan)

            # Calculate debt to equity ratio
            total_debt = stock_data.info.get('totalDebt', np.nan)
            if not np.isnan(shareholders_equity):
                debt_to_equity_ratio = total_debt / shareholders_equity
            else:
                debt_to_equity_ratio = np.nan

            # Create a dictionary to store the fundamental features
            fundamental_feature = {
                'Ticker': ticker,
                'Market Cap': market_cap,
                'P/E Ratio': pe_ratio,
                'P/B Ratio': pb_ratio,
                'Dividend Yield': dividend_yield,
                'ROE': roe,
                'EPS': eps,
                'Debt to Equity Ratio': debt_to_equity_ratio,
                # Add other features here
            }

            fundamental_features.append(fundamental_feature)

        return pd.DataFrame(fundamental_features)

# Example usage
tickers = ['AAPL', 'MSFT', 'GOOGL']
helper = stocks_helper_class()
fundamental_features_df = helper.create_fundamental_features(tickers)
print(fundamental_features_df)
