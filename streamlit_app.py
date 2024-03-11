import streamlit as st


# Add navigation to different pages
page = st.sidebar.radio("Select Page", ['Stock_Statergies', 'Stock_Option_Analysis','Stocks_Analysis','ML_Models','Backtesting_statergies'])

if page == 'Stock_Statergies':
    from Stock_Statergy_page import Stocks_statergy_page_class
    Statergy = st.sidebar.selectbox('Select the statergy:', ['EMA','Breakout_signal','Super Trend'])
    # Sidebar layout
    if Statergy=='EMA':
        st.sidebar.title('Input Parameters')
        stock_statergy_ema = Stocks_statergy_page_class()
        stock_statergy_ema.ema_func()
    elif Statergy=='Breakout_signal':
        st.sidebar.title('Input Parameters')
        stock_statergy_breakout = Stocks_statergy_page_class()
        df_full=stock_statergy_breakout.breakout_func()
        
        



elif page == 'Stock_Option_Analysis':
    from option_analyse_page import OptionsChainAnalyzer_class
    Statergy = st.sidebar.selectbox('Select the Analysis:', ['Gap_Analysis', 'OI_Analysis','Option_Greeks_Analysis'])
    options_analyzer = OptionsChainAnalyzer_class()
    options_analyzer.option_analysis_page_func(Statergy)

elif page == 'Stocks_Analysis':
    Stock_analysis = st.sidebar.selectbox('Select the Analysis:', ['Fundamental_Analysis', 'Technical_Analysis','Other_Analysis'])
    if Stock_analysis=='Fundamental_Analysis':

        ticker = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
        'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
        'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GRASIM.NS',
        'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'ICICIBANK.NS', 'IBULHSGFIN.NS', 'IOC.NS',
        'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
        'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS',
        'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'VEDL.NS',
        'WIPRO.NS', 'YESBANK.NS', 'ZEEL.NS','ADANIPORTS.BO', 'ASIANPAINT.BO', 'AXISBANK.BO', 'BAJAJ-AUTO.BO',
        'BAJFINANCE.BO', 'BAJAJFINSV.BO', 'BPCL.BO', 'BHARTIARTL.BO']  # Add more if needed

        from stocks_helper_functions import stocks_helper_class
        fundamentals=stocks_helper_class()
        
        fun_df=fundamentals.create_fundamental_features(ticker)
        st.write(fun_df)
elif page == 'ML_Models':
    pass        
elif page == 'Backtesting_statergies':
    pass        