import streamlit as st


# Add navigation to different pages
page = st.sidebar.radio("Select Page", ['Stock_Statergies', 'Stock_Option_Analysis'])

if page == 'Stock_Statergies':
    from Stock_Statergy_page import Stocks_statergy_page_class
    Statergy = st.sidebar.selectbox('Select the statergy:', ['EMA', 'Super Trend'])
    # Sidebar layout
    st.sidebar.title('Input Parameters')
    stock_statergy = Stocks_statergy_page_class()
    stock_statergy.option_analysis_page_func(Statergy)


elif page == 'Stock_Option_Analysis':
    from option_analyse_page import OptionsChainAnalyzer_class
    Statergy = st.sidebar.selectbox('Select the Analysis:', ['Gap_Analysis', 'OI_Analysis','Option_Greeks_Analysis'])
    options_analyzer = OptionsChainAnalyzer_class()
    options_analyzer.option_analysis_page_func(Statergy)
