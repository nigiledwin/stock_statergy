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
from Stock_Statergy_page import Stocks_statergy_page_class

# Add navigation to different pages
page = st.sidebar.radio("Select Page", ['Stock_Statergies', 'Stock_Option_Analysis'])

if page == 'Stock_Statergies':
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
