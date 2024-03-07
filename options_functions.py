import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
import calendar

class options_functions_class:
    def __init__(self):
        pass
    def add_date(self,data):
        data['Date_only'] = pd.to_datetime(data['Date']).dt.date
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Day_of_Week'] = data['Date'].dt.day_name()
        return data 
    
    def gapup_gapdown(self,data):
        data['PrevClose'] = data['Close'].shift(1)
        data['Gap'] = data['Open'] - data['PrevClose']
        data['Gap_Up'] = np.where(data['Gap'] > 0, data['Gap'], np.nan)
        data['Gap_Down'] = np.where(data['Gap'] < 0, data['Gap'], np.nan)
        return data 

        

    # Function to determine if a given date is an expiry day
    def is_expiry_day(self,date):
        # Check if the date is the last Thursday of the month (monthly expiry)
        if date.weekday() == 2 and date.day > (calendar.monthrange(date.year, date.month)[1] - 7):
            return 'Monthly Expiry'
        # Check if the date is a Friday (weekly expiry for indices)
        elif date.weekday() == 2:
            return 'Weekly Expiry'
        else:
            return 'Not Expiry Day'
        
    # Function determine the gap up basket based on its values    
    def gap_per_cat(self,gap):
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