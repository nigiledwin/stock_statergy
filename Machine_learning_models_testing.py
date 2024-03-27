import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class ML_model_class:
    def __init__(self):
        pass

    def ML_model_func_rf(self,df):


        # Select features and target variable
        features = ['MA10', 'MA50', 'EMA10', 'EMA50', 'RSI', 'EMA12', 'EMA26', 'MACD',
                    '20MA', '20STD', 'UpperBand', 'LowerBand']
        target = 'Close_Next_Day'

        # Split the dataset into features and target variable
        X = df[features].iloc[:-1]
        y = df[target].iloc[:-1]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error:", mse)

        X_pred_last = df[features].iloc[-1:].values
        y_pred_close = model.predict(X_pred_last)

        # Calculate R2 score
        r2 = r2_score(y_test, y_pred)
        print("R2 Score:", r2)
        print(y_pred_close)
        print(mse)

        return r2,y_pred_close,mse

df_nse = pd.read_csv('stock_data_NSE_1d.csv')
df_ADANIPORTS=df_nse[df_nse['Symbol']=='ADANIPORTS.NS']
#print(df_ADANIPORTS)

#creating features usingg feature engineering class

from ML_feature_engineering import feature_engg_class
feature_df=feature_engg_class()
df_ADANIPORTS=feature_df.feature_engg_func(df_ADANIPORTS)


#Applying ML functions
from Machine_learning_models_testing import ML_model_class
ML_df=ML_model_class()
model=ML_df.ML_model_func_rf(df_ADANIPORTS)



