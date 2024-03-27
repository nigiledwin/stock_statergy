import pandas as pd
class ML_model_run_class:
    def __init__(self):
        pass
    def model_run_fun(self):

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

        return model,df_ADANIPORTS

