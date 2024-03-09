import numpy as np
import pandas as pd

class option_greeks_class:
    def __init__(self):
        pass

    def option_greeks_calc(self, df_final, spot_price,default_volatility=0.2,min_volatility=0.01):
        from py_vollib.black_scholes.implied_volatility import implied_volatility
        from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, rho
        from py_lets_be_rational.exceptions import BelowIntrinsicException

        t = 0.001
        r = 0.1

        # Call option calculations for call options
        flag = 'c'
        iv_ce = []
        for index, row in df_final.iterrows():
            try:
                iv = implied_volatility(row['lastPrice_CE'], spot_price, row['strikePrice_CE'], t, r, flag)
                if iv < min_volatility:
                    iv = default_volatility
            except BelowIntrinsicException:
                iv = default_volatility  # Set default volatility if BelowIntrinsicException occurs
            iv_ce.append(iv)
        df_final['IV_CE'] = iv_ce
        df_final['delta_val_CE'] = df_final.apply(lambda row: delta(flag, spot_price, row['strikePrice_CE'], t, r, row['IV_CE']), axis=1)
        df_final['rho_val_CE'] = df_final.apply(lambda row: rho(flag, spot_price, row['strikePrice_CE'], t, r, row['IV_CE']), axis=1)
        df_final['gamma_val_CE'] = df_final.apply(lambda row: gamma(flag, spot_price, row['strikePrice_CE'], t, r, row['IV_CE']), axis=1)
        df_final['theta_val_CE'] = df_final.apply(lambda row: theta(flag, spot_price, row['strikePrice_CE'], t, r, row['IV_CE']), axis=1)

        # Call option calculations for put options
        flag = 'p'
        iv_pe = []
        for index, row in df_final.iterrows():
            try:
                iv = implied_volatility(row['lastPrice_PE'], spot_price, row['strikePrice_PE'], t, r, flag)
                if iv < min_volatility:
                    iv = default_volatility
            except BelowIntrinsicException:
                iv = default_volatility  # Set default volatility if BelowIntrinsicException occurs
            iv_pe.append(iv)
        df_final['IV_PE'] = iv_pe
        df_final['delta_val_PE'] = df_final.apply(lambda row: delta(flag, spot_price, row['strikePrice_PE'], t, r, row['IV_PE']), axis=1)
        df_final['rho_val_PE'] = df_final.apply(lambda row: rho(flag, spot_price, row['strikePrice_PE'], t, r, row['IV_PE']), axis=1)
        df_final['gamma_val_PE'] = df_final.apply(lambda row: gamma(flag, spot_price, row['strikePrice_PE'], t, r, row['IV_PE']), axis=1)
        df_final['theta_val_PE'] = df_final.apply(lambda row: theta(flag, spot_price, row['strikePrice_PE'], t, r, row['IV_PE']), axis=1)
        columns_to_keep = ['IV_CE','delta_val_CE','rho_val_CE','gamma_val_CE','theta_val_CE','lastPrice_CE','StrikePriceStatus_CE','strikePrice_CE',
                           'IV_PE','delta_val_PE','rho_val_PE','gamma_val_PE','theta_val_PE','lastPrice_PE']
        df_greeks=df_final[columns_to_keep].copy()
        return df_greeks
