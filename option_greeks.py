import streamlit as st

class option_greeks:
    @staticmethod
    def option_greeks_analyser(price, spot_price, strike_price):
        from py_vollib.black_scholes.implied_volatility import implied_volatility
        from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, rho

        t = 0.001
        r = 0.1
        flag = 'c'

        IV = implied_volatility(price, spot_price, strike_price, t, r, flag)
        delta_val = delta(flag, spot_price, strike_price, t, r, IV)
        rho_val = rho(flag, spot_price, strike_price, t, r, IV)
        gamma_val = gamma(flag, spot_price, strike_price, t, r, IV)
        theta_val = theta(flag, spot_price, strike_price, t, r, IV)

        return IV, delta_val, rho_val, gamma_val, theta_val

# Usage example
if __name__ == "__main__":
    IV, delta_val, rho_val, gamma_val, theta_val = option_greeks.option_greeks_analyser(120, 17500, 17560)
    print("Implied Volatility:", IV)
    print("Delta:", delta_val)
    print("Rho:", rho_val)
    print("Gamma:", gamma_val)
    print("Theta:", theta_val)
