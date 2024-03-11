from backtesting import Backtest, Strategy

# Calculate Simple Moving Average (SMA)
period = 20  # Adjust the period as needed
df['SMA'] = df['Close'].rolling(window=period).mean()

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class MyStrategy(Strategy):
    def init(self):
        # Initialize any necessary variables or state here
        pass
    
    def next(self):
        # Define your trading strategy logic here
        if crossover(self.data['Close'], self.data['SMA']):
            self.buy()
        elif crossover(self.data['SMA'], self.data['Close']):
            self.sell()

# Create a backtest scenario
bt = Backtest(df, MyStrategy, cash=10000, commission=0.002)

# Run the backtest
results = bt.run()

# Analyze the results
print(results)

# Plot the backtest results
bt.plot()
