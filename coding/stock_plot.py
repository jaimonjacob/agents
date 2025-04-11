# filename: stock_plot.py
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Define the tickers
tickers = ['META', 'TSLA']

# Fetch the data for the last 1 year
data = yf.download(tickers, period='1y')

# Extract the 'Close' prices
close_prices = data['Close']

# Calculate the percentage change from the initial price
returns = close_prices.apply(lambda x: x / x[0] * 100)

# Plot the returns
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(returns.index, returns[ticker], label=ticker)

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Percentage Change')
plt.title('META and TSLA Stock Price Change (Last 1 Year)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()