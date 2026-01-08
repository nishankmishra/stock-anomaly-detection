import yfinance as yf
import pandas as pd
from sklearn.ensemble import IsolationForest

stock = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
stock.reset_index(inplace=True)

stock['Returns'] = stock['Close'].pct_change()
stock['Volume_Change'] = stock['Volume'].pct_change()
stock.dropna(inplace=True)

features = stock[['Returns', 'Volume_Change']]
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
stock['Anomaly'] = model.fit_predict(features)

print(stock[stock['Anomaly'] == -1].head())
