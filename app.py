import streamlit as st

st.set_page_config(page_title="Stock Anomaly Detection", layout="wide")

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


st.title("📈 Stock Market Anomaly Detection Dashboard")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
contamination = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.1, 0.02)

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

data['Returns'] = data['Close'].pct_change()
data['Volume_Change'] = data['Volume'].pct_change()
data.dropna(inplace=True)

features = data[['Returns', 'Volume_Change']]
model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
data['Anomaly'] = model.fit_predict(features)

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(data['Date'], data['Close'], label='Close Price')
anomalies = data[data['Anomaly'] == -1]
ax.scatter(anomalies['Date'], anomalies['Close'], color='red', label='Anomaly')
ax.legend()
st.pyplot(fig)

st.subheader("Detected Anomalies")
st.dataframe(anomalies[['Date', 'Close', 'Volume']])
