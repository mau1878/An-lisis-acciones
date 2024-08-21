import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

st.title("Stock Price Analysis")

# User inputs
ticker = st.text_input("Ingrese el ticker del stock:", "AAPL").upper()
start_date = st.date_input("Seleccione la fecha de inicio:")

# Fetch stock data
st.write(f"Fetching data for {ticker} from {start_date} onwards...")
data = yf.download(ticker, start=start_date)

# Calculate monthly price variations
data['Month'] = data.index.to_period('M')
monthly_data = data.resample('M').ffill()
monthly_data['Monthly Change (%)'] = monthly_data['Adj Close'].pct_change() * 100

# Plot monthly price variations
st.write("### Monthly Price Variations")
fig = px.line(monthly_data, x=monthly_data.index, y='Monthly Change (%)',
              title=f"Monthly Price Variations of {ticker}",
              labels={'Monthly Change (%)': 'Monthly Change (%)'})
fig.update_traces(mode='lines+markers')
st.plotly_chart(fig)

# Histogram with Gaussian
st.write("### Histogram of Monthly Price Variations with Gaussian Fit")
monthly_changes = monthly_data['Monthly Change (%)'].dropna()

fig, ax = plt.subplots()
sns.histplot(monthly_changes, kde=False, stat="density", ax=ax)
mu, std = norm.fit(monthly_changes)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
ax.plot(x, p, 'k', linewidth=2)
ax.set_title(f"Histogram of {ticker} Monthly Changes with Gaussian Fit")
ax.set_xlabel("Monthly Change (%)")
ax.set_ylabel("Density")
st.pyplot(fig)

# Display statistical summary
st.write("### Statistical Summary")
st.write(monthly_changes.describe())

