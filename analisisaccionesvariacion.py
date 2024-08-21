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
start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))

# Fetch stock data
st.write(f"Fetching data for {ticker} from {start_date} onwards...")
data = yf.download(ticker, start=start_date)

# Check if data is available
if data.empty:
    st.error("No data available for the selected stock and date range.")
else:
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

    # Histogram with Gaussian and percentiles
    st.write("### Histogram of Monthly Price Variations with Gaussian Fit")
    monthly_changes = monthly_data['Monthly Change (%)'].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(monthly_changes, kde=False, stat="density", color="skyblue", ax=ax, binwidth=2)
    
    # Fit Gaussian distribution
    mu, std = norm.fit(monthly_changes)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    
    # Percentiles
    percentiles = [25, 50, 75]
    for percentile in percentiles:
        perc_value = np.percentile(monthly_changes, percentile)
        ax.axvline(perc_value, color='red', linestyle='--', label=f'{percentile}th Percentile')
        ax.text(perc_value, ax.get_ylim()[1]*0.9, f'{perc_value:.2f}', color='red')

    ax.set_title(f"Histogram of {ticker} Monthly Changes with Gaussian Fit")
    ax.set_xlabel("Monthly Change (%)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # Display statistical summary
    st.write("### Statistical Summary")
    st.write(monthly_changes.describe())
