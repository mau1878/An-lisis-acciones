import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
    
    # Percentiles with different colors and vertical labels
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, percentile in enumerate(percentiles):
        perc_value = np.percentile(monthly_changes, percentile)
        ax.axvline(perc_value, color=colors[i], linestyle='--', label=f'{percentile}th Percentile')
        ax.text(perc_value, ax.get_ylim()[1]*0.9, f'{perc_value:.2f}', color=colors[i],
                rotation=90, verticalalignment='center', horizontalalignment='right')

    ax.set_title(f"Histogram of {ticker} Monthly Changes with Gaussian Fit")
    ax.set_xlabel("Monthly Change (%)")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

    # Heatmap of monthly variations
    st.write("### Heatmap of Monthly Price Variations")
    monthly_pivot = monthly_data.pivot_table(values='Monthly Change (%)', index=monthly_data.index.year, columns=monthly_data.index.month, aggfunc='mean')
    
    # Define a custom colormap with greens for positive values and reds for negative values
    colors = ['red', 'white', 'green']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(monthly_pivot, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, center=0, ax=ax)
    plt.title(f"Heatmap of Monthly Price Variations for {ticker}")
    plt.xlabel("Month")
    plt.ylabel("Year")
    st.pyplot(fig)

    # Monthly and yearly average changes
    st.write("### Average Monthly Changes")
    avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Monthly Change (%)'].mean()
    avg_monthly_changes.index = pd.to_datetime(avg_monthly_changes.index, format='%m').strftime('%B')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_monthly_changes.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Average Monthly Changes")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Monthly Change (%)")
    st.pyplot(fig)

    st.write("### Average Yearly Changes")
    avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Monthly Change (%)'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_yearly_changes.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Average Yearly Changes")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Yearly Change (%)")
    st.pyplot(fig)

    # Display statistical summary
    st.write("### Statistical Summary")
    st.write(monthly_changes.describe())
