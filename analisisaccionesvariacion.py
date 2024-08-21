import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import re

st.title("Análisis de Precios de Acciones")

# User inputs
input_ratio = st.text_input("Ingrese el ticker o la razón de las acciones:", "YPFD.BA/YPF")
start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))
end_date = st.date_input("Seleccione la fecha de fin:", value=pd.to_datetime('today'))

# Function to fetch data for all tickers
def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        ticker = ticker.upper()  # Ensure ticker is uppercase
        st.write(f"Descargando datos para el ticker {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No hay datos disponibles para el ticker {ticker} en el rango de fechas seleccionado.")
            return None
        data[ticker] = df
    return data

# Function to align dates and fill missing values
def align_dates(data):
    # Determine the date range based on the first ticker
    first_ticker_dates = data[list(data.keys())[0]].index
    
    # Reindex all data to the date range of the first ticker
    for ticker in data:
        data[ticker] = data[ticker].reindex(first_ticker_dates)
        data[ticker] = data[ticker].ffill()  # Fill missing values with the previous available data
    
    return data

# Function to evaluate the ratio expression
def evaluate_ratio(ratio_str, data):
    # Extract tickers and operators
    tokens = re.split(r'([/*])', ratio_str.replace(' ', ''))
    tickers = [token for token in tokens if token and token not in '/*']
    operators = [token for token in tokens if token in '/*']
    
    # Convert all tickers to uppercase
    tickers = [ticker.upper() for ticker in tickers]
    
    # Verify all tickers are in the data
    missing_tickers = [ticker for ticker in tickers if ticker not in data]
    if missing_tickers:
        st.error(f"Tickers no disponibles en los datos: {', '.join(missing_tickers)}")
        return None

    # Compute the ratio
    result = None
    for i, ticker in enumerate(tickers):
        if result is None:
            result = data[ticker]['Adj Close']
        else:
            if operators[i-1] == '*':
                result *= data[ticker]['Adj Close']
            elif operators[i-1] == '/':
                result /= data[ticker]['Adj Close']
    
    return result

# Process the ratio input
st.write(f"Obteniendo datos para la razón {input_ratio} desde {start_date} hasta {end_date}...")
tickers = re.findall(r'\b\w+\.\w+', input_ratio)
data = fetch_data(tickers, start_date, end_date)

if data:
    # Align dates and handle missing values
    data = align_dates(data)
    
    # Evaluate ratio
    ratio_data = evaluate_ratio(input_ratio, data)
    
    if ratio_data is not None:
        # Calculate monthly price variations
        ratio_data = ratio_data.to_frame(name='Adjusted Close')
        ratio_data.index = pd.to_datetime(ratio_data.index)
        ratio_data['Month'] = ratio_data.index.to_period('M')
        monthly_data = ratio_data.resample('M').ffill()
        monthly_data['Cambio Mensual (%)'] = monthly_data['Adjusted Close'].pct_change() * 100

        # Plot monthly price variations
        st.write("### Variaciones Mensuales de Precios")
        fig = px.line(monthly_data, x=monthly_data.index, y='Cambio Mensual (%)',
                      title=f"Variaciones Mensuales de {input_ratio}",
                      labels={'Cambio Mensual (%)': 'Cambio Mensual (%)'})
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig)

        # Histogram with Gaussian and percentiles
        st.write("### Histograma de Variaciones Mensuales con Ajuste de Gauss")
        monthly_changes = monthly_data['Cambio Mensual (%)'].dropna()

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
            ax.axvline(perc_value, color=colors[i], linestyle='--', label=f'{percentile}º Percentil')
            ax.text(perc_value, ax.get_ylim()[1]*0.9, f'{perc_value:.2f}', color=colors[i],
                    rotation=90, verticalalignment='center', horizontalalignment='right')

        ax.set_title(f"Histograma de Cambios Mensuales con Ajuste de Gauss")
        ax.set_xlabel("Cambio Mensual (%)")
        ax.set_ylabel("Densidad")
        ax.legend()
        st.pyplot(fig)

        # Heatmap of monthly variations
        st.write("### Mapa de Calor de Variaciones Mensuales")
        monthly_pivot = monthly_data.pivot_table(values='Cambio Mensual (%)', index=monthly_data.index.year, columns=monthly_data.index.month, aggfunc='mean')
        
        # Define a custom colormap with greens for positive values and reds for negative values
        colors = ['red', 'white', 'green']
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(monthly_pivot, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, center=0, ax=ax)
        plt.title(f"Mapa de Calor de Variaciones Mensuales para {input_ratio}")
        plt.xlabel("Mes")
        plt.ylabel("Año")
        st.pyplot(fig)

        # Monthly and yearly average changes
        st.write("### Cambios Promedio Mensuales")
        avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].mean()
        avg_monthly_changes.index = pd.to_datetime(avg_monthly_changes.index, format='%m').strftime('%B')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_monthly_changes.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Cambios Promedio Mensuales")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Cambio Promedio Mensual (%)")
        st.pyplot(fig)

        st.write("### Cambios Promedio Anuales")
        avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_yearly_changes.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Cambios Promedio Anuales")
        ax.set_xlabel("Año")
        ax.set_ylabel("Cambio Promedio Anual (%)")
        st.pyplot(fig)

        # Display statistical summary
        st.write("### Resumen Estadístico")
        st.write(monthly_changes.describe())
