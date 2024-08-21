import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
import re

# Define custom colormap
def get_custom_cmap():
    colors = ['red', 'white', 'green']
    return LinearSegmentedColormap.from_list('custom_diverging', colors)

# Function to fetch data
def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        ticker = ticker.upper()
        st.write(f"Intentando descargar datos para el ticker {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                st.warning(f"No hay datos disponibles para el ticker {ticker} en el rango de fechas seleccionado.")
            else:
                data[ticker] = df
        except Exception as e:
            st.error(f"Error al descargar datos para el ticker {ticker}: {e}")
    return data

# Function to align dates and fill missing values
def align_dates(data):
    if not data:
        return {}
    
    first_ticker_dates = data[list(data.keys())[0]].index
    
    for ticker in data:
        data[ticker] = data[ticker].reindex(first_ticker_dates)
        data[ticker] = data[ticker].ffill()  # Forward fill missing values
    
    return data

# Function to evaluate the ratio
def evaluate_ratio(main_ticker, ratio_divisor, data):
    if not main_ticker:
        st.error("El ticker principal no puede estar vacío.")
        return None

    # Process the ratio with the optional divisor
    if ratio_divisor:
        main_ticker = main_ticker.upper()
        ratio_divisor = ratio_divisor.upper()

        if main_ticker in data:
            result = data[main_ticker]['Adj Close']
            if ratio_divisor in data:
                result /= data[ratio_divisor]['Adj Close']
            else:
                st.error(f"Divisor no disponible en los datos: {ratio_divisor}")
                return None
        else:
            st.error(f"Ticker principal no disponible en los datos: {main_ticker}")
            return None
    else:
        main_ticker = main_ticker.upper()
        if main_ticker in data:
            result = data[main_ticker]['Adj Close']
        else:
            st.error(f"Ticker principal no disponible en los datos: {main_ticker}")
            return None

    return result

# Streamlit app
st.title("Análisis de Precios de Acciones")

# User inputs
main_ticker = st.text_input("Ingrese el ticker principal:")
ratio_divisor = st.text_input("Ingrese el ticker o ratio divisor (opcional):")

start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))
end_date = st.date_input("Seleccione la fecha de fin:", value=pd.to_datetime('today'))

# Option to choose between average and median for monthly and yearly graphs
metric_option = st.radio("Seleccione la métrica para los gráficos mensuales y anuales:", ("Promedio", "Mediana"))

# Extract tickers from the inputs
tickers = {main_ticker, ratio_divisor}
tickers = {ticker.upper() for ticker in tickers if ticker}

data = fetch_data(tickers, start_date, end_date)

if data:
    data = align_dates(data)
    
    # Evaluate ratio
    ratio_data = evaluate_ratio(main_ticker, ratio_divisor, data)
    
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
                      title=f"Variaciones Mensuales de {main_ticker}" + (f" / {ratio_divisor}" if ratio_divisor else ""),
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
        cmap = get_custom_cmap()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(monthly_pivot, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, center=0, ax=ax)
        plt.title(f"Mapa de Calor de Variaciones Mensuales para {main_ticker}" + (f" / {ratio_divisor}" if ratio_divisor else ""))
        plt.xlabel("Mes")
        plt.ylabel("Año")
        st.pyplot(fig)

        # Monthly and yearly average/median changes
        st.write(f"### Cambios Promedio {metric_option} Mensuales")
        if metric_option == "Promedio":
            avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].mean()
        else:
            avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].median()
        avg_monthly_changes.index = pd.to_datetime(avg_monthly_changes.index, format='%m').strftime('%B')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_monthly_changes.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f"Cambios Promedio {metric_option} Mensuales")
        ax.set_xlabel("Mes")
        ax.set_ylabel(f"Cambio {metric_option} Mensual (%)")
        st.pyplot(fig)

        st.write(f"### Cambios Promedio {metric_option} Anuales")
        if metric_option == "Promedio":
            avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].mean()
        else:
            avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].median()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_yearly_changes.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f"Cambios Promedio {metric_option} Anuales")
        ax.set_xlabel("Año")
        ax.set_ylabel(f"Cambio {metric_option} Anual (%)")
        st.pyplot(fig)
