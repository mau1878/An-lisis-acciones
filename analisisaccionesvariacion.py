import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

st.title("Análisis de Precios de Acciones")

# User inputs
ticker = st.text_input("Ingrese el ticker de la acción:", "AAPL").upper()
start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))

# Fetch stock data
st.write(f"Obteniendo datos para {ticker} desde {start_date} en adelante...")
data = yf.download(ticker, start=start_date)

# Check if data is available
if data.empty:
    st.error("No hay datos disponibles para la acción y el rango de fechas seleccionados.")
else:
    # Calculate monthly price variations
    data['Month'] = data.index.to_period('M')
    monthly_data = data.resample('M').ffill()
    monthly_data['Cambio Mensual (%)'] = monthly_data['Adj Close'].pct_change() * 100

    # Plot monthly price variations
    st.write("### Variaciones Mensuales de Precios")
    fig = px.line(monthly_data, x=monthly_data.index, y='Cambio Mensual (%)',
                  title=f"Variaciones Mensuales de {ticker}",
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

    ax.set_title(f"Histograma de Cambios Mensuales de {ticker} con Ajuste de Gauss")
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
    plt.title(f"Mapa de Calor de Variaciones Mensuales para {ticker}")
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
