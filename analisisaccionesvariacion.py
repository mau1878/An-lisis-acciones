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

# User inputs for stock and ratio
ticker = st.text_input("Ingrese el ticker de la acción:", "AAPL").upper()
start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))
ratio_input = st.text_input("Ingrese la razón de acciones (por ejemplo: 'YPFD.BA/YPF', 'GGAL.BA*10/GGAL', 'METR.BA/(GGAL.BA*10/GGAL)')", "")

# Function to fetch data and compute the ratio
def fetch_and_compute_ratio(ratio_expr, start_date):
    tickers = set(re.findall(r'[A-Z]+\.[A-Z]+', ratio_expr))  # Extract tickers from the ratio expression
    data = {}
    
    for ticker in tickers:
        df = yf.download(ticker, start=start_date)
        if not df.empty:
            df['Month'] = df.index.to_period('M')
            df = df.resample('M').ffill()
            df[ticker] = df['Adj Close']
            data[ticker] = df
        else:
            st.warning(f"No hay datos disponibles para el ticker {ticker}.")
            return None

    # Combine all dataframes
    combined_df = pd.concat(data.values(), axis=1)
    combined_df.columns = data.keys()

    # Compute the ratio
    ratio = combined_df.eval(ratio_expr)
    ratio.name = 'Ratio'
    
    return combined_df, ratio

# Fetch stock data and compute ratio
st.write(f"Obteniendo datos para la razón '{ratio_input}' desde {start_date} en adelante...")
combined_data, ratio = fetch_and_compute_ratio(ratio_input, start_date)

if combined_data is not None and ratio is not None:
    # Plot ratio
    st.write("### Variación del Ratio")
    fig = px.line(ratio, x=ratio.index, y='Ratio',
                  title=f"Variación del Ratio: {ratio_input}",
                  labels={'Ratio': 'Ratio'})
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig)

    # Histogram with Gaussian and percentiles
    st.write("### Histograma de Variaciones del Ratio con Ajuste de Gauss")
    ratio_changes = ratio.pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ratio_changes, kde=False, stat="density", color="skyblue", ax=ax, binwidth=2)
    
    # Fit Gaussian distribution
    mu, std = norm.fit(ratio_changes)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    
    # Percentiles with different colors and vertical labels
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, percentile in enumerate(percentiles):
        perc_value = np.percentile(ratio_changes, percentile)
        ax.axvline(perc_value, color=colors[i], linestyle='--', label=f'{percentile}º Percentil')
        ax.text(perc_value, ax.get_ylim()[1]*0.9, f'{perc_value:.2f}', color=colors[i],
                rotation=90, verticalalignment='center', horizontalalignment='right')

    ax.set_title(f"Histograma de Cambios del Ratio con Ajuste de Gauss")
    ax.set_xlabel("Cambio (%)")
    ax.set_ylabel("Densidad")
    ax.legend()
    st.pyplot(fig)

    # Heatmap of monthly variations
    st.write("### Mapa de Calor de Variaciones Mensuales del Ratio")
    monthly_pivot = ratio.to_frame().pivot_table(values='Ratio', index=ratio.index.year, columns=ratio.index.month, aggfunc='mean')
    
    # Define a custom colormap with greens for positive values and reds for negative values
    colors = ['red', 'white', 'green']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(monthly_pivot, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, center=0, ax=ax)
    plt.title(f"Mapa de Calor de Variaciones Mensuales del Ratio")
    plt.xlabel("Mes")
    plt.ylabel("Año")
    st.pyplot(fig)

    # Monthly and yearly average changes
    st.write("### Cambios Promedio Mensuales del Ratio")
    avg_monthly_changes = ratio.groupby(ratio.index.month).mean()
    avg_monthly_changes.index = pd.to_datetime(avg_monthly_changes.index, format='%m').strftime('%B')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_monthly_changes.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Cambios Promedio Mensuales del Ratio")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cambio Promedio Mensual (%)")
    st.pyplot(fig)

    st.write("### Cambios Promedio Anuales del Ratio")
    avg_yearly_changes = ratio.groupby(ratio.index.year).mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_yearly_changes.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Cambios Promedio Anuales del Ratio")
    ax.set_xlabel("Año")
    ax.set_ylabel("Cambio Promedio Anual (%)")
    st.pyplot(fig)

    # Display statistical summary
    st.write("### Resumen Estadístico")
    st.write(ratio.describe())
