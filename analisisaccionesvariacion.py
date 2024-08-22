import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap

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
def evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ypfd_ratio=False):
    if not main_ticker:
        st.error("El ticker principal no puede estar vacío.")
        return None
    
    main_ticker = main_ticker.upper()

    # Apply YPFD.BA/YPF ratio if the option is activated
    if apply_ypfd_ratio:
        st.write(f"Aplicando la razón YPFD.BA/YPF al ticker {main_ticker}...")
        if 'YPFD.BA' in data and 'YPF' in data:
            result = data[main_ticker]['Adj Close'] / (data['YPFD.BA']['Adj Close'] / data['YPF']['Adj Close'])
        else:
            st.error("No hay datos disponibles para YPFD.BA o YPF.")
            return None
    else:
        result = data[main_ticker]['Adj Close']

    # Process the additional ratio with optional divisors
    if second_ticker and third_ticker:
        second_ticker = second_ticker.upper()
        third_ticker = third_ticker.upper()

        if second_ticker in data:
            if third_ticker in data:
                result /= (data[second_ticker]['Adj Close'] / data[third_ticker]['Adj Close'])
            else:
                st.error(f"El tercer divisor no está disponible en los datos: {third_ticker}")
                return None
        else:
            st.error(f"El segundo divisor no está disponible en los datos: {second_ticker}")
            return None
    elif second_ticker:
        second_ticker = second_ticker.upper()

        if second_ticker in data:
            result /= data[second_ticker]['Adj Close']
        else:
            st.error(f"El segundo divisor no está disponible en los datos: {second_ticker}")
            return None

    return result

# Function to calculate streaks
def calculate_streaks(changes):
    streaks = {'positive': [], 'negative': []}
    current_streak = 0
    streak_type = None
    
    for change in changes:
        if change > 0:
            if streak_type == 'positive':
                current_streak += 1
            else:
                if streak_type == 'negative' and current_streak > 0:
                    streaks['negative'].append(current_streak)
                streak_type = 'positive'
                current_streak = 1
        elif change < 0:
            if streak_type == 'negative':
                current_streak += 1
            else:
                if streak_type == 'positive' and current_streak > 0:
                    streaks['positive'].append(current_streak)
                streak_type = 'negative'
                current_streak = 1
        else:
            if current_streak > 0:
                streaks[streak_type].append(current_streak)
            streak_type = None
            current_streak = 0
    
    if current_streak > 0:
        streaks[streak_type].append(current_streak)
    
    return streaks

# Streamlit app
st.title("Análisis de Precios de Acciones - MTaurus - X: https://x.com/MTaurus_ok")

# User option to apply the YPFD.BA/YPF ratio
apply_ypfd_ratio = st.checkbox("Dividir el ticker principal por dólar CCL de YPF", value=False)

# User inputs
main_ticker = st.text_input("Ingrese el ticker principal (por ejemplo GGAL.BA o METR.BA o AAPL o BMA:")
second_ticker = st.text_input("Ingrese el segundo ticker o ratio divisor (opcional):")
third_ticker = st.text_input("Ingrese el tercer ticker o ratio divisor (opcional):")

start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))
end_date = st.date_input("Seleccione la fecha de fin:", value=pd.to_datetime('today'))

# Option to choose between average and median for monthly and yearly graphs
metric_option = st.radio("Seleccione la métrica para los gráficos mensuales y anuales:", ("Promedio", "Mediana"))

# Extract tickers from the inputs
tickers = {main_ticker, second_ticker, third_ticker}
tickers = {ticker.upper() for ticker in tickers if ticker}
if apply_ypfd_ratio:
    tickers.update({'YPFD.BA', 'YPF'})

data = fetch_data(tickers, start_date, end_date)

if data:
    data = align_dates(data)
    
    # Evaluate ratio
    ratio_data = evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ypfd_ratio)
    
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
                      title=f"Variaciones Mensuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""),
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
        sns.heatmap(monthly_pivot, annot=True, cmap=cmap, center=0, fmt=".1f", linewidths=.5, ax=ax)
        ax.set_title(f"Mapa de Calor de Variaciones Mensuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        ax.set_xlabel("Mes")
        ax.set_ylabel("Año")
        st.pyplot(fig)

        # Calculate streaks
        st.write("### Análisis de Rachas")
        streaks = calculate_streaks(monthly_changes)
        streak_summary = {
            'Total Rachas Positivas': len(streaks['positive']),
            'Total Rachas Negativas': len(streaks['negative']),
            'Mayor Racha Positiva': max(streaks['positive']) if streaks['positive'] else 0,
            'Mayor Racha Negativa': max(streaks['negative']) if streaks['negative'] else 0
        }
        
        st.write("#### Resumen de Rachas")
        for key, value in streak_summary.items():
            st.write(f"{key}: {value}")

        # Rank streaks
        streak_rank = pd.DataFrame({
            'Racha Positiva': streaks['positive'],
            'Racha Negativa': streaks['negative']
        }).stack().reset_index(level=1, drop=True).rename('Racha').sort_values(ascending=False)
        
        st.write("#### Rachas Ordenadas")
        st.write(streak_rank)

        # Plotting streak rank as a bar chart
        st.write("### Ranking de Rachas")
        fig = px.bar(streak_rank, y=streak_rank.index, x='Racha', orientation='h', color=streak_rank.values, 
                     color_continuous_scale=cmap.colors, title="Ranking de Rachas Positivas y Negativas")
        st.plotly_chart(fig)
