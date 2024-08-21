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
def evaluate_ratio(main_ticker, second_ticker, third_ticker, data):
    if not main_ticker:
        st.error("El ticker principal no puede estar vacío.")
        return None

    # Process the ratio with optional divisors
    if second_ticker and third_ticker:
        main_ticker = main_ticker.upper()
        second_ticker = second_ticker.upper()
        third_ticker = third_ticker.upper()

        if main_ticker in data:
            result = data[main_ticker]['Adj Close']
            if second_ticker in data:
                if third_ticker in data:
                    result /= (data[second_ticker]['Adj Close'] / data[third_ticker]['Adj Close'])
                else:
                    st.error(f"El tercer divisor no está disponible en los datos: {third_ticker}")
                    return None
            else:
                st.error(f"El segundo divisor no está disponible en los datos: {second_ticker}")
                return None
        else:
            st.error(f"El ticker principal no está disponible en los datos: {main_ticker}")
            return None
    elif second_ticker:
        main_ticker = main_ticker.upper()
        second_ticker = second_ticker.upper()

        if main_ticker in data:
            result = data[main_ticker]['Adj Close']
            if second_ticker in data:
                result /= data[second_ticker]['Adj Close']
            else:
                st.error(f"El segundo divisor no está disponible en los datos: {second_ticker}")
                return None
        else:
            st.error(f"El ticker principal no está disponible en los datos: {main_ticker}")
            return None
    else:
        main_ticker = main_ticker.upper()
        if main_ticker in data:
            result = data[main_ticker]['Adj Close']
        else:
            st.error(f"El ticker principal no está disponible en los datos: {main_ticker}")
            return None

    return result

# Calculate streaks and variations
def calculate_streaks_and_variations(data):
    daily_changes = data['Adjusted Close'].pct_change() * 100
    positive_streaks = []
    negative_streaks = []
    current_streak = 0
    current_streak_type = None
    
    for change in daily_changes:
        if change > 0:
            if current_streak_type == 'positive':
                current_streak += 1
            else:
                if current_streak_type == 'negative':
                    negative_streaks.append(current_streak)
                current_streak = 1
                current_streak_type = 'positive'
        elif change < 0:
            if current_streak_type == 'negative':
                current_streak += 1
            else:
                if current_streak_type == 'positive':
                    positive_streaks.append(current_streak)
                current_streak = 1
                current_streak_type = 'negative'
        else:
            if current_streak_type == 'positive':
                positive_streaks.append(current_streak)
            elif current_streak_type == 'negative':
                negative_streaks.append(current_streak)
            current_streak = 0
            current_streak_type = None
    
    if current_streak_type == 'positive':
        positive_streaks.append(current_streak)
    elif current_streak_type == 'negative':
        negative_streaks.append(current_streak)
    
    max_positive_streak = max(positive_streaks) if positive_streaks else 0
    max_negative_streak = max(negative_streaks) if negative_streaks else 0
    avg_positive_streak = np.mean(positive_streaks) if positive_streaks else 0
    avg_negative_streak = np.mean(negative_streaks) if negative_streaks else 0
    max_variation = daily_changes.max()
    min_variation = daily_changes.min()
    dispersion = daily_changes.std()

    return {
        'Max Positive Streak': max_positive_streak,
        'Max Negative Streak': max_negative_streak,
        'Avg Positive Streak': avg_positive_streak,
        'Avg Negative Streak': avg_negative_streak,
        'Max Variation': max_variation,
        'Min Variation': min_variation,
        'Dispersion': dispersion
    }

# Streamlit app
st.title("Análisis de Precios de Acciones")

# User inputs
main_ticker = st.text_input("Ingrese el ticker principal:")
second_ticker = st.text_input("Ingrese el segundo ticker o ratio divisor (opcional):")
third_ticker = st.text_input("Ingrese el tercer ticker o ratio divisor (opcional):")

start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('2000-01-01'))
end_date = st.date_input("Seleccione la fecha de fin:", value=pd.to_datetime('today'))

# Option to choose between average and median for monthly and yearly graphs
metric_option = st.radio("Seleccione la métrica para los gráficos mensuales y anuales:", ("Promedio", "Mediana"))

# Extract tickers from the inputs
tickers = {main_ticker, second_ticker, third_ticker}
tickers = {ticker.upper() for ticker in tickers if ticker}

data = fetch_data(tickers, start_date, end_date)

if data:
    data = align_dates(data)
    
    # Evaluate ratio
    ratio_data = evaluate_ratio(main_ticker, second_ticker, third_ticker, data)
    
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

        ax.set_title(f"Histograma de Variaciones Mensuales con Ajuste de Gauss")
        ax.set_xlabel("Cambio Mensual (%)")
        ax.set_ylabel("Densidad")
        ax.legend()
        st.pyplot(fig)

        # Heatmap for monthly changes
        st.write("### Mapa de Calor de Variaciones Mensuales")
        heatmap_data = monthly_data.pivot_table(index=monthly_data.index.to_period('M').strftime('%Y-%m'), columns='Month', values='Cambio Mensual (%)')
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap=get_custom_cmap(), annot=False, fmt=".1f", linewidths=0.5)
        plt.title(f"Mapa de Calor de Variaciones Mensuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        plt.xlabel("Mes")
        plt.ylabel("Año-Mes")
        st.pyplot(fig)

        # Calculate and display additional metrics
        st.write("### Métricas Adicionales")
        metrics = calculate_streaks_and_variations(ratio_data)
        for metric, value in metrics.items():
            st.write(f"{metric}: {value:.2f}")

        # Monthly and yearly average/median changes
        st.write("### Cambios Promedio y Mediana Anuales")
        yearly_data = ratio_data.resample('Y').ffill()
        if metric_option == "Promedio":
            avg_yearly_changes = yearly_data.groupby(yearly_data.index.year)['Adjusted Close'].pct_change().mean() * 100
        else:
            avg_yearly_changes = yearly_data.groupby(yearly_data.index.year)['Adjusted Close'].pct_change().median() * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        avg_yearly_changes.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f"Cambios Promedio {metric_option} Anuales")
        ax.set_xlabel("Año")
        ax.set_ylabel(f"Cambio {metric_option} Anual (%)")
        st.pyplot(fig)
