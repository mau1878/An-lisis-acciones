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

# Function to calculate longest streaks
def calculate_streaks(series):
    streaks = []
    current_streak = {'type': None, 'length': 0, 'start': None, 'end': None}

    for date, value in series.items():
        streak_type = 'positive' if value > 0 else 'negative'
        
        if current_streak['type'] is None:
            current_streak['type'] = streak_type
            current_streak['start'] = date
            current_streak['length'] = 1
        elif current_streak['type'] == streak_type:
            current_streak['length'] += 1
            current_streak['end'] = date
        else:
            streaks.append(current_streak.copy())
            current_streak = {'type': streak_type, 'length': 1, 'start': date, 'end': date}

    if current_streak['length'] > 0:
        streaks.append(current_streak)

    return sorted(streaks, key=lambda x: x['length'], reverse=True)

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
        # Calculate monthly and yearly price variations
        ratio_data = ratio_data.to_frame(name='Adjusted Close')
        ratio_data.index = pd.to_datetime(ratio_data.index)
        ratio_data['Month'] = ratio_data.index.to_period('M')
        ratio_data['Year'] = ratio_data.index.to_period('Y')
        
        # Monthly and yearly variations
        monthly_data = ratio_data.resample('M').ffill()
        monthly_data['Cambio Mensual (%)'] = monthly_data['Adjusted Close'].pct_change() * 100
        
        yearly_data = ratio_data.resample('Y').ffill()
        yearly_data['Cambio Anual (%)'] = yearly_data['Adjusted Close'].pct_change() * 100
        
        # Monthly bar plot
        st.write("### Variaciones Mensuales de Precios")
        fig = px.bar(monthly_data, x=monthly_data.index, y='Cambio Mensual (%)',
                      title=f"Variaciones Mensuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""),
                      labels={'Cambio Mensual (%)': 'Cambio Mensual (%)'})
        st.plotly_chart(fig)

        # Yearly bar plot
        st.write("### Variaciones Anuales de Precios")
        fig_yearly = px.bar(yearly_data, x=yearly_data.index, y='Cambio Anual (%)',
                           title=f"Variaciones Anuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""),
                           labels={'Cambio Anual (%)': 'Cambio Anual (%)'})
        st.plotly_chart(fig_yearly)

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
            ax.axvline(perc_value, color=colors[i], linestyle='--')
            ax.text(perc_value, ax.get_ylim()[1] * 0.9, f'{percentile}th Percentile', color=colors[i], rotation=90)

        plt.title(f'Histograma de Cambios Mensuales de {main_ticker}')
        plt.xlabel('Cambio Mensual (%)')
        plt.ylabel('Densidad')
        st.pyplot()

        # Heatmap of monthly changes
        st.write("### Mapa de Calor de Variaciones Mensuales")
        monthly_pivot = monthly_data.pivot_table(values='Cambio Mensual (%)', index=monthly_data.index.year, columns=monthly_data.index.month)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(monthly_pivot, cmap=get_custom_cmap(), annot=True, fmt=".1f", linewidths=0.5)
        plt.title(f"Mapa de Calor de Variaciones Mensuales de {main_ticker}")
        plt.xlabel("Mes")
        plt.ylabel("Año")
        st.pyplot()

        # Calculate and display longest streaks
        st.write("### Rachas de Cambios Positivos y Negativos Más Largas")

        streaks = calculate_streaks(monthly_data['Cambio Mensual (%)'])

        longest_positive_streak = next((s for s in streaks if s['type'] == 'positive'), None)
        longest_negative_streak = next((s for s in streaks if s['type'] == 'negative'), None)

        if longest_positive_streak:
            st.write(f"**Racha Positiva Más Larga:**")
            st.write(f"Inicio: {longest_positive_streak['start'].strftime('%Y-%m-%d')}")
            st.write(f"Fin: {longest_positive_streak['end'].strftime('%Y-%m-%d')}")
            st.write(f"Duración: {longest_positive_streak['length']} meses")

        if longest_negative_streak:
            st.write(f"**Racha Negativa Más Larga:**")
            st.write(f"Inicio: {longest_negative_streak['start'].strftime('%Y-%m-%d')}")
            st.write(f"Fin: {longest_negative_streak['end'].strftime('%Y-%m-%d')}")
            st.write(f"Duración: {longest_negative_streak['length']} meses")

        st.write("### Ranking de Rachas Positivas y Negativas")

        if streaks:
            df_streaks = pd.DataFrame(streaks)
            df_streaks['Start Date'] = df_streaks['start']
            df_streaks['End Date'] = df_streaks['end']
            df_streaks['Length'] = df_streaks['length']
            df_streaks.drop(['start', 'end'], axis=1, inplace=True)
            st.write(df_streaks)
