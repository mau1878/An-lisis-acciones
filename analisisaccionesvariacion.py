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

# Function to calculate streaks
def calculate_streaks(data):
    positive_streaks = []
    negative_streaks = []
    
    current_streak = 0
    streak_type = None
    
    for change in data:
        if change > 0:
            if streak_type == 'negative':
                negative_streaks.append(current_streak)
                current_streak = 0
            current_streak += 1
            streak_type = 'positive'
        elif change < 0:
            if streak_type == 'positive':
                positive_streaks.append(current_streak)
                current_streak = 0
            current_streak += 1
            streak_type = 'negative'
        else:
            if streak_type:
                if streak_type == 'positive':
                    positive_streaks.append(current_streak)
                elif streak_type == 'negative':
                    negative_streaks.append(current_streak)
                current_streak = 0
                streak_type = None
    
    # Append the last streak
    if streak_type == 'positive':
        positive_streaks.append(current_streak)
    elif streak_type == 'negative':
        negative_streaks.append(current_streak)
    
    return {
        'longest_positive_streak': max(positive_streaks, default=0),
        'longest_negative_streak': max(negative_streaks, default=0),
        'avg_positive_streak': np.mean(positive_streaks) if positive_streaks else 0,
        'avg_negative_streak': np.mean(negative_streaks) if negative_streaks else 0
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
        # Convert ratio data to DataFrame
        ratio_data = ratio_data.to_frame(name='Adjusted Close')
        ratio_data.index = pd.to_datetime(ratio_data.index)
        
        # Calculate monthly and yearly price variations
        ratio_data['Month'] = ratio_data.index.to_period('M')
        monthly_data = ratio_data.resample('M').ffill()
        monthly_data['Cambio Mensual (%)'] = monthly_data['Adjusted Close'].pct_change() * 100
        
        ratio_data['Year'] = ratio_data.index.to_period('Y')
        yearly_data = ratio_data.resample('Y').ffill()
        yearly_data['Cambio Anual (%)'] = yearly_data['Adjusted Close'].pct_change() * 100
        
        # Plot monthly price variations
        st.write("### Variaciones Mensuales de Precios")
        if not monthly_data['Cambio Mensual (%)'].dropna().empty:
            fig = px.line(monthly_data, x=monthly_data.index, y='Cambio Mensual (%)',
                          title=f"Variaciones Mensuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""),
                          labels={'Cambio Mensual (%)': 'Cambio Mensual (%)'})
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
        else:
            st.warning("No hay datos disponibles para mostrar las variaciones mensuales.")

        # Plot yearly price variations
        st.write("### Variaciones Anuales de Precios")
        if not yearly_data['Cambio Anual (%)'].dropna().empty:
            fig = px.line(yearly_data, x=yearly_data.index, y='Cambio Anual (%)',
                          title=f"Variaciones Anuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""),
                          labels={'Cambio Anual (%)': 'Cambio Anual (%)'})
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig)
        else:
            st.warning("No hay datos disponibles para mostrar las variaciones anuales.")
        
        # Histogram with Gaussian and percentiles
        st.write("### Histograma de Variaciones Mensuales con Ajuste de Gauss")
        monthly_changes = monthly_data['Cambio Mensual (%)'].dropna()

        if not monthly_changes.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(monthly_changes, kde=False, stat="density", color="skyblue", ax=ax, binwidth=2)
            
            # Fit Gaussian distribution
            mu, std = norm.fit(monthly_changes)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2)
            
            # Percentiles with different colors and vertical labels
            percentiles = [5, 95]
            for perc in percentiles:
                perc_value = np.percentile(monthly_changes, perc)
                ax.axvline(perc_value, color='red' if perc == 5 else 'green', linestyle='--', label=f'{perc}° Percentil')
                ax.text(perc_value, ax.get_ylim()[1] * 0.8, f'{perc}° Percentil', color='red' if perc == 5 else 'green', ha='center')
            
            ax.set_title(f'Histograma de Variaciones Mensuales y Ajuste de Gauss de {main_ticker}' + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
            ax.set_xlabel('Cambio Mensual (%)')
            ax.set_ylabel('Densidad')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No hay datos disponibles para mostrar el histograma.")

        # Heatmap of monthly changes
        st.write("### Mapa de Calor de Variaciones Mensuales")
        monthly_pivot = monthly_data.pivot_table(index=monthly_data.index.to_period('Y').astype(str), columns=monthly_data.index.to_period('M').astype(str), values='Cambio Mensual (%)')
        if not monthly_pivot.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(monthly_pivot, cmap=get_custom_cmap(), annot=True, fmt=".1f", ax=ax, center=0)
            ax.set_title(f"Mapa de Calor de Variaciones Mensuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
            ax.set_xlabel('Mes')
            ax.set_ylabel('Año')
            st.pyplot(fig)
        else:
            st.warning("No hay datos suficientes para mostrar el mapa de calor.")
        
        # Monthly and Yearly median/average bar plots
        st.write("### Gráficos de Mediana y Promedio Mensuales y Anuales")
        
        if metric_option == "Promedio":
            monthly_median = monthly_data['Cambio Mensual (%)'].resample('M').mean()
            yearly_median = yearly_data['Cambio Anual (%)'].resample('Y').mean()
            monthly_avg = monthly_data['Cambio Mensual (%)'].resample('M').mean()
            yearly_avg = yearly_data['Cambio Anual (%)'].resample('Y').mean()
        else:
            monthly_median = monthly_data['Cambio Mensual (%)'].resample('M').median()
            yearly_median = yearly_data['Cambio Anual (%)'].resample('Y').median()
            monthly_avg = monthly_data['Cambio Mensual (%)'].resample('M').median()
            yearly_avg = yearly_data['Cambio Anual (%)'].resample('Y').median()

        # Monthly median and average bar plots
        if not monthly_median.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_median.plot(kind='bar', color='orange', ax=ax, label='Mediana Mensual')
            monthly_avg.plot(kind='bar', color='blue', ax=ax, alpha=0.5, label='Promedio Mensual')
            ax.set_title(f'Mediana y Promedio Mensuales de {main_ticker}' + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
            ax.set_xlabel('Mes')
            ax.set_ylabel('Cambio Mensual (%)')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No hay datos suficientes para mostrar el gráfico de mediana mensual.")
        
        # Yearly median and average bar plots
        if not yearly_median.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            yearly_median.plot(kind='bar', color='orange', ax=ax, label='Mediana Anual')
            yearly_avg.plot(kind='bar', color='blue', ax=ax, alpha=0.5, label='Promedio Anual')
            ax.set_title(f'Mediana y Promedio Anuales de {main_ticker}' + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
            ax.set_xlabel('Año')
            ax.set_ylabel('Cambio Anual (%)')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No hay datos suficientes para mostrar el gráfico de mediana anual.")

        # Calculate streaks
        st.write("### Estadísticas de Rachas")
        changes = monthly_data['Cambio Mensual (%)'].dropna().values
        streak_stats = calculate_streaks(changes)
        
        st.write(f"Racha positiva más larga: {streak_stats['longest_positive_streak']} meses")
        st.write(f"Racha negativa más larga: {streak_stats['longest_negative_streak']} meses")
        st.write(f"Racha positiva promedio: {streak_stats['avg_positive_streak']:.2f} meses")
        st.write(f"Racha negativa promedio: {streak_stats['avg_negative_streak']:.2f} meses")

    else:
        st.warning("No se pudo calcular el ratio o no hay datos suficientes.")
