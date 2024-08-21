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

# Calculate streaks and dispersion
def calculate_streaks(data):
    positive_streaks = []
    negative_streaks = []
    
    # Calculate monthly changes
    monthly_changes = data['Cambio Mensual (%)'].dropna()
    streak_length = 0
    last_sign = np.sign(monthly_changes.iloc[0])

    for change in monthly_changes:
        current_sign = np.sign(change)
        if current_sign == last_sign:
            streak_length += 1
        else:
            if last_sign > 0:
                positive_streaks.append(streak_length)
            elif last_sign < 0:
                negative_streaks.append(streak_length)
            streak_length = 1
            last_sign = current_sign
    
    # Append the last streak
    if streak_length > 0:
        if last_sign > 0:
            positive_streaks.append(streak_length)
        elif last_sign < 0:
            negative_streaks.append(streak_length)

    return {
        'longest_positive_streak': max(positive_streaks, default=0),
        'longest_negative_streak': max(negative_streaks, default=0),
        'average_positive_streak': np.mean(positive_streaks) if positive_streaks else 0,
        'average_negative_streak': np.mean(negative_streaks) if negative_streaks else 0,
        'max_variation': monthly_changes.max() if not monthly_changes.empty else 0,
        'min_variation': monthly_changes.min() if not monthly_changes.empty else 0,
        'dispersion': np.std(monthly_changes) if not monthly_changes.empty else 0
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
        
        # Calculate monthly price variations
        ratio_data['Month'] = ratio_data.index.to_period('M')
        monthly_data = ratio_data.resample('ME').ffill()
        monthly_data['Cambio Mensual (%)'] = monthly_data['Adjusted Close'].pct_change() * 100
        
        # Calculate yearly price variations
        ratio_data['Year'] = ratio_data.index.to_period('Y')
        yearly_data = ratio_data.resample('YE').ffill()
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
            
            # Fit Gaussian
            mu, std = norm.fit(monthly_changes)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2)
            ax.set_title('Histograma de Variaciones Mensuales con Ajuste de Gauss')
            ax.set_xlabel('Variación Mensual (%)')
            ax.set_ylabel('Densidad')
            
            # Percentiles
            p5 = np.percentile(monthly_changes, 5)
            p95 = np.percentile(monthly_changes, 95)
            ax.axvline(p5, color='r', linestyle='--', label='5º Percentil')
            ax.axvline(p95, color='g', linestyle='--', label='95º Percentil')
            ax.legend()

            st.pyplot(fig)
        else:
            st.warning("No hay datos suficientes para mostrar el histograma.")

        # Heatmap of monthly changes
        st.write("### Mapa de Calor de Variaciones Mensuales")
        monthly_pivot = monthly_data.pivot_table(index=monthly_data.index.year, columns=monthly_data.index.month, values='Cambio Mensual (%)', aggfunc='mean' if metric_option == "Promedio" else 'median')
        
        if not monthly_pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(monthly_pivot, cmap=get_custom_cmap(), annot=True, fmt=".1f", ax=ax, center=0)
            ax.set_title(f"Mapa de Calor de Variaciones Mensuales de {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
            ax.set_xlabel('Mes')
            ax.set_ylabel('Año')
            st.pyplot(fig)
        else:
            st.warning("No hay datos suficientes para mostrar el mapa de calor.")

        # Calculate and display statistics
        st.write("### Estadísticas de Variaciones Mensuales")
        stats = calculate_streaks(monthly_data)
        st.write(f"**Mayor racha positiva:** {stats['longest_positive_streak']} meses")
        st.write(f"**Mayor racha negativa:** {stats['longest_negative_streak']} meses")
        st.write(f"**Promedio de rachas positivas:** {stats['average_positive_streak']:.2f} meses")
        st.write(f"**Promedio de rachas negativas:** {stats['average_negative_streak']:.2f} meses")
        st.write(f"**Variación máxima:** {stats['max_variation']:.2f}%")
        st.write(f"**Variación mínima:** {stats['min_variation']:.2f}%")
        st.write(f"**Dispersion:** {stats['dispersion']:.2f}")

    else:
        st.warning("No se pudo calcular el ratio o no hay datos suficientes.")
