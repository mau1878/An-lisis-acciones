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

# Streamlit app
st.title("Análisis de Variación Mensual de Precios de Acciones, ETFs e Índices - MTaurus - X: https://x.com/MTaurus_ok")

# User option to apply the YPFD.BA/YPF ratio
apply_ypfd_ratio = st.checkbox("Dividir el ticker principal por dólar CCL de YPF", value=False)

# User inputs
main_ticker = st.text_input("Ingrese el ticker principal (por ejemplo GGAL.BA o METR.BA o AAPL o BMA:")
second_ticker = st.text_input("Ingrese el segundo ticker o ratio divisor (opcional):")
third_ticker = st.text_input("Ingrese el tercer ticker o ratio divisor (opcional):")

start_date = st.date_input("Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('1990-01-01'))
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
        if isinstance(ratio_data, pd.Series):
            ratio_data = ratio_data.to_frame(name='Adj Close')
        ratio_data.index = pd.to_datetime(ratio_data.index)
        ratio_data.loc[:, 'Month'] = ratio_data.index.to_period('M')
        monthly_data = ratio_data.resample('ME').ffill()
        monthly_data['Cambio Mensual (%)'] = monthly_data['Adj Close'].pct_change() * 100

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

        ax.set_title(f"Histograma de Cambios Mensuales con Ajuste de Gauss para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        ax.set_xlabel("Cambio Mensual (%)")
        ax.set_ylabel("Densidad")
        ax.legend()
        # Add watermark   
        plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey', ha='center', va='center', alpha=0.5, transform=ax.transAxes)
        st.pyplot(fig)

        # Heatmap of monthly variations
        st.write("### Mapa de Calor de Variaciones Mensuales")
        monthly_pivot = monthly_data.pivot_table(values='Cambio Mensual (%)', index=monthly_data.index.year, columns=monthly_data.index.month, aggfunc='mean')
        
        # Define a custom colormap with greens for positive values and reds for negative values
        cmap = get_custom_cmap()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(monthly_pivot, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, center=0, ax=ax)
        plt.title(f"Mapa de Calor de Variaciones Mensuales para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        plt.xlabel("Mes")
        plt.ylabel("Año")
        # Add watermark   
        plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey', ha='center', va='center', alpha=0.5, transform=ax.transAxes)
        st.pyplot(fig)

        # Monthly and yearly average/median changes
        st.write(f"### Cambios {metric_option} Mensuales")
        if metric_option == "Promedio":
            avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].mean()
        else:
            avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].median()
        avg_monthly_changes.index = pd.to_datetime(avg_monthly_changes.index, format='%m').strftime('%B')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_monthly_changes.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f"Cambios {metric_option} Mensuales para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        ax.set_xlabel("Mes")
        ax.set_ylabel(f"{metric_option} de Cambio Mensual (%)")
        # Add watermark   
        plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey', ha='center', va='center', alpha=0.5, transform=ax.transAxes)
        st.pyplot(fig)
        
        st.write(f"### Cambios {metric_option} Anuales")
        if metric_option == "Promedio":
            avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].mean()
        else:
            avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].median()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_yearly_changes.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f"Cambios {metric_option} Anuales para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        ax.set_xlabel("Año")
        ax.set_ylabel(f"{metric_option} de Cambio Anual (%)")
        # Add watermark   
        plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey', ha='center', va='center', alpha=0.5, transform=ax.transAxes)
        st.pyplot(fig)

        # NEW: Rank months by the number of positive and negative values
        st.write("### Ranking de Meses por Número de Valores Positivos y Negativos")
        monthly_positive_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.month).apply(lambda x: (x > 0).sum())
        monthly_negative_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.month).apply(lambda x: (x < 0).sum())
        
        monthly_rank_df = pd.DataFrame({
            'Mes': pd.to_datetime(monthly_positive_count.index, format='%m').strftime('%B'),
            'Positivos': monthly_positive_count.values,
            'Negativos': monthly_negative_count.values
        }).sort_values(by='Positivos', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_rank_df.set_index('Mes')[['Positivos', 'Negativos']].plot(kind='bar', ax=ax)
        ax.set_title(f"Ranking de Meses por Número de Variaciones Intermensuales Positivas y Negativas para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        ax.set_ylabel("Número de Valores")
        # Add watermark   
        plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey', ha='center', va='center', alpha=0.5, transform=ax.transAxes)
        st.pyplot(fig)

        # NEW: Rank years by the number of positive and negative values
        st.write("### Ranking de Años por Número de Valores Positivos y Negativos")
        yearly_positive_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.year).apply(lambda x: (x > 0).sum())
        yearly_negative_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.year).apply(lambda x: (x < 0).sum())
        
        yearly_rank_df = pd.DataFrame({
            'Año': yearly_positive_count.index,
            'Positivos': yearly_positive_count.values,
            'Negativos': yearly_negative_count.values
        }).sort_values(by='Positivos', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_rank_df.set_index('Año')[['Positivos', 'Negativos']].plot(kind='bar', ax=ax)
        ax.set_title(f"Ranking de Años por Número de Variaciones Intermensuales Positivas y Negativas para {main_ticker}" + (f" / {second_ticker}" if second_ticker else "") + (f" / {third_ticker}" if third_ticker else ""))
        ax.set_ylabel("Número de Valores")
        # Add watermark   
        plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey', ha='center', va='center', alpha=0.5, transform=ax.transAxes)
        st.pyplot(fig)
       
        # Existing functions and code
        # Updated calculate_streaks function
        # Updated calculate_streaks function
        # Updated calculate_streaks function
        def calculate_streaks(data):
            streaks = []
            current_streak = {'value': None, 'start': None, 'end': None, 'length': 0}
            
            for i in range(len(data)):
                if current_streak['value'] is None:
                    # Initialize the first streak
                    current_streak['value'] = data[i]
                    current_streak['start'] = i
                    current_streak['end'] = i
                    current_streak['length'] = 1
                elif (data[i] > 0 and current_streak['value'] > 0) or (data[i] <= 0 and current_streak['value'] <= 0):
                    # Continue the current streak if the value direction is the same
                    current_streak['end'] = i
                    current_streak['length'] += 1
                else:
                    # End the current streak and start a new one
                    streaks.append(current_streak)
                    current_streak = {
                        'value': data[i],
                        'start': i,
                        'end': i,
                        'length': 1
                    }
            
            # Append the last streak
            if current_streak['length'] > 0:
                streaks.append(current_streak)
            
            # Convert start and end indices to datetime if data is a Pandas Series with datetime index
            if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
                for streak in streaks:
                    streak['start'] = data.index[streak['start']]
                    streak['end'] = data.index[streak['end']]
                    # Calculate the duration in months
                    streak['length'] = (streak['end'].year - streak['start'].year) * 12 + (streak['end'].month - streak['start'].month) + 1
            
            return streaks
        
        # Apply the function to your data
        # Calculate streaks from the monthly data
        streaks = calculate_streaks(monthly_data['Cambio Mensual (%)'].dropna())
        
        # Ensure the streaks list is correctly structured
        streaks_data = []
        
        for streak in streaks:
            streak_type = 'positive' if streak['value'] > 0 else 'negative'
            streaks_data.append({
                'type': streak_type,
                'start': streak['start'].strftime('%Y-%m'),  # Format to Year-Month
                'end': streak['end'].strftime('%Y-%m'),      # Format to Year-Month
                'length': streak['length']
            })
        
        # Create DataFrame from the streaks list
        streaks_df = pd.DataFrame(streaks_data)
        
        # Check if 'type' column exists before filtering
        if 'type' in streaks_df.columns:
            positive_streaks = streaks_df[streaks_df['type'] == 'positive'].sort_values(by='length', ascending=False)
            negative_streaks = streaks_df[streaks_df['type'] == 'negative'].sort_values(by='length', ascending=False)
        else:
            # Handle the case where the 'type' column is missing
            st.warning("The 'type' column is missing in the streaks DataFrame. No streaks to analyze.")
            positive_streaks = pd.DataFrame()
            negative_streaks = pd.DataFrame()
        
        # Select top 10 longest streaks
        top_positive_streaks = positive_streaks.head(10).reset_index(drop=True)
        top_negative_streaks = negative_streaks.head(10).reset_index(drop=True)
        
        # Display the streaks in a table format
        st.write("#### Rachas Positivas más Largas")
        st.dataframe(top_positive_streaks[['start', 'end', 'length']].rename(columns={'start': 'Inicio', 'end': 'Fin', 'length': 'Duración (meses)'}))
        
        st.write("#### Rachas Negativas más Largas")
        st.dataframe(top_negative_streaks[['start', 'end', 'length']].rename(columns={'start': 'Inicio', 'end': 'Fin', 'length': 'Duración (meses)'}))
