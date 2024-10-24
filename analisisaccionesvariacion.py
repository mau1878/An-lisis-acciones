import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap

# Configuración de la página de Streamlit
st.set_page_config(
  page_title="Análisis de Variación Mensual de Precios",
  layout="wide",
  initial_sidebar_state="expanded",
)

# Definir un colormap personalizado
def get_custom_cmap():
  colors = ['red', 'white', 'green']
  return LinearSegmentedColormap.from_list('custom_diverging', colors)

# Función para descargar y procesar los datos de yfinance
def fetch_data(tickers, start_date, end_date):
  data = {}
  for ticker in tickers:
      ticker = ticker.upper()
      st.write(f"Intentando descargar datos para el ticker **{ticker}**...")
      try:
          df = yf.download(ticker, start=start_date, end=end_date)
          if df.empty:
              st.warning(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas seleccionado.")
          else:
              # Aplanar columnas si tienen múltiples niveles
              if isinstance(df.columns, pd.MultiIndex):
                  df.columns = df.columns.get_level_values(0)
              data[ticker] = df
              st.write(f"**Columnas para {ticker}:** {df.columns.tolist()}")
      except Exception as e:
          st.error(f"Error al descargar datos para el ticker **{ticker}**: {e}")
  return data

# Función para alinear las fechas y rellenar valores faltantes
def align_dates(data):
  if not data:
      return {}
  
  first_ticker = list(data.keys())[0]
  first_ticker_dates = data[first_ticker].index
  
  for ticker in data:
      data[ticker] = data[ticker].reindex(first_ticker_dates)
      data[ticker] = data[ticker].ffill()  # Rellenar valores faltantes hacia adelante
  
  return data

# Función para evaluar el ratio según los tickers proporcionados
def evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ypfd_ratio=False):
  if not main_ticker:
      st.error("El ticker principal no puede estar vacío.")
      return None
  
  main_ticker = main_ticker.upper()

  # Aplicar la razón YPFD.BA/YPF si la opción está activada
  if apply_ypfd_ratio:
      st.write(f"Aplicando la razón YPFD.BA/YPF al ticker **{main_ticker}**...")
      if 'YPFD.BA' in data and 'YPF' in data:
          if 'Adj Close' in data['YPF'].columns and 'Adj Close' in data['YPFD.BA'].columns:
              result = data[main_ticker]['Adj Close'] / (data['YPFD.BA']['Adj Close'] / data['YPF']['Adj Close'])
          else:
              st.error("No hay datos disponibles para 'Adj Close' en **YPFD.BA** o **YPF**.")
              return None
      else:
          st.error("No hay datos disponibles para **YPFD.BA** o **YPF**.")
          return None
  else:
      if 'Adj Close' not in data[main_ticker].columns:
          st.error(f"'Adj Close' no está disponible para el ticker **{main_ticker}**.")
          return None
      result = data[main_ticker]['Adj Close']

  # Procesar el ratio adicional con divisores opcionales
  if second_ticker and third_ticker:
      second_ticker = second_ticker.upper()
      third_ticker = third_ticker.upper()

      if second_ticker in data:
          if third_ticker in data:
              if 'Adj Close' in data[second_ticker].columns and 'Adj Close' in data[third_ticker].columns:
                  divisor = data[second_ticker]['Adj Close'] / data[third_ticker]['Adj Close']
                  result = result / divisor
              else:
                  st.error(f"'Adj Close' no está disponible para **{third_ticker}**.")
                  return None
          else:
              st.error(f"El tercer divisor no está disponible en los datos: **{third_ticker}**.")
              return None
      else:
          st.error(f"El segundo divisor no está disponible en los datos: **{second_ticker}**.")
          return None
  elif second_ticker:
      second_ticker = second_ticker.upper()

      if second_ticker in data:
          if 'Adj Close' in data[second_ticker].columns:
              result = result / data[second_ticker]['Adj Close']
          else:
              st.error(f"'Adj Close' no está disponible para el segundo ticker: **{second_ticker}**.")
              return None
      else:
          st.error(f"El segundo divisor no está disponible en los datos: **{second_ticker}**.")
          return None

  return result

# Función para calcular rachas positivas y negativas
def calculate_streaks(data):
  streaks = []
  current_streak = {'value': None, 'start': None, 'end': None, 'length': 0}
  
  for i in range(len(data)):
      if current_streak['value'] is None:
          # Inicializar la primera racha
          current_streak['value'] = data[i]
          current_streak['start'] = i
          current_streak['end'] = i
          current_streak['length'] = 1
      elif (data[i] > 0 and current_streak['value'] > 0) or (data[i] <= 0 and current_streak['value'] <= 0):
          # Continuar la racha actual si la dirección del valor es la misma
          current_streak['end'] = i
          current_streak['length'] += 1
      else:
          # Terminar la racha actual y empezar una nueva
          streaks.append(current_streak)
          current_streak = {
              'value': data[i],
              'start': i,
              'end': i,
              'length': 1
          }
  
  # Añadir la última racha
  if current_streak['length'] > 0:
      streaks.append(current_streak)
  
  # Convertir los índices de inicio y fin a datetime si los datos tienen un índice datetime
  if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
      for streak in streaks:
          streak['start'] = data.index[streak['start']]
          streak['end'] = data.index[streak['end']]
          # Calcular la duración en meses
          streak['length'] = (streak['end'].year - streak['start'].year) * 12 + (streak['end'].month - streak['start'].month) + 1
  
  return streaks

# Título de la aplicación
st.title("📈 Análisis de Variación Mensual de Precios de Acciones, ETFs e Índices - MTaurus")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Opciones del usuario para aplicar la razón YPFD.BA/YPF
apply_ypfd_ratio = st.checkbox("🔄 Dividir el ticker principal por dólar CCL de YPF", value=False)

# Entradas del usuario
main_ticker = st.text_input("🖊️ Ingrese el ticker principal (por ejemplo: GGAL.BA, METR.BA, AAPL, BMA):")
second_ticker = st.text_input("➕ Ingrese el segundo ticker o ratio divisor (opcional):")
third_ticker = st.text_input("➕ Ingrese el tercer ticker o ratio divisor (opcional):")

# Selección de fechas
start_date = st.date_input("📅 Seleccione la fecha de inicio:", value=pd.to_datetime('2010-01-01'), min_value=pd.to_datetime('1990-01-01'))
end_date = st.date_input("📅 Seleccione la fecha de fin:", value=pd.to_datetime('today'))

# Opción para elegir entre promedio y mediana para los gráficos
metric_option = st.radio("📊 Seleccione la métrica para los gráficos mensuales y anuales:", ("Promedio", "Mediana"))

# Extraer tickers de las entradas
tickers = {main_ticker, second_ticker, third_ticker}
tickers = {ticker.upper() for ticker in tickers if ticker}
if apply_ypfd_ratio:
  tickers.update({'YPFD.BA', 'YPF'})

# Descargar datos
data = fetch_data(tickers, start_date, end_date)

# Procesar y analizar los datos si se han descargado correctamente
if data:
  # Alinear fechas y rellenar valores faltantes
  data = align_dates(data)
  
  # Evaluar el ratio según los tickers seleccionados
  ratio_data = evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ypfd_ratio)
  
  if ratio_data is not None:
      # Convertir a DataFrame si es una Serie
      if isinstance(ratio_data, pd.Series):
          ratio_data = ratio_data.to_frame(name='Adj Close')
      
      ratio_data.index = pd.to_datetime(ratio_data.index)
      
      # Crear una copia del DataFrame para evitar SettingWithCopyWarning
      ratio_data = ratio_data.copy()
      ratio_data['Month'] = ratio_data.index.to_period('M')
      
      # Resamplear a mensual y rellenar hacia adelante
      monthly_data = ratio_data.resample('M').ffill()
      
      # Verificar si 'Adj Close' está presente después del resampleo
      if 'Adj Close' not in monthly_data.columns:
          st.error("'Adj Close' no está disponible en los datos después del resampleo.")
      else:
          # Calcular el cambio mensual en porcentaje
          monthly_data['Cambio Mensual (%)'] = monthly_data['Adj Close'].pct_change() * 100

          # -------------------------------
          # Visualizaciones y Análisis
          # -------------------------------
          
          # 1. Variaciones Mensuales de Precios
          st.write("### 📉 Variaciones Mensuales de Precios")
          fig = px.line(
              monthly_data,
              x=monthly_data.index,
              y='Cambio Mensual (%)',
              title=f"Variaciones Mensuales de {main_ticker}" +
                    (f" / {second_ticker}" if second_ticker else "") +
                    (f" / {third_ticker}" if third_ticker else ""),
              labels={'Cambio Mensual (%)': 'Cambio Mensual (%)'}
          )
          fig.update_traces(mode='lines+markers')
          st.plotly_chart(fig, use_container_width=True)

          # 2. Histograma de Variaciones Mensuales con Ajuste de Gauss
          st.write("### 📊 Histograma de Variaciones Mensuales con Ajuste de Gauss")
          monthly_changes = monthly_data['Cambio Mensual (%)'].dropna()

          fig, ax = plt.subplots(figsize=(10, 6))
          sns.histplot(monthly_changes, kde=False, stat="density", color="skyblue", ax=ax, binwidth=2)
          
          # Ajuste de distribución Gaussiana
          mu, std = norm.fit(monthly_changes)
          xmin, xmax = ax.get_xlim()
          x = np.linspace(xmin, xmax, 100)
          p = norm.pdf(x, mu, std)
          ax.plot(x, p, 'k', linewidth=2)

          # Cálculo y plot de percentiles con diferentes colores y etiquetas verticales
          percentiles = [5, 25, 50, 75, 95]
          colors = ['red', 'orange', 'green', 'blue', 'purple']
          for i, percentile in enumerate(percentiles):
              perc_value = np.percentile(monthly_changes, percentile)
              ax.axvline(perc_value, color=colors[i], linestyle='--', label=f'{percentile}º Percentil')
              ax.text(perc_value, ax.get_ylim()[1]*0.9, f'{perc_value:.2f}', color=colors[i],
                      rotation=90, verticalalignment='center', horizontalalignment='right')

          ax.set_title(f"Histograma de Cambios Mensuales con Ajuste de Gauss para {main_ticker}" +
                       (f" / {second_ticker}" if second_ticker else "") +
                       (f" / {third_ticker}" if third_ticker else ""))
          ax.set_xlabel("Cambio Mensual (%)")
          ax.set_ylabel("Densidad")
          ax.legend()
          # Añadir watermark
          plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
                   ha='center', va='center', alpha=0.5, transform=ax.transAxes)
          st.pyplot(fig)

          # 3. Mapa de Calor de Variaciones Mensuales
          st.write("### 🔥 Mapa de Calor de Variaciones Mensuales")
          monthly_pivot = monthly_data.pivot_table(
              values='Cambio Mensual (%)',
              index=monthly_data.index.year,
              columns=monthly_data.index.month,
              aggfunc='mean'
          )
          
          # Definir un colormap personalizado (rojo para negativo, verde para positivo)
          cmap = get_custom_cmap()
          
          fig, ax = plt.subplots(figsize=(12, 8))
          sns.heatmap(
              monthly_pivot,
              cmap=cmap,
              annot=True,
              fmt=".2f",
              linewidths=0.5,
              center=0,
              ax=ax
          )
          plt.title(f"Mapa de Calor de Variaciones Mensuales para {main_ticker}" +
                    (f" / {second_ticker}" if second_ticker else "") +
                    (f" / {third_ticker}" if third_ticker else ""))
          plt.xlabel("Mes")
          plt.ylabel("Año")
          # Añadir watermark
          plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
                   ha='center', va='center', alpha=0.5, transform=ax.transAxes)
          st.pyplot(fig)

          # 4. Cambios Mensuales y Anuales Promedio o Mediana
          st.write(f"### 📈 Cambios {metric_option} Mensuales")
          if metric_option == "Promedio":
              avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].mean()
          else:
              avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].median()
          avg_monthly_changes.index = pd.to_datetime(avg_monthly_changes.index, format='%m').strftime('%B')

          fig, ax = plt.subplots(figsize=(12, 6))
          avg_monthly_changes.plot(kind='bar', color='skyblue', ax=ax)
          ax.set_title(f"Cambios {metric_option} Mensuales para {main_ticker}" +
                       (f" / {second_ticker}" if second_ticker else "") +
                       (f" / {third_ticker}" if third_ticker else ""))
          ax.set_xlabel("Mes")
          ax.set_ylabel(f"{metric_option} de Cambio Mensual (%)")
          # Añadir watermark
          plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
                   ha='center', va='center', alpha=0.5, transform=ax.transAxes)
          st.pyplot(fig)
          
          st.write(f"### 📈 Cambios {metric_option} Anuales")
          if metric_option == "Promedio":
              avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].mean()
          else:
              avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].median()
          
          fig, ax = plt.subplots(figsize=(14, 6))
          avg_yearly_changes.plot(kind='bar', color='skyblue', ax=ax)
          ax.set_title(f"Cambios {metric_option} Anuales para {main_ticker}" +
                       (f" / {second_ticker}" if second_ticker else "") +
                       (f" / {third_ticker}" if third_ticker else ""))
          ax.set_xlabel("Año")
          ax.set_ylabel(f"{metric_option} de Cambio Anual (%)")
          # Añadir watermark
          plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
                   ha='center', va='center', alpha=0.5, transform=ax.transAxes)
          st.pyplot(fig)

          # 5. Ranking de Meses por Número de Valores Positivos y Negativos
          st.write("### 📊 Ranking de Meses por Número de Valores Positivos y Negativos")
          monthly_positive_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.month).apply(lambda x: (x > 0).sum())
          monthly_negative_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.month).apply(lambda x: (x < 0).sum())
          
          monthly_rank_df = pd.DataFrame({
              'Mes': pd.to_datetime(monthly_positive_count.index, format='%m').strftime('%B'),
              'Positivos': monthly_positive_count.values,
              'Negativos': monthly_negative_count.values
          }).sort_values(by='Positivos', ascending=False)
          
          fig, ax = plt.subplots(figsize=(14, 8))
          monthly_rank_df.set_index('Mes')[['Positivos', 'Negativos']].plot(kind='bar', ax=ax)
          ax.set_title(f"Ranking de Meses por Número de Variaciones Intermensuales Positivas y Negativas para {main_ticker}" +
                       (f" / {second_ticker}" if second_ticker else "") +
                       (f" / {third_ticker}" if third_ticker else ""))
          ax.set_ylabel("Número de Valores")
          # Añadir watermark
          plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
                   ha='center', va='center', alpha=0.5, transform=ax.transAxes)
          st.pyplot(fig)

          # 6. Ranking de Años por Número de Valores Positivos y Negativos
          st.write("### 📊 Ranking de Años por Número de Valores Positivos y Negativos")
          yearly_positive_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.year).apply(lambda x: (x > 0).sum())
          yearly_negative_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.year).apply(lambda x: (x < 0).sum())
          
          yearly_rank_df = pd.DataFrame({
              'Año': yearly_positive_count.index,
              'Positivos': yearly_positive_count.values,
              'Negativos': yearly_negative_count.values
          }).sort_values(by='Positivos', ascending=False)
          
          fig, ax = plt.subplots(figsize=(14, 8))
          yearly_rank_df.set_index('Año')[['Positivos', 'Negativos']].plot(kind='bar', ax=ax)
          ax.set_title(f"Ranking de Años por Número de Variaciones Intermensuales Positivas y Negativas para {main_ticker}" +
                       (f" / {second_ticker}" if second_ticker else "") +
                       (f" / {third_ticker}" if third_ticker else ""))
          ax.set_ylabel("Número de Valores")
          # Añadir watermark
          plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
                   ha='center', va='center', alpha=0.5, transform=ax.transAxes)
          st.pyplot(fig)
         
          # 7. Rachas Positivas y Negativas más Largas
          st.write("### 📈 Rachas Positivas y Negativas más Largas")
          
          # Calcular rachas desde los datos mensuales
          streaks = calculate_streaks(monthly_data['Cambio Mensual (%)'].dropna())
          
          # Procesar las rachas para crear un DataFrame
          streaks_data = []
          for streak in streaks:
              streak_type = 'Positive' if streak['value'] > 0 else 'Negative'
              streaks_data.append({
                  'Type': streak_type,
                  'Start': streak['start'].strftime('%Y-%m'),  # Formato Año-Mes
                  'End': streak['end'].strftime('%Y-%m'),      # Formato Año-Mes
                  'Duration (months)': streak['length']
              })
          
          # Crear DataFrame de rachas
          streaks_df = pd.DataFrame(streaks_data)
          
          # Verificar si la columna 'Type' existe antes de filtrar
          if 'Type' in streaks_df.columns:
              positive_streaks = streaks_df[streaks_df['Type'] == 'Positive'].sort_values(by='Duration (months)', ascending=False)
              negative_streaks = streaks_df[streaks_df['Type'] == 'Negative'].sort_values(by='Duration (months)', ascending=False)
          else:
              st.warning("La columna 'Type' no existe en el DataFrame de rachas. No hay rachas para analizar.")
              positive_streaks = pd.DataFrame()
              negative_streaks = pd.DataFrame()
          
          # Seleccionar las 10 rachas más largas
          top_positive_streaks = positive_streaks.head(10).reset_index(drop=True)
          top_negative_streaks = negative_streaks.head(10).reset_index(drop=True)
          
          # Mostrar las rachas en tablas
          st.write("#### 🔝 **10 Rachas Positivas más Largas**")
          if not top_positive_streaks.empty:
              st.dataframe(top_positive_streaks[['Start', 'End', 'Duration (months)']].rename(
                  columns={'Start': 'Inicio', 'End': 'Fin', 'Duration (months)': 'Duración (meses)'}
              ))
          else:
              st.write("No hay rachas positivas para mostrar.")
          
          st.write("#### 🔻 **10 Rachas Negativas más Largas**")
          if not top_negative_streaks.empty:
              st.dataframe(top_negative_streaks[['Start', 'End', 'Duration (months)']].rename(
                  columns={'Start': 'Inicio', 'End': 'Fin', 'Duration (months)': 'Duración (meses)'}
              ))
          else:
              st.write("No hay rachas negativas para mostrar.")

# Información adicional o footer (opcional)
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")
