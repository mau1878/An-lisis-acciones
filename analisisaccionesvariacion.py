import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
import requests
from datetime import datetime
import logging
import urllib3
from curl_cffi import requests as cffi_requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis de Variación de Precios",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Utility Functions
def get_custom_cmap(color_order='red_white_green'):
    if color_order == 'red_white_green':
        colors = ['red', 'white', 'green']
    else:  # green_white_red
        colors = ['green', 'white', 'red']
    return LinearSegmentedColormap.from_list('custom_diverging', colors)

def ajustar_precios_por_splits(df, ticker):
    """
    Ajusta los precios en caso de splits. Por ahora, esta función no realiza ajustes.
    Se puede implementar según las necesidades específicas.
    """
    return df

def validate_ticker_format(ticker, data_source):
    if not ticker:
        return True  # Tickers vacíos están permitidos (para campos opcionales)
    ticker = ticker.upper()
    return True  # Permitir cualquier formato de ticker para todas las fuentes

# Data Download Functions
def descargar_datos_yfinance(ticker, start, end):
    try:
        session = cffi_requests.Session(impersonate="chrome124")
        stock_data = yf.download(ticker, start=start, end=end, progress=False, session=session)
        if stock_data.empty:
            logger.warning(f"No se encontraron datos para el ticker {ticker} en el rango de fechas seleccionado.")
            return pd.DataFrame()
        stock_data = stock_data.reset_index()
        if isinstance(stock_data.columns, pd.MultiIndex):
            if ('Close', ticker) in stock_data.columns:
                close_price = stock_data[('Close', ticker)]
                var_name = ticker.replace('.', '_')
                df = pd.DataFrame({
                    'Date': stock_data['Date'],
                    var_name: close_price
                })
            elif ('Adj Close', ticker) in stock_data.columns:
                close_price = stock_data[('Adj Close', ticker)]
                var_name = ticker.replace('.', '_')
                df = pd.DataFrame({
                    'Date': stock_data['Date'],
                    var_name: close_price
                })
            else:
                logger.error(f"No se encontró 'Close' o 'Adj Close' para el ticker {ticker}.")
                return pd.DataFrame()
        else:
            if 'Close' in stock_data.columns:
                close_col = 'Close'
            elif 'Adj Close' in stock_data.columns:
                close_col = 'Adj Close'
            else:
                logger.error(f"No se encontró 'Close' o 'Adj Close' para el ticker {ticker}.")
                return pd.DataFrame()
            var_name = ticker.replace('.', '_')
            df = pd.DataFrame({
                'Date': stock_data['Date'],
                var_name: stock_data[close_col]
            })
        df = ajustar_precios_por_splits(df, ticker)
        return df
    except Exception as e:
        logger.error(f"Error al descargar datos para el ticker {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        cookies = {
            'ChyrpSession': '0e2b2109d60de6da45154b542afb5768',
            'i18next': 'es',
            'PHPSESSID': '5b8da4e0d96ab5149f4973232931f033',
        }
        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'dnt': '1',
            'referer': 'https://analisistecnico.com.ar/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }
        symbol = ticker.replace('.BA', '')
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }
        response = requests.get(
            'https://analisistecnico.com.ar/services/datafeed/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('s') != 'ok':
                logger.error(f"Error en la respuesta de la API para {ticker}: {data.get('s')}")
                return pd.DataFrame()
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c'],
                'High': data['h'],
                'Low': data['l'],
                'Open': data['o'],
                'Volume': data['v']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df = ajustar_precios_por_splits(df, ticker)
            var_name = ticker.replace('.', '_')
            df = df[['Date', 'Close']].rename(columns={'Close': var_name})
            return df
        else:
            logger.error(f"Error al obtener datos para el ticker {ticker}: Código de estado {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error al descargar datos de analisistecnico para el ticker {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_iol(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        cookies = {
            'intencionApertura': '0',
            '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1',
            'isLogged': '1',
            'uid': '1107644',
        }
        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'referer': 'https://iol.invertironline.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }
        symbol = ticker.replace('.BA', '')
        params = {
            'symbolName': symbol,
            'exchange': 'BCBA',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
            'resolution': 'D',
        }
        response = requests.get(
            'https://iol.invertironline.com/api/cotizaciones/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('status') != 'ok' or 'bars' not in data:
                logger.error(f"Error en la respuesta de la API para {ticker}: {data.get('status')}")
                return pd.DataFrame()
            bars = data['bars']
            df = pd.DataFrame(bars)
            df = df.rename(columns={
                'time': 'Date',
                'close': 'Close',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['Date'], unit='s')
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df = ajustar_precios_por_splits(df, ticker)
            var_name = ticker.replace('.', '_')
            df = df[['Date', 'Close']].rename(columns={'Close': var_name})
            return df
        else:
            logger.error(f"Error al obtener datos para el ticker {ticker}: Código de estado {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error al descargar datos de IOL para el ticker {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_byma(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        cookies = {
            'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338',
            '_fbp': 'fb.2.1728347943669.954945632708052302',
        }
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,es-AR;q=0.8,es;q=0.7,en-DE;q=0.6,en;q=0.5,en-US;q=0.4',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': 'https://open.bymadata.com.ar/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }
        symbol = ticker.replace('.BA', '') + ' 24HS'
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(
            'https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
            params=params,
            cookies=cookies,
            headers=headers,
            verify=False
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('s') != 'ok':
                logger.error(f"Error en la respuesta de la API para {ticker}: {data.get('s')}")
                return pd.DataFrame()
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c'],
                'High': data['h'],
                'Low': data['l'],
                'Open': data['o'],
                'Volume': data['v']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df = ajustar_precios_por_splits(df, ticker)
            var_name = ticker.replace('.', '_')
            df = df[['Date', 'Close']].rename(columns={'Close': var_name})
            return df
        else:
            logger.error(f"Error al obtener datos para el ticker {ticker}: Código de estado {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error al descargar datos de ByMA para el ticker {ticker}: {e}")
        return pd.DataFrame()

# Data Processing Functions
def fetch_data(tickers, start_date, end_date, data_source='yfinance'):
    data = {}
    for ticker in tickers:
        ticker = ticker.upper()
        st.write(f"Intentando descargar datos para el ticker **{ticker}** desde {data_source}...")
        try:
            if data_source == 'yfinance':
                df = descargar_datos_yfinance(ticker, start_date, end_date)
            elif data_source == 'analisistecnico':
                df = descargar_datos_analisistecnico(ticker, start_date, end_date)
            elif data_source == 'iol':
                df = descargar_datos_iol(ticker, start_date, end_date)
            elif data_source == 'byma':
                df = descargar_datos_byma(ticker, start_date, end_date)
            else:
                st.error(f"Fuente de datos desconocida: {data_source}")
                continue
            if df.empty:
                st.warning(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas seleccionado.")
            else:
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                data[ticker] = df
                st.write(f"**Columnas para {ticker}:** {df.columns.tolist()}")
        except Exception as e:
            st.error(f"Error al descargar datos para el ticker **{ticker}**: {e}")
    return data

def align_dates(data):
    if not data:
        return {}
    all_dates = pd.Index([])
    for df in data.values():
        all_dates = all_dates.union(df.index)
    for ticker in data:
        data[ticker] = data[ticker].reindex(all_dates)
        data[ticker] = data[ticker].ffill()
    return data

def evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ccl_ratio=False, data_source='yfinance'):
    if not main_ticker:
        st.error("El ticker principal no puede estar vacío.")
        return None
    main_ticker = main_ticker.upper()
    var_main = main_ticker.replace('.', '_')
    if main_ticker not in data:
        st.error(f"No hay datos disponibles para el ticker {main_ticker}")
        return None
    if not data[main_ticker].empty:
        result = data[main_ticker][var_main]
    else:
        st.error(f"No hay datos disponibles para el ticker {main_ticker}")
        return None
    if apply_ccl_ratio:
        if data_source == 'yfinance':
            st.write(f"Aplicando la razón YPFD.BA/YPF al ticker **{main_ticker}**...")
            ccl_tickers = ['YPFD.BA', 'YPF']
            if all(ticker in data for ticker in ccl_tickers):
                var_ypfd = 'YPFD_BA'
                var_ypf = 'YPF'
                if var_ypfd in data['YPFD.BA'].columns and var_ypf in data['YPF'].columns:
                    ypfd_data = data['YPFD.BA'][var_ypfd]
                    ypf_data = data['YPF'][var_ypf]
                    ratio = ypfd_data / ypf_data.replace(0, np.nan)
                    result = result / ratio
                else:
                    st.error("No hay columnas 'YPFD_BA' o 'YPF' disponibles en los datos.")
                    return None
            else:
                st.error("No hay datos disponibles para **YPFD.BA** o **YPF**.")
                return None
        else:
            st.write(f"Aplicando la razón GD30/GD30C al ticker **{main_ticker}**...")
            ccl_tickers = ['GD30', 'GD30C']
            if all(ticker in data for ticker in ccl_tickers):
                var_GD30 = 'GD30'
                var_GD30c = 'GD30C'
                if var_GD30 in data['GD30'].columns and var_GD30c in data['GD30C'].columns:
                    GD30_data = data['GD30'][var_GD30]
                    GD30c_data = data['GD30C'][var_GD30c]
                    ratio = GD30_data / GD30c_data.replace(0, np.nan)
                    result = result / ratio
                else:
                    st.error("No hay columnas 'GD30' o 'GD30C' disponibles en los datos.")
                    return None
            else:
                st.error("No hay datos disponibles para **GD30** o **GD30C**.")
                return None
    if second_ticker and third_ticker:
        second_ticker = second_ticker.upper()
        third_ticker = third_ticker.upper()
        var_second = second_ticker.replace('.', '_')
        var_third = third_ticker.replace('.', '_')
        if second_ticker in data and third_ticker in data:
            if var_second in data[second_ticker].columns and var_third in data[third_ticker].columns:
                second_data = data[second_ticker][var_second]
                third_data = data[third_ticker][var_third]
                ratio = second_data / third_data.replace(0, np.nan)
                result = result / ratio
            else:
                st.error(f"No hay columnas '{var_second}' o '{var_third}' disponibles en los datos.")
                return None
        else:
            st.error(f"No hay datos disponibles para {second_ticker} o {third_ticker}")
            return None
    elif second_ticker:
        second_ticker = second_ticker.upper()
        var_second = second_ticker.replace('.', '_')
        if second_ticker in data:
            if var_second in data[second_ticker].columns:
                second_data = data[second_ticker][var_second]
                ratio = second_data.replace(0, np.nan)
                result = result / ratio
            else:
                st.error(f"No hay columnas '{var_second}' disponibles en los datos.")
                return None
        else:
            st.error(f"No hay datos disponibles para {second_ticker}")
            return None
    return result

def calculate_streaks(data):
    streaks = []
    current_streak = {'value': None, 'start': None, 'end': None, 'length': 0}
    for i in range(len(data)):
        if current_streak['value'] is None:
            current_streak['value'] = data.iloc[i]
            current_streak['start'] = data.index[i]
            current_streak['end'] = data.index[i]
            current_streak['length'] = 1
        elif (data.iloc[i] > 0 and current_streak['value'] > 0) or (data.iloc[i] <= 0 and current_streak['value'] <= 0):
            current_streak['end'] = data.index[i]
            current_streak['length'] += 1
        else:
            streaks.append(current_streak)
            current_streak = {
                'value': data.iloc[i],
                'start': data.index[i],
                'end': data.index[i],
                'length': 1
            }
    if current_streak['length'] > 0:
        streaks.append(current_streak)
    streaks_df = pd.DataFrame(streaks)
    return streaks_df

def create_period_ranking(monthly_data, main_ticker, second_ticker, third_ticker, analysis_period, period_label):
    st.write(f"### 📊 Ranking de {period_label}es por Número de Valores Positivos y Negativos")
    if analysis_period == "Mes a Mes":
        period_index = monthly_data.index.month
        period_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        period_range = range(1, 13)
        period_label_axis = "Mes"
    else:
        period_index = monthly_data.index.quarter
        period_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        period_range = range(1, 5)
        period_label_axis = "Trimestre"
    period_positive_count = monthly_data[f'Cambio {period_label} (%)'].groupby(period_index).apply(lambda x: (x > 0).sum())
    period_negative_count = monthly_data[f'Cambio {period_label} (%)'].groupby(period_index).apply(lambda x: (x < 0).sum())
    period_rank_df = pd.DataFrame({
        period_label_axis: [period_names[i] for i in period_range],
        'Positivos': [period_positive_count.get(i, 0) for i in period_range],
        'Negativos': [period_negative_count.get(i, 0) for i in period_range]
    })
    fig, ax = plt.subplots(figsize=(8 if analysis_period == "Trimestre a Trimestre" else 14, 8))
    x = np.arange(len(period_rank_df[period_label_axis]))
    width = 0.35
    ax.bar(x - width/2, period_rank_df['Positivos'], width, label='Positivos', color='green')
    ax.bar(x + width/2, period_rank_df['Negativos'], width, label='Negativos', color='red')
    ax.set_title(f"Ranking de {period_label}es por Número de Variaciones Inter{period_label.lower()}es Positivas y Negativas para {main_ticker}" +
                 (f" / {second_ticker}" if second_ticker else "") +
                 (f" / {third_ticker}" if third_ticker else ""))
    ax.set_ylabel("Número de Valores")
    ax.set_xticks(x)
    ax.set_xticklabels(period_rank_df[period_label_axis], rotation=45)
    ax.legend()
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
             ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    plt.tight_layout()
    st.pyplot(fig)

def create_yearly_ranking(monthly_data, main_ticker, second_ticker, third_ticker, period_label):
    st.write("### 📊 Ranking de Años por Número de Valores Positivos y Negativos")
    yearly_positive_count = monthly_data[f'Cambio {period_label} (%)'].groupby(monthly_data.index.year).apply(lambda x: (x > 0).sum())
    yearly_negative_count = monthly_data[f'Cambio {period_label} (%)'].groupby(monthly_data.index.year).apply(lambda x: (x < 0).sum())
    years = sorted(yearly_positive_count.index)
    yearly_rank_df = pd.DataFrame({
        'Año': years,
        'Positivos': [yearly_positive_count.get(year, 0) for year in years],
        'Negativos': [yearly_negative_count.get(year, 0) for year in years]
    })
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(yearly_rank_df['Año']))
    width = 0.35
    ax.bar(x - width/2, yearly_rank_df['Positivos'], width, label='Positivos', color='green')
    ax.bar(x + width/2, yearly_rank_df['Negativos'], width, label='Negativos', color='red')
    ax.set_title(f"Ranking de Años por Número de Variaciones Inter{period_label.lower()}es Positivas y Negativas para {main_ticker}" +
                 (f" / {second_ticker}" if second_ticker else "") +
                 (f" / {third_ticker}" if third_ticker else ""))
    ax.set_ylabel("Número de Valores")
    ax.set_xticks(x)
    ax.set_xticklabels(yearly_rank_df['Año'], rotation=45)
    ax.legend()
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
             ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    plt.tight_layout()
    st.pyplot(fig)

def create_visualizations(monthly_data, main_ticker, second_ticker, third_ticker, metric_option, color_order, analysis_period, period_label):
    st.write(f"### 📉 Variaciones {period_label}es de Precios")
    fig = px.line(
        monthly_data,
        x=monthly_data.index,
        y=f'Cambio {period_label} (%)',
        title=f"Variaciones {period_label}es de {main_ticker}" +
              (f" / {second_ticker}" if second_ticker else "") +
              (f" / {third_ticker}" if third_ticker else ""),
        labels={f'Cambio {period_label} (%)': f'Cambio {period_label} (%)'}
    )
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)
    create_histogram_with_gaussian(monthly_data, main_ticker, second_ticker, third_ticker, period_label)
    create_period_heatmap(monthly_data, main_ticker, second_ticker, third_ticker, color_order, analysis_period, period_label)
    create_average_changes_visualization(monthly_data, metric_option, main_ticker, second_ticker, third_ticker, analysis_period, period_label)
    create_period_ranking(monthly_data, main_ticker, second_ticker, third_ticker, analysis_period, period_label)
    create_yearly_ranking(monthly_data, main_ticker, second_ticker, third_ticker, period_label)
    analyze_streaks(monthly_data, main_ticker, period_label)
    display_statistics(monthly_data, period_label)

def create_histogram_with_gaussian(monthly_data, main_ticker, second_ticker, third_ticker, period_label):
    st.write(f"### 📊 Histograma de Variaciones {period_label}es con Ajuste de Gauss")
    monthly_changes = monthly_data[f'Cambio {period_label} (%)'].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(monthly_changes, kde=False, stat="density", color="skyblue", ax=ax, binwidth=2)
    mu, std = norm.fit(monthly_changes)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    percentiles = [5, 25, 50, 75, 95]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, percentile in enumerate(percentiles):
        perc_value = np.percentile(monthly_changes, percentile)
        ax.axvline(perc_value, color=colors[i], linestyle='--', label=f'{percentile}º Percentil')
        ax.text(perc_value, ax.get_ylim()[1]*0.9, f'{perc_value:.2f}', color=colors[i],
                rotation=90, verticalalignment='center', horizontalalignment='right')
    ax.set_title(f"Histograma de Cambios {period_label}es con Ajuste de Gauss para {main_ticker}" +
                 (f" / {second_ticker}" if second_ticker else "") +
                 (f" / {third_ticker}" if third_ticker else ""))
    ax.set_xlabel(f"Cambio {period_label} (%)")
    ax.set_ylabel("Densidad")
    ax.legend()
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
             ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    st.pyplot(fig)

def create_average_changes_visualization(monthly_data, metric_option, main_ticker, second_ticker, third_ticker, analysis_period, period_label):
    st.write(f"### 📈 Cambios {metric_option} {period_label}es")
    if analysis_period == "Mes a Mes":
        period_index = monthly_data.index.month
        period_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        period_label_axis = "Mes"
    else:
        period_index = monthly_data.index.quarter
        period_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        period_label_axis = "Trimestre"
    if metric_option == "Promedio":
        avg_period_changes = monthly_data.groupby(period_index)[f'Cambio {period_label} (%)'].mean()
    else:
        avg_period_changes = monthly_data.groupby(period_index)[f'Cambio {period_label} (%)'].median()
    avg_period_changes.index = [period_names[i] for i in avg_period_changes.index]
    fig, ax = plt.subplots(figsize=(12 if analysis_period == "Mes a Mes" else 6, 6))
    bars = ax.bar(range(len(avg_period_changes)), avg_period_changes)
    for bar in bars:
        if bar.get_height() >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    ax.set_title(f"Cambios {metric_option} {period_label}es para {main_ticker}" +
                 (f" / {second_ticker}" if second_ticker else "") +
                 (f" / {third_ticker}" if third_ticker else ""))
    ax.set_xlabel(period_label_axis)
    ax.set_ylabel(f"{metric_option} de Cambio {period_label} (%)")
    plt.xticks(range(len(avg_period_changes)), avg_period_changes.index, rotation=45)
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
             ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    plt.tight_layout()
    st.pyplot(fig)
    st.write(f"### 📈 Cambios {metric_option} Anuales")
    if metric_option == "Promedio":
        avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)[f'Cambio {period_label} (%)'].mean()
    else:
        avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)[f'Cambio {period_label} (%)'].median()
    avg_yearly_changes = avg_yearly_changes.sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(avg_yearly_changes)), avg_yearly_changes)
    for bar in bars:
        if bar.get_height() >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    ax.set_title(f"Cambios {metric_option} Anuales para {main_ticker}" +
                 (f" / {second_ticker}" if second_ticker else "") +
                 (f" / {third_ticker}" if third_ticker else ""))
    ax.set_xlabel("Año")
    ax.set_ylabel(f"{metric_option} de Cambio Anual (%)")
    plt.xticks(range(len(avg_yearly_changes)), avg_yearly_changes.index, rotation=45)
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
             ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    plt.tight_layout()
    st.pyplot(fig)

def create_period_heatmap(monthly_data, main_ticker, second_ticker, third_ticker, color_order, analysis_period, period_label):
    st.write(f"### 🔥 Mapa de Calor de Variaciones {period_label}es")
    if analysis_period == "Mes a Mes":
        period_index = monthly_data.index.month
        period_names = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }
        period_label_axis = "Mes"
    else:
        period_index = monthly_data.index.quarter
        period_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        period_label_axis = "Trimestre"
    monthly_pivot = monthly_data.pivot_table(
        values=f'Cambio {period_label} (%)',
        index=monthly_data.index.year,
        columns=period_index,
        aggfunc='mean'
    )
    num_years = len(monthly_pivot.index)
    base_height = 6
    height_per_year = 0.4
    fig_height = max(base_height, height_per_year * num_years)
    fig_width = 6 if analysis_period == "Trimestre a Trimestre" else 12
    cmap = get_custom_cmap(color_order)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        monthly_pivot,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        center=0,
        ax=ax,
        annot_kws={"size": min(12, max(8, 120 / num_years))},
        cbar_kws={'label': f'Cambio {period_label} (%)'}
    )
    plt.title(f"Mapa de Calor de Variaciones {period_label}es para {main_ticker}" +
              (f" / {second_ticker}" if second_ticker else "") +
              (f" / {third_ticker}" if third_ticker else ""),
              pad=20)
    plt.xlabel(period_label_axis)
    plt.ylabel("Año")
    ax.set_xticklabels([period_names.get(int(x), x) for x in monthly_pivot.columns], rotation=45)
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
             ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    plt.tight_layout()
    st.pyplot(fig)

def analyze_streaks(monthly_data, main_ticker, period_label):
    st.write(f"### 📊 Análisis de Rachas de Variaciones {period_label}es")
    monthly_changes = monthly_data[f'Cambio {period_label} (%)'].dropna()
    streaks_df = calculate_streaks(monthly_changes)
    if not streaks_df.empty:
        st.write(f"**Rachas de variaciones {period_label.lower()}es positivas y negativas para {main_ticker}:**")
        for _, row in streaks_df.iterrows():
            direction = "Positiva" if row['value'] > 0 else "Negativa"
            st.write(f"- Racha {direction}: {row['length']} {period_label.lower()}es, desde {row['start'].date()} hasta {row['end'].date()}")
    else:
        st.write(f"No se encontraron rachas para {main_ticker}.")

def display_statistics(monthly_data, period_label):
    st.write(f"### 📊 Estadísticas Descriptivas")
    monthly_changes = monthly_data[f'Cambio {period_label} (%)'].dropna()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Promedio {period_label}", f"{monthly_changes.mean():.2f}%")
        st.metric(f"Mediana {period_label}", f"{monthly_changes.median():.2f}%")
    with col2:
        st.metric(f"Máximo {period_label}", f"{monthly_changes.max():.2f}%")
        st.metric(f"Mínimo {period_label}", f"{monthly_changes.min():.2f}%")
    with col3:
        st.metric(f"Volatilidad {period_label}", f"{monthly_changes.std():.2f}%")
        positive_periods = (monthly_changes > 0).sum()
        total_periods = len(monthly_changes)
        st.metric(f"% {period_label}es Positivos", f"{(positive_periods/total_periods*100):.1f}%")

# Main UI Layout
def main():
    st.title("📈 Análisis de Variación de Precios de Acciones, ETFs e Índices - MTaurus")
    st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")
    data_source = st.selectbox(
        "🔄 Seleccione la fuente de datos:",
        options=['yfinance', 'analisistecnico', 'iol', 'byma'],
        help="Seleccione la fuente desde donde desea obtener los datos"
    )
    st.markdown("""
    ### ℹ️ Información sobre las fuentes de datos:
    - **yfinance**: Datos globales de Yahoo Finance
    - **analisistecnico**: Datos del mercado argentino de AnálisisTécnico
    - **iol**: Datos del mercado argentino de InvertirOnline
    - **byma**: Datos del mercado argentino de BYMA
    """)
    if data_source == 'yfinance':
        ccl_pair = ('YPFD.BA', 'YPF')
        ccl_checkbox_text = "🔄 Dividir el ticker principal por dólar CCL de YPF (YPFD.BA/YPF)"
    else:
        ccl_pair = ('GD30', 'GD30C')
        ccl_checkbox_text = "🔄 Dividir el ticker principal por dólar CCL (GD30/GD30C)"
    apply_ccl_ratio = st.checkbox(ccl_checkbox_text, value=False)
    main_ticker = st.text_input("🖊️ Ingrese el ticker principal (por ejemplo: GGAL.BA, METR.BA, AAPL, BMA):")
    second_ticker = st.text_input("➕ Ingrese el segundo ticker o ratio divisor (opcional):")
    third_ticker = st.text_input("➕ Ingrese el tercer ticker o ratio divisor (opcional):")
    start_date = st.date_input(
        "📅 Seleccione la fecha de inicio:",
        value=pd.to_datetime('2010-01-01').date(),
        min_value=pd.to_datetime('1920-01-01').date()
    )
    end_date = st.date_input(
        "📅 Seleccione la fecha de fin:",
        value=pd.to_datetime('today').date()
    )
    analysis_period = st.radio(
        "📊 Seleccione el período de análisis:",
        ("Mes a Mes", "Trimestre a Trimestre"),
        help="Seleccione si desea analizar las variaciones mensuales o trimestrales."
    )
    period_map = {
        "Mes a Mes": 'M',
        "Trimestre a Trimestre": 'Q'
    }
    resample_freq = period_map[analysis_period]
    period_label = "Mensual" if analysis_period == "Mes a Mes" else "Trimestral"
    metric_option = st.radio(
        f"📊 Seleccione la métrica para los gráficos {period_label.lower()}es:",
        ("Promedio", "Mediana")
    )
    color_order = st.selectbox(
        "🌈 Seleccione el esquema de colores para el mapa de calor:",
        options=['Red-White-Green', 'Green-White-Red'],
        format_func=lambda x: 'Rojo-Blanco-Verde' if x == 'Red-White-Green' else 'Verde-Blanco-Rojo'
    )
    color_order_map = {
        'Red-White-Green': 'red_white_green',
        'Green-White-Red': 'green_white_red'
    }
    selected_color_order = color_order_map[color_order]
    tickers = {main_ticker, second_ticker, third_ticker}
    tickers = {ticker.upper() for ticker in tickers if ticker}
    if apply_ccl_ratio:
        tickers.update(set(ccl_pair))
    if all(validate_ticker_format(ticker, data_source) for ticker in tickers if ticker):
        data = fetch_data(tickers, start_date, end_date, data_source)
        if data:
            data = align_dates(data)
            ratio_data = evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ccl_ratio, data_source)
            if ratio_data is not None:
                if isinstance(ratio_data, pd.Series):
                    ratio_data = ratio_data.to_frame(name='Adj Close')
                ratio_data.index = pd.to_datetime(ratio_data.index)
                ratio_data = ratio_data.copy()
                ratio_data['Period'] = ratio_data.index.to_period(resample_freq)
                monthly_data = ratio_data.resample(resample_freq).ffill()
                main_var = main_ticker.replace('.', '_')
                if 'Adj Close' in monthly_data.columns:
                    pct_change_col = 'Adj Close'
                elif main_var in monthly_data.columns:
                    pct_change_col = main_var
                else:
                    st.error(f"'{main_var}' no está disponible en los datos después del resampleo.")
                    return
                monthly_data[f'Cambio {period_label} (%)'] = monthly_data[pct_change_col].pct_change() * 100
                create_visualizations(monthly_data, main_ticker, second_ticker, third_ticker, metric_option, selected_color_order, analysis_period, period_label)

# Main execution
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")
