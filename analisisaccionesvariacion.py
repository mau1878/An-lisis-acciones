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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis de Variación Mensual de Precios",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Utility Functions
def get_custom_cmap():
    colors = ['red', 'white', 'green']
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
    # Aquí se puede agregar validaciones específicas según la fuente de datos
    return True  # Permitir cualquier formato de ticker para todas las fuentes

# Data Download Functions
def descargar_datos_yfinance(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end, progress=False)

        if stock_data.empty:
            logger.warning(f"No se encontraron datos para el ticker {ticker} en el rango de fechas seleccionado.")
            return pd.DataFrame()

        stock_data = stock_data.reset_index()

        # Manejar tanto columnas de un solo nivel como MultiIndex
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Extraer 'Close' o 'Adj Close' según esté disponible
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
            # Columnas de un solo nivel
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

        # Ajustar precios por splits si es necesario
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

    # Obtener todas las fechas
    all_dates = pd.Index([])
    for df in data.values():
        all_dates = all_dates.union(df.index)

    # Reindexar cada DataFrame para tener las mismas fechas
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
                    # Evitar división por cero
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
                    # Evitar división por cero
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
                # Evitar división por cero
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
                # Evitar división por cero
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

    # Convert to pandas-friendly format
    streaks_df = pd.DataFrame(streaks)
    return streaks_df

def create_monthly_ranking(monthly_data, main_ticker, second_ticker, third_ticker):
    st.write("### 📊 Ranking de Meses por Número de Valores Positivos y Negativos")
    monthly_positive_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.month).apply(lambda x: (x > 0).sum())
    monthly_negative_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.month).apply(lambda x: (x < 0).sum())

    # Crear DataFrame con nombres de meses en español
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    monthly_rank_df = pd.DataFrame({
        'Mes': [month_names[i] for i in range(1, 13)],
        'Positivos': [monthly_positive_count.get(i, 0) for i in range(1, 13)],
        'Negativos': [monthly_negative_count.get(i, 0) for i in range(1, 13)]
    })

    fig, ax = plt.subplots(figsize=(14, 8))

    # Posiciones para las barras
    x = np.arange(len(monthly_rank_df['Mes']))
    width = 0.35  # Ancho de las barras

    # Crear las barras
    ax.bar(x - width/2, monthly_rank_df['Positivos'], width, label='Positivos', color='green')
    ax.bar(x + width/2, monthly_rank_df['Negativos'], width, label='Negativos', color='red')

    # Personalizar el gráfico
    ax.set_title(f"Ranking de Meses por Número de Variaciones Intermensuales Positivas y Negativas para {main_ticker}" +
                (f" / {second_ticker}" if second_ticker else "") +
                (f" / {third_ticker}" if third_ticker else ""))
    ax.set_ylabel("Número de Valores")
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_rank_df['Mes'], rotation=45)
    ax.legend()

    # Agregar marca de agua
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
            ha='center', va='center', alpha=0.5, transform=ax.transAxes)

    # Ajustar el diseño para evitar recortes
    plt.tight_layout()
    st.pyplot(fig)

def create_yearly_ranking(monthly_data, main_ticker, second_ticker, third_ticker):
    st.write("### 📊 Ranking de Años por Número de Valores Positivos y Negativos")
    yearly_positive_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.year).apply(lambda x: (x > 0).sum())
    yearly_negative_count = monthly_data['Cambio Mensual (%)'].groupby(monthly_data.index.year).apply(lambda x: (x < 0).sum())

    # Crear DataFrame con orden cronológico
    years = sorted(yearly_positive_count.index)
    yearly_rank_df = pd.DataFrame({
        'Año': years,
        'Positivos': [yearly_positive_count.get(year, 0) for year in years],
        'Negativos': [yearly_negative_count.get(year, 0) for year in years]
    })

    fig, ax = plt.subplots(figsize=(14, 8))

    # Posiciones para las barras
    x = np.arange(len(yearly_rank_df['Año']))
    width = 0.35  # Ancho de las barras

    # Crear las barras
    ax.bar(x - width/2, yearly_rank_df['Positivos'], width, label='Positivos', color='green')
    ax.bar(x + width/2, yearly_rank_df['Negativos'], width, label='Negativos', color='red')

    # Personalizar el gráfico
    ax.set_title(f"Ranking de Años por Número de Variaciones Intermensuales Positivas y Negativas para {main_ticker}" +
                (f" / {second_ticker}" if second_ticker else "") +
                (f" / {third_ticker}" if third_ticker else ""))
    ax.set_ylabel("Número de Valores")
    ax.set_xticks(x)
    ax.set_xticklabels(yearly_rank_df['Año'], rotation=45)
    ax.legend()

    # Agregar marca de agua
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
            ha='center', va='center', alpha=0.5, transform=ax.transAxes)

    # Ajustar el diseño para evitar recortes
    plt.tight_layout()
    st.pyplot(fig)

# Main UI Layout
def main():
    # Title and Header
    st.title("📈 Análisis de Variación Mensual de Precios de Acciones, ETFs e Índices - MTaurus")
    st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

    # Data Source Selection
    data_source = st.selectbox(
        "🔄 Seleccione la fuente de datos:",
        options=['yfinance', 'analisistecnico', 'iol', 'byma'],
        help="Seleccione la fuente desde donde desea obtener los datos"
    )

    # Information about data sources
    st.markdown("""
    ### ℹ️ Información sobre las fuentes de datos:
    - **yfinance**: Datos globales de Yahoo Finance
    - **analisistecnico**: Datos del mercado argentino de AnálisisTécnico
    - **iol**: Datos del mercado argentino de InvertirOnline
    - **byma**: Datos del mercado argentino de BYMA
    """)

    # CCL Configuration
    if data_source == 'yfinance':
        ccl_pair = ('YPFD.BA', 'YPF')
        ccl_checkbox_text = "🔄 Dividir el ticker principal por dólar CCL de YPF (YPFD.BA/YPF)"
    else:
        ccl_pair = ('GD30', 'GD30C')
        ccl_checkbox_text = "🔄 Dividir el ticker principal por dólar CCL (GD30/GD30C)"

    apply_ccl_ratio = st.checkbox(ccl_checkbox_text, value=False)

    # User Inputs
    main_ticker = st.text_input("🖊️ Ingrese el ticker principal (por ejemplo: GGAL.BA, METR.BA, AAPL, BMA):")
    second_ticker = st.text_input("➕ Ingrese el segundo ticker o ratio divisor (opcional):")
    third_ticker = st.text_input("➕ Ingrese el tercer ticker o ratio divisor (opcional):")

    # Date Selection
    start_date = st.date_input(
        "📅 Seleccione la fecha de inicio:",
        value=pd.to_datetime('2010-01-01').date(),
        min_value=pd.to_datetime('1920-01-01').date()
    )
    end_date = st.date_input(
        "📅 Seleccione la fecha de fin:",
        value=pd.to_datetime('today').date()
    )

    # Metric Selection
    metric_option = st.radio(
        "📊 Seleccione la métrica para los gráficos mensuales y anuales:",
        ("Promedio", "Mediana")
    )

    # Process tickers
    tickers = {main_ticker, second_ticker, third_ticker}
    tickers = {ticker.upper() for ticker in tickers if ticker}
    if apply_ccl_ratio:
        tickers.update(set(ccl_pair))

    # Validate and process data
    if all(validate_ticker_format(ticker, data_source) for ticker in tickers if ticker):
        data = fetch_data(tickers, start_date, end_date, data_source)

        if data:
            # Align dates and fill missing values
            data = align_dates(data)

            # Evaluate ratio based on selected tickers
            ratio_data = evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ccl_ratio, data_source)

            if ratio_data is not None:
                # Convert to DataFrame if it's a Series
                if isinstance(ratio_data, pd.Series):
                    ratio_data = ratio_data.to_frame(name='Adj Close')

                ratio_data.index = pd.to_datetime(ratio_data.index)

                # Create a copy of DataFrame to avoid SettingWithCopyWarning
                ratio_data = ratio_data.copy()
                ratio_data['Month'] = ratio_data.index.to_period('M')

                # Resample to monthly and forward fill
                monthly_data = ratio_data.resample('M').ffill()

                # Verify if 'Adj Close' is present after resampling
                main_var = main_ticker.replace('.', '_')
                if 'Adj Close' in monthly_data.columns:
                    pct_change_col = 'Adj Close'
                elif main_var in monthly_data.columns:
                    pct_change_col = main_var
                else:
                    st.error(f"'{main_var}' no está disponible en los datos después del resampleo.")
                    return

                # Calculate monthly percentage change
                monthly_data['Cambio Mensual (%)'] = monthly_data[pct_change_col].pct_change() * 100

                # Visualizations
                create_visualizations(monthly_data, main_ticker, second_ticker, third_ticker, metric_option)

def create_histogram_with_gaussian(monthly_data, main_ticker, second_ticker, third_ticker):
    st.write("### 📊 Histograma de Variaciones Mensuales con Ajuste de Gauss")
    monthly_changes = monthly_data['Cambio Mensual (%)'].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(monthly_changes, kde=False, stat="density", color="skyblue", ax=ax, binwidth=2)

    # Gaussian distribution fit
    mu, std = norm.fit(monthly_changes)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    # Calculate and plot percentiles
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
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
            ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    st.pyplot(fig)

def create_average_changes_visualization(monthly_data, metric_option, main_ticker, second_ticker, third_ticker):
    # Monthly Changes
    st.write(f"### 📈 Cambios {metric_option} Mensuales")

    # Calculate monthly changes based on selected metric
    if metric_option == "Promedio":
        avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].mean()
    else:
        avg_monthly_changes = monthly_data.groupby(monthly_data.index.month)['Cambio Mensual (%)'].median()

    # Convert month numbers to Spanish month names
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    avg_monthly_changes.index = [month_names[i] for i in avg_monthly_changes.index]

    # Create monthly plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(avg_monthly_changes)), avg_monthly_changes)

    # Color bars based on values
    for bar in bars:
        if bar.get_height() >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    ax.set_title(f"Cambios {metric_option} Mensuales para {main_ticker}" +
                (f" / {second_ticker}" if second_ticker else "") +
                (f" / {third_ticker}" if third_ticker else ""))
    ax.set_xlabel("Mes")
    ax.set_ylabel(f"{metric_option} de Cambio Mensual (%)")
    plt.xticks(range(len(avg_monthly_changes)), avg_monthly_changes.index, rotation=45)

    # Agregar marca de agua
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
            ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    plt.tight_layout()
    st.pyplot(fig)

    # Yearly Changes
    st.write(f"### 📈 Cambios {metric_option} Anuales")

    # Calculate yearly changes based on selected metric
    if metric_option == "Promedio":
        avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].mean()
    else:
        avg_yearly_changes = monthly_data.groupby(monthly_data.index.year)['Cambio Mensual (%)'].median()

    # Sort years chronologically
    avg_yearly_changes = avg_yearly_changes.sort_index()

    # Create yearly plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(avg_yearly_changes)), avg_yearly_changes)

    # Color bars based on values
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

    # Agregar marca de agua
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
            ha='center', va='center', alpha=0.5, transform=ax.transAxes)

    # Ajustar el diseño para evitar recortes
    plt.tight_layout()
    st.pyplot(fig)

def create_monthly_heatmap(monthly_data, main_ticker, second_ticker, third_ticker):
    st.write("### 🔥 Mapa de Calor de Variaciones Mensuales")
    monthly_pivot = monthly_data.pivot_table(
        values='Cambio Mensual (%)',
        index=monthly_data.index.year,
        columns=monthly_data.index.month,
        aggfunc='mean'
    )

    # Definir colormap personalizado (rojo para negativo, verde para positivo)
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
    # Agregar marca de agua
    plt.text(0.5, 0.01, "MTaurus - X: MTaurus_ok", fontsize=12, color='grey',
             ha='center', va='center', alpha=0.5, transform=ax.transAxes)
    st.pyplot(fig)

def analyze_streaks(monthly_data, main_ticker):
    st.write("### 📈 Análisis de Rachas")

    monthly_changes = monthly_data['Cambio Mensual (%)'].dropna()
    streaks = calculate_streaks(monthly_changes)

    # Separar rachas positivas y negativas
    positive_streaks = streaks[streaks['value'] > 0]
    negative_streaks = streaks[streaks['value'] <= 0]

    # Mostrar estadísticas
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Rachas Positivas")
        if not positive_streaks.empty:
            max_positive = positive_streaks.loc[positive_streaks['length'].idxmax()]
            st.write(f"**Racha más larga:** {max_positive['length']} meses")
            st.write(f"**Desde:** {max_positive['start'].strftime('%Y-%m')}")
            st.write(f"**Hasta:** {max_positive['end'].strftime('%Y-%m')}")
        else:
            st.write("No hay rachas positivas.")

    with col2:
        st.write("#### Rachas Negativas")
        if not negative_streaks.empty:
            max_negative = negative_streaks.loc[negative_streaks['length'].idxmax()]
            st.write(f"**Racha más larga:** {max_negative['length']} meses")
            st.write(f"**Desde:** {max_negative['start'].strftime('%Y-%m')}")
            st.write(f"**Hasta:** {max_negative['end'].strftime('%Y-%m')}")
        else:
            st.write("No hay rachas negativas.")

def create_visualizations(monthly_data, main_ticker, second_ticker, third_ticker, metric_option):
    # 1. Monthly Price Variations
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

    # 2. Histogram with Gaussian Fit
    create_histogram_with_gaussian(monthly_data, main_ticker, second_ticker, third_ticker)

    # 3. Monthly Heatmap
    create_monthly_heatmap(monthly_data, main_ticker, second_ticker, third_ticker)

    # 4. Average Changes Visualization
    create_average_changes_visualization(monthly_data, metric_option, main_ticker, second_ticker, third_ticker)

    # 5. Rankings
    create_monthly_ranking(monthly_data, main_ticker, second_ticker, third_ticker)
    create_yearly_ranking(monthly_data, main_ticker, second_ticker, third_ticker)

    # 6. Streak Analysis
    analyze_streaks(monthly_data, main_ticker)

    # 7. Statistics
    display_statistics(monthly_data)

def display_statistics(monthly_data):
    st.write("### 📊 Estadísticas Descriptivas")

    monthly_changes = monthly_data['Cambio Mensual (%)'].dropna()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Promedio Mensual", f"{monthly_changes.mean():.2f}%")
        st.metric("Mediana Mensual", f"{monthly_changes.median():.2f}%")

    with col2:
        st.metric("Máximo Mensual", f"{monthly_changes.max():.2f}%")
        st.metric("Mínimo Mensual", f"{monthly_changes.min():.2f}%")

    with col3:
        st.metric("Volatilidad Mensual", f"{monthly_changes.std():.2f}%")
        positive_months = (monthly_changes > 0).sum()
        total_months = len(monthly_changes)
        st.metric("% Meses Positivos", f"{(positive_months/total_months*100):.1f}%")

# Main execution
if __name__ == "__main__":
    main()

# Footer
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")
