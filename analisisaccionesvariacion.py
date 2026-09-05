import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import requests
from datetime import datetime
import logging
import os
import urllib3
from curl_cffi import requests as cffi_requests

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Análisis de Variación de Precios - MTaurus",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DARK MODE GLOBAL ───
plt.style.use('dark_background')

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#0e1117",
    "figure.facecolor": "#0e1117",
    "grid.color": "#444444",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "legend.facecolor": "#0e1117",
    "legend.edgecolor": "#555555",
})

plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#0e1117',
    'axes.edgecolor': '#666666',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'grid.color': '#444444',
})

def apply_dark_theme(ax):
    ax.set_facecolor('#0e1117')
    ax.figure.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='#444444', alpha=0.5)
    if ax.get_legend():
        legend = ax.get_legend()
        legend.get_frame().set_facecolor('#0e1117')
        legend.get_frame().set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color('white')

def add_watermark(ax, fontsize=28, alpha=0.25):
    ax.text(0.5, 0.5, "MTaurus - X: MTaurus_ok",
            fontsize=fontsize, color='white', alpha=alpha,
            ha='center', va='center', rotation=-42,
            transform=ax.transAxes, fontweight='bold', zorder=999)

def get_custom_cmap(color_order='red_white_green'):
    if color_order == 'red_white_green':
        colors = ['#d32f2f', '#ffffff', '#388e3c']
    else:
        colors = ['#388e3c', '#ffffff', '#d32f2f']
    return LinearSegmentedColormap.from_list('custom_diverging', colors)

def get_yearly_cmap(color_order='red_white_green'):
    # Mismo espíritu rojo/verde que el mensual, pero en un tono más oscuro/saturado
    # para que se distinga a simple vista sin mezclar las dos escalas.
    if color_order == 'red_white_green':
        colors = ['#7f0000', '#ffffff', '#0d3d1a']
    else:
        colors = ['#0d3d1a', '#ffffff', '#7f0000']
    return LinearSegmentedColormap.from_list('yearly_diverging', colors)

def calculate_yearly_changes(daily_data, price_col):
    # Mismo método que mensual/trimestral: último valor del período vs último del período anterior
    yearly_price = daily_data[price_col].resample('YE').last()
    yearly_change = yearly_price.pct_change() * 100
    yearly_change.index = yearly_change.index.year
    return yearly_change

def ajustar_precios_por_splits(df, ticker):
    return df

def validate_ticker_format(ticker, data_source):
    if not ticker:
        return True
    return True

# ─── DESCARGA DE DATOS ───
def descargar_datos_yfinance(ticker, start, end):
    try:
        session = cffi_requests.Session(impersonate="chrome124")
        stock_data = yf.download(ticker, start=start, end=end, progress=False, session=session)
        if stock_data.empty:
            logger.warning(f"No datos para {ticker} en yfinance")
            return pd.DataFrame()
        stock_data = stock_data.reset_index()
        if isinstance(stock_data.columns, pd.MultiIndex):
            if ('Adj Close', ticker) in stock_data.columns:
                close_price = stock_data[('Adj Close', ticker)]
            elif ('Close', ticker) in stock_data.columns:
                close_price = stock_data[('Close', ticker)]
            else:
                return pd.DataFrame()
            var_name = ticker.replace('.', '_')
            df = pd.DataFrame({'Date': stock_data['Date'], var_name: close_price})
        else:
            close_col = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'
            var_name = ticker.replace('.', '_')
            df = pd.DataFrame({'Date': stock_data['Date'], var_name: stock_data[close_col]})
        df = ajustar_precios_por_splits(df, ticker)
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error yfinance {ticker}: {e}")
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
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        symbol = ticker.replace('.BA', '')
        params = {'symbol': symbol, 'resolution': 'D', 'from': str(from_timestamp), 'to': str(to_timestamp)}
        response = requests.get('https://analisistecnico.com.ar/services/datafeed/history', params=params, cookies=cookies, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get('s') != 'ok':
                return pd.DataFrame()
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates('Date')
            df = ajustar_precios_por_splits(df, ticker)
            var_name = ticker.replace('.', '_')
            df = df.set_index('Date').rename(columns={'Close': var_name})
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error analisistecnico {ticker}: {e}")
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
            'referer': 'https://iol.invertironline.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        symbol = ticker.replace('.BA', '')
        params = {'symbolName': symbol, 'exchange': 'BCBA', 'from': str(from_timestamp), 'to': str(to_timestamp), 'resolution': 'D'}
        response = requests.get('https://iol.invertironline.com/api/cotizaciones/history', params=params, cookies=cookies, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') != 'ok' or 'bars' not in data:
                return pd.DataFrame()
            df = pd.DataFrame(data['bars'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time').rename(columns={'close': ticker.replace('.', '_')})[['close']]
            df = ajustar_precios_por_splits(df, ticker)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error IOL {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_byma(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        cookies = {'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338'}
        headers = {'Accept': 'application/json', 'Referer': 'https://open.bymadata.com.ar/'}
        symbol = ticker.replace('.BA', '') + ' 24HS'
        params = {'symbol': symbol, 'resolution': 'D', 'from': str(from_timestamp), 'to': str(to_timestamp)}
        urllib3.disable_warnings()
        response = requests.get('https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
                                params=params, cookies=cookies, headers=headers, verify=False)
        if response.status_code == 200:
            data = response.json()
            if data.get('s') != 'ok':
                return pd.DataFrame()
            df = pd.DataFrame({'Date': pd.to_datetime(data['t'], unit='s'), 'Close': data['c']})
            df = df.set_index('Date').rename(columns={'Close': ticker.replace('.', '_')})
            df = ajustar_precios_por_splits(df, ticker)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error ByMA {ticker}: {e}")
        return pd.DataFrame()

# ─── HISTÓRICO EXTENDIDO PARA ^MERV (Stooq, 1988-1996) ───
# yfinance solo tiene ^MERV desde el 8/10/1996. Este archivo (Stooq, ticker ^MRV)
# completa el tramo 1988-04-04 a 1996-10-07 (día calendario justo anterior al
# arranque de yfinance, sin superposición). Colocar el archivo en el repo en:
#   data/mrv_historico.txt
MERVAL_HISTORICO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mrv_historico.txt')
MERVAL_HISTORICO_CUTOFF = pd.Timestamp('1996-10-08')  # primer día con datos de yfinance

@st.cache_data
def cargar_historico_merval_stooq():
    try:
        hist = pd.read_csv(MERVAL_HISTORICO_PATH)
        hist['Date'] = pd.to_datetime(hist['<DATE>'], format='%Y%m%d')
        hist = hist[hist['Date'] < MERVAL_HISTORICO_CUTOFF]
        hist = hist[['Date', '<CLOSE>']].sort_values('Date').drop_duplicates('Date')
        return hist.set_index('Date')['<CLOSE>']
    except Exception as e:
        logger.error(f"Error cargando histórico Merval (Stooq): {e}")
        return pd.Series(dtype=float)

def extender_con_historico_merval(df, ticker, start_date):
    """Si el ticker es ^MERV y el usuario pidió datos desde antes del 8/10/1996,
    completa 1988-1996 con el histórico de Stooq (^MRV)."""
    if ticker.upper() != '^MERV':
        return df
    start_ts = pd.Timestamp(start_date)
    if start_ts >= MERVAL_HISTORICO_CUTOFF:
        return df

    hist = cargar_historico_merval_stooq()
    if hist.empty:
        return df
    hist = hist[hist.index >= start_ts]
    if hist.empty:
        return df

    var_name = ticker.replace('.', '_')
    hist_df = hist.to_frame(name=var_name)

    if df.empty:
        return hist_df
    return pd.concat([hist_df, df[~df.index.isin(hist_df.index)]]).sort_index()

# ─── CCL HISTÓRICO PARA ^MERV (1988-2003) Y RATIO YPFD.BA/YPF ───
# Tramo histórico (Excel del usuario + fuente oficial BCRA "EvolucionMoneda"):
#   - com_3501: 4/4/1988 a 31/12/1991
#   - "TIPO DE CAMBIO - MONEDA DE CURSO LEGAL" (BCRA, serie oficial completa
#     y sin baches): 2/1/1992 a 21/3/2003
# La fecha de corte (24/3/2003) se determinó comparando día a día esta serie
# oficial contra el ratio real YPFD.BA/YPF: es la primera fecha desde la que
# la diferencia entre ambas queda sostenidamente por debajo del 3% (antes de
# esa fecha, sobre todo en oct-2002, la diferencia llega a superar el 15%).
# Colocar el archivo en el repo en: data/merval_ccl_historico.csv
MERVAL_CCL_HISTORICO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'merval_ccl_historico.csv')
MERVAL_CCL_HISTORICO_CUTOFF = pd.Timestamp('2003-03-24')  # desde acá manda el ratio YPFD.BA/YPF

@st.cache_data(ttl="1d")
def descargar_ypfd_ypf_crudo():
    """Descarga YPFD.BA y YPF con precio sin ajustar por DIVIDENDOS
    (auto_adjust=False). El ratio de CCL necesita esto: el ADR cobra
    dividendos en USD y la acción local en ARS, así que sus historiales de
    ajuste por dividendos no son comparables entre sí y distorsionan el
    ratio si se usa Adj Close (verificado con datos reales: en Convertibilidad
    el ratio debía dar ~1.0 y con Adj Close daba ~21).

    Nota sobre splits: en yfinance 1.x, auto_adjust=False solo evita el
    ajuste por dividendos -- el ajuste por SPLITS se aplica siempre, de forma
    consistente en toda la serie histórica (verificado: da el mismo resultado
    sin importar el rango pedido). Por eso el multiplicador ADR:acción local
    es constante (x10) en calcular_ratio_ypfd_ypf, sin necesidad de detectar
    fechas de split a mano."""
    try:
        ypfd = yf.download('YPFD.BA', start='1996-01-01', progress=False, auto_adjust=False)
        ypf = yf.download('YPF', start='1996-01-01', progress=False, auto_adjust=False)

        def get_close(d):
            if isinstance(d.columns, pd.MultiIndex):
                d = d.droplevel(1, axis=1)
            return d['Close'] if 'Close' in d.columns else pd.Series(dtype=float)

        return get_close(ypfd).dropna(), get_close(ypf).dropna()
    except Exception as e:
        logger.error(f"Error descargando YPFD.BA/YPF crudo: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

def calcular_ratio_ypfd_ypf(start_date, end_date):
    ypfd, ypf = descargar_ypfd_ypf_crudo()
    if ypfd.empty or ypf.empty:
        return pd.Series(dtype=float)
    combined = pd.DataFrame({'YPFD': ypfd, 'YPF': ypf}).dropna()
    combined = combined[combined.index >= pd.Timestamp(start_date)]
    if combined.empty:
        return pd.Series(dtype=float)
    # 1 ADR YPF equivale a 10 acciones locales YPFD.BA
    return (combined['YPFD'] * 10) / combined['YPF']

@st.cache_data
def cargar_ccl_historico_merval():
    try:
        hist = pd.read_csv(MERVAL_CCL_HISTORICO_PATH, parse_dates=['Date'])
        hist = hist.sort_values('Date').drop_duplicates('Date').set_index('Date')
        return hist['TipoCambio']
    except Exception as e:
        logger.error(f"Error cargando CCL histórico Merval: {e}")
        return pd.Series(dtype=float)

def fetch_data(tickers, start_date, end_date, data_source):
    data = {}
    for ticker in tickers:
        ticker = ticker.upper()
        if data_source == 'yfinance':
            df = descargar_datos_yfinance(ticker, start_date, end_date)
        elif data_source == 'analisistecnico':
            df = descargar_datos_analisistecnico(ticker, start_date, end_date)
        elif data_source == 'iol':
            df = descargar_datos_iol(ticker, start_date, end_date)
        elif data_source == 'byma':
            df = descargar_datos_byma(ticker, start_date, end_date)
        else:
            df = pd.DataFrame()

        df = extender_con_historico_merval(df, ticker, start_date)

        if not df.empty:
            data[ticker] = df
    return data

def align_dates(data):
    if not data:
        return {}
    all_dates = pd.Index([])
    for df in data.values():
        all_dates = all_dates.union(df.index)
    for ticker in data:
        data[ticker] = data[ticker].reindex(all_dates).ffill()
    return data
def evaluate_ratio(main_ticker, second_ticker, third_ticker, data, apply_ccl_ratio, data_source):
    if not main_ticker or main_ticker not in data or data[main_ticker].empty:
        return None

    var_main = main_ticker.replace('.', '_')
    result = data[main_ticker][var_main]

    if apply_ccl_ratio:
        if data_source == 'yfinance':
            if main_ticker.upper() == '^MERV':
                ratio = pd.Series(index=result.index, dtype=float)

                # Tramo histórico (1988 - 2/1/2000): com_3501 (BCRA)
                hist_ccl = cargar_ccl_historico_merval()
                if not hist_ccl.empty:
                    idx_hist = result.index[result.index < MERVAL_CCL_HISTORICO_CUTOFF]
                    ratio.loc[idx_hist] = hist_ccl.reindex(idx_hist)

                # Tramo moderno (desde 3/1/2000): ratio YPFD.BA/YPF crudo, corregido por el split
                idx_moderno = result.index[result.index >= MERVAL_CCL_HISTORICO_CUTOFF]
                if len(idx_moderno) > 0:
                    ratio_ypf = calcular_ratio_ypfd_ypf(idx_moderno.min(), idx_moderno.max())
                    if not ratio_ypf.empty:
                        ratio.loc[idx_moderno] = ratio_ypf.reindex(idx_moderno)

                result = result / ratio
            elif 'YPFD.BA' in data and 'YPF' in data:
                ratio_ypf = calcular_ratio_ypfd_ypf(result.index.min(), result.index.max())
                if not ratio_ypf.empty:
                    result = result / ratio_ypf.reindex(result.index)
        else:
            if 'GD30' in data and 'GD30C' in data:
                ratio = data['GD30']['GD30'] / data['GD30C']['GD30C']
                result = result / ratio

    if second_ticker and third_ticker and second_ticker in data and third_ticker in data:
        var2 = second_ticker.replace('.', '_')
        var3 = third_ticker.replace('.', '_')
        ratio = data[second_ticker][var2] / data[third_ticker][var3]
        result = result / ratio
    elif second_ticker and second_ticker in data:
        var2 = second_ticker.replace('.', '_')
        result = result / data[second_ticker][var2]

    return result

def calculate_streaks(series):
    streaks = []
    current_value = None
    current_start = None
    current_length = 0
    for idx, val in series.items():
        if current_value is None:
            current_value = val > 0
            current_start = idx
            current_length = 1
        elif (val > 0) == current_value:
            current_length += 1
        else:
            streaks.append({
                'start': current_start,
                'end': idx - pd.Timedelta(days=1) if 'D' in series.index.freqstr else idx,
                'length': current_length,
                'value': 1 if current_value else -1
            })
            current_value = val > 0
            current_start = idx
            current_length = 1
    if current_length > 0:
        streaks.append({
            'start': current_start,
            'end': series.index[-1],
            'length': current_length,
            'value': 1 if current_value else -1
        })
    return pd.DataFrame(streaks)

def calculate_drawdown(prices):
    if len(prices) < 2:
        return pd.Series(index=prices.index, data=0, name='Drawdown (%)')
    peak = prices.cummax()
    dd = (prices - peak) / peak * 100
    return dd

def create_drawdown_visualization(prices, main_ticker, second=None, third=None, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader("Drawdown del Ticker Principal")
    dd = calculate_drawdown(prices)
    max_dd = dd.min()
    current_dd = dd.iloc[-1]
    col1, col2 = st.columns(2)
    col1.metric("Máximo Drawdown", f"{max_dd:.2f}%")
    col2.metric("Drawdown Actual", f"{current_dd:.2f}%")
    
    title = f"Drawdown - {main_ticker}" + (f" / {second}" if second else "") + \
            (f" / {third}" if third else "") + ccl_text
    
    fig = px.area(dd.reset_index(), x='index', y=dd.name,
                  title=title, template='plotly_dark',
                  labels={'value': 'Drawdown (%)'})
    fig.update_traces(line_color='crimson', fillcolor='rgba(220,20,60,0.3)')
    fig.add_annotation(text="MTaurus - X: MTaurus_ok", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font_size=38, opacity=0.22, textangle=-42)
    st.plotly_chart(fig, use_container_width=True)

# ─── VISUALIZACIONES ───
def create_histogram_with_gaussian(monthly_data, main, sec, third, period_label, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader(f"Histograma de Cambios {period_label}es")
    changes = monthly_data[f'Cambio {period_label} (%)'].dropna()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(changes, kde=False, stat="density", color="#1e88e5", ax=ax, binwidth=2)
    mu, std = norm.fit(changes)
    x = np.linspace(changes.min(), changes.max(), 100)
    ax.plot(x, norm.pdf(x, mu, std), 'white', lw=2.5)
    for p, c in zip([5,25,50,75,95], ['#ef5350','#ffb300','#4caf50','#42a5f5','#ab47bc']):
        val = np.percentile(changes, p)
        ax.axvline(val, color=c, ls='--', alpha=0.8)
        ax.text(val+0.5, ax.get_ylim()[1]*0.92, f'{val:.1f}', color=c, fontsize=10)
    ax.set_title(f"Histograma {period_label} - {main}" + (f" / {sec}" if sec else "") + (f" / {third}" if third else "")+ ccl_text)
    apply_dark_theme(ax)
    add_watermark(ax)
    st.pyplot(fig)

def create_period_heatmap(monthly_data, main, sec, third, color_order, analysis_period, period_label,
                          daily_data, price_col, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader(f"Mapa de Calor {period_label} (+ columna de cambio anual)")
    if analysis_period == "Mes a Mes":
        idx = monthly_data.index.month
        names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    else:
        idx = monthly_data.index.quarter
        names = ['Q1','Q2','Q3','Q4']
    pivot = monthly_data.pivot_table(values=f'Cambio {period_label} (%)',
                                     index=monthly_data.index.year, columns=idx, aggfunc='mean')

    # Columna extra con el cambio anual (mismo método last-vs-last, calculado sobre el precio diario)
    yearly_change = calculate_yearly_changes(daily_data, price_col)
    combined = pivot.copy()
    n_period_cols = pivot.shape[1]
    combined['__gap__'] = np.nan          # columna en blanco: separa visualmente meses/trimestres de "Año"
    combined['Año'] = yearly_change.reindex(pivot.index)
    gap_col_idx = n_period_cols
    year_col_idx = n_period_cols + 1

    mask_main = np.zeros(combined.shape, dtype=bool)
    mask_main[:, gap_col_idx] = True      # ocultar la columna vacía en el heatmap principal
    mask_main[:, year_col_idx] = True     # ocultar la columna 'Año' en el heatmap principal

    mask_year = np.ones(combined.shape, dtype=bool)
    mask_year[:, year_col_idx] = combined['Año'].isna().values  # mostrar solo donde hay dato anual

    valid_period_vals = pivot.values[~np.isnan(pivot.values)]
    if valid_period_vals.size > 0:
        period_vmin = min(-100, valid_period_vals.min())
        period_vmax = max(valid_period_vals.max(), 1)
    else:
        period_vmin, period_vmax = -100, 100
    period_norm = TwoSlopeNorm(vmin=period_vmin, vcenter=0, vmax=period_vmax)

    valid_year_vals = combined['Año'].dropna()
    if not valid_year_vals.empty:
        year_vmin = min(-100, valid_year_vals.min())
        year_vmax = max(valid_year_vals.max(), 1)
    else:
        year_vmin, year_vmax = -100, 100
    year_norm = TwoSlopeNorm(vmin=year_vmin, vcenter=0, vmax=year_vmax)

    fig, ax = plt.subplots(figsize=(13.5, max(6, len(pivot)*0.4)))
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="3%", pad=0.6)
    cax2 = divider.append_axes("right", size="3%", pad=0.9)

    sns.heatmap(combined, mask=mask_main, cmap=get_custom_cmap(color_order), annot=True, fmt=".1f",
                norm=period_norm, linewidths=0.5, linecolor='#0e1117', ax=ax, cbar_ax=cax1,
                cbar_kws={'label': f'Cambio {period_label} (%)'})
    sns.heatmap(combined, mask=mask_year, cmap=get_yearly_cmap(color_order), annot=True, fmt=".1f",
                norm=year_norm, linewidths=0.5, linecolor='#0e1117',
                ax=ax, cbar_ax=cax2, cbar_kws={'label': 'Cambio Anual (%)'})

    ax.set_xticklabels([names[i-1] for i in pivot.columns] + ['', 'Año'], rotation=45)
    ax.set_title(f"Heatmap {period_label} - {main}" + (f" / {sec}" if sec else "") + (f" / {third}" if third else "")+ ccl_text)
    apply_dark_theme(ax)

    # Repetir el año dentro de la columna en blanco, para no tener que mirar hasta la izquierda
    for i, year in enumerate(combined.index):
        ax.text(gap_col_idx + 0.5, i + 0.5, str(year), color='white', ha='center', va='center',
                fontsize=9, fontweight='bold')

    for cax in (cax1, cax2):
        cax.tick_params(colors='white')
        cax.yaxis.label.set_color('white')
        cax.set_facecolor('#0e1117')
    add_watermark(ax)
    st.pyplot(fig)

def create_average_changes_visualization(monthly_data, metric, main, sec, third, analysis_period, period_label, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader(f"Cambios {metric} {period_label}es")
    if analysis_period == "Mes a Mes":
        grp = monthly_data.index.month
        lbl = "Mes"
        names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    else:
        grp = monthly_data.index.quarter
        lbl = "Trimestre"
        names = ['Q1','Q2','Q3','Q4']
    col_name = f'Cambio {period_label} (%)'

    if metric == "Promedio":
        avg = monthly_data.groupby(grp)[col_name].mean()
    elif metric == "Mediana":
        avg = monthly_data.groupby(grp)[col_name].median()
    else:
        st.error(f"Métrica no soportada: {metric}")
        return

    avg.index = [names[i-1] for i in avg.index]
    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(avg.index, avg, color=['#4caf50' if v>=0 else '#ef5350' for v in avg])
    ax.set_title(f"{metric} por {lbl} - {main}" + (f" / {sec}" if sec else "") + (f" / {third}" if third else "")+ ccl_text)
    apply_dark_theme(ax)
    add_watermark(ax)
    st.pyplot(fig)

def create_period_ranking(monthly_data, main, sec, third, analysis_period, period_label,
                          daily_data, price_col, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader(f"Ranking {period_label}es Positivos/Negativos (+ total de años)")
    if analysis_period == "Mes a Mes":
        grp = monthly_data.index.month
        names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    else:
        grp = monthly_data.index.quarter
        names = ['Q1','Q2','Q3','Q4']
    pos = monthly_data.groupby(grp)[f'Cambio {period_label} (%)'].apply(lambda x: (x>0).sum())
    neg = monthly_data.groupby(grp)[f'Cambio {period_label} (%)'].apply(lambda x: (x<0).sum())

    # Bucket extra: total histórico de años positivos vs negativos (mismo cálculo last-vs-last que el heatmap)
    yearly_change = calculate_yearly_changes(daily_data, price_col).dropna()
    pos_years = int((yearly_change > 0).sum())
    neg_years = int((yearly_change < 0).sum())

    x_months = np.arange(len(names))
    x_year = len(names) + 1  # deja un hueco de 1 posición como separación real, no solo una línea

    fig, ax = plt.subplots(figsize=(11,6))
    ax.bar(x_months - 0.2, pos, 0.4, label='Positivos', color='#4caf50')
    ax.bar(x_months + 0.2, neg, 0.4, label='Negativos', color='#ef5350')
    ax.bar(x_year - 0.2, pos_years, 0.4, color='#4caf50')
    ax.bar(x_year + 0.2, neg_years, 0.4, color='#ef5350')
    ax.axvline((x_months[-1] + x_year) / 2, color='white', ls=':', alpha=0.4)
    ax.set_xticks(list(x_months) + [x_year])
    ax.set_xticklabels(names + ['Total Años\n(histórico)'], rotation=45)
    ax.legend()
    ax.set_title(f"Ranking {period_label} - {main}" + (f" / {sec}" if sec else "") + (f" / {third}" if third else "")+ ccl_text)
    apply_dark_theme(ax)
    add_watermark(ax)
    st.pyplot(fig)

def create_yearly_ranking(monthly_data, main, sec, third, period_label, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader("Ranking Anual Positivos/Negativos")
    pos = monthly_data.groupby(monthly_data.index.year)[f'Cambio {period_label} (%)'].apply(lambda x: (x>0).sum())
    neg = monthly_data.groupby(monthly_data.index.year)[f'Cambio {period_label} (%)'].apply(lambda x: (x<0).sum())
    years = sorted(pos.index)
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(years))
    ax.bar(x - 0.2, [pos.get(y,0) for y in years], 0.4, label='Positivos', color='#4caf50')
    ax.bar(x + 0.2, [neg.get(y,0) for y in years], 0.4, label='Negativos', color='#ef5350')
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.legend()
    ax.set_title(f"Ranking Anual - {main}" + (f" / {sec}" if sec else "") + (f" / {third}" if third else "")+ ccl_text)
    apply_dark_theme(ax)
    add_watermark(ax)
    st.pyplot(fig)

def display_statistics(monthly_data, period_label):
    st.subheader("Estadísticas Descriptivas")
    ch = monthly_data[f'Cambio {period_label} (%)'].dropna()
    cols = st.columns(3)
    cols[0].metric("Promedio", f"{ch.mean():.2f}%")
    cols[0].metric("Mediana", f"{ch.median():.2f}%")
    cols[1].metric("Máximo", f"{ch.max():.2f}%")
    cols[1].metric("Mínimo", f"{ch.min():.2f}%")
    cols[2].metric("Volatilidad", f"{ch.std():.2f}%")
    cols[2].metric("% Positivos", f"{(ch>0).mean()*100:.1f}%")

def analyze_streaks(monthly_data, main_ticker, period_label):
    ch = monthly_data[f'Cambio {period_label} (%)'].dropna()
    if len(ch) < 3:
        st.info("Datos insuficientes para rachas.")
        return
    streaks = calculate_streaks(ch)
    if streaks.empty:
        st.info("No se detectaron rachas significativas.")
        return
    for _, r in streaks.iterrows():
        dir_str = "positiva" if r['value'] > 0 else "negativa"
        st.write(f"Racha **{dir_str}** de **{r['length']}** {period_label.lower()}es | {r['start'].date()} → {r['end'].date()}")

def create_visualizations(monthly_data, main, sec, third, metric_opt, color_ord, 
                         anal_per, per_lbl, daily_data, price_col, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    title_suf = f" / {sec}" if sec else ""
    title_suf += f" / {third}" if third else ""

    # Gráfico de líneas principal
    fig_line = px.line(monthly_data, x=monthly_data.index, y=f'Cambio {per_lbl} (%)',
                       title=f"Variaciones {per_lbl}es - {main}{title_suf}{ccl_text}",
                       template='plotly_dark')
    fig_line.update_traces(mode='lines+markers')
    fig_line.add_annotation(text="MTaurus - X: MTaurus_ok", xref="paper", yref="paper", 
                            x=0.5, y=0.5, showarrow=False, font_size=34, 
                            opacity=0.22, textangle=-42)
    st.plotly_chart(fig_line, use_container_width=True)

    # Llamadas a las funciones hijas (ahora con el parámetro apply_ccl)
    create_histogram_with_gaussian(monthly_data, main, sec, third, per_lbl, apply_ccl)
    create_period_heatmap(monthly_data, main, sec, third, color_ord, anal_per, per_lbl, daily_data, price_col, apply_ccl)
    create_average_changes_visualization(monthly_data, metric_opt, main, sec, third, anal_per, per_lbl, apply_ccl)
    create_period_ranking(monthly_data, main, sec, third, anal_per, per_lbl, daily_data, price_col, apply_ccl)
    create_yearly_ranking(monthly_data, main, sec, third, per_lbl, apply_ccl)
    
    display_statistics(monthly_data, per_lbl)
    
    if price_col in daily_data.columns:
        create_drawdown_visualization(daily_data[price_col], main, sec, third, apply_ccl)
    
    with st.expander(f"📊 Análisis de Rachas ({per_lbl.lower()}es)", expanded=False):
        analyze_streaks(monthly_data, main, per_lbl)

# ─── APP PRINCIPAL ───
def main():
    st.title("📈 Análisis de Variación de Precios - MTaurus")
    st.markdown("Seguinos en [X → @MTaurus_ok](https://x.com/MTaurus_ok)")

    data_src = st.selectbox(
        "Fuente de datos",
        options=['yfinance', 'analisistecnico', 'iol', 'byma'],
        help="Elegí de dónde bajar los precios históricos.\n\n"
             "- yfinance: datos globales (Yahoo Finance), incluye muchos tickers argentinos\n"
             "- analisistecnico / iol / byma: fuentes argentinas específicas (pueden tener más precisión local pero dependen de cookies/sesiones)"
    )

    apply_ccl = st.checkbox(
        "Aplicar ratio CCL",
        value=False,
        help="Marca esta opción si querés 'dolarizar' el ticker principal dividiendo su precio por el dólar CCL.\n\n"
             "- Con yfinance: usa YPFD.BA / YPF\n"
             "- Con otras fuentes: usa GD30 / GD30C\n\n"
             "Muy útil para ver la performance en dólares CCL y comparar con activos en el exterior."
    )

    main_ticker = st.text_input(
        "Ticker principal",
        value="",
        help="El activo que querés analizar en profundidad (aparece en todos los títulos y gráficos principales).\n\n"
             "Ejemplos: GGAL.BA, AAPL, BMA, AL30, MELI, YPF, TSLA"
    )

    sec_ticker = st.text_input(
        "Segundo ticker (opcional)",
        value="",
        help="Úsalo para crear un ratio o dividir el principal por este ticker.\n\n"
             "Ejemplos comunes:\n"
             "- Dividir un ADR por su equivalente en pesos (GGAL / GGAL.BA)\n"
             "- Comparar con un banco o sector (COME / BMA)\n"
             "- Normalizar por otro activo"
    )

    third_ticker = st.text_input(
        "Tercer ticker (opcional)",
        value="",
        help="Permite agregar un segundo divisor en cadena.\n\n"
             "Ejemplo: principal / segundo / tercero\n"
             "Útil para ratios más complejos (poco común, pero disponible)."
    )

    col1, col2 = st.columns(2)
    with col1:
        start_dt = st.date_input(
            "Desde",
            value=pd.to_datetime("2010-01-01").date(),  # This remains the default selected date
            min_value=pd.to_datetime("1920-01-01").date(), # <--- ADD THIS LINE
            help="Fecha más antigua desde la cual descargar datos históricos.\n\n"
                 "Cuanto más atrás, más completo el análisis (pero puede tardar más en cargar)."
        )
    with col2:
        end_dt = st.date_input(
            "Hasta",
            value=datetime.today().date(),
            help="Fecha más reciente de los datos.\n\n"
                 "Por defecto es hoy. Podés poner una fecha anterior si querés comparar períodos específicos."
        )

    period_choice = st.radio(
        "Período de análisis",
        options=["Mes a Mes", "Trimestre a Trimestre"],
        help="Define cómo agrupamos y calculamos las variaciones:\n\n"
             "- Mes a Mes: variación mes contra mes anterior (más detalle)\n"
             "- Trimestre a Trimestre: variación trimestre contra trimestre anterior (más estable, menos ruido)"
    )
    freq = 'ME' if period_choice == "Mes a Mes" else 'QE'          # ← CAMBIO AQUÍ
    per_label = "Mensual" if period_choice == "Mes a Mes" else "Trimestral"

    metric_choice = st.radio(
        "Métrica",
        options=["Promedio", "Mediana"],
        help="Qué valor representativo usar en barras, heatmap y resúmenes:\n\n"
             "- Promedio: sensible a valores extremos (outliers)\n"
             "- Mediana: más robusta, ignora mejor los valores muy altos/bajos"
    )

    color_choice = st.selectbox(
        "Colores Heatmap",
        options=["Rojo → Blanco → Verde", "Verde → Blanco → Rojo"],
        help="Orden de colores en el mapa de calor:\n\n"
             "- Rojo → Blanco → Verde: rojo = caídas fuertes, verde = subas fuertes (el más intuitivo para la mayoría)\n"
             "- Verde → Blanco → Rojo: al revés (a veces preferido en finanzas para que positivo sea verde)"
    )
    cmap_key = 'red_white_green' if "Rojo" in color_choice else 'green_white_red'

    # Resto del código sigue igual (tickers_set, if st.button("Analizar") ... )
    # ...

    tickers_set = {t for t in [main_ticker, sec_ticker, third_ticker] if t}
    if apply_ccl:
        tickers_set |= {'YPFD.BA', 'YPF'} if data_src == 'yfinance' else {'GD30', 'GD30C'}

    if st.button("Analizar", type="primary") and main_ticker:
        with st.spinner("Cargando datos..."):
            raw_data = fetch_data(tickers_set, start_dt, end_dt, data_src)
            if not raw_data:
                st.error("No se obtuvieron datos.")
                return
            aligned_data = align_dates(raw_data)
            ratio_series = evaluate_ratio(main_ticker, sec_ticker, third_ticker, aligned_data, apply_ccl, data_src)
            if ratio_series is None or ratio_series.empty:
                st.error("No se pudo generar la serie ajustada.")
                return

            df_daily = ratio_series.to_frame(name='Price')
            df_daily.index = pd.to_datetime(df_daily.index)

            # ─── DEBUG TEMPORAL: sacar después de confirmar el nuevo empalme (24/3/2003) ───
            if apply_ccl and main_ticker.upper() == '^MERV':
                st.warning("🔧 Debug temporal activo — ver el expander arriba de los gráficos")
                with st.expander("🔧 Debug temporal: detalle mar-2003 (nuevo punto de empalme)", expanded=True):
                    st.write(f"Versión de yfinance: {yf.__version__}")
                    ypfd_dbg, ypf_dbg = descargar_ypfd_ypf_crudo()
                    ratio_dbg = (ypfd_dbg * 10) / ypf_dbg
                    st.write("Ratio YPFD.BA/YPF (x10 ya aplicado) alrededor del nuevo empalme:")
                    st.dataframe(ratio_dbg.loc['2003-03-10':'2003-04-10'])
                    st.write("df_daily (Price ya con CCL aplicado) alrededor del nuevo empalme:")
                    st.dataframe(df_daily.loc['2003-03-10':'2003-04-10'])

            df_period = df_daily.resample(freq).last()
            df_period[f'Cambio {per_label} (%)'] = df_period['Price'].pct_change() * 100

            create_visualizations(
                df_period, main_ticker, sec_ticker, third_ticker,
                metric_choice, cmap_key, period_choice, per_label,
                df_daily, 'Price',
                apply_ccl   # ← este parámetro ya está correcto
            )

    st.markdown("---")
    st.caption("© 2025 MTaurus • @MTaurus_ok • Buenos Aires")

if __name__ == "__main__":
    main()
