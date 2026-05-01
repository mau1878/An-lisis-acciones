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
            if 'YPFD.BA' in data and 'YPF' in data:
                ratio = data['YPFD.BA']['YPFD_BA'] / data['YPF']['YPF']
                result = result / ratio
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

def create_period_heatmap(monthly_data, main, sec, third, color_order, analysis_period, period_label, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader(f"Mapa de Calor {period_label}")
    if analysis_period == "Mes a Mes":
        idx = monthly_data.index.month
        names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    else:
        idx = monthly_data.index.quarter
        names = ['Q1','Q2','Q3','Q4']
    pivot = monthly_data.pivot_table(values=f'Cambio {period_label} (%)',
                                     index=monthly_data.index.year, columns=idx, aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.4)))
    sns.heatmap(pivot, cmap=get_custom_cmap(color_order), annot=True, fmt=".1f",
                center=0, linewidths=0.5, ax=ax)
    ax.set_xticklabels([names[i-1] for i in pivot.columns], rotation=45)
    ax.set_title(f"Heatmap {period_label} - {main}" + (f" / {sec}" if sec else "") + (f" / {third}" if third else "")+ ccl_text)
    apply_dark_theme(ax)
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

def create_period_ranking(monthly_data, main, sec, third, analysis_period, period_label, apply_ccl=False):
    ccl_text = " - CCL aplicado" if apply_ccl else ""
    st.subheader(f"Ranking {period_label}es Positivos/Negativos")
    if analysis_period == "Mes a Mes":
        grp = monthly_data.index.month
        names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    else:
        grp = monthly_data.index.quarter
        names = ['Q1','Q2','Q3','Q4']
    pos = monthly_data.groupby(grp)[f'Cambio {period_label} (%)'].apply(lambda x: (x>0).sum())
    neg = monthly_data.groupby(grp)[f'Cambio {period_label} (%)'].apply(lambda x: (x<0).sum())
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, pos, 0.4, label='Positivos', color='#4caf50')
    ax.bar(x + 0.2, neg, 0.4, label='Negativos', color='#ef5350')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
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
    create_period_heatmap(monthly_data, main, sec, third, color_ord, anal_per, per_lbl, apply_ccl)
    create_average_changes_visualization(monthly_data, metric_opt, main, sec, third, anal_per, per_lbl, apply_ccl)
    create_period_ranking(monthly_data, main, sec, third, anal_per, per_lbl, apply_ccl)
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
