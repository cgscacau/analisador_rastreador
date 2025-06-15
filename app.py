# =============================================================================
# Importação das bibliotecas necessárias
# =============================================================================
import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# =============================================================================
# Configuração da Página do Streamlit
# =============================================================================
st.set_page_config(
    page_title="Análise Quant de Ações",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Funções de Carregamento e Coleta de Dados (Back-End)
# =============================================================================
def load_tickers_from_file(filename):
    """Carrega uma lista de tickers de um arquivo de texto."""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        st.error(f"Arquivo de tickers não encontrado: {filename}.")
        return []

TICKERS_B3 = load_tickers_from_file('ibov_tickers.txt')
TICKERS_US = load_tickers_from_file('sp500_tickers.txt')

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """Busca dados fundamentalistas, históricos e de dividendos de um ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2y")
        dividends = stock.dividends
        if hist.empty or not info.get('symbol'):
            return None, None, None
        return info, hist, dividends
    except Exception as e:
        # <<< MELHORIA: NÃO MOSTRA O ERRO 429 NA INTERFACE DO USUÁRIO >>>
        if "429" not in str(e):
            st.error(f"Erro ao buscar dados para {ticker}: {e}")
        return None, None, None

# =============================================================================
# Funções de Análise Fundamentalista
# =============================================================================
def run_fundamental_analysis(tickers):
    """Executa a análise fundamentalista com cálculo manual de DY e pausas."""
    data = []
    progress_bar = st.progress(0, text="Buscando dados fundamentalistas...")
    
    for i, ticker in enumerate(tickers):
        info, hist, dividends = get_stock_data(ticker)
        
        # <<< CORREÇÃO: AUMENTA A PAUSA PARA 0.5 SEGUNDOS >>>
        time.sleep(0.5)

        if info and hist is not None and not hist.empty and info.get('trailingPE'):
            today = pd.Timestamp.now()
            one_year_ago = today - pd.DateOffset(years=1)
            ttm_dividends = dividends[dividends.index.tz_localize(None) > one_year_ago].sum()
            last_price = hist['Close'].iloc[-1]
            calculated_dy = (ttm_dividends / last_price) if last_price > 0 else 0
            
            data.append({
                'Ticker': ticker, 'P/L': info.get('trailingPE'), 'P/VP': info.get('priceToBook'),
                'ROE': info.get('returnOnEquity'), 'Dividend Yield': calculated_dy,
                'Dívida/PL': info.get('debtToEquity'), 'Marg. Líquida': info.get('profitMargins'),
                'ROIC (proxy)': info.get('returnOnAssets') or (info.get('freeCashflow', 0) / info.get('totalAssets', 1)),
                'Earnings Yield': 1 / info.get('trailingPE') if info.get('trailingPE') else 0
            })
        progress_bar.progress((i + 1) / len(tickers), text=f"Analisando {ticker}...")
        
    progress_bar.empty()

    if not data: return pd.DataFrame()
    df = pd.DataFrame(data).set_index('Ticker')
    df.dropna(subset=['P/L', 'P/VP'], inplace=True)
    
    df['Piotroski (Simples)'] = (df['ROE'].fillna(0) > 0).astype(int) + (df['P/L'].fillna(-1) > 0).astype(int) + (df['Dívida/PL'].fillna(999) < 200).astype(int) + (df['Marg. Líquida'].fillna(0) > 0).astype(int)
    df['Magic_Rank'] = df['ROIC (proxy)'].rank(ascending=False) + df['Earnings Yield'].rank(ascending=False)
    df['Ranking_Final'] = df['P/L'].rank() * 0.4 + df['P/VP'].rank() * 0.3 + df['ROE'].rank(ascending=False) * 0.3
    return df

# =============================================================================
# Funções de Análise Técnica (sem alterações)
# =============================================================================
def run_technical_analysis(hist_data):
    if hist_data is None or len(hist_data) < 200: return None, "Dados Insuficientes", None, None
    hist_data.ta.sma(length=9, append=True, col_names=('SMA_9',)); hist_data.ta.sma(length=21, append=True, col_names=('SMA_21',));
    hist_data.ta.rsi(length=14, append=True, col_names=('RSI_14',)); hist_data.ta.macd(fast=12, slow=26, signal=9, append=True); 
    hist_data.ta.stoch(k=14, d=3, smooth_k=3, append=True); hist_data.ta.ichimoku(append=True)
    last, prev = hist_data.iloc[-1], hist_data.iloc[-2]
    tendencia = "Lateral"
    if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']: tendencia = "Alta"
    elif last['Close'] < last['ISA_9'] and last['Close'] < last['ISB_26']: tendencia = "Baixa"
    sinais = []
    if last['SMA_9'] > last['SMA_21'] and prev['SMA_9'] <= prev['SMA_21']: sinais.append("🟢 **Compra**: Cruz. Médias (9>21)")
    if last['SMA_9'] < last['SMA_21'] and prev['SMA_9'] >= prev['SMA_21']: sinais.append("🔴 **Venda**: Cruz. Médias (9<21)")
    if last['RSI_14'] < 30: sinais.append(f"🟢 **Compra**: RSI Sobrevendido ({last['RSI_14']:.1f})")
    if last['RSI_14'] > 70: sinais.append(f"🔴 **Venda**: RSI Sobrecomprado ({last['RSI_14']:.1f})")
    if last['MACD_12_26_9'] > last['MACDs_12_26_9'] and prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']: sinais.append("🟢 **Compra**: Cruz. MACD")
    if last['MACD_12_26_9'] < last['MACDs_12_26_9'] and prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']: sinais.append("🔴 **Venda**: Cruz. MACD")
    if last['STOCHk_14_3_3'] < 20 and last['STOCHd_14_3_3'] < 20: sinais.append(f"🟢 **Compra**: Estoc. Sobrevendido")
    if last['STOCHk_14_3_3'] > 80 and last['STOCHd_14_3_3'] > 80: sinais.append(f"🔴 **Venda**: Estoc. Sobrecomprado")
    if last['ITS_9'] > last['IKS_26'] and prev['ITS_9'] <= prev['IKS_26']: sinais.append("🟢 **Compra**: Cruz. Ichimoku")
    if last['ITS_9'] < last['IKS_26'] and prev['ITS_9'] >= prev['IKS_26']: sinais.append("🔴 **Venda**: Cruz. Ichimoku")
    sinais_compra, sinais_venda = len([s for s in sinais if "Compra" in s]), len([s for s in sinais if "Venda" in s])
    status_tecnico = "Potencial de Compra" if sinais_compra > sinais_venda else "Potencial de Venda" if sinais_venda > sinais_compra else "Neutro"
    return hist_data, tendencia, status_tecnico, "\n".join(sinais)

def plot_technical_analysis(df, ticker):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(f'Preço & Ichimoku - {ticker}', 'RSI', 'MACD', 'Estocástico'), row_heights=[0.6, 0.1, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candles'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ISA_9'], line=dict(color='rgba(0,0,0,0)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ISB_26'], line=dict(color='rgba(0,0,0,0)'), name='Nuvem Ichimoku', fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)' if df['ISA_9'].iloc[-1] > df['ISB_26'].iloc[-1] else 'rgba(255, 0, 0, 0.2)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ITS_9'], name='Tenkan-sen', line=dict(color='blue', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['IKS_26'], name='Kijun-sen', line=dict(color='red', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI'), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color='red'); fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color='green')
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], name='MACD'), row=3, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name='Signal'), row=3, col=1); fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='Histogram'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STOCHk_14_3_3'], name='%K'), row=4, col=1); fig.add_trace(go.Scatter(x=df.index, y=df['STOCHd_14_3_3'], name='%D'), row=4, col=1); fig.add_hline(y=80, row=4, col=1, line_dash="dash", line_color='red'); fig.add_hline(y=20, row=4, col=1, line_dash="dash", line_color='green')
    fig.update_layout(height=800, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Interface do Streamlit (UI) com Melhorias Estéticas
# =============================================================================
st.title("🚀 Análise Quantitativa de Ações")
st.markdown("Uma ferramenta para análise fundamentalista e técnica.")
st.sidebar.header("Configurações de Análise")
market = st.sidebar.radio("Escolha o Mercado:", ('Bolsa Brasileira (B3)', 'Bolsa Americana (S&P 500)'))
tickers_list = TICKERS_B3 if market == 'Bolsa Brasileira (B3)' else TICKERS_US
default_tickers = tickers_list[:10] if tickers_list else []
selected_tickers = st.sidebar.multiselect("Selecione os Tickers:", tickers_list, default=default_tickers)

if st.sidebar.button("Executar Análise", type="primary"):
    if not selected_tickers: st.warning("Por favor, selecione pelo menos um ticker.")
    else:
        st.header("📊 Análise Fundamentalista Comparativa")
        df_fundamentals = run_fundamental_analysis(selected_tickers)
        if not df_fundamentals.empty:
            table_styles = {'props': [('font-size', '18px'), ('text-align', 'center')]}
            tab1, tab2, tab3 = st.tabs(["Ranking Customizado", "Magic Formula", "Piotroski Score"])
            with tab1:
                st.dataframe(df_fundamentals.sort_values('Ranking_Final')[['P/L', 'P/VP', 'ROE', 'Dividend Yield', 'Ranking_Final']]
                             .style.format({'P/L': '{:.2f}', 'P/VP': '{:.2f}', 'ROE': '{:.2%}', 'Dividend Yield': '{:.2%}', 'Ranking_Final': '{:.1f}'})
                             .set_table_styles([table_styles])
                             .background_gradient(cmap='Greens_r', subset=['Ranking_Final']), use_container_width=True)
            with tab2:
                st.dataframe(df_fundamentals.sort_values('Magic_Rank')[['ROIC (proxy)', 'Earnings Yield', 'Magic_Rank']]
                             .style.format({'ROIC (proxy)': '{:.2%}', 'Earnings Yield': '{:.2%}', 'Magic_Rank': '{:.0f}'})
                             .set_table_styles([table_styles])
                             .background_gradient(cmap='Greens_r', subset=['Magic_Rank']), use_container_width=True)
            with tab3:
                st.dataframe(df_fundamentals.sort_values('Piotroski (Simples)', ascending=False)[['Piotroski (Simples)']]
                             .style.set_table_styles([table_styles])
                             .background_gradient(cmap='Greens', subset=['Piotroski (Simples)']), use_container_width=True)
            st.session_state['fundamental_results'] = df_fundamentals.index.tolist()
        else:
            st.warning("Não foi possível buscar os dados. A API pode estar temporariamente sobrecarregada. Tente novamente em alguns instantes ou com menos tickers.")

if 'fundamental_results' in st.session_state and st.session_state['fundamental_results']:
    st.header("📈 Análise Técnica Individual")
    selected_ticker_for_tech = st.selectbox("Selecione um ticker:", st.session_state['fundamental_results'])
    if selected_ticker_for_tech:
        with st.spinner(f"Gerando análise técnica para {selected_ticker_for_tech}..."):
            info, hist_data, dividends = get_stock_data(selected_ticker_for_tech)
            if hist_data is not None and info:
                df_tech, tendencia, status_tecnico, sinais = run_technical_analysis(hist_data)
                if df_tech is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tendência (Ichimoku)", tendencia)
                    col2.metric("Status dos Sinais", status_tecnico)
                    with col3.expander("Ver Sinais Detalhados"): st.markdown(sinais if sinais else "Nenhum sinal técnico claro.")
                    
                    st.subheader("Resumo da Análise")
                    today_sum = pd.Timestamp.now(); one_year_ago_sum = today_sum - pd.DateOffset(years=1)
                    ttm_dividends_sum = dividends[dividends.index.tz_localize(None) > one_year_ago_sum].sum()
                    last_price_sum = hist_data['Close'].iloc[-1]
                    correct_dy = (ttm_dividends_sum / last_price_sum) if last_price_sum > 0 else 0
                    
                    fundamental_summary = f"Do ponto de vista **fundamentalista**, {info.get('longName', selected_ticker_for_tech)} apresenta P/L de `{info.get('trailingPE'):.2f}`, P/VP de `{info.get('priceToBook'):.2f}` e Dividend Yield de `{correct_dy:.2%}`."
                    technical_summary = f"Do ponto de vista **técnico**, a tendência principal é de **{tendencia}** e o balanço dos sinais aponta para **{status_tecnico}**."
                    st.markdown(fundamental_summary); st.markdown(technical_summary)
                    plot_technical_analysis(df_tech, selected_ticker_for_tech)