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
# Funções de Carregamento e Análise de Dados (Back-End)
# =============================================================================
def load_tickers_from_file(filename):
    """Carrega uma lista de tickers de um arquivo de texto."""
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        return []

@st.cache_data(ttl=3600 * 4) # Cache de 4 horas
def carregar_dados_fundamentalistas(tickers):
    """
    Função principal que busca todos os dados fundamentalistas de uma vez,
    com tratamento de erro silencioso.
    """
    data = []
    progress_bar = st.progress(0, text="Buscando dados fundamentalistas...")
    
    for i, ticker_str in enumerate(tickers):
        try:
            # Pausa para não sobrecarregar a API
            time.sleep(0.1)
            
            ticker = yf.Ticker(ticker_str)
            info = ticker.info

            # Validação mínima para garantir que temos dados
            if not info or info.get('trailingPE') is None:
                continue # Pula para o próximo ticker se não houver dados essenciais

            # Cálculo manual do DY para maior precisão
            preco_atual = info.get('currentPrice', info.get('previousClose', 0))
            if preco_atual == 0: continue
            divs_12m = ticker.dividends.last('365D').sum()
            calculated_dy = (divs_12m / preco_atual) if preco_atual > 0 else 0
            
            data.append({
                'Ticker': ticker_str, 'P/L': info.get('trailingPE'), 'P/VP': info.get('priceToBook'),
                'ROE': info.get('returnOnEquity'), 'Dividend Yield': calculated_dy,
                'Dívida/PL': info.get('debtToEquity'), 'Marg. Líquida': info.get('profitMargins'),
                'ROIC (proxy)': info.get('returnOnAssets') or (info.get('freeCashflow', 0) / info.get('totalAssets', 1)),
                'Earnings Yield': 1 / info.get('trailingPE')
            })
        except Exception:
            # Tratamento de erro silencioso. Se falhar, simplesmente pula.
            pass
        finally:
            progress_bar.progress((i + 1) / len(tickers), text=f"Buscando: {ticker_str}")

    progress_bar.empty()
    if not data: return pd.DataFrame()
    
    df = pd.DataFrame(data).set_index('Ticker')
    df.dropna(subset=['P/L', 'P/VP'], inplace=True)
    
    # Cálculos de Ranking
    df['Piotroski (Simples)'] = (df['ROE'].fillna(0) > 0).astype(int) + (df['P/L'].fillna(-1) > 0).astype(int) + (df['Dívida/PL'].fillna(999) < 200).astype(int) + (df['Marg. Líquida'].fillna(0) > 0).astype(int)
    df['Magic_Rank'] = df['ROIC (proxy)'].rank(ascending=False) + df['Earnings Yield'].rank(ascending=False)
    df['Ranking_Final'] = df['P/L'].rank() * 0.4 + df['P/VP'].rank() * 0.3 + df['ROE'].rank(ascending=False) * 0.3
    return df

@st.cache_data(ttl=3600)
def get_technical_data(ticker):
    """Busca dados para a análise técnica individual."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return None
        
        # Cálculo dos indicadores
        hist.ta.sma(length=9, append=True, col_names=('SMA_9',)); hist.ta.sma(length=21, append=True, col_names=('SMA_21',));
        hist.ta.rsi(length=14, append=True, col_names=('RSI_14',)); hist.ta.macd(fast=12, slow=26, signal=9, append=True); 
        hist.ta.stoch(k=14, d=3, smooth_k=3, append=True); hist.ta.ichimoku(append=True)
        return hist
    except Exception:
        return None

def run_technical_analysis(hist_data):
    """Interpreta os dados técnicos já calculados."""
    if hist_data is None or len(hist_data) < 200: return "Dados Insuficientes", None, None
    
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
    
    return tendencia, status_tecnico, "\n".join(sinais)

def plot_technical_analysis(df, ticker):
    """Plota os gráficos de análise técnica."""
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
# Interface do Streamlit (UI) - Fluxo Principal
# =============================================================================
st.title("🚀 Análise Quantitativa de Ações")
st.markdown("Uma ferramenta para análise fundamentalista e técnica.")

# --- Carregamento Único dos Dados ---
TICKERS_B3 = load_tickers_from_file('ibov_tickers.txt')
TICKERS_US = load_tickers_from_file('sp500_tickers.txt')
df_completo_b3 = carregar_dados_fundamentalistas(TICKERS_B3)
df_completo_us = carregar_dados_fundamentalistas(TICKERS_US)

st.sidebar.header("Configurações de Análise")
market = st.sidebar.radio("Escolha o Mercado:", ('Bolsa Brasileira (B3)', 'Bolsa Americana (S&P 500)'))

if market == 'Bolsa Brasileira (B3)':
    df_mercado = df_completo_b3
    tickers_list = df_mercado.index.tolist() if not df_mercado.empty else []
else:
    df_mercado = df_completo_us
    tickers_list = df_mercado.index.tolist() if not df_mercado.empty else []

default_tickers = tickers_list[:10] if tickers_list else []
selected_tickers = st.sidebar.multiselect("Selecione os Tickers:", tickers_list, default=default_tickers)

if st.sidebar.button("Executar Análise", type="primary"):
    if not selected_tickers:
        st.warning("Por favor, selecione pelo menos um ticker.")
    elif df_mercado.empty:
        st.error("A base de dados fundamentalistas está vazia. Tente recarregar a página.")
    else:
        # Filtra os dados já carregados, sem chamar a API novamente
        df_fundamentals = df_mercado[df_mercado.index.isin(selected_tickers)]
        
        st.header("📊 Análise Fundamentalista Comparativa")
        if not df_fundamentals.empty:
            table_styles = {'props': [('font-size', '18px'), ('text-align', 'center')]}
            tab1, tab2, tab3 = st.tabs(["Ranking Customizado", "Magic Formula", "Piotroski Score"])
            with tab1:
                st.dataframe(df_fundamentals.sort_values('Ranking_Final')[['P/L', 'P/VP', 'ROE', 'Dividend Yield', 'Ranking_Final']]
                             .style.format({'P/L': '{:.2f}', 'P/VP': '{:.2f}', 'ROE': '{:.2%}', 'Dividend Yield': '{:.2%}', 'Ranking_Final': '{:.1f}'})
                             .set_table_styles([table_styles]).background_gradient(cmap='Greens_r', subset=['Ranking_Final']), use_container_width=True)
            with tab2:
                st.dataframe(df_fundamentals.sort_values('Magic_Rank')[['ROIC (proxy)', 'Earnings Yield', 'Magic_Rank']]
                             .style.format({'ROIC (proxy)': '{:.2%}', 'Earnings Yield': '{:.2%}', 'Magic_Rank': '{:.0f}'})
                             .set_table_styles([table_styles]).background_gradient(cmap='Greens_r', subset=['Magic_Rank']), use_container_width=True)
            with tab3:
                st.dataframe(df_fundamentals.sort_values('Piotroski (Simples)', ascending=False)[['Piotroski (Simples)']]
                             .style.set_table_styles([table_styles]).background_gradient(cmap='Greens', subset=['Piotroski (Simples)']), use_container_width=True)
            
            # Análise Técnica para o primeiro ticker selecionado como exemplo
            if not df_fundamentals.empty:
                st.header("📈 Análise Técnica Individual")
                ticker_para_analise_tecnica = df_fundamentals.index[0]
                
                with st.spinner(f"Gerando análise técnica para {ticker_para_analise_tecnica}..."):
                    hist_data_tecnica = get_technical_data(ticker_para_analise_tecnica)
                    
                    if hist_data_tecnica is not None:
                        tendencia, status_tecnico, sinais = run_technical_analysis(hist_data_tecnica)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Tendência (Ichimoku)", tendencia)
                        col2.metric("Status dos Sinais", status_tecnico)
                        with col3.expander("Ver Sinais Detalhados"): st.markdown(sinais if sinais else "Nenhum sinal técnico claro.")
                        
                        plot_technical_analysis(hist_data_tecnica, ticker_para_analise_tecnica)
                    else:
                        st.warning(f"Não foi possível gerar a análise técnica para {ticker_para_analise_tecnica}.")
        else:
            st.warning("Nenhum dos tickers selecionados possui dados fundamentalistas disponíveis.")