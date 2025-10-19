import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta  # Biblioteca alternativa

st.set_page_config(page_title="Analisador de Ações", layout="wide")

def calcular_indicadores(df):
    """Calcula indicadores técnicos usando a biblioteca ta"""
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    # Bandas de Bollinger
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    # Médias Móveis
    df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_12'] = ta.trend.EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=df['Close'], window=26).ema_indicator()
    
    # ATR (Average True Range)
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], 
        low=df['Low'], 
        close=df['Close'], 
        window=14
    ).average_true_range()
    
    # Estocástico
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['STOCH_k'] = stoch.stoch()
    df['STOCH_d'] = stoch.stoch_signal()
    
    return df

def gerar_sinais(df):
    """Gera sinais de compra/venda baseados nos indicadores"""
    sinais = []
    
    ultima_linha = df.iloc[-1]
    penultima_linha = df.iloc[-2] if len(df) > 1 else None
    
    # Sinal RSI
    if not pd.isna(ultima_linha['RSI']):
        if ultima_linha['RSI'] < 30:
            sinais.append(("🟢 COMPRA", "RSI está em zona de sobrevenda (< 30)"))
        elif ultima_linha['RSI'] > 70:
            sinais.append(("🔴 VENDA", "RSI está em zona de sobrecompra (> 70)"))
    
    # Sinal MACD
    if penultima_linha is not None and not pd.isna(ultima_linha['MACD']):
        if (penultima_linha['MACD'] < penultima_linha['MACD_signal'] and 
            ultima_linha['MACD'] > ultima_linha['MACD_signal']):
            sinais.append(("🟢 COMPRA", "MACD cruzou acima da linha de sinal"))
        elif (penultima_linha['MACD'] > penultima_linha['MACD_signal'] and 
              ultima_linha['MACD'] < ultima_linha['MACD_signal']):
            sinais.append(("🔴 VENDA", "MACD cruzou abaixo da linha de sinal"))
    
    # Sinal Bandas de Bollinger
    if not pd.isna(ultima_linha['BB_lower']):
        if ultima_linha['Close'] < ultima_linha['BB_lower']:
            sinais.append(("🟢 COMPRA", "Preço abaixo da banda inferior de Bollinger"))
        elif ultima_linha['Close'] > ultima_linha['BB_upper']:
            sinais.append(("🔴 VENDA", "Preço acima da banda superior de Bollinger"))
    
    # Sinal Médias Móveis
    if penultima_linha is not None and not pd.isna(ultima_linha['SMA_20']):
        if (penultima_linha['SMA_20'] < penultima_linha['SMA_50'] and 
            ultima_linha['SMA_20'] > ultima_linha['SMA_50']):
            sinais.append(("🟢 COMPRA", "Cruzamento dourado: SMA20 cruzou acima da SMA50"))
        elif (penultima_linha['SMA_20'] > penultima_linha['SMA_50'] and 
              ultima_linha['SMA_20'] < ultima_linha['SMA_50']):
            sinais.append(("🔴 VENDA", "Cruzamento da morte: SMA20 cruzou abaixo da SMA50"))
    
    # Sinal Estocástico
    if not pd.isna(ultima_linha['STOCH_k']):
        if ultima_linha['STOCH_k'] < 20:
            sinais.append(("🟢 COMPRA", "Estocástico em zona de sobrevenda (< 20)"))
        elif ultima_linha['STOCH_k'] > 80:
            sinais.append(("🔴 VENDA", "Estocástico em zona de sobrecompra (> 80)"))
    
    return sinais

def criar_grafico_candlestick(df, ticker):
    """Cria gráfico de candlestick com indicadores"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Preço'
    ))
    
    # Bandas de Bollinger
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_upper'],
        name='BB Superior',
        line=dict(color='gray', dash='dash'),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_middle'],
        name='BB Média',
        line=dict(color='blue', dash='dash'),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_lower'],
        name='BB Inferior',
        line=dict(color='gray', dash='dash'),
        opacity=0.5,
        fill='tonexty'
    ))
    
    # Médias Móveis
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='red', width=1)
    ))
    
    fig.update_layout(
        title=f'{ticker} - Análise Técnica',
        yaxis_title='Preço (R$)',
        xaxis_title='Data',
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def criar_grafico_rsi(df):
    """Cria gráfico do RSI"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # Linhas de referência
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevenda")
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title='RSI (Relative Strength Index)',
        yaxis_title='RSI',
        xaxis_title='Data',
        template='plotly_dark',
        height=300
    )
    
    return fig

def criar_grafico_macd(df):
    """Cria gráfico do MACD"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_signal'],
        name='Sinal',
        line=dict(color='red', width=2)
    ))
    
    # Histograma
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_hist'],
        name='Histograma',
        marker_color=colors,
        opacity=0.5
    ))
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        yaxis_title='MACD',
        xaxis_title='Data',
        template='plotly_dark',
        height=300
    )
    
    return fig

# Interface Streamlit
st.title("📈 Analisador Técnico de Ações")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    ticker = st.text_input("Ticker da Ação", value="PETR4.SA", help="Ex: PETR4.SA, VALE3.SA, ITUB4.SA")
    
    periodo = st.selectbox(
        "Período de Análise",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )
    
    intervalo = st.selectbox(
        "Intervalo",
        options=["1d", "1wk", "1mo"],
        index=0
    )
    
    analisar = st.button("🔍 Analisar", type="primary", use_container_width=True)

# Conteúdo principal
if analisar:
    try:
        with st.spinner(f"Carregando dados de {ticker}..."):
            # Baixar dados
            stock = yf.Ticker(ticker)
            df = stock.history(period=periodo, interval=intervalo)
            
            if df.empty:
                st.error("❌ Não foi possível carregar os dados. Verifique o ticker.")
            else:
                # Calcular indicadores
                df = calcular_indicadores(df)
                
                # Informações básicas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Preço Atual", f"R$ {df['Close'].iloc[-1]:.2f}")
                with col2:
                    variacao = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100)
                    st.metric("Variação Diária", f"{variacao:.2f}%")
                with col3:
                    st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
                with col4:
                    rsi_value = df['RSI'].iloc[-1]
                    if not pd.isna(rsi_value):
                        st.metric("RSI", f"{rsi_value:.2f}")
                    else:
                        st.metric("RSI", "N/A")
                
                st.markdown("---")
                
                # Sinais de Trading
                st.subheader("🎯 Sinais de Trading")
                sinais = gerar_sinais(df)
                
                if sinais:
                    for sinal, descricao in sinais:
                        if "COMPRA" in sinal:
                            st.success(f"{sinal}: {descricao}")
                        else:
                            st.error(f"{sinal}: {descricao}")
                else:
                    st.info("ℹ️ Nenhum sinal forte identificado no momento.")
                
                st.markdown("---")
                
                # Gráficos
                st.subheader("📊 Gráficos")
                
                # Gráfico principal
                st.plotly_chart(criar_grafico_candlestick(df, ticker), use_container_width=True)
                
                # Gráficos de indicadores
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(criar_grafico_rsi(df), use_container_width=True)
                
                with col2:
                    st.plotly_chart(criar_grafico_macd(df), use_container_width=True)
                
                # Tabela de dados
                with st.expander("📋 Ver Dados Detalhados"):
                    st.dataframe(df.tail(20).iloc[::-1], use_container_width=True)
                
    except Exception as e:
        st.error(f"❌ Erro ao processar: {str(e)}")
        st.info("💡 Dica: Verifique se o ticker está correto e tente novamente.")
else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em 'Analisar'")
