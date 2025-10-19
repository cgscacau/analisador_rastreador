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


def calcular_score_compra_venda(df):
    """Calcula um score de compra/venda baseado em múltiplos indicadores"""
    score = 0
    detalhes = []
    
    ultima_linha = df.iloc[-1]
    penultima_linha = df.iloc[-2] if len(df) > 1 else None
    
    # RSI (peso: 2)
    if not pd.isna(ultima_linha['RSI']):
        if ultima_linha['RSI'] < 30:
            score += 2
            detalhes.append(("RSI Sobrevenda", 2, "Bullish"))
        elif ultima_linha['RSI'] < 40:
            score += 1
            detalhes.append(("RSI Baixo", 1, "Bullish"))
        elif ultima_linha['RSI'] > 70:
            score -= 2
            detalhes.append(("RSI Sobrecompra", -2, "Bearish"))
        elif ultima_linha['RSI'] > 60:
            score -= 1
            detalhes.append(("RSI Alto", -1, "Bearish"))
    
    # MACD (peso: 2)
    if penultima_linha is not None and not pd.isna(ultima_linha['MACD']):
        if (penultima_linha['MACD'] < penultima_linha['MACD_signal'] and 
            ultima_linha['MACD'] > ultima_linha['MACD_signal']):
            score += 2
            detalhes.append(("MACD Cruzamento Alta", 2, "Bullish"))
        elif (penultima_linha['MACD'] > penultima_linha['MACD_signal'] and 
              ultima_linha['MACD'] < ultima_linha['MACD_signal']):
            score -= 2
            detalhes.append(("MACD Cruzamento Baixa", -2, "Bearish"))
        elif ultima_linha['MACD'] > ultima_linha['MACD_signal']:
            score += 0.5
            detalhes.append(("MACD Positivo", 0.5, "Bullish"))
        else:
            score -= 0.5
            detalhes.append(("MACD Negativo", -0.5, "Bearish"))
    
    # Bandas de Bollinger (peso: 1.5)
    if not pd.isna(ultima_linha['BB_lower']):
        bb_position = (ultima_linha['Close'] - ultima_linha['BB_lower']) / (ultima_linha['BB_upper'] - ultima_linha['BB_lower'])
        if bb_position < 0.2:
            score += 1.5
            detalhes.append(("Preço na Banda Inferior", 1.5, "Bullish"))
        elif bb_position > 0.8:
            score -= 1.5
            detalhes.append(("Preço na Banda Superior", -1.5, "Bearish"))
    
    # Médias Móveis (peso: 2)
    if not pd.isna(ultima_linha['SMA_20']) and not pd.isna(ultima_linha['SMA_50']):
        if ultima_linha['SMA_20'] > ultima_linha['SMA_50']:
            score += 1
            detalhes.append(("SMA20 > SMA50", 1, "Bullish"))
        else:
            score -= 1
            detalhes.append(("SMA20 < SMA50", -1, "Bearish"))
        
        if ultima_linha['Close'] > ultima_linha['SMA_20']:
            score += 0.5
            detalhes.append(("Preço > SMA20", 0.5, "Bullish"))
        else:
            score -= 0.5
            detalhes.append(("Preço < SMA20", -0.5, "Bearish"))
        
        if penultima_linha is not None:
            if (penultima_linha['SMA_20'] < penultima_linha['SMA_50'] and 
                ultima_linha['SMA_20'] > ultima_linha['SMA_50']):
                score += 1.5
                detalhes.append(("Golden Cross", 1.5, "Bullish"))
            elif (penultima_linha['SMA_20'] > penultima_linha['SMA_50'] and 
                  ultima_linha['SMA_20'] < ultima_linha['SMA_50']):
                score -= 1.5
                detalhes.append(("Death Cross", -1.5, "Bearish"))
    
    # Estocástico (peso: 1.5)
    if not pd.isna(ultima_linha['STOCH_k']):
        if ultima_linha['STOCH_k'] < 20:
            score += 1.5
            detalhes.append(("Estocástico Sobrevenda", 1.5, "Bullish"))
        elif ultima_linha['STOCH_k'] > 80:
            score -= 1.5
            detalhes.append(("Estocástico Sobrecompra", -1.5, "Bearish"))
    
    # Momentum (baseado em variação de preço)
    if penultima_linha is not None:
        variacao_5d = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100) if len(df) >= 5 else 0
        if variacao_5d > 5:
            score += 1
            detalhes.append(("Momentum Positivo (5d)", 1, "Bullish"))
        elif variacao_5d < -5:
            score -= 1
            detalhes.append(("Momentum Negativo (5d)", -1, "Bearish"))
    
    return score, detalhes

def calcular_metricas_risco(df):
    """Calcula métricas de risco e retorno"""
    returns = df['Close'].pct_change().dropna()
    
    # Retorno esperado (anualizado)
    retorno_medio_diario = returns.mean()
    retorno_anual = retorno_medio_diario * 252 * 100
    
    # Volatilidade (anualizada)
    volatilidade_diaria = returns.std()
    volatilidade_anual = volatilidade_diaria * (252 ** 0.5) * 100
    
    # Sharpe Ratio (assumindo taxa livre de risco de 10% ao ano)
    taxa_livre_risco = 0.10
    sharpe_ratio = (retorno_anual / 100 - taxa_livre_risco) / (volatilidade_anual / 100) if volatilidade_anual != 0 else 0
    
    # Drawdown máximo
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # VaR (Value at Risk) - 95% de confiança
    var_95 = returns.quantile(0.05) * 100
    
    # Classificação de risco
    if volatilidade_anual < 20:
        nivel_risco = "Baixo"
        cor_risco = "🟢"
    elif volatilidade_anual < 35:
        nivel_risco = "Moderado"
        cor_risco = "🟡"
    else:
        nivel_risco = "Alto"
        cor_risco = "🔴"
    
    return {
        'retorno_anual': retorno_anual,
        'volatilidade_anual': volatilidade_anual,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'nivel_risco': nivel_risco,
        'cor_risco': cor_risco
    }

def gerar_recomendacao_estrategia(score, metricas_risco, df):
    """Gera recomendação de estratégia baseada em score e risco"""
    ultima_linha = df.iloc[-1]
    
    # Determinar recomendação principal
    if score >= 5:
        recomendacao = "COMPRA FORTE"
        emoji = "🟢🟢"
        confianca = "Alta"
    elif score >= 2:
        recomendacao = "COMPRA"
        emoji = "🟢"
        confianca = "Moderada"
    elif score >= -2:
        recomendacao = "NEUTRO / AGUARDAR"
        emoji = "⚪"
        confianca = "Baixa"
    elif score >= -5:
        recomendacao = "VENDA"
        emoji = "🔴"
        confianca = "Moderada"
    else:
        recomendacao = "VENDA FORTE"
        emoji = "🔴🔴"
        confianca = "Alta"
    
    # Estratégia baseada em risco-retorno
    if metricas_risco['nivel_risco'] == "Baixo":
        if score > 0:
            estrategia = "**Estratégia Conservadora:** Posição de longo prazo com baixa volatilidade. Ideal para investidores avessos ao risco."
            alocacao = "Alocação sugerida: 60-80% do capital disponível para este ativo"
        else:
            estrategia = "**Estratégia Conservadora:** Manter distância ou aguardar melhores pontos de entrada. Ativo de baixo risco mas sem sinais positivos."
            alocacao = "Alocação sugerida: 0-20% do capital disponível"
    
    elif metricas_risco['nivel_risco'] == "Moderado":
        if score > 2:
            estrategia = "**Estratégia Balanceada:** Boa oportunidade com risco controlado. Considere entrada gradual com stop loss."
            alocacao = "Alocação sugerida: 40-60% do capital disponível"
        elif score < -2:
            estrategia = "**Estratégia Balanceada:** Sinais negativos com volatilidade moderada. Considere reduzir exposição ou realizar lucros."
            alocacao = "Alocação sugerida: 0-30% do capital disponível"
        else:
            estrategia = "**Estratégia Balanceada:** Momento indefinido. Aguarde sinais mais claros antes de tomar posição."
            alocacao = "Alocação sugerida: 20-40% do capital disponível (apenas para quem já está posicionado)"
    
    else:  # Alto risco
        if score > 3:
            estrategia = "**Estratégia Agressiva:** Alta volatilidade com sinais positivos. Oportunidade para traders experientes com gestão de risco rigorosa."
            alocacao = "Alocação sugerida: 20-40% do capital disponível (apenas para perfil agressivo)"
        elif score < -3:
            estrategia = "**Estratégia Agressiva:** Alta volatilidade com sinais negativos. Considere posições vendidas (short) ou evite o ativo."
            alocacao = "Alocação sugerida: 0-10% do capital disponível"
        else:
            estrategia = "**Estratégia Agressiva:** Alta volatilidade sem direção clara. Extremamente arriscado para novas posições."
            alocacao = "Alocação sugerida: 0-20% do capital disponível (somente para traders experientes)"
    
    # Níveis de stop loss e take profit
    atr = ultima_linha['ATR'] if not pd.isna(ultima_linha['ATR']) else ultima_linha['Close'] * 0.02
    
    if score > 0:  # Cenário de compra
        stop_loss = ultima_linha['Close'] - (2 * atr)
        take_profit_1 = ultima_linha['Close'] + (2 * atr)
        take_profit_2 = ultima_linha['Close'] + (4 * atr)
        
        niveis = f"""
**Níveis Sugeridos para Compra:**
- **Entrada:** R$ {ultima_linha['Close']:.2f}
- **Stop Loss:** R$ {stop_loss:.2f} ({((stop_loss/ultima_linha['Close']-1)*100):.2f}%)
- **Take Profit 1:** R$ {take_profit_1:.2f} ({((take_profit_1/ultima_linha['Close']-1)*100):.2f}%)
- **Take Profit 2:** R$ {take_profit_2:.2f} ({((take_profit_2/ultima_linha['Close']-1)*100):.2f}%)
- **Relação Risco/Retorno:** 1:{abs((take_profit_1-ultima_linha['Close'])/(ultima_linha['Close']-stop_loss)):.2f}
"""
    else:  # Cenário de venda
        stop_loss = ultima_linha['Close'] + (2 * atr)
        take_profit_1 = ultima_linha['Close'] - (2 * atr)
        take_profit_2 = ultima_linha['Close'] - (4 * atr)
        
        niveis = f"""
**Níveis Sugeridos para Venda:**
- **Saída/Realização:** R$ {ultima_linha['Close']:.2f}
- **Stop Loss (se short):** R$ {stop_loss:.2f} ({((stop_loss/ultima_linha['Close']-1)*100):.2f}%)
- **Suporte 1:** R$ {take_profit_1:.2f} ({((take_profit_1/ultima_linha['Close']-1)*100):.2f}%)
- **Suporte 2:** R$ {take_profit_2:.2f} ({((take_profit_2/ultima_linha['Close']-1)*100):.2f}%)
"""
    
    return {
        'recomendacao': recomendacao,
        'emoji': emoji,
        'confianca': confianca,
        'estrategia': estrategia,
        'alocacao': alocacao,
        'niveis': niveis,
        'score': score
    }

def exibir_resumo_analitico(df):
    """Exibe o resumo analítico completo"""
    
    # Calcular score e métricas
    score, detalhes = calcular_score_compra_venda(df)
    metricas_risco = calcular_metricas_risco(df)
    recomendacao = gerar_recomendacao_estrategia(score, metricas_risco, df)
    
    st.header("📊 Resumo Analítico e Recomendação")
    
    # Card principal de recomendação
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### {recomendacao['emoji']} {recomendacao['recomendacao']}")
        st.markdown(f"**Confiança:** {recomendacao['confianca']}")
        st.markdown(f"**Score Técnico:** {score:.1f}/10")
    
    with col2:
        st.metric("Nível de Risco", f"{metricas_risco['cor_risco']} {metricas_risco['nivel_risco']}")
        st.metric("Sharpe Ratio", f"{metricas_risco['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Retorno Anual", f"{metricas_risco['retorno_anual']:.2f}%")
        st.metric("Volatilidade", f"{metricas_risco['volatilidade_anual']:.2f}%")
    
    st.markdown("---")
    
    # Estratégia detalhada
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🎯 Estratégia Recomendada")
        st.markdown(recomendacao['estrategia'])
        st.info(recomendacao['alocacao'])
        
        st.markdown("### 📍 Níveis de Operação")
        st.markdown(recomendacao['niveis'])
    
    with col2:
        st.markdown("### 📈 Métricas de Risco-Retorno")
        
        st.markdown(f"""
        **Retorno Esperado (Anual):** {metricas_risco['retorno_anual']:.2f}%
        
        **Volatilidade (Anual):** {metricas_risco['volatilidade_anual']:.2f}%
        
        **Sharpe Ratio:** {metricas_risco['sharpe_ratio']:.2f}
        {'✅ Excelente' if metricas_risco['sharpe_ratio'] > 1 else '⚠️ Moderado' if metricas_risco['sharpe_ratio'] > 0 else '❌ Ruim'}
        
        **Drawdown Máximo:** {metricas_risco['max_drawdown']:.2f}%
        
        **VaR (95%):** {metricas_risco['var_95']:.2f}%
        *Perda máxima esperada em 95% dos dias*
        """)
        
        # Interpretação do Sharpe Ratio
        st.markdown("---")
        st.markdown("**Interpretação do Sharpe:**")
        if metricas_risco['sharpe_ratio'] > 2:
            st.success("Excelente relação risco-retorno")
        elif metricas_risco['sharpe_ratio'] > 1:
            st.success("Boa relação risco-retorno")
        elif metricas_risco['sharpe_ratio'] > 0:
            st.warning("Relação risco-retorno moderada")
        else:
            st.error("Relação risco-retorno desfavorável")
    
    st.markdown("---")
    
    # Detalhamento dos indicadores
    with st.expander("🔍 Detalhamento dos Indicadores Técnicos"):
        st.markdown("### Contribuição de cada indicador para o score:")
        
        df_detalhes = pd.DataFrame(detalhes, columns=['Indicador', 'Pontos', 'Tendência'])
        df_detalhes = df_detalhes.sort_values('Pontos', ascending=False)
        
        # Colorir baseado na tendência
        def colorir_linha(row):
            if row['Tendência'] == 'Bullish':
                return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
            else:
                return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
        
        st.dataframe(
            df_detalhes.style.apply(colorir_linha, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown(f"""
        **Score Total:** {score:.1f} pontos
        
        **Interpretação:**
        - Score > 5: Forte sinal de compra
        - Score 2-5: Sinal de compra moderado
        - Score -2 a 2: Neutro, aguardar
        - Score -5 a -2: Sinal de venda moderado
        - Score < -5: Forte sinal de venda
        """)



