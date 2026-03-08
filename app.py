import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from indicadores_tecnicos import (
    calcular_indicadores,
    calcular_score_compra_venda,
    calcular_metricas_risco,
    gerar_recomendacao_estrategia,
    gerar_sinais,
    simular_retorno_media,
    otimizar_retorno_media
)
from visualizacao import criar_grafico_unificado, criar_grafico_backtest, criar_grafico_operacao_atual

st.set_page_config(page_title="Analisador de Ações", layout="wide")

# ============================================================================
# CACHE DE DADOS
# ============================================================================
@st.cache_data(ttl=3600)
def obter_dados_yahoo(ticker, periodo, intervalo):
    stock = yf.Ticker(ticker)
    df = stock.history(period=periodo, interval=intervalo)
    return df

def processar_indicadores(df):
    return calcular_indicadores(df.copy())

def executar_backtest(df, p_mm, p_dev, p_tp, p_sl):
    return simular_retorno_media(df.copy(), p_mm, p_dev, p_tp, p_sl)

def executar_otimizacao(df):
    return otimizar_retorno_media(df.copy())

# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO INTERNAS (Resumo)
# ============================================================================
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


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

st.title("📈 Analisador Técnico de Ações")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    with st.form(key="config_form"):
        ticker_input = st.text_input("Ticker da Ação", value=st.session_state.get('ticker', 'PETR4.SA'), help="Ex: PETR4.SA, VALE3.SA, ITUB4.SA")
        
        _periodo_opts = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
        _periodo_idx = _periodo_opts.index(st.session_state.get('periodo', '6mo')) if st.session_state.get('periodo', '6mo') in _periodo_opts else 2
        periodo_input = st.selectbox(
            "Período de Análise",
            options=_periodo_opts,
            index=_periodo_idx
        )
        
        _intervalo_opts = ["1d", "1wk", "1mo"]
        _intervalo_idx = _intervalo_opts.index(st.session_state.get('intervalo', '1d')) if st.session_state.get('intervalo', '1d') in _intervalo_opts else 0
        intervalo_input = st.selectbox(
            "Intervalo",
            options=_intervalo_opts,
            index=_intervalo_idx
        )
        
        analisar = st.form_submit_button("🔍 Analisar", type="primary", use_container_width=True)

if analisar:
    st.session_state.analisar_clicado = True
    st.session_state.ticker = ticker_input
    st.session_state.periodo = periodo_input
    st.session_state.intervalo = intervalo_input

ticker = st.session_state.get('ticker', 'PETR4.SA')
periodo = st.session_state.get('periodo', '6mo')
intervalo = st.session_state.get('intervalo', '1d')


# Conteúdo principal
if st.session_state.get('analisar_clicado', False):
    try:
        with st.spinner(f"Carregando dados de {ticker}..."):
            # Baixar dados
            df = obter_dados_yahoo(ticker, periodo, intervalo)
            
            if df.empty:
                st.error("❌ Não foi possível carregar os dados. Verifique o ticker.")
            else:
                # Calcular indicadores
                df = processar_indicadores(df)
                
                # Instanciar as abas
                tab1, tab2, tab3 = st.tabs(["📊 Dashboard e Gráficos", "📋 Dados Detalhados", "🔁 Otimização de Retorno à Média"])
                
                with tab1:
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
                    
                    # RESUMO ANALÍTICO
                    exibir_resumo_analitico(df)
                    
                    st.markdown("---")
                    
                    # Sinais de Trading
                    st.subheader("🎯 Sinais de Trading Rápidos")
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
                    st.subheader("📊 Gráficos Completos")
                    st.plotly_chart(criar_grafico_unificado(df, ticker), use_container_width=True)
                
                with tab2:
                    st.subheader("📋 Tabela de Histórico e Indicadores")
                    st.dataframe(df.tail(100).iloc[::-1], use_container_width=True)
                
                with tab3:
                    st.subheader("⚙️ Otimização de Retorno à Média (Regressão Linear)")
                    st.markdown("Testa a estratégia de comprar na Banda Inferior de Regressão Linear e vender no alvo, stop ou na Linha Central.")
                    
                    if 'p_mm' not in st.session_state: st.session_state.p_mm = 20
                    if 'p_dev' not in st.session_state: st.session_state.p_dev = 2.0
                    if 'p_tp' not in st.session_state: st.session_state.p_tp = 0.05
                    if 'p_sl' not in st.session_state: st.session_state.p_sl = 0.03
                    if 'otimizando' not in st.session_state: st.session_state.otimizando = False
                    
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        b_mm = st.number_input("Período da Regressão (Canal)", min_value=10, max_value=100, value=int(st.session_state.p_mm), step=5)
                        b_dev = st.number_input("Desvios para Entrada (Linhas LR)", min_value=1.0, max_value=4.0, value=float(st.session_state.p_dev), step=0.5)
                    with col_p2:
                        b_tp = st.number_input("Take Profit %", min_value=0.01, max_value=0.20, value=float(st.session_state.p_tp), step=0.01)
                        b_sl = st.number_input("Stop Loss %", min_value=0.01, max_value=0.20, value=float(st.session_state.p_sl), step=0.01)
                    
                    if st.button("✨ Otimizar Melhores Parâmetros", type="primary"):
                        barra = st.progress(0, text="🔍 Iniciando varredura de parâmetros...")
                        
                        from itertools import product as _product
                        periodos_mm = [10, 20, 30, 40, 50, 60, 90]
                        desvios = [1.5, 2.0, 2.5, 3.0]
                        take_profits = [0.03, 0.05, 0.08, 0.10, 0.15]
                        stop_losses = [0.02, 0.03, 0.05, 0.08]
                        
                        combos = [(p, d, tp, sl) for p, d, tp, sl in _product(periodos_mm, desvios, take_profits, stop_losses) if sl < tp]
                        total = len(combos)
                        
                        melhor_capital = 0
                        melhores_p = None
                        
                        for i, (p_mm, p_dev, p_tp, p_sl) in enumerate(combos):
                            if i % 10 == 0:
                                pct = int((i / total) * 100)
                                barra.progress(pct, text=f"🔍 Testando combinação {i}/{total}... ({pct}%)")
                            _, stats_c = simular_retorno_media(df, p_mm, p_dev, p_tp, p_sl)
                            if stats_c['capital_final'] > melhor_capital:
                                melhor_capital = stats_c['capital_final']
                                melhores_p = {'periodo_mm': p_mm, 'desvios_entrada': p_dev, 'take_profit_pct': p_tp, 'stop_loss_pct': p_sl}
                        
                        barra.progress(100, text="✅ Otimização concluída!")
                        
                        if melhores_p:
                            st.session_state.p_mm = melhores_p['periodo_mm']
                            st.session_state.p_dev = melhores_p['desvios_entrada']
                            st.session_state.p_tp = melhores_p['take_profit_pct']
                            st.session_state.p_sl = melhores_p['stop_loss_pct']
                            st.success(f"✅ Melhores parâmetros aplicados! Período={melhores_p['periodo_mm']} barras, Desvio={melhores_p['desvios_entrada']}, TP={melhores_p['take_profit_pct']*100:.0f}%, SL={melhores_p['stop_loss_pct']*100:.0f}%")
                            st.rerun()

                    df_bt, stats = executar_backtest(df, b_mm, b_dev, b_tp, b_sl)
                    
                    st.markdown("---")
                    st.plotly_chart(criar_grafico_backtest(df_bt, ticker), use_container_width=True)
                    
                    rm_c1, rm_c2, rm_c3, rm_c4 = st.columns(4)
                    rm_c1.metric("Retorno Estratégia", f"{stats['retorno_pct']:.2f}%")
                    rm_c2.metric("Trades Realizados", stats['trades_realizados'])
                    rm_c3.metric("Probabilidade de Acerto (Win Rate)", f"{stats['win_rate']:.1f}%")
                    
                    retorno_bh = ((df_bt['Buy_and_Hold'].iloc[-1] - 1000) / 1000) * 100
                    rm_c4.metric("Comparação (Buy&Hold)", f"{retorno_bh:.2f}%")
                    
                    # ==== OPORTUNIDADE ATUAL ====
                    st.markdown("---")
                    st.subheader("🔎 Avaliação do Trade Mapeado Atualmente")
                    
                    ultimo_fechamento = df_bt['Close'].iloc[-1]
                    entrada_lr = df_bt['Banda_Inferior'].iloc[-1]
                    alvo_media = df_bt['LRL'].iloc[-1]
                    stop_estimado = entrada_lr * (1 - b_sl)
                    alvo_estimado = entrada_lr * (1 + b_tp)
                    
                    distancia_entrada = ((ultimo_fechamento - entrada_lr) / entrada_lr) * 100
                    
                    st.markdown("---")
                    
                    if ultimo_fechamento <= entrada_lr * 1.01: # margem de 1%
                        st.success("🟢 **SINAL DE ENTRADA ATIVO!**")
                        st.markdown(f"O preço de `{ultimo_fechamento:.2f}` está testando a Banda Inferior Otimizada. É hora do Trade!")
                    elif distancia_entrada < 5.0:
                        st.warning("🟡 **PROXIMO À ZONA DE COMPRA**")
                        st.markdown(f"Faltam apenas `{distancia_entrada:.2f}%` de queda para encostar na linha ideal.")
                    else:
                        st.info("⚪ **FORA DA ZONA DE TRADE**")
                        st.markdown(f"O preço está `{distancia_entrada:.2f}%` seguro acima da banda de entrada. Aguardar.")

                    st.markdown(f"""
                    **🎯 Plano de Voo (Se comprar na Linha Otimizada agora):**
                    - ⬇️ **Entrada Ideal:** R$ {entrada_lr:.2f}
                    - 🎯 **Alvo Take Profit (+{b_tp*100:.0f}%):** R$ {alvo_estimado:.2f} *(Ou R$ {alvo_media:.2f} se encostar na Média Central)*
                    - 🛑 **Stop Loss Proteção (-{b_sl*100:.0f}%):** R$ {stop_estimado:.2f}
                    - 📊 **Probabilidade Histórica:** **{stats['win_rate']:.1f}%** de chance da ação bater no alvo antes do stop (Baseado nos {stats['trades_realizados']} trades simulados)
                    """)
                    
                    st.plotly_chart(criar_grafico_operacao_atual(df_bt, ticker, period_bars=int(b_mm)+40, window_lr=int(b_mm), desvios_lr=float(b_dev)), use_container_width=True)
                
    except Exception as e:
        st.error(f"❌ Erro ao processar: {str(e)}")
        st.info("💡 Dica: Verifique se o ticker está correto e tente novamente.")
else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em 'Analisar'")
