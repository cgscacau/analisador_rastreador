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
    otimizar_sma,
    otimizar_todos_indicadores,
    otimizar_retorno_media,
    calcular_total_combos_indicadores,
    calcular_total_combos_lr
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
        ticker_input = st.text_input("Ticker da Ação", value=st.session_state.get('ticker_raw', 'PETR4'), help="Ex: PETR4, VALE3, ITUB4 (sem precisar do .SA)")
        
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
        
        passo_input = st.number_input(
            "Passo Otimização (1 a 10)", 
            min_value=1, max_value=10, 
            value=st.session_state.get('passo_opt', 2), 
            step=1,
            help="Menor passo = testes mais precisos (demora mais). Maior passo = mais rápido."
        )
        
        # Feedback visual da quantidade de passos
        total_combos = calcular_total_combos_indicadores(passo_input)
        tempo_estimado_seg = total_combos * 0.003  # aprox 3ms por check simples
        tempo_str = f"~{int(tempo_estimado_seg)} seg" if tempo_estimado_seg < 60 else f"~{int(tempo_estimado_seg // 60)} min"
        st.caption(f"🧪 Testará **{total_combos:,}** combinações ({tempo_str})".replace(',', '.'))
        
        analisar = st.form_submit_button("🔍 Analisar", type="primary", use_container_width=True)

if analisar:
    st.session_state.analisar_clicado = True
    # Normaliza o ticker: adiciona '.SA' automaticamente se for B3
    _t = ticker_input.strip().upper()
    if '.' not in _t:
        _t = _t + '.SA'
    st.session_state.ticker_raw = ticker_input.strip().upper()  # guarda o valor sem sufixo para o input
    st.session_state.ticker = _t
    st.session_state.periodo = periodo_input
    st.session_state.intervalo = intervalo_input
    st.session_state.passo_opt = passo_input

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
                # Otimização Automática Inicial
                passo_atual = st.session_state.get('passo_opt', 2)
                hash_analise = f"{ticker}_{periodo}_{intervalo}_{passo_atual}"
                
                if st.session_state.get('last_optimized_ticker') != hash_analise:
                    with st.spinner(f"✨ Primeira análise: Buscando os melhores parâmetros para {ticker}..."):
                        barra_opt = st.progress(0, text="🔍 Iniciando otimização completa de indicadores...")
                        def _atualiza(pct, msg):
                            barra_opt.progress(pct, text=msg)
                        
                        res = otimizar_todos_indicadores(df, callback_progresso=_atualiza, passo=passo_atual)
                        
                        # Salva todos os períodos otimizados no estado
                        for k in ('rsi_period', 'sma_curta', 'sma_longa', 'macd_fast', 'macd_slow', 'macd_signal', 'stoch_window', 'lr_window', 'rsi_buy_thresh'):
                            if k in res:
                                st.session_state[k] = res[k]
                                
                        st.session_state['last_optimized_ticker'] = hash_analise
                        barra_opt.empty()
                        st.toast(f"Indicadores otimizados automaticamente para {ticker}!", icon="🎯")

                # Calcular indicadores (usando períodos otimizados se disponíveis)
                df = calcular_indicadores(df,
                    sma_curta    = st.session_state.get('sma_curta', 20),
                    sma_longa    = st.session_state.get('sma_longa', 50),
                    rsi_period   = st.session_state.get('rsi_period', 14),
                    macd_fast    = st.session_state.get('macd_fast', 12),
                    macd_slow    = st.session_state.get('macd_slow', 26),
                    macd_signal  = st.session_state.get('macd_signal', 9),
                    stoch_window = st.session_state.get('stoch_window', 14),
                    lr_window    = st.session_state.get('lr_window', 20),
                )
                
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
                    
                    # ========= SCORE DE CONFIANÇA VISUAL =========
                    score, detalhes_score = calcular_score_compra_venda(df)
                    max_score = 10.5  # soma máxima possível dos pesos
                    score_pct = max(0, min(100, int(((score + max_score) / (2 * max_score)) * 100)))
                    
                    bullish_items = [(n, v) for n, v, d in detalhes_score if d == 'Bullish']
                    bearish_items = [(n, v) for n, v, d in detalhes_score if d == 'Bearish']
                    
                    if score >= 3:
                        score_label = "🟢 COMPRA / BULLISH"
                        score_color = "#00c853"
                        score_emoji = "🟢"
                    elif score <= -3:
                        score_label = "🔴 VENDA / BEARISH"
                        score_color = "#f44336"
                        score_emoji = "🔴"
                    else:
                        score_label = "🟡 NEUTRO / AGUARDAR"
                        score_color = "#ff9800"
                        score_emoji = "🟡"

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                border-radius: 12px; padding: 20px; margin-bottom: 16px;
                                border-left: 5px solid {score_color};">
                        <div style="display: flex; align-items: center; gap: 20px;">
                            <div style="font-size: 48px;">{score_emoji}</div>
                            <div>
                                <div style="color: {score_color}; font-size: 22px; font-weight: bold;">{score_label}</div>
                                <div style="color: #aaa; font-size: 14px;">Score Composto: {score:.1f} pontos &nbsp;|&nbsp; Confiança: {score_pct}%</div>
                            </div>
                            <div style="margin-left: auto; text-align: right;">
                                <div style="color: #4caf50; font-size: 13px;">Fatores Bullish: {len(bullish_items)}</div>
                                <div style="color: #f44336; font-size: 13px;">Fatores Bearish: {len(bearish_items)}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        if bullish_items:
                            st.markdown("**✅ Fatores Positivos:**")
                            for nome, valor in bullish_items:
                                st.markdown(f"&nbsp;&nbsp;`+{valor}` {nome}")
                    with sc2:
                        if bearish_items:
                            st.markdown("**❌ Fatores Negativos:**")
                            for nome, valor in bearish_items:
                                st.markdown(f"&nbsp;&nbsp;`{valor}` {nome}")
                    
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
                    
                    # ========= OTIMIZADOR COMPLETO DE INDICADORES =========
                    _params_labels = {
                        'RSI': f"RSI({st.session_state.get('rsi_period', 14)})",
                        'SMA': f"SMA({st.session_state.get('sma_curta', 20)}/{st.session_state.get('sma_longa', 50)})",
                        'MACD': f"MACD({st.session_state.get('macd_fast', 12)}/{st.session_state.get('macd_slow', 26)})",
                        'Stoch': f"Stoch({st.session_state.get('stoch_window', 14)})",
                        'LR': f"LR({st.session_state.get('lr_window', 20)})",
                    }
                    st.markdown("📐 **Parâmetros Atuais:** " + " | ".join(f"`{v}`" for v in _params_labels.values()))
                    
                    if st.button("⚡ Otimizar Todos os Indicadores", type="secondary"):
                        barra_opt = st.progress(0, text="🔍 Iniciando otimização completa de indicadores...")
                        def _atualiza(pct, msg):
                            barra_opt.progress(pct, text=msg)
                            
                        passo_atual = st.session_state.get('passo_opt', 2)
                        res = otimizar_todos_indicadores(df, callback_progresso=_atualiza, passo=passo_atual)
                        
                        # Salva todos os períodos otimizados no estado
                        for k in ('rsi_period', 'sma_curta', 'sma_longa', 'macd_fast', 'macd_slow', 'macd_signal', 'stoch_window', 'lr_window'):
                            if k in res:
                                st.session_state[k] = res[k]
                        st.success(
                            f"✅ Melhor combinação encontrada! "
                            f"RSI({res.get('rsi_period',14)}&lt;{res.get('rsi_buy_thresh',30)}) | "
                            f"SMA({res.get('sma_curta',20)}/{res.get('sma_longa',50)}) | "
                            f"MACD({res.get('macd_fast',12)}/{res.get('macd_slow',26)}) | "
                            f"Stoch({res.get('stoch_window',14)}) | LR({res.get('lr_window',20)}) "
                            f"→ Retorno: {res.get('retorno_pct',0):.1f}% | Win Rate: {res.get('win_rate',0):.0f}% | {res.get('n_trades',0)} trades"
                        )
                        st.rerun()
                    
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
                    if 'passo_lr' not in st.session_state: st.session_state.passo_lr = 2
                    if 'otimizando' not in st.session_state: st.session_state.otimizando = False
                    
                    st.markdown("##### Configuração Manual")
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1:
                        b_mm = st.number_input("Período da Regressão (Canal)", min_value=10, max_value=100, value=int(st.session_state.p_mm), step=5)
                        b_dev = st.number_input("Desvios para Entrada", min_value=1.0, max_value=4.0, value=float(st.session_state.p_dev), step=0.5)
                    with col_p2:
                        b_tp = st.number_input("Take Profit %", min_value=0.01, max_value=0.20, value=float(st.session_state.p_tp), step=0.01)
                        b_sl = st.number_input("Stop Loss %", min_value=0.01, max_value=0.20, value=float(st.session_state.p_sl), step=0.01)
                    with col_p3:
                        passo_lr_input = st.number_input("Passo do Otimizador (1-10)", min_value=1, max_value=10, value=int(st.session_state.passo_lr), step=1, help="Define o tamanho do salto testado para encontrar o Pote de Ouro.")
                    
                    st.session_state.passo_lr = passo_lr_input
                    
                    total_combos_lr = calcular_total_combos_lr(passo_lr_input)
                    tempo_lr_seg = total_combos_lr * 0.001 # Aprox 1ms por simulação simples
                    tempo_lr_str = f"~{int(tempo_lr_seg)} seg" if tempo_lr_seg < 60 else f"~{int(tempo_lr_seg // 60)} min"
                    st.caption(f"🧪 O otimizador varrerá **{total_combos_lr:,}** combinações (Tempo estimado: {tempo_lr_str})".replace(',', '.'))
                    
                    if st.button("✨ Otimizar Melhores Parâmetros", type="primary"):
                        barra = st.progress(0, text="🔍 Iniciando varredura de parâmetros...")
                        
                        from itertools import product as _product
                        passo = int(st.session_state.get('passo_lr', 2))
                        
                        periodos_mm = list(range(10, 101, max(passo * 5, 5)))         # 10, 20, 30... ou 10, 15, 20...
                        desvios = [x/10.0 for x in range(10, 41, max(passo, 2))]      # 1.0, 1.2, 1.4... ou 1.0, 1.5...
                        take_profits = [x/100.0 for x in range(3, 22, max(passo, 1))] # 3%, 5%, 7%...
                        stop_losses = [x/100.0 for x in range(2, 16, max(passo, 1))]  # 2%, 4%, 6%...
                        
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
