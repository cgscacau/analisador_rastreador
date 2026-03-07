import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from indicadores_tecnicos import (
    calcular_indicadores,
    calcular_score_compra_venda,
    calcular_metricas_risco,
    gerar_recomendacao_estrategia,
    gerar_sinais
)
from visualizacao import criar_grafico_unificado

st.set_page_config(page_title="Analisador de Ações", layout="wide")

# ============================================================================
# CACHE DE DADOS
# ============================================================================
@st.cache_data(ttl=3600)
def obter_dados_yahoo(ticker, periodo, intervalo):
    stock = yf.Ticker(ticker)
    df = stock.history(period=periodo, interval=intervalo)
    return df

@st.cache_data
def processar_indicadores(df):
    return calcular_indicadores(df.copy())

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
        
        analisar = st.form_submit_button("🔍 Analisar", type="primary", use_container_width=True)

# Conteúdo principal
if analisar:
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
                tab1, tab2 = st.tabs(["📊 Dashboard e Gráficos", "📋 Dados Detalhados"])
                
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
                
    except Exception as e:
        st.error(f"❌ Erro ao processar: {str(e)}")
        st.info("💡 Dica: Verifique se o ticker está correto e tente novamente.")
else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em 'Analisar'")
