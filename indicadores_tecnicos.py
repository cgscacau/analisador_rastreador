import pandas as pd
import numpy as np
import ta
from itertools import product

def calcular_indicadores(df, **kwargs):
    """Calcula indicadores técnicos. Todos os períodos são configuravéis via kwargs."""
    
    rsi_period   = kwargs.get('rsi_period', 14)
    macd_fast    = kwargs.get('macd_fast', 12)
    macd_slow    = kwargs.get('macd_slow', 26)
    macd_signal  = kwargs.get('macd_signal', 9)
    stoch_window = kwargs.get('stoch_window', 14)
    lr_window    = kwargs.get('lr_window', 20)
    lr_dev       = kwargs.get('lr_dev', 2)
    sma_curta    = kwargs.get('sma_curta', 20)
    sma_longa    = kwargs.get('sma_longa', 50)
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_period).rsi()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    # Canal de Regressão Linear
    def _calc_lr(series):
        if len(series) < lr_window: return np.nan
        x = np.arange(lr_window)
        y = series.values
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * (lr_window - 1)

    df['LR_middle'] = df['Close'].rolling(window=lr_window).apply(_calc_lr, raw=False)
    std_lr = df['Close'].rolling(window=lr_window).std()
    df['LR_upper'] = df['LR_middle'] + (lr_dev * std_lr)
    df['LR_lower'] = df['LR_middle'] - (lr_dev * std_lr)
    
    # Médias Móveis
    df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=sma_curta).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=sma_longa).sma_indicator()
    df['EMA_12'] = ta.trend.EMAIndicator(close=df['Close'], window=macd_fast).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=df['Close'], window=macd_slow).ema_indicator()
    
    # ATR (Average True Range)
    df['ATR'] = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    ).average_true_range()
    
    # Estocástico
    stoch = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=stoch_window, smooth_window=3
    )
    df['STOCH_k'] = stoch.stoch()
    df['STOCH_d'] = stoch.stoch_signal()
    
    return df


def otimizar_sma(df, callback_progresso=None):
    """Versão simplificada legada — chama otimizar_todos_indicadores internamente."""
    resultado = otimizar_todos_indicadores(df, callback_progresso=callback_progresso)
    return {
        'sma_curta': resultado['sma_curta'],
        'sma_longa': resultado['sma_longa'],
        'retorno_pct': resultado['retorno_pct'],
        'n_trades': resultado['n_trades'],
    }


def otimizar_todos_indicadores(df, callback_progresso=None, passo=2):
    """
    Otimização conjunta de todos os períodos dos indicadores técnicos.
    Testa combinações de RSI, SMA, MACD e Estocástico usando uma estratégia
    de entrada/saída baseada em score composto calculado inline.
    Retorna o conjunto de parâmetros com maior retorno histórico.
    """
    # Espaços de busca de cada indicador com passo (step) dinâmico
    rsi_periods       = list(range(10, 22, passo))
    rsi_buy_thresh    = list(range(26, 36, passo))
    sma_curtos        = list(range(6, 22, passo))
    # Para SMA longa, multiplicamos o passo para varrer um espectro maior sem explodir permutações
    sma_longos        = list(range(40, 121, max(passo * 5, 10))) 
    macd_fasts        = list(range(8, 16, passo))
    macd_slows        = list(range(20, 30, passo))
    stoch_windows     = list(range(10, 18, passo))
    lr_windows        = list(range(16, 32, passo))

    combos = []
    for rp, rb, sc, sl, mf, ms, sw, lw in product(
        rsi_periods, rsi_buy_thresh, sma_curtos, sma_longos,
        macd_fasts, macd_slows, stoch_windows, lr_windows
    ):
        if sc < sl and mf < ms:
            combos.append((rp, rb, sc, sl, mf, ms, sw, lw))

    total = len(combos)
    melhor_retorno = -999.0
    melhores_params = None
    melhor_n_trades = 0

    for i, (rp, rb, sc, sl, mf, ms, sw, lw) in enumerate(combos):
        if callback_progresso and i % 20 == 0:
            pct = int((i / total) * 100)
            callback_progresso(pct, f"🔍 Combo {i}/{total} — RSI({rp}/{rb}) SMA({sc}/{sl}) MACD({mf}/{ms}) Stoch({sw}) LR({lw})")

        try:
            # Calcula SMAs e RSI inline (mais rápido que calcular_indicadores completo)
            sma_c = df['Close'].rolling(sc).mean()
            sma_l = df['Close'].rolling(sl).mean()
            rsi_s = ta.momentum.RSIIndicator(close=df['Close'], window=rp).rsi()
            stoch_s = ta.momentum.StochasticOscillator(
                high=df['High'], low=df['Low'], close=df['Close'],
                window=sw, smooth_window=3
            ).stoch()
            macd_o = ta.trend.MACD(close=df['Close'], window_fast=mf, window_slow=ms, window_sign=9)
            macd_line = macd_o.macd()
            macd_sig  = macd_o.macd_signal()

            capital = 1000.0
            posicionado = False
            preco_entrada = 0.0
            n_trades = 0
            n_wins = 0

            for j in range(ms + 1, len(df)):
                score = 0
                # RSI
                if not pd.isna(rsi_s.iloc[j]):
                    if rsi_s.iloc[j] < rb:        score += 2
                    elif rsi_s.iloc[j] > (100 - rb): score -= 2
                # SMA
                if not pd.isna(sma_c.iloc[j]) and not pd.isna(sma_l.iloc[j]):
                    score += 1 if sma_c.iloc[j] > sma_l.iloc[j] else -1
                    score += 0.5 if df['Close'].iloc[j] > sma_c.iloc[j] else -0.5
                # MACD
                if not pd.isna(macd_line.iloc[j]) and not pd.isna(macd_sig.iloc[j]):
                    score += 1 if macd_line.iloc[j] > macd_sig.iloc[j] else -1
                # Stoch
                if not pd.isna(stoch_s.iloc[j]):
                    if stoch_s.iloc[j] < 20: score += 1.5
                    elif stoch_s.iloc[j] > 80: score -= 1.5

                if not posicionado and score >= 3:
                    posicionado = True
                    preco_entrada = df['Close'].iloc[j]
                elif posicionado and score <= -2:
                    ret = (df['Close'].iloc[j] - preco_entrada) / preco_entrada
                    capital *= (1 + ret)
                    if ret > 0: n_wins += 1
                    posicionado = False
                    n_trades += 1

            retorno_final = ((capital - 1000) / 1000) * 100
            if retorno_final > melhor_retorno and n_trades >= 2:
                melhor_retorno = retorno_final
                melhor_n_trades = n_trades
                melhores_params = {
                    'rsi_period': rp, 'rsi_buy_thresh': rb,
                    'sma_curta': sc, 'sma_longa': sl,
                    'macd_fast': mf, 'macd_slow': ms,
                    'stoch_window': sw, 'lr_window': lw,
                    'retorno_pct': retorno_final,
                    'n_trades': n_trades,
                    'win_rate': round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0
                }
        except Exception:
            continue

    if callback_progresso:
        callback_progresso(100, "✅ Otimização completa de indicadores concluída!")

    if melhores_params is None:
        melhores_params = {
            'rsi_period': 14, 'rsi_buy_thresh': 30,
            'sma_curta': 20, 'sma_longa': 50,
            'macd_fast': 12, 'macd_slow': 26,
            'stoch_window': 14, 'lr_window': 20,
            'retorno_pct': 0.0, 'n_trades': 0, 'win_rate': 0.0
        }

    return melhores_params


# ============================================================================
# FUNÇÕES DE ANÁLISE E SCORING
# ============================================================================

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
    
    # Canal de Regressão Linear (peso: 1.5)
    if 'LR_lower' in df.columns and not pd.isna(ultima_linha['LR_lower']):
        lr_position = (ultima_linha['Close'] - ultima_linha['LR_lower']) / (ultima_linha['LR_upper'] - ultima_linha['LR_lower'])
        if lr_position < 0.2:
            score += 1.5
            detalhes.append(("Preço na Banda Inferior (LR)", 1.5, "Bullish"))
        elif lr_position > 0.8:
            score -= 1.5
            detalhes.append(("Preço na Banda Superior (LR)", -1.5, "Bearish"))
    
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
    if len(df) >= 5:
        variacao_5d = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100)
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
    
    # Sinal Canal de Regressão Linear
    if 'LR_lower' in df.columns and not pd.isna(ultima_linha['LR_lower']):
        if ultima_linha['Close'] < ultima_linha['LR_lower']:
            sinais.append(("🟢 COMPRA", "Preço abaixo da banda inferior de Regressão"))
        elif ultima_linha['Close'] > ultima_linha['LR_upper']:
            sinais.append(("🔴 VENDA", "Preço acima da banda superior de Regressão"))
    
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


import numpy as np
from itertools import product

# ============================================================================
# BACKTESTING RETORNO À MÉDIA
# ============================================================================

def simular_retorno_media(df, periodo_mm=20, desvios_entrada=2.0, take_profit_pct=0.05, stop_loss_pct=0.03):
    """
    Simula uma estratégia de retorno à média comprando na banda inferior das
    Linhas de Regressão Linear e vendendo no alvo (TP), stop (SL) ou no retorno à regressão média.
    Retorna o dataframe com os resultados (curva de capital) e estatísticas.
    """
    df_bt = df[['Open', 'High', 'Low', 'Close']].copy()
    
    window = int(periodo_mm)
    
    def calculate_bt_lr(series):
        if len(series) < window: return np.nan
        x = np.arange(window)
        y = series.values
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * (window - 1)
        
    df_bt['LRL'] = df_bt['Close'].rolling(window=window).apply(calculate_bt_lr, raw=False)
    std = df_bt['Close'].rolling(window=window).std()
    df_bt['Banda_Inferior'] = df_bt['LRL'] - (desvios_entrada * std)
    
    posicionado = False
    preco_entrada = 0.0
    capital_inicial = 1000.0
    capital_atual = capital_inicial
    
    capital_curve = np.zeros(len(df_bt))
    capital_curve[:] = capital_inicial
    
    sinais_compra = [] 
    sinais_venda = []  
    
    trades_realizados = 0
    trades_vencedores = 0
    
    for i in range(periodo_mm, len(df_bt)):
        preco_hoje = df_bt['Close'].iloc[i]
        
        if not posicionado:
            if df_bt['Close'].iloc[i-1] < df_bt['Banda_Inferior'].iloc[i-1]:
                posicionado = True
                preco_entrada = preco_hoje 
                sinais_compra.append((df_bt.index[i], preco_entrada))
        else:
            retorno_atual = (preco_hoje - preco_entrada) / preco_entrada
            
            saiu = False
            motivo_saida = ""
            
            if retorno_atual >= take_profit_pct:
                saiu = True
                motivo_saida = "TP"
            elif retorno_atual <= -stop_loss_pct:
                saiu = True
                motivo_saida = "SL"
            elif preco_hoje >= df_bt['LRL'].iloc[i]:
                saiu = True
                motivo_saida = "Media"
                
            if saiu:
                posicionado = False
                resultado_financeiro = (preco_hoje - preco_entrada) / preco_entrada
                capital_atual = capital_atual * (1 + resultado_financeiro)
                
                sinais_venda.append((df_bt.index[i], preco_hoje, motivo_saida))
                
                trades_realizados += 1
                if resultado_financeiro > 0:
                    trades_vencedores += 1
        
        if posicionado:
            retorno_aberto = (preco_hoje - preco_entrada) / preco_entrada
            capital_curve[i] = capital_atual * (1 + retorno_aberto)
        else:
            capital_curve[i] = capital_atual

    df_bt['Capital'] = capital_curve
    
    preco_inicio = df_bt['Close'].iloc[periodo_mm] if len(df_bt) > periodo_mm else df_bt['Close'].iloc[0]
    df_bt['Buy_and_Hold'] = capital_inicial * (df_bt['Close'] / preco_inicio)
    df_bt.iloc[0:periodo_mm, df_bt.columns.get_loc('Buy_and_Hold')] = capital_inicial 
    
    retorno_final = ((capital_atual - capital_inicial) / capital_inicial) * 100
    win_rate = (trades_vencedores / trades_realizados * 100) if trades_realizados > 0 else 0
    
    estatisticas = {
        'capital_final': capital_atual,
        'retorno_pct': retorno_final,
        'trades_realizados': trades_realizados,
        'win_rate': win_rate,
        'sinais_compra': sinais_compra,
        'sinais_venda': sinais_venda
    }
    
    return df_bt, estatisticas


def otimizar_retorno_media(df, max_trials=1000, passo=2):
    """
    Testa combinações de parâmetros fazendo um Grid Search (Brute Force controlada).
    """
    periodos_mm = list(range(10, 101, max(passo * 5, 5)))         # 10, 20, 30... ou 10, 15, 20...
    desvios = [x/10.0 for x in range(10, 41, max(passo, 2))]      # 1.0, 1.2, 1.4... ou 1.0, 1.5...
    take_profits = [x/100.0 for x in range(3, 22, max(passo, 1))] # 3%, 5%, 7%...
    stop_losses = [x/100.0 for x in range(2, 16, max(passo, 1))]  # 2%, 4%, 6%...
    
    todas_combinacoes = list(product(periodos_mm, desvios, take_profits, stop_losses))
    
    melhor_capital = 0
    melhores_params = None
    melhor_stats = None
    
    combinacoes_testar = todas_combinacoes[:max_trials]
    
    for comb in combinacoes_testar:
        p_mm, p_dev, p_tp, p_sl = comb
        
        if p_sl >= p_tp:
            continue
            
        _, stats = simular_retorno_media(df, p_mm, p_dev, p_tp, p_sl)
        
        if stats['capital_final'] > melhor_capital:
            melhor_capital = stats['capital_final']
            melhores_params = {
                'periodo_mm': p_mm,
                'desvios_entrada': p_dev,
                'take_profit_pct': p_tp,
                'stop_loss_pct': p_sl
            }
            melhor_stats = stats
            
    if melhores_params is None:
        melhores_params = {
            'periodo_mm': 20,
            'desvios_entrada': 2.0,
            'take_profit_pct': 0.05,
            'stop_loss_pct': 0.03
        }
        _, melhor_stats = simular_retorno_media(df, 20, 2.0, 0.05, 0.03)
        
    return melhores_params, melhor_stats
