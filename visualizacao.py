import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def criar_grafico_unificado(df, ticker):
    """Cria gráfico unificado com subplots (Candlestick, Volume, RSI e MACD)"""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=('Preço', 'Volume', 'RSI', 'MACD'))
    
    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Preço',
        showlegend=False
    ), row=1, col=1)
    
    # Canal de Regressão Linear
    fig.add_trace(go.Scatter(
        x=df.index, y=df['LR_upper'],
        name='LR Superior',
        line=dict(color='gray', dash='dash'),
        opacity=0.5
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['LR_middle'],
        name='LR Média',
        line=dict(color='blue', dash='dash'),
        opacity=0.5
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['LR_lower'],
        name='LR Inferior',
        line=dict(color='gray', dash='dash'),
        opacity=0.5,
        fill='tonexty'
    ), row=1, col=1)
    
    # Médias Móveis
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='red', width=1)
    ), row=1, col=1)
    
    # 2. Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    # 3. RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple', width=2),
        showlegend=False
    ), row=3, col=1)
    # Linhas de referência do RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # 4. MACD
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue', width=2),
        showlegend=False
    ), row=4, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_signal'],
        name='Sinal MACD',
        line=dict(color='red', width=2),
        showlegend=False
    ), row=4, col=1)
    
    # Histograma MACD
    macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_hist'],
        name='Hist. MACD',
        marker_color=macd_colors,
        opacity=0.5,
        showlegend=False
    ), row=4, col=1)
    
    fig.update_layout(
        title=f'{ticker} - Análise Técnica',
        template='plotly_dark',
        height=800,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig


def criar_grafico_backtest(df_bt, ticker):
    """Cria o gráfico de Evolução do Capital vs Buy & Hold"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_bt.index,
        y=df_bt['Capital'],
        name='Estratégia Retorno à Média',
        line=dict(color='#00ffcc', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_bt.index,
        y=df_bt['Buy_and_Hold'],
        name='Buy & Hold (Referência)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{ticker} - Evolução do Capital (Backtest)',
        yaxis_title='Capital (R$)',
        xaxis_title='Data',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def criar_grafico_operacao_atual(df_bt, ticker, period_bars=90, window_lr=None, desvios_lr=None):
    """Cria um gráfico focado no momento atual com o canal de Regressão Linear reto"""
    # Recorta as últimas barras do dataframe
    df_zoom = df_bt.tail(period_bars).copy()
    
    fig = go.Figure()

    # Preço
    fig.add_trace(go.Candlestick(
        x=df_zoom.index,
        open=df_zoom['Open'],
        high=df_zoom['High'],
        low=df_zoom['Low'],
        close=df_zoom['Close'],
        name='Preço',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    # Calcula Linhas Curvas por padrão ou Linhas Retas se parâmetros forem passados
    if window_lr is not None and desvios_lr is not None and len(df_bt) >= window_lr:
        df_lr = df_bt.tail(window_lr)
        x_vals = np.arange(window_lr)
        y_vals = df_lr['Close'].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        
        lr_line = intercept + slope * x_vals
        std_dev = df_lr['Close'].std()
        lr_upper = lr_line + (desvios_lr * std_dev)
        lr_lower = lr_line - (desvios_lr * std_dev)
        
        # Banda Superior Reta
        fig.add_trace(go.Scatter(
            x=df_lr.index, y=lr_upper,
            name='Banda Superior (LR)',
            line=dict(color='rgba(255, 100, 100, 0.8)', dash='dash'),
            mode='lines'
        ))
        
        # Linha Central Reta
        fig.add_trace(go.Scatter(
            x=df_lr.index, y=lr_line,
            name='Linha Central (Média)',
            line=dict(color='rgba(100, 200, 255, 0.8)', width=2),
            mode='lines'
        ))
        
        # Banda Inferior Reta
        fig.add_trace(go.Scatter(
            x=df_lr.index, y=lr_lower,
            name='Banda Inferior (Compra)',
            line=dict(color='rgba(100, 255, 100, 0.8)', dash='dash'),
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(100, 255, 100, 0.1)' 
        ))
    else:
        # Fallback para o backtest de curvas 
        if 'LRL' in df_zoom.columns:
            fig.add_trace(go.Scatter(
                x=df_zoom.index, y=df_zoom['LRL'],
                name='LR Média (Curva)',
                line=dict(color='blue', dash='dash'),
                opacity=0.7
            ))
        if 'Banda_Inferior' in df_zoom.columns:
            fig.add_trace(go.Scatter(
                x=df_zoom.index, y=df_zoom['Banda_Inferior'],
                name='LR Inferior (Curva)',
                line=dict(color='gray', dash='dash'),
                opacity=0.7,
                fill='tonexty'
            ))

    titulo_extra = f" (Canal Reto: {window_lr} barras)" if window_lr else ""
    fig.update_layout(
        title=f'{ticker} - Momento Atual{titulo_extra}',
        template='plotly_dark',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

