import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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
    
    # Bandas de Bollinger
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_upper'],
        name='BB Superior',
        line=dict(color='gray', dash='dash'),
        opacity=0.5
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_middle'],
        name='BB Média',
        line=dict(color='blue', dash='dash'),
        opacity=0.5
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_lower'],
        name='BB Inferior',
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
