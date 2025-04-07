import plotly.graph_objects as go
import pandas as pd

def create_stock_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a candlestick chart for stock data
    """
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])

    fig.update_layout(
        title='Stock Price Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        height=500,
        template='plotly_white',
        xaxis_rangeslider_visible=False
    )

    return fig

def create_indicator_chart(df: pd.DataFrame, indicator: str, title: str) -> go.Figure:
    """
    Create a line chart for technical indicators
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[indicator],
        mode='lines',
        name=indicator
    ))

    fig.update_layout(
        title=title,
        yaxis_title='Value',
        xaxis_title='Date',
        height=300,
        template='plotly_white'
    )

    return fig
