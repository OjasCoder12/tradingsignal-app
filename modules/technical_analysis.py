import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def calculate_all_indicators(self):
        """Calculate all technical indicators for analysis"""
        # Simple Moving Averages
        self.df['SMA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA200'] = self.df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        self.df['EMA12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
        
        # Golden/Death Cross
        self.df['Golden_Cross'] = (self.df['SMA50'] > self.df['SMA200']) & (self.df['SMA50'].shift(1) <= self.df['SMA200'].shift(1))
        self.df['Death_Cross'] = (self.df['SMA50'] < self.df['SMA200']) & (self.df['SMA50'].shift(1) >= self.df['SMA200'].shift(1))
        
        # Bollinger Bands (20,2)
        self.df['BB_Middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['BB_Std'] = self.df['Close'].rolling(window=20).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + 2 * self.df['BB_Std']
        self.df['BB_Lower'] = self.df['BB_Middle'] - 2 * self.df['BB_Std']
        
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # ADX (Average Directional Index)
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14).mean()
        
        # Parabolic SAR
        self.calculate_psar()
        
        # Stochastic Oscillator
        low_14 = self.df['Low'].rolling(window=14).min()
        high_14 = self.df['High'].rolling(window=14).max()
        self.df['%K'] = 100 * ((self.df['Close'] - low_14) / (high_14 - low_14))
        self.df['%D'] = self.df['%K'].rolling(window=3).mean()
        
        # On-Balance Volume (OBV)
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        
        return self.df
        
    def calculate_psar(self, af_start=0.02, af_inc=0.02, af_max=0.2):
        """Calculate Parabolic SAR"""
        high = self.df['High']
        low = self.df['Low']
        
        # Starting values
        psar = [high[0]]
        ep = [high[0]]  # Extreme point
        af = [af_start]  # Acceleration factor
        trend = [1]  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(high)):
            # Previous values
            psar_prev = psar[-1]
            ep_prev = ep[-1]
            af_prev = af[-1]
            trend_prev = trend[-1]
            
            # Current PSAR value
            if trend_prev == 1:  # Uptrend
                psar_current = psar_prev + af_prev * (ep_prev - psar_prev)
                # Check if PSAR crosses below price
                if psar_current > low[i]:
                    psar_current = ep_prev
                    trend_current = -1
                    ep_current = low[i]
                    af_current = af_start
                else:
                    trend_current = 1
                    if high[i] > ep_prev:
                        ep_current = high[i]
                        af_current = min(af_prev + af_inc, af_max)
                    else:
                        ep_current = ep_prev
                        af_current = af_prev
            else:  # Downtrend
                psar_current = psar_prev + af_prev * (ep_prev - psar_prev)
                # Check if PSAR crosses above price
                if psar_current < high[i]:
                    psar_current = ep_prev
                    trend_current = 1
                    ep_current = high[i]
                    af_current = af_start
                else:
                    trend_current = -1
                    if low[i] < ep_prev:
                        ep_current = low[i]
                        af_current = min(af_prev + af_inc, af_max)
                    else:
                        ep_current = ep_prev
                        af_current = af_prev
            
            psar.append(psar_current)
            ep.append(ep_current)
            af.append(af_current)
            trend.append(trend_current)
        
        self.df['PSAR'] = psar
        self.df['PSAR_Trend'] = trend
        
        return self.df
        
    def calculate_fibonacci_levels(self, start_idx, end_idx):
        """Calculate Fibonacci retracement levels"""
        start_price = self.df['Close'].iloc[start_idx]
        end_price = self.df['Close'].iloc[end_idx]
        diff = end_price - start_price
        
        # Common Fibonacci levels
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        fib_levels = {}
        
        if diff > 0:  # Uptrend, calculate retracements
            for level in levels:
                fib_levels[level] = end_price - diff * level
        else:  # Downtrend, calculate retracements
            for level in levels:
                fib_levels[level] = start_price + abs(diff) * level
                
        return fib_levels

    def plot_moving_averages(self):
        """Create moving averages chart with golden/death crosses"""
        self.df['SMA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA200'] = self.df['Close'].rolling(window=200).mean()
        
        # Calculate crosses for highlighting
        self.df['Golden_Cross'] = (self.df['SMA50'] > self.df['SMA200']) & (self.df['SMA50'].shift(1) <= self.df['SMA200'].shift(1))
        self.df['Death_Cross'] = (self.df['SMA50'] < self.df['SMA200']) & (self.df['SMA50'].shift(1) >= self.df['SMA200'].shift(1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Close'],
            name='Price',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['SMA20'],
            name='SMA20',
            line=dict(color='orange')
        ))

        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['SMA50'],
            name='SMA50',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['SMA200'],
            name='SMA200',
            line=dict(color='purple')
        ))
        
        # Add markers for golden crosses
        golden_cross_idx = self.df[self.df['Golden_Cross']].index
        if not golden_cross_idx.empty:
            fig.add_trace(go.Scatter(
                x=golden_cross_idx,
                y=self.df.loc[golden_cross_idx, 'Close'],
                mode='markers',
                marker=dict(symbol='star', size=12, color='gold'),
                name='Golden Cross'
            ))
        
        # Add markers for death crosses
        death_cross_idx = self.df[self.df['Death_Cross']].index
        if not death_cross_idx.empty:
            fig.add_trace(go.Scatter(
                x=death_cross_idx,
                y=self.df.loc[death_cross_idx, 'Close'],
                mode='markers',
                marker=dict(symbol='x', size=12, color='black'),
                name='Death Cross'
            ))

        fig.update_layout(
            title='Moving Averages with Golden/Death Crosses',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400
        )

        return fig

    def plot_rsi(self):
        """Create RSI chart"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))

        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.add_hline(y=50, line_dash="dot", line_color="gray")

        fig.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=300
        )

        return fig
        
    def plot_bollinger_bands(self):
        """Create Bollinger Bands chart"""
        self.df['BB_Middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['BB_Std'] = self.df['Close'].rolling(window=20).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + 2 * self.df['BB_Std']
        self.df['BB_Lower'] = self.df['BB_Middle'] - 2 * self.df['BB_Std']
        
        fig = go.Figure()
        
        # Add price
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Close'],
            name='Price',
            line=dict(color='blue')
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['BB_Upper'],
            name='Upper Band',
            line=dict(color='red', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['BB_Middle'],
            name='Middle Band (SMA20)',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['BB_Lower'],
            name='Lower Band',
            line=dict(color='green', dash='dash')
        ))
        
        # Fill between upper and lower bands
        fig.add_trace(go.Scatter(
            x=self.df.index.tolist() + self.df.index.tolist()[::-1],
            y=self.df['BB_Upper'].tolist() + self.df['BB_Lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Bollinger Bands (20,2)',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400
        )
        
        return fig

    def plot_macd(self):
        """Create MACD chart"""
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = signal
        self.df['MACD_Hist'] = histogram

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
                           
        # Add price chart with EMA lines
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Close'],
            name='Price',
            line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=exp1,
            name='EMA12',
            line=dict(color='orange')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=exp2,
            name='EMA26',
            line=dict(color='purple')
        ), row=1, col=1)

        # Add MACD and signal lines
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=macd,
            name='MACD',
            line=dict(color='blue')
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=signal,
            name='Signal',
            line=dict(color='orange')
        ), row=2, col=1)
        
        # Add histogram as bar chart
        colors = ['red' if val < 0 else 'green' for val in histogram]
        
        fig.add_trace(go.Bar(
            x=self.df.index,
            y=histogram,
            name='Histogram',
            marker_color=colors
        ), row=2, col=1)

        fig.update_layout(
            title='MACD Indicator',
            xaxis_title='Date',
            height=500
        )

        return fig
        
    def plot_parabolic_sar(self):
        """Create Parabolic SAR chart"""
        # Make sure PSAR is calculated
        if 'PSAR' not in self.df.columns:
            self.calculate_psar()
            
        fig = go.Figure()
        
        # Add price candles
        fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='Price'
        ))
        
        # Add Parabolic SAR dots
        bullish = self.df[self.df['PSAR_Trend'] == 1]
        bearish = self.df[self.df['PSAR_Trend'] == -1]
        
        fig.add_trace(go.Scatter(
            x=bullish.index,
            y=bullish['PSAR'],
            mode='markers',
            marker=dict(symbol='circle', size=5, color='green'),
            name='Bullish SAR'
        ))
        
        fig.add_trace(go.Scatter(
            x=bearish.index,
            y=bearish['PSAR'],
            mode='markers',
            marker=dict(symbol='circle', size=5, color='red'),
            name='Bearish SAR'
        ))
        
        fig.update_layout(
            title='Parabolic SAR',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500
        )
        
        return fig
        
    def plot_stochastic(self):
        """Create Stochastic Oscillator chart"""
        # Calculate Stochastic Oscillator
        low_14 = self.df['Low'].rolling(window=14).min()
        high_14 = self.df['High'].rolling(window=14).max()
        self.df['%K'] = 100 * ((self.df['Close'] - low_14) / (high_14 - low_14))
        self.df['%D'] = self.df['%K'].rolling(window=3).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['%K'],
            name='%K',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['%D'],
            name='%D',
            line=dict(color='orange')
        ))
        
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig.update_layout(
            title='Stochastic Oscillator',
            xaxis_title='Date',
            yaxis_title='Value',
            height=300
        )
        
        return fig
        
    def plot_obv(self):
        """Create On-Balance Volume chart"""
        # Calculate OBV
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Add price
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Close'],
            name='Price',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Add OBV
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['OBV'],
            name='OBV',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.update_layout(
            title='On-Balance Volume (OBV)',
            xaxis_title='Date',
            height=500
        )
        
        return fig
        
    def plot_adx(self):
        """Create ADX chart"""
        # Calculate True Range
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Calculate +DI and -DI
        up_move = self.df['High'].diff()
        down_move = self.df['Low'].diff(-1).abs()
        
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        
        # Calculate ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=14).mean()
        
        self.df['ADX'] = adx
        self.df['Plus_DI'] = plus_di
        self.df['Minus_DI'] = minus_di
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=adx,
            name='ADX',
            line=dict(color='black')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=plus_di,
            name='+DI',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=minus_di,
            name='-DI',
            line=dict(color='red')
        ))
        
        fig.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Strength")
        
        fig.update_layout(
            title='Average Directional Index (ADX)',
            xaxis_title='Date',
            yaxis_title='Value',
            height=400
        )
        
        return fig