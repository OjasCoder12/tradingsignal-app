import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from .market_analysis import MarketAnalyzer
from .notifications import NotificationManager

class StockMonitor:
    def __init__(self):
        self.stocks = []
        self.predictions = {}
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.notification_manager = NotificationManager()
        self.last_notification_time = None

    def load_stocks_from_file(self, filepath: str) -> List[str]:
        """Load stock symbols from a text file"""
        try:
            with open(filepath, 'r') as f:
                self.stocks = [line.strip() for line in f.readlines() if line.strip()]
            return self.stocks
        except Exception as e:
            raise Exception(f"Error loading stocks from file: {str(e)}")

    def prepare_prediction_data(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for prediction"""
        try:
            # Get historical data
            stock = yf.Ticker(symbol)
            df = stock.history(period='1y')

            # Create features
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()

            # Create target (next day's price)
            df['Target'] = df['Close'].shift(-1)

            # Drop NaN values
            df = df.dropna()

            # Prepare features and target
            features = ['Close', 'MA5', 'MA20', 'RSI', 'Daily_Return', 'Volume_Change']
            X = df[features].values
            y = df['Target'].values

            return X, y
        except Exception as e:
            raise Exception(f"Error preparing prediction data: {str(e)}")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def predict_future_price(self, symbol: str, days_ahead: int = 7) -> Dict:
        """Predict future stock price"""
        try:
            X, y = self.prepare_prediction_data(symbol)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.model.fit(X_train, y_train)

            # Make prediction
            last_data = X[-1:]
            future_price = self.model.predict(last_data)[0]

            # Calculate confidence score
            confidence = self.model.score(X_test, y_test)

            return {
                'predicted_price': future_price,
                'confidence': confidence
            }
        except Exception as e:
            raise Exception(f"Error predicting price: {str(e)}")

    def generate_trading_signal(self, symbol: str, quantity: int = 100) -> Dict:
        """Generate trading signal with quantity recommendation"""
        try:
            market_analyzer = MarketAnalyzer(symbol)
            stock = yf.Ticker(symbol)
            current_data = stock.history(period='1d')
            current_price = current_data['Close'].iloc[-1]

            # Get prediction
            prediction = self.predict_future_price(symbol)
            predicted_price = prediction['predicted_price']
            confidence = prediction['confidence']

            # Get market analysis
            df = stock.history(period='3mo')
            trading_decision = market_analyzer.get_trading_decision(df)

            # Determine optimal quantity based on risk-reward
            risk_reward = trading_decision['profit_potential']['risk_reward_ratio']
            adjusted_quantity = int(quantity * min(1, risk_reward/2))

            # Calculate price change percentage
            price_change_pct = ((predicted_price - current_price) / current_price) * 100

            # Generate signal
            if price_change_pct > 5 and confidence > 0.7:
                signal = "BUY"
                quantity = adjusted_quantity
            elif price_change_pct < -5 and confidence > 0.7:
                signal = "SELL"
                quantity = adjusted_quantity
            else:
                signal = "HOLD"
                quantity = 0

            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'confidence': confidence,
                'signal': signal,
                'recommended_quantity': quantity,
                'trading_decision': trading_decision
            }
        except Exception as e:
            raise Exception(f"Error generating trading signal: {str(e)}")

    def should_send_notification(self) -> bool:
        """Check if it's time to send a notification"""
        current_time = datetime.now()

        # Send notification if:
        # 1. It's the first notification
        # 2. It's been at least 15 minutes since the last notification
        if (not self.last_notification_time or 
            (current_time - self.last_notification_time).total_seconds() >= 900):
            self.last_notification_time = current_time
            return True
        return False

    def monitor_stocks(self) -> List[Dict]:
        """Monitor all stocks and generate signals"""
        signals = []
        for symbol in self.stocks:
            try:
                signal = self.generate_trading_signal(symbol)
                signals.append(signal)
            except Exception as e:
                print(f"Error monitoring {symbol}: {str(e)}")
                continue

        # Send notifications if needed
        if signals and self.should_send_notification():
            self.notification_manager.send_notification(signals)

        return signals