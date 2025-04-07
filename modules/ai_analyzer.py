import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AIAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.prepare_features()

    def prepare_features(self):
        """Prepare technical features for AI analysis"""
        # Basic features
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['LogReturns'] = np.log(self.df['Close']/self.df['Close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            self.df[f'MA{window}'] = self.df['Close'].rolling(window=window).mean()
            self.df[f'MA_Ratio_{window}'] = self.df['Close'] / self.df[f'MA{window}']
        
        # Volatility
        self.df['Volatility_20'] = self.df['Returns'].rolling(window=20).std()
        self.df['Volatility_60'] = self.df['Returns'].rolling(window=60).std()
        
        # Price momentum
        for window in [5, 10, 20]:
            self.df[f'Momentum_{window}'] = self.df['Close'] / self.df['Close'].shift(window)
        
        # Volume features
        self.df['Volume_Change'] = self.df['Volume'].pct_change()
        self.df['Volume_MA10'] = self.df['Volume'].rolling(window=10).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_MA10']
        
        # Technical indicators
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.df['EMA12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # Bollinger Bands
        self.df['BB_Middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['BB_Std'] = self.df['Close'].rolling(window=20).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + 2 * self.df['BB_Std']
        self.df['BB_Lower'] = self.df['BB_Middle'] - 2 * self.df['BB_Std']
        self.df['BB_Width'] = (self.df['BB_Upper'] - self.df['BB_Lower']) / self.df['BB_Middle']
        self.df['BB_Position'] = (self.df['Close'] - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'])
        
        # Day of week, month features
        self.df['DayOfWeek'] = pd.to_datetime(self.df.index).dayofweek
        self.df['Month'] = pd.to_datetime(self.df.index).month
        self.df['Year'] = pd.to_datetime(self.df.index).year
        
        # Target variables
        self.df['Target_Direction'] = np.where(self.df['Returns'] > 0, 1, 0)
        
        # Forward returns for prediction
        for days in [1, 3, 5, 10]:
            self.df[f'Forward_Return_{days}d'] = self.df['Close'].pct_change(periods=days).shift(-days)
        
        # Drop rows with NaN values
        self.df = self.df.dropna()

    def get_features(self):
        """Get feature set for model training"""
        # Select most important features for prediction
        basic_features = [
            'MA5', 'MA10', 'MA20', 'MA50', 'Volatility_20',
            'Momentum_5', 'Momentum_10', 'Volume_Ratio',
            'RSI', 'MACD', 'MACD_Hist', 'BB_Position', 'BB_Width',
            'DayOfWeek', 'Month'
        ]
        
        # Check which features exist in the dataframe
        available_features = [f for f in basic_features if f in self.df.columns]
        return available_features

    def train_classification_model(self, prediction_horizon=1):
        """Train a classification model to predict price direction"""
        target_col = 'Target_Direction'
        features = self.get_features()
        
        # Ensure we have enough data
        if len(self.df) < 30:
            return "Not enough data for model training"
        
        # Use data up to prediction_horizon days ago to avoid lookahead bias
        train_df = self.df.iloc[:-prediction_horizon] if prediction_horizon > 0 else self.df
        
        # Prepare features and target
        X = train_df[features].values
        y = train_df[target_col].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
        
        # Store model and scaler
        self.models['direction_classifier'] = {
            'model': model,
            'scaler': scaler,
            'features': features
        }
        
        return model

    def train_regression_model(self, forecast_days=5):
        """Train a regression model to predict future price"""
        target_col = f'Forward_Return_{forecast_days}d'
        
        if target_col not in self.df.columns:
            self.df[target_col] = self.df['Close'].pct_change(periods=forecast_days).shift(-forecast_days)
        
        features = self.get_features()
        
        # Ensure we have enough data
        if len(self.df) < 30 + forecast_days:
            return "Not enough data for model training"
        
        # Use data up to forecast_days days ago to avoid lookahead bias
        train_df = self.df.iloc[:-forecast_days]
        
        # Prepare features and target
        X = train_df[features].values
        y = train_df[target_col].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
        
        # Store model and scaler
        self.models[f'price_regressor_{forecast_days}d'] = {
            'model': model,
            'scaler': scaler,
            'features': features
        }
        
        return model

    def predict_trend(self):
        """Predict market trend using Random Forest classification"""
        try:
            # Train model if not already trained
            if 'direction_classifier' not in self.models:
                self.train_classification_model()
            
            model_info = self.models['direction_classifier']
            model = model_info['model']
            scaler = model_info['scaler']
            features = model_info['features']
            
            # Get latest feature values
            latest_features = self.df[features].iloc[-1:].values
            
            # Scale features
            X_scaled = scaler.transform(latest_features)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            prediction_proba = model.predict_proba(X_scaled)[0]
            confidence = max(prediction_proba) * 100
            
            # Get feature importance
            importances = model.feature_importances_
            feature_importance = dict(zip(features, importances))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            trend = "Bullish 📈" if prediction == 1 else "Bearish 📉"
            feature_info = ", ".join([f"{f}: {v:.3f}" for f, v in top_features])
            
            return f"{trend} (Confidence: {confidence:.1f}%, Key indicators: {feature_info})"
            
        except Exception as e:
            return f"Unable to generate trend prediction: {str(e)}"

    def predict_price_movement(self, days=5):
        """Predict price movement over specified days"""
        try:
            # Train model if not already trained
            model_key = f'price_regressor_{days}d'
            if model_key not in self.models:
                self.train_regression_model(forecast_days=days)
            
            if model_key not in self.models:
                return None, 0  # Model couldn't be trained
            
            model_info = self.models[model_key]
            model = model_info['model']
            scaler = model_info['scaler']
            features = model_info['features']
            
            # Get latest feature values
            latest_features = self.df[features].iloc[-1:].values
            
            # Scale features
            X_scaled = scaler.transform(latest_features)
            
            # Predict
            predicted_return = model.predict(X_scaled)[0]
            current_price = self.df['Close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)
            
            # Calculate a pseudo-confidence based on model feature importance 
            # and historical prediction errors
            # (this is a simplified approach)
            feature_importances = model.feature_importances_
            confidence = min(90, 50 + (sum(feature_importances) * 100))
            
            return predicted_price, confidence
            
        except Exception as e:
            print(f"Error predicting price movement: {str(e)}")
            return None, 0

    def monte_carlo_simulation(self, days=30, simulations=1000):
        """Run Monte Carlo simulation for price prediction"""
        try:
            # Get historical returns statistics
            returns = self.df['Returns'].dropna()
            mu = returns.mean()
            sigma = returns.std()
            
            # Last closing price
            last_price = self.df['Close'].iloc[-1]
            
            # Initialize simulation array
            simulation_df = pd.DataFrame()
            
            # Run simulations
            for i in range(simulations):
                prices = [last_price]
                for day in range(days):
                    # Generate random return from normal distribution
                    daily_return = np.random.normal(mu, sigma)
                    # Calculate next price
                    price = prices[-1] * (1 + daily_return)
                    prices.append(price)
                
                simulation_df[i] = prices
            
            # Calculate statistics
            final_prices = simulation_df.iloc[-1]
            mean_price = final_prices.mean()
            
            # Calculate confidence intervals
            confidence_5 = final_prices.quantile(0.05)
            confidence_95 = final_prices.quantile(0.95)
            
            # Calculate probability of profit
            prob_profit = (final_prices > last_price).mean() * 100
            
            results = {
                'mean_price': mean_price,
                'min_price': final_prices.min(),
                'max_price': final_prices.max(),
                'confidence_5': confidence_5,
                'confidence_95': confidence_95,
                'prob_profit': prob_profit,
                'simulations': simulation_df
            }
            
            return results
            
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {str(e)}")
            return None

    def support_resistance_ml(self):
        """Identify support and resistance levels using machine learning"""
        try:
            # Use local maxima and minima
            prices = self.df['Close'].values
            n = len(prices)
            
            pivots = []
            
            # Find local extrema with at least 5 points on each side
            window = 5
            for i in range(window, n-window):
                if all(prices[i] > prices[i-j] for j in range(1, window+1)) and all(prices[i] > prices[i+j] for j in range(1, window+1)):
                    pivots.append((i, prices[i], 'resistance'))
                elif all(prices[i] < prices[i-j] for j in range(1, window+1)) and all(prices[i] < prices[i+j] for j in range(1, window+1)):
                    pivots.append((i, prices[i], 'support'))
            
            # Group similar levels
            tolerance = 0.02  # 2% tolerance
            clusters = []
            
            for pivot in pivots:
                idx, price, pivot_type = pivot
                
                # Check if this price is close to any existing cluster
                found_cluster = False
                for i, (cluster_price, cluster_type, points) in enumerate(clusters):
                    if abs(price - cluster_price) / cluster_price < tolerance and cluster_type == pivot_type:
                        # Update cluster with new point
                        new_points = points + [idx]
                        new_price = sum(prices[p] for p in new_points) / len(new_points)
                        clusters[i] = (new_price, cluster_type, new_points)
                        found_cluster = True
                        break
                
                if not found_cluster:
                    # Create new cluster
                    clusters.append((price, pivot_type, [idx]))
            
            # Sort by strength (number of points)
            clusters.sort(key=lambda x: len(x[2]), reverse=True)
            
            # Return top levels
            support_levels = [price for price, level_type, _ in clusters if level_type == 'support'][:3]
            resistance_levels = [price for price, level_type, _ in clusters if level_type == 'resistance'][:3]
            
            return {
                'support': support_levels, 
                'resistance': resistance_levels
            }
            
        except Exception as e:
            print(f"Error identifying support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}
            
    def predict_with_multiple_models(self):
        """Combine predictions from multiple models"""
        results = {}
        
        # Trend prediction (direction)
        results['trend_prediction'] = self.predict_trend()
        
        # Price predictions for different time horizons
        for days in [1, 5, 10]:
            predicted_price, confidence = self.predict_price_movement(days=days)
            if predicted_price is not None:
                current_price = self.df['Close'].iloc[-1]
                pct_change = ((predicted_price / current_price) - 1) * 100
                results[f'{days}d_price'] = predicted_price
                results[f'{days}d_change'] = pct_change
                results[f'{days}d_confidence'] = confidence
        
        # Support and resistance levels
        sr_levels = self.support_resistance_ml()
        results['support_levels'] = sr_levels['support']
        results['resistance_levels'] = sr_levels['resistance']
        
        # Monte Carlo simulation for longer-term forecast (30 days)
        monte_carlo = self.monte_carlo_simulation(days=30, simulations=1000)
        if monte_carlo:
            results['monte_carlo'] = {
                'mean_price': monte_carlo['mean_price'],
                'confidence_5': monte_carlo['confidence_5'],
                'confidence_95': monte_carlo['confidence_95'],
                'prob_profit': monte_carlo['prob_profit']
            }
        
        return results
