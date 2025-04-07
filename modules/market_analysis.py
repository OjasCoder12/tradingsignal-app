import trafilatura
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class MarketAnalyzer:
    def __init__(self, symbol: str, target_currency: str = None):
        self.symbol = symbol

        # Check if it's an Indian stock
        self.is_indian = symbol.endswith('.NS') or symbol.endswith('.BO')
        self.exchange = 'NSE' if symbol.endswith('.NS') else 'BSE' if symbol.endswith('.BO') else 'US'

        # Set base and target currencies
        self.base_currency = 'INR' if self.is_indian else 'USD'
        self.target_currency = target_currency or self.base_currency

        # Cache for exchange rates
        self._exchange_rates = {}

        # Add Indian-specific news sources if applicable
        base_sources = [
            f"https://finance.yahoo.com/quote/{symbol}/news",
            f"https://seekingalpha.com/symbol/{symbol}",
            f"https://www.marketwatch.com/investing/stock/{symbol}"
        ]

        indian_sources = []
        if self.is_indian:
            base_symbol = symbol.split('.')[0]
            indian_sources = [
                f"https://www.moneycontrol.com/india/stockpricequote/{base_symbol}",
                f"https://economictimes.indiatimes.com/markets/stocks/news"
            ]

        self.news_sources = base_sources + indian_sources
        self.stock = yf.Ticker(symbol)

    def _get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate between two currencies"""
        if from_currency == to_currency:
            return 1.0

        cache_key = f"{from_currency}{to_currency}"
        if cache_key in self._exchange_rates:
            return self._exchange_rates[cache_key]

        try:
            # Use Yahoo Finance to get exchange rate
            symbol = f"{from_currency}{to_currency}=X"
            forex = yf.Ticker(symbol)
            rate = forex.info.get('regularMarketPrice', None)

            if rate:
                self._exchange_rates[cache_key] = rate
                return rate

            # Fallback to common rates if YF fails
            fallback_rates = {
                'USDINR': 83.0,
                'INRUSD': 1/83.0,
                'EURUSD': 1.08,
                'GBPUSD': 1.26,
            }
            return fallback_rates.get(cache_key, 1.0)

        except Exception:
            # Return default rate if fetching fails
            if from_currency == 'USD' and to_currency == 'INR':
                return 83.0
            elif from_currency == 'INR' and to_currency == 'USD':
                return 1/83.0
            return 1.0

    def convert_currency(self, value: float, from_currency: str, to_currency: str) -> float:
        """Convert value from one currency to another"""
        if not value or from_currency == to_currency:
            return value
        rate = self._get_exchange_rate(from_currency, to_currency)
        return value * rate

    def _format_currency(self, value: float, currency: str = None) -> str:
        """Format currency value with appropriate symbol and scale"""
        if not value:
            return 'N/A'

        currency = currency or self.target_currency
        symbols = {'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£'}
        symbol = symbols.get(currency, currency)

        if currency == 'INR':
            value_cr = value / 1e7
            if value_cr >= 100000:  # 1000 Cr+
                return f"{symbol}{value_cr/1000:.2f}K Cr"
            elif value_cr >= 1:  # 1 Cr+
                return f"{symbol}{value_cr:.2f} Cr"
            else:
                value_lakh = value / 1e5
                return f"{symbol}{value_lakh:.2f}L"
        else:
            if value >= 1e12:  # Trillion
                return f"{symbol}{value/1e12:.2f}T"
            elif value >= 1e9:  # Billion
                return f"{symbol}{value/1e9:.2f}B"
            elif value >= 1e6:  # Million
                return f"{symbol}{value/1e6:.2f}M"
            else:
                return f"{symbol}{value:.2f}"

    def get_company_fundamentals(self) -> Dict:
        """Get company fundamental data with currency conversion"""
        try:
            info = self.stock.info

            # Get and convert market cap
            market_cap = info.get('marketCap', 0)
            if market_cap:
                market_cap = self.convert_currency(market_cap, self.base_currency, self.target_currency)
            market_cap_str = self._format_currency(market_cap, self.target_currency)

            # Get and convert current price
            current_price = info.get('currentPrice', None)
            if current_price:
                current_price = self.convert_currency(current_price, self.base_currency, self.target_currency)

            # Get and convert target price
            target_price = info.get('targetMeanPrice', None)
            if target_price:
                target_price = self.convert_currency(target_price, self.base_currency, self.target_currency)

            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': market_cap_str,
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'peg_ratio': info.get('pegRatio', 'N/A'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'profit_margin': info.get('profitMargin', 0) * 100 if info.get('profitMargin') else 0,
                'debt_to_equity': info.get('debtToEquity', 'N/A'),
                'recommendation': info.get('recommendationKey', 'N/A').upper(),
                'target_price': target_price,
                'current_price': current_price,
                'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
                'currency': {'symbol': '₹' if self.target_currency == 'INR' else '$', 'code': self.target_currency}
            }
        except Exception as e:
            return {
                'error': f"Could not fetch fundamental data: {str(e)}"
            }

    def _convert_to_inr(self, value: float) -> float:
        """Convert USD value to INR if it's an Indian stock"""
        if self.is_indian and value:
            return value * self._get_exchange_rate('USD', 'INR')
        return value

    def _format_market_cap(self, market_cap: float) -> str:
        """Format market cap with appropriate currency and scale"""
        return self._format_currency(market_cap)

    def get_stock_news(self) -> List[Dict]:
        """Fetch recent news articles about the stock"""
        news_items = []

        for url in self.news_sources:
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded)
                    if text:
                        news_items.append({
                            'source': url,
                            'content': text,
                            'date': datetime.now().strftime('%Y-%m-%d')
                        })
            except Exception as e:
                continue

        return news_items

    def calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        prices = df['Close'].values
        window = min(20, len(prices))

        resistance = max(prices[-window:])
        support = min(prices[-window:])

        return support, resistance

    def predict_price_movement(self, df: pd.DataFrame) -> Dict:
        """Predict potential price movement and profit/loss"""
        current_price = df['Close'].iloc[-1]

        # Calculate price momentum
        returns = df['Close'].pct_change()
        momentum = returns.mean() * 100
        volatility = returns.std() * 100

        # Calculate support and resistance
        support, resistance = self.calculate_support_resistance(df)

        # Calculate potential profit targets
        upside_target = min(resistance * 1.05, current_price * (1 + abs(momentum) * 2))
        downside_risk = max(support * 0.95, current_price * (1 - abs(momentum) * 2))

        return {
            'momentum': momentum,
            'volatility': volatility,
            'support': support,
            'resistance': resistance,
            'upside_target': upside_target,
            'downside_risk': downside_risk,
            'risk_reward_ratio': (upside_target - current_price) / (current_price - downside_risk) if current_price != downside_risk else 0
        }

    def analyze_market_sentiment(self, df: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """
        Analyze market sentiment using technical indicators and price action
        Returns: (sentiment, confidence_score, signals)
        """
        signals = []

        # Calculate technical indicators
        last_close = df['Close'].iloc[-1]
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]

        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # Volume analysis
        avg_volume = df['Volume'].mean()
        latest_volume = df['Volume'].iloc[-1]
        volume_ratio = latest_volume / avg_volume

        # Score calculation
        score = 0
        confidence = 0

        # Price momentum analysis
        price_movement = self.predict_price_movement(df)

        if price_movement['momentum'] > 0:
            score += 0.5
            signals.append("📈 Positive price momentum")
        elif price_movement['momentum'] < 0:
            score -= 0.5
            signals.append("📉 Negative price momentum")

        # Support/Resistance analysis
        if last_close < price_movement['support'] * 1.02:
            score += 1
            signals.append("💪 Price near support level - potential bounce")
        elif last_close > price_movement['resistance'] * 0.98:
            score -= 1
            signals.append("⚠️ Price near resistance level - potential reversal")

        # Risk/Reward analysis
        if price_movement['risk_reward_ratio'] > 2:
            score += 1
            signals.append("✨ Favorable risk/reward ratio")
            confidence += 0.1

        # Trend analysis
        if last_close > sma_20 > sma_50:
            score += 1
            signals.append("⬆️ Strong uptrend: Price above both moving averages")
        elif last_close < sma_20 < sma_50:
            score -= 1
            signals.append("⬇️ Strong downtrend: Price below both moving averages")

        # RSI analysis
        if rsi > 70:
            score -= 0.5
            signals.append("⚠️ Overbought on RSI")
        elif rsi < 30:
            score += 0.5
            signals.append("✅ Oversold on RSI")

        # Volume confirmation
        if volume_ratio > 1.5:
            confidence += 0.2
            signals.append("📈 Strong volume support")
        elif volume_ratio < 0.5:
            confidence -= 0.1
            signals.append("📉 Low volume: weak trend")

        # Normalize score and set sentiment
        if score > 1:
            sentiment = "STRONGLY BULLISH"
        elif score > 0:
            sentiment = "BULLISH"
        elif score < -1:
            sentiment = "STRONGLY BEARISH"
        elif score < 0:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        confidence_score = min(0.9, 0.5 + abs(score) * 0.2 + confidence)

        return sentiment, confidence_score, signals

    def detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect common chart patterns in price data
        Returns a list of detected patterns with details
        """
        detected_patterns = []
        
        # Ensure we have enough data
        if len(df) < 50:
            return detected_patterns
            
        # Get price data
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Detection window (last N candles)
        window = min(60, len(df) - 1)
        
        # Function to check if price is near a level within tolerance
        def is_near(price1, price2, tolerance=0.02):
            return abs(price1 - price2) / price2 < tolerance
        
        # Head and Shoulders pattern detection
        # Look for 3 peaks with middle peak higher
        try:
            # Find local maxima
            peaks = []
            for i in range(5, window):
                if high[-(i-2)] > high[-(i-1)] and high[-(i-2)] > high[-(i-3)] and \
                   high[-(i-2)] > high[-(i)] and high[-(i-2)] > high[-(i+1)]:
                    peaks.append((len(close) - i + 2, high[-(i-2)]))
            
            # Need at least 3 peaks for H&S
            if len(peaks) >= 3:
                # Sort peaks by price
                peaks.sort(key=lambda x: x[1], reverse=True)
                
                # Get index of highest peak
                highest_idx = peaks[0][0]
                
                # Get peaks before and after the highest peak
                left_peaks = [p for p in peaks[1:] if p[0] < highest_idx]
                right_peaks = [p for p in peaks[1:] if p[0] > highest_idx]
                
                # If we have peaks on both sides
                if left_peaks and right_peaks:
                    left_shoulder = max(left_peaks, key=lambda x: x[1])
                    right_shoulder = max(right_peaks, key=lambda x: x[1])
                    
                    # Check if shoulders are at similar heights
                    if is_near(left_shoulder[1], right_shoulder[1], 0.05):
                        detected_patterns.append({
                            'pattern': 'Head and Shoulders',
                            'type': 'Bearish Reversal',
                            'confidence': 0.7,
                            'description': 'Bearish reversal pattern with three peaks, middle peak (head) higher than shoulders'
                        })
        except Exception as e:
            pass
            
        # Double Top pattern detection
        try:
            recent_high = max(high[-window:])
            high_indices = [i for i in range(window) if is_near(high[-(i+1)], recent_high, 0.03)]
            
            if len(high_indices) >= 2 and max(high_indices) - min(high_indices) >= 10:
                # Check if there's a significant dip between the tops
                min_between = min(low[-max(high_indices):-(min(high_indices)+1)])
                if (recent_high - min_between) / recent_high > 0.03:
                    # Check if volume is lower on second peak
                    vol_first = volume[-(max(high_indices)+1)]
                    vol_second = volume[-(min(high_indices)+1)]
                    
                    confidence = 0.6
                    if vol_second < vol_first:
                        confidence += 0.2
                    
                    detected_patterns.append({
                        'pattern': 'Double Top',
                        'type': 'Bearish Reversal',
                        'confidence': confidence,
                        'description': 'Bearish reversal pattern showing two distinct tops at similar price levels'
                    })
        except Exception as e:
            pass
            
        # Double Bottom pattern detection
        try:
            recent_low = min(low[-window:])
            low_indices = [i for i in range(window) if is_near(low[-(i+1)], recent_low, 0.03)]
            
            if len(low_indices) >= 2 and max(low_indices) - min(low_indices) >= 10:
                # Check if there's a significant bounce between the bottoms
                max_between = max(high[-max(low_indices):-(min(low_indices)+1)])
                if (max_between - recent_low) / recent_low > 0.03:
                    # Check if volume is higher on second bottom
                    vol_first = volume[-(max(low_indices)+1)]
                    vol_second = volume[-(min(low_indices)+1)]
                    
                    confidence = 0.6
                    if vol_second > vol_first:
                        confidence += 0.2
                    
                    detected_patterns.append({
                        'pattern': 'Double Bottom',
                        'type': 'Bullish Reversal',
                        'confidence': confidence,
                        'description': 'Bullish reversal pattern showing two distinct bottoms at similar price levels'
                    })
        except Exception as e:
            pass
            
        # Bullish/Bearish Engulfing patterns (strong reversal signals)
        try:
            for i in range(1, window-1):
                # Check for bullish engulfing
                if df['Open'].iloc[-i] > df['Close'].iloc[-(i+1)] and \
                   df['Close'].iloc[-i] > df['Open'].iloc[-(i+1)] and \
                   df['Close'].iloc[-i] > df['Open'].iloc[-i] and \
                   df['Open'].iloc[-(i+1)] > df['Close'].iloc[-(i+1)]:
                    
                    # Confirm with prior downtrend
                    if df['Close'].iloc[-(i+2)] > df['Close'].iloc[-(i+1)]:
                        detected_patterns.append({
                            'pattern': 'Bullish Engulfing',
                            'type': 'Bullish Reversal',
                            'confidence': 0.65,
                            'description': 'Strong bullish reversal pattern where a green candle completely engulfs the previous red candle'
                        })
                        break
                
                # Check for bearish engulfing
                if df['Open'].iloc[-i] < df['Close'].iloc[-(i+1)] and \
                   df['Close'].iloc[-i] < df['Open'].iloc[-(i+1)] and \
                   df['Close'].iloc[-i] < df['Open'].iloc[-i] and \
                   df['Open'].iloc[-(i+1)] < df['Close'].iloc[-(i+1)]:
                    
                    # Confirm with prior uptrend
                    if df['Close'].iloc[-(i+2)] < df['Close'].iloc[-(i+1)]:
                        detected_patterns.append({
                            'pattern': 'Bearish Engulfing',
                            'type': 'Bearish Reversal',
                            'confidence': 0.65,
                            'description': 'Strong bearish reversal pattern where a red candle completely engulfs the previous green candle'
                        })
                        break
        except Exception as e:
            pass
            
        # Cup and Handle pattern (bullish continuation)
        try:
            # Define a cup shape period - looking for U-shaped price movement
            cup_window = min(40, window - 10)
            
            # Skip this pattern detection if not enough data
            if len(df) >= cup_window + 20:
                cup_section = df['Close'].iloc[-(cup_window+10):-10].values
                
                # Check if first and last points are at similar levels
                if is_near(cup_section[0], cup_section[-1], 0.05):
                    # Check if middle point is significantly lower (cup shape)
                    mid_idx = len(cup_section) // 2
                    mid_price = cup_section[mid_idx]
                    
                    cup_depth = (cup_section[0] - mid_price) / cup_section[0]
                    
                    if cup_depth > 0.1:  # At least 10% cup depth
                        # Check for handle (small downward drift after cup)
                        handle = df['Close'].iloc[-10:].values
                        
                        # Handle should be a small downward drift
                        if handle[0] > handle[len(handle)//2] and handle[-1] > handle[len(handle)//2]:
                            detected_patterns.append({
                                'pattern': 'Cup and Handle',
                                'type': 'Bullish Continuation',
                                'confidence': 0.7,
                                'description': 'Bullish continuation pattern with U-shaped cup followed by smaller handle retracement'
                            })
        except Exception as e:
            pass
            
        # Identify triangles (continuation patterns)
        try:
            # Use at least 15 candles for triangle detection
            triangle_window = min(30, window)
            
            # Get highs and lows
            period_highs = high[-triangle_window:]
            period_lows = low[-triangle_window:]
            
            # Calculate trend lines (linear regression)
            days = np.array(range(triangle_window))
            
            # High trendline
            high_coef = np.polyfit(days, period_highs, 1)
            high_trend = high_coef[0]
            
            # Low trendline
            low_coef = np.polyfit(days, period_lows, 1)
            low_trend = low_coef[0]
            
            # Check for triangle patterns
            
            # Ascending Triangle (flat top, rising bottom)
            if abs(high_trend) < 0.001 and low_trend > 0.001:
                detected_patterns.append({
                    'pattern': 'Ascending Triangle',
                    'type': 'Bullish Continuation',
                    'confidence': 0.6,
                    'description': 'Bullish continuation pattern with flat resistance and rising support'
                })
            
            # Descending Triangle (flat bottom, falling top)
            elif abs(low_trend) < 0.001 and high_trend < -0.001:
                detected_patterns.append({
                    'pattern': 'Descending Triangle',
                    'type': 'Bearish Continuation',
                    'confidence': 0.6,
                    'description': 'Bearish continuation pattern with flat support and falling resistance'
                })
            
            # Symmetrical Triangle (converging trendlines)
            elif high_trend < -0.001 and low_trend > 0.001:
                detected_patterns.append({
                    'pattern': 'Symmetrical Triangle',
                    'type': 'Continuation',
                    'confidence': 0.55,
                    'description': 'Continuation pattern with converging support and resistance trendlines'
                })
        except Exception as e:
            pass
            
        # Flag or Pennant patterns (short-term continuation after strong move)
        try:
            # Check for a strong prior move (flag pole)
            flag_window = min(15, window // 2)
            pole_window = min(10, flag_window)
            
            # Calculate price change for pole
            pole_change = (df['Close'].iloc[-flag_window] - df['Close'].iloc[-(flag_window+pole_window)]) / df['Close'].iloc[-(flag_window+pole_window)]
            
            # Significant move (more than 5%)
            if abs(pole_change) > 0.05:
                flag_highs = high[-flag_window:]
                flag_lows = low[-flag_window:]
                
                # Check if recent price action is consolidating (lower volatility)
                recent_vol = np.std(close[-flag_window:]) / np.mean(close[-flag_window:])
                previous_vol = np.std(close[-(flag_window+pole_window):-flag_window]) / np.mean(close[-(flag_window+pole_window):-flag_window])
                
                if recent_vol < previous_vol * 0.7:  # Volatility decreased significantly
                    # Determine if bullish or bearish based on pole direction
                    pattern_type = "Bullish" if pole_change > 0 else "Bearish"
                    
                    # Determine if flag or pennant based on shape
                    # Flag has parallel trend lines, pennant is more triangular
                    days = np.array(range(flag_window))
                    
                    # High and low trendlines
                    high_coef = np.polyfit(days, flag_highs, 1)
                    low_coef = np.polyfit(days, flag_lows, 1)
                    
                    # If trendlines are roughly parallel, it's a flag
                    if abs(high_coef[0] - low_coef[0]) < 0.001:
                        detected_patterns.append({
                            'pattern': f'{pattern_type} Flag',
                            'type': f'{pattern_type} Continuation',
                            'confidence': 0.6,
                            'description': f'{pattern_type} continuation pattern showing consolidation after strong move'
                        })
                    else:
                        # Otherwise, it's a pennant
                        detected_patterns.append({
                            'pattern': f'{pattern_type} Pennant',
                            'type': f'{pattern_type} Continuation',
                            'confidence': 0.6,
                            'description': f'{pattern_type} continuation pattern showing triangular consolidation after strong move'
                        })
        except Exception as e:
            pass
        
        # Identify doji candlesticks (indecision/reversal signal)
        try:
            for i in range(3):
                # Get latest candles
                candle_open = df['Open'].iloc[-(i+1)]
                candle_close = df['Close'].iloc[-(i+1)]
                candle_high = df['High'].iloc[-(i+1)]
                candle_low = df['Low'].iloc[-(i+1)]
                
                body_size = abs(candle_close - candle_open)
                total_range = candle_high - candle_low
                
                # Doji has very small body compared to range
                if total_range > 0 and body_size / total_range < 0.1:
                    # Prior trend direction
                    prior_trend = "Uptrend" if df['Close'].iloc[-(i+2)] > df['Close'].iloc[-(i+7)] else "Downtrend"
                    
                    detected_patterns.append({
                        'pattern': 'Doji',
                        'type': 'Reversal/Indecision',
                        'confidence': 0.5,
                        'description': f'Indecision candlestick with small body in {prior_trend}, potential reversal signal'
                    })
                    break
        except Exception as e:
            pass
        
        return detected_patterns
                
    def analyze_advanced_fundamentals(self) -> Dict:
        """Advanced fundamental analysis using multiple metrics"""
        fundamentals = self.get_company_fundamentals()
        
        # Skip if we don't have basic fundamentals
        if 'error' in fundamentals:
            return fundamentals
            
        advanced_metrics = {}
        
        # Get ticker info 
        info = self.stock.info
        
        # PEG ratio (PE/Growth) - lower is better, below 1 is attractive
        advanced_metrics['peg_ratio'] = fundamentals.get('peg_ratio', 'N/A')
        
        # Price to Sales ratio
        if 'priceToSalesTrailing12Months' in info:
            advanced_metrics['price_to_sales'] = info['priceToSalesTrailing12Months']
        
        # Price to Book ratio
        if 'priceToBook' in info:
            advanced_metrics['price_to_book'] = info['priceToBook']
        
        # Return on Equity (%)
        if 'returnOnEquity' in info:
            advanced_metrics['roe'] = info['returnOnEquity'] * 100 if info['returnOnEquity'] else 0
        
        # Return on Assets (%)
        if 'returnOnAssets' in info:
            advanced_metrics['roa'] = info['returnOnAssets'] * 100 if info['returnOnAssets'] else 0
        
        # Debt to Equity ratio
        advanced_metrics['debt_to_equity'] = fundamentals.get('debt_to_equity', 'N/A')
        
        # Current Ratio (Current Assets / Current Liabilities)
        if 'currentRatio' in info:
            advanced_metrics['current_ratio'] = info['currentRatio']
        
        # Quick Ratio ((Current Assets - Inventory) / Current Liabilities)
        if 'quickRatio' in info:
            advanced_metrics['quick_ratio'] = info['quickRatio']
        
        # EBITDA Margins
        if 'ebitdaMargins' in info:
            advanced_metrics['ebitda_margin'] = info['ebitdaMargins'] * 100 if info['ebitdaMargins'] else 0
        
        # Gross Margins
        if 'grossMargins' in info:
            advanced_metrics['gross_margin'] = info['grossMargins'] * 100 if info['grossMargins'] else 0
        
        # Net Profit Margin
        advanced_metrics['profit_margin'] = fundamentals.get('profit_margin', 0)
        
        # Dividend related metrics
        advanced_metrics['dividend_yield'] = fundamentals.get('dividend_yield', 0)
        
        if 'dividendRate' in info:
            advanced_metrics['dividend_rate'] = info['dividendRate']
        
        if 'payoutRatio' in info:
            advanced_metrics['payout_ratio'] = info['payoutRatio'] * 100 if info['payoutRatio'] else 0
        
        # Earnings Growth
        advanced_metrics['earnings_growth'] = fundamentals.get('earnings_growth', 0)
        
        # Revenue Growth
        if 'revenueGrowth' in info:
            advanced_metrics['revenue_growth'] = info['revenueGrowth'] * 100 if info['revenueGrowth'] else 0
        
        # Analyst coverage
        if 'numberOfAnalystOpinions' in info:
            advanced_metrics['analyst_count'] = info['numberOfAnalystOpinions']
        
        # Institutional ownership percentage
        if 'heldPercentInstitutions' in info:
            advanced_metrics['institutional_ownership'] = info['heldPercentInstitutions'] * 100 if info['heldPercentInstitutions'] else 0
        
        # Calculate Piotroski F-Score (simple version)
        f_score = 0
        
        # Profitability criteria
        if advanced_metrics.get('roa', 0) > 0:
            f_score += 1
        if advanced_metrics.get('profit_margin', 0) > 0:
            f_score += 1
        if advanced_metrics.get('earnings_growth', 0) > 0:
            f_score += 1
        
        # Leverage, Liquidity criteria
        if advanced_metrics.get('debt_to_equity', float('inf')) != 'N/A':
            if float(advanced_metrics['debt_to_equity']) < 1:
                f_score += 1
        if advanced_metrics.get('current_ratio', 0) > 1:
            f_score += 1
        
        # Operating efficiency
        if advanced_metrics.get('gross_margin', 0) > 20:
            f_score += 1
        
        advanced_metrics['piotroski_score'] = f_score
        
        # Score interpretation
        if f_score >= 8:
            advanced_metrics['financial_health'] = "Excellent"
        elif f_score >= 6:
            advanced_metrics['financial_health'] = "Good"
        elif f_score >= 4:
            advanced_metrics['financial_health'] = "Average"
        else:
            advanced_metrics['financial_health'] = "Poor"
        
        # Value assessment based on combined metrics
        value_score = 0
        
        # PE ratio check (lower is better)
        if fundamentals.get('pe_ratio') != 'N/A':
            pe = float(fundamentals['pe_ratio'])
            if pe < 15:
                value_score += 2
            elif pe < 25:
                value_score += 1
            elif pe > 40:
                value_score -= 1
        
        # PEG ratio check (lower is better, below 1 is attractive)
        if advanced_metrics.get('peg_ratio') != 'N/A':
            peg = float(advanced_metrics['peg_ratio'])
            if peg < 1:
                value_score += 2
            elif peg < 2:
                value_score += 1
            elif peg > 3:
                value_score -= 1
        
        # Dividend yield check (higher is better for income)
        if advanced_metrics.get('dividend_yield', 0) > 4:
            value_score += 2
        elif advanced_metrics.get('dividend_yield', 0) > 2:
            value_score += 1
        
        # Growth checks
        if advanced_metrics.get('earnings_growth', 0) > 15:
            value_score += 2
        elif advanced_metrics.get('earnings_growth', 0) > 8:
            value_score += 1
        
        if advanced_metrics.get('revenue_growth', 0) > 15:
            value_score += 1
        
        # Profitability checks
        if advanced_metrics.get('profit_margin', 0) > 20:
            value_score += 2
        elif advanced_metrics.get('profit_margin', 0) > 10:
            value_score += 1
        
        if advanced_metrics.get('roe', 0) > 20:
            value_score += 2
        elif advanced_metrics.get('roe', 0) > 15:
            value_score += 1
        
        # Assign value rating
        if value_score >= 8:
            advanced_metrics['value_rating'] = "Excellent Value"
        elif value_score >= 5:
            advanced_metrics['value_rating'] = "Good Value"
        elif value_score >= 2:
            advanced_metrics['value_rating'] = "Fair Value"
        elif value_score >= 0:
            advanced_metrics['value_rating'] = "Fully Valued"
        else:
            advanced_metrics['value_rating'] = "Overvalued"
        
        # Add to fundamentals
        fundamentals['advanced_metrics'] = advanced_metrics
        
        return fundamentals
                
    def get_trading_decision(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading decision based on technical and fundamental analysis
        Includes chart pattern detection and advanced analysis
        """
        sentiment, confidence, signals = self.analyze_market_sentiment(df)
        fundamentals = self.analyze_advanced_fundamentals()
        price_movement = self.predict_price_movement(df)
        
        # Detect chart patterns
        chart_patterns = self.detect_chart_patterns(df)
        
        # Process pattern signals and add to signals list
        pattern_score = 0
        for pattern in chart_patterns:
            pattern_type = pattern['type'].split()[0]  # Get first word (Bullish/Bearish)
            pattern_confidence = pattern['confidence']
            
            # Add to signals
            signals.append(f"📊 {pattern['pattern']} detected: {pattern['description']}")
            
            # Adjust score based on pattern type and confidence
            if pattern_type == "Bullish":
                pattern_score += pattern_confidence
            elif pattern_type == "Bearish":
                pattern_score -= pattern_confidence

        # Get recent price movement
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100)

        # Combine technical and fundamental analysis
        technical_score = 1 if sentiment.endswith("BULLISH") else -1 if sentiment.endswith("BEARISH") else 0
        
        # Add pattern score influence
        technical_score += pattern_score * 0.5  # Weight pattern detection at 50% of other technical signals
        
        fundamental_score = 0

        # Analyze fundamentals
        if fundamentals.get('recommendation') in ['BUY', 'STRONG_BUY']:
            fundamental_score += 1
        elif fundamentals.get('recommendation') in ['SELL', 'STRONG_SELL']:
            fundamental_score -= 1

        # Check PE ratio
        if fundamentals.get('pe_ratio') != 'N/A':
            if float(fundamentals['pe_ratio']) < 15:
                fundamental_score += 0.5
                signals.append("✅ Attractive PE ratio indicates potential value")
            elif float(fundamentals['pe_ratio']) > 30:
                fundamental_score -= 0.5
                signals.append("⚠️ High PE ratio suggests overvaluation")

        # Growth prospects
        if fundamentals.get('earnings_growth', 0) > 10:
            fundamental_score += 0.5
            signals.append("🌱 Strong earnings growth potential")

        # Price targets
        if fundamentals.get('target_price'):
            current_price = df['Close'].iloc[-1]
            price_target = fundamentals['target_price']
            potential_return = ((price_target - current_price) / current_price) * 100

            if potential_return > 20:
                fundamental_score += 1
                signals.append(f"🎯 Analyst target suggests {potential_return:.1f}% upside potential")
            elif potential_return < -20:
                fundamental_score -= 1
                signals.append(f"⚠️ Analyst target suggests {-potential_return:.1f}% downside risk")
                
        # Advanced metrics from Piotroski Score
        if 'advanced_metrics' in fundamentals:
            adv = fundamentals['advanced_metrics']
            signals.append(f"📊 Financial Health: {adv.get('financial_health', 'N/A')} (F-Score: {adv.get('piotroski_score', 'N/A')})")
            signals.append(f"💰 Value Assessment: {adv.get('value_rating', 'N/A')}")
            
            # Add score adjustment based on financial health
            if adv.get('financial_health') == "Excellent":
                fundamental_score += 0.5
            elif adv.get('financial_health') == "Poor":
                fundamental_score -= 0.5
                
            # Add score adjustment based on value rating
            if adv.get('value_rating') == "Excellent Value":
                fundamental_score += 0.5
            elif adv.get('value_rating') == "Overvalued":
                fundamental_score -= 0.5
                
            # Add key metrics to signals
            if adv.get('roe', 0) > 15:
                signals.append(f"✨ Strong ROE: {adv.get('roe', 0):.1f}%")
            if adv.get('profit_margin', 0) > 15:
                signals.append(f"💼 High Profit Margin: {adv.get('profit_margin', 0):.1f}%")
            
            # Add debt warning if applicable
            if adv.get('debt_to_equity', 'N/A') != 'N/A':
                if float(adv['debt_to_equity']) > 2:
                    signals.append(f"⚠️ High Debt-to-Equity Ratio: {adv['debt_to_equity']}")

        # Combined score
        total_score = technical_score + fundamental_score

        # Check if it's an Indian stock
        is_indian = self.symbol.endswith('.NS') or self.symbol.endswith('.BO')
        exchange = 'NSE' if self.symbol.endswith('.NS') else 'BSE' if self.symbol.endswith('.BO') else 'US'

        # Add Indian market specific context if applicable
        if is_indian:
            signals.append(f"🇮🇳 Trading on {exchange} | Indian Market Analysis")

            # For NSE/BSE stocks, consider market timing
            from datetime import datetime, time
            now = datetime.now()
            # Indian market trading hours: 9:15 AM to 3:30 PM IST, Mon-Fri
            market_open = time(9, 15)
            market_close = time(15, 30)

            if now.time() >= market_open and now.time() <= market_close and now.weekday() < 5:
                signals.append("📊 Indian Market Currently Open")
            else:
                signals.append("📊 Indian Market Currently Closed")

        # Add detected chart patterns to the decision
        pattern_details = []
        for pattern in chart_patterns:
            pattern_details.append({
                'name': pattern['pattern'],
                'type': pattern['type'],
                'confidence': pattern['confidence']
            })

        decision = {
            'sentiment': sentiment,
            'confidence': confidence,
            'price_change_5d': f"{price_change:.2f}%",
            'current_price': fundamentals.get('current_price', df['Close'].iloc[-1]),
            'market_cap': fundamentals.get('market_cap', 'N/A'),
            'recommendation': "HOLD",
            'signals': signals,
            'fundamentals': fundamentals,
            'exchange': exchange if is_indian else 'US',
            'chart_patterns': pattern_details,
            'profit_potential': {
                'upside_target': price_movement['upside_target'],
                'downside_risk': price_movement['downside_risk'],
                'risk_reward_ratio': price_movement['risk_reward_ratio']
            }
        }

        # Clear decision logic with profit targets
        if total_score >= 2 and confidence > 0.6:
            decision['recommendation'] = "STRONG BUY"
            decision['action_color'] = "green"
            signals.append(f"🎯 Strong buy signal with potential {((price_movement['upside_target']/df['Close'].iloc[-1] - 1) * 100):.1f}% upside")
        elif total_score >= 1:
            decision['recommendation'] = "BUY"
            decision['action_color'] = "lime"
            signals.append("✅ Moderate buy signal with positive risk/reward")
        elif total_score <= -2 and confidence > 0.6:
            decision['recommendation'] = "STRONG SELL"
            decision['action_color'] = "red"
            signals.append(f"⛔ Strong sell signal with potential {((1 - price_movement['downside_risk']/df['Close'].iloc[-1]) * 100):.1f}% downside")
        elif total_score <= -1:
            decision['recommendation'] = "SELL"
            decision['action_color'] = "orange"
            signals.append("⚠️ Moderate sell signal with negative risk/reward")
        else:
            decision['recommendation'] = "HOLD"
            decision['action_color'] = "gray"
            signals.append("⏸️ No clear edge - maintain current position")

        return decision