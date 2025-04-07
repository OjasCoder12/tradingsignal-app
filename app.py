import streamlit as st
import pandas as pd
import time
import os  # Added for environment variable setting
from pygments.lexers import go
from modules.data_fetcher import StockDataFetcher
from modules.technical_analysis import TechnicalAnalyzer
from modules.ai_analyzer import AIAnalyzer
from modules.visualization import create_stock_chart, create_indicator_chart
from modules.market_analysis import MarketAnalyzer
from modules.stock_monitor import StockMonitor

st.set_page_config(
    page_title="Stock Market Analysis Tool",
    page_icon="📈",
    layout="wide"
)


class StockAnalysisApp:
    def __init__(self):
        self.initialize_session_state()
        if not self.show_disclaimer():
            st.stop()
        self.setup_sidebar()
        self.main()

    def show_disclaimer(self):
        if 'terms_accepted' not in st.session_state:
            st.session_state.terms_accepted = False

        if not st.session_state.terms_accepted:
            st.markdown("""
            ## ⚠️ IMPORTANT DISCLAIMER

            Please read and accept the following terms and conditions before proceeding:

            1. This tool is for informational purposes only and does not constitute financial advice.
            2. Stock trading involves substantial risk and may result in the loss of part or all of your investment.
            3. The developers and providers of this tool:
                - Are not financial advisors
                - Are not responsible for any trading decisions you make
                - Make no guarantees about the accuracy of the information provided
                - Are not liable for any losses or damages resulting from use of this tool
            4. Past performance does not guarantee future results.
            5. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
            """)

            def on_accept():
                st.session_state.terms_accepted = True

            def on_decline():
                st.session_state.terms_accepted = False
                st.error("You must accept the terms to use this application.")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.button("I Accept", on_click=on_accept, type="primary")
            with col2:
                st.button("I Decline", on_click=on_decline)

            return st.session_state.terms_accepted
        return True

    def initialize_session_state(self):
        if 'alert_threshold' not in st.session_state:
            st.session_state.alert_threshold = 5.0
        if 'selected_strategy' not in st.session_state:
            st.session_state.selected_strategy = 'SMA Crossover'
        if 'stock_list_file' not in st.session_state:
            st.session_state.stock_list_file = None
        if 'monitor_active' not in st.session_state:
            st.session_state.monitor_active = False
        if 'quantity' not in st.session_state:
            st.session_state.quantity = 100
        if 'user_phone' not in st.session_state:  # Added for phone number handling.
            st.session_state.user_phone = ""
        if 'target_currency' not in st.session_state:
            st.session_state.target_currency = None

    def setup_sidebar(self):
        st.sidebar.title("📊 Analysis Settings")

        # Currency Selection
        st.sidebar.subheader("💱 Currency Settings")
        target_currency = st.sidebar.selectbox(
            "Select Display Currency",
            options=['Auto', 'USD', 'INR', 'EUR', 'GBP'],
            help="Auto will use INR for Indian stocks and USD for others"
        )
        st.session_state.target_currency = None if target_currency == 'Auto' else target_currency

        # Phone Number Input for Notifications
        st.sidebar.subheader("📱 Notification Settings")
        user_phone = st.sidebar.text_input(
            "Enter your phone number (e.g., +1234567890)",
            value=st.session_state.get('user_phone', ''),
            help="Enter your phone number to receive SMS alerts"
        )

        if user_phone:
            if user_phone.startswith('+') and len(user_phone) >= 10:
                st.session_state.user_phone = user_phone
                os.environ['USER_PHONE_NUMBER'] = user_phone
            else:
                st.sidebar.error("Please enter a valid phone number starting with '+' country code")

        # Stock Monitoring Section
        st.sidebar.subheader("📈 Stock Monitoring")
        uploaded_file = st.sidebar.file_uploader("Upload Stock List (TXT)", type=['txt'])
        if uploaded_file:
            st.session_state.stock_list_file = uploaded_file

        st.session_state.quantity = st.sidebar.number_input(
            "Default Quantity",
            min_value=1,
            value=100,
            step=10
        )

        st.session_state.monitor_active = st.sidebar.checkbox(
            "Enable Auto-Monitoring",
            value=st.session_state.monitor_active
        )

        # Original sidebar components
        st.sidebar.subheader("Market Selection")
        market = st.sidebar.radio(
            "Select Market",
            options=["US", "India (NSE)", "India (BSE)"]
        )

        # Stock Selection
        st.sidebar.subheader("Stock Selection")

        # Default symbols based on market
        default_symbols = {
            "US": "AAPL",
            "India (NSE)": "RELIANCE.NS",
            "India (BSE)": "RELIANCE.BO"
        }

        # Suggestions based on market
        suggestions = ""
        if market == "India (NSE)":
            suggestions = "e.g., RELIANCE.NS, INFY.NS, HDFCBANK.NS, TCS.NS, SBIN.NS"
        elif market == "India (BSE)":
            suggestions = "e.g., RELIANCE.BO, INFY.BO, HDFCBANK.BO, TCS.BO, SBIN.BO"
        else:
            suggestions = "e.g., AAPL, MSFT, GOOGL, AMZN, TSLA"

        symbol = st.sidebar.text_input(
            f"Enter Stock Symbol ({suggestions})",
            value=default_symbols[market]
        ).upper()

        period = st.sidebar.selectbox(
            "Select Time Period",
            options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3
        )

        # Alert Configuration
        st.sidebar.subheader("🔔 Price Alerts")
        alert_threshold = st.sidebar.slider(
            "Alert me when price changes by (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )

        # Indian Market Index Info
        if "India" in market:
            st.sidebar.subheader("🇮🇳 Indian Market Indices")
            if market == "India (NSE)":
                st.sidebar.info("NIFTY 50: ^NSEI\nNIFTY Bank: ^NSEBANK\nNIFTY IT: ^CNXIT")
            else:
                st.sidebar.info("SENSEX: ^BSESN\nBSE 100: ^BSE100\nBSE 200: ^BSE200")

        st.session_state.update({
            'symbol': symbol,
            'period': period,
            'alert_threshold': alert_threshold,
            'market': market
        })

    def display_monitoring_dashboard(self, stock_monitor):
        """Display the stock monitoring dashboard"""
        st.subheader("📊 Stock Monitoring Dashboard")
        st.markdown("*Auto-updates every 15 seconds*")

        if st.session_state.stock_list_file:
            # Save uploaded file
            with open("temp_stocks.txt", "wb") as f:
                f.write(st.session_state.stock_list_file.getvalue())

            # Load and monitor stocks
            try:
                stock_monitor.load_stocks_from_file("temp_stocks.txt")
                signals = stock_monitor.monitor_stocks()

                # Display signals in a table
                for signal in signals:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            f"{signal['symbol']}",
                            f"{signal['trading_decision']['fundamentals']['currency']}{signal['current_price']:.2f}",
                            f"{signal['price_change_pct']:.2f}%"
                        )

                    with col2:
                        st.metric(
                            "Predicted Price",
                            f"{signal['trading_decision']['fundamentals']['currency']}{signal['predicted_price']:.2f}",
                            f"Confidence: {signal['confidence']:.2%}"
                        )

                    with col3:
                        signal_color = {
                            'BUY': 'green',
                            'SELL': 'red',
                            'HOLD': 'gray'
                        }[signal['signal']]

                        st.markdown(
                            f"""
                            <div style='padding: 10px; background-color: {signal_color}30; border-radius: 5px; text-align: center;'>
                            <h3 style='color: {signal_color};'>{signal['signal']}</h3>
                            <p>Qty: {signal['recommended_quantity']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with col4:
                        st.write("Signals:")
                        for trading_signal in signal['trading_decision']['signals'][:3]:
                            st.write(f"- {trading_signal}")

                    st.markdown("---")

            except Exception as e:
                st.error(f"Error monitoring stocks: {str(e)}")
        else:
            st.info("Please upload a text file with stock symbols (one per line)")

    def main(self):
        st.title("📈 Smart Stock Analysis Tool")
        st.markdown("*Making investment decisions easier with AI-powered analysis*")

        # Initialize stock monitor
        stock_monitor = StockMonitor()

        # Display monitoring dashboard if active
        if st.session_state.monitor_active:
            self.display_monitoring_dashboard(stock_monitor)
            time.sleep(15)  # Wait 15 seconds before next update
            st.experimental_rerun()  # Rerun the app to update

        # Original main content
        if "India" in st.session_state.get('market', ''):
            st.info("🇮🇳 Analyzing Indian Stock Market data | Exchange: " +
                    ("NSE" if "NSE" in st.session_state.market else "BSE"))

        try:
            # Fetch Data
            with st.spinner('Fetching stock data...'):
                data_fetcher = StockDataFetcher()
                df = data_fetcher.get_stock_data(
                    st.session_state.symbol,
                    st.session_state.period
                )

            # Market Analysis with currency conversion
            market_analyzer = MarketAnalyzer(
                st.session_state.symbol,
                target_currency=st.session_state.target_currency
            )
            trading_decision = market_analyzer.get_trading_decision(df)
            currency_info = trading_decision['fundamentals'].get('currency', {'symbol': '$', 'code': 'USD'})
            currency_symbol = currency_info['symbol']

            # Create main dashboard
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"{trading_decision['fundamentals']['name']} ({st.session_state.symbol})")
                fig = create_stock_chart(df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Current Price and Market Cap
                st.subheader("💰 Price Information")
                col3, col4 = st.columns(2)

                with col3:
                    current_price = trading_decision['current_price']
                    price_change = float(trading_decision['price_change_5d'].rstrip('%'))
                    st.metric(
                        "Current Price",
                        f"{currency_symbol}{current_price:.2f}",
                        f"{price_change:.2f}%"
                    )

                with col4:
                    st.metric(
                        "Market Cap",
                        trading_decision['market_cap']
                    )

                # Profit/Loss Potential
                st.subheader("📊 Profit/Loss Potential")
                profit_potential = trading_decision['profit_potential']

                col5, col6 = st.columns(2)
                with col5:
                    upside = (profit_potential['upside_target'] / current_price - 1) * 100
                    st.metric(
                        "Upside Target",
                        f"{currency_symbol}{profit_potential['upside_target']:.2f}",
                        f"+{upside:.1f}%"
                    )

                with col6:
                    downside = (1 - profit_potential['downside_risk'] / current_price) * 100
                    st.metric(
                        "Downside Risk",
                        f"{currency_symbol}{profit_potential['downside_risk']:.2f}",
                        f"-{downside:.1f}%"
                    )

                st.metric(
                    "Risk/Reward Ratio",
                    f"{profit_potential['risk_reward_ratio']:.2f}",
                    "Good" if profit_potential['risk_reward_ratio'] > 2 else "Fair" if profit_potential[
                                                                                           'risk_reward_ratio'] > 1 else "Poor"
                )

                # Trading Decision
                st.markdown(
                    f"""
                    # Trading Signal
                    <div style='padding: 20px; background-color: {trading_decision['action_color']}30; border-radius: 10px;'>
                    <h2 style='text-align: center; color: {trading_decision['action_color']};'>
                    {trading_decision['recommendation']}
                    </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Company Fundamentals
                st.subheader("🏢 Company Fundamentals")
                fund = trading_decision['fundamentals']
                metrics = {
                    "Sector": fund['sector'],
                    "Industry": fund['industry'],
                    "P/E Ratio": fund['pe_ratio'],
                    "Forward P/E": fund['forward_pe'],
                    "PEG Ratio": fund['peg_ratio'],
                    "Dividend Yield": f"{fund['dividend_yield']:.2f}%",
                    "Profit Margin": f"{fund['profit_margin']:.2f}%",
                    "Earnings Growth": f"{fund.get('earnings_growth', 0):.1f}%"
                }

                for key, value in metrics.items():
                    st.text(f"{key}: {value}")

                # Trading Signals
                st.subheader("🎯 Trading Signals")
                for signal in trading_decision['signals']:
                    st.markdown(f"- {signal}")

                # Alert Check
                if abs(price_change) >= st.session_state.alert_threshold:
                    st.warning(f"⚠️ Price Alert: {price_change:.2f}% change detected!")

            # Technical Analysis
            st.subheader("📈 Technical Analysis")
            tech_analyzer = TechnicalAnalyzer(df)

            # Create tabs for different indicators
            tabs = ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Parabolic SAR", "Stochastic", "ADX",
                    "Volume Analysis", "Pattern Detection"]
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(tabs)

            with tab1:
                st.markdown("*Moving averages help identify trends and golden/death crosses*")
                ma_fig = tech_analyzer.plot_moving_averages()
                st.plotly_chart(ma_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About Moving Averages & Crosses"):
                    st.markdown("""
                    - **SMA (Simple Moving Average)**: Average price over specified periods
                    - **Golden Cross**: When 50-day SMA crosses above 200-day SMA (bullish)
                    - **Death Cross**: When 50-day SMA crosses below 200-day SMA (bearish)
                    - **Trading Strategy**: Buy when price > SMA (uptrend), Sell when price < SMA (downtrend)
                    """)

            with tab2:
                st.markdown("*Bollinger Bands measure volatility and potential reversal points*")
                bb_fig = tech_analyzer.plot_bollinger_bands()
                st.plotly_chart(bb_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About Bollinger Bands"):
                    st.markdown("""
                    - **Upper Band**: SMA20 + (2 × Standard Deviation)
                    - **Middle Band**: 20-day SMA
                    - **Lower Band**: SMA20 - (2 × Standard Deviation)
                    - **Squeeze**: When bands narrow, volatility is low and breakout is possible
                    - **Trading Signals**: Price near upper band may be overbought, price near lower band may be oversold
                    """)

            with tab3:
                st.markdown("*RSI shows overbought/oversold conditions*")
                rsi_fig = tech_analyzer.plot_rsi()
                st.plotly_chart(rsi_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About RSI"):
                    st.markdown("""
                    - **Overbought**: RSI > 70 indicates potential selling opportunity
                    - **Oversold**: RSI < 30 indicates potential buying opportunity
                    - **Divergence**: When price makes new high/low but RSI doesn't, suggests potential reversal
                    - **Range-bound markets**: RSI oscillates between 30-70
                    - **Trending markets**: RSI can remain in overbought/oversold territories for extended periods
                    """)

            with tab4:
                st.markdown("*MACD indicates momentum and trend changes*")
                macd_fig = tech_analyzer.plot_macd()
                st.plotly_chart(macd_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About MACD"):
                    st.markdown("""
                    - **MACD Line**: Difference between 12-day EMA and 26-day EMA
                    - **Signal Line**: 9-day EMA of MACD Line
                    - **Histogram**: Difference between MACD Line and Signal Line
                    - **Bullish Signal**: MACD crosses above Signal Line
                    - **Bearish Signal**: MACD crosses below Signal Line
                    - **Zero Line Crossover**: MACD crossing above/below zero indicates trend change
                    """)

            with tab5:
                st.markdown("*Parabolic SAR identifies potential reversal points and stop-loss levels*")
                psar_fig = tech_analyzer.plot_parabolic_sar()
                st.plotly_chart(psar_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About Parabolic SAR"):
                    st.markdown("""
                    - **Dots Below Price**: Bullish signal, dots serve as stop-loss levels
                    - **Dots Above Price**: Bearish signal, dots serve as stop-loss levels
                    - **Dot Flip**: When dots switch positions, indicates trend reversal
                    - **Best used**: In trending markets, not sideways/ranging markets
                    - **Stop and Reverse**: When price crosses PSAR, reverse the position
                    """)

            with tab6:
                st.markdown("*Stochastic Oscillator identifies overbought/oversold levels and momentum*")
                stoch_fig = tech_analyzer.plot_stochastic()
                st.plotly_chart(stoch_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About Stochastic Oscillator"):
                    st.markdown("""
                    - **%K Line**: Fast stochastic comparing current price to high-low range
                    - **%D Line**: 3-day SMA of %K (slower signal line)
                    - **Overbought**: Above 80 indicates possible reversal lower
                    - **Oversold**: Below 20 indicates possible reversal higher
                    - **Crossovers**: %K crossing above/below %D generates trading signals
                    - **Divergence**: When price makes new high/low but stochastic doesn't, suggests potential reversal
                    """)

            with tab7:
                st.markdown("*ADX measures trend strength regardless of direction*")
                adx_fig = tech_analyzer.plot_adx()
                st.plotly_chart(adx_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About ADX"):
                    st.markdown("""
                    - **ADX Line**: Measures trend strength (not direction)
                    - **+DI Line**: Measures uptrend strength
                    - **-DI Line**: Measures downtrend strength
                    - **ADX > 25**: Strong trend present
                    - **ADX < 20**: Weak or no trend (ranging market)
                    - **Crossovers**: +DI crossing above -DI suggests bullish trend, opposite for bearish
                    """)

            with tab8:
                st.markdown("*On-Balance Volume (OBV) confirms price trends with volume*")
                obv_fig = tech_analyzer.plot_obv()
                st.plotly_chart(obv_fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About OBV"):
                    st.markdown("""
                    - **Rising OBV**: Volume flows into the security (buying pressure)
                    - **Falling OBV**: Volume flows out of the security (selling pressure)
                    - **Divergence**: When price moves one way but OBV moves opposite, suggests potential reversal
                    - **Confirmation**: OBV should confirm price trends (move in same direction)
                    - **Leading Indicator**: OBV changes can precede price changes
                    """)

            with tab9:
                st.markdown("*Chart Patterns and Fibonacci Analysis*")

                # Select range for Fibonacci analysis
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.selectbox("Select Start Date for Fibonacci", options=df.index,
                                              format_func=lambda x: x.strftime('%Y-%m-%d'))
                with col2:
                    end_date = st.selectbox("Select End Date for Fibonacci", options=df.index, index=len(df.index) - 1,
                                            format_func=lambda x: x.strftime('%Y-%m-%d'))

                # Convert dates to indices for fibonacci calculation
                start_idx = df.index.get_loc(start_date)
                end_idx = df.index.get_loc(end_date)

                # Calculate Fibonacci levels
                fib_levels = tech_analyzer.calculate_fibonacci_levels(start_idx, end_idx)

                # Create Fibonacci chart
                fig = go.Figure()

                # Add candlestick chart
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))

                # Add Fibonacci levels
                colors = ['purple', 'blue', 'teal', 'green', 'orange', 'red']
                for i, (level, price) in enumerate(fib_levels.items()):
                    if level in [0, 1]:  # Skip 0% and 100% levels to reduce clutter
                        continue
                    fig.add_hline(
                        y=price,
                        line_dash="dash",
                        line_color=colors[i % len(colors)],
                        annotation_text=f"Fib {level * 100}%: {price:.2f}",
                        annotation_position="left"
                    )

                fig.update_layout(
                    title='Fibonacci Retracement Levels',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Educational tip
                with st.expander("📚 About Chart Patterns & Fibonacci"):
                    st.markdown("""
                    **Common Chart Patterns:**
                    - **Head & Shoulders**: Reversal pattern with three peaks (middle highest)
                    - **Double Top/Bottom**: Reversal pattern with two similar peaks/troughs
                    - **Cup & Handle**: Bullish continuation pattern resembling a teacup
                    - **Triangle Patterns**: Continuation patterns showing consolidation

                    **Fibonacci Retracement Levels:**
                    - **38.2%, 50%, 61.8%**: Key retracement levels where price may find support/resistance
                    - **Use in Uptrends**: Measure from significant low to high
                    - **Use in Downtrends**: Measure from significant high to low
                    - **Trading Strategy**: Enter trades at retracement levels with confirmation
                    """)

                # Pattern detection info
                st.subheader("🔍 Pattern Detection")

                # Simple chart pattern detection - this is a basic implementation
                # A comprehensive implementation would require more complex algorithms

                # Check for potential double top/bottom
                rolling_window = min(30, len(df) - 1)
                recent_high = df['High'][-rolling_window:].max()
                recent_low = df['Low'][-rolling_window:].min()

                high_indices = df[-rolling_window:].index[df[-rolling_window:]['High'] > 0.95 * recent_high]
                low_indices = df[-rolling_window:].index[df[-rolling_window:]['Low'] < 1.05 * recent_low]

                # Check for potential double top
                if len(high_indices) >= 2 and (high_indices[-1] - high_indices[0]).days > 5:
                    st.info(
                        "🔔 **Potential Double Top Pattern Detected**: A bearish reversal pattern suggesting a possible downtrend.")

                # Check for potential double bottom
                if len(low_indices) >= 2 and (low_indices[-1] - low_indices[0]).days > 5:
                    st.info(
                        "🔔 **Potential Double Bottom Pattern Detected**: A bullish reversal pattern suggesting a possible uptrend.")

                # Check for potential head and shoulders
                # This is a very simplified check
                if len(high_indices) >= 3:
                    highs = df.loc[high_indices, 'High'].values
                    if highs[1] > highs[0] and highs[1] > highs[2]:
                        st.info(
                            "🔔 **Potential Head and Shoulders Pattern Detected**: A bearish reversal pattern suggesting a possible downtrend.")

            # AI Prediction Section
            st.subheader("🤖 AI-Powered Price Predictions")

            ai_analyzer = AIAnalyzer(df)
            ai_predictions = ai_analyzer.predict_with_multiple_models()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Trend Analysis")
                st.markdown(f"**AI Trend Prediction:** {ai_predictions.get('trend_prediction', 'Unable to predict')}")

                # Support and Resistance
                st.subheader("🎯 Key Price Levels")
                support_levels = ai_predictions.get('support_levels', [])
                resistance_levels = ai_predictions.get('resistance_levels', [])

                if support_levels:
                    st.write("**Support Levels:**")
                    for i, level in enumerate(support_levels):
                        st.markdown(f"- S{i + 1}: {currency_symbol}{level:.2f}")

                if resistance_levels:
                    st.write("**Resistance Levels:**")
                    for i, level in enumerate(resistance_levels):
                        st.markdown(f"- R{i + 1}: {currency_symbol}{level:.2f}")

            with col2:
                st.subheader("Price Forecasts")
                current_price = df['Close'].iloc[-1]

                # Display predictions for different time periods
                for days in [5, 10]:
                    if f'{days}d_price' in ai_predictions:
                        pred_price = ai_predictions[f'{days}d_price']
                        pred_change = ai_predictions[f'{days}d_change']
                        confidence = ai_predictions[f'{days}d_confidence']

                        change_color = "green" if pred_change > 0 else "red"

                        st.markdown(
                            f"""
                            <div style='padding: 10px; border-radius: 5px; margin-bottom: 10px; 
                                border: 1px solid {change_color}; background-color: {change_color}10;'>
                                <h4>{days}-Day Forecast</h4>
                                <p style='font-size: 18px;'>
                                    {currency_symbol}{pred_price:.2f} 
                                    <span style='color: {change_color}'>({pred_change:+.2f}%)</span>
                                </p>
                                <p>Confidence: {confidence:.1f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Monte Carlo simulation results
            if 'monte_carlo' in ai_predictions:
                mc = ai_predictions['monte_carlo']
                st.subheader("📊 30-Day Monte Carlo Simulation")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Expected Price",
                        f"{currency_symbol}{mc['mean_price']:.2f}",
                        f"{((mc['mean_price'] / current_price) - 1) * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "95% Confidence Range",
                        f"{currency_symbol}{mc['confidence_5']:.2f} - {currency_symbol}{mc['confidence_95']:.2f}"
                    )

                with col3:
                    st.metric(
                        "Probability of Profit",
                        f"{mc['prob_profit']:.1f}%"
                    )

                # Educational info about Monte Carlo
                with st.expander("📚 About Monte Carlo Simulation"):
                    st.markdown("""
                    **Monte Carlo Simulation** runs 1,000 random price paths to estimate possible outcomes:

                    - **Expected Price**: Average final price across all simulations
                    - **Confidence Range**: 90% of simulations fall within this range
                    - **Probability of Profit**: Percentage of simulations resulting in price increase

                    While useful for understanding potential scenarios, these are statistical projections, not guarantees.
                    """)

            # Market News
            st.subheader("📰 Latest Market News")
            news_items = market_analyzer.get_stock_news()

            if news_items:
                for item in news_items:
                    with st.expander(f"News from {item['source'].split('/')[2]}"):
                        st.write(item['content'][:500] + "...")
                        st.caption(f"Published: {item['date']}")
            else:
                st.info("No recent news articles found.")

            # Educational Resources
            with st.expander("📚 Educational Resources"):
                st.markdown("""
                ### Technical Analysis Resources

                **Chart Patterns**
                - Head & Shoulders: Bearish reversal pattern with three peaks (middle highest)
                - Double Top/Bottom: Reversal patterns signaling market tops and bottoms
                - Cup & Handle: Bullish continuation pattern resembling a teacup with handle
                - Triangle Patterns: Continuation patterns showing consolidation
                - Flag & Pennant: Short-term consolidation before trend continuation

                **Technical Indicators**
                - Moving Averages: Help identify trends and support/resistance levels
                - RSI (Relative Strength Index): Measures overbought/oversold conditions
                - MACD: Shows momentum and potential trend changes
                - Bollinger Bands: Measure volatility and potential price extremes
                - Fibonacci Retracement: Key levels where price may find support/resistance

                ### Fundamental Analysis Resources

                **Key Metrics**
                - PE Ratio: Price to Earnings - measures valuation relative to earnings
                - PEG Ratio: PE to Growth - adjusts PE for expected growth rate
                - ROE: Return on Equity - measures profitability vs. shareholder equity
                - Debt-to-Equity: Measures financial leverage and risk
                - Profit Margin: Net profit as percentage of revenue

                ### Risk Management Principles

                1. Position Sizing: Never risk more than 1-2% of capital on any single trade
                2. Stop-Loss Orders: Always use stops to limit potential losses
                3. Diversification: Don't concentrate too much in one sector/stock
                4. Risk/Reward Ratio: Look for opportunities with at least 2:1 ratio
                5. Correlation Analysis: Ensure portfolio components aren't highly correlated
                """)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please enter a valid stock symbol and try again.")


if __name__ == "__main__":
    app = StockAnalysisApp()