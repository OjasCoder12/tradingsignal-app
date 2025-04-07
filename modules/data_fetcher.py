import yfinance as yf
import pandas as pd

class StockDataFetcher:
    def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        """
        try:
            # Add exchange suffix if not present for Indian stocks
            if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
                # Try NSE first, then BSE if NSE fails
                try:
                    df = yf.Ticker(f"{symbol}.NS").history(period=period)
                    if not df.empty:
                        return df
                except:
                    pass

                try:
                    df = yf.Ticker(f"{symbol}.BO").history(period=period)
                    if not df.empty:
                        return df
                except:
                    pass

                raise ValueError(f"No data found for symbol {symbol} on NSE or BSE")

            # If suffix is already present, fetch directly
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return df

        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")