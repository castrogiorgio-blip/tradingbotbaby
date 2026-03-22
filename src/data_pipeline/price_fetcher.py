"""
Price Data Fetcher — pulls historical and recent OHLCV data from Alpaca.

Usage:
    from src.data_pipeline.price_fetcher import PriceFetcher
    fetcher = PriceFetcher()
    df = fetcher.get_historical_bars("AAPL", days=365*3)
    df = fetcher.get_latest_bars(["AAPL", "MSFT", "SPY"])
"""
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_api_keys, get_all_tickers, DATA_DIR


class PriceFetcher:
    """Fetches stock price data from Alpaca Markets API."""

    def __init__(self):
        keys = get_api_keys()
        self.client = StockHistoricalDataClient(
            api_key=keys["alpaca_api_key"],
            secret_key=keys["alpaca_secret_key"],
        )
        self.data_dir = DATA_DIR / "raw" / "prices"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PriceFetcher initialized with Alpaca client")

    def get_historical_bars(
        self,
        symbol: str,
        days: int = 365 * 3,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """
        Fetch historical daily bars for a single symbol.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            days: Number of days of history to fetch (default: 3 years)
            timeframe: Bar timeframe (default: daily)

        Returns:
            DataFrame with columns: open, high, low, close, volume, trade_count, vwap
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        logger.info(f"Fetching {days} days of {symbol} data ({start.date()} → {end.date()})")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        bars = self.client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df

        # If multi-index (symbol, timestamp), drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone for simplicity
        df = df.sort_index()

        logger.info(f"Got {len(df)} bars for {symbol} ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    def get_latest_bars(self, symbols: list[str] = None) -> dict[str, pd.Series]:
        """
        Fetch the most recent bar for multiple symbols.

        Args:
            symbols: List of tickers. If None, fetches all tickers from config.

        Returns:
            Dict mapping symbol → latest bar as a pandas Series
        """
        if symbols is None:
            symbols = get_all_tickers()

        logger.info(f"Fetching latest bars for {len(symbols)} symbols")

        request = StockLatestBarRequest(symbol_or_symbols=symbols)
        bars = self.client.get_stock_latest_bar(request)

        result = {}
        for symbol, bar in bars.items():
            result[symbol] = pd.Series({
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "trade_count": bar.trade_count,
                "vwap": bar.vwap,
                "timestamp": bar.timestamp,
            })

        logger.info(f"Got latest bars for {len(result)} symbols")
        return result

    def get_bulk_historical(
        self,
        symbols: list[str] = None,
        days: int = 365 * 3,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical bars for all symbols in the watchlist.

        Args:
            symbols: List of tickers. If None, uses all from config.
            days: Days of history per symbol.

        Returns:
            Dict mapping symbol → DataFrame of OHLCV bars
        """
        if symbols is None:
            symbols = get_all_tickers()

        logger.info(f"Fetching bulk historical data for {len(symbols)} symbols, {days} days each")

        all_data = {}
        for symbol in symbols:
            try:
                df = self.get_historical_bars(symbol, days=days)
                if not df.empty:
                    all_data[symbol] = df
                    # Save to disk
                    filepath = self.data_dir / f"{symbol}_daily.csv"
                    df.to_csv(filepath)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        logger.info(f"Successfully fetched data for {len(all_data)}/{len(symbols)} symbols")
        return all_data

    def save_to_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """Save a DataFrame to CSV in the data directory."""
        filepath = self.data_dir / f"{symbol}_daily.csv"
        df.to_csv(filepath)
        logger.info(f"Saved {symbol} data to {filepath}")
        return str(filepath)


if __name__ == "__main__":
    # Quick test
    fetcher = PriceFetcher()
    df = fetcher.get_historical_bars("SPY", days=30)
    print(f"\nSPY last 30 days ({len(df)} bars):")
    print(df.tail())
    print(f"\nColumns: {list(df.columns)}")
