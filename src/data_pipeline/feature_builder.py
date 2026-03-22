"""
Feature Builder — merges all data sources into a single ML-ready dataset.

This is the central piece that combines:
  1. Price data + technical indicators (from Alpaca)
  2. Economic indicators (from FRED)
  3. News sentiment scores (from Finnhub + FinBERT)

The output is a single DataFrame per symbol with all features aligned by date,
ready to be fed into the ML models.

Usage:
    from src.data_pipeline.feature_builder import FeatureBuilder
    builder = FeatureBuilder()
    df = builder.build_features("AAPL", days=365*3)
    bulk = builder.build_all_features(days=365*3)
"""
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_all_tickers, DATA_DIR
from src.data_pipeline.price_fetcher import PriceFetcher
from src.data_pipeline.indicator_engine import IndicatorEngine
from src.data_pipeline.economic_fetcher import EconomicFetcher
from src.data_pipeline.news_fetcher import NewsFetcher
from src.data_pipeline.event_calendar import EventCalendar


class FeatureBuilder:
    """Builds ML-ready feature datasets by merging all data sources."""

    def __init__(self, use_finbert: bool = True):
        self.price_fetcher = PriceFetcher()
        self.indicator_engine = IndicatorEngine()
        self.economic_fetcher = EconomicFetcher()
        self.news_fetcher = NewsFetcher(use_finbert=use_finbert)
        self.event_calendar = EventCalendar()
        self.data_dir = DATA_DIR / "processed"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("FeatureBuilder initialized (with event calendar)")

    def build_features(
        self,
        symbol: str,
        days: int = 365 * 3,
        include_news: bool = True,
        news_lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Build complete feature set for a single symbol.

        Args:
            symbol: Stock ticker
            days: Days of price history
            include_news: Whether to include news sentiment features
            news_lookback_days: Days of news to fetch (API rate limits)

        Returns:
            DataFrame with all features, indexed by date
        """
        logger.info(f"Building features for {symbol} ({days} days)")

        # --- 1. Price data ---
        price_df = self.price_fetcher.get_historical_bars(symbol, days=days)
        if price_df.empty:
            logger.error(f"No price data for {symbol}, skipping")
            return pd.DataFrame()

        # --- 2. Technical indicators ---
        df = self.indicator_engine.add_all_indicators(price_df)

        # --- 3. Economic indicators ---
        econ_df = self.economic_fetcher.get_all_indicators(days=days)
        if not econ_df.empty:
            # Merge on date (left join to keep all trading days)
            econ_df.index = pd.to_datetime(econ_df.index)
            df = df.merge(econ_df, left_index=True, right_index=True, how="left")
            # Forward fill economic data (reported less frequently)
            econ_cols = econ_df.columns.tolist()
            df[econ_cols] = df[econ_cols].ffill()

        # --- 4. News sentiment ---
        if include_news:
            try:
                news_df = self.news_fetcher.get_daily_sentiment_summary(
                    symbol, days=news_lookback_days
                )
                if not news_df.empty:
                    news_df.index = pd.to_datetime(news_df.index)
                    df = df.merge(news_df, left_index=True, right_index=True, how="left")
                    # Fill days without news with neutral values
                    news_cols = news_df.columns.tolist()
                    df["news_count"] = df.get("news_count", 0).fillna(0)
                    for col in news_cols:
                        if col != "news_count":
                            df[col] = df[col].fillna(0)
            except Exception as e:
                logger.warning(f"News sentiment failed for {symbol}: {e}")

        # --- 5. Event calendar features ---
        try:
            events_df = self.event_calendar.encode_events(df.index, symbol)
            if not events_df.empty:
                df = df.merge(events_df, left_index=True, right_index=True, how="left")
                df = df.fillna(0)
                logger.info(f"  Added {len(events_df.columns)} event calendar features")
        except Exception as e:
            logger.warning(f"Event calendar failed for {symbol}: {e}")

        # --- 6. Add target variable ---
        df = self._add_target(df)

        # --- 7. Clean up ---
        # Drop rows with too many NaN values (early rows before indicators warm up)
        min_valid = len(df.columns) * 0.5  # Require at least 50% non-null
        df = df.dropna(thresh=int(min_valid))

        logger.info(
            f"{symbol} features: {len(df)} rows × {len(df.columns)} columns "
            f"({df.index[0].date()} → {df.index[-1].date()})"
        )

        # Save to disk
        filepath = self.data_dir / f"{symbol}_features.csv"
        df.to_csv(filepath)
        logger.info(f"Saved to {filepath}")

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add the prediction target: next-day price direction.

        target_direction: 1 if next day's close > today's close, else 0
        target_return: Next day's percentage return
        """
        # Defragment the DataFrame first to avoid PerformanceWarning
        df = df.copy()

        df["target_return"] = df["close"].shift(-1) / df["close"] - 1
        df["target_direction"] = (df["target_return"] > 0).astype(int)

        # Also add multi-day targets for longer-horizon predictions
        df["target_return_5d"] = df["close"].shift(-5) / df["close"] - 1
        df["target_direction_5d"] = (df["target_return_5d"] > 0).astype(int)

        return df

    def build_all_features(
        self,
        symbols: list[str] = None,
        days: int = 365 * 3,
        include_news: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Build features for all symbols in the watchlist.

        Args:
            symbols: List of tickers (None = all from config)
            days: Days of history
            include_news: Include sentiment features

        Returns:
            Dict mapping symbol → feature DataFrame
        """
        if symbols is None:
            symbols = get_all_tickers()

        logger.info(f"Building features for {len(symbols)} symbols")

        # Fetch economic data once (shared across all symbols)
        # This is already cached in the economic fetcher
        all_features = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Processing {symbol}")
            try:
                df = self.build_features(
                    symbol,
                    days=days,
                    include_news=include_news,
                    news_lookback_days=min(30, days),
                )
                if not df.empty:
                    all_features[symbol] = df
            except Exception as e:
                logger.error(f"Failed to build features for {symbol}: {e}")

        logger.info(f"Built features for {len(all_features)}/{len(symbols)} symbols")
        return all_features

    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """Get a summary of the feature dataset for debugging."""
        return {
            "shape": df.shape,
            "date_range": f"{df.index[0].date()} → {df.index[-1].date()}",
            "n_features": len(df.columns),
            "null_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
            "target_balance": df["target_direction"].value_counts().to_dict()
            if "target_direction" in df.columns
            else None,
        }


if __name__ == "__main__":
    # Quick test — build features for SPY with no FinBERT (faster)
    builder = FeatureBuilder(use_finbert=False)

    df = builder.build_features("SPY", days=365, include_news=False)
    print(f"\nFeature set for SPY:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns ({len(df.columns)}):")
    for col in sorted(df.columns):
        null_pct = df[col].isnull().mean() * 100
        print(f"    {col:<40} {null_pct:.1f}% null")

    if "target_direction" in df.columns:
        print(f"\n  Target balance:")
        print(f"    Up days: {(df['target_direction'] == 1).sum()}")
        print(f"    Down days: {(df['target_direction'] == 0).sum()}")
