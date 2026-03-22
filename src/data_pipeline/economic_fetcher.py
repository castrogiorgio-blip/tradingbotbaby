"""
Economic Data Fetcher — pulls macroeconomic indicators from FRED (Federal Reserve).

These indicators (interest rates, CPI, unemployment, etc.) are important
features for the ML model because they influence market direction.

Usage:
    from src.data_pipeline.economic_fetcher import EconomicFetcher
    fetcher = EconomicFetcher()
    df = fetcher.get_all_indicators(days=365*3)
"""
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_api_keys, get_tickers, DATA_DIR


class EconomicFetcher:
    """Fetches macroeconomic indicator data from FRED."""

    def __init__(self):
        keys = get_api_keys()
        self.fred = Fred(api_key=keys["fred_api_key"])
        self.data_dir = DATA_DIR / "raw" / "economic"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load indicator list from config
        tickers_config = get_tickers()
        self.indicators = tickers_config.get("economic_indicators", [])
        logger.info(f"EconomicFetcher initialized with {len(self.indicators)} indicators")

    # Human-readable names for FRED series
    SERIES_NAMES = {
        "FEDFUNDS": "fed_funds_rate",
        "CPIAUCSL": "cpi",
        "UNRATE": "unemployment_rate",
        "GDP": "gdp",
        "T10Y2Y": "yield_spread_10y2y",
        "VIXCLS": "vix",
        "DGS10": "treasury_10y",
        "DTWEXBGS": "dollar_index",
        "UMCSENT": "consumer_sentiment",
        "INDPRO": "industrial_production",
    }

    def get_single_indicator(
        self,
        series_id: str,
        days: int = 365 * 3,
    ) -> pd.Series:
        """
        Fetch a single FRED series.

        Args:
            series_id: FRED series ID (e.g., "FEDFUNDS")
            days: Number of days of history

        Returns:
            pandas Series with datetime index
        """
        start = datetime.now() - timedelta(days=days)
        name = self.SERIES_NAMES.get(series_id, series_id.lower())

        try:
            data = self.fred.get_series(series_id, observation_start=start)
            data.name = name
            logger.info(f"Fetched {series_id} ({name}): {len(data)} observations")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return pd.Series(name=name, dtype=float)

    def get_all_indicators(self, days: int = 365 * 3) -> pd.DataFrame:
        """
        Fetch all configured economic indicators and merge into one DataFrame.

        Economic data comes at different frequencies (daily, monthly, quarterly).
        We forward-fill to align everything to a daily frequency.

        Args:
            days: Number of days of history

        Returns:
            DataFrame with daily frequency, one column per indicator
        """
        logger.info(f"Fetching {len(self.indicators)} economic indicators")

        all_series = {}
        for series_id in self.indicators:
            series = self.get_single_indicator(series_id, days=days)
            if not series.empty:
                all_series[series.name] = series

        if not all_series:
            logger.warning("No economic data fetched")
            return pd.DataFrame()

        # Merge all series into one DataFrame
        df = pd.DataFrame(all_series)

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Resample to daily frequency and forward-fill
        # (economic data is often monthly/quarterly — ffill carries the last known value)
        df = df.resample("D").last()
        df = df.ffill()

        # Drop weekends (no trading)
        df = df[df.index.dayofweek < 5]

        # Compute derived features
        df = self._add_derived_features(df)

        logger.info(f"Economic data: {len(df)} days, {len(df.columns)} columns")

        # Save to disk
        filepath = self.data_dir / "economic_indicators.csv"
        df.to_csv(filepath)
        logger.info(f"Saved economic data to {filepath}")

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed features from economic indicators."""
        # Rate of change for each indicator
        for col in df.columns:
            if df[col].notna().sum() > 20:
                df[f"{col}_change_1m"] = df[col].pct_change(20)  # ~1 month
                df[f"{col}_change_3m"] = df[col].pct_change(63)  # ~3 months

        # Yield curve inversion signal (recession predictor)
        if "yield_spread_10y2y" in df.columns:
            df["yield_curve_inverted"] = (df["yield_spread_10y2y"] < 0).astype(int)

        # VIX zones
        if "vix" in df.columns:
            df["vix_high"] = (df["vix"] > 25).astype(int)      # Elevated fear
            df["vix_extreme"] = (df["vix"] > 35).astype(int)    # Panic

        return df

    def get_latest_values(self) -> dict:
        """Get the most recent value for each indicator."""
        latest = {}
        for series_id in self.indicators:
            name = self.SERIES_NAMES.get(series_id, series_id.lower())
            try:
                data = self.fred.get_series(series_id)
                if not data.empty:
                    latest[name] = {
                        "value": float(data.iloc[-1]),
                        "date": str(data.index[-1].date()),
                    }
            except Exception as e:
                logger.error(f"Failed to get latest {series_id}: {e}")

        return latest


if __name__ == "__main__":
    fetcher = EconomicFetcher()

    # Fetch all indicators
    df = fetcher.get_all_indicators(days=365)
    print(f"\nEconomic data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLatest values:")
    print(df.tail(3))

    # Show latest snapshot
    print("\n--- Latest indicator values ---")
    latest = fetcher.get_latest_values()
    for name, info in latest.items():
        print(f"  {name}: {info['value']:.2f} (as of {info['date']})")
