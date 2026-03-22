"""
Technical Indicator Engine — computes 50+ indicators from OHLCV data.

Uses the 'ta' library (Technical Analysis) for indicator calculations.
These features become inputs to our ML models.

Usage:
    from src.data_pipeline.indicator_engine import IndicatorEngine
    engine = IndicatorEngine()
    df_with_indicators = engine.add_all_indicators(price_df)
"""
import pandas as pd
import numpy as np
import ta
from loguru import logger


class IndicatorEngine:
    """Computes technical indicators from OHLCV price data."""

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to a price DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with 50+ additional indicator columns
        """
        df = df.copy()
        logger.info(f"Computing indicators for {len(df)} bars")

        # === Trend Indicators ===
        df = self._add_moving_averages(df)
        df = self._add_macd(df)
        df = self._add_adx(df)
        df = self._add_ichimoku(df)

        # === Momentum Indicators ===
        df = self._add_rsi(df)
        df = self._add_stochastic(df)
        df = self._add_williams_r(df)
        df = self._add_cci(df)
        df = self._add_roc(df)

        # === Volatility Indicators ===
        df = self._add_bollinger_bands(df)
        df = self._add_atr(df)

        # === Volume Indicators ===
        df = self._add_volume_indicators(df)

        # === Price-Derived Features ===
        df = self._add_price_features(df)

        # === Candlestick Pattern Signals ===
        df = self._add_candle_patterns(df)

        orig_cols = 7  # open, high, low, close, volume, trade_count, vwap
        n_indicators = len(df.columns) - orig_cols
        logger.info(f"Added {n_indicators} indicator columns. Total columns: {len(df.columns)}")

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple and Exponential Moving Averages at multiple periods."""
        for period in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{period}"] = ta.trend.sma_indicator(df["close"], window=period)
            df[f"ema_{period}"] = ta.trend.ema_indicator(df["close"], window=period)

        # Moving average crossover signals
        df["sma_cross_20_50"] = (df["sma_20"] > df["sma_50"]).astype(int)
        df["sma_cross_50_200"] = (df["sma_50"] > df["sma_200"]).astype(int)
        df["ema_cross_10_20"] = (df["ema_10"] > df["ema_20"]).astype(int)

        # Price relative to MAs
        df["price_vs_sma_20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        df["price_vs_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]
        df["price_vs_sma_200"] = (df["close"] - df["sma_200"]) / df["sma_200"]

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD — Moving Average Convergence Divergence."""
        macd_obj = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_histogram"] = macd_obj.macd_diff()
        df["macd_cross_signal"] = (df["macd"] > df["macd_signal"]).astype(int)
        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX — Average Directional Index (trend strength)."""
        adx_obj = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx_14"] = adx_obj.adx()
        df["adx_pos_14"] = adx_obj.adx_pos()
        df["adx_neg_14"] = adx_obj.adx_neg()
        return df

    def _add_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ichimoku Cloud components."""
        ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
        df["ichimoku_a"] = ichi.ichimoku_a()
        df["ichimoku_b"] = ichi.ichimoku_b()
        df["ichimoku_base"] = ichi.ichimoku_base_line()
        df["ichimoku_conv"] = ichi.ichimoku_conversion_line()
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI — Relative Strength Index at multiple periods."""
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = ta.momentum.rsi(df["close"], window=period)

        # RSI zones
        df["rsi_14_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_14_oversold"] = (df["rsi_14"] < 30).astype(int)

        return df

    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic Oscillator."""
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        return df

    def _add_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Williams %R."""
        df["williams_r_14"] = ta.momentum.williams_r(
            df["high"], df["low"], df["close"], lbp=14
        )
        return df

    def _add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI — Commodity Channel Index."""
        df["cci_20"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)
        return df

    def _add_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rate of Change at multiple periods."""
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = ta.momentum.roc(df["close"], window=period)
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands."""
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR — Average True Range (volatility)."""
        for period in [7, 14, 21]:
            atr_val = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=period
            )
            df[f"atr_{period}"] = atr_val
            df[f"atr_{period}_pct"] = atr_val / df["close"]
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators."""
        # On-Balance Volume
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

        # Volume SMA
        df["volume_sma_20"] = ta.trend.sma_indicator(df["volume"].astype(float), window=20)
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

        # Accumulation/Distribution
        df["ad_line"] = ta.volume.acc_dist_index(
            df["high"], df["low"], df["close"], df["volume"]
        )

        # Money Flow Index
        df["mfi_14"] = ta.volume.money_flow_index(
            df["high"], df["low"], df["close"], df["volume"], window=14
        )

        # VWAP ratio
        if "vwap" in df.columns:
            df["vwap_ratio"] = df["close"] / df["vwap"].replace(0, np.nan)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-derived features useful for ML."""
        # Daily returns
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["return_20d"] = df["close"].pct_change(20)

        # Volatility (rolling std of returns)
        df["volatility_10d"] = df["return_1d"].rolling(10).std()
        df["volatility_20d"] = df["return_1d"].rolling(20).std()

        # High-Low range
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]
        df["daily_range_avg_10"] = df["daily_range"].rolling(10).mean()

        # Gap (open vs previous close)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Distance from 52-week high/low
        df["high_52w"] = df["high"].rolling(252).max()
        df["low_52w"] = df["low"].rolling(252).min()
        df["pct_from_52w_high"] = (df["close"] - df["high_52w"]) / df["high_52w"]
        df["pct_from_52w_low"] = (df["close"] - df["low_52w"]) / df["low_52w"]

        # Day of week and month (calendar features)
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        return df

    def _add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic candlestick pattern detection."""
        body = abs(df["close"] - df["open"])
        full_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["candle_body_ratio"] = body / full_range

        df["candle_bullish"] = (df["close"] > df["open"]).astype(int)
        df["candle_doji"] = (df["candle_body_ratio"] < 0.1).astype(int)

        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        df["candle_hammer"] = (
            (lower_shadow > 2 * body) & (upper_shadow < body)
        ).astype(int)

        return df


if __name__ == "__main__":
    from src.data_pipeline.price_fetcher import PriceFetcher

    fetcher = PriceFetcher()
    df = fetcher.get_historical_bars("SPY", days=365)

    engine = IndicatorEngine()
    df_indicators = engine.add_all_indicators(df)

    print(f"\nOriginal columns: 7")
    print(f"After indicators: {len(df_indicators.columns)}")
    print(f"\nAll columns:\n{sorted(df_indicators.columns.tolist())}")
