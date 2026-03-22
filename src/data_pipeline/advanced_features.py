"""
Advanced Feature Builder — adds regime and mean-reversion features to the ML pipeline.

This module supplements the existing indicator_engine with specialized features:
  1. Mean-Reversion Features: identify overextension, potential pullbacks
  2. Regime Features: characterize market structure, trend strength, volatility environment

These features are optimized for a ~750 sample dataset (selective feature set).
No look-ahead bias — all calculations use only past/current data.

Usage:
    from src.data_pipeline.advanced_features import AdvancedFeatureBuilder
    builder = AdvancedFeatureBuilder()
    df_advanced = builder.build_advanced_features(df)
    names = builder.get_feature_names()
"""
import pandas as pd
import numpy as np
from loguru import logger


class AdvancedFeatureBuilder:
    """Builds advanced regime and mean-reversion features on top of existing feature set."""

    def __init__(self):
        """Initialize the advanced feature builder."""
        self.logger = logger

    def build_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all advanced features to an existing DataFrame with OHLCV + existing indicators.

        Args:
            df: DataFrame with columns: open, high, low, close, volume, vwap, and existing
                technical indicator columns (from IndicatorEngine). Must be sorted by date.

        Returns:
            DataFrame with all new columns added. Original columns preserved.
            New features will have NaN in their warmup period.
        """
        df = df.copy()
        self.logger.info(f"Building advanced features for {len(df)} rows")

        # Mean-Reversion Features
        df = self._add_zscore_features(df)
        df = self._add_bollinger_features(df)
        df = self._add_rsi_divergence(df)
        df = self._add_mean_reversion_score(df)
        df = self._add_vwap_distance(df)
        df = self._add_overnight_gap_zscore(df)
        df = self._add_intraday_reversal(df)

        # Regime Features
        df = self._add_adx_regime(df)
        df = self._add_trend_strength(df)
        df = self._add_volatility_regime(df)
        df = self._add_vol_of_vol(df)
        df = self._add_correlation_regime(df)
        df = self._add_momentum_regime(df)
        df = self._add_market_breadth_proxy(df)
        df = self._add_regime_change_flag(df)
        df = self._add_hurst_exponent_approx(df)

        n_new = len(self.get_feature_names())
        self.logger.info(f"Added {n_new} advanced feature columns. Total: {len(df.columns)}")

        return df

    def get_feature_names(self) -> list[str]:
        """
        Return the list of feature names added by this builder.

        Returns:
            List of column names for all advanced features.
        """
        mean_reversion = [
            "zscore_20",
            "zscore_50",
            "bb_position",
            "bb_squeeze",
            "rsi_divergence",
            "mean_reversion_score",
            "distance_from_vwap",
            "overnight_gap_zscore",
            "intraday_reversal",
        ]

        regime = [
            "adx_regime",
            "trend_strength",
            "volatility_regime",
            "vol_of_vol",
            "correlation_regime",
            "momentum_regime",
            "market_breadth_proxy",
            "regime_change_flag",
            "hurst_exponent_approx",
        ]

        return mean_reversion + regime

    def get_feature_importance_hints(self) -> dict[str, str]:
        """
        Return a dict mapping feature name to its intended use case.

        Returns:
            Dict with feature names as keys and trading logic descriptions as values.
        """
        return {
            # Mean-Reversion Features
            "zscore_20": "Distance from 20-day mean (std units). Extremes signal reversals.",
            "zscore_50": "Distance from 50-day mean (std units). Intermediate-term extension.",
            "bb_position": "Position within 20-period Bollinger Bands (0=lower, 1=upper). Extremes mean-revert.",
            "bb_squeeze": "Bollinger Band width percentile. Low = potential breakout/compression.",
            "rsi_divergence": "Divergence between RSI trend and price trend (14d). Signals reversal weakness.",
            "mean_reversion_score": "Composite score combining zscore, BB position, and RSI extremes.",
            "distance_from_vwap": "% distance from volume-weighted average price. Gap implies mean reversion.",
            "overnight_gap_zscore": "Z-score of overnight gaps (open vs prev close). Extreme gaps tend to revert.",
            "intraday_reversal": "(close-open)/(high-low). Measures within-bar reversal strength.",

            # Regime Features
            "adx_regime": "ADX smoothed over 5 periods. >25 = strong trend, <20 = ranging.",
            "trend_strength": "Composite trend strength: ADX * |DI+ - DI-| / 100.",
            "volatility_regime": "Current realized vol / 60-day avg vol. >1 = high vol regime.",
            "vol_of_vol": "Std of rolling 10-day volatility over 30 days. Indicates vol clustering.",
            "correlation_regime": "Rolling 20-day autocorrelation. >0.3 = trending, <0 = mean-reverting.",
            "momentum_regime": "SMA20 / SMA50 ratio. >1 = uptrend, <1 = downtrend.",
            "market_breadth_proxy": "Proportion of up days in last 20. 0.5 = neutral, extremes indicate regime.",
            "regime_change_flag": "Binary flag when ADX crosses 20 or 25. Signals regime transition.",
            "hurst_exponent_approx": ">0.5 = trending, <0.5 = mean-reverting. Indicates price structure.",
        }

    # ========== MEAN-REVERSION FEATURES ==========

    def _add_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add z-score features over 20 and 50-day windows.

        Z-score = (price - mean) / std. Measures how far price is from its mean in standard
        deviations. Extremes (>2 or <-2) signal potential mean-reversion opportunities.

        Args:
            df: DataFrame with 'close' column.

        Returns:
            DataFrame with zscore_20 and zscore_50 columns added.
        """
        for period in [20, 50]:
            rolling_mean = df["close"].rolling(period).mean()
            rolling_std = df["close"].rolling(period).std()
            df[f"zscore_{period}"] = (df["close"] - rolling_mean) / rolling_std.replace(0, np.nan)

        return df

    def _add_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Band position and squeeze features.

        bb_position: normalized position within bands (0=lower band, 0.5=middle, 1=upper).
        bb_squeeze: percentile of BB width over 50 periods. Low values suggest compression
                    before a breakout; high values suggest expansion.

        Args:
            df: DataFrame with 'close' column, or pre-computed BB columns from indicator_engine.

        Returns:
            DataFrame with bb_position and bb_squeeze columns added.
        """
        # Use pre-computed Bollinger Bands if available; otherwise compute
        if "bb_upper" in df.columns and "bb_middle" in df.columns and "bb_lower" in df.columns:
            bb_upper = df["bb_upper"]
            bb_middle = df["bb_middle"]
            bb_lower = df["bb_lower"]
        else:
            # Fallback: compute simple BB
            bb_middle = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std

        # Position within bands (bounded [0, 1])
        bb_width = bb_upper - bb_lower
        bb_position = (df["close"] - bb_lower) / bb_width.replace(0, np.nan)
        df["bb_position"] = np.clip(bb_position, 0, 1)

        # Squeeze: percentile of BB width over 50 periods
        bb_widths = bb_upper - bb_lower
        bb_squeeze = bb_widths.rolling(50).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-9) if len(x) == 50 else np.nan,
            raw=False
        )
        df["bb_squeeze"] = bb_squeeze

        return df

    def _add_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI divergence: difference between RSI trend and price trend.

        Bullish divergence: price makes lower lows but RSI makes higher lows (signals reversal up).
        Bearish divergence: price makes higher highs but RSI makes lower highs (signals reversal down).

        Computes: trend of RSI_14 - trend of close, both over 14 periods.

        Args:
            df: DataFrame with 'close' and 'rsi_14' columns.

        Returns:
            DataFrame with rsi_divergence column added.
        """
        if "rsi_14" not in df.columns:
            # Fallback: compute RSI ourselves
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = df["rsi_14"]

        # Trend: simple price change over 14 periods
        price_trend = df["close"].diff(14)
        rsi_trend = rsi.diff(14)

        # Divergence: if price and RSI trends move in opposite directions, it's a divergence signal
        df["rsi_divergence"] = rsi_trend - price_trend.fillna(0)

        return df

    def _add_mean_reversion_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite mean-reversion score.

        Combines:
          - zscore (how extended is the price)
          - bb_position (where within bands)
          - RSI extremes (overbought/oversold)

        Higher score = stronger mean-reversion signal.

        Args:
            df: DataFrame with zscore_20, bb_position, and rsi_14 columns.

        Returns:
            DataFrame with mean_reversion_score column added.
        """
        # Component 1: zscore magnitude (normalized)
        if "zscore_20" in df.columns:
            zscore_component = np.abs(df["zscore_20"]).fillna(0) / 3.0  # 3σ as reference
        else:
            zscore_component = 0

        # Component 2: BB position extremes (farther from middle = higher score)
        if "bb_position" in df.columns:
            bb_component = 1.0 - np.abs(df["bb_position"] - 0.5) / 0.5  # Inverted: extremes get high score
            bb_component = bb_component.fillna(0)
        else:
            bb_component = 0

        # Component 3: RSI extremes (overbought >70 or oversold <30)
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"].fillna(50)
            rsi_component = np.maximum(
                (rsi > 70) * (rsi - 70) / 30,
                (rsi < 30) * (30 - rsi) / 30
            )
        else:
            rsi_component = 0

        # Composite score (weighted average)
        df["mean_reversion_score"] = (zscore_component * 0.4 + bb_component * 0.3 + rsi_component * 0.3)

        return df

    def _add_vwap_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add distance from VWAP as a mean-reversion signal.

        VWAP is the volume-weighted average price. If available, compute % distance.
        If not available, compute a volume-weighted price proxy.

        Args:
            df: DataFrame with 'close', 'volume', and optionally 'vwap' columns.

        Returns:
            DataFrame with distance_from_vwap column added.
        """
        if "vwap" in df.columns and df["vwap"].notna().sum() > 0:
            vwap = df["vwap"]
        else:
            # Compute simple volume-weighted price proxy
            # This is a very rough approximation; a proper VWAP would cumsum vol-weighted prices
            hlc = (df["high"] + df["low"] + df["close"]) / 3
            vwap = (hlc * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()

        distance = ((df["close"] - vwap) / vwap.replace(0, np.nan)) * 100
        df["distance_from_vwap"] = np.clip(distance, -20, 20)  # Clip to reasonable range

        return df

    def _add_overnight_gap_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add z-score of overnight gaps.

        Overnight gap = (open - prev_close) / prev_close. Compute z-score over rolling 20-day window.
        Large gaps (extreme zscore) tend to mean-revert intraday.

        Args:
            df: DataFrame with 'open' and 'close' columns.

        Returns:
            DataFrame with overnight_gap_zscore column added.
        """
        gap = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        gap_mean = gap.rolling(20).mean()
        gap_std = gap.rolling(20).std()

        gap_zscore = (gap - gap_mean) / gap_std.replace(0, np.nan)
        df["overnight_gap_zscore"] = gap_zscore

        return df

    def _add_intraday_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intraday reversal strength.

        Formula: (close - open) / (high - low). Measures how much the bar closed vs its full range.
        Values close to 1 = bullish reversal (opened near low, closed near high).
        Values close to 0 or negative = bearish reversal.
        This identifies bars with strong internal reversals.

        Args:
            df: DataFrame with 'open', 'close', 'high', 'low' columns.

        Returns:
            DataFrame with intraday_reversal column added.
        """
        full_range = df["high"] - df["low"]
        intraday_move = df["close"] - df["open"]

        intraday_reversal = intraday_move / full_range.replace(0, np.nan)
        df["intraday_reversal"] = np.clip(intraday_reversal, -1, 1)

        return df

    # ========== REGIME FEATURES ==========

    def _add_adx_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ADX regime feature (smoothed ADX).

        ADX (Average Directional Index) from indicator_engine measures trend strength.
        >25: strong trend, 20-25: trending, <20: no trend/ranging.

        Smooth it over 5 periods to reduce noise and identify regime shifts.

        Args:
            df: DataFrame with 'adx_14' column (from IndicatorEngine).

        Returns:
            DataFrame with adx_regime column added.
        """
        if "adx_14" in df.columns:
            df["adx_regime"] = df["adx_14"].rolling(5).mean()
        else:
            # Fallback: use a dummy feature
            df["adx_regime"] = np.nan

        return df

    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite trend strength.

        Combines ADX with directional indicator spread:
        trend_strength = ADX * |DI+ - DI-| / 100

        Higher values = stronger trend in a clear direction.

        Args:
            df: DataFrame with 'adx_14', 'adx_pos_14', 'adx_neg_14' columns.

        Returns:
            DataFrame with trend_strength column added.
        """
        if "adx_14" in df.columns and "adx_pos_14" in df.columns and "adx_neg_14" in df.columns:
            adx = df["adx_14"].fillna(0)
            di_spread = np.abs(df["adx_pos_14"] - df["adx_neg_14"]).fillna(0)
            df["trend_strength"] = (adx * di_spread) / 100
        else:
            df["trend_strength"] = np.nan

        return df

    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility regime.

        volatility_regime = current realized vol / 60-day average vol.
        >1: elevated volatility, <1: low volatility.

        Helps identify when the market is quiet vs. choppy.

        Args:
            df: DataFrame with 'return_1d' column (from IndicatorEngine).

        Returns:
            DataFrame with volatility_regime column added.
        """
        if "return_1d" not in df.columns:
            # Compute returns
            df["return_1d"] = df["close"].pct_change(1)

        current_vol = df["return_1d"].rolling(10).std()
        avg_vol = df["return_1d"].rolling(60).std()

        volatility_regime = current_vol / avg_vol.replace(0, np.nan)
        df["volatility_regime"] = volatility_regime

        return df

    def _add_vol_of_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add vol-of-vol: volatility of volatility.

        Measures the variability of volatility over a 30-day window.
        High vol-of-vol = uncertain/choppy regime; low = stable vol environment.

        Args:
            df: DataFrame with 'return_1d' column.

        Returns:
            DataFrame with vol_of_vol column added.
        """
        if "return_1d" not in df.columns:
            df["return_1d"] = df["close"].pct_change(1)

        rolling_vols = df["return_1d"].rolling(10).std()
        vol_of_vol = rolling_vols.rolling(30).std()

        df["vol_of_vol"] = vol_of_vol

        return df

    def _add_correlation_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add correlation regime using autocorrelation of returns.

        Autocorrelation of 1-day returns measures mean-reversion vs. trending.
        >0.3: strong trending (momentum), <0: mean-reverting, ~0: random walk.

        Args:
            df: DataFrame with 'return_1d' column.

        Returns:
            DataFrame with correlation_regime column added.
        """
        if "return_1d" not in df.columns:
            df["return_1d"] = df["close"].pct_change(1)

        returns = df["return_1d"]

        # Compute rolling 20-day autocorrelation
        def rolling_autocorr(x):
            if len(x) < 2 or x.std() == 0:
                return np.nan
            return x.autocorr()

        correlation_regime = returns.rolling(20).apply(rolling_autocorr, raw=False)
        df["correlation_regime"] = correlation_regime

        return df

    def _add_momentum_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum regime indicator.

        momentum_regime = SMA20 / SMA50.
        >1: 20-period MA above 50-period (uptrend/bullish), <1: downtrend.

        Values near 1.0 = ranging; values farther from 1.0 = stronger momentum.

        Args:
            df: DataFrame with 'sma_20' and 'sma_50' columns.

        Returns:
            DataFrame with momentum_regime column added.
        """
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["momentum_regime"] = df["sma_20"] / df["sma_50"].replace(0, np.nan)
        else:
            # Fallback: compute SMAs
            sma20 = df["close"].rolling(20).mean()
            sma50 = df["close"].rolling(50).mean()
            df["momentum_regime"] = sma20 / sma50.replace(0, np.nan)

        return df

    def _add_market_breadth_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market breadth proxy.

        Computes the proportion of up days in the last 20 days.
        0.5 = neutral, >0.5 = more up days (bullish), <0.5 = more down days (bearish).

        Args:
            df: DataFrame with 'close' column.

        Returns:
            DataFrame with market_breadth_proxy column added.
        """
        returns = df["close"].pct_change()
        up_days = (returns > 0).astype(int)

        breadth = up_days.rolling(20).sum() / 20.0
        df["market_breadth_proxy"] = breadth

        return df

    def _add_regime_change_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime change flag.

        Binary flag that signals when ADX crosses certain thresholds (20 or 25),
        which indicates a transition from ranging to trending or vice versa.

        This is useful for detecting when the regime has just shifted.

        Args:
            df: DataFrame with 'adx_14' column.

        Returns:
            DataFrame with regime_change_flag column added (1 = transition detected, 0 = no).
        """
        if "adx_14" not in df.columns:
            df["regime_change_flag"] = 0
            return df

        adx = df["adx_14"].fillna(0)

        # Check if ADX crosses the 20 or 25 threshold
        cross_up_20 = (adx.shift(1) <= 20) & (adx > 20)
        cross_down_20 = (adx.shift(1) > 20) & (adx <= 20)
        cross_up_25 = (adx.shift(1) <= 25) & (adx > 25)
        cross_down_25 = (adx.shift(1) > 25) & (adx <= 25)

        regime_flag = (cross_up_20 | cross_down_20 | cross_up_25 | cross_down_25).astype(int)
        df["regime_change_flag"] = regime_flag

        return df

    def _add_hurst_exponent_approx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add approximate Hurst exponent.

        The Hurst exponent characterizes price behavior:
        H > 0.5: trending (persistence), H < 0.5: mean-reverting, H ≈ 0.5: random walk.

        This is an approximation using rescaled range analysis over 50-period windows.

        Args:
            df: DataFrame with 'close' column.

        Returns:
            DataFrame with hurst_exponent_approx column added.
        """
        def hurst_approx(ts):
            """Approximate Hurst exponent for a time series."""
            if len(ts) < 10:
                return np.nan

            # Calculate log returns
            returns = np.log(ts / ts.shift(1)).dropna().values

            if len(returns) < 2:
                return np.nan

            # Simple approximation: rescaled range
            mean_ret = np.mean(returns)
            Y = np.cumsum(returns - mean_ret)

            R = np.max(Y) - np.min(Y)
            S = np.std(returns, ddof=1)

            if S == 0:
                return np.nan

            # Very rough approximation of Hurst exponent
            hurst = np.log(R / S) / np.log(len(returns))
            return np.clip(hurst, 0.2, 0.8)  # Reasonable bounds

        hurst = df["close"].rolling(50).apply(hurst_approx, raw=False)
        df["hurst_exponent_approx"] = hurst

        return df


if __name__ == "__main__":
    # Quick test: load a sample DataFrame and build advanced features
    print("Advanced Feature Builder - Unit Test")
    print("=" * 50)

    # Create dummy data for testing
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    dummy_data = {
        "open": np.random.uniform(100, 105, 100),
        "high": np.random.uniform(105, 110, 100),
        "low": np.random.uniform(95, 100, 100),
        "close": np.random.uniform(100, 105, 100),
        "volume": np.random.uniform(1e6, 2e6, 100),
        "trade_count": np.random.uniform(1000, 2000, 100),
        "vwap": np.random.uniform(100, 105, 100),
    }
    df_test = pd.DataFrame(dummy_data, index=dates)

    # Add dummy indicators
    df_test["rsi_14"] = np.random.uniform(30, 70, 100)
    df_test["adx_14"] = np.random.uniform(15, 35, 100)
    df_test["adx_pos_14"] = np.random.uniform(10, 30, 100)
    df_test["adx_neg_14"] = np.random.uniform(10, 30, 100)
    df_test["sma_20"] = df_test["close"].rolling(20).mean()
    df_test["sma_50"] = df_test["close"].rolling(50).mean()
    df_test["bb_upper"] = df_test["close"].rolling(20).mean() + 2 * df_test["close"].rolling(20).std()
    df_test["bb_middle"] = df_test["close"].rolling(20).mean()
    df_test["bb_lower"] = df_test["close"].rolling(20).mean() - 2 * df_test["close"].rolling(20).std()
    df_test["return_1d"] = df_test["close"].pct_change()

    # Build advanced features
    builder = AdvancedFeatureBuilder()
    df_advanced = builder.build_advanced_features(df_test)

    print(f"\nOriginal shape: {df_test.shape}")
    print(f"Advanced shape: {df_advanced.shape}")
    print(f"New features added: {len(builder.get_feature_names())}")
    print(f"\nNew feature names:")
    for name in builder.get_feature_names():
        print(f"  - {name}")

    print(f"\nFeature importance hints:")
    for name, desc in list(builder.get_feature_importance_hints().items())[:3]:
        print(f"  {name}: {desc[:70]}...")

    print("\nSample statistics for new features:")
    new_features = builder.get_feature_names()
    for feat in new_features[:3]:
        if feat in df_advanced.columns:
            print(f"  {feat}: mean={df_advanced[feat].mean():.4f}, std={df_advanced[feat].std():.4f}")
