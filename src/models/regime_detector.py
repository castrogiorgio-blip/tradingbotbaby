"""
Market Regime Detection System

This module classifies market regimes and filters trades accordingly.
It uses technical indicators to identify:
  - TRENDING_UP: strong uptrends
  - TRENDING_DOWN: strong downtrends
  - MEAN_REVERTING: low-volatility, oscillating market
  - HIGH_VOLATILITY: elevated uncertainty

Trade sizing is adjusted based on detected regime to align strategies with market conditions.

Usage:
    from src.models.regime_detector import RegimeDetector, AllocationFilter
    detector = RegimeDetector()
    regime, confidence = detector.detect_regime(df)
    allocation = filter.filter_signal(signal, regime, confidence)
"""

from enum import Enum
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    MEAN_REVERTING = "MEAN_REVERTING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    Detects market regime based on technical indicators.

    Classifies the market into one of four regimes based on trend strength (ADX),
    direction (DI+/DI-), moving average alignment, volatility, and price patterns.
    """

    # Thresholds for regime classification
    ADX_STRONG_TREND = 25        # ADX above this indicates strong trend
    ADX_WEAK_TREND = 20          # ADX below this indicates weak/ranging
    RSI_MEAN_REV_MIN = 40        # RSI lower bound for mean reversion
    RSI_MEAN_REV_MAX = 60        # RSI upper bound for mean reversion
    VIX_HIGH_VOL = 25            # VIX above this indicates high volatility
    VOL_EXPANSION_THRESHOLD = 1.5  # Realized vol / 60-day avg ratio
    BB_CONTRACTION_THRESHOLD = 0.3  # Bollinger Band width threshold

    def __init__(self):
        """Initialize regime detector with default thresholds."""
        pass

    def detect_regime(
        self,
        df: pd.DataFrame
    ) -> Tuple[MarketRegime, float]:
        """
        Detect the current market regime from the latest bar.

        Args:
            df: DataFrame with OHLCV and technical indicators.
               Must contain: close, sma_50, sma_200, adx_14, adx_pos_14, adx_neg_14,
                           rsi_14, bb_width, atr_14, volatility_20d

        Returns:
            Tuple of (regime, confidence) where confidence is 0-1.
        """
        if df.empty or len(df) < 1:
            return MarketRegime.UNKNOWN, 0.0

        # Get latest values
        row = df.iloc[-1]

        # Check required columns
        required = ["close", "sma_50", "sma_200", "adx_14", "adx_pos_14", "adx_neg_14"]
        if not all(col in df.columns for col in required):
            return MarketRegime.UNKNOWN, 0.0

        # Fill missing indicators with safe defaults
        adx = row.get("adx_14", 0)
        di_pos = row.get("adx_pos_14", 0)
        di_neg = row.get("adx_neg_14", 0)
        rsi = row.get("rsi_14", 50)
        bb_width = row.get("bb_width", 0)
        atr = row.get("atr_14", 0)
        close = row.get("close", 0)
        sma_50 = row.get("sma_50", 0)
        sma_200 = row.get("sma_200", 0)

        # Calculate volatility ratio (realized vol vs 60-day avg)
        vol_ratio = self._calculate_volatility_ratio(df)

        # Detect high volatility first (overrides other regimes)
        if self._is_high_volatility(df, row, vol_ratio, atr):
            confidence = self._calculate_volatility_confidence(df, row, vol_ratio, atr)
            return MarketRegime.HIGH_VOLATILITY, confidence

        # Check for trending regimes
        if adx > self.ADX_STRONG_TREND:
            if di_pos > di_neg and sma_50 > sma_200:
                # Uptrend confirmation
                if close > sma_50:
                    confidence = self._calculate_uptrend_confidence(df, row, adx, di_pos, di_neg)
                    return MarketRegime.TRENDING_UP, confidence

            elif di_neg > di_pos and sma_50 < sma_200:
                # Downtrend confirmation
                if close < sma_50:
                    confidence = self._calculate_downtrend_confidence(df, row, adx, di_neg, di_pos)
                    return MarketRegime.TRENDING_DOWN, confidence

        # Check for mean-reverting regime
        if adx < self.ADX_WEAK_TREND:
            if self.RSI_MEAN_REV_MIN < rsi < self.RSI_MEAN_REV_MAX:
                if bb_width < self.BB_CONTRACTION_THRESHOLD and bb_width > 0:
                    confidence = self._calculate_mean_reversion_confidence(df, row, adx, rsi, bb_width)
                    return MarketRegime.MEAN_REVERTING, confidence

        # Default to unknown
        return MarketRegime.UNKNOWN, 0.0

    def get_regime_history(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute regime for each bar in the DataFrame.

        Args:
            df: DataFrame with indicators

        Returns:
            Series with regime labels, indexed by date
        """
        regimes = []

        for i in range(len(df)):
            subset = df.iloc[:i+1]
            regime, _ = self.detect_regime(subset)
            regimes.append(regime.value)

        return pd.Series(regimes, index=df.index, name="regime")

    def filter_signal(
        self,
        signal: float,
        regime: MarketRegime,
        confidence: float
    ) -> Dict[str, float]:
        """
        Adjust signal strength and position sizing based on regime.

        Args:
            signal: Original signal strength (typically 0-1)
            regime: Detected market regime
            confidence: Regime detection confidence (0-1)

        Returns:
            Dict with keys:
              - 'adjusted_signal': modified signal strength
              - 'position_size_multiplier': how much to scale position
              - 'should_trade': boolean, whether to take the signal
              - 'reason': string explaining the adjustment
        """
        # Convert string signals to numeric (-1 to 1)
        if isinstance(signal, str):
            signal_map = {"BUY_CALL": 0.6, "BUY_PUT": -0.6, "HOLD": 0.0}
            signal = signal_map.get(signal, 0.0)

        # Base position sizing and signal filtering
        if regime == MarketRegime.TRENDING_UP:
            # Prefer trend-following signals
            if signal > 0:  # Bullish signal
                return {
                    'adjusted_signal': signal,
                    'position_size_multiplier': 1.0,
                    'should_trade': True,
                    'reason': 'Uptrend regime: full position for long signals'
                }
            else:  # Bearish/mean-reversion signal
                return {
                    'adjusted_signal': signal * 0.5,
                    'position_size_multiplier': 0.3,
                    'should_trade': signal < -0.3,
                    'reason': 'Uptrend regime: skip mean-reversion, reduce shorts'
                }

        elif regime == MarketRegime.TRENDING_DOWN:
            # Prefer trend-following signals
            if signal < 0:  # Bearish signal
                return {
                    'adjusted_signal': signal,
                    'position_size_multiplier': 1.0,
                    'should_trade': True,
                    'reason': 'Downtrend regime: full position for short signals'
                }
            else:  # Bullish/mean-reversion signal
                return {
                    'adjusted_signal': signal * 0.5,
                    'position_size_multiplier': 0.3,
                    'should_trade': signal > 0.3,
                    'reason': 'Downtrend regime: skip mean-reversion, reduce longs'
                }

        elif regime == MarketRegime.MEAN_REVERTING:
            # Prefer mean-reversion signals (smaller moves in both directions)
            abs_signal = abs(signal)
            if 0.3 < abs_signal < 0.8:  # Moderate mean-reversion signals
                return {
                    'adjusted_signal': signal,
                    'position_size_multiplier': 1.0,
                    'should_trade': True,
                    'reason': 'Mean-reversion regime: full position for oscillator signals'
                }
            elif abs_signal > 0.8:  # Strong trend signal in ranging market
                return {
                    'adjusted_signal': signal * 0.6,
                    'position_size_multiplier': 0.4,
                    'should_trade': confidence > 0.6,
                    'reason': 'Mean-reversion regime: reduce trend signals'
                }
            else:  # Weak signals
                return {
                    'adjusted_signal': signal * 0.3,
                    'position_size_multiplier': 0.2,
                    'should_trade': False,
                    'reason': 'Mean-reversion regime: signal too weak'
                }

        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Reduce all positions, only high-confidence signals
            if confidence < 0.4:
                return {
                    'adjusted_signal': signal * 0.3,
                    'position_size_multiplier': 0.0,
                    'should_trade': False,
                    'reason': 'High volatility: regime confidence too low'
                }

            if abs(signal) > 0.5:  # Only take high-confidence signals
                return {
                    'adjusted_signal': signal * 0.8,
                    'position_size_multiplier': 0.5,
                    'should_trade': True,
                    'reason': 'High volatility: 50% position size for strong signals only'
                }
            else:
                return {
                    'adjusted_signal': signal * 0.5,
                    'position_size_multiplier': 0.25,
                    'should_trade': abs(signal) > 0.4,
                    'reason': 'High volatility: reduced sizing'
                }

        else:  # UNKNOWN regime
            # Conservative: use default 70% sizing
            return {
                'adjusted_signal': signal,
                'position_size_multiplier': 0.7,
                'should_trade': abs(signal) > 0.3,
                'reason': 'Unknown regime: conservative 70% default sizing'
            }

    def get_regime_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compute summary statistics for each regime period in the data.

        Args:
            df: DataFrame with OHLC data and indicators

        Returns:
            Dict mapping regime names to statistics dicts with:
              - 'count': number of bars in regime
              - 'pct_of_total': percentage of total bars
              - 'avg_return': average daily return
              - 'volatility': realized volatility
              - 'max_drawdown': max drawdown during regime
              - 'sharpe_ratio': Sharpe ratio (annualized)
        """
        regime_series = self.get_regime_history(df)

        # Calculate daily returns
        returns = df['close'].pct_change()

        stats = {}

        for regime in MarketRegime:
            if regime == MarketRegime.UNKNOWN:
                continue

            mask = regime_series == regime.value
            if not mask.any():
                continue

            regime_returns = returns[mask].dropna()

            if len(regime_returns) == 0:
                continue

            # Basic stats
            count = mask.sum()
            pct = count / len(df) * 100

            # Return metrics
            avg_ret = regime_returns.mean()
            volatility = regime_returns.std()

            # Sharpe ratio (annualized, assuming 252 trading days)
            if volatility > 0:
                sharpe = (avg_ret / volatility) * np.sqrt(252)
            else:
                sharpe = 0

            # Max drawdown during regime
            cumulative = (1 + regime_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()

            stats[regime.value] = {
                'count': int(count),
                'pct_of_total': float(pct),
                'avg_daily_return': float(avg_ret),
                'volatility': float(volatility),
                'max_drawdown': float(max_dd),
                'sharpe_ratio': float(sharpe),
            }

        return stats

    # ========== Helper Methods ==========

    def _calculate_volatility_ratio(self, df: pd.DataFrame) -> float:
        """Calculate realized volatility ratio (current / 60-day average)."""
        if len(df) < 60:
            return 1.0

        returns = df['close'].pct_change().dropna()
        current_vol = returns.iloc[-20:].std()  # Last 20 days
        long_vol = returns.iloc[-60:].std()      # Last 60 days

        if long_vol == 0:
            return 1.0

        return current_vol / long_vol

    def _is_high_volatility(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        vol_ratio: float,
        atr: float
    ) -> bool:
        """Check if market exhibits high volatility characteristics."""
        # Check realized vol expansion
        if vol_ratio > self.VOL_EXPANSION_THRESHOLD:
            return True

        # Check ATR expansion (ATR increasing relative to 20-period average)
        if 'atr_14' in df.columns and len(df) > 20:
            atr_avg = df['atr_14'].iloc[-20:-1].mean()
            if atr_avg > 0 and atr / atr_avg > 1.3:
                return True

        return False

    def _calculate_volatility_confidence(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        vol_ratio: float,
        atr: float
    ) -> float:
        """Calculate confidence of high volatility regime."""
        confidence = 0.0

        # Realized vol expansion
        if vol_ratio > self.VOL_EXPANSION_THRESHOLD:
            confidence += min(0.6, vol_ratio / 2.0)

        # ATR expansion
        if 'atr_14' in df.columns and len(df) > 20:
            atr_avg = df['atr_14'].iloc[-20:-1].mean()
            if atr_avg > 0:
                atr_ratio = atr / atr_avg
                if atr_ratio > 1.3:
                    confidence += min(0.4, (atr_ratio - 1.0) * 0.5)

        return min(1.0, confidence)

    def _calculate_uptrend_confidence(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        adx: float,
        di_pos: float,
        di_neg: float
    ) -> float:
        """Calculate confidence of uptrend regime."""
        confidence = 0.0

        # ADX strength (normalized 0-1)
        confidence += min(1.0, (adx - self.ADX_STRONG_TREND) / 30)

        # DI+ vs DI- spread
        if di_pos + di_neg > 0:
            di_ratio = di_pos / (di_pos + di_neg)
            confidence += (di_ratio - 0.5) * 2  # Range 0-1

        # Price position relative to SMA
        if 'sma_50' in df.columns and row.get('sma_50', 0) > 0:
            close = row.get('close', 0)
            sma_50 = row.get('sma_50', 0)
            if close > sma_50:
                pct_above = (close - sma_50) / sma_50
                confidence += min(0.3, pct_above * 10)

        return min(1.0, confidence)

    def _calculate_downtrend_confidence(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        adx: float,
        di_neg: float,
        di_pos: float
    ) -> float:
        """Calculate confidence of downtrend regime."""
        confidence = 0.0

        # ADX strength
        confidence += min(1.0, (adx - self.ADX_STRONG_TREND) / 30)

        # DI- vs DI+ spread
        if di_neg + di_pos > 0:
            di_ratio = di_neg / (di_neg + di_pos)
            confidence += (di_ratio - 0.5) * 2

        # Price position relative to SMA
        if 'sma_50' in df.columns and row.get('sma_50', 0) > 0:
            close = row.get('close', 0)
            sma_50 = row.get('sma_50', 0)
            if close < sma_50:
                pct_below = (sma_50 - close) / sma_50
                confidence += min(0.3, pct_below * 10)

        return min(1.0, confidence)

    def _calculate_mean_reversion_confidence(
        self,
        df: pd.DataFrame,
        row: pd.Series,
        adx: float,
        rsi: float,
        bb_width: float
    ) -> float:
        """Calculate confidence of mean-reversion regime."""
        confidence = 0.0

        # Low ADX indicates weak trend
        if adx < self.ADX_WEAK_TREND:
            confidence += (self.ADX_WEAK_TREND - adx) / self.ADX_WEAK_TREND * 0.4

        # RSI in neutral zone
        rsi_distance = min(rsi - self.RSI_MEAN_REV_MIN, self.RSI_MEAN_REV_MAX - rsi)
        rsi_neutral = rsi_distance / ((self.RSI_MEAN_REV_MAX - self.RSI_MEAN_REV_MIN) / 2)
        confidence += rsi_neutral * 0.35

        # Bollinger Band contraction
        if bb_width > 0 and bb_width < self.BB_CONTRACTION_THRESHOLD:
            confidence += (self.BB_CONTRACTION_THRESHOLD - bb_width) / self.BB_CONTRACTION_THRESHOLD * 0.25

        return min(1.0, confidence)


class AllocationFilter:
    """
    Decides trade sizing and filtering based on market regime and signal strength.

    Wraps RegimeDetector to provide a simple interface for position sizing.
    """

    def __init__(self, regime_detector: Optional[RegimeDetector] = None):
        """
        Initialize allocation filter.

        Args:
            regime_detector: RegimeDetector instance (creates new one if None)
        """
        self.detector = regime_detector or RegimeDetector()

    def get_position_size(
        self,
        df: pd.DataFrame,
        signal: float,
        base_size: float = 1.0
    ) -> float:
        """
        Determine position size (as fraction of base_size) given current regime.

        Args:
            df: OHLCV DataFrame with indicators
            signal: Signal strength (-1 to 1)
            base_size: Reference position size

        Returns:
            Recommended position size (0 to base_size)
        """
        regime, confidence = self.detector.detect_regime(df)
        adjusted = self.detector.filter_signal(signal, regime, confidence)

        if not adjusted['should_trade']:
            return 0.0

        return base_size * adjusted['position_size_multiplier']

    def get_regime_and_sizing(
        self,
        df: pd.DataFrame,
        signal: float,
        base_size: float = 1.0
    ) -> Dict:
        """
        Get regime info and sizing recommendation.

        Args:
            df: OHLCV DataFrame with indicators
            signal: Signal strength
            base_size: Reference position size

        Returns:
            Dict with regime, confidence, position_size, and adjustment details
        """
        regime, confidence = self.detector.detect_regime(df)
        # Convert string signals for filter_signal (it handles conversion internally)
        adjusted = self.detector.filter_signal(signal, regime, confidence)

        # Safe float conversion for string signals
        signal_num = adjusted['adjusted_signal']

        return {
            'regime': regime.value,
            'confidence': float(confidence),
            'signal': float(signal_num),
            'adjusted_signal': float(adjusted['adjusted_signal']),
            'position_size': base_size * adjusted['position_size_multiplier'],
            'should_trade': adjusted['should_trade'],
            'reason': adjusted['reason'],
        }
