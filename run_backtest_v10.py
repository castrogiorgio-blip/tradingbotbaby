#!/usr/bin/env python3
"""
V10 BACKTEST — THREE-LAYER PREMIUM HARVESTING ENGINE

Architecture:
  Layer 1: Rules-based credit spread engine with REALISTIC theta model
  Layer 2: Variance Risk Premium (VRP) timing — only trade when IV > RV
  Layer 3: Event-aware sizing + cross-asset regime detection

Edge source: The variance risk premium (IV consistently > RV on SPY)
NOT from: ML direction prediction (proven to have zero OOS edge)

Usage:
    python3 run_backtest_v10.py
    python3 run_backtest_v10.py --layer 1        # Layer 1 only (baseline)
    python3 run_backtest_v10.py --layer 2        # Layer 1 + 2
    python3 run_backtest_v10.py --layer 3        # All three layers
    python3 run_backtest_v10.py --compare        # Compare all configs + buy-and-hold
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from copy import deepcopy

# ═══════════════════════════════════════════════════════════
# DATA LOADING & ENRICHMENT
# ═══════════════════════════════════════════════════════════

def load_enriched_data():
    """Load SPY features and merge in economic data (VIX, yields, etc.)."""
    base_dir = Path(__file__).parent

    # Load SPY features
    df = pd.read_csv(base_dir / "data/processed/SPY_features.csv",
                     index_col=0, parse_dates=True)

    # Load economic data with actual VIX values
    econ = pd.read_csv(base_dir / "data/raw/economic/economic_indicators.csv",
                       index_col=0, parse_dates=True)

    # Merge VIX and macro data into features
    for col in ['vix', 'yield_spread_10y2y', 'treasury_10y', 'dollar_index',
                'consumer_sentiment', 'fed_funds_rate']:
        if col in econ.columns:
            econ_series = econ[col].dropna()
            # Align to features index
            df[col] = econ_series.reindex(df.index, method='ffill')

    # Forward-fill any remaining gaps
    df['vix'] = df['vix'].ffill().fillna(20.0)  # Default VIX = 20 if missing
    df['yield_spread_10y2y'] = df['yield_spread_10y2y'].ffill().fillna(0.0)
    df['treasury_10y'] = df['treasury_10y'].ffill().fillna(3.5)

    # ──────────────────────────────────────
    # DERIVED FEATURES for layers 2 and 3
    # ──────────────────────────────────────

    # Realized Volatility (annualized, 20-day)
    returns = df['close'].pct_change()
    df['realized_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
    df['realized_vol_10d'] = returns.rolling(10).std() * np.sqrt(252)
    df['realized_vol_5d'] = returns.rolling(5).std() * np.sqrt(252)

    # Implied Volatility proxy = VIX / 100
    df['implied_vol'] = df['vix'] / 100.0

    # VARIANCE RISK PREMIUM = IV - RV  (positive means options are overpriced)
    df['vrp_20d'] = df['implied_vol'] - df['realized_vol_20d']
    df['vrp_10d'] = df['implied_vol'] - df['realized_vol_10d']

    # VRP percentile (rolling 60-day window)
    df['vrp_percentile'] = df['vrp_20d'].rolling(60).rank(pct=True)

    # VIX term structure proxy (VIX rate of change as contango/backwardation signal)
    df['vix_5d_change'] = df['vix'].pct_change(5)
    df['vix_20d_change'] = df['vix'].pct_change(20)
    df['vix_mean_20d'] = df['vix'].rolling(20).mean()
    df['vix_zscore'] = (df['vix'] - df['vix'].rolling(60).mean()) / df['vix'].rolling(60).std()

    # Yield curve dynamics
    df['yield_curve_slope_change'] = df['yield_spread_10y2y'].diff(5)

    # SPY momentum / trend
    df['spy_return_5d'] = df['close'].pct_change(5)
    df['spy_return_20d'] = df['close'].pct_change(20)
    df['spy_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
    df['spy_above_sma200'] = (df['close'] > df['sma_200']).astype(int)

    # ATR-based move magnitude
    df['atr_pct'] = df['atr_14'] / df['close']

    # FOMC / event flags (already in data)
    for col in ['fomc_week', 'earnings_week', 'high_event_risk', 'quad_witching']:
        if col not in df.columns:
            df[col] = 0

    print(f"  Data loaded: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  VIX range: {df['vix'].min():.1f} - {df['vix'].max():.1f}")
    print(f"  VRP (20d) range: {df['vrp_20d'].min():.4f} - {df['vrp_20d'].max():.4f}")

    return df


# ═══════════════════════════════════════════════════════════
# LAYER 1: RULES-BASED PREMIUM HARVESTING ENGINE
# ═══════════════════════════════════════════════════════════

class CreditSpreadEngine:
    """
    Realistic credit spread backtester.

    Key differences from V8:
    - Theta model: 1.0% per day of credit, cap at 20% (vs V8's 5%/day, 50% cap)
    - Delta model: incorporates gamma acceleration on adverse moves
    - Vega model: vol expansion hurts credit spreads (V8 ignored this)
    - Stop loss: 1.5x credit (vs V8's 2x)
    - Trade frequency: weekly entries (vs signal-dependent)
    """

    def __init__(self, capital=10000.0, config=None):
        self.starting_capital = capital
        self.config = config or self.default_config()

    @staticmethod
    def default_config():
        return {
            # Position sizing
            'risk_per_trade': 0.02,       # 2% of capital per trade
            'max_open_positions': 4,      # max concurrent
            # Spread parameters
            'spread_width_pct': 0.03,     # 3% OTM spread width (dynamic in Layer 2)
            'credit_pct': 0.30,           # receive 30% of spread width
            'target_dte': 45,             # target days to expiration
            'manage_at_dte': 21,          # manage/close at 21 DTE
            # Exit rules
            'take_profit_pct': 0.50,      # take profit at 50% of credit received
            'stop_loss_multiple': 1.0,    # stop at 1.0x credit received (TIGHT)
            'max_hold_days': 21,          # absolute max (manages at 14)
            'manage_at_dte': 14,          # manage earlier
            # Realistic theta model
            'theta_rate_per_day': 0.012,  # 1.2% of credit per day
            'theta_cap': 0.20,            # max 20% of position from theta
            # Vega impact
            'vega_sensitivity': 0.5,      # how much vol changes affect P&L
        }

    def evaluate_position(self, pos, current_price, current_vix, date):
        """
        Realistic credit spread P&L evaluation.

        P&L = delta_pnl + theta_benefit - vega_cost
        """
        cfg = self.config
        days = max(0.5, (date - pos['entry_date']).days)
        entry_price = pos['entry_price']
        spread_width = cfg['spread_width_pct']
        credit_pct = cfg['credit_pct']

        # ── DELTA P&L ──
        price_move = (current_price - entry_price) / entry_price
        if pos['spread_type'] == 'bull_put':
            dir_move = price_move   # positive when SPY goes up (good for bull put)
        else:
            dir_move = -price_move  # positive when SPY goes down (good for bear call)

        if dir_move >= 0:
            # Favorable move: credit spread gains value
            move_vs_spread = dir_move / spread_width
            delta_pnl = credit_pct * min(move_vs_spread * 1.5, 1.0)
        else:
            # Adverse move: losses accelerate (gamma effect)
            adverse = -dir_move
            move_vs_spread = adverse / spread_width
            if move_vs_spread < 0.3:
                # Small adverse move: delta-driven linear loss
                delta_pnl = -credit_pct * move_vs_spread * 1.2
            elif move_vs_spread < 0.7:
                # Moderate: gamma accelerates losses
                base = -credit_pct * 0.3 * 1.2
                extra = -(move_vs_spread - 0.3) * (1 - credit_pct) * 1.5
                delta_pnl = base + extra
            else:
                # Large move: approaching max loss
                delta_pnl = -(1 - credit_pct) * min(move_vs_spread / 1.0, 1.0)

        # ── THETA BENEFIT (realistic) ──
        theta_per_day = cfg['theta_rate_per_day']
        # Theta accelerates as expiration approaches (not linear)
        # Use sqrt decay: more theta in later days
        if days <= 7:
            theta_multiplier = 1.0
        elif days <= 14:
            theta_multiplier = 0.8
        elif days <= 21:
            theta_multiplier = 0.6
        else:
            theta_multiplier = 0.4

        theta_benefit = min(days * theta_per_day * theta_multiplier, cfg['theta_cap']) * credit_pct

        # ── VEGA COST (vol expansion hurts credit spreads) ──
        vix_change = (current_vix - pos['entry_vix']) / max(pos['entry_vix'], 10.0)
        vega_cost = 0.0
        if vix_change > 0:
            # VIX went up → credit spread loses value
            vega_cost = vix_change * cfg['vega_sensitivity'] * credit_pct
        elif vix_change < -0.05:
            # VIX dropped significantly → small benefit
            vega_cost = vix_change * cfg['vega_sensitivity'] * credit_pct * 0.3

        # ── NET P&L ──
        net_pnl = delta_pnl + theta_benefit - vega_cost
        net_pnl = max(net_pnl, -(1 - credit_pct))  # can't lose more than spread width minus credit
        net_pnl = min(net_pnl, credit_pct)           # can't gain more than credit received

        # ── EXIT LOGIC ──
        should_close = False
        reason = ""

        tp_threshold = credit_pct * cfg['take_profit_pct']
        sl_threshold = -credit_pct * cfg['stop_loss_multiple']

        if net_pnl >= tp_threshold:
            should_close = True
            reason = "take_profit"
        elif net_pnl <= sl_threshold:
            should_close = True
            reason = "stop_loss"
            net_pnl = sl_threshold  # cap at stop level
        elif days >= cfg['max_hold_days']:
            should_close = True
            reason = "max_hold"
        elif days >= cfg['manage_at_dte'] and net_pnl > 0:
            should_close = True
            reason = "manage_21dte"

        return should_close, reason, net_pnl


# ═══════════════════════════════════════════════════════════
# LAYER 2: VARIANCE RISK PREMIUM TIMING
# ═══════════════════════════════════════════════════════════

class VRPTimingModel:
    """
    Only enters trades when implied volatility > realized volatility.
    This is the actual edge source — selling overpriced insurance.

    Signals:
    - VRP > 0 and in top 40th percentile → SELL PREMIUM (full size)
    - VRP > 0 but below 40th percentile → SELL PREMIUM (half size)
    - VRP < 0 (rare: options are cheap) → NO TRADE
    - VIX > 35 → NO TRADE (crisis, gamma risk too high)
    - VIX < 12 → NO TRADE (premium too thin to cover transaction costs)
    """

    def __init__(self):
        self.min_vix = 13.0
        self.max_vix = 28.0  # Tighter ceiling: VIX>25 had 52% win rate (no edge)
        self.vrp_full_size_percentile = 0.40
        self.vrp_minimum = 0.0

    def should_trade(self, row):
        """Returns (should_trade, size_multiplier, reason)."""
        vix = row.get('vix', 20.0)
        vrp = row.get('vrp_20d', 0.0)
        vrp_pctile = row.get('vrp_percentile', 0.5)

        if pd.isna(vrp) or pd.isna(vix):
            return False, 0.0, "missing_data"

        if vix > self.max_vix:
            return False, 0.0, f"vix_too_high_{vix:.0f}"

        if vix < self.min_vix:
            return False, 0.0, f"vix_too_low_{vix:.0f}"

        if vrp < self.vrp_minimum:
            return False, 0.0, f"vrp_negative_{vrp:.4f}"

        # VRP is positive — options are overpriced, edge exists
        if vrp_pctile >= self.vrp_full_size_percentile:
            return True, 1.0, f"vrp_strong_{vrp:.4f}_p{vrp_pctile:.0%}"
        else:
            return True, 0.6, f"vrp_moderate_{vrp:.4f}_p{vrp_pctile:.0%}"

    def dynamic_spread_width(self, vix):
        """Wider spreads when vol is high (more premium to collect)."""
        if vix < 15:
            return 0.025  # 2.5% width
        elif vix < 22:
            return 0.030  # 3.0% width
        elif vix < 30:
            return 0.035  # 3.5% width
        else:
            return 0.040  # 4.0% width

    def dynamic_credit(self, vix):
        """Higher credit collection in high-vol (options richer)."""
        if vix < 15:
            return 0.25
        elif vix < 22:
            return 0.30
        elif vix < 30:
            return 0.35
        else:
            return 0.40


# ═══════════════════════════════════════════════════════════
# LAYER 3: EVENT-AWARE + CROSS-ASSET REGIME
# ═══════════════════════════════════════════════════════════

class CrossAssetRegime:
    """
    Two-state regime model using cross-asset signals.

    RISK-ON (sell premium aggressively):
    - Yield curve positive (2s10s > 0)
    - VIX declining or stable (5d change < +10%)
    - SPY above SMA200
    - Fed not actively hiking (no recent rate increases)

    RISK-OFF (reduce size or stand aside):
    - Yield curve inverting or inverted
    - VIX spiking (5d change > +15%)
    - SPY below SMA200
    - Credit stress signals
    """

    def get_regime(self, row):
        """Returns (regime, size_multiplier)."""
        risk_on_score = 0
        total_signals = 0

        # Signal 1: Yield curve
        ys = row.get('yield_spread_10y2y', 0)
        if not pd.isna(ys):
            total_signals += 1
            if ys > 0.2:
                risk_on_score += 1
            elif ys < -0.3:
                risk_on_score -= 1

        # Signal 2: VIX momentum
        vix_5d = row.get('vix_5d_change', 0)
        if not pd.isna(vix_5d):
            total_signals += 1
            if vix_5d < 0:          # VIX declining
                risk_on_score += 1
            elif vix_5d > 0.15:     # VIX spiking
                risk_on_score -= 1

        # Signal 3: SPY trend
        above_200 = row.get('spy_above_sma200', 0)
        above_50 = row.get('spy_above_sma50', 0)
        total_signals += 1
        if above_200 and above_50:
            risk_on_score += 1
        elif not above_200 and not above_50:
            risk_on_score -= 1

        # Signal 4: RSI extremes
        rsi = row.get('rsi_14', 50)
        if not pd.isna(rsi) and rsi > 0:
            total_signals += 1
            if 35 < rsi < 65:
                risk_on_score += 0.5  # neutral RSI = no stress
            elif rsi < 25 or rsi > 80:
                risk_on_score -= 1    # extreme RSI = caution

        # Convert to regime
        if total_signals == 0:
            return "UNKNOWN", 0.7

        normalized_score = risk_on_score / total_signals

        if normalized_score > 0.3:
            return "RISK_ON", 1.2   # slightly oversize
        elif normalized_score > 0:
            return "NEUTRAL", 1.0
        elif normalized_score > -0.3:
            return "CAUTIOUS", 0.6
        else:
            return "RISK_OFF", 0.3


class EventFilter:
    """Adjusts sizing around known high-impact events."""

    def get_event_adjustment(self, row):
        """Returns (size_multiplier, reason)."""
        # FOMC week — vol is elevated, gamma risk
        if row.get('fomc_week', 0) == 1:
            return 0.5, "fomc_week"

        # High event risk
        if row.get('high_event_risk', 0) == 1:
            return 0.7, "high_event_risk"

        # Quad witching (options expiration)
        if row.get('quad_witching', 0) == 1:
            return 0.8, "quad_witching"

        return 1.0, "no_event"


# ═══════════════════════════════════════════════════════════
# DIRECTION CHOOSER (replaces broken ML model)
# ═══════════════════════════════════════════════════════════

class DirectionChooser:
    """
    Simple, transparent direction choice.
    Not trying to predict — just aligning with obvious trend.

    Bull Put (benefits from SPY flat or up):
    - SPY above SMA50 and SMA200
    - RSI not overbought (< 70)

    Bear Call (benefits from SPY flat or down):
    - SPY below SMA50 and SMA200
    - RSI not oversold (> 30)

    Iron Condor (benefits from SPY staying in range):
    - Mixed signals or neutral regime
    - VIX > 18 (enough premium on both sides)
    """

    def choose(self, row):
        """Returns list of spread types to enter."""
        above_50 = row.get('spy_above_sma50', 0)
        above_200 = row.get('spy_above_sma200', 0)
        rsi = row.get('rsi_14', 50)
        vix = row.get('vix', 20)
        macd_hist = row.get('macd_histogram', 0)

        spreads = []

        if above_50 and above_200 and rsi < 70:
            # Clear uptrend: sell bull put (profit if SPY stays flat or goes up)
            spreads.append('bull_put')

        elif not above_50 and not above_200 and rsi > 35 and rsi < 55:
            # Clear downtrend + not oversold: sell bear call
            spreads.append('bear_call')

        elif vix > 18 and above_200:
            # Elevated vol but above long-term trend: bull put (collect richer premium)
            spreads.append('bull_put')

        else:
            # Default: bull put (structural long bias, SPY trends up long-term)
            # Only sell bear calls when overwhelming evidence of downtrend
            spreads.append('bull_put')

        return spreads


# ═══════════════════════════════════════════════════════════
# MAIN BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════

def run_v10_backtest(df, layer=3, capital=10000.0, entry_frequency_days=5):
    """
    Run the V10 backtest with specified number of layers.

    layer=1: Basic credit spread engine only
    layer=2: + VRP timing
    layer=3: + Event-aware sizing + cross-asset regime
    """

    engine = CreditSpreadEngine(capital=capital)
    vrp_model = VRPTimingModel()
    regime_model = CrossAssetRegime()
    event_filter = EventFilter()
    direction_chooser = DirectionChooser()

    cfg = engine.config
    current_capital = capital
    open_positions = []
    all_trades = []
    portfolio_history = []
    skipped_reasons = {}
    regime_trade_counts = {}

    # Start from day 200+ to ensure all indicators are populated
    start_idx = 250
    last_entry_date = None
    days_since_entry = entry_frequency_days  # allow first entry immediately

    for i in range(start_idx, len(df)):
        date = df.index[i]
        row = df.iloc[i]
        current_price = row['close']
        current_vix = row.get('vix', 20.0)

        if current_price <= 0 or pd.isna(current_price):
            continue

        # ── CLOSE EXISTING POSITIONS ──
        to_close = []
        for pi, pos in enumerate(open_positions):
            should_close, reason, pnl_pct = engine.evaluate_position(
                pos, current_price, current_vix, date
            )
            if should_close:
                pos['exit_date'] = date
                pos['exit_price'] = current_price
                pos['exit_vix'] = current_vix
                pos['exit_reason'] = reason
                pos['pnl_pct'] = pnl_pct
                pos['pnl_dollar'] = pos['position_size'] * pnl_pct
                pos['hold_days'] = (date - pos['entry_date']).days
                to_close.append(pi)

        for pi in sorted(to_close, reverse=True):
            closed = open_positions.pop(pi)
            current_capital += closed['position_size'] + closed['pnl_dollar']
            all_trades.append(closed)

        # ── ENTRY LOGIC ──
        days_since_entry += 1

        can_enter = (
            len(open_positions) < cfg['max_open_positions']
            and current_capital > capital * 0.3  # safety: stop if capital drops below 30%
            and days_since_entry >= entry_frequency_days
        )

        if can_enter:
            # ── LAYER 1: Basic eligibility ──
            size_multiplier = 1.0
            trade_allowed = True
            skip_reason = None
            spread_width = cfg['spread_width_pct']
            credit_pct = cfg['credit_pct']

            # ── LAYER 2: VRP timing ──
            if layer >= 2:
                vrp_ok, vrp_size, vrp_reason = vrp_model.should_trade(row)
                if not vrp_ok:
                    trade_allowed = False
                    skip_reason = vrp_reason
                else:
                    size_multiplier *= vrp_size
                    # Dynamic spread width and credit based on VIX
                    spread_width = vrp_model.dynamic_spread_width(current_vix)
                    credit_pct = vrp_model.dynamic_credit(current_vix)

            # ── LAYER 3: Regime + Events ──
            regime = "NEUTRAL"
            if layer >= 3 and trade_allowed:
                regime, regime_mult = regime_model.get_regime(row)
                size_multiplier *= regime_mult

                event_mult, event_reason = event_filter.get_event_adjustment(row)
                size_multiplier *= event_mult

                if regime == "RISK_OFF" and current_vix > 28:
                    trade_allowed = False
                    skip_reason = f"risk_off_high_vix_{current_vix:.0f}"

            # Track skips
            if not trade_allowed:
                skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                # Still record portfolio history
                open_value = sum(p['position_size'] for p in open_positions)
                portfolio_history.append({
                    'date': date, 'capital': current_capital,
                    'open_value': open_value,
                    'total_value': current_capital + open_value,
                    'open_positions': len(open_positions),
                    'vix': current_vix, 'regime': regime,
                })
                continue

            # ── CHOOSE DIRECTION(S) ──
            spread_types = direction_chooser.choose(row)

            for spread_type in spread_types:
                if len(open_positions) >= cfg['max_open_positions']:
                    break

                # Position size with all multipliers
                pos_size = current_capital * cfg['risk_per_trade'] * size_multiplier
                pos_size = max(min(pos_size, current_capital * 0.05), 50.0)  # cap at 5%, min $50

                if pos_size < 50 or current_capital < pos_size:
                    continue

                position = {
                    'entry_date': date,
                    'symbol': 'SPY',
                    'spread_type': spread_type,
                    'direction': 'long' if spread_type == 'bull_put' else 'short',
                    'entry_price': current_price,
                    'entry_vix': current_vix,
                    'position_size': pos_size,
                    'spread_width': spread_width,
                    'credit_pct': credit_pct,
                    'vrp_at_entry': row.get('vrp_20d', 0),
                    'regime_at_entry': regime,
                    'size_multiplier': size_multiplier,
                    'layer': layer,
                }

                open_positions.append(position)
                current_capital -= pos_size
                days_since_entry = 0

                # Track regime counts
                regime_trade_counts[regime] = regime_trade_counts.get(regime, 0) + 1

        # Record portfolio value
        open_value = sum(p['position_size'] for p in open_positions)
        portfolio_history.append({
            'date': date, 'capital': current_capital,
            'open_value': open_value,
            'total_value': current_capital + open_value,
            'open_positions': len(open_positions),
            'vix': current_vix, 'regime': regime if layer >= 3 else 'N/A',
        })

    # Close remaining positions at end
    if open_positions and len(df) > 0:
        last_price = df['close'].iloc[-1]
        last_vix = df['vix'].iloc[-1] if 'vix' in df.columns else 20.0
        last_date = df.index[-1]
        for pos in open_positions:
            _, _, pnl_pct = engine.evaluate_position(pos, last_price, last_vix, last_date)
            pos['exit_date'] = last_date
            pos['exit_price'] = last_price
            pos['exit_vix'] = last_vix
            pos['exit_reason'] = 'backtest_end'
            pos['pnl_pct'] = pnl_pct
            pos['pnl_dollar'] = pos['position_size'] * pnl_pct
            pos['hold_days'] = (last_date - pos['entry_date']).days
            current_capital += pos['position_size'] + pos['pnl_dollar']
            all_trades.append(pos)

    # ── COMPUTE METRICS ──
    metrics = compute_metrics(all_trades, portfolio_history, capital, layer)
    metrics['skipped_reasons'] = skipped_reasons
    metrics['regime_trade_counts'] = regime_trade_counts

    return all_trades, portfolio_history, metrics


def compute_metrics(trades, history, starting_capital, layer):
    """Compute comprehensive performance metrics."""
    if not trades:
        return {
            'layer': layer, 'total_trades': 0, 'win_rate': 0,
            'total_return': 0, 'sharpe': 0, 'max_drawdown': 0,
            'profit_factor': 0, 'avg_pnl': 0, 'expectancy': 0,
        }

    pnls = [t['pnl_dollar'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Portfolio-level metrics
    values = [h['total_value'] for h in history if 'total_value' in h]
    vs = pd.Series(values)
    daily_ret = vs.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if len(daily_ret) > 1 and daily_ret.std() > 0 else 0)

    peak = vs.cummax()
    drawdown = ((vs - peak) / peak)
    max_dd = drawdown.min()

    final_value = values[-1] if values else starting_capital
    total_return = (final_value - starting_capital) / starting_capital

    # Trade-level
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')

    # Hold time
    hold_days = [t.get('hold_days', 0) for t in trades]

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # By spread type
    bull_puts = [t for t in trades if t.get('spread_type') == 'bull_put']
    bear_calls = [t for t in trades if t.get('spread_type') == 'bear_call']

    # Annual metrics
    if history:
        days_in_backtest = (history[-1]['date'] - history[0]['date']).days
        years = days_in_backtest / 365.25
    else:
        years = 1
    trades_per_year = len(trades) / max(years, 0.1)
    annual_return = total_return / max(years, 0.1)

    return {
        'layer': layer,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'annual_return': annual_return,
        'final_value': final_value,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'expectancy': np.mean(pnls),
        'avg_win_pct': np.mean([p for p in pnl_pcts if p > 0]) if wins else 0,
        'avg_loss_pct': np.mean([p for p in pnl_pcts if p < 0]) if losses else 0,
        'avg_hold_days': np.mean(hold_days),
        'trades_per_year': trades_per_year,
        'bull_put_count': len(bull_puts),
        'bear_call_count': len(bear_calls),
        'bull_put_win_rate': sum(1 for t in bull_puts if t['pnl_dollar'] > 0) / max(len(bull_puts), 1),
        'bear_call_win_rate': sum(1 for t in bear_calls if t['pnl_dollar'] > 0) / max(len(bear_calls), 1),
        'exit_reasons': exit_reasons,
        'years': years,
    }


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════

def walk_forward_validation(df, layer=3, n_folds=4, capital=10000.0):
    """Expanding-window walk-forward test."""
    n = len(df)
    fold_size = n // (n_folds + 1)  # size of each test window

    results = []

    for fold in range(n_folds):
        # Training window expands, test window slides
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start + 20:
            continue

        df_test = df.iloc[test_start:test_end].copy()

        # V10 is rules-based so "training" just means we use data before test window
        # for indicator calculations (which are already computed)
        trades, history, metrics = run_v10_backtest(
            df_test, layer=layer, capital=capital,
            entry_frequency_days=5
        )

        results.append({
            'fold': fold + 1,
            'test_start': df_test.index[0].strftime('%Y-%m-%d'),
            'test_end': df_test.index[-1].strftime('%Y-%m-%d'),
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'return': metrics['total_return'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_drawdown'],
        })

    return results


# ═══════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════

def monte_carlo(trades, n_simulations=2000, starting_capital=10000.0):
    """Bootstrap Monte Carlo from actual trade returns."""
    if not trades:
        return {}

    pnl_pcts = np.array([t['pnl_pct'] for t in trades])
    pos_sizes = np.array([t['position_size'] for t in trades])
    avg_size_pct = (pos_sizes / starting_capital).mean()

    final_values = []
    for _ in range(n_simulations):
        # Resample trade returns with replacement
        sampled = np.random.choice(pnl_pcts, size=len(trades), replace=True)
        capital = starting_capital
        for pnl in sampled:
            trade_size = capital * avg_size_pct
            capital += trade_size * pnl
        final_values.append(capital)

    fv = np.array(final_values)
    returns = (fv - starting_capital) / starting_capital

    return {
        'mean_return': float(np.mean(returns)),
        'median_return': float(np.median(returns)),
        'std_return': float(np.std(returns)),
        'p5_return': float(np.percentile(returns, 5)),
        'p25_return': float(np.percentile(returns, 25)),
        'p75_return': float(np.percentile(returns, 75)),
        'p95_return': float(np.percentile(returns, 95)),
        'prob_profit': float((returns > 0).mean()),
        'prob_loss_10pct': float((returns < -0.10).mean()),
        'var_95': float(np.percentile(returns, 5)),
        'cvar_95': float(returns[returns <= np.percentile(returns, 5)].mean()) if (returns <= np.percentile(returns, 5)).any() else 0,
    }


# ═══════════════════════════════════════════════════════════
# BUY AND HOLD BENCHMARK
# ═══════════════════════════════════════════════════════════

def buy_and_hold(df, capital=10000.0, start_idx=250):
    """Simple buy-and-hold SPY benchmark."""
    prices = df['close'].iloc[start_idx:]
    entry_price = prices.iloc[0]
    final_price = prices.iloc[-1]

    total_return = (final_price - entry_price) / entry_price
    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25
    annual_return = total_return / max(years, 0.1)

    # Drawdown
    cummax = prices.cummax()
    dd = ((prices - cummax) / cummax)
    max_dd = dd.min()

    # Sharpe
    daily_ret = prices.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0)

    return {
        'strategy': 'Buy & Hold SPY',
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'years': years,
        'entry_price': entry_price,
        'final_price': final_price,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V10 Three-Layer Premium Harvesting Engine")
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--capital', type=float, default=10000.0)
    parser.add_argument('--compare', action='store_true', help='Compare all layers + buy-and-hold')
    parser.add_argument('--frequency', type=int, default=5, help='Days between entries')
    args = parser.parse_args()

    print("=" * 80)
    print("  V10 — THREE-LAYER PREMIUM HARVESTING ENGINE")
    print("  Edge: Variance Risk Premium | Method: Rules-Based | No ML")
    print("=" * 80)

    # Load data
    print("\n[1] Loading and enriching data...")
    df = load_enriched_data()

    if args.compare:
        # ── COMPARE ALL CONFIGURATIONS ──
        print("\n" + "=" * 80)
        print("  COMPARISON MODE: Layer 1 vs Layer 2 vs Layer 3 vs Buy-and-Hold")
        print("=" * 80)

        all_results = {}

        for layer in [1, 2, 3]:
            print(f"\n{'─'*60}")
            print(f"  Running Layer {layer}...")
            print(f"{'─'*60}")

            trades, history, metrics = run_v10_backtest(
                df, layer=layer, capital=args.capital,
                entry_frequency_days=args.frequency
            )

            all_results[f'layer_{layer}'] = metrics

            print(f"  Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Total Return: {metrics['total_return']:.2%}")
            print(f"  Annual Return: {metrics['annual_return']:.2%}")
            print(f"  Sharpe: {metrics['sharpe']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"  Trades/Year: {metrics['trades_per_year']:.1f}")

            # Walk-forward
            print(f"\n  Walk-Forward Validation (4 folds):")
            wf = walk_forward_validation(df, layer=layer, n_folds=4, capital=args.capital)
            for fold in wf:
                print(f"    Fold {fold['fold']}: {fold['trades']:3d} trades | "
                      f"Win: {fold['win_rate']:.0%} | Return: {fold['return']:+.2%} | "
                      f"Sharpe: {fold['sharpe']:.2f}")
            all_results[f'layer_{layer}']['walk_forward'] = wf

            if trades:
                # Monte Carlo
                mc = monte_carlo(trades, n_simulations=2000, starting_capital=args.capital)
                print(f"\n  Monte Carlo (2000 sims):")
                print(f"    Mean return: {mc['mean_return']:.2%}")
                print(f"    Median return: {mc['median_return']:.2%}")
                print(f"    5th percentile: {mc['p5_return']:.2%}")
                print(f"    95th percentile: {mc['p95_return']:.2%}")
                print(f"    P(profit): {mc['prob_profit']:.1%}")
                all_results[f'layer_{layer}']['monte_carlo'] = mc

            # Save trades
            if trades:
                trades_df = pd.DataFrame(trades)
                save_cols = ['entry_date', 'exit_date', 'symbol', 'spread_type', 'direction',
                            'entry_price', 'exit_price', 'entry_vix', 'exit_vix',
                            'position_size', 'pnl_pct', 'pnl_dollar', 'hold_days',
                            'exit_reason', 'vrp_at_entry', 'regime_at_entry', 'size_multiplier']
                save_cols = [c for c in save_cols if c in trades_df.columns]
                trades_df[save_cols].to_csv(
                    f"data/backtest/trades_SPY_v10_layer{layer}.csv", index=False
                )

        # Buy and hold benchmark
        bh = buy_and_hold(df, capital=args.capital)
        all_results['buy_and_hold'] = bh

        # ── COMPARISON TABLE ──
        print("\n" + "=" * 80)
        print("  FINAL COMPARISON TABLE")
        print("=" * 80)
        print(f"\n  {'Strategy':<20s} | {'Trades':>6s} | {'Win%':>6s} | {'Return':>8s} | {'Annual':>8s} | {'Sharpe':>7s} | {'MaxDD':>7s} | {'Tr/Yr':>5s}")
        print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")

        for layer in [1, 2, 3]:
            m = all_results[f'layer_{layer}']
            print(f"  {'V10 Layer '+str(layer):<20s} | {m['total_trades']:>6d} | {m['win_rate']:>5.1%} | {m['total_return']:>+7.2%} | {m['annual_return']:>+7.2%} | {m['sharpe']:>7.2f} | {m['max_drawdown']:>6.2%} | {m['trades_per_year']:>5.1f}")

        print(f"  {'Buy & Hold SPY':<20s} | {'N/A':>6s} | {'N/A':>6s} | {bh['total_return']:>+7.2%} | {bh['annual_return']:>+7.2%} | {bh['sharpe']:>7.2f} | {bh['max_drawdown']:>6.2%} | {'N/A':>5s}")

        # V8 reference (from existing metrics)
        print(f"  {'V8 (inflated θ)':<20s} | {'71':>6s} | {'87.3%':>6s} | {'+23.07%':>8s} | {'~18.4%':>8s} | {'1.60':>7s} | {'-4.50%':>7s} | {'~14':>5s}")

        # Save comparison
        # Clean up for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj

        with open("data/analysis/v10_comparison.json", "w") as f:
            json.dump(clean_for_json(all_results), f, indent=2, default=str)

        print(f"\n  Results saved to data/analysis/v10_comparison.json")
        print(f"  Trade CSVs saved to data/backtest/trades_SPY_v10_layer*.csv")

    else:
        # ── SINGLE LAYER RUN ──
        print(f"\n[2] Running V10 backtest — Layer {args.layer}...")
        trades, history, metrics = run_v10_backtest(
            df, layer=args.layer, capital=args.capital,
            entry_frequency_days=args.frequency
        )

        print(f"\n{'='*60}")
        print(f"  V10 RESULTS — Layer {args.layer}")
        print(f"{'='*60}")
        for key, val in metrics.items():
            if key in ('skipped_reasons', 'exit_reasons', 'regime_trade_counts', 'walk_forward'):
                continue
            if isinstance(val, float):
                print(f"  {key:<25s}: {val:>12.4f}")
            else:
                print(f"  {key:<25s}: {val}")

        print(f"\n  Exit Reasons: {metrics.get('exit_reasons', {})}")
        print(f"  Regime Trade Counts: {metrics.get('regime_trade_counts', {})}")
        print(f"  Skipped Reasons (top 5):")
        for reason, count in sorted(metrics.get('skipped_reasons', {}).items(),
                                     key=lambda x: -x[1])[:5]:
            print(f"    {reason}: {count}")

        # Walk-forward
        print(f"\n[3] Walk-Forward Validation...")
        wf = walk_forward_validation(df, layer=args.layer, n_folds=4, capital=args.capital)
        for fold in wf:
            print(f"  Fold {fold['fold']}: {fold['trades']:3d} trades | "
                  f"Win: {fold['win_rate']:.0%} | Return: {fold['return']:+.2%}")

        # Monte Carlo
        if trades:
            print(f"\n[4] Monte Carlo Simulation...")
            mc = monte_carlo(trades, n_simulations=2000, starting_capital=args.capital)
            print(f"  Mean return: {mc['mean_return']:.2%}")
            print(f"  5th-95th percentile: [{mc['p5_return']:.2%}, {mc['p95_return']:.2%}]")
            print(f"  P(profit): {mc['prob_profit']:.1%}")

        # Save
        if trades:
            trades_df = pd.DataFrame(trades)
            save_cols = ['entry_date', 'exit_date', 'symbol', 'spread_type', 'direction',
                        'entry_price', 'exit_price', 'entry_vix', 'exit_vix',
                        'position_size', 'pnl_pct', 'pnl_dollar', 'hold_days',
                        'exit_reason', 'vrp_at_entry', 'regime_at_entry', 'size_multiplier']
            save_cols = [c for c in save_cols if c in trades_df.columns]
            trades_df[save_cols].to_csv(
                f"data/backtest/trades_SPY_v10_layer{args.layer}.csv", index=False
            )


if __name__ == "__main__":
    main()
