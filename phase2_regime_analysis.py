"""
Phase 2: Deep Regime and Correlation Analysis for V8 Trading Strategy
This script performs comprehensive regime classification, profitability analysis,
and stress testing on credit spread trades.
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

BASE_DIR = Path('/sessions/lucid-eloquent-euler/mnt/mytradingbot')
DATA_DIR = BASE_DIR / 'data'
BACKTEST_DIR = DATA_DIR / 'backtest'
PROCESSED_DIR = DATA_DIR / 'processed'
ANALYSIS_DIR = DATA_DIR / 'analysis'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 2: REGIME AND CORRELATION ANALYSIS - V8 TRADING STRATEGY")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")

# Load credit spread trades
credit_trades = pd.read_csv(BACKTEST_DIR / 'trades_SPY_options_credit.csv')
credit_trades['entry_date'] = pd.to_datetime(credit_trades['entry_date'])
credit_trades['exit_date'] = pd.to_datetime(credit_trades['exit_date'])

# Load directional trades
directional_trades = pd.read_csv(BACKTEST_DIR / 'trades_SPY.csv')
directional_trades['entry_date'] = pd.to_datetime(directional_trades['entry_date'])
directional_trades['exit_date'] = pd.to_datetime(directional_trades['exit_date'])

# Load features with indicators
features = pd.read_csv(PROCESSED_DIR / 'SPY_features.csv')
features['timestamp'] = pd.to_datetime(features['timestamp'])
features = features.sort_values('timestamp').reset_index(drop=True)

print(f"  Loaded {len(credit_trades)} credit spread trades")
print(f"  Loaded {len(directional_trades)} directional trades")
print(f"  Loaded {len(features)} feature rows")

# ============================================================================
# 2. REGIME CLASSIFICATION
# ============================================================================

print("\n[2/6] Classifying regimes for each day...")

def classify_regime(row):
    """
    Classify market regime based on ADX, DI+, DI-, SMA, and price position.
    Follows logic: TRENDING_UP, TRENDING_DOWN, MEAN_REVERTING, HIGH_VOLATILITY, UNKNOWN
    """
    adx = row['adx_14']
    di_plus = row['adx_pos_14']
    di_minus = row['adx_neg_14']
    sma_50 = row['sma_50']
    sma_200 = row['sma_200']
    close = row['close']
    atr_pct = row['atr_14_pct'] if 'atr_14_pct' in row.index and pd.notna(row['atr_14_pct']) else 0

    # Handle missing/invalid data
    if pd.isna(adx) or pd.isna(di_plus) or pd.isna(di_minus) or pd.isna(sma_50) or pd.isna(sma_200) or pd.isna(close):
        return 'UNKNOWN'

    # Skip if all indicators are zero (initialization period)
    if adx == 0 and di_plus == 0 and di_minus == 0:
        return 'UNKNOWN'

    # HIGH_VOLATILITY: high ATR regardless of trend
    if atr_pct > 3.0:
        return 'HIGH_VOLATILITY'

    # TRENDING_UP: strong uptrend signals
    if adx > 25 and di_plus > di_minus and sma_50 > sma_200 and close > sma_50:
        return 'TRENDING_UP'

    # TRENDING_DOWN: strong downtrend signals
    if adx > 25 and di_minus > di_plus and sma_50 < sma_200 and close < sma_50:
        return 'TRENDING_DOWN'

    # MEAN_REVERTING: weak trend (ADX < 25) with moderate price positioning
    if adx < 25 and sma_50 > sma_200:
        if (close < sma_50 and close > sma_200) or (close > sma_50 and close < sma_200):
            return 'MEAN_REVERTING'

    if adx < 25 and sma_50 < sma_200:
        if (close < sma_50 and close > sma_200) or (close > sma_50 and close < sma_200):
            return 'MEAN_REVERTING'

    # Default to MEAN_REVERTING if weak signals
    if adx < 25:
        return 'MEAN_REVERTING'

    return 'UNKNOWN'

features['regime'] = features.apply(classify_regime, axis=1)

# Regime distribution
regime_dist = features['regime'].value_counts(normalize=True) * 100
regime_counts = features['regime'].value_counts()

print("\nRegime Distribution:")
print(f"  {regime_dist.to_string()}")
print(f"\nRegime Counts:")
for regime, count in regime_counts.items():
    print(f"  {regime}: {count} days ({regime_dist[regime]:.1f}%)")

unknown_pct = regime_dist.get('UNKNOWN', 0)
print(f"\n  >>> UNKNOWN regime percentage: {unknown_pct:.1f}% (expected ~72%)")

results = {
    'regime_distribution': regime_dist.to_dict(),
    'regime_counts': regime_counts.to_dict(),
    'unknown_percentage': float(unknown_pct)
}

# ============================================================================
# 3. REGIME PROFITABILITY MATRIX
# ============================================================================

print("\n[3/6] Computing regime profitability matrix...")

# Create a mapping from timestamp to regime
regime_map = dict(zip(features['timestamp'].dt.date, features['regime']))

# Match trades to regimes at entry
credit_trades['entry_date_only'] = credit_trades['entry_date'].dt.date
credit_trades['regime_at_entry'] = credit_trades['entry_date_only'].map(regime_map)

# Classify trade directions (bull put = short, bear call = long, etc.)
def classify_trade_type(row):
    """Classify credit spread trades as Bull Put or Bear Call"""
    direction = row['direction']
    signal = row['signal']

    # Bull Put Spread: Short Put (sell OTM puts) -> short direction + BUY_PUT signal
    if direction == 'short' and signal == 'BUY_PUT':
        return 'Bull_Put'

    # Bear Call Spread: Short Call (sell OTM calls) -> long direction + BUY_CALL signal
    # But actually credit spreads sell calls, so check the pattern
    if direction == 'long' and signal == 'BUY_CALL':
        return 'Bear_Call'

    if direction == 'short' and signal == 'BUY_CALL':
        return 'Bear_Call'

    if direction == 'long' and signal == 'BUY_PUT':
        return 'Bull_Put'

    return 'Other'

credit_trades['trade_type'] = credit_trades.apply(classify_trade_type, axis=1)

# Build regime profitability matrix
regimes = features['regime'].unique()
regime_stats = {}

for regime in sorted(regimes):
    trades_in_regime = credit_trades[credit_trades['regime_at_entry'] == regime]

    if len(trades_in_regime) == 0:
        continue

    total_pnl_dollar = trades_in_regime['pnl_dollar'].sum()
    avg_pnl_dollar = trades_in_regime['pnl_dollar'].mean()
    avg_pnl_pct = trades_in_regime['pnl_pct'].mean()

    # Win rate
    winning_trades = (trades_in_regime['pnl_dollar'] > 0).sum()
    win_rate = winning_trades / len(trades_in_regime)

    # By trade direction
    bull_put = trades_in_regime[trades_in_regime['trade_type'] == 'Bull_Put']
    bear_call = trades_in_regime[trades_in_regime['trade_type'] == 'Bear_Call']

    regime_stats[regime] = {
        'trade_count': len(trades_in_regime),
        'win_rate': float(win_rate),
        'avg_pnl_dollar': float(avg_pnl_dollar),
        'avg_pnl_pct': float(avg_pnl_pct),
        'total_pnl_dollar': float(total_pnl_dollar),
        'bull_put_trades': len(bull_put),
        'bull_put_win_rate': float((bull_put['pnl_dollar'] > 0).sum() / len(bull_put)) if len(bull_put) > 0 else 0,
        'bull_put_avg_pnl': float(bull_put['pnl_dollar'].mean()) if len(bull_put) > 0 else 0,
        'bear_call_trades': len(bear_call),
        'bear_call_win_rate': float((bear_call['pnl_dollar'] > 0).sum() / len(bear_call)) if len(bear_call) > 0 else 0,
        'bear_call_avg_pnl': float(bear_call['pnl_dollar'].mean()) if len(bear_call) > 0 else 0,
    }

print("\nREGIME PROFITABILITY MATRIX:")
print("-" * 100)
print(f"{'Regime':<18} {'Trades':>8} {'Win%':>8} {'Avg P&L$':>12} {'Total P&L$':>12} {'Bull Put':>10} {'Bear Call':>10}")
print("-" * 100)

for regime in sorted(regime_stats.keys()):
    stats_dict = regime_stats[regime]
    print(f"{regime:<18} {stats_dict['trade_count']:>8} "
          f"{stats_dict['win_rate']*100:>7.1f}% "
          f"{stats_dict['avg_pnl_dollar']:>11.2f} "
          f"{stats_dict['total_pnl_dollar']:>11.2f} "
          f"{stats_dict['bull_put_win_rate']*100:>9.1f}% "
          f"{stats_dict['bear_call_win_rate']*100:>9.1f}%")

results['regime_profitability'] = regime_stats

# ============================================================================
# 4. CROSS-ASSET CORRELATION SIGNALS - VIX
# ============================================================================

print("\n[4/6] Analyzing VIX correlation with trade outcomes...")

# Match VIX values at entry
vix_map = dict(zip(features['timestamp'].dt.date, features['vix']))
credit_trades['vix_at_entry'] = credit_trades['entry_date_only'].map(vix_map)

# Remove trades without VIX data
trades_with_vix = credit_trades[credit_trades['vix_at_entry'].notna()].copy()

print(f"\n  Trades with VIX data: {len(trades_with_vix)} / {len(credit_trades)}")

if len(trades_with_vix) > 0:
    # Correlation between VIX and P&L
    correlation = trades_with_vix[['vix_at_entry', 'pnl_pct']].corr().iloc[0, 1]
    print(f"  Correlation (VIX at entry vs P&L %): {correlation:.4f}")

    # VIX bins
    vix_bins = [0, 15, 20, 25, 30, 100]
    vix_labels = ['<15', '15-20', '20-25', '25-30', '>30']
    trades_with_vix['vix_bin'] = pd.cut(trades_with_vix['vix_at_entry'], bins=vix_bins, labels=vix_labels, include_lowest=True)

    vix_stats = {}
    print("\nVIX RANGE ANALYSIS:")
    print("-" * 70)
    print(f"{'VIX Range':<12} {'Trades':>8} {'Win%':>8} {'Avg P&L$':>12} {'Total P&L$':>12}")
    print("-" * 70)

    for label in vix_labels:
        bin_trades = trades_with_vix[trades_with_vix['vix_bin'] == label]
        if len(bin_trades) == 0:
            continue

        win_rate = (bin_trades['pnl_dollar'] > 0).sum() / len(bin_trades)
        avg_pnl = bin_trades['pnl_dollar'].mean()
        total_pnl = bin_trades['pnl_dollar'].sum()

        vix_stats[label] = {
            'trade_count': len(bin_trades),
            'win_rate': float(win_rate),
            'avg_pnl_dollar': float(avg_pnl),
            'total_pnl_dollar': float(total_pnl)
        }

        print(f"{label:<12} {len(bin_trades):>8} {win_rate*100:>7.1f}% {avg_pnl:>11.2f} {total_pnl:>11.2f}")

    results['vix_correlation'] = float(correlation)
    results['vix_range_analysis'] = vix_stats

# ============================================================================
# 5. TAIL RISK ANALYSIS
# ============================================================================

print("\n[5/6] Analyzing tail risk and worst-case scenarios...")

# Compute daily returns
features['daily_return'] = features['close'].pct_change() * 100

# Find worst 5% of days
worst_5pct_threshold = features['daily_return'].quantile(0.05)
worst_days = features[features['daily_return'] <= worst_5pct_threshold].copy()

print(f"\nWorst 5% of days: {len(worst_days)} days")
print(f"  Worst day return: {features['daily_return'].min():.2f}%")
print(f"  5% threshold: {worst_5pct_threshold:.2f}%")

# For each worst day, find trades that were open
def was_trade_open(trade_entry, trade_exit, worst_day_date):
    """Check if a trade was open on a specific date"""
    return trade_entry.date() <= worst_day_date <= trade_exit.date()

open_during_worst = []
for _, worst_day in worst_days.iterrows():
    worst_date = worst_day['timestamp'].date()
    trades_open = credit_trades[credit_trades.apply(
        lambda row: was_trade_open(row['entry_date'], row['exit_date'], worst_date), axis=1
    )]
    if len(trades_open) > 0:
        open_during_worst.append({
            'date': worst_date,
            'daily_return': worst_day['daily_return'],
            'trades_open': len(trades_open),
            'avg_pnl_on_day': trades_open['pnl_dollar'].mean()
        })

print(f"\nTrades open during worst 5% days: {len(open_during_worst)} days with open trades")

tail_risk_stats = {
    'worst_5pct_days': len(worst_days),
    'days_with_open_trades': len(open_during_worst),
    'worst_day_return_pct': float(features['daily_return'].min()),
    'worst_5pct_threshold': float(worst_5pct_threshold)
}

if len(open_during_worst) > 0:
    total_pnl_worst = sum([d['avg_pnl_on_day'] * d['trades_open'] for d in open_during_worst])
    avg_pnl_worst = np.mean([d['avg_pnl_on_day'] for d in open_during_worst])
    tail_risk_stats['avg_pnl_during_worst_days'] = float(avg_pnl_worst)
    tail_risk_stats['total_pnl_during_worst_days'] = float(total_pnl_worst)

    print(f"  Average P&L on those days: ${avg_pnl_worst:.2f}")
    print(f"  Total P&L across worst days: ${total_pnl_worst:.2f}")

results['tail_risk_analysis'] = tail_risk_stats

# ============================================================================
# 6. STRESS TEST - WORST DRAWDOWN PERIODS
# ============================================================================

print("\n[6/6] Stress testing worst drawdown periods...")

# Calculate rolling returns to find drawdown periods
features['cumulative_return'] = (1 + features['daily_return'] / 100).cumprod()

# Rolling maximum
features['rolling_max'] = features['cumulative_return'].rolling(window=250, min_periods=1).max()

# Drawdown
features['drawdown'] = (features['cumulative_return'] - features['rolling_max']) / features['rolling_max'] * 100

# Find worst drawdown periods (top 5 worst 20-day rolling returns)
features['rolling_20d_return'] = features['daily_return'].rolling(window=20, min_periods=1).sum()
worst_periods = features.nsmallest(5, 'rolling_20d_return')[['timestamp', 'rolling_20d_return', 'drawdown']]

print("\nWorst 5 drawdown periods (20-day rolling returns):")
print("-" * 70)

stress_test_results = []

for _, period in worst_periods.iterrows():
    period_date = period['timestamp'].date()
    rolling_return = period['rolling_20d_return']

    # Find trades active around this period
    period_start = period['timestamp'] - timedelta(days=20)
    period_end = period['timestamp']

    trades_in_period = credit_trades[
        (credit_trades['entry_date'] >= period_start) &
        (credit_trades['entry_date'] <= period_end)
    ]

    if len(trades_in_period) > 0:
        losing_trades = (trades_in_period['pnl_dollar'] < 0).sum()
        loss_rate = losing_trades / len(trades_in_period)
        avg_loss = trades_in_period[trades_in_period['pnl_dollar'] < 0]['pnl_dollar'].mean()

        stress_test_results.append({
            'period_end_date': str(period_date),
            'rolling_20d_return': float(rolling_return),
            'trades_entered': len(trades_in_period),
            'losing_trades': losing_trades,
            'loss_rate': float(loss_rate),
            'avg_loss_of_losing': float(avg_loss) if losing_trades > 0 else 0
        })

        print(f"  {period_date}: {rolling_return:>7.2f}% return, "
              f"{len(trades_in_period):>3} trades, "
              f"{loss_rate*100:>5.1f}% losing")

results['stress_test'] = {
    'worst_periods': stress_test_results
}

# ============================================================================
# 7. SERIAL CORRELATION - DO WINNERS/LOSERS CLUSTER?
# ============================================================================

print("\nAnalyzing serial correlation of trades...")

# Add profitable flag
credit_trades['is_winning'] = credit_trades['pnl_dollar'] > 0
credit_trades['month'] = credit_trades['entry_date'].dt.to_period('M')

# Serial correlation by regime
print("\nSerial Correlation by Regime:")
regime_win_clusters = {}
for regime in sorted(regimes):
    trades_in_regime = credit_trades[credit_trades['regime_at_entry'] == regime].sort_values('entry_date')
    if len(trades_in_regime) < 2:
        continue

    # Check if winners/losers cluster
    wins = trades_in_regime['is_winning'].values
    if len(wins) > 1:
        # Simple clustering: count consecutive same outcomes
        clusters = 1
        for i in range(1, len(wins)):
            if wins[i] != wins[i-1]:
                clusters += 1

        # Expected clusters if random
        win_pct = np.mean(wins)
        expected_clusters = len(wins) * (2 * win_pct * (1 - win_pct))

        regime_win_clusters[regime] = {
            'observed_clusters': clusters,
            'expected_clusters': float(expected_clusters),
            'clustering_ratio': float(clusters / expected_clusters) if expected_clusters > 0 else 0
        }

# Serial correlation by month
print("\nTrade Outcomes by Month:")
monthly_stats = {}
for month, group in credit_trades.groupby('month'):
    win_rate = (group['pnl_dollar'] > 0).sum() / len(group)
    monthly_stats[str(month)] = {
        'trades': len(group),
        'win_rate': float(win_rate),
        'avg_pnl': float(group['pnl_dollar'].mean())
    }

results['serial_correlation'] = {
    'regime_clustering': regime_win_clusters,
    'monthly_stats': monthly_stats
}

print(f"  Analyzed {len(credit_trades)} trades across {len(monthly_stats)} months")

# ============================================================================
# 8. SUMMARY AND SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS...")
print("="*80)

# Summary statistics
total_trades = len(credit_trades)
total_pnl = credit_trades['pnl_dollar'].sum()
overall_win_rate = (credit_trades['pnl_dollar'] > 0).sum() / total_trades
avg_pnl = credit_trades['pnl_dollar'].mean()

# Helper function to convert numpy types to Python types
def convert_to_serializable(obj):
    """Recursively convert numpy types to Python types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj

summary = {
    'analysis_date': datetime.now().isoformat(),
    'data_period': f"{credit_trades['entry_date'].min().date()} to {credit_trades['entry_date'].max().date()}",
    'total_credit_spread_trades': int(total_trades),
    'overall_win_rate': float(overall_win_rate),
    'total_pnl_dollar': float(total_pnl),
    'avg_pnl_per_trade': float(avg_pnl),
    'regime_analysis': convert_to_serializable(results)
}

# Save to JSON
output_file = ANALYSIS_DIR / 'phase2_results.json'
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to: {output_file}")

# Print summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nTotal Credit Spread Trades: {total_trades}")
print(f"Overall Win Rate: {overall_win_rate*100:.1f}%")
print(f"Total P&L: ${total_pnl:.2f}")
print(f"Average P&L per Trade: ${avg_pnl:.2f}")
print(f"\nUnknown Regime Percentage: {unknown_pct:.1f}%")
print(f"\nKey Finding: {unknown_pct:.1f}% of days classified as UNKNOWN (expected ~72%)")

# Identify best regime
best_regime = max(regime_stats.items(), key=lambda x: x[1]['win_rate'])
print(f"\nBest regime by win rate: {best_regime[0]} ({best_regime[1]['win_rate']*100:.1f}%)")

# Identify worst drawdown
worst_drawdown_info = worst_periods.iloc[0] if len(worst_periods) > 0 else None
if worst_drawdown_info is not None:
    print(f"Worst 20-day period: {worst_drawdown_info['timestamp'].date()} ({worst_drawdown_info['rolling_20d_return']:.2f}% return)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
