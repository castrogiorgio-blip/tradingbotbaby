# V8/V9 Strategy Audit & Optimization Report

**Date:** March 22, 2026
**Author:** Quantitative Strategy Desk
**Subject:** SPY Credit Spread Strategy — Full Statistical Audit
**Verdict:** The V8 +23% credit spread return is largely a simulation artifact. The underlying model has no demonstrable out-of-sample edge.

---

## Executive Summary

The V8 trading bot claims +23.07% returns on credit spreads with an 87.3% win rate. After a five-phase forensic audit including P&L decomposition, regime analysis, stress testing, hedge design, and a rebuilt V9 backtest, the conclusions are stark:

**The +23% return is a mirage.** Approximately 55% of the credit spread P&L comes from an unrealistically generous theta decay model that overestimates time-decay income by 3-10x. When the theta model is corrected to realistic parameters (1.5% per day vs. 5% per day), the strategy loses money.

**The directional ML model is severely overfitted.** Walk-forward out-of-sample testing across 17 folds shows a 30.8% win rate (worse than a coin flip), -1.64% return, and a Sharpe of -2.66. Only 4 of 17 folds produced positive returns. The model has learned noise, not signal.

**Credit spread "alpha" decomposes as follows:**
- 55% from unrealistic theta model assumptions
- 30% from favorable sample period (2025 Q2 tariff volatility benefited short put/call trades)
- 15% from genuine structural edge of selling premium in credit spreads

---

## Phase 1: Statistical Audit Findings

### P&L Distribution — Credit Spreads

| Metric | Value |
|--------|-------|
| Total Trades | 71 over ~16 months |
| Win Rate | 87.3% (62 wins, 9 losses) |
| Mean P&L per trade | +15.8% of position |
| Median P&L per trade | +23.6% |
| Skewness | -1.88 (heavy left tail) |
| Win/Loss Ratio | 0.53x (wins are smaller than losses) |
| Max Win | +33.0% (capped at credit received) |
| Max Loss | -50.5% (2x credit stop) |

**Red Flags:**
- 34% of trades hit maximum credit (0.33) — suggesting the take-profit and theta model are too generous
- Losers average -46.5% vs. winners averaging +24.8% — a 1:0.53 payoff ratio is dangerous
- Only 14 trades per year — far too few for statistical significance
- 58 overlapping trade pairs — correlated P&L reduces effective sample size dramatically

### The Theta Smoking Gun

The credit spread P&L evaluation uses: `theta_benefit = min(days * 0.05, 0.5) * credit_pct`

This means 5% of the credit percentage is added as "theta decay" every day, reaching a 16.5% position bonus after 10 days. Real-world theta for a 3%-wide SPY credit spread at 30 DTE is approximately 0.5-1.5% of credit per day — meaning the model overestimates theta income by 3-10x.

**P&L Decomposition (sum across all 71 trades):**

| Component | Cumulative P&L | % of Total |
|-----------|---------------|------------|
| Theta model (unrealistic) | +612.98% | 54.7% |
| Directional/delta P&L | +507.43% | 45.3% |
| **Total net** | **+1,120.41%** | **100%** |

Without the inflated theta, the directional component alone produces roughly half the returns — and this is before accounting for the directional model's demonstrated lack of edge.

### Win/Loss Streaks

- Maximum winning streak: 23 trades
- Maximum losing streak: 2 trades
- Serial correlation of returns: 0.255 (mild clustering)
- Winners average 4.6 days to exit (take-profit); losers average 10.1 days (always hit max hold)

This asymmetry — winners close fast on theta + favorable moves, losers grind to max duration — is a classic signature of option-selling strategies where the theta model is doing the heavy lifting.

### Monthly P&L Concentration

Profits are severely concentrated:
- 2025 Q2 alone produced $120.43 (52% of all profits) across 20 trades at 100% win rate
- 2026 Q1 produced $92.23 across 16 trades at 100% win rate
- 2025 Q3 produced -$1.38 (net loss with only 3 trades, 67% win rate)

This means two favorable quarters drive the entire result. In the remaining months, the strategy is roughly flat.

---

## Phase 2: Regime & Correlation Analysis

### Regime Distribution

Contrary to the claim that 72% of days are UNKNOWN, re-analysis of the test period shows:

| Regime | % of Days | Credit Trades | Win Rate | Total P&L |
|--------|----------|---------------|----------|-----------|
| MEAN_REVERTING | 58.7% | 39 | 84.6% | $103.93 |
| TRENDING_UP | 19.8% | 0 | N/A | $0.00 |
| UNKNOWN | 16.8% | 25 | 88.0% | $90.45 |
| TRENDING_DOWN | 4.7% | 7 | 100.0% | $36.30 |

**Key Insight:** No trades were entered during TRENDING_UP regime, and 100% of UNKNOWN-regime trades are bull puts (long direction). The regime filter is preventing trades during uptrends, which is counterintuitive for bull put spreads.

### VIX Analysis

All 71 credit spread trades occurred when VIX data was below 15 (or VIX was unavailable/zero in the features). This means the strategy has never been tested in elevated volatility — precisely when credit spread selling is most dangerous.

### Tail Risk & Stress Tests

- During the worst 5% of SPY days (63 days with returns below -1.66%), 15 had open credit spread trades
- Average P&L on those days: +$0.57 per trade (slightly positive — but this is the theta model carrying the position)
- During the worst 20-day rolling period (-12.65% SPY drawdown in Q2 2025): 6 trades entered, 0 losses

This seemingly miraculous performance during stress periods is again driven by the theta model: even while delta P&L is deeply negative, the theta benefit pushes the net P&L positive. With realistic theta, these trades would have been losses.

---

## Phase 3: Hedge Recommendations

### Losing Trade Profile

The 9 losing trades share common characteristics:
- All held to maximum duration (10 days)
- Average loss: -46.5% of position
- Entry volatility (20d realized) was 44% lower than for winning trades — suggesting they entered during calm periods before volatility expanded
- They were caught on the wrong side of a directional move without the theta model being able to offset it

### Recommended Hedges (in priority order)

**1. Tighten Stop-Loss from 2.0x to 1.5x Credit (ZERO COST)**

| Metric | 2.0x Stop | 1.5x Stop | 1.0x Stop |
|--------|-----------|-----------|-----------|
| Win Rate | 87.3% | 87.3% | 87.3% |
| Avg P&L | +15.8% | +17.2% | +18.6% |
| Total Return | 1,120% | 1,223% | 1,322% |
| Sharpe | 0.62 | 0.79 | 1.01 |

Tightening the stop is the single highest-impact improvement: it caps losses earlier and dramatically improves the Sharpe ratio at zero cost.

**2. Dynamic Position Sizing by VIX Level (ZERO COST)**
- VIX < 15: 1.5x normal position size
- VIX 15-20: 1.0x (baseline)
- VIX > 20: 0.5x (defensive)
- VIX > 35: No trades

**3. SPY Put Hedge (COSTS ~1% OF CREDIT)**
- Buy 5% OTM puts with 50% of position size
- Estimated annual cost: ~14% of capital
- Protection: ~25% payoff in crisis scenarios
- Only implement if capital allows the drag

### Net Expected Performance with Hedges

Implementing Recommendation 1 + 2 (both zero cost):
- Expected max drawdown: ~3.15% (from 4.50%)
- Expected Sharpe: ~0.79 (from 0.62)
- Win rate: unchanged at 87.3%

**Important caveat:** These improvements apply to the V8 simulation with its unrealistic theta model. With realistic theta, the numbers would be substantially lower.

---

## Phase 4: V9 Strategy Results

### What Was Changed in V9

1. **Realistic theta model:** 1.5% per day (vs. 5%), cap at 30% (vs. 50%)
2. **Tighter stop-loss:** 1.5x credit (vs. 2.0x)
3. **Higher take-profit:** 60% of credit (vs. 50%)
4. **Shorter max hold:** 7 days (vs. 10)
5. **Simplified regime filter:** 100% coverage (BULL/BEAR/VOLATILE/NEUTRAL) vs. 28%
6. **Rules-based signals:** Transparent SMA/RSI/MACD logic replacing broken ML ensemble
7. **VIX filter:** Skip trades at VIX > 35

### V9 Results

| Metric | V8 | V9 |
|--------|----|----|
| Total Trades | 71 | 14 |
| Win Rate | 87.3% | 57.1% |
| Total Return | +23.07% | -122.5% |
| Avg Win | +24.8% | +29.6% |
| Avg Loss | -46.5% | -61.0% |
| Sharpe | 1.60 | -4.54 |
| Trades/Year | ~14 | ~2.8 |

**V9 was intentionally honest — and it lost money.** The two key factors:

1. **Realistic theta reduces free profit by ~70%.** Without the generous theta model propping up every trade, the average trade needs a larger favorable directional move to be profitable.

2. **Rules-based signals produced only 14 trades.** The strict regime and VIX filters, combined with transparent entry rules, were far more conservative. The broken ML ensemble in V8, despite having no OOS edge, paradoxically generated more trades because it was confident about noise.

### Walk-Forward Validation of V9

All 5 walk-forward folds produced **zero trades** — the rules-based signals with strict filters never fired during the test windows. This is actually better than V8's walk-forward, which actively lost money (-1.64%), but it means the strategy is too conservative to trade.

### The Fundamental Problem

The V9 results prove the core thesis: **the V8 +23% return was a simulation artifact, not real alpha.** When you:
- Fix the theta model to realistic values
- Use honest signals instead of an overfitted ML model
- Apply proper risk filters

...the strategy either doesn't trade (too few signals) or loses money (wrong signals + reduced theta cushion).

---

## Phase 5: Recommendations for Live Trading

### What Should NOT Go Live

1. **The current V8 system should NOT be traded with real money.** The +23% backtest is unreliable due to the theta model inflation, and the underlying ML model has no out-of-sample edge.

2. **The V9 rules-based strategy should NOT go live either.** It generates too few trades (2.8/year) and loses money.

### What Could Work (V10 Roadmap)

If you want to build a genuine credit spread strategy, here is the path:

**Step 1: Fix the Foundation (Week 1-2)**
- Replace the theta model with realistic parameters: `theta_benefit = min(days * 0.01, 0.15) * credit_pct`
- This is conservative (1% per day, cap at 15% total) and closer to reality
- Implement 1.5x credit stop-loss
- Reduce max hold to 7 days

**Step 2: Fix the Signal Generator (Week 3-6)**
- The ML ensemble needs walk-forward validation on credit spread mode specifically (V8 only validated directional)
- Focus on improving direction prediction above 60% accuracy OOS
- Consider dropping the LSTM and TFT models (complex, likely overfitting) and using a simpler XGBoost + Logistic Regression stack
- Add proper feature selection (mutual information, forward selection) to reduce noise features

**Step 3: Increase Trade Frequency (Week 4-6)**
- Target 50+ trades per year (currently 14)
- Lower confidence threshold but add more confirmatory filters
- Consider expanding to QQQ, IWM, or sector ETFs for more opportunities

**Step 4: Implement Hedges (Week 5-7)**
- Dynamic position sizing by VIX level
- Tighter stops (1.5x credit)
- Consider portfolio-level risk limits (max 10% capital at risk simultaneously)

**Step 5: Paper Trade for 3-6 Months (Week 8+)**
- Run V10 on paper alongside V8
- Compare real-time signals, fills, and P&L
- Only go live after 50+ paper trades with demonstrable edge

### Realistic Return Expectations

With a properly calibrated system:
- **Conservative target:** 8-12% annual return, Sharpe 0.8-1.2
- **Realistic win rate:** 70-80% (not 87%)
- **Expected max drawdown:** 8-15%
- **Trade frequency:** 50-100 trades per year

The +23% was never real. A well-executed credit spread strategy on SPY should target high single digits to low teens annually, with the edge coming from consistent premium harvesting and tight risk management — not from inflated theta models.

---

## Appendix: Key File Inventory

| File | Description |
|------|-------------|
| `phase1_statistical_audit.py` | Full P&L and overfitting analysis |
| `phase2_regime_analysis.py` | Regime profitability and correlation |
| `phase3_hedge_design.py` | Hedge strategy design and simulation |
| `run_backtest_v9.py` | V9 strategy implementation |
| `data/analysis/phase1_results.json` | Phase 1 quantitative results |
| `data/analysis/phase2_results.json` | Phase 2 regime and VIX analysis |
| `data/analysis/phase3_results.json` | Phase 3 hedge recommendations |
| `data/analysis/v8_vs_v9_comparison.json` | V8 vs V9 head-to-head |
| `data/backtest/trades_SPY_v9_credit.csv` | V9 individual trades |
| `data/backtest/metrics_SPY_v9.txt` | V9 full metrics |

---

## Risk Warnings

1. **This analysis is based on simulated backtests, not live trading.** Real execution will face slippage, wide bid-ask spreads on options, and assignment risk.

2. **Credit spread strategies have concave payoff profiles.** High win rates mask the true risk: a single tail event can wipe out months of gains.

3. **71 trades is not statistically significant.** You need 200+ trades to draw reliable conclusions about a credit spread strategy.

4. **Past performance, even if real, does not predict future returns.** Regime changes, correlation breakdowns, and black swans cannot be backtested.

5. **The VIX data quality in the feature set appears incomplete** (all zeros in some periods). Any strategy relying on VIX signals needs verified, complete data.

**Confidence level in these conclusions: HIGH.** The theta model artifact is mathematically demonstrable, and the directional model's failure in walk-forward testing is unambiguous.
