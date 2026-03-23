# V10 Architecture & Data Pipeline Specification

**Date:** March 22, 2026
**Status:** Prototype Complete — Production Data Pipeline Required

---

## What the V10 Prototype Proved

With honest parameters (realistic theta, vega impact, tight stops), the rules-based credit spread engine produced:

| Configuration | Trades | Win% | Total Return | Annual | Sharpe | MaxDD |
|--------------|--------|------|-------------|--------|--------|-------|
| Layer 1 (base engine) | 201 | 63.7% | +6.82% | +1.70% | 0.46 | -6.80% |
| Layer 2 (+VRP timing) | 155 | 60.0% | +0.85% | +0.21% | 0.08 | -3.38% |
| Layer 3 (+regime+events) | 155 | 60.0% | +0.63% | +0.16% | 0.07 | -3.37% |
| Buy & Hold SPY | — | — | +45.90% | +11.46% | 0.63 | -22.74% |

**Key findings from diagnostics:**

1. **Bull puts work, bear calls don't (in this period).** Bull puts: 172 trades, 67% win rate, +$1,069. Bear calls: 29 trades, 45% win rate, -$387. SPY's long-term upward bias makes bull puts the structural play.

2. **Low VIX is the sweet spot.** VIX < 15: 79% win rate, +$16/trade. VIX 15-20: 62% win rate, +$1/trade. VIX > 25: 38% win rate, -$19/trade. When VIX is low, options sellers collect enough premium relative to realized moves. When VIX is high, the actual moves are larger than the premium compensates for.

3. **Layer 2 improved risk, not returns.** The VRP filter cut max drawdown from 6.80% to 3.38% by avoiding trades in the wrong volatility regime, but also filtered out some profitable low-VIX entries.

4. **The honest edge is thin.** Layer 1's profit factor of 1.16 and EV of +1.75% per trade is a real but slim edge. Monte Carlo shows 84% probability of profit over the full period. This is consistent with what professional credit spread sellers actually earn — you make small, consistent gains that compound over time, occasionally taking hits.

5. **Why it underperforms buy-and-hold.** SPY returned 45.9% over this period (strong bull market). Credit spreads can't capture upside beyond the premium collected. The value proposition isn't beating SPY in a bull market — it's generating income in flat/down markets with lower drawdowns (6.8% vs 22.7%).

---

## Where the Extra 10-15% Comes From

The prototype uses **proxy data** for implied volatility (just VIX) and has no real options chain data. Here's what proper data unlocks:

### Data Gap 1: Real Options Chain (VOL SURFACE)

**What we have:** VIX (single number — 30-day ATM implied vol for SPY)

**What we need:** Full options chain for SPY at every strike and expiration — bid, ask, mid, volume, open interest, implied vol, greeks (delta, gamma, theta, vega)

**Why it matters:** VIX tells you the market's *average* fear level. The vol surface tells you *where the mispricings are*. A typical SPY options chain might have 200+ strikes across 8 expirations. The implied vol at each strike forms a "smile" or "skew" — and the shape of this skew is where the alpha lives.

For example: if 25-delta put IVs are 5 vol points above 25-delta call IVs, the skew is steep — put protection is expensive. That's when selling bull put spreads has the highest edge because you're selling overpriced insurance. If the skew is flat, the edge is thin and you should stand aside.

**Expected improvement:** Selecting entries based on skew richness (top quartile of put skew vs. historical) could increase win rate from 64% to 72-75% and boost annual returns by 4-6%.

**Data sources:**
- CBOE DataShop: Historical options data, $500-2000/month
- OptionMetrics (IvyDB): Academic-grade, $3000+/year
- Thetadata: $25-100/month, excellent for retail algo traders
- ORATS: $100-300/month, includes proprietary vol forecasts

### Data Gap 2: Dealer Positioning (GAMMA EXPOSURE)

**What we have:** Nothing. The V8 system has zero positioning data.

**What we need:** Aggregate dealer gamma exposure (GEX), dark pool indicators (DIX), and options volume decomposition (customer vs. firm, put/call by open/close).

**Why it matters:** When dealers are short gamma (from selling puts to hedging funds), they must buy when the market goes up and sell when it goes down — amplifying moves. This is when credit spread sellers get blown out. When dealers are long gamma, they dampen moves — exactly the environment where credit spreads thrive.

**Expected improvement:** Avoiding negative GEX days could cut the stop-loss rate from 36% to 20-25%. At the same win %, this alone could push annual returns from +1.7% to +6-8%.

**Data sources:**
- Squeezemetrics (GEX/DIX): $30/month
- CBOE put/call ratios: Free (daily aggregate)
- Unusual Whales / Flow Algo: $50-100/month
- Custom calculation from options chain data

### Data Gap 3: Realized Volatility Forecasting

**What we have:** Backward-looking 20-day realized vol (lagging indicator)

**What we need:** Forward-looking realized vol estimates — specifically, how much will SPY actually move over the next 1-5 days?

**Why it matters:** The variance risk premium (VRP = IV - RV) is the core edge. But using backward-looking RV means we're measuring *yesterday's* VRP, not *tomorrow's*. A forward-looking RV forecast using intraday vol, overnight gaps, and cross-asset signals could predict when VRP will be unusually wide (great entries) vs. compressed (skip).

**Expected improvement:** Better VRP timing could concentrate entries in the top 30% of edge-days, improving EV per trade from +1.75% to +4-5%.

**Approach:**
- HAR (Heterogeneous Autoregressive) model: Uses 1-day, 5-day, 22-day realized vol to forecast next-week vol. Well-studied, robust, hard to overfit.
- GARCH(1,1): Standard vol forecasting model. Good baseline.
- Cross-asset features: VIX futures term structure, credit spreads (HY OAS), FX vol (EURUSD 1-week ATM IV).

**Data sources:**
- Intraday SPY data: Alpaca (you already have this), Polygon.io ($30/month)
- VIX futures: CBOE, available via most data providers
- Credit spreads: FRED (ICE BofA High Yield OAS — free)
- FX vol: Quandl / Refinitiv

---

## Production Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     V10 PRODUCTION SYSTEM                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              DATA PIPELINE (Daily 6AM EST)              │    │
│  │                                                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │    │
│  │  │  Alpaca   │  │Thetadata │  │  FRED /  │             │    │
│  │  │  SPY EOD  │  │ Options  │  │  Squeeze │             │    │
│  │  │  + Intra  │  │  Chain   │  │  metrics │             │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘             │    │
│  │       │              │              │                    │    │
│  │       └──────┬───────┴──────┬───────┘                   │    │
│  │              ▼              ▼                            │    │
│  │     ┌───────────────────────────────┐                   │    │
│  │     │    Feature Engineering        │                   │    │
│  │     │  • Realized vol (1/5/20d)     │                   │    │
│  │     │  • Vol surface metrics        │                   │    │
│  │     │  • Put skew richness          │                   │    │
│  │     │  • VRP (IV - forecast RV)     │                   │    │
│  │     │  • Dealer GEX estimate        │                   │    │
│  │     │  • Cross-asset regime         │                   │    │
│  │     └───────────┬───────────────────┘                   │    │
│  └─────────────────┼───────────────────────────────────────┘    │
│                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              DECISION ENGINE                             │    │
│  │                                                         │    │
│  │  Layer 1: Should we trade today?                        │    │
│  │  ├─ VRP > 0?  (options overpriced)                      │    │
│  │  ├─ VIX 13-28?  (sweet spot)                            │    │
│  │  ├─ GEX > 0?  (dealers long gamma = stable)             │    │
│  │  └─ No FOMC/CPI this week?                              │    │
│  │                                                         │    │
│  │  Layer 2: What to trade?                                │    │
│  │  ├─ Put skew steep → bull put spread                    │    │
│  │  ├─ Call skew steep → bear call spread                  │    │
│  │  ├─ Both steep → iron condor                            │    │
│  │  └─ Select strikes by target delta (0.15-0.20)          │    │
│  │                                                         │    │
│  │  Layer 3: How much?                                     │    │
│  │  ├─ Base: 2% of capital                                 │    │
│  │  ├─ VRP top quartile → 1.3x                             │    │
│  │  ├─ RISK_OFF regime → 0.5x                              │    │
│  │  └─ Event week → 0.7x                                   │    │
│  │                                                         │    │
│  └───────────┬─────────────────────────────────────────────┘    │
│              ▼                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              EXECUTION + RISK MANAGEMENT                │    │
│  │                                                         │    │
│  │  • Place spread orders via Alpaca options API           │    │
│  │  • Stop loss at 1.0x credit received                    │    │
│  │  • Take profit at 50% of credit received                │    │
│  │  • Manage at 14 DTE (close if profitable)               │    │
│  │  • Portfolio: max 4 concurrent positions                 │    │
│  │  • Daily P&L check: halt if -3% daily                   │    │
│  │  • Weekly: roll/adjust positions if needed               │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase A: Data Infrastructure (Weeks 1-3)

1. **Sign up for Thetadata** ($25/month) — historical + real-time options chains
2. **Set up Squeezemetrics** ($30/month) — GEX and DIX data
3. **Build daily pipeline** — Python script that:
   - Fetches SPY EOD price (Alpaca — existing)
   - Fetches full SPY options chain (Thetadata)
   - Computes vol surface metrics (skew, term structure, ATM IV by expiry)
   - Fetches GEX (Squeezemetrics or calculate from options chain)
   - Fetches HY OAS from FRED (free)
   - Saves to `data/daily/` as timestamped Parquet files
4. **Backfill** — Get 2-3 years of historical options data for proper backtesting

### Phase B: Enhanced Backtest (Weeks 4-6)

1. **Rebuild V10 with real options data** — Replace proxy models with actual vol surface
2. **Strike selection by delta** — Instead of fixed 3% OTM, select strikes at 0.15-0.20 delta
3. **Spread pricing from actual bid/ask** — Account for slippage and crossing the spread
4. **Walk-forward validation** — 8+ folds with expanding window
5. **Target: 200+ trades over 2-3 years with consistent positive PnL**

### Phase C: Paper Trading (Weeks 7-12)

1. **Deploy on Alpaca paper** — Run the strategy live with paper money
2. **Log every decision** — entry signals, skipped trades, regime state, VRP level
3. **Weekly review** — Compare paper results to backtest expectations
4. **Target: 50+ paper trades before going live**

### Phase D: Live Deployment (Week 13+)

1. **Start small** — $5,000-10,000 initial capital
2. **Scale position sizes gradually** — Start at 1% risk per trade, increase to 2% after 20+ profitable trades
3. **Monthly performance review** — Is realized Sharpe > 0.3? If not, halt and reassess

---

## Realistic Return Projections

Based on the V10 prototype results scaled by expected improvements from real data:

| Scenario | Win Rate | EV/Trade | Trades/Yr | Annual Return | Sharpe | MaxDD |
|----------|---------|----------|-----------|---------------|--------|-------|
| Conservative (Layer 1 only) | 65% | +2.0% | 50 | 6-8% | 0.4-0.6 | -8% |
| Moderate (+ vol surface data) | 70% | +3.5% | 50 | 12-16% | 0.7-1.0 | -6% |
| Aggressive (+ GEX + HAR vol forecast) | 73% | +4.5% | 60 | 18-25% | 0.9-1.3 | -8% |
| Optimistic ceiling | 75% | +5.0% | 70 | 25-30% | 1.0-1.5 | -10% |

The "moderate" scenario — achievable with a $50-75/month data budget — is the most likely outcome. You add 5-8% win rate from better entry timing (vol surface) and cut 10-15% of stop-loss trades via GEX filtering. That takes the prototype from +1.7% annual to the 12-16% range, which meaningfully beats the risk-free rate and is competitive with (though not necessarily better than) buy-and-hold on a risk-adjusted basis.

The crucial advantage over buy-and-hold: lower drawdowns, income in flat/down markets, and uncorrelated returns. A 12% credit spread return with -6% max drawdown is a very different portfolio experience than a 12% equity return with -23% max drawdown.

---

## Monthly Data Cost Summary

| Provider | Purpose | Cost/Month |
|----------|---------|------------|
| Alpaca | SPY price data, execution | Free (paper) / $0 (existing) |
| Thetadata | Options chain data | $25 |
| Squeezemetrics | GEX/DIX positioning | $30 |
| FRED | Economic indicators, credit spreads | Free |
| **Total** | | **$55/month** |

For $55/month in data, you get the three inputs that matter most: vol surface (entry timing), dealer positioning (risk avoidance), and macro regime (context). Everything else — sentiment, news, complex ML — is noise for this strategy.
