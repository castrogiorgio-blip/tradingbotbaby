# TradingBot ML — Project Blueprint

## 1. Executive Summary

This project builds a local, automated ML-powered trading system that:

- Scrapes financial data, economic indicators, and news every morning before 7 AM EST
- Runs ML models to predict next-day price movements
- Generates options trade recommendations (calls/puts) with stop-loss and take-profit levels
- Serves a local web dashboard with three portfolio sections: Blue-Chip, High-Risk/High-Reward, and Longer-Horizon
- Optionally executes trades autonomously via broker API

Starting capital: **$1,000** · Goal: **Steady, risk-managed returns**

> **Important disclaimer:** This is a technical project for educational and personal use. All trading involves risk. Past performance of any model does not guarantee future results. You should never risk money you can't afford to lose.

---

## 2. Recommended Stack (and Why)

### 2.1 Asset Focus: US Stocks & ETFs with Options

**Why this is the highest-success starting point for your project:**

- **Data richness** — More free, high-quality historical data is available for US equities than any other asset class. Every API, every open-source project, and every tutorial defaults to US stocks.
- **Options accessibility** — With Alpaca's commission-free options API, you can trade single-leg calls/puts and even multi-leg strategies (spreads, straddles) at zero commission.
- **$1K compatibility** — Fractional shares let you diversify even with $1K. For options, you can start with low-priced contracts on liquid ETFs (SPY, QQQ, IWM) where premiums can be as low as $10–50 per contract.
- **Expandable later** — Once the system is stable, we can add a forex module via MetaTrader 5 (which does have a Python API, but is primarily designed for forex/CFDs, not US stock options).

### 2.2 Broker: Alpaca Markets

| Feature | Alpaca | Interactive Brokers | MetaTrader 5 |
|---|---|---|---|
| Commission-free stocks | ✅ | ❌ | N/A (forex) |
| Commission-free options | ✅ | ❌ | ❌ (no options) |
| Python SDK | ✅ Excellent | ✅ Good | ✅ Windows-only |
| Paper trading | ✅ Free | ✅ Free | ✅ Free |
| Options API | ✅ Level 1–3 | ✅ Full | ❌ |
| Minimum deposit | $0 | $0 | Broker-dependent |
| REST + WebSocket | ✅ | ✅ | Via MetaApi cloud |
| Best for this project | ⭐ Yes | Phase 2 option | Forex addon later |

**Verdict:** Start with **Alpaca** for stocks/options. Add MetaTrader 5 later if you want a forex module.

### 2.3 ML Framework: FinRL + Custom Models

| Framework | Strengths | Weaknesses |
|---|---|---|
| **FinRL** | Purpose-built for trading RL, great tutorials, active community | RL can be overkill for directional predictions |
| **QLib (Microsoft)** | Excellent data pipeline, traditional ML + RL, production-grade | Steeper learning curve |
| **Custom (our approach)** | Tailored to your exact needs | More work upfront |

**Our approach:** Use **FinRL** as a reference architecture and borrow its environment/backtesting modules, but build a **custom prediction pipeline** combining:

1. **Gradient Boosting (XGBoost/LightGBM)** — For tabular features (technical indicators, economic data). These are the workhorses of quantitative finance and consistently outperform deep learning on structured data.
2. **LSTM/Transformer** — For sequential price patterns where temporal context matters.
3. **NLP Sentiment Model (FinBERT)** — For news/announcement impact scoring. FinBERT is a pre-trained model specifically fine-tuned on financial text.
4. **Ensemble Layer** — Combines all three model outputs into a final signal with confidence score.

### 2.4 Data Sources (All Free Tiers)

| Data Type | Source | Free Tier |
|---|---|---|
| Historical prices (daily) | **Alpaca Market Data** | Included with account |
| Real-time quotes | **Alpaca** | Included (IEX feed) |
| Technical indicators | **Computed locally** | Using `ta-lib` or `pandas-ta` |
| Economic indicators | **FRED API** (Federal Reserve) | Unlimited, free |
| Earnings/fundamentals | **Alpha Vantage** | 25 requests/day (free) |
| News + sentiment | **Finnhub** | 60 calls/min (free) |
| SEC filings/events | **SEC EDGAR** | Free, unlimited |
| Options chain data | **Alpaca** | Included with account |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL WEB DASHBOARD                       │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │Blue-Chip │  │ High-Risk/    │  │  Longer Horizon      │  │
│  │Portfolio │  │ High-Reward   │  │  (Weekly/Monthly)    │  │
│  └──────────┘  └───────────────┘  └──────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Trade Log · P&L Tracker · Model Confidence Scores  │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Flask/FastAPI  │
              │  Local Server   │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐  ┌───▼────┐
    │ Trading │  │ ML Engine │  │Scheduler│
    │ Engine  │  │           │  │(APScheduler
    │(Alpaca) │  │ Predict & │  │ or cron)│
    └─────────┘  │ Recommend │  └─────────┘
                 └─────┬─────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐  ┌───▼────────┐
    │Price &  │  │Economic   │  │News &      │
    │Technical│  │Indicators │  │Sentiment   │
    │Data     │  │(FRED)     │  │(Finnhub +  │
    │(Alpaca) │  │           │  │ FinBERT)   │
    └─────────┘  └───────────┘  └────────────┘
```

### 3.1 Component Breakdown

**A. Data Pipeline (runs daily at 6:00 AM EST)**

```
6:00 AM  → Fetch latest daily bars (Alpaca)
6:05 AM  → Compute 50+ technical indicators (pandas-ta)
6:10 AM  → Fetch economic indicators (FRED: CPI, jobs, GDP, Fed rate)
6:15 AM  → Fetch overnight news + compute sentiment (Finnhub + FinBERT)
6:20 AM  → Fetch options chain snapshots (Alpaca)
6:25 AM  → Merge all features into prediction-ready dataset
6:30 AM  → Run ML models → Generate trade signals
6:45 AM  → Post recommendations to dashboard
6:50 AM  → (Optional) Execute trades via Alpaca API
```

**B. ML Engine**

Three specialized models feed into one ensemble:

1. **XGBoost Classifier** — Trained on 200+ features (price patterns, volume, indicators, economic data). Predicts direction (up/down) with probability.
2. **LSTM Network** — Trained on 60-day rolling windows of OHLCV + indicator sequences. Captures temporal patterns.
3. **FinBERT Sentiment Scorer** — Processes last 24h of news headlines per ticker. Outputs sentiment score (-1 to +1).
4. **Ensemble Meta-Model** — Takes the three outputs + their historical accuracy as features. Produces final signal: BUY_CALL, BUY_PUT, or HOLD, plus a confidence score (0–100%).

**C. Trade Signal → Options Recommendation**

Once we have a directional prediction with confidence:

```
IF confidence > 70% AND direction = UP:
    → Recommend CALL option
    → Select strike: nearest ATM or slightly OTM (delta ~0.40)
    → Expiration: 7–14 DTE (days to expiration)
    → Stop-loss: 30% of premium paid
    → Take-profit: 50-100% of premium paid

IF confidence > 70% AND direction = DOWN:
    → Recommend PUT option (same logic)

IF confidence < 70%:
    → HOLD / No trade
```

**D. Portfolio Sections**

| Section | Strategy | Typical Holdings | Rebalance |
|---|---|---|---|
| **Blue-Chip** | Conservative, high-confidence only | SPY, QQQ, AAPL, MSFT options + shares | Weekly |
| **High-Risk / High-Reward** | Lower confidence threshold, OTM options | Small-cap, volatile tickers, earnings plays | Daily |
| **Longer Horizon** | Swing trades, 30-90 DTE options, LEAPS | Sector ETFs, macro-driven positions | Monthly |

**E. Risk Management (Critical for $1K)**

- **Position sizing:** Never risk more than 2-5% of portfolio on a single trade ($20-50)
- **Daily loss limit:** Stop trading if daily losses exceed 5% of portfolio
- **Diversification:** Maximum 3 open positions at a time with $1K
- **Options-specific:** Only buy options (no selling/writing — limited risk to premium paid)
- **Paper trade first:** Run the system in paper-trading mode for at least 4-8 weeks before using real money

---

## 4. Technology Stack

```
Language:       Python 3.11+
Web Framework:  Flask (simple) or FastAPI (async, modern)
Frontend:       HTML/CSS/JS with Chart.js or Plotly for charts
                (Single-page app, no framework needed for personal use)
ML Libraries:   scikit-learn, XGBoost, LightGBM, PyTorch (LSTM)
NLP:            transformers (HuggingFace), FinBERT model
Data:           pandas, numpy, pandas-ta (technical indicators)
Scheduling:     APScheduler (in-app) or system cron
Database:       SQLite (simple, no server needed, perfect for local use)
Broker API:     alpaca-py (official Alpaca Python SDK)
Data APIs:      fredapi, finnhub-python, alpha_vantage
Backtesting:    backtrader or custom (using FinRL's environment)
```

---

## 5. Project Directory Structure

```
mytradingbot/
├── config/
│   ├── settings.yaml          # API keys, model params, thresholds
│   └── tickers.yaml           # Watchlist per portfolio section
├── data/
│   ├── raw/                   # Raw API responses (cached)
│   ├── processed/             # Feature-engineered datasets
│   ├── models/                # Saved trained models (.pkl, .pt)
│   └── predictions/           # Daily prediction logs
├── src/
│   ├── data_pipeline/
│   │   ├── price_fetcher.py       # Alpaca historical/realtime data
│   │   ├── indicator_engine.py    # Technical indicator computation
│   │   ├── economic_fetcher.py    # FRED economic data
│   │   ├── news_fetcher.py        # Finnhub news + FinBERT sentiment
│   │   ├── options_fetcher.py     # Options chain data
│   │   └── feature_builder.py     # Merge all into ML-ready features
│   ├── models/
│   │   ├── xgboost_model.py       # Gradient boosting classifier
│   │   ├── lstm_model.py          # LSTM sequential model
│   │   ├── sentiment_model.py     # FinBERT wrapper
│   │   ├── ensemble.py            # Meta-model combining all three
│   │   └── trainer.py             # Training/retraining logic
│   ├── trading/
│   │   ├── signal_generator.py    # Model output → trade signal
│   │   ├── options_selector.py    # Signal → specific option contract
│   │   ├── risk_manager.py        # Position sizing, stop-loss logic
│   │   └── executor.py            # Alpaca API trade execution
│   ├── backtest/
│   │   ├── backtester.py          # Historical strategy testing
│   │   └── metrics.py             # Sharpe, drawdown, win rate, etc.
│   └── web/
│       ├── app.py                 # Flask/FastAPI main app
│       ├── templates/
│       │   └── dashboard.html     # Main dashboard page
│       └── static/
│           ├── css/
│           └── js/
├── scheduler/
│   └── daily_pipeline.py      # Orchestrates the 6AM daily run
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_analysis.ipynb
├── tests/
├── requirements.txt
├── docker-compose.yml         # Optional: containerize everything
└── README.md
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1–2)
- Set up project structure and Python environment
- Create Alpaca paper trading account
- Build price data fetcher + technical indicator engine
- Build economic data fetcher (FRED)
- Store everything in SQLite
- **Deliverable:** Can fetch and store historical data for any ticker

### Phase 2: ML Models (Weeks 3–5)
- Feature engineering (200+ features from price, volume, indicators, economic data)
- Train XGBoost baseline model on 3+ years of historical data
- Build and train LSTM model on price sequences
- Integrate FinBERT for news sentiment scoring
- Build ensemble meta-model
- **Deliverable:** Models that predict next-day direction with measurable accuracy

### Phase 3: Backtesting (Weeks 5–6)
- Build backtesting framework
- Test strategy on 2+ years of out-of-sample data
- Measure: Sharpe ratio, max drawdown, win rate, profit factor
- Tune risk parameters (stop-loss %, take-profit %, position sizing)
- **Deliverable:** Backtest report showing historical performance

### Phase 4: Web Dashboard (Weeks 6–8)
- Build Flask/FastAPI local server
- Create dashboard with three portfolio sections
- Add trade recommendation cards (ticker, direction, strike, expiry, SL/TP)
- Add P&L tracker and model confidence visualizations
- Add historical performance charts
- **Deliverable:** Working local dashboard at localhost:5000

### Phase 5: Automation (Weeks 8–10)
- Set up daily scheduler (6 AM EST pipeline)
- Connect to Alpaca paper trading for automated execution
- Add notification system (email/desktop alerts)
- Run in paper-trading mode for 4–8 weeks
- **Deliverable:** Fully automated daily pipeline in paper-trading mode

### Phase 6: Live Trading (After 4–8 Weeks of Paper Trading)
- Analyze paper trading results
- Switch to live trading with small position sizes
- Monitor and refine continuously
- **Deliverable:** Live automated trading with real capital

### Phase 7 (Future): Expansion
- Add forex module via MetaTrader 5
- Add commodity futures
- Add reinforcement learning (FinRL-based) for portfolio optimization
- Add more sophisticated options strategies (spreads, iron condors)

---

## 7. Key Open-Source Projects to Leverage

| Project | What We'll Use From It | GitHub |
|---|---|---|
| **FinRL** | Environment design, backtesting patterns, RL models (later) | AI4Finance-Foundation/FinRL |
| **QLib** | Data pipeline patterns, feature engineering recipes | microsoft/qlib |
| **FinBERT** | Pre-trained financial sentiment model | ProsusAI/finBERT |
| **alpaca-py** | Official Alpaca trading SDK | alpacahq/alpaca-py |
| **pandas-ta** | 130+ technical indicators | twopirllc/pandas-ta |
| **backtrader** | Backtesting engine (alternative to custom) | mementum/backtrader |

---

## 8. Answering Your MetaTrader Question

**Can MetaTrader execute trades autonomously?** Yes — MetaTrader 5 supports automated trading through Expert Advisors (EAs) and has a Python API (`metatrader5` package). However, there are important caveats:

- MT5's Python library only works on **Windows** (it communicates with the MT5 terminal via IPC)
- MT5 is primarily designed for **forex and CFDs**, not US stock options
- For US stock options (our primary focus), **Alpaca is far better suited**
- MT5 could be added as a **Phase 7 addon** specifically for forex pairs if you want to expand later

---

## 9. Realistic Expectations with $1,000

Being transparent about what's achievable:

- **Options trading with $1K is possible but tight.** Each option contract controls 100 shares, so you'll be buying low-priced contracts ($0.10–$0.50 per share = $10–$50 per contract). This limits you to liquid, lower-priced underlyings.
- **A good target: 1–3% monthly return** after the system is tuned. That's $10–$30/month at the start, compounding over time.
- **Expect losses during development.** This is why paper trading for weeks/months is essential.
- **The ML model won't be right every time.** Even a 55–60% win rate can be profitable with proper risk management (favorable risk/reward ratios).
- **Consider mixing:** Use some capital for share positions (via Blue-Chip section) and some for options (via the other sections). This provides more stability.

---

## 10. Immediate Next Steps

1. **Create an Alpaca account** at [alpaca.markets](https://alpaca.markets) — it's free, and you get paper trading immediately
2. **Get free API keys** from: FRED (fred.stlouisfed.org), Finnhub (finnhub.io), Alpha Vantage (alphavantage.co)
3. **We start building Phase 1** — I'll set up the project structure and data pipeline right here in your workspace

---

*Blueprint created: March 21, 2026*
*Project: TradingBot ML v0.1*
