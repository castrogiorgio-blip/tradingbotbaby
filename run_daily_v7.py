#!/usr/bin/env python3
"""
Daily Prediction Runner V7 — Credit Spread Signals + Volatility Filter

Generates daily predictions with three signal tiers:
  1. DIRECTIONAL: Stock-level BUY_CALL / BUY_PUT / HOLD
  2. CREDIT_SPREAD: Whether to sell a credit spread (bull put or bear call)
  3. VOL_ASSESSMENT: Whether today is a good day for options at all

Uses the proven V6 ensemble (57% accuracy, +0.88% directional, +18.21% credit spreads).

Usage:
    python3 run_daily_v7.py                     # All watchlist tickers
    python3 run_daily_v7.py --symbols SPY AAPL  # Specific tickers
"""
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_all_tickers, DATA_DIR
from src.data_pipeline.feature_builder import FeatureBuilder
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel

try:
    from src.models.tft_model import TFTModel
    HAS_TFT = True
except ImportError:
    HAS_TFT = False

try:
    from src.models.tabnet_model import TabNetModel
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False


# ═══════════════════════════════════════════════════════════
# V6 Ensemble — proven 57% accuracy, baked-in formula
# ═══════════════════════════════════════════════════════════

class V6Ensemble:
    """
    Self-contained ensemble with the V6 confidence formula.
    Does NOT depend on settings.yaml or ensemble.py.
    """

    WEIGHTS = {
        "xgboost": 0.30, "lstm": 0.15, "tft": 0.25,
        "tabnet": 0.20, "sentiment": 0.10,
    }
    CONFIDENCE_THRESHOLD = 0.15

    def predict(self, xgb_prob, lstm_prob, tft_prob=0.5,
                tabnet_prob=0.5, sent_prob=0.5, sent_conf=0.0):
        model_probs = {
            "xgboost": xgb_prob, "lstm": lstm_prob, "tft": tft_prob,
            "tabnet": tabnet_prob, "sentiment": sent_prob,
        }
        ew = self.WEIGHTS.copy()

        # Redistribute inactive model weights
        for name, prob, inactive in [
            ("sentiment", sent_prob, sent_conf < 0.1),
            ("tft", tft_prob, tft_prob == 0.5),
            ("tabnet", tabnet_prob, tabnet_prob == 0.5),
        ]:
            if inactive:
                w = ew.pop(name, 0)
                tr = sum(ew.values())
                if tr > 0:
                    for k in ew:
                        ew[k] += w * (ew[k] / tr)
                ew[name] = 0.0

        total_w = sum(ew.values())
        if total_w > 0:
            ew = {k: v / total_w for k, v in ew.items()}

        combined = sum(ew.get(k, 0) * model_probs[k] for k in model_probs)
        direction = "UP" if combined > 0.5 else "DOWN"

        # Agreement and conviction
        active = {k: v for k, v in model_probs.items() if ew.get(k, 0) > 0.05}
        agreements = sum(1 for p in active.values() if (p > 0.5) == (direction == "UP"))
        n_active = max(len(active), 1)
        agreement_ratio = agreements / n_active

        convictions = [abs(p - 0.5) * 2 for p in active.values()]
        avg_conv = np.mean(convictions) if convictions else 0
        max_conv = max(convictions) if convictions else 0

        # V6 confidence formula
        if agreement_ratio >= 0.8:
            confidence = 0.30 + avg_conv * 0.5 + max_conv * 0.2
        elif agreement_ratio >= 0.6:
            confidence = 0.15 + avg_conv * 0.5 + max_conv * 0.15
        else:
            confidence = avg_conv * 0.3
        confidence = min(confidence, 1.0)

        signal = ("BUY_CALL" if direction == "UP" else "BUY_PUT") \
                 if confidence >= self.CONFIDENCE_THRESHOLD else "HOLD"

        return {
            "signal": signal,
            "probability": float(combined),
            "confidence": float(confidence),
            "direction": direction,
            "agreement_ratio": float(agreement_ratio),
            "avg_conviction": float(avg_conv),
        }


# ═══════════════════════════════════════════════════════════
# Volatility Filter — assesses whether today is good for options
# ═══════════════════════════════════════════════════════════

class VolatilityFilter:
    """Scores today's volatility environment for options trading."""

    def assess(self, df, lookback=20):
        """
        Score the latest day's volatility environment.

        Returns dict with:
          - vol_score: 0.0-1.0 (higher = better for options)
          - is_high_vol: True if score >= 0.3
          - components: breakdown of score
        """
        if len(df) < lookback + 1:
            return {"vol_score": 0.0, "is_high_vol": False, "components": {}}

        latest = df.iloc[-1]
        recent = df.iloc[-(lookback + 1):]

        score = 0.0
        components = {}

        # 1. Realized vol ratio (current vs historical)
        returns = recent["close"].pct_change().dropna()
        if len(returns) >= 5:
            recent_vol = returns.iloc[-5:].std()
            hist_vol = returns.std()
            if hist_vol > 0:
                vol_ratio = recent_vol / hist_vol
                if vol_ratio > 1.5:
                    score += 0.3
                    components["vol_ratio"] = f"{vol_ratio:.2f} (HIGH)"
                elif vol_ratio > 1.2:
                    score += 0.15
                    components["vol_ratio"] = f"{vol_ratio:.2f} (elevated)"
                else:
                    components["vol_ratio"] = f"{vol_ratio:.2f} (normal)"

        # 2. VIX level
        vix_val = latest.get("vix", 0)
        if vix_val and not np.isnan(vix_val):
            if vix_val > 25:
                score += 0.3
                components["vix"] = f"{vix_val:.1f} (HIGH)"
            elif vix_val > 20:
                score += 0.15
                components["vix"] = f"{vix_val:.1f} (elevated)"
            else:
                components["vix"] = f"{vix_val:.1f} (low)"

        # 3. Today's absolute return
        if len(returns) > 0:
            today_ret = abs(returns.iloc[-1])
            if today_ret > 0.015:
                score += 0.25
                components["daily_move"] = f"{today_ret*100:.2f}% (large)"
            elif today_ret > 0.008:
                score += 0.1
                components["daily_move"] = f"{today_ret*100:.2f}% (moderate)"
            else:
                components["daily_move"] = f"{today_ret*100:.2f}% (small)"

        # 4. Event proximity
        for col in ["fomc_week", "earnings_week", "high_event_risk"]:
            if col in df.columns:
                val = latest.get(col, 0)
                if val and val > 0:
                    score += 0.2
                    components["event"] = col
                    break

        score = min(score, 1.0)

        return {
            "vol_score": round(score, 3),
            "is_high_vol": score >= 0.3,
            "components": components,
        }


# ═══════════════════════════════════════════════════════════
# Credit Spread Signal Generator
# ═══════════════════════════════════════════════════════════

def generate_credit_spread_signal(prediction, vol_assessment):
    """
    Given a directional prediction and volatility assessment,
    determine whether to recommend a credit spread trade.

    Credit spread strategy (from V7.1 backtest — +18.21% return):
      - Bullish → sell bull put spread (profit if stock stays above short put)
      - Bearish → sell bear call spread (profit if stock stays below short call)
      - Only trade on high-vol days (vol_score >= 0.3)
      - 3% spread width, 33% credit collection
      - TP at 50% of credit, SL at 2x credit, 10-day max hold
    """
    signal = prediction["signal"]
    confidence = prediction["confidence"]

    if signal == "HOLD" or not vol_assessment["is_high_vol"]:
        return {
            "credit_spread_signal": "NO_TRADE",
            "reason": "HOLD signal" if signal == "HOLD" else "Low volatility day",
            "spread_type": None,
            "parameters": None,
        }

    spread_type = "bull_put" if signal == "BUY_CALL" else "bear_call"

    # Position sizing guidance (% of capital per spread)
    if confidence >= 0.25:
        size_pct = 0.04  # 4% of capital
        size_label = "full"
    elif confidence >= 0.20:
        size_pct = 0.03  # 3%
        size_label = "standard"
    else:
        size_pct = 0.02  # 2%
        size_label = "reduced"

    return {
        "credit_spread_signal": "SELL_SPREAD",
        "spread_type": spread_type,
        "reason": f"{signal} + high vol (score={vol_assessment['vol_score']:.2f})",
        "parameters": {
            "spread_width": "3% of stock price",
            "target_credit": "~33% of spread width",
            "take_profit": "50% of credit received",
            "stop_loss": "2x credit received",
            "max_hold_days": 10,
            "suggested_dte": "14-21 days",
            "position_size": f"{size_pct*100:.0f}% of capital ({size_label})",
        },
    }


# ═══════════════════════════════════════════════════════════
# Main Daily Runner
# ═══════════════════════════════════════════════════════════

def run_daily_v7(symbols=None):
    """Generate V7 predictions with credit spread signals."""
    if symbols is None:
        symbols = get_all_tickers()

    print(f"\n{'='*60}")
    print(f"  DAILY PREDICTION RUN V7 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Features: V6 ensemble + credit spreads + vol filter")
    print(f"{'='*60}")

    builder = FeatureBuilder(use_finbert=False)
    ensemble = V6Ensemble()
    vol_filter = VolatilityFilter()

    all_predictions = {}

    for symbol in symbols:
        print(f"\n--- Processing {symbol} ---")
        try:
            # Build features (90 days for indicators + LSTM window)
            df = builder.build_features(symbol, days=120, include_news=False)
            if df.empty or len(df) < 70:
                print(f"  Insufficient data for {symbol}, skipping")
                continue

            # Load XGBoost
            xgb = XGBoostModel()
            try:
                xgb.load(symbol)
            except FileNotFoundError:
                try:
                    xgb.load("SPY")
                except FileNotFoundError:
                    print(f"  No XGBoost model for {symbol}, skipping")
                    continue

            X, _ = xgb.prepare_features(df)
            if len(X) == 0:
                continue

            # Align features with saved model (model may lack event calendar cols)
            if hasattr(xgb.model, 'feature_names_in_'):
                model_features = list(xgb.model.feature_names_in_)
            elif hasattr(xgb, 'feature_names') and xgb.feature_names:
                model_features = xgb.feature_names
            else:
                model_features = None

            if model_features is not None:
                # Drop any columns the model wasn't trained on
                extra_cols = [c for c in X.columns if c not in model_features]
                if extra_cols:
                    print(f"  Dropping {len(extra_cols)} extra features not in saved model: {extra_cols[:3]}...")
                    X = X[[c for c in X.columns if c in model_features]]
                # Add any missing columns as 0 (shouldn't happen normally)
                missing_cols = [c for c in model_features if c not in X.columns]
                for c in missing_cols:
                    X[c] = 0
                X = X[model_features]  # Ensure correct order

            latest_X = X.iloc[[-1]]
            xgb_prob = xgb.predict_proba(latest_X)[0]

            # TabNet
            tabnet_prob = 0.5
            if HAS_TABNET:
                try:
                    tabnet = TabNetModel()
                    tabnet.feature_names = xgb.feature_names
                    try:
                        tabnet.load(symbol)
                    except FileNotFoundError:
                        tabnet.load("SPY")
                    tabnet_prob = tabnet.predict_proba(latest_X)[0]
                except Exception:
                    tabnet_prob = 0.5

            # LSTM (3-run average)
            lstm_prob = 0.5
            try:
                lstm = LSTMModel()
                X_seq, _ = lstm.prepare_sequences(df, fit_scaler=True)
                if len(X_seq) > 0:
                    all_probs = []
                    for seed in [42, 123, 777]:
                        lr = LSTMModel()
                        lr.scaler = lstm.scaler
                        lr.n_features = lstm.n_features
                        try:
                            lr.load(symbol)
                        except FileNotFoundError:
                            try:
                                lr.load("SPY")
                            except FileNotFoundError:
                                break
                        probs = lr.predict_proba(X_seq[-1:])
                        if len(probs) > 0:
                            all_probs.append(probs[0])
                    if all_probs:
                        lstm_prob = np.mean(all_probs)
            except Exception:
                lstm_prob = 0.5

            # TFT
            tft_prob = 0.5
            if HAS_TFT:
                try:
                    tft = TFTModel()
                    X_tft, _ = tft.prepare_sequences(df, fit_scaler=True)
                    if len(X_tft) > 0:
                        try:
                            tft.load(symbol)
                        except FileNotFoundError:
                            tft.load("SPY")
                        tft_prob = tft.predict_proba(X_tft[-1:])[0]
                except Exception:
                    tft_prob = 0.5

            # V6 Ensemble prediction
            prediction = ensemble.predict(
                xgb_prob=xgb_prob, lstm_prob=lstm_prob,
                tft_prob=tft_prob, tabnet_prob=tabnet_prob,
            )

            # Volatility assessment
            vol_assessment = vol_filter.assess(df)

            # Credit spread signal
            credit_signal = generate_credit_spread_signal(prediction, vol_assessment)

            # Current price
            current_price = float(df["close"].iloc[-1])

            # Combine everything
            full_prediction = {
                **prediction,
                "symbol": symbol,
                "current_price": current_price,
                "date": str(df.index[-1].date()),
                "model_probs": {
                    "xgboost": round(float(xgb_prob), 4),
                    "lstm": round(float(lstm_prob), 4),
                    "tft": round(float(tft_prob), 4),
                    "tabnet": round(float(tabnet_prob), 4),
                },
                "volatility": vol_assessment,
                "credit_spread": credit_signal,
            }

            all_predictions[symbol] = full_prediction

            # Print summary
            cs = credit_signal["credit_spread_signal"]
            vol_s = vol_assessment["vol_score"]
            print(
                f"  {prediction['signal']:>10}  "
                f"conf={prediction['confidence']:.3f}  "
                f"prob={prediction['probability']:.3f}  "
                f"price=${current_price:.2f}  "
                f"vol={vol_s:.2f}  "
                f"spread={cs}"
            )
            if cs == "SELL_SPREAD":
                print(f"  → {credit_signal['spread_type'].upper()}: "
                      f"{credit_signal['reason']}")

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DAILY RECOMMENDATIONS V7")
    print(f"{'='*60}")

    # Directional signals
    actionable = {s: p for s, p in all_predictions.items() if p["signal"] != "HOLD"}
    holds = {s: p for s, p in all_predictions.items() if p["signal"] == "HOLD"}

    if actionable:
        print(f"\n  DIRECTIONAL SIGNALS ({len(actionable)}):")
        for sym, pred in sorted(actionable.items(), key=lambda x: -x[1]["confidence"]):
            print(f"    {pred['signal']:>10}  {sym:<6}  "
                  f"conf={pred['confidence']:.3f}  "
                  f"price=${pred['current_price']:.2f}")

    if holds:
        print(f"\n  HOLD ({len(holds)}): {', '.join(holds.keys())}")

    # Credit spread recommendations
    spreads = {s: p for s, p in all_predictions.items()
               if p["credit_spread"]["credit_spread_signal"] == "SELL_SPREAD"}

    if spreads:
        print(f"\n  CREDIT SPREAD OPPORTUNITIES ({len(spreads)}):")
        for sym, pred in sorted(spreads.items(), key=lambda x: -x[1]["confidence"]):
            cs = pred["credit_spread"]
            print(f"    {cs['spread_type'].upper():<12}  {sym:<6}  "
                  f"conf={pred['confidence']:.3f}  "
                  f"vol={pred['volatility']['vol_score']:.2f}  "
                  f"size={cs['parameters']['position_size']}")
    else:
        print(f"\n  CREDIT SPREADS: No opportunities today (low vol or HOLD signals)")

    # Volatility summary
    high_vol = [s for s, p in all_predictions.items() if p["volatility"]["is_high_vol"]]
    low_vol = [s for s, p in all_predictions.items() if not p["volatility"]["is_high_vol"]]
    print(f"\n  VOL ENVIRONMENT: {len(high_vol)} high-vol, {len(low_vol)} low-vol")
    if high_vol:
        for sym in high_vol:
            v = all_predictions[sym]["volatility"]
            print(f"    {sym:<6} vol={v['vol_score']:.2f}  "
                  f"{', '.join(f'{k}={v}' for k, v in v['components'].items())}")

    # ── Save predictions ──────────────────────────────────────
    pred_dir = DATA_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "run_date": datetime.now().isoformat(),
        "version": "V7.1",
        "predictions": all_predictions,
        "summary": {
            "total_symbols": len(symbols),
            "processed": len(all_predictions),
            "buy_call": sum(1 for p in all_predictions.values() if p["signal"] == "BUY_CALL"),
            "buy_put": sum(1 for p in all_predictions.values() if p["signal"] == "BUY_PUT"),
            "hold": sum(1 for p in all_predictions.values() if p["signal"] == "HOLD"),
            "credit_spread_opportunities": len(spreads),
            "high_vol_tickers": len(high_vol),
        },
    }

    # Save as latest
    latest_path = pred_dir / "latest_predictions.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Predictions saved to {latest_path}")

    # Save dated copy
    date_str = datetime.now().strftime("%Y%m%d")
    dated_path = pred_dir / f"daily_v7_{date_str}.json"
    with open(dated_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"{'='*60}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily V7 predictions")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols")
    args = parser.parse_args()

    result = run_daily_v7(args.symbols)

    if result:
        s = result["summary"]
        print(f"\n  Daily V7 Complete:")
        print(f"  Processed: {s['processed']}/{s['total_symbols']}")
        print(f"  BUY CALL: {s['buy_call']} | BUY PUT: {s['buy_put']} | HOLD: {s['hold']}")
        print(f"  Credit Spread Opportunities: {s['credit_spread_opportunities']}")
        print(f"  High-Vol Tickers: {s['high_vol_tickers']}")
