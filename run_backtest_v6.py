#!/usr/bin/env python3
"""
Backtest V6.1 — Standalone script with all fixes baked in.
This file exists because edits to run_backtest.py weren't syncing.
All logic is self-contained here.

Usage:
    python3 run_backtest_v6.py
    python3 run_backtest_v6.py --symbol AAPL --days 1825 --capital 1000
"""
import argparse
import numpy as np
import pandas as pd
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import DATA_DIR
from src.data_pipeline.feature_builder import FeatureBuilder
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.backtest.backtester import Backtester

# Try importing new models (may not be available)
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


class FixedEnsemble:
    """
    Self-contained ensemble with the V6.1 confidence formula.
    Does NOT read from settings.yaml to avoid stale config issues.
    """
    def __init__(self):
        # V6.1 hardcoded weights
        self.weights = {
            "xgboost": 0.30,
            "lstm": 0.15,
            "tft": 0.25,
            "tabnet": 0.20,
            "sentiment": 0.10,
        }
        self.confidence_threshold = 0.15
        print(f"  FixedEnsemble V6.1: threshold={self.confidence_threshold}, "
              f"weights={self.weights}")

    def predict_batch(self, xgb_probs, lstm_probs, tft_probs=None,
                      tabnet_probs=None, sent_probs=None, sent_confs=None):
        n = len(xgb_probs)
        if tft_probs is None:
            tft_probs = np.full(n, 0.5)
        if tabnet_probs is None:
            tabnet_probs = np.full(n, 0.5)
        if sent_probs is None:
            sent_probs = np.full(n, 0.5)
        if sent_confs is None:
            sent_confs = np.full(n, 0.0)

        results = []
        for i in range(n):
            results.append(self._predict_single(
                xgb_probs[i], lstm_probs[i], tft_probs[i],
                tabnet_probs[i], sent_probs[i], sent_confs[i]
            ))
        return results

    def _predict_single(self, xgb_prob, lstm_prob, tft_prob, tabnet_prob,
                        sent_prob, sent_conf):
        model_probs = {
            "xgboost": xgb_prob,
            "lstm": lstm_prob,
            "tft": tft_prob,
            "tabnet": tabnet_prob,
            "sentiment": sent_prob,
        }

        # Build effective weights — redistribute from inactive models
        ew = self.weights.copy()

        # Sentiment: drop if no confidence
        if sent_conf < 0.1:
            sw = ew.pop("sentiment")
            tr = sum(ew.values())
            if tr > 0:
                for k in ew:
                    ew[k] += sw * (ew[k] / tr)
            ew["sentiment"] = 0.0
        else:
            ew["sentiment"] *= sent_conf

        # TFT: drop if not available (prob == 0.5)
        if tft_prob == 0.5:
            tw = ew.pop("tft", 0)
            tr = sum(ew.values())
            if tr > 0:
                for k in ew:
                    ew[k] += tw * (ew[k] / tr)
            ew["tft"] = 0.0

        # TabNet: same
        if tabnet_prob == 0.5:
            tbw = ew.pop("tabnet", 0)
            tr = sum(ew.values())
            if tr > 0:
                for k in ew:
                    ew[k] += tbw * (ew[k] / tr)
            ew["tabnet"] = 0.0

        # Normalize
        total_w = sum(ew.values())
        if total_w > 0:
            ew = {k: v / total_w for k, v in ew.items()}

        # Combined probability
        combined_prob = sum(ew.get(k, 0) * model_probs[k] for k in model_probs)

        # Direction
        direction = "UP" if combined_prob > 0.5 else "DOWN"

        # Active models (weight > 5%)
        active = {k: v for k, v in model_probs.items() if ew.get(k, 0) > 0.05}

        # Agreement
        agreements = sum(1 for p in active.values()
                        if (p > 0.5) == (direction == "UP"))
        n_active = len(active) if active else 1
        agreement_ratio = agreements / n_active

        # Convictions
        convictions = [abs(p - 0.5) * 2 for p in active.values()]
        avg_conv = np.mean(convictions) if convictions else 0
        max_conv = max(convictions) if convictions else 0

        # V6.1 confidence formula
        if agreement_ratio >= 0.8:
            confidence = 0.30 + avg_conv * 0.5 + max_conv * 0.2
        elif agreement_ratio >= 0.6:
            confidence = 0.15 + avg_conv * 0.5 + max_conv * 0.15
        else:
            confidence = avg_conv * 0.3

        confidence = min(confidence, 1.0)

        # Signal
        if confidence >= self.confidence_threshold:
            signal = "BUY_CALL" if direction == "UP" else "BUY_PUT"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "probability": float(combined_prob),
            "confidence": float(confidence),
            "direction": direction,
        }


class FixedBacktester:
    """
    Self-contained backtester with V6.1 options fixes.
    Does NOT read options settings from config to avoid stale values.
    """
    def __init__(self, starting_capital=1000.0, mode="directional"):
        self.mode = mode
        self.starting_capital = starting_capital
        self.max_risk_per_trade = 0.03
        self.daily_loss_limit = 0.05
        self.max_open_positions = 5

        # Directional settings
        self.dir_sl = 0.03
        self.dir_tp = 0.04
        self.max_hold_days = 5

        # Options settings — V6.3 (realistic leverage model)
        # Real ATM option math for SPY:
        #   Premium ≈ 1-2% of stock price for 14 DTE
        #   Delta ≈ 0.50, so delta_leverage ≈ delta / (premium/stock) ≈ 0.50/0.015 ≈ 33x
        #   We use 12x conservatively (accounts for bid-ask spread, slippage, IV crush)
        #   Daily theta ≈ 3-5% of premium early, accelerating to 8-10% near expiry
        self.opt_sl = 0.20           # 20% loss → cut it fast
        self.opt_tp = 0.25           # 25% gain → take profit
        self.default_dte = 14        # 2-week options
        self.opt_min_conf = 0.15     # Match directional threshold
        self.opt_max_hold_days = 3   # 3-day max hold (minimize theta exposure)

        # Directional min confidence — match ensemble threshold
        self.dir_min_conf = 0.15

        self.results_dir = DATA_DIR / "backtest"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self, predictions, price_df, dates=None):
        if dates is None:
            dates = price_df.index[-len(predictions):]

        capital = self.starting_capital
        portfolio_history = []
        trades = []
        open_positions = []
        daily_pnl = 0.0
        daily_start_capital = capital

        min_conf = self.opt_min_conf if self.mode == "options" else self.dir_min_conf

        for i, (pred, date) in enumerate(zip(predictions, dates)):
            if i > 0 and dates[i].date() != dates[i-1].date():
                daily_pnl = 0.0
                daily_start_capital = capital

            current_price = price_df.loc[date, "close"] if date in price_df.index else None
            if current_price is None:
                portfolio_history.append({"date": date, "capital": capital,
                                         "open_positions": len(open_positions)})
                continue

            # Close positions
            to_close = []
            for pi, pos in enumerate(open_positions):
                should_close, reason, pnl_pct = self._eval(pos, current_price, date)
                if should_close:
                    pos["exit_date"] = date
                    pos["exit_price"] = current_price
                    pos["exit_reason"] = reason
                    pos["pnl_pct"] = pnl_pct
                    pos["pnl_dollar"] = pos["position_size"] * pnl_pct
                    to_close.append(pi)

            for pi in sorted(to_close, reverse=True):
                closed = open_positions.pop(pi)
                capital += closed["position_size"] + closed["pnl_dollar"]
                daily_pnl += closed["pnl_dollar"]
                trades.append(closed)

            # Daily loss limit
            if daily_start_capital > 0:
                if -daily_pnl / daily_start_capital >= self.daily_loss_limit:
                    portfolio_history.append({"date": date, "capital": capital,
                                             "open_positions": len(open_positions)})
                    continue

            # Open new position
            signal = pred.get("signal", "HOLD")
            confidence = pred.get("confidence", 0)
            probability = pred.get("probability", 0.5)

            can_open = (signal != "HOLD"
                       and confidence >= min_conf
                       and len(open_positions) < self.max_open_positions)

            if can_open:
                conf_scale = min(0.5 + confidence, 1.5)
                pos_size = capital * self.max_risk_per_trade * conf_scale

                if pos_size >= 10 and capital >= 50:
                    direction = "long" if signal == "BUY_CALL" else "short"
                    position = {
                        "entry_date": date,
                        "symbol": "SPY",
                        "direction": direction,
                        "signal": signal,
                        "entry_price": current_price,
                        "position_size": pos_size,
                        "confidence": confidence,
                        "probability": probability,
                        "mode": self.mode,
                    }
                    if self.mode == "options":
                        position["option_type"] = "call" if signal == "BUY_CALL" else "put"
                        position["dte"] = self.default_dte
                    open_positions.append(position)
                    capital -= pos_size

            # Record
            open_value = sum(
                pos["position_size"] * max(0, 1 + self._eval(pos, current_price, date)[2])
                for pos in open_positions
            )
            portfolio_history.append({
                "date": date, "capital": capital,
                "open_value": max(0, open_value),
                "total_value": capital + max(0, open_value),
                "open_positions": len(open_positions),
            })

        # Close remaining
        last_price = price_df["close"].iloc[-1]
        for pos in open_positions:
            _, _, pnl_pct = self._eval(pos, last_price, dates[-1], force=True)
            pos["exit_date"] = dates[-1]
            pos["exit_price"] = last_price
            pos["exit_reason"] = "backtest_end"
            pos["pnl_pct"] = pnl_pct
            pos["pnl_dollar"] = pos["position_size"] * pnl_pct
            capital += pos["position_size"] + pos["pnl_dollar"]
            trades.append(pos)

        metrics = self._metrics(trades, portfolio_history)
        return {"trades": trades, "portfolio_history": portfolio_history, "metrics": metrics}

    def _eval(self, pos, price, date, force=False):
        if self.mode == "directional":
            return self._eval_dir(pos, price, date, force)
        return self._eval_opt(pos, price, date, force)

    def _eval_dir(self, pos, price, date, force=False):
        pct = (price - pos["entry_price"]) / pos["entry_price"]
        days = max(1, (date - pos["entry_date"]).days)
        pnl = pct if pos["direction"] == "long" else -pct

        if force:
            return True, "backtest_end", pnl
        if pnl <= -self.dir_sl:
            return True, "stop_loss", -self.dir_sl
        if pnl >= self.dir_tp:
            return True, "take_profit", self.dir_tp
        if days >= self.max_hold_days:
            return True, "max_hold", pnl
        return False, "", pnl

    def _eval_opt(self, pos, price, date, force=False):
        """
        V6.3 Options P&L model — realistic ATM option leverage.

        Real-world ATM option math (SPY, 14 DTE, ~15% IV):
          Premium ≈ 1.2-1.5% of stock price
          Delta ≈ 0.50 → delta_leverage = 0.50 / 0.013 ≈ 38x (theoretical)
          We use 12x (conservative: accounts for bid-ask, slippage, IV crush)
          Theta ≈ 3-4% of premium per day early, 7-10% near expiry
          Total theta at expiry: ~80% of premium (ATM options are mostly time value)

        Strategy: 3-day max hold to keep theta under control.
          Day 1 theta: ~3% | Day 2: ~6% cumulative | Day 3: ~9% cumulative
          A 1% stock move → ~12% option gain vs ~3% theta cost → net +9% day 1
        """
        pct = (price - pos["entry_price"]) / pos["entry_price"]
        dte = pos["dte"]
        days = max(0.5, (date - pos["entry_date"]).days)

        # Theta: sqrt decay, realistic for ATM options
        # ~80% of ATM premium is time value that decays to zero
        theta_budget = 0.80
        time_frac = min(days / dte, 1.0)
        cum_theta = theta_budget * (1.0 - np.sqrt(max(0, 1.0 - time_frac)))

        # Delta leverage: ATM 14 DTE ≈ 12x (conservative)
        # Real theoretical is ~30-40x but slippage, spread, and IV crush reduce it
        base_delta = 12.0
        dir_move = pct if pos["option_type"] == "call" else -pct

        if dir_move > 0.005:
            # Going ITM: delta rises toward 1.0, effective leverage increases
            delta = base_delta + min(dir_move * 30, 8.0)  # up to 20x
        elif dir_move < -0.005:
            # Going OTM: delta drops, leverage collapses
            delta = max(base_delta * (1 + dir_move * 8), 2.0)  # min 2x
        else:
            delta = base_delta

        # Vega: large moves increase IV → benefits long option holders
        abs_move = abs(pct)
        vega = max(0, (abs_move - 0.005)) * 2.0  # significant for short-dated

        # Gamma: ATM short-dated options have very high gamma
        # Benefits from moves in EITHER direction (convexity)
        gamma = max(0, abs_move - 0.003) * 0.5

        if pos["option_type"] == "call":
            raw_pnl = pct * delta + vega + gamma - cum_theta
        else:
            raw_pnl = -pct * delta + vega + gamma - cum_theta

        opt_pnl = max(raw_pnl, -1.0)

        if force:
            return True, "backtest_end", opt_pnl
        if opt_pnl <= -self.opt_sl:
            return True, "stop_loss", -self.opt_sl
        if opt_pnl >= self.opt_tp:
            return True, "take_profit", self.opt_tp
        if days >= self.opt_max_hold_days:
            return True, "time_exit", opt_pnl
        if days >= dte:
            return True, "expiration", opt_pnl
        return False, "", opt_pnl

    def _metrics(self, trades, history):
        if not trades:
            return {
                "total_trades": 0, "final_portfolio_value": self.starting_capital,
                "total_return": 0, "win_rate": 0, "direction_accuracy": 0,
                "sharpe_ratio": 0, "max_drawdown": 0, "profit_factor": 0,
                "expectancy": 0, "mode": self.mode, "exit_reasons": {},
            }

        pnls = [t["pnl_dollar"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_pnl = sum(pnls)

        values = [h.get("total_value", h.get("capital", self.starting_capital))
                  for h in history]
        vs = pd.Series(values)
        daily_ret = vs.pct_change().dropna()

        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if len(daily_ret) > 1 and daily_ret.std() > 0 else 0)
        cummax = vs.cummax()
        dd = ((vs - cummax) / cummax).min()

        correct = sum(1 for t in trades
                     if (t["direction"] == "long" and t.get("exit_price", 0) > t["entry_price"])
                     or (t["direction"] == "short" and t.get("exit_price", 0) < t["entry_price"]))

        exit_reasons = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "direction_accuracy": correct / len(trades),
            "total_pnl": total_pnl,
            "expectancy": total_pnl / len(trades),
            "profit_factor": abs(sum(wins) / sum(losses)) if losses else float("inf"),
            "sharpe_ratio": sharpe,
            "max_drawdown": dd,
            "total_return": (values[-1] - self.starting_capital) / self.starting_capital,
            "final_portfolio_value": values[-1],
            "mode": self.mode,
            "exit_reasons": exit_reasons,
            "long_trades": sum(1 for t in trades if t["direction"] == "long"),
            "short_trades": sum(1 for t in trades if t["direction"] == "short"),
        }


def run_backtest_v6(symbol="SPY", days=365*5, capital=1000.0, mode="both", test_ratio=0.30):
    print(f"\n{'='*60}")
    print(f"  BACKTEST V6.1 — {symbol} — ${capital} capital")
    print(f"  {days} days of data, {test_ratio*100:.0f}% test split")
    print(f"{'='*60}\n")

    # Step 1: Build features
    print("Step 1: Building features...")
    builder = FeatureBuilder(use_finbert=False)
    df = builder.build_features(symbol, days=days, include_news=False)

    if df.empty or len(df) < 100:
        print(f"ERROR: Insufficient data for {symbol} (got {len(df)} rows)")
        return None

    # Step 2: Split
    print("\nStep 2: Splitting data...")
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * 0.85)

    df_train = df.iloc[:val_start]
    df_val = df.iloc[val_start:test_start]
    df_test = df.iloc[test_start:]
    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Step 3: Train XGBoost
    print("\nStep 3: Training XGBoost...")
    xgb = XGBoostModel()
    X_train, y_train = xgb.prepare_features(df_train)
    X_val, y_val = xgb.prepare_features(df_val)
    X_test, y_test = xgb.prepare_features(df_test)
    xgb.train(X_train, y_train, eval_set=(X_val, y_val))

    # Step 3b: Train TabNet
    tabnet_test_probs = None
    if HAS_TABNET:
        print("\nStep 3b: Training TabNet...")
        try:
            tabnet = TabNetModel()
            tabnet.feature_names = xgb.feature_names
            tabnet.train(X_train, y_train, eval_set=(X_val, y_val))
            tabnet_test_probs = tabnet.predict_proba(X_test)
            print(f"  TabNet mean prob: {tabnet_test_probs.mean():.4f}")
        except Exception as e:
            print(f"  TabNet failed: {e}")

    # Step 4: Train LSTM
    N_RUNS = 3
    print(f"\nStep 4: Training LSTM ({N_RUNS} runs)...")
    lstm = LSTMModel()
    X_train_seq, y_train_seq = lstm.prepare_sequences(df_train, fit_scaler=True)
    X_val_seq, y_val_seq = lstm.prepare_sequences(df_val, fit_scaler=False)
    X_test_seq, y_test_seq = lstm.prepare_sequences(df_test, fit_scaler=False)

    lstm_ok = len(X_train_seq) > 0 and len(X_val_seq) > 0
    lstm_test_probs = None

    if lstm_ok:
        all_probs = []
        for ri, seed in enumerate([42, 123, 777]):
            print(f"  LSTM run {ri+1}/{N_RUNS} (seed={seed})...")
            lr = LSTMModel()
            lr._seed = seed
            lr.scaler = lstm.scaler
            lr.n_features = lstm.n_features
            lr.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            probs = lr.predict_proba(X_test_seq)
            all_probs.append(probs)
            print(f"    mean prob: {probs.mean():.4f}")
        lstm_test_probs = np.mean(all_probs, axis=0)

    # Step 4b: Train TFT
    tft_test_probs = None
    if HAS_TFT:
        print("\nStep 4b: Training TFT...")
        try:
            tft = TFTModel()
            X_train_tft, y_train_tft = tft.prepare_sequences(df_train, fit_scaler=True)
            X_val_tft, y_val_tft = tft.prepare_sequences(df_val, fit_scaler=False)
            X_test_tft, y_test_tft = tft.prepare_sequences(df_test, fit_scaler=False)
            if len(X_train_tft) > 0 and len(X_val_tft) > 0:
                tft.train(X_train_tft, y_train_tft, X_val_tft, y_val_tft)
                tft_test_probs = tft.predict_proba(X_test_tft)
                print(f"  TFT mean prob: {tft_test_probs.mean():.4f}")
        except Exception as e:
            print(f"  TFT failed: {e}")

    # Step 5: Ensemble predictions
    print("\nStep 5: Generating ensemble predictions...")
    ensemble = FixedEnsemble()

    xgb_test_probs = xgb.predict_proba(X_test)

    # Align to shortest
    n_align = len(xgb_test_probs)
    if lstm_ok and lstm_test_probs is not None:
        n_align = min(n_align, len(lstm_test_probs))
    if tft_test_probs is not None:
        n_align = min(n_align, len(tft_test_probs))
    if tabnet_test_probs is not None:
        n_align = min(n_align, len(tabnet_test_probs))

    xgb_a = xgb_test_probs[-n_align:]
    lstm_a = lstm_test_probs[-n_align:] if lstm_test_probs is not None else np.full(n_align, 0.5)
    tft_a = tft_test_probs[-n_align:] if tft_test_probs is not None else np.full(n_align, 0.5)
    tab_a = tabnet_test_probs[-n_align:] if tabnet_test_probs is not None else np.full(n_align, 0.5)
    sent_p = np.full(n_align, 0.5)
    sent_c = np.full(n_align, 0.0)

    predictions = ensemble.predict_batch(xgb_a, lstm_a, tft_a, tab_a, sent_p, sent_c)
    test_dates = df_test.index[-n_align:]

    # === DIAGNOSTICS ===
    signals = [p["signal"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]
    probs = [p["probability"] for p in predictions]

    print(f"\n{'='*60}")
    print(f"  PREDICTION DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Signals: BUY_CALL={signals.count('BUY_CALL')}, "
          f"BUY_PUT={signals.count('BUY_PUT')}, HOLD={signals.count('HOLD')}")
    print(f"\n  Model probability ranges:")
    print(f"    XGBoost:  min={xgb_a.min():.4f}  max={xgb_a.max():.4f}  mean={xgb_a.mean():.4f}  std={xgb_a.std():.4f}")
    if lstm_test_probs is not None:
        print(f"    LSTM:     min={lstm_a.min():.4f}  max={lstm_a.max():.4f}  mean={lstm_a.mean():.4f}  std={lstm_a.std():.4f}")
    if tft_test_probs is not None:
        print(f"    TFT:      min={tft_a.min():.4f}  max={tft_a.max():.4f}  mean={tft_a.mean():.4f}  std={tft_a.std():.4f}")
    if tabnet_test_probs is not None:
        print(f"    TabNet:   min={tab_a.min():.4f}  max={tab_a.max():.4f}  mean={tab_a.mean():.4f}  std={tab_a.std():.4f}")

    conf_arr = np.array(confidences)
    print(f"\n  Confidence: min={conf_arr.min():.4f}  max={conf_arr.max():.4f}  mean={conf_arr.mean():.4f}")
    print(f"  Confidence percentiles:")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    {pct}th: {np.percentile(conf_arr, pct):.4f}")
    print(f"  Ensemble threshold: {ensemble.confidence_threshold}")
    print(f"  Backtester dir threshold: 0.15 | options threshold: 0.15")

    actionable = sum(1 for s in signals if s != "HOLD")
    print(f"\n  Actionable signals: {actionable}/{len(signals)} ({actionable/len(signals)*100:.0f}%)")
    print(f"{'='*60}\n")

    # Step 6: Run backtests
    all_results = {}
    modes = ["directional", "options"] if mode == "both" else [mode]

    for bt_mode in modes:
        print(f"Step 6: Running {bt_mode.upper()} backtest...")
        bt = FixedBacktester(starting_capital=capital, mode=bt_mode)
        results = bt.run(predictions, df_test, test_dates)
        all_results[bt_mode] = results

        m = results["metrics"]
        print(f"  {bt_mode.upper()}: {m['total_trades']} trades, "
              f"win={m['win_rate']*100:.1f}%, ret={m['total_return']*100:+.2f}%, "
              f"final=${m['final_portfolio_value']:,.2f}")
        if m.get("exit_reasons"):
            print(f"  Exit reasons: {m['exit_reasons']}")
        print()

    # Step 7: Monte Carlo
    print("Step 7: Running Monte Carlo (1000 simulations)...")
    from src.backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(n_simulations=1000)

    primary_mode = "directional" if "directional" in all_results else mode
    primary = all_results[primary_mode]

    mc_results = mc.run(primary, starting_capital=capital)
    mc.print_report(mc_results)
    mc.save_results(mc_results, symbol)

    # Save predictions
    pred_df = pd.DataFrame([{
        "date": str(d.date()),
        "signal": p["signal"],
        "probability": p["probability"],
        "confidence": p["confidence"],
        "direction": p["direction"],
    } for d, p in zip(test_dates, predictions)])
    pred_path = DATA_DIR / "backtest" / f"predictions_{symbol}.csv"
    pred_df.to_csv(pred_path, index=False)

    return primary, all_results


def print_comparison(all_results, capital):
    if len(all_results) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  MODE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Directional':>20} {'Options':>20}")
    print(f"{'-'*65}")

    dm = all_results.get("directional", {}).get("metrics", {})
    om = all_results.get("options", {}).get("metrics", {})

    rows = [
        ("Final Value", f"${dm.get('final_portfolio_value',0):,.2f}", f"${om.get('final_portfolio_value',0):,.2f}"),
        ("Total Return", f"{dm.get('total_return',0)*100:+.2f}%", f"{om.get('total_return',0)*100:+.2f}%"),
        ("Total Trades", f"{dm.get('total_trades',0)}", f"{om.get('total_trades',0)}"),
        ("Win Rate", f"{dm.get('win_rate',0)*100:.1f}%", f"{om.get('win_rate',0)*100:.1f}%"),
        ("Direction Accuracy", f"{dm.get('direction_accuracy',0)*100:.1f}%", f"{om.get('direction_accuracy',0)*100:.1f}%"),
        ("Profit Factor", f"{dm.get('profit_factor',0):.2f}", f"{om.get('profit_factor',0):.2f}"),
        ("Sharpe Ratio", f"{dm.get('sharpe_ratio',0):.2f}", f"{om.get('sharpe_ratio',0):.2f}"),
        ("Max Drawdown", f"{dm.get('max_drawdown',0)*100:.1f}%", f"{om.get('max_drawdown',0)*100:.1f}%"),
        ("Expectancy/Trade", f"${dm.get('expectancy',0):+,.2f}", f"${om.get('expectancy',0):+,.2f}"),
    ]
    for label, dv, ov in rows:
        print(f"{label:<25} {dv:>20} {ov:>20}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(">>> BACKTEST V6.1 — STANDALONE WITH ALL FIXES <<<")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Run trading backtest V6.1")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--days", type=int, default=365*5)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--mode", default="both", choices=["directional", "options", "both"])
    parser.add_argument("--test-ratio", type=float, default=0.30)
    args = parser.parse_args()

    result = run_backtest_v6(args.symbol, args.days, args.capital, args.mode, args.test_ratio)

    if result:
        primary, all_results = result

        m = primary.get("metrics", {})
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY ({m.get('mode','directional').upper()})")
        print(f"{'='*60}")
        print(f"Starting Capital:  ${args.capital:,.2f}")
        print(f"Final Value:       ${m.get('final_portfolio_value',0):,.2f}")
        print(f"Total Return:      {m.get('total_return',0)*100:+.2f}%")
        print(f"Win Rate:          {m.get('win_rate',0)*100:.1f}%")
        print(f"Direction Accuracy:{m.get('direction_accuracy',0)*100:.1f}%")
        print(f"Sharpe Ratio:      {m.get('sharpe_ratio',0):.2f}")
        print(f"Max Drawdown:      {m.get('max_drawdown',0)*100:.1f}%")
        print(f"Total Trades:      {m.get('total_trades',0)}")
        print(f"Profit Factor:     {m.get('profit_factor',0):.2f}")
        print(f"Expectancy/Trade:  ${m.get('expectancy',0):+,.2f}")
        print(f"{'='*60}")

        if len(all_results) > 1:
            print_comparison(all_results, args.capital)
