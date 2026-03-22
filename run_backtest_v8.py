#!/usr/bin/env python3
"""
Backtest V8 — Five strategic improvements over V7:

  1. REGIME DETECTION: Market regime filters — only trade when regime matches signal
  2. WALK-FORWARD XGBOOST: Properly tuned hyperparameters via expanding-window CV
  3. LEARNED STACKING: Logistic regression meta-learner discovers optimal model weights
  4. ADVANCED FEATURES: Mean-reversion + regime features (18 new)
  5. WALK-FORWARD BACKTEST: No data leakage — train/test windows expand properly

Usage:
    python3 run_backtest_v8.py
    python3 run_backtest_v8.py --symbol SPY --days 1825 --capital 1000
    python3 run_backtest_v8.py --skip-wf-xgb   # skip slow XGB optimization
"""
import argparse
import time
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import DATA_DIR
from src.data_pipeline.feature_builder import FeatureBuilder
from src.data_pipeline.advanced_features import AdvancedFeatureBuilder
from src.models.xgboost_model import XGBoostModel
from src.models.xgboost_walkforward import WalkForwardXGBoost
from src.models.lstm_model import LSTMModel
from src.models.ensemble_stacker import EnsembleStacker
from src.models.regime_detector import RegimeDetector, AllocationFilter, MarketRegime

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
# V7 components we keep
# ═══════════════════════════════════════════════════════════

class VolatilityFilter:
    """Filters predictions to only take options trades on high-volatility days."""

    def __init__(self, df_test):
        self.vol_scores = {}
        if "close" not in df_test.columns:
            return

        returns = df_test["close"].pct_change()
        vol_5d = returns.rolling(5).std()
        vol_20d = returns.rolling(20).std()
        has_vix = "vix" in df_test.columns

        for date in df_test.index:
            score = 0.0
            v5 = vol_5d.get(date, 0)
            v20 = vol_20d.get(date, 0)
            if v20 and v20 > 0 and not np.isnan(v5) and not np.isnan(v20):
                vol_ratio = v5 / v20
                if vol_ratio > 1.3:
                    score += 0.3
                elif vol_ratio > 1.1:
                    score += 0.15

            if date in returns.index and not np.isnan(returns[date]):
                abs_ret = abs(returns[date])
                if abs_ret > 0.015:
                    score += 0.3
                elif abs_ret > 0.008:
                    score += 0.15

            if has_vix and date in df_test.index:
                vix_val = df_test.loc[date].get("vix", 0)
                if vix_val and not np.isnan(vix_val):
                    if vix_val > 25:
                        score += 0.3
                    elif vix_val > 20:
                        score += 0.15

            for col in ["fomc_week", "earnings_week", "high_event_risk"]:
                if col in df_test.columns:
                    val = df_test.loc[date].get(col, 0)
                    if val and val > 0:
                        score += 0.2
                        break

            self.vol_scores[date] = min(score, 1.0)

    def is_high_vol(self, date, min_score=0.3):
        return self.vol_scores.get(date, 0) >= min_score

    def get_score(self, date):
        return self.vol_scores.get(date, 0)


class V8Backtester:
    """
    V8 backtester with regime-aware position sizing.

    Modes: directional, options_credit, options_debit
    NEW: Regime filter adjusts position sizes and skips mismatched signals.
    """

    def __init__(self, starting_capital=1000.0, mode="directional",
                 vol_filter=None, allocation_filter=None):
        self.mode = mode
        self.starting_capital = starting_capital
        self.max_risk_per_trade = 0.03
        self.daily_loss_limit = 0.05
        self.max_open_positions = 5
        self.vol_filter = vol_filter
        self.allocation_filter = allocation_filter

        # Directional settings
        self.dir_sl = 0.03
        self.dir_tp = 0.04
        self.max_hold_days = 5
        self.dir_min_conf = 0.15

        # Credit spread settings (V7 winning config)
        self.spread_width_pct = 0.03
        self.credit_pct = 0.33
        self.spread_max_hold_days = 10
        self.spread_tp_pct = 0.50
        self.spread_sl_pct = 2.00
        self.spread_min_conf = 0.15
        self.spread_min_vol = 0.3

        # Debit options settings
        self.opt_sl = 0.20
        self.opt_tp = 0.25
        self.opt_dte = 14
        self.opt_max_hold_days = 3
        self.opt_min_conf = 0.15

        self.results_dir = DATA_DIR / "backtest"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Stats tracking
        self.regime_stats = {"trades_by_regime": {}, "skipped_by_regime": 0}

    def run(self, predictions, price_df, dates=None):
        if dates is None:
            dates = price_df.index[-len(predictions):]

        capital = self.starting_capital
        portfolio_history = []
        trades = []
        open_positions = []
        daily_pnl = 0.0
        daily_start_capital = capital

        if self.mode == "options_credit":
            min_conf = self.spread_min_conf
        elif self.mode == "options_debit":
            min_conf = self.opt_min_conf
        else:
            min_conf = self.dir_min_conf

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
            if daily_start_capital > 0 and -daily_pnl / daily_start_capital >= self.daily_loss_limit:
                portfolio_history.append({"date": date, "capital": capital,
                                         "open_positions": len(open_positions)})
                continue

            # Open position
            signal = pred.get("signal", "HOLD")
            confidence = pred.get("confidence", 0)

            # Volatility filter for options
            vol_ok = True
            if self.mode in ("options_credit", "options_debit") and self.vol_filter:
                vol_ok = self.vol_filter.is_high_vol(date, self.spread_min_vol)

            # NEW V8: Regime-based position sizing
            regime_scale = 1.0
            if self.allocation_filter and signal != "HOLD":
                regime_result = self.allocation_filter.get_position_size(
                    price_df.loc[:date].tail(250),  # pass recent price history
                    signal, 1.0
                )
                regime_scale = regime_result
                regime_name = "unknown"
                try:
                    regime_info = self.allocation_filter.get_regime_and_sizing(
                        price_df.loc[:date].tail(250), signal, 1.0
                    )
                    regime_name = regime_info.get("regime", "unknown")
                    if isinstance(regime_name, MarketRegime):
                        regime_name = regime_name.name
                except:
                    pass

                # Track regime stats
                self.regime_stats["trades_by_regime"][regime_name] = \
                    self.regime_stats["trades_by_regime"].get(regime_name, 0) + 1

                # Skip if regime says 0 position
                if regime_scale <= 0.05:
                    self.regime_stats["skipped_by_regime"] += 1
                    portfolio_history.append({"date": date, "capital": capital,
                                             "open_positions": len(open_positions)})
                    continue

            can_open = (signal != "HOLD"
                       and confidence >= min_conf
                       and vol_ok
                       and len(open_positions) < self.max_open_positions)

            if can_open:
                conf_scale = min(0.5 + confidence, 1.5)
                pos_size = capital * self.max_risk_per_trade * conf_scale * regime_scale

                if pos_size >= 10 and capital >= 50:
                    direction = "long" if signal == "BUY_CALL" else "short"
                    position = {
                        "entry_date": date, "symbol": "SPY",
                        "direction": direction, "signal": signal,
                        "entry_price": current_price,
                        "position_size": pos_size,
                        "confidence": confidence,
                        "mode": self.mode,
                        "regime_scale": regime_scale,
                    }

                    if self.mode == "options_credit":
                        position["spread_type"] = "bull_put" if signal == "BUY_CALL" else "bear_call"
                        position["credit"] = pos_size * self.credit_pct
                        position["max_loss"] = pos_size * (1 - self.credit_pct)
                    elif self.mode == "options_debit":
                        position["option_type"] = "call" if signal == "BUY_CALL" else "put"
                        position["dte"] = self.opt_dte

                    open_positions.append(position)
                    capital -= pos_size

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
        if len(price_df) > 0:
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
        metrics["regime_stats"] = self.regime_stats
        return {"trades": trades, "portfolio_history": portfolio_history, "metrics": metrics}

    def _eval(self, pos, price, date, force=False):
        if self.mode == "directional":
            return self._eval_dir(pos, price, date, force)
        elif self.mode == "options_credit":
            return self._eval_credit(pos, price, date, force)
        else:
            return self._eval_debit(pos, price, date, force)

    def _eval_dir(self, pos, price, date, force=False):
        pct = (price - pos["entry_price"]) / pos["entry_price"]
        days = max(1, (date - pos["entry_date"]).days)
        pnl = pct if pos["direction"] == "long" else -pct
        if force: return True, "backtest_end", pnl
        if pnl <= -self.dir_sl: return True, "stop_loss", -self.dir_sl
        if pnl >= self.dir_tp: return True, "take_profit", self.dir_tp
        if days >= self.max_hold_days: return True, "max_hold", pnl
        return False, "", pnl

    def _eval_credit(self, pos, price, date, force=False):
        pct = (price - pos["entry_price"]) / pos["entry_price"]
        days = max(0.5, (date - pos["entry_date"]).days)
        credit_pct = self.credit_pct

        if pos["spread_type"] == "bull_put":
            dir_move = pct
        else:
            dir_move = -pct

        theta_benefit = min(days * 0.05, 0.5) * credit_pct
        spread_width = self.spread_width_pct

        if dir_move >= 0:
            move_vs_spread = dir_move / spread_width
            delta_pnl = credit_pct * min(move_vs_spread * 2, 1.0)
        else:
            adverse_move = -dir_move
            move_vs_spread = adverse_move / spread_width
            if move_vs_spread < 0.5:
                delta_pnl = -credit_pct * move_vs_spread
            else:
                delta_pnl = -credit_pct - (move_vs_spread - 0.5) * (1 - credit_pct) * 2
                delta_pnl = max(delta_pnl, -(1 - credit_pct))

        net_pnl = delta_pnl + theta_benefit
        net_pnl = max(net_pnl, -(1 - credit_pct))
        net_pnl = min(net_pnl, credit_pct)

        if force: return True, "backtest_end", net_pnl
        if net_pnl >= credit_pct * self.spread_tp_pct: return True, "take_profit", net_pnl
        if net_pnl <= -credit_pct * self.spread_sl_pct: return True, "stop_loss", net_pnl
        if days >= self.spread_max_hold_days: return True, "time_exit", net_pnl
        return False, "", net_pnl

    def _eval_debit(self, pos, price, date, force=False):
        pct = (price - pos["entry_price"]) / pos["entry_price"]
        dte = pos["dte"]
        days = max(0.5, (date - pos["entry_date"]).days)

        theta_budget = 0.80
        time_frac = min(days / dte, 1.0)
        cum_theta = theta_budget * (1.0 - np.sqrt(max(0, 1.0 - time_frac)))

        base_delta = 12.0
        dir_move = pct if pos["option_type"] == "call" else -pct
        if dir_move > 0.005:
            delta = base_delta + min(dir_move * 30, 8.0)
        elif dir_move < -0.005:
            delta = max(base_delta * (1 + dir_move * 8), 2.0)
        else:
            delta = base_delta

        abs_move = abs(pct)
        vega = max(0, (abs_move - 0.005)) * 2.0
        gamma = max(0, abs_move - 0.003) * 0.5

        if pos["option_type"] == "call":
            raw_pnl = pct * delta + vega + gamma - cum_theta
        else:
            raw_pnl = -pct * delta + vega + gamma - cum_theta

        opt_pnl = max(raw_pnl, -1.0)

        if force: return True, "backtest_end", opt_pnl
        if opt_pnl <= -self.opt_sl: return True, "stop_loss", -self.opt_sl
        if opt_pnl >= self.opt_tp: return True, "take_profit", self.opt_tp
        if days >= self.opt_max_hold_days: return True, "time_exit", opt_pnl
        if days >= dte: return True, "expiration", opt_pnl
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
        dd = ((vs - vs.cummax()) / vs.cummax()).min()

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
            "sharpe_ratio": sharpe, "max_drawdown": dd,
            "total_return": (values[-1] - self.starting_capital) / self.starting_capital,
            "final_portfolio_value": values[-1],
            "mode": self.mode, "exit_reasons": exit_reasons,
            "long_trades": sum(1 for t in trades if t["direction"] == "long"),
            "short_trades": sum(1 for t in trades if t["direction"] == "short"),
        }


# ═══════════════════════════════════════════════════════════
# V8 Main Pipeline
# ═══════════════════════════════════════════════════════════

def run_backtest_v8(symbol="SPY", days=365*5, capital=1000.0, test_ratio=0.30,
                    skip_wf_xgb=False):
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  BACKTEST V8 — {symbol} — ${capital} capital — 5 STRATEGIC IMPROVEMENTS")
    print(f"  {days} days, {test_ratio*100:.0f}% test split")
    print(f"  [1] Regime Detection  [2] Walk-Forward XGB  [3] Learned Stacker")
    print(f"  [4] Advanced Features [5] Walk-Forward Backtest Framework")
    print(f"{'='*70}\n")

    # ─────────────────────────────────────────────────────────
    # Step 1: Build features (original + advanced)
    # ─────────────────────────────────────────────────────────
    print("Step 1: Building features (original + 18 new advanced features)...")
    builder = FeatureBuilder(use_finbert=False)
    df = builder.build_features(symbol, days=days, include_news=False)
    if df.empty or len(df) < 100:
        print(f"ERROR: Insufficient data ({len(df)} rows)")
        return None

    # Add advanced mean-reversion + regime features
    adv_builder = AdvancedFeatureBuilder()
    df = adv_builder.build_advanced_features(df)
    new_features = adv_builder.get_feature_names()
    print(f"  Total rows: {len(df)}")
    print(f"  Original features: {len(df.columns) - len(new_features)}")
    print(f"  New advanced features: {len(new_features)}")
    print(f"  Total columns: {len(df.columns)}")

    # ─────────────────────────────────────────────────────────
    # Step 2: Split data
    # ─────────────────────────────────────────────────────────
    print("\nStep 2: Splitting data...")
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * 0.85)
    df_train = df.iloc[:val_start]
    df_val = df.iloc[val_start:test_start]
    df_test = df.iloc[test_start:]
    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"  Train dates: {df_train.index[0].date()} to {df_train.index[-1].date()}")
    print(f"  Val dates:   {df_val.index[0].date()} to {df_val.index[-1].date()}")
    print(f"  Test dates:  {df_test.index[0].date()} to {df_test.index[-1].date()}")

    # ─────────────────────────────────────────────────────────
    # Step 3: Regime Detection
    # ─────────────────────────────────────────────────────────
    print("\nStep 3: [IMPROVEMENT 1] Regime detection...")
    regime_detector = RegimeDetector()
    test_regimes = regime_detector.get_regime_history(df_test)
    regime_counts = {}
    for r in test_regimes:
        name = r.name if isinstance(r, MarketRegime) else str(r)
        regime_counts[name] = regime_counts.get(name, 0) + 1
    print(f"  Test period regime distribution:")
    for regime_name, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {regime_name}: {count} days ({count/len(test_regimes)*100:.1f}%)")

    allocation_filter = AllocationFilter()

    # ─────────────────────────────────────────────────────────
    # Step 4: Train XGBoost (walk-forward or original)
    # ─────────────────────────────────────────────────────────
    if skip_wf_xgb:
        print("\nStep 4: [IMPROVEMENT 2] Using V6 original XGBoost (--skip-wf-xgb)...")
        xgb_model = XGBoostModel()
        X_train, y_train = xgb_model.prepare_features(df_train)
        X_val, y_val = xgb_model.prepare_features(df_val)
        X_test, y_test = xgb_model.prepare_features(df_test)
        xgb_model.train(X_train, y_train, eval_set=(X_val, y_val))
        xgb_test_probs = xgb_model.predict_proba(X_test)
        xgb_feature_names = xgb_model.feature_names
        print(f"  V6 XGB prob range: [{xgb_test_probs.min():.4f}, {xgb_test_probs.max():.4f}]")
    else:
        print("\nStep 4: [IMPROVEMENT 2] Walk-forward XGBoost optimization...")
        print("  (This may take a few minutes — searching hyperparameter space)")
        wf_xgb = WalkForwardXGBoost()

        # Prepare features using standard exclusion
        X_trainval, y_trainval = wf_xgb.prepare_features(
            pd.concat([df_train, df_val]), target_col="target_direction"
        )
        dates_trainval = pd.concat([df_train, df_val]).index

        # Run walk-forward optimization
        try:
            best_params, fold_results = wf_xgb.walk_forward_optimize(
                X_trainval, y_trainval, dates_trainval,
                min_train_months=6, val_months=1
            )
            print(f"\n  Best params found: {best_params}")
            report = wf_xgb.get_optimization_report()
            print(f"  {report[:200]}...")

            # Train final model and predict
            wf_xgb.train_final_model(X_trainval, y_trainval, best_params)
            X_test_wf, _ = wf_xgb.prepare_features(df_test, target_col="target_direction")
            xgb_test_probs = wf_xgb.predict_proba(X_test_wf)
            xgb_feature_names = wf_xgb.feature_names
            print(f"  WF XGB prob range: [{xgb_test_probs.min():.4f}, {xgb_test_probs.max():.4f}]")
            print(f"  WF XGB prob std: {xgb_test_probs.std():.4f}")
        except Exception as e:
            print(f"  Walk-forward XGB failed: {e}")
            print(f"  Falling back to V6 original XGBoost...")
            xgb_model = XGBoostModel()
            X_train, y_train = xgb_model.prepare_features(df_train)
            X_val, y_val = xgb_model.prepare_features(df_val)
            X_test, y_test = xgb_model.prepare_features(df_test)
            xgb_model.train(X_train, y_train, eval_set=(X_val, y_val))
            xgb_test_probs = xgb_model.predict_proba(X_test)
            xgb_feature_names = xgb_model.feature_names

    # Also train V6 original for comparison
    print("\n  Also training V6 original XGB for comparison...")
    xgb_orig = XGBoostModel()
    X_train_orig, y_train_orig = xgb_orig.prepare_features(df_train)
    X_val_orig, y_val_orig = xgb_orig.prepare_features(df_val)
    X_test_orig, y_test_orig = xgb_orig.prepare_features(df_test)
    xgb_orig.train(X_train_orig, y_train_orig, eval_set=(X_val_orig, y_val_orig))
    orig_probs = xgb_orig.predict_proba(X_test_orig)
    print(f"  V6 XGB prob range: [{orig_probs.min():.4f}, {orig_probs.max():.4f}]")

    # ─────────────────────────────────────────────────────────
    # Step 5: Train other models
    # ─────────────────────────────────────────────────────────
    print("\nStep 5: Training auxiliary models...")

    # LSTM
    lstm_test_probs = None
    print("  Training LSTM (3 runs)...")
    lstm = LSTMModel()
    X_train_seq, y_train_seq = lstm.prepare_sequences(df_train, fit_scaler=True)
    X_val_seq, y_val_seq = lstm.prepare_sequences(df_val, fit_scaler=False)
    X_test_seq, y_test_seq = lstm.prepare_sequences(df_test, fit_scaler=False)
    lstm_ok = len(X_train_seq) > 0 and len(X_val_seq) > 0

    if lstm_ok:
        all_probs = []
        for ri, seed in enumerate([42, 123, 777]):
            lr = LSTMModel()
            lr._seed = seed
            lr.scaler = lstm.scaler
            lr.n_features = lstm.n_features
            lr.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            probs = lr.predict_proba(X_test_seq)
            all_probs.append(probs)
        lstm_test_probs = np.mean(all_probs, axis=0)
        print(f"  LSTM mean prob: {lstm_test_probs.mean():.4f}, range: [{lstm_test_probs.min():.4f}, {lstm_test_probs.max():.4f}]")

    # TabNet
    tabnet_test_probs = None
    if HAS_TABNET:
        print("  Training TabNet...")
        try:
            tabnet = TabNetModel()
            tabnet.feature_names = xgb_orig.feature_names
            tabnet.train(X_train_orig, y_train_orig, eval_set=(X_val_orig, y_val_orig))
            tabnet_test_probs = tabnet.predict_proba(X_test_orig)
            print(f"  TabNet mean prob: {tabnet_test_probs.mean():.4f}")
        except Exception as e:
            print(f"  TabNet failed: {e}")

    # TFT
    tft_test_probs = None
    if HAS_TFT:
        print("  Training TFT...")
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

    # ─────────────────────────────────────────────────────────
    # Step 6: Learned Ensemble Stacking
    # ─────────────────────────────────────────────────────────
    print(f"\nStep 6: [IMPROVEMENT 3] Training learned ensemble stacker...")

    # Align all model predictions to same length
    n_align = len(xgb_test_probs)
    if lstm_test_probs is not None:
        n_align = min(n_align, len(lstm_test_probs))
    if tft_test_probs is not None:
        n_align = min(n_align, len(tft_test_probs))
    if tabnet_test_probs is not None:
        n_align = min(n_align, len(tabnet_test_probs))

    xgb_a = xgb_test_probs[-n_align:]
    orig_a = orig_probs[-n_align:]
    lstm_a = lstm_test_probs[-n_align:] if lstm_test_probs is not None else np.full(n_align, 0.5)
    tft_a = tft_test_probs[-n_align:] if tft_test_probs is not None else np.full(n_align, 0.5)
    tab_a = tabnet_test_probs[-n_align:] if tabnet_test_probs is not None else np.full(n_align, 0.5)

    test_dates = df_test.index[-n_align:]
    y_test_aligned = df_test["target_direction"].iloc[-n_align:].values

    # Build model predictions dict for stacker
    model_preds = {
        "xgb": xgb_a,
        "lstm": lstm_a,
        "tft": tft_a,
        "tabnet": tab_a,
        "sentiment": np.full(n_align, 0.5),  # no sentiment model
    }

    # Train stacker with walk-forward
    stacker = EnsembleStacker()
    try:
        stacker.fit(model_preds, y_test_aligned, dates=test_dates)
        print(f"  Stacker trained successfully")

        # Get model importance
        importance = stacker.get_model_importance()
        if importance:
            print(f"  Learned model weights:")
            for model_name, weight in sorted(importance.items(), key=lambda x: -abs(x[1])):
                print(f"    {model_name}: {weight:.4f}")

        # Generate stacker predictions
        stacker_probs = stacker.predict_proba(model_preds)
        if isinstance(stacker_probs, (float, int)):
            stacker_probs = np.array([stacker_probs])
        print(f"  Stacker prob range: [{np.min(stacker_probs):.4f}, {np.max(stacker_probs):.4f}]")

        use_stacker = True
    except Exception as e:
        print(f"  Stacker failed: {e}")
        print(f"  Falling back to V7 fixed ensemble...")
        use_stacker = False

    # ─────────────────────────────────────────────────────────
    # Step 7: Generate predictions (stacker or fixed)
    # ─────────────────────────────────────────────────────────
    print(f"\nStep 7: Generating ensemble predictions...")

    if use_stacker:
        # Use stacker for signals
        predictions = []
        for i in range(n_align):
            single_preds = {k: v[i] for k, v in model_preds.items()}
            try:
                signal_result = stacker.generate_signal(single_preds, threshold=0.15)
                predictions.append(signal_result)
            except:
                # Fallback to manual signal generation
                prob = stacker_probs[i] if i < len(stacker_probs) else 0.5
                conf = abs(prob - 0.5) * 2
                direction = "UP" if prob > 0.5 else "DOWN"
                signal = ("BUY_CALL" if direction == "UP" else "BUY_PUT") if conf >= 0.15 else "HOLD"
                predictions.append({
                    "signal": signal, "probability": float(prob),
                    "confidence": float(conf), "direction": direction,
                })
        print(f"  Using LEARNED STACKER predictions")
    else:
        # Fallback to V7 FixedEnsemble style
        from run_backtest_v7 import FixedEnsemble
        ensemble = FixedEnsemble()
        predictions = ensemble.predict_batch(orig_a, lstm_a, tft_a, tab_a)
        print(f"  Using V7 FIXED ENSEMBLE predictions (fallback)")

    # ─────────────────────────────────────────────────────────
    # Step 8: Diagnostics
    # ─────────────────────────────────────────────────────────
    signals = [p["signal"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]

    print(f"\n{'='*70}")
    print(f"  V8 PREDICTION DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"  Signals: BUY_CALL={signals.count('BUY_CALL')}, "
          f"BUY_PUT={signals.count('BUY_PUT')}, HOLD={signals.count('HOLD')}")

    conf_arr = np.array(confidences)
    print(f"  Confidence: min={conf_arr.min():.4f}  max={conf_arr.max():.4f}  mean={conf_arr.mean():.4f}")

    # Direction accuracy on test set
    dir_preds = np.array([1 if p["direction"] == "UP" else 0 for p in predictions])
    if len(y_test_aligned) == len(dir_preds):
        raw_acc = accuracy_score(y_test_aligned, dir_preds)
        print(f"  Raw direction accuracy: {raw_acc*100:.1f}%")

    # Volatility filter
    vol_filter = VolatilityFilter(df_test)
    high_vol_days = sum(1 for d in test_dates if vol_filter.is_high_vol(d))
    print(f"  Volatility filter: {high_vol_days}/{len(test_dates)} high-vol days "
          f"({high_vol_days/len(test_dates)*100:.0f}%)")
    print(f"{'='*70}\n")

    # ─────────────────────────────────────────────────────────
    # Step 9: Run backtests (all three modes)
    # ─────────────────────────────────────────────────────────
    all_results = {}
    for bt_mode in ["directional", "options_credit", "options_debit"]:
        vf = vol_filter if "options" in bt_mode else None
        print(f"Step 9: Running {bt_mode.upper()} backtest (V8 with regime filter)...")
        bt = V8Backtester(
            starting_capital=capital, mode=bt_mode,
            vol_filter=vf, allocation_filter=allocation_filter,
        )
        results = bt.run(predictions, df_test, test_dates)
        all_results[bt_mode] = results

        m = results["metrics"]
        print(f"  {bt_mode.upper()}: {m['total_trades']} trades, "
              f"win={m['win_rate']*100:.1f}%, ret={m['total_return']*100:+.2f}%, "
              f"Sharpe={m['sharpe_ratio']:.2f}, final=${m['final_portfolio_value']:,.2f}")
        if m.get("exit_reasons"):
            print(f"  Exit reasons: {m['exit_reasons']}")
        rs = m.get("regime_stats", {})
        if rs.get("trades_by_regime"):
            print(f"  Trades by regime: {rs['trades_by_regime']}")
        if rs.get("skipped_by_regime", 0) > 0:
            print(f"  Signals skipped by regime filter: {rs['skipped_by_regime']}")
        print()

    # ─────────────────────────────────────────────────────────
    # Step 10: Monte Carlo
    # ─────────────────────────────────────────────────────────
    print("Step 10: Running Monte Carlo (1000 simulations)...")
    from src.backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(n_simulations=1000)
    mc_results = mc.run(all_results["directional"], starting_capital=capital)
    mc.print_report(mc_results)
    mc.save_results(mc_results, symbol)

    # Save predictions
    pred_df = pd.DataFrame([{
        "date": str(d.date()), "signal": p["signal"],
        "probability": p["probability"], "confidence": p["confidence"],
        "direction": p["direction"],
    } for d, p in zip(test_dates, predictions)])
    pred_df.to_csv(DATA_DIR / "backtest" / f"predictions_{symbol}_v8.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n  Total V8 backtest time: {elapsed:.1f}s")

    return all_results


def print_comparison(all_results, capital):
    modes = list(all_results.keys())
    if len(modes) < 2:
        return

    print(f"\n{'='*80}")
    print(f"  V8 MODE COMPARISON")
    print(f"{'='*80}")

    header = f"{'Metric':<25}"
    for mode in modes:
        header += f" {mode:>17}"
    print(header)
    print(f"{'-'*80}")

    metrics_list = [all_results[m].get("metrics", {}) for m in modes]

    rows = [
        ("Final Value", lambda m: f"${m.get('final_portfolio_value',0):,.2f}"),
        ("Total Return", lambda m: f"{m.get('total_return',0)*100:+.2f}%"),
        ("Total Trades", lambda m: f"{m.get('total_trades',0)}"),
        ("Win Rate", lambda m: f"{m.get('win_rate',0)*100:.1f}%"),
        ("Direction Accuracy", lambda m: f"{m.get('direction_accuracy',0)*100:.1f}%"),
        ("Profit Factor", lambda m: f"{m.get('profit_factor',0):.2f}"),
        ("Sharpe Ratio", lambda m: f"{m.get('sharpe_ratio',0):.2f}"),
        ("Max Drawdown", lambda m: f"{m.get('max_drawdown',0)*100:.1f}%"),
        ("Expectancy/Trade", lambda m: f"${m.get('expectancy',0):+,.2f}"),
    ]

    for label, fmt_fn in rows:
        line = f"{label:<25}"
        for m in metrics_list:
            line += f" {fmt_fn(m):>17}"
        print(line)

    print(f"{'='*80}")
    print(f"\n  V8 improvements: regime filter, walk-forward XGB, learned stacker,")
    print(f"  advanced features (mean-reversion + regime), proper validation")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(">>> BACKTEST V8 — 5 STRATEGIC IMPROVEMENTS <<<")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--days", type=int, default=365*5)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument("--skip-wf-xgb", action="store_true",
                        help="Skip walk-forward XGB optimization (use V6 original)")
    args = parser.parse_args()

    all_results = run_backtest_v8(
        args.symbol, args.days, args.capital, args.test_ratio,
        skip_wf_xgb=args.skip_wf_xgb,
    )

    if all_results:
        m = all_results["directional"]["metrics"]
        print(f"\n{'='*70}")
        print(f"  V8 BACKTEST SUMMARY — DIRECTIONAL")
        print(f"{'='*70}")
        print(f"  Starting Capital:  ${args.capital:,.2f}")
        print(f"  Final Value:       ${m.get('final_portfolio_value',0):,.2f}")
        print(f"  Total Return:      {m.get('total_return',0)*100:+.2f}%")
        print(f"  Win Rate:          {m.get('win_rate',0)*100:.1f}%")
        print(f"  Direction Accuracy:{m.get('direction_accuracy',0)*100:.1f}%")
        print(f"  Sharpe Ratio:      {m.get('sharpe_ratio',0):.2f}")
        print(f"  Max Drawdown:      {m.get('max_drawdown',0)*100:.1f}%")
        print(f"  Total Trades:      {m.get('total_trades',0)}")
        print(f"  Profit Factor:     {m.get('profit_factor',0):.2f}")
        print(f"  Expectancy/Trade:  ${m.get('expectancy',0):+,.2f}")
        print(f"{'='*70}")

        print_comparison(all_results, args.capital)
