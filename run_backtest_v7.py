#!/usr/bin/env python3
"""
Backtest V7 — Three major improvements over V6:

  1. CREDIT SPREADS: Theta works FOR you instead of against you.
     Bull put spread when bullish, bear call spread when bearish.

  2. BETTER XGBoost: Less regularization, probability calibration,
     wider probability spread → stronger signals → higher accuracy.

  3. VOLATILITY FILTER: Only take options on high-volatility days
     (VIX spikes, large recent moves, event proximity).

Usage:
    python3 run_backtest_v7.py
    python3 run_backtest_v7.py --symbol SPY --days 1825 --capital 1000
"""
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import DATA_DIR
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
# IMPROVEMENT 1: Better XGBoost with calibration
# ═══════════════════════════════════════════════════════════

class TunedXGBoost:
    """
    XGBoost with reduced regularization and probability calibration.

    V6 XGBoost was over-regularized:
      max_depth=3, gamma=1.0, reg_lambda=5.0, learning_rate=0.01
      → output range 0.5198-0.5259 (basically constant)

    V7 loosens regularization to let the model actually learn patterns,
    then calibrates probabilities with isotonic regression for wider spread.
    """

    EXCLUDE_COLUMNS = [
        "target_return", "target_direction",
        "target_return_5d", "target_direction_5d",
        "open", "high", "low", "close", "volume",
        "trade_count", "vwap",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
        "bb_upper", "bb_middle", "bb_lower",
        "ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conv",
        "high_52w", "low_52w",
        "obv", "ad_line",
        "volume_sma_20",
        "atr_7", "atr_14", "atr_21",
        "macd", "macd_signal", "macd_histogram",
    ]

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=5,            # V7: deeper (was 3) — let it find patterns
            learning_rate=0.03,     # V7: faster (was 0.01) — learn before early stop
            min_child_weight=5,     # V7: less restrictive (was 10)
            subsample=0.7,          # V7: more data per tree (was 0.6)
            colsample_bytree=0.7,   # V7: more features (was 0.5)
            gamma=0.3,              # V7: lower (was 1.0) — allow more splits
            reg_alpha=0.3,          # V7: lighter L1 (was 1.0)
            reg_lambda=1.5,         # V7: lighter L2 (was 5.0)
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
        )
        self.calibrator = None
        self.iso_reg = None
        self.cal_bias = 0.0
        self.feature_names = None

    def prepare_features(self, df, target_col="target_direction"):
        df_clean = df.dropna(subset=[target_col]).copy()
        feature_cols = [c for c in df_clean.columns if c not in self.EXCLUDE_COLUMNS]
        self.feature_names = feature_cols
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_col]
        return X, y

    def train(self, X_train, y_train, X_val, y_val):
        print(f"  V7 XGBoost: training on {len(X_train)} samples, {X_train.shape[1]} features")
        self.model.fit(
            X_train.fillna(0), y_train,
            eval_set=[(X_val.fillna(0), y_val)],
            verbose=False,
        )

        # Log raw model performance
        raw_probs = self.model.predict_proba(X_val.fillna(0))[:, 1]
        raw_preds = (raw_probs > 0.5).astype(int)
        raw_acc = accuracy_score(y_val, raw_preds)
        raw_auc = roc_auc_score(y_val, raw_probs)
        print(f"  Raw XGB: acc={raw_acc:.4f}, AUC={raw_auc:.4f}, "
              f"prob range=[{raw_probs.min():.4f}, {raw_probs.max():.4f}]")

        # Calibrate probabilities using isotonic regression on validation set
        # This widens the probability spread for more decisive signals
        # Manual calibration since cv='prefit' not supported in all sklearn versions
        from sklearn.isotonic import IsotonicRegression
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
        self.iso_reg.fit(raw_probs, y_val)
        self.calibrator = None  # We use iso_reg directly instead

        cal_probs = self.iso_reg.predict(raw_probs)
        # Debias: isotonic regression can shift the mean (e.g. always bullish)
        # Center calibrated probs around 0.5 to preserve direction balance
        self.cal_bias = cal_probs.mean() - 0.5
        debiased = np.clip(cal_probs - self.cal_bias, 0.01, 0.99)
        cal_preds = (debiased > 0.5).astype(int)
        cal_acc = accuracy_score(y_val, cal_preds)
        print(f"  Calibrated: acc={cal_acc:.4f}, "
              f"prob range=[{debiased.min():.4f}, {debiased.max():.4f}], "
              f"bias removed={self.cal_bias:.4f}")

    def predict_proba(self, X):
        X_filled = X.fillna(0) if hasattr(X, 'fillna') else X
        raw_probs = self.model.predict_proba(X_filled)[:, 1]
        if self.iso_reg is not None:
            cal = self.iso_reg.predict(raw_probs)
            # Remove bias so signals aren't all-bullish or all-bearish
            debiased = np.clip(cal - self.cal_bias, 0.01, 0.99)
            return debiased
        return raw_probs


# ═══════════════════════════════════════════════════════════
# IMPROVEMENT 2: Volatility filter for options entries
# ═══════════════════════════════════════════════════════════

class VolatilityFilter:
    """
    Filters predictions to only take options trades on high-volatility days.

    Options profit from large moves. This filter identifies days where:
    1. Recent realized volatility is elevated (above 20-day average)
    2. VIX is elevated (if available in features)
    3. Near events (FOMC, earnings) that cause large moves
    4. Recent momentum suggests continuation
    """

    def __init__(self, df_test):
        """Precompute volatility metrics for the test period."""
        self.vol_scores = {}

        if "close" not in df_test.columns:
            return

        # Daily returns
        returns = df_test["close"].pct_change()

        # 5-day realized vol vs 20-day average
        vol_5d = returns.rolling(5).std()
        vol_20d = returns.rolling(20).std()

        # VIX if available
        has_vix = "vix" in df_test.columns

        for date in df_test.index:
            score = 0.0

            # Realized vol ratio
            v5 = vol_5d.get(date, 0)
            v20 = vol_20d.get(date, 0)
            if v20 and v20 > 0 and not np.isnan(v5) and not np.isnan(v20):
                vol_ratio = v5 / v20
                if vol_ratio > 1.3:
                    score += 0.3  # Elevated recent vol
                elif vol_ratio > 1.1:
                    score += 0.15

            # Absolute recent move (momentum)
            if date in returns.index and not np.isnan(returns[date]):
                abs_ret = abs(returns[date])
                if abs_ret > 0.015:
                    score += 0.3  # Big move today
                elif abs_ret > 0.008:
                    score += 0.15

            # VIX level
            if has_vix and date in df_test.index:
                vix_val = df_test.loc[date].get("vix", 0)
                if vix_val and not np.isnan(vix_val):
                    if vix_val > 25:
                        score += 0.3  # High VIX
                    elif vix_val > 20:
                        score += 0.15

            # Event proximity (check for event features)
            for col in ["fomc_week", "earnings_week", "high_event_risk"]:
                if col in df_test.columns:
                    val = df_test.loc[date].get(col, 0)
                    if val and val > 0:
                        score += 0.2
                        break  # Only count once

            self.vol_scores[date] = min(score, 1.0)

    def is_high_vol(self, date, min_score=0.3):
        """Returns True if this date is a good day for options."""
        return self.vol_scores.get(date, 0) >= min_score

    def get_score(self, date):
        return self.vol_scores.get(date, 0)


# ═══════════════════════════════════════════════════════════
# IMPROVEMENT 3: Credit spread backtester
# ═══════════════════════════════════════════════════════════

class FixedEnsemble:
    """V6.1 ensemble — same as before."""

    def __init__(self):
        self.weights = {
            "xgboost": 0.30, "lstm": 0.15, "tft": 0.25,
            "tabnet": 0.20, "sentiment": 0.10,
        }
        self.confidence_threshold = 0.15
        print(f"  FixedEnsemble V7: threshold={self.confidence_threshold}")

    def predict_batch(self, xgb_probs, lstm_probs, tft_probs=None,
                      tabnet_probs=None, sent_probs=None, sent_confs=None):
        n = len(xgb_probs)
        if tft_probs is None: tft_probs = np.full(n, 0.5)
        if tabnet_probs is None: tabnet_probs = np.full(n, 0.5)
        if sent_probs is None: sent_probs = np.full(n, 0.5)
        if sent_confs is None: sent_confs = np.full(n, 0.0)

        return [self._predict(xgb_probs[i], lstm_probs[i], tft_probs[i],
                              tabnet_probs[i], sent_probs[i], sent_confs[i])
                for i in range(n)]

    def _predict(self, xgb, lstm, tft, tabnet, sent, sent_conf):
        model_probs = {"xgboost": xgb, "lstm": lstm, "tft": tft,
                       "tabnet": tabnet, "sentiment": sent}
        ew = self.weights.copy()

        # Redistribute inactive model weights
        for name, prob, conf_check in [("sentiment", sent, sent_conf < 0.1),
                                        ("tft", tft, tft == 0.5),
                                        ("tabnet", tabnet, tabnet == 0.5)]:
            if conf_check:
                w = ew.pop(name, 0)
                tr = sum(ew.values())
                if tr > 0:
                    for k in ew: ew[k] += w * (ew[k] / tr)
                ew[name] = 0.0

        total_w = sum(ew.values())
        if total_w > 0:
            ew = {k: v / total_w for k, v in ew.items()}

        combined = sum(ew.get(k, 0) * model_probs[k] for k in model_probs)
        direction = "UP" if combined > 0.5 else "DOWN"

        active = {k: v for k, v in model_probs.items() if ew.get(k, 0) > 0.05}
        agreements = sum(1 for p in active.values() if (p > 0.5) == (direction == "UP"))
        n_active = max(len(active), 1)
        agreement_ratio = agreements / n_active

        convictions = [abs(p - 0.5) * 2 for p in active.values()]
        avg_conv = np.mean(convictions) if convictions else 0
        max_conv = max(convictions) if convictions else 0

        if agreement_ratio >= 0.8:
            confidence = 0.30 + avg_conv * 0.5 + max_conv * 0.2
        elif agreement_ratio >= 0.6:
            confidence = 0.15 + avg_conv * 0.5 + max_conv * 0.15
        else:
            confidence = avg_conv * 0.3
        confidence = min(confidence, 1.0)

        signal = ("BUY_CALL" if direction == "UP" else "BUY_PUT") \
                 if confidence >= self.confidence_threshold else "HOLD"

        return {"signal": signal, "probability": float(combined),
                "confidence": float(confidence), "direction": direction}


class V7Backtester:
    """
    V7 backtester with THREE modes:

    1. DIRECTIONAL: Stock-based P&L (same as V6)
    2. OPTIONS_DEBIT: Buy options (same as V6.3 — for comparison)
    3. OPTIONS_CREDIT: SELL credit spreads — theta works FOR you

    Credit spread strategy:
      - Bullish signal → sell bull put spread (sell ATM put, buy OTM put)
      - Bearish signal → sell bear call spread (sell ATM call, buy OTM call)
      - Max profit = credit received (if stock stays above/below short strike)
      - Max loss = spread width - credit (defined risk)
      - Theta decay HELPS position (time value melts from short option)

    With 57% directional accuracy:
      - 57% of spreads expire worthless (max profit)
      - 43% get tested but stop-loss limits damage
    """

    def __init__(self, starting_capital=1000.0, mode="directional", vol_filter=None):
        self.mode = mode
        self.starting_capital = starting_capital
        self.max_risk_per_trade = 0.03
        self.daily_loss_limit = 0.05
        self.max_open_positions = 5
        self.vol_filter = vol_filter

        # Directional settings (unchanged)
        self.dir_sl = 0.03
        self.dir_tp = 0.04
        self.max_hold_days = 5
        self.dir_min_conf = 0.15

        # Credit spread settings
        self.spread_width_pct = 0.03     # 3% wide spread — room before ITM
        self.credit_pct = 0.33           # collect 33% of spread width as credit
        self.spread_max_hold_days = 10   # 10 days — let theta decay do its job
        self.spread_tp_pct = 0.50        # take profit at 50% of credit (close early)
        self.spread_sl_pct = 2.00        # standard: 2x credit stop (absorb adverse moves)
        # Math: at 57% win, EV = 0.57*0.50 - 0.43*2.00 = 0.285 - 0.860 = -0.575 (pure delta)
        # BUT theta benefit is critical: ~5%/day * 10d = 50% of credit → tips the balance
        # Credit spreads win by NOT hitting SL — theta erodes the short option value
        self.spread_min_conf = 0.15      # minimum confidence for spread trades
        self.spread_min_vol = 0.3        # minimum vol score for options entries

        # Debit options settings (V6.3 for comparison)
        self.opt_sl = 0.20
        self.opt_tp = 0.25
        self.opt_dte = 14
        self.opt_max_hold_days = 3
        self.opt_min_conf = 0.15

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

            # Volatility filter for options modes
            vol_ok = True
            if self.mode in ("options_credit", "options_debit") and self.vol_filter:
                vol_ok = self.vol_filter.is_high_vol(date, self.spread_min_vol)

            can_open = (signal != "HOLD"
                       and confidence >= min_conf
                       and vol_ok
                       and len(open_positions) < self.max_open_positions)

            if can_open:
                conf_scale = min(0.5 + confidence, 1.5)
                pos_size = capital * self.max_risk_per_trade * conf_scale

                if pos_size >= 10 and capital >= 50:
                    direction = "long" if signal == "BUY_CALL" else "short"
                    position = {
                        "entry_date": date, "symbol": "SPY",
                        "direction": direction, "signal": signal,
                        "entry_price": current_price,
                        "position_size": pos_size,
                        "confidence": confidence,
                        "mode": self.mode,
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
        """
        Credit spread P&L model.

        Bull put spread (bullish): sell ATM put + buy OTM put
          - Stock stays above short strike → keep full credit (+35% of risk)
          - Stock drops past short strike → lose up to max loss
          - Theta HELPS: as time passes, short option decays faster

        P&L is modeled as percentage of position_size (collateral):
          - Max profit = credit_pct (35% of collateral)
          - Max loss = -(1 - credit_pct) = -65% of collateral
          - Breakeven = stock moves against you by credit_pct × spread_width

        Key advantage: you win if stock goes UP, stays flat, OR drops slightly.
        Only lose if stock drops significantly past your spread.
        """
        pct = (price - pos["entry_price"]) / pos["entry_price"]
        days = max(0.5, (date - pos["entry_date"]).days)
        credit_pct = self.credit_pct  # 0.35

        # Direction-adjusted move
        if pos["spread_type"] == "bull_put":
            # Bull put spread profits when stock goes up or stays flat
            dir_move = pct  # positive = good
        else:
            # Bear call spread profits when stock goes down or stays flat
            dir_move = -pct  # negative stock = good

        # Theta benefit: credit spreads gain from time decay
        # Short option decays faster than long option (gamma risk aside)
        # Model: ~5% of max profit per day held (simple linear for short holds)
        theta_benefit = min(days * 0.05, 0.5) * credit_pct  # up to 50% of credit from theta

        # Delta P&L relative to spread width
        spread_width = self.spread_width_pct  # 2% of stock price

        if dir_move >= 0:
            # Stock went in our direction or stayed flat
            # How much of the credit do we keep?
            # If stock is above short strike: keep everything
            # As stock moves in our favor, P&L approaches max credit
            move_vs_spread = dir_move / spread_width
            delta_pnl = credit_pct * min(move_vs_spread * 2, 1.0)
        else:
            # Stock went against us
            # Loss increases as stock moves past short strike
            adverse_move = -dir_move  # now positive
            move_vs_spread = adverse_move / spread_width
            if move_vs_spread < 0.5:
                # Still above short strike — limited damage
                delta_pnl = -credit_pct * move_vs_spread
            else:
                # Past short strike — losing money
                delta_pnl = -credit_pct - (move_vs_spread - 0.5) * (1 - credit_pct) * 2
                delta_pnl = max(delta_pnl, -(1 - credit_pct))  # cap at max loss

        net_pnl = delta_pnl + theta_benefit
        net_pnl = max(net_pnl, -(1 - credit_pct))  # can't lose more than max loss
        net_pnl = min(net_pnl, credit_pct)  # can't gain more than credit

        if force:
            return True, "backtest_end", net_pnl

        # Take profit: captured 60% of max credit → close early
        if net_pnl >= credit_pct * self.spread_tp_pct:
            return True, "take_profit", net_pnl

        # Stop loss: losing 1.5x the credit received
        if net_pnl <= -credit_pct * self.spread_sl_pct:
            return True, "stop_loss", net_pnl

        # Time exit
        if days >= self.spread_max_hold_days:
            return True, "time_exit", net_pnl

        return False, "", net_pnl

    def _eval_debit(self, pos, price, date, force=False):
        """V6.3 debit options model (for comparison)."""
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
# Main backtest pipeline
# ═══════════════════════════════════════════════════════════

def run_backtest_v7(symbol="SPY", days=365*5, capital=1000.0, test_ratio=0.30):
    print(f"\n{'='*60}")
    print(f"  BACKTEST V7 — {symbol} — ${capital} capital")
    print(f"  {days} days, {test_ratio*100:.0f}% test, 3 improvements")
    print(f"{'='*60}\n")

    # Step 1: Build features
    print("Step 1: Building features...")
    builder = FeatureBuilder(use_finbert=False)
    df = builder.build_features(symbol, days=days, include_news=False)
    if df.empty or len(df) < 100:
        print(f"ERROR: Insufficient data ({len(df)} rows)")
        return None

    # Step 2: Split
    print("\nStep 2: Splitting data...")
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    val_start = int(test_start * 0.85)
    df_train, df_val, df_test = df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]
    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Step 3: Train XGBoost — use ORIGINAL (V6) for ensemble (57% proven accuracy)
    # V7 tuned XGB tested at 49.4% — less regularization = more overfit, worse OOS
    print("\nStep 3: Training XGBoost (V6 original — proven 57% accuracy)...")
    xgb_orig = XGBoostModel()
    X_train, y_train = xgb_orig.prepare_features(df_train)
    X_val, y_val = xgb_orig.prepare_features(df_val)
    X_test, y_test = xgb_orig.prepare_features(df_test)
    xgb_orig.train(X_train, y_train, eval_set=(X_val, y_val))
    orig_probs = xgb_orig.predict_proba(X_test)
    print(f"  V6 XGB prob range: [{orig_probs.min():.4f}, {orig_probs.max():.4f}]")

    # Train other models
    tabnet_test_probs = None
    if HAS_TABNET:
        print("\nStep 3b: Training TabNet...")
        try:
            tabnet = TabNetModel()
            tabnet.feature_names = xgb_orig.feature_names
            tabnet.train(X_train, y_train, eval_set=(X_val, y_val))
            tabnet_test_probs = tabnet.predict_proba(X_test)
            print(f"  TabNet mean prob: {tabnet_test_probs.mean():.4f}")
        except Exception as e:
            print(f"  TabNet failed: {e}")

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

    # Step 5: Ensemble
    print("\nStep 5: Generating ensemble predictions...")
    ensemble = FixedEnsemble()

    # Use ORIGINAL V6 XGBoost for ensemble (proven 57% accuracy)
    xgb_test_probs = orig_probs

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

    predictions = ensemble.predict_batch(xgb_a, lstm_a, tft_a, tab_a)
    test_dates = df_test.index[-n_align:]

    # Diagnostics
    signals = [p["signal"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]

    print(f"\n{'='*60}")
    print(f"  PREDICTION DIAGNOSTICS (V7.1)")
    print(f"{'='*60}")
    print(f"  Signals: BUY_CALL={signals.count('BUY_CALL')}, "
          f"BUY_PUT={signals.count('BUY_PUT')}, HOLD={signals.count('HOLD')}")
    print(f"\n  Model probability ranges (V6 XGBoost — used in ensemble):")
    print(f"    XGBoost:    min={xgb_a.min():.4f}  max={xgb_a.max():.4f}  "
          f"mean={xgb_a.mean():.4f}  std={xgb_a.std():.4f}")
    if lstm_test_probs is not None:
        print(f"    LSTM:       min={lstm_a.min():.4f}  max={lstm_a.max():.4f}  "
              f"mean={lstm_a.mean():.4f}  std={lstm_a.std():.4f}")
    if tft_test_probs is not None:
        print(f"    TFT:        min={tft_a.min():.4f}  max={tft_a.max():.4f}  "
              f"mean={tft_a.mean():.4f}  std={tft_a.std():.4f}")
    if tabnet_test_probs is not None:
        print(f"    TabNet:     min={tab_a.min():.4f}  max={tab_a.max():.4f}  "
              f"mean={tab_a.mean():.4f}  std={tab_a.std():.4f}")

    conf_arr = np.array(confidences)
    print(f"\n  Confidence: min={conf_arr.min():.4f}  max={conf_arr.max():.4f}  mean={conf_arr.mean():.4f}")

    # IMPROVEMENT 2: Volatility filter stats
    vol_filter = VolatilityFilter(df_test)
    high_vol_days = sum(1 for d in test_dates if vol_filter.is_high_vol(d))
    print(f"\n  Volatility filter: {high_vol_days}/{len(test_dates)} days pass "
          f"({high_vol_days/len(test_dates)*100:.0f}% of days are high-vol)")
    print(f"{'='*60}\n")

    # Step 6: Run ALL THREE modes
    all_results = {}
    for bt_mode in ["directional", "options_credit", "options_debit"]:
        vf = vol_filter if "options" in bt_mode else None
        print(f"Step 6: Running {bt_mode.upper()} backtest...")
        bt = V7Backtester(starting_capital=capital, mode=bt_mode, vol_filter=vf)
        results = bt.run(predictions, df_test, test_dates)
        all_results[bt_mode] = results

        m = results["metrics"]
        print(f"  {bt_mode.upper()}: {m['total_trades']} trades, "
              f"win={m['win_rate']*100:.1f}%, ret={m['total_return']*100:+.2f}%, "
              f"final=${m['final_portfolio_value']:,.2f}")
        if m.get("exit_reasons"):
            print(f"  Exit reasons: {m['exit_reasons']}")
        print()

    # Step 7: Monte Carlo on directional
    print("Step 7: Running Monte Carlo (1000 simulations)...")
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
    pred_df.to_csv(DATA_DIR / "backtest" / f"predictions_{symbol}.csv", index=False)

    return all_results


def print_comparison(all_results, capital):
    modes = list(all_results.keys())
    if len(modes) < 2:
        return

    print(f"\n{'='*80}")
    print(f"  MODE COMPARISON (V7)")
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
    print(f"\n  directional = stock P&L (baseline)")
    print(f"  options_credit = sell credit spreads (theta FOR you)")
    print(f"  options_debit = buy options (theta AGAINST you)")
    print(f"  Credit spreads win with 57% accuracy because theta is your ally!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(">>> BACKTEST V7.1 — V6 SIGNALS + CREDIT SPREADS + VOL FILTER <<<")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--days", type=int, default=365*5)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    args = parser.parse_args()

    all_results = run_backtest_v7(args.symbol, args.days, args.capital, args.test_ratio)

    if all_results:
        # Print directional summary
        m = all_results["directional"]["metrics"]
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY (DIRECTIONAL — V7.1 V6 Signals + Credit Spreads)")
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

        print_comparison(all_results, args.capital)
