"""
Walk-Forward Backtester — Implements expanding-window walk-forward backtesting.

Walk-forward backtesting prevents data leakage by:
  1. Training on months 1-6 (train_window)
  2. Testing on month 7 (test_window), recording out-of-sample performance
  3. Retraining on months 1-7
  4. Testing on month 8, etc.

This gives realistic performance estimates since the test set is never seen during training.

Supports two strategy modes:
  - DIRECTIONAL: Stock-based trades (simple delta)
  - CREDIT_SPREAD: Sell bull put / bear call spreads (theta works for you)
  - BOTH: Run both modes in parallel for comparison

Usage:
    from src.backtest.walkforward_backtester import WalkForwardBacktester

    wf = WalkForwardBacktester(
        min_train_days=120,
        test_days=21,
        strategy_mode="directional",
    )
    results = wf.run_walkforward(
        df, feature_cols, target_col,
        model_factory=lambda X, y: my_model.fit(X, y)
    )

    print(wf.get_aggregate_metrics())
    wf.plot_summary()
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import get_settings, DATA_DIR


class WalkForwardBacktester:
    """Walk-forward backtester with expanding training windows and fixed test windows."""

    VALID_STRATEGIES = ("directional", "credit_spread", "both")

    def __init__(
        self,
        min_train_days: int = 120,
        test_days: int = 21,
        step_days: int = None,
        retrain_models: bool = True,
        strategy_mode: str = "directional",
        starting_capital: float = 1000.0,
    ):
        """
        Initialize walk-forward backtester.

        Args:
            min_train_days: Minimum training window in trading days (default ~6 months)
            test_days: Fixed test window size in trading days (default ~1 month)
            step_days: How far to advance between folds (default = test_days, no overlap)
            retrain_models: If True, retrain models each fold; if False, only retrain stacker
            strategy_mode: 'directional', 'credit_spread', or 'both'
            starting_capital: Starting portfolio value per fold
        """
        if strategy_mode not in self.VALID_STRATEGIES:
            raise ValueError(f"strategy_mode must be one of {self.VALID_STRATEGIES}")

        self.min_train_days = min_train_days
        self.test_days = test_days
        self.step_days = step_days or test_days
        self.retrain_models = retrain_models
        self.strategy_mode = strategy_mode
        self.starting_capital = starting_capital

        settings = get_settings()
        trading = settings.get("trading", {})

        # Directional mode settings
        self.dir_sl = 0.03  # 3% stop loss
        self.dir_tp = 0.04  # 4% take profit
        self.max_hold_days = 5

        # Credit spread settings
        self.spread_width_pct = 0.03  # 3% wide spread
        self.credit_pct = 0.33  # 33% of spread as credit
        self.spread_tp_pct = 0.50  # 50% of credit for take profit
        self.spread_sl_pct = 2.00  # 2x credit for stop loss
        self.spread_max_hold_days = 10

        self.max_risk_per_trade = trading.get("max_portfolio_risk_per_trade", 0.03)
        self.daily_loss_limit = trading.get("daily_loss_limit", 0.05)
        self.max_open_positions = trading.get("max_open_positions", 5)

        # Results storage
        self.fold_results = []
        self.fold_trades = []
        self.fold_predictions = []
        self.combined_portfolio_history = []
        self.results_dir = DATA_DIR / "backtest_walkforward"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"WalkForwardBacktester initialized: "
            f"min_train={min_train_days}d, test={test_days}d, step={self.step_days}d, "
            f"mode={strategy_mode}"
        )

    # ── Main walk-forward loop ────────────────────────────────────

    def run_walkforward(self, df, feature_cols, target_col, model_factory):
        """
        Run full walk-forward backtest.

        Args:
            df: DataFrame with all data, datetime-indexed
            feature_cols: List of feature column names
            target_col: Target column name (binary: 0 or 1)
            model_factory: Callable(X_train, y_train) -> object with predict_proba(X_test)

        Returns:
            Dict with fold_results, aggregate_metrics, and other diagnostics
        """
        df_clean = df.dropna(subset=[target_col]).copy()
        if len(df_clean) < self.min_train_days + self.test_days:
            logger.error(
                f"Insufficient data: {len(df_clean)} rows < "
                f"{self.min_train_days + self.test_days} required"
            )
            return None

        # Convert index to trading days if not already
        dates = df_clean.index
        n_total = len(df_clean)

        logger.info(f"Starting walk-forward: {n_total} total rows, {len(dates)} dates")

        fold_num = 0
        train_end = self.min_train_days

        while train_end + self.test_days <= n_total:
            fold_num += 1
            test_start = train_end
            test_end = min(train_end + self.test_days, n_total)

            logger.info(
                f"\nFold {fold_num}: train={train_end} rows "
                f"({dates[0].date()}-{dates[train_end-1].date()}), "
                f"test={test_end - test_start} rows "
                f"({dates[test_start].date()}-{dates[test_end-1].date()})"
            )

            # Split data
            df_train = df_clean.iloc[:train_end]
            df_test = df_clean.iloc[test_start:test_end]

            X_train = df_train[feature_cols].fillna(0).values
            y_train = df_train[target_col].values
            X_test = df_test[feature_cols].fillna(0).values
            y_test = df_test[target_col].values
            test_dates = df_test.index

            # Train model
            try:
                model = model_factory(X_train, y_train)
                test_probs = model.predict_proba(X_test)
                if len(test_probs.shape) > 1:
                    test_probs = test_probs[:, 1]
            except Exception as e:
                logger.error(f"Model training failed in fold {fold_num}: {e}")
                train_end += self.step_days
                continue

            # Generate signals (simple ensemble: directional)
            predictions = self._generate_predictions(test_probs)

            # Run backtest for this fold
            if self.strategy_mode in ("directional", "both"):
                fold_results_dir = self._run_fold_backtest(
                    predictions, df_test, test_dates, fold_num, mode="directional"
                )
                if fold_results_dir:
                    self.fold_results.append(fold_results_dir)

            if self.strategy_mode in ("credit_spread", "both"):
                fold_results_cs = self._run_fold_backtest(
                    predictions, df_test, test_dates, fold_num, mode="credit_spread"
                )
                if fold_results_cs:
                    self.fold_results.append(fold_results_cs)

            # Record predictions for aggregate analysis
            fold_pred_record = {
                "fold": fold_num,
                "test_dates": test_dates.tolist(),
                "test_probs": test_probs,
                "y_test": y_test,
                "signals": [p["signal"] for p in predictions],
                "confidences": [p["confidence"] for p in predictions],
            }
            self.fold_predictions.append(fold_pred_record)

            train_end += self.step_days

        if not self.fold_results:
            logger.error("No folds completed successfully")
            return None

        logger.info(f"\nCompleted {len(self.fold_results)} folds")

        # Aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics()

        results = {
            "fold_results": self.fold_results,
            "aggregate_metrics": aggregate_metrics,
            "fold_predictions": self.fold_predictions,
            "config": {
                "min_train_days": self.min_train_days,
                "test_days": self.test_days,
                "step_days": self.step_days,
                "strategy_mode": self.strategy_mode,
                "starting_capital": self.starting_capital,
            },
        }

        self._log_summary(results)
        return results

    # ── Fold backtesting ─────────────────────────────────────────

    def _run_fold_backtest(self, predictions, df_test, test_dates, fold_num, mode):
        """Run a single fold's backtest in specified mode."""
        capital = self.starting_capital
        portfolio_history = []
        trades = []
        open_positions = []
        daily_pnl = 0.0
        daily_start_capital = capital

        if mode == "directional":
            min_conf = 0.15
        else:  # credit_spread
            min_conf = 0.15

        for i, (pred, date) in enumerate(zip(predictions, test_dates)):
            # Reset daily tracking
            if i > 0 and test_dates[i].date() != test_dates[i - 1].date():
                daily_pnl = 0.0
                daily_start_capital = capital

            current_price = df_test.loc[date, "close"] if date in df_test.index else None
            if current_price is None:
                portfolio_history.append(
                    {"date": date, "capital": capital, "open_positions": len(open_positions)}
                )
                continue

            # Evaluate and close existing positions
            positions_to_close = []
            for pos_idx, pos in enumerate(open_positions):
                should_close, exit_reason, pnl_pct = self._eval_position(
                    pos, current_price, date, mode
                )
                if should_close:
                    pos["exit_date"] = date
                    pos["exit_price"] = current_price
                    pos["exit_reason"] = exit_reason
                    pos["pnl_pct"] = pnl_pct
                    pos["pnl_dollar"] = pos["position_size"] * pnl_pct
                    positions_to_close.append(pos_idx)

            # Close positions (reverse order)
            for pos_idx in sorted(positions_to_close, reverse=True):
                closed_pos = open_positions.pop(pos_idx)
                capital += closed_pos["position_size"] + closed_pos["pnl_dollar"]
                daily_pnl += closed_pos["pnl_dollar"]
                trades.append(closed_pos)

            # Check daily loss limit
            if daily_start_capital > 0:
                daily_loss_pct = -daily_pnl / daily_start_capital
                if daily_loss_pct >= self.daily_loss_limit:
                    portfolio_history.append(
                        {"date": date, "capital": capital, "open_positions": len(open_positions)}
                    )
                    continue

            # Open new position
            signal = pred.get("signal", "HOLD")
            confidence = pred.get("confidence", 0)

            can_open = (
                signal != "HOLD"
                and confidence >= min_conf
                and len(open_positions) < self.max_open_positions
            )

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
                        "mode": mode,
                    }

                    if mode == "credit_spread":
                        position["spread_type"] = (
                            "bull_put" if signal == "BUY_CALL" else "bear_call"
                        )
                        position["credit"] = pos_size * self.credit_pct
                        position["max_loss"] = pos_size * (1 - self.credit_pct)

                    open_positions.append(position)
                    capital -= pos_size

            # Record portfolio state
            open_value = sum(
                pos["position_size"] * max(0, 1 + self._eval_position(pos, current_price, date, mode)[2])
                for pos in open_positions
            )

            portfolio_history.append(
                {
                    "date": date,
                    "capital": capital,
                    "open_value": max(0, open_value),
                    "total_value": capital + max(0, open_value),
                    "open_positions": len(open_positions),
                }
            )

        # Close remaining positions at last price
        last_price = df_test["close"].iloc[-1]
        for pos in open_positions:
            _, _, pnl_pct = self._eval_position(pos, last_price, test_dates[-1], mode, force_close=True)
            pos["exit_date"] = test_dates[-1]
            pos["exit_price"] = last_price
            pos["exit_reason"] = "backtest_end"
            pos["pnl_pct"] = pnl_pct
            pos["pnl_dollar"] = pos["position_size"] * pnl_pct
            capital += pos["position_size"] + pos["pnl_dollar"]
            trades.append(pos)

        # Calculate metrics for this fold
        metrics = self._calculate_fold_metrics(trades, portfolio_history, mode)

        fold_result = {
            "fold": fold_num,
            "mode": mode,
            "trades": trades,
            "portfolio_history": portfolio_history,
            "metrics": metrics,
        }

        self.fold_trades.append((fold_num, mode, trades))
        self.combined_portfolio_history.extend(
            [{**h, "fold": fold_num, "mode": mode} for h in portfolio_history]
        )

        logger.info(
            f"  Fold {fold_num} ({mode}): {metrics['total_trades']} trades, "
            f"return={metrics['total_return']*100:+.2f}%, "
            f"sharpe={metrics['sharpe_ratio']:.2f}"
        )

        return fold_result

    # ── Position evaluation ──────────────────────────────────────

    def _eval_position(self, pos, current_price, date, mode, force_close=False):
        """Evaluate position and determine if it should be closed."""
        if mode == "directional":
            return self._eval_directional(pos, current_price, date, force_close)
        else:  # credit_spread
            return self._eval_credit_spread(pos, current_price, date, force_close)

    def _eval_directional(self, pos, current_price, date, force_close=False):
        """Directional mode: stock-based P&L."""
        price_change_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        days_held = max(1, (date - pos["entry_date"]).days)

        if pos["direction"] == "long":
            pnl_pct = price_change_pct
        else:
            pnl_pct = -price_change_pct

        if force_close:
            return True, "backtest_end", pnl_pct

        if pnl_pct <= -self.dir_sl:
            return True, "stop_loss", -self.dir_sl

        if pnl_pct >= self.dir_tp:
            return True, "take_profit", self.dir_tp

        if days_held >= self.max_hold_days:
            return True, "max_hold", pnl_pct

        return False, "", pnl_pct

    def _eval_credit_spread(self, pos, current_price, date, force_close=False):
        """Credit spread P&L model with theta benefit."""
        price_change_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        days_held = max(0.5, (date - pos["entry_date"]).days)
        credit_pct = self.credit_pct

        # Direction-adjusted move
        if pos["spread_type"] == "bull_put":
            # Profits when stock goes up
            dir_move = price_change_pct
        else:
            # Bear call: profits when stock goes down
            dir_move = -price_change_pct

        # Theta benefit: credit spreads gain from time decay
        theta_benefit = min(days_held * 0.05, 0.5) * credit_pct

        spread_width = self.spread_width_pct

        if dir_move >= 0:
            # Stock went in our direction
            move_vs_spread = dir_move / spread_width
            delta_pnl = credit_pct * min(move_vs_spread * 2, 1.0)
        else:
            # Stock went against us
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

        if force_close:
            return True, "backtest_end", net_pnl

        # Take profit: at 50% of max credit
        if net_pnl >= credit_pct * self.spread_tp_pct:
            return True, "take_profit", net_pnl

        # Stop loss: at 2x credit
        if net_pnl <= -credit_pct * self.spread_sl_pct:
            return True, "stop_loss", net_pnl

        # Time exit: after max hold days
        if days_held >= self.spread_max_hold_days:
            return True, "time_exit", net_pnl

        return False, "", net_pnl

    # ── Prediction generation ────────────────────────────────────

    def _generate_predictions(self, test_probs):
        """Generate signals from probabilities."""
        signals = []
        for prob in test_probs:
            # Simple logic: prob > 0.5 = bullish, < 0.5 = bearish
            direction = "UP" if prob > 0.5 else "DOWN"
            conviction = abs(prob - 0.5) * 2  # 0 to 1 scale

            # Confidence: how far from 0.5?
            confidence = min(conviction * 0.5, 1.0)

            if confidence >= 0.15:
                signal = "BUY_CALL" if direction == "UP" else "BUY_PUT"
            else:
                signal = "HOLD"

            signals.append({
                "signal": signal,
                "probability": float(prob),
                "confidence": float(confidence),
                "direction": direction,
            })

        return signals

    # ── Metrics calculation ──────────────────────────────────────

    def _calculate_fold_metrics(self, trades, portfolio_history, mode):
        """Calculate performance metrics for a single fold."""
        if not trades:
            return {
                "error": "No trades executed",
                "total_trades": 0,
                "final_portfolio_value": self.starting_capital,
                "total_return": 0,
                "win_rate": 0,
                "direction_accuracy": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "expectancy": 0,
                "mode": mode,
            }

        pnls = [t["pnl_dollar"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]

        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(winning_trades) / len(trades) if trades else 0

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = (
            abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float("inf")
        )

        expectancy = total_pnl / len(trades) if trades else 0

        # Portfolio metrics
        if portfolio_history:
            values = [
                p.get("total_value", p.get("capital", self.starting_capital))
                for p in portfolio_history
            ]
            values_series = pd.Series(values)

            daily_returns = values_series.pct_change().dropna()

            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0

            cummax = values_series.cummax()
            drawdown = (values_series - cummax) / cummax
            max_drawdown = drawdown.min()

            final_value = values[-1]
            total_return = (final_value - self.starting_capital) / self.starting_capital

            if len(values_series) > 0 and max_drawdown != 0:
                annual_return = total_return * (252 / max(len(values_series), 1))
                calmar = abs(annual_return / max_drawdown)
            else:
                calmar = 0
        else:
            sharpe = 0
            max_drawdown = 0
            final_value = self.starting_capital
            total_return = 0
            calmar = 0

        # Direction accuracy
        correct_direction = 0
        for t in trades:
            price_move = t.get("exit_price", 0) - t.get("entry_price", 0)
            if t.get("direction") == "long" and price_move > 0:
                correct_direction += 1
            elif t.get("direction") == "short" and price_move < 0:
                correct_direction += 1
        direction_accuracy = correct_direction / len(trades) if trades else 0

        exit_reasons = {}
        for t in trades:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        metrics = {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "direction_accuracy": direction_accuracy,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": np.mean(pnls),
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "total_return": total_return,
            "final_portfolio_value": final_value,
            "exit_reasons": exit_reasons,
            "mode": mode,
        }

        return metrics

    def _compute_aggregate_metrics(self):
        """Compute aggregate metrics across all folds."""
        if not self.fold_results:
            return {}

        # Group by strategy mode
        results_by_mode = defaultdict(list)
        for fold_result in self.fold_results:
            mode = fold_result["mode"]
            results_by_mode[mode].append(fold_result)

        aggregate = {}

        for mode, results_list in results_by_mode.items():
            all_trades = []
            all_pnls = []
            all_returns = []
            all_sharpes = []
            all_accuracies = []
            all_win_rates = []

            for fold_result in results_list:
                metrics = fold_result["metrics"]
                all_trades.extend(fold_result["trades"])

                if metrics.get("total_trades", 0) > 0:
                    all_pnls.append(metrics["total_pnl"])
                    all_returns.append(metrics["total_return"])
                    all_sharpes.append(metrics["sharpe_ratio"])
                    all_accuracies.append(metrics["direction_accuracy"])
                    all_win_rates.append(metrics["win_rate"])

            # Aggregate across folds
            n_folds = len(results_list)
            total_trades = len(all_trades)

            if all_trades:
                winning = sum(1 for t in all_trades if t["pnl_dollar"] > 0)
                win_rate_agg = winning / total_trades
                all_pnls_trades = [t["pnl_dollar"] for t in all_trades]
                avg_pnl = np.mean(all_pnls_trades)
                total_pnl_agg = sum(all_pnls_trades)
            else:
                win_rate_agg = 0
                avg_pnl = 0
                total_pnl_agg = 0

            aggregate[mode] = {
                "n_folds": n_folds,
                "total_trades": total_trades,
                "avg_trades_per_fold": total_trades / n_folds if n_folds > 0 else 0,
                "total_pnl": total_pnl_agg,
                "avg_pnl_per_trade": avg_pnl,
                "win_rate": win_rate_agg,
                "avg_win_rate_per_fold": np.mean(all_win_rates) if all_win_rates else 0,
                "avg_direction_accuracy": np.mean(all_accuracies) if all_accuracies else 0,
                "avg_sharpe": np.mean(all_sharpes) if all_sharpes else 0,
                "avg_return_per_fold": np.mean(all_returns) if all_returns else 0,
                "total_return_agg": sum(all_returns) if all_returns else 0,
            }

        return aggregate

    # ── Output methods ───────────────────────────────────────────

    def get_fold_results(self):
        """Return per-fold metrics."""
        return self.fold_results

    def get_aggregate_metrics(self):
        """Return aggregate metrics across all folds."""
        return self._compute_aggregate_metrics()

    def get_equity_curve(self, mode=None):
        """Return combined equity curve from all out-of-sample periods."""
        if not self.combined_portfolio_history:
            return pd.DataFrame()

        df_hist = pd.DataFrame(self.combined_portfolio_history)

        if mode:
            df_hist = df_hist[df_hist["mode"] == mode]

        df_hist = df_hist.sort_values("date").reset_index(drop=True)
        return df_hist

    def plot_summary(self, filename=None):
        """Generate text-based summary and save to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.results_dir / f"walkforward_summary_{timestamp}.txt"
        else:
            filename = Path(filename)

        with open(filename, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("WALK-FORWARD BACKTEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Configuration:\n")
            f.write(f"  Min training window: {self.min_train_days} trading days\n")
            f.write(f"  Test window: {self.test_days} trading days\n")
            f.write(f"  Step between folds: {self.step_days} trading days\n")
            f.write(f"  Strategy mode: {self.strategy_mode}\n")
            f.write(f"  Starting capital per fold: ${self.starting_capital:,.2f}\n")
            f.write(f"  Folds completed: {len(self.fold_results)}\n\n")

            aggregate_metrics = self.get_aggregate_metrics()

            for mode, metrics in aggregate_metrics.items():
                f.write(f"\n{mode.upper()} MODE\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Folds: {metrics['n_folds']}\n")
                f.write(f"  Total trades (out-of-sample): {metrics['total_trades']}\n")
                f.write(f"  Avg trades per fold: {metrics['avg_trades_per_fold']:.1f}\n")
                f.write(f"  Total P&L: ${metrics['total_pnl']:+,.2f}\n")
                f.write(f"  Avg P&L per trade: ${metrics['avg_pnl_per_trade']:+,.2f}\n")
                f.write(f"  Win rate: {metrics['win_rate']*100:.1f}%\n")
                f.write(f"  Avg win rate per fold: {metrics['avg_win_rate_per_fold']*100:.1f}%\n")
                f.write(f"  Avg direction accuracy: {metrics['avg_direction_accuracy']*100:.1f}%\n")
                f.write(f"  Avg Sharpe ratio: {metrics['avg_sharpe']:.2f}\n")
                f.write(f"  Avg return per fold: {metrics['avg_return_per_fold']*100:+.2f}%\n")
                f.write(f"  Aggregate return: {metrics['total_return_agg']*100:+.2f}%\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("FOLD BREAKDOWN\n")
            f.write("=" * 80 + "\n\n")

            for fold_result in self.fold_results:
                fold = fold_result["fold"]
                mode = fold_result["mode"]
                metrics = fold_result["metrics"]

                f.write(f"Fold {fold} ({mode}):\n")
                if "error" in metrics:
                    f.write(f"  ERROR: {metrics['error']}\n")
                else:
                    f.write(f"  Trades: {metrics['total_trades']}\n")
                    f.write(f"  Win rate: {metrics['win_rate']*100:.1f}%\n")
                    f.write(f"  Return: {metrics['total_return']*100:+.2f}%\n")
                    f.write(f"  Sharpe: {metrics['sharpe_ratio']:.2f}\n")
                    f.write(f"  Max DD: {metrics['max_drawdown']*100:.1f}%\n")
                    f.write(f"  Direction accuracy: {metrics['direction_accuracy']*100:.1f}%\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write(f"Report saved at {datetime.now().isoformat()}\n")

        logger.info(f"Summary saved to {filename}")
        return filename

    def _log_summary(self, results):
        """Log a formatted summary of results."""
        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD BACKTEST COMPLETED")
        logger.info("=" * 80)

        aggregate = results.get("aggregate_metrics", {})

        for mode, metrics in aggregate.items():
            logger.info(f"\n{mode.upper()} MODE:")
            logger.info(f"  Folds: {metrics['n_folds']}")
            logger.info(f"  Total trades: {metrics['total_trades']}")
            logger.info(f"  Win rate: {metrics['win_rate']*100:.1f}%")
            logger.info(f"  Avg direction accuracy: {metrics['avg_direction_accuracy']*100:.1f}%")
            logger.info(f"  Total P&L: ${metrics['total_pnl']:+,.2f}")
            logger.info(f"  Avg Sharpe: {metrics['avg_sharpe']:.2f}")

        logger.info("=" * 80)

    def compare_strategies(self):
        """Compare directional vs credit spread performance (if both were run)."""
        aggregate = self.get_aggregate_metrics()

        if "directional" not in aggregate or "credit_spread" not in aggregate:
            return None

        dir_m = aggregate["directional"]
        cs_m = aggregate["credit_spread"]

        comparison = {
            "metric": [
                "Win rate",
                "Total P&L",
                "Avg P&L/trade",
                "Direction accuracy",
                "Avg Sharpe",
                "Trades executed",
            ],
            "directional": [
                f"{dir_m['win_rate']*100:.1f}%",
                f"${dir_m['total_pnl']:+,.2f}",
                f"${dir_m['avg_pnl_per_trade']:+,.2f}",
                f"{dir_m['avg_direction_accuracy']*100:.1f}%",
                f"{dir_m['avg_sharpe']:.2f}",
                f"{dir_m['total_trades']}",
            ],
            "credit_spread": [
                f"{cs_m['win_rate']*100:.1f}%",
                f"${cs_m['total_pnl']:+,.2f}",
                f"${cs_m['avg_pnl_per_trade']:+,.2f}",
                f"{cs_m['avg_direction_accuracy']*100:.1f}%",
                f"{cs_m['avg_sharpe']:.2f}",
                f"{cs_m['total_trades']}",
            ],
        }

        logger.info("\n" + "=" * 80)
        logger.info("STRATEGY COMPARISON")
        logger.info("=" * 80)

        for i, metric in enumerate(comparison["metric"]):
            dir_val = comparison["directional"][i]
            cs_val = comparison["credit_spread"][i]
            logger.info(f"{metric:30s} | {dir_val:20s} | {cs_val:20s}")

        logger.info("=" * 80)

        return comparison
