"""
Backtester — simulates the trading strategy on historical data.

Two modes:
  1. DIRECTIONAL (default): Trades the underlying stock based on signals.
     This honestly measures whether the ML model's directional predictions
     are correct. No options complexity — pure signal accuracy evaluation.
     P&L = position_size × price_change (long for BUY_CALL, short for BUY_PUT).

  2. OPTIONS: Simulates buying calls/puts with realistic non-linear theta decay.
     Theta starts low (~0.5%/day early) and accelerates near expiration (~5%/day).
     Use this mode once directional accuracy is confirmed.

Tracks:
  - Each trade (entry, exit, P&L)
  - Portfolio value over time
  - Key performance metrics (Sharpe, drawdown, win rate)

Usage:
    from src.backtest.backtester import Backtester
    bt = Backtester(starting_capital=1000, mode="directional")
    results = bt.run(predictions, price_data)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, DATA_DIR


class Backtester:
    """Simulates trading strategy on historical predictions."""

    VALID_MODES = ("directional", "options")

    def __init__(self, starting_capital: float = 1000.0, mode: str = "directional"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        settings = get_settings()
        trading = settings.get("trading", {})

        self.mode = mode
        self.starting_capital = starting_capital
        self.max_risk_per_trade = trading.get("max_portfolio_risk_per_trade", 0.03)
        self.daily_loss_limit = trading.get("daily_loss_limit", 0.05)
        self.max_open_positions = trading.get("max_open_positions", 5)

        # Directional mode settings
        self.directional_stop_loss = 0.03   # 3% adverse move on the stock → cut loss
        self.directional_take_profit = 0.04  # 4% favorable move → take profit
        self.max_hold_days = 5               # Close after 5 days max (short-term trading)

        # Options mode settings (only used in options mode)
        options = trading.get("options", {})
        self.stop_loss_pct = options.get("stop_loss_pct", 0.35)
        self.take_profit_pct = options.get("take_profit_pct", 0.50)
        self.default_dte = options.get("default_dte_range", [21, 35])  # 3-5 week options = sweet spot
        self.options_min_confidence = 0.25  # Slightly above directional (theta is a headwind)
        self.options_max_hold_pct = 0.50   # Close at 50% of DTE regardless (avoid last-week theta crush)

        self.results_dir = DATA_DIR / "backtest"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Backtester initialized — mode={mode}, capital=${starting_capital}, "
                     f"risk/trade={self.max_risk_per_trade*100}%, "
                     f"max_positions={self.max_open_positions}")

    # ── Main loop ────────────────────────────────────────────────

    def run(
        self,
        predictions: list[dict],
        price_df: pd.DataFrame,
        dates: pd.DatetimeIndex = None,
    ) -> dict:
        """
        Run backtest on historical predictions.

        Args:
            predictions: List of ensemble prediction dicts (signal, probability, confidence)
            price_df: DataFrame with at least 'close' column, datetime-indexed
            dates: Corresponding dates for each prediction

        Returns:
            Dict with trades, portfolio history, and performance metrics
        """
        if dates is None:
            dates = price_df.index[-len(predictions):]

        capital = self.starting_capital
        portfolio_history = []
        trades = []
        open_positions = []
        daily_pnl = 0.0
        daily_start_capital = capital

        logger.info(f"Running backtest ({self.mode} mode): {len(predictions)} predictions, "
                     f"${capital} starting capital")

        for i, (pred, date) in enumerate(zip(predictions, dates)):
            # Reset daily tracking at start of new day
            if i > 0 and dates[i].date() != dates[i - 1].date():
                daily_pnl = 0.0
                daily_start_capital = capital

            current_price = price_df.loc[date, "close"] if date in price_df.index else None
            if current_price is None:
                portfolio_history.append({
                    "date": date, "capital": capital,
                    "open_positions": len(open_positions),
                })
                continue

            # --- Check and close existing positions ---
            positions_to_close = []
            for pos_idx, pos in enumerate(open_positions):
                should_close, exit_reason, pnl_pct = self._evaluate_position(
                    pos, current_price, date
                )
                if should_close:
                    pos["exit_date"] = date
                    pos["exit_price"] = current_price
                    pos["exit_reason"] = exit_reason
                    pos["pnl_pct"] = pnl_pct
                    pos["pnl_dollar"] = pos["position_size"] * pnl_pct
                    positions_to_close.append(pos_idx)

            # Close positions (reverse order to preserve indices)
            for pos_idx in sorted(positions_to_close, reverse=True):
                closed_pos = open_positions.pop(pos_idx)
                capital += closed_pos["position_size"] + closed_pos["pnl_dollar"]
                daily_pnl += closed_pos["pnl_dollar"]
                trades.append(closed_pos)

            # --- Check daily loss limit ---
            if daily_start_capital > 0:
                daily_loss_pct = -daily_pnl / daily_start_capital
                if daily_loss_pct >= self.daily_loss_limit:
                    portfolio_history.append({
                        "date": date, "capital": capital,
                        "open_positions": len(open_positions),
                        "daily_loss_limit_hit": True,
                    })
                    continue

            # --- Open new position if signal is actionable ---
            signal = pred.get("signal", "HOLD")
            confidence = pred.get("confidence", 0)
            probability = pred.get("probability", 0.5)

            # Options mode requires slightly higher confidence (theta is a headwind)
            if self.mode == "options":
                min_trade_confidence = self.options_min_confidence  # 0.25
            else:
                min_trade_confidence = 0.15  # Directional: match ensemble threshold

            can_open = (
                signal != "HOLD"
                and confidence >= min_trade_confidence
                and len(open_positions) < self.max_open_positions
            )

            if can_open:
                position = self._open_position(
                    signal, confidence, probability, current_price, capital, date
                )
                if position is not None:
                    open_positions.append(position)
                    capital -= position["position_size"]

            # Record portfolio state
            open_value = self._estimate_open_value(open_positions, current_price, date)

            portfolio_history.append({
                "date": date,
                "capital": capital,
                "open_value": max(0, open_value),
                "total_value": capital + max(0, open_value),
                "open_positions": len(open_positions),
            })

        # Close any remaining open positions at last price
        last_price = price_df["close"].iloc[-1]
        for pos in open_positions:
            _, _, pnl_pct = self._evaluate_position(pos, last_price, dates[-1], force_close=True)
            pos["exit_date"] = dates[-1]
            pos["exit_price"] = last_price
            pos["exit_reason"] = "backtest_end"
            pos["pnl_pct"] = pnl_pct
            pos["pnl_dollar"] = pos["position_size"] * pnl_pct
            capital += pos["position_size"] + pos["pnl_dollar"]
            trades.append(pos)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, portfolio_history)

        results = {
            "trades": trades,
            "portfolio_history": portfolio_history,
            "metrics": metrics,
            "config": {
                "starting_capital": self.starting_capital,
                "mode": self.mode,
                "max_risk_per_trade": self.max_risk_per_trade,
                "max_open_positions": self.max_open_positions,
                "directional_stop_loss": self.directional_stop_loss if self.mode == "directional" else None,
                "directional_take_profit": self.directional_take_profit if self.mode == "directional" else None,
                "options_stop_loss": self.stop_loss_pct if self.mode == "options" else None,
                "options_take_profit": self.take_profit_pct if self.mode == "options" else None,
            },
        }

        self._log_summary(results)
        return results

    # ── Position management ──────────────────────────────────────

    def _open_position(self, signal, confidence, probability, current_price, capital, date):
        """Create a new position based on the signal and mode."""
        # Position sizing: risk a fixed % of current capital per trade
        # Scale by confidence — higher confidence = larger bet (within limits)
        confidence_scale = 0.5 + confidence  # range ~0.7 to 1.5
        risk_amount = capital * self.max_risk_per_trade * min(confidence_scale, 1.5)
        position_size = risk_amount

        # Don't trade if capital is too low
        if position_size < 10 or capital < 50:
            return None

        direction = "long" if signal == "BUY_CALL" else "short"

        position = {
            "entry_date": date,
            "symbol": "SPY",
            "direction": direction,
            "signal": signal,
            "entry_price": current_price,
            "position_size": position_size,
            "confidence": confidence,
            "probability": probability,
            "mode": self.mode,
        }

        if self.mode == "options":
            position["option_type"] = "call" if signal == "BUY_CALL" else "put"
            position["dte"] = np.mean(self.default_dte)

        return position

    def _evaluate_position(self, pos, current_price, date, force_close=False):
        """
        Evaluate if a position should be closed and calculate its P&L.

        Returns: (should_close: bool, exit_reason: str, pnl_pct: float)
        """
        if self.mode == "directional":
            return self._evaluate_directional(pos, current_price, date, force_close)
        else:
            return self._evaluate_options(pos, current_price, date, force_close)

    def _evaluate_directional(self, pos, current_price, date, force_close=False):
        """
        Directional mode: simple stock-based P&L.
        Long position profits when price goes up, short profits when price goes down.
        """
        price_change_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        days_held = max(1, (date - pos["entry_date"]).days)

        # P&L depends on direction
        if pos["direction"] == "long":
            pnl_pct = price_change_pct
        else:
            pnl_pct = -price_change_pct

        if force_close:
            return True, "backtest_end", pnl_pct

        # Check stop loss (stock moves against us by X%)
        if pnl_pct <= -self.directional_stop_loss:
            return True, "stop_loss", -self.directional_stop_loss

        # Check take profit (stock moves in our favor by X%)
        if pnl_pct >= self.directional_take_profit:
            return True, "take_profit", self.directional_take_profit

        # Check max hold duration
        if days_held >= self.max_hold_days:
            return True, "max_hold", pnl_pct

        return False, "", pnl_pct

    def _evaluate_options(self, pos, current_price, date, force_close=False):
        """
        Options mode: realistic non-linear theta decay model.

        Theta decay follows an exponential curve:
        - Early in the option's life (>7 DTE): ~0.5-1% per day
        - Mid-life (3-7 DTE): ~1.5-2.5% per day
        - Near expiration (<3 DTE): ~3-5% per day

        This uses a sqrt-based approximation of the Black-Scholes theta curve:
        Cumulative theta ≈ total_theta_at_expiry × (1 - sqrt(time_remaining / total_time))
        """
        price_change_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
        dte = pos["dte"]
        days_held = max(0.5, (date - pos["entry_date"]).days)  # min half day

        # Non-linear theta: sqrt decay model
        # For 28-day ATM options, theta is much gentler in first half
        # Key insight: close before the "theta cliff" (last 40% of DTE)
        total_theta_budget = 0.30  # 30% premium loss if held to expiry (28 DTE, ATM)
        time_fraction = min(days_held / dte, 1.0)
        cumulative_theta = total_theta_budget * (1.0 - np.sqrt(max(0, 1.0 - time_fraction)))

        # Delta: ATM ~0.50, but effective leverage depends on moneyness
        # As the option goes ITM, delta approaches 1.0 → more leverage
        # As OTM, delta drops → less leverage
        base_delta = 2.0
        # If the stock has moved >1% in our favor, delta increases (going ITM)
        if pos["option_type"] == "call":
            directional_move = price_change_pct
        else:
            directional_move = -price_change_pct

        if directional_move > 0.01:
            # Going ITM → delta approaches 1.0 → leverage increases
            delta_leverage = base_delta + min(directional_move * 5, 1.0)
        elif directional_move < -0.01:
            # Going OTM → delta drops → leverage decreases
            delta_leverage = max(base_delta + directional_move * 3, 0.5)
        else:
            delta_leverage = base_delta

        # Vega: large moves increase IV → benefits option holders
        abs_move = abs(price_change_pct)
        vega_bonus = max(0, (abs_move - 0.01)) * 0.4

        # Gamma scalping benefit: if the stock swings back and forth, ATM options benefit
        # Simplified: give a small bonus proportional to move magnitude (regardless of direction)
        gamma_benefit = max(0, abs_move - 0.005) * 0.1

        if pos["option_type"] == "call":
            raw_pnl = price_change_pct * delta_leverage + vega_bonus + gamma_benefit - cumulative_theta
        else:
            raw_pnl = -price_change_pct * delta_leverage + vega_bonus + gamma_benefit - cumulative_theta

        option_pnl_pct = max(raw_pnl, -1.0)

        if force_close:
            return True, "backtest_end", option_pnl_pct

        # Check stop loss
        if option_pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss", -self.stop_loss_pct

        # Check take profit
        if option_pnl_pct >= self.take_profit_pct:
            return True, "take_profit", self.take_profit_pct

        # CRITICAL: Close BEFORE theta crush zone (at 50% of DTE)
        # e.g., for 28 DTE option, close by day 14 regardless
        max_hold = dte * self.options_max_hold_pct
        if days_held >= max_hold:
            return True, "theta_exit", option_pnl_pct

        # Check expiration (safety net)
        if days_held >= dte:
            return True, "expiration", option_pnl_pct

        return False, "", option_pnl_pct

    def _estimate_open_value(self, open_positions, current_price, date):
        """Estimate the current value of all open positions."""
        total_value = 0
        for pos in open_positions:
            _, _, pnl_pct = self._evaluate_position(pos, current_price, date)
            # Position value = original size + unrealized P&L
            pos_value = pos["position_size"] * max(0, 1 + pnl_pct)
            total_value += pos_value
        return total_value

    # ── Metrics ──────────────────────────────────────────────────

    def _calculate_metrics(self, trades: list, portfolio_history: list) -> dict:
        """Calculate performance metrics from backtest results."""
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
                "mode": self.mode,
            }

        pnls = [t["pnl_dollar"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]

        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(winning_trades) / len(trades) if trades else 0

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float("inf")

        # Expectancy: avg $ per trade
        expectancy = total_pnl / len(trades) if trades else 0

        # Portfolio value series
        if portfolio_history:
            values = [p.get("total_value", p.get("capital", self.starting_capital))
                      for p in portfolio_history]
            values_series = pd.Series(values)

            # Daily returns
            daily_returns = values_series.pct_change().dropna()

            # Sharpe ratio (annualized, assuming 252 trading days)
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0

            # Max drawdown
            cummax = values_series.cummax()
            drawdown = (values_series - cummax) / cummax
            max_drawdown = drawdown.min()

            # Calmar ratio (annualized return / max drawdown)
            final_value = values[-1]
            total_return = (final_value - self.starting_capital) / self.starting_capital

            n_days = len(values_series)
            if n_days > 0 and max_drawdown != 0:
                annual_return = total_return * (252 / max(n_days, 1))
                calmar = abs(annual_return / max_drawdown)
            else:
                calmar = 0
        else:
            sharpe = 0
            max_drawdown = 0
            final_value = self.starting_capital
            total_return = 0
            calmar = 0

        # Trade duration stats
        durations = []
        for t in trades:
            if "entry_date" in t and "exit_date" in t:
                dur = (t["exit_date"] - t["entry_date"]).days
                durations.append(dur)

        # Exit reason distribution
        exit_reasons = {}
        for t in trades:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # Direction accuracy: how often does the model correctly predict up/down?
        correct_direction = 0
        for t in trades:
            price_move = t.get("exit_price", 0) - t.get("entry_price", 0)
            if t.get("direction") == "long" and price_move > 0:
                correct_direction += 1
            elif t.get("direction") == "short" and price_move < 0:
                correct_direction += 1
        direction_accuracy = correct_direction / len(trades) if trades else 0

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
            "avg_trade_duration_days": np.mean(durations) if durations else 0,
            "exit_reasons": exit_reasons,
            "long_trades": sum(1 for t in trades if t.get("direction") == "long"),
            "short_trades": sum(1 for t in trades if t.get("direction") == "short"),
            "mode": self.mode,
        }

        return metrics

    def _log_summary(self, results: dict):
        """Log a formatted summary of backtest results."""
        m = results["metrics"]
        if "error" in m:
            logger.warning(f"Backtest produced no trades: {m['error']}")
            print(f"\n  *** {self.mode.upper()} BACKTEST: 0 TRADES ***")
            print(f"  This means no predictions passed the confidence threshold.")
            if self.mode == "options":
                print(f"  Options min confidence: {self.options_min_confidence}")
            else:
                print(f"  Directional min confidence: 0.20")
            print(f"  Check the PREDICTION DIAGNOSTICS above for details.\n")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTEST RESULTS ({self.mode.upper()} MODE)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Capital:    ${self.starting_capital:,.2f}")
        logger.info(f"Final Value:         ${m['final_portfolio_value']:,.2f}")
        logger.info(f"Total Return:        {m['total_return']*100:+.2f}%")
        logger.info(f"Total P&L:           ${m['total_pnl']:+,.2f}")
        logger.info(f"")
        logger.info(f"Total Trades:        {m['total_trades']}")
        logger.info(f"Win Rate:            {m['win_rate']*100:.1f}%")
        logger.info(f"Direction Accuracy:  {m['direction_accuracy']*100:.1f}%")
        logger.info(f"Expectancy/Trade:    ${m['expectancy']:+,.2f}")
        logger.info(f"Avg Win:             ${m['avg_win']:+,.2f}")
        logger.info(f"Avg Loss:            ${m['avg_loss']:+,.2f}")
        logger.info(f"Profit Factor:       {m['profit_factor']:.2f}")
        logger.info(f"")
        logger.info(f"Sharpe Ratio:        {m['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown:        {m['max_drawdown']*100:.1f}%")
        logger.info(f"Calmar Ratio:        {m['calmar_ratio']:.2f}")
        logger.info(f"Avg Trade Duration:  {m['avg_trade_duration_days']:.1f} days")
        logger.info(f"")
        logger.info(f"Longs: {m['long_trades']} | Shorts: {m['short_trades']}")
        logger.info(f"Exit Reasons: {m['exit_reasons']}")
        logger.info(f"{'='*60}")

    def save_results(self, results: dict, symbol: str = "SPY"):
        """Save backtest results to CSV files."""
        mode_suffix = f"_{self.mode}" if self.mode != "directional" else ""

        # Save trades
        if results["trades"]:
            trades_df = pd.DataFrame(results["trades"])
            trades_path = self.results_dir / f"trades_{symbol}{mode_suffix}.csv"
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Trades saved to {trades_path}")

        # Save portfolio history
        if results["portfolio_history"]:
            hist_df = pd.DataFrame(results["portfolio_history"])
            hist_path = self.results_dir / f"portfolio_{symbol}{mode_suffix}.csv"
            hist_df.to_csv(hist_path, index=False)
            logger.info(f"Portfolio history saved to {hist_path}")

        # Save metrics summary
        metrics_path = self.results_dir / f"metrics_{symbol}{mode_suffix}.txt"
        with open(metrics_path, "w") as f:
            f.write(f"Backtest Results for {symbol} ({self.mode} mode)\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            for key, value in results["metrics"].items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Metrics saved to {metrics_path}")
