"""
Monte Carlo Simulator — stress-tests the trading strategy under uncertainty.

Runs thousands of randomized simulations to answer:
  - What's the probability of making money?
  - What's the worst-case scenario (5th percentile)?
  - What's the expected range of outcomes?
  - How robust is the strategy to random variation?

Three simulation methods:
  1. TRADE SHUFFLE — randomly reorders actual trades to test path dependence
  2. BOOTSTRAP — resamples trades with replacement to simulate alternative histories
  3. NOISE INJECTION — adds random noise to predictions to test robustness

Usage:
    from src.backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(n_simulations=1000)
    mc_results = mc.run(backtest_results)
    mc.print_report(mc_results)
"""
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import DATA_DIR


class MonteCarloSimulator:
    """Monte Carlo simulation engine for trading strategy analysis."""

    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        """
        Args:
            n_simulations: Number of simulation runs
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self.results_dir = DATA_DIR / "backtest"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MonteCarloSimulator initialized ({n_simulations} simulations)")

    def run(
        self,
        backtest_results: dict,
        starting_capital: float = 1000.0,
        method: str = "all",
    ) -> dict:
        """
        Run Monte Carlo simulations on backtest results.

        Args:
            backtest_results: Output from Backtester.run()
            starting_capital: Starting portfolio value
            method: "shuffle", "bootstrap", "noise", or "all"

        Returns:
            Dict with simulation results, statistics, and confidence intervals
        """
        trades = backtest_results.get("trades", [])
        if len(trades) < 3:
            logger.warning(f"Only {len(trades)} trades — need at least 3 for Monte Carlo")
            return {"error": "Insufficient trades for simulation"}

        np.random.seed(self.seed)

        all_results = {}

        if method in ("shuffle", "all"):
            logger.info("Running TRADE SHUFFLE simulations...")
            all_results["shuffle"] = self._run_shuffle(trades, starting_capital)

        if method in ("bootstrap", "all"):
            logger.info("Running BOOTSTRAP simulations...")
            all_results["bootstrap"] = self._run_bootstrap(trades, starting_capital)

        if method in ("noise", "all"):
            logger.info("Running NOISE INJECTION simulations...")
            predictions = backtest_results.get("predictions", [])
            price_history = backtest_results.get("portfolio_history", [])
            all_results["noise"] = self._run_noise(trades, starting_capital)

        # Combine all simulation outcomes for overall statistics
        all_final_values = []
        all_returns = []
        for method_name, method_results in all_results.items():
            if "final_values" in method_results:
                all_final_values.extend(method_results["final_values"])
                all_returns.extend(method_results["returns"])

        combined_stats = self._compute_statistics(
            all_final_values, all_returns, starting_capital, "Combined"
        )

        return {
            "methods": all_results,
            "combined": combined_stats,
            "n_simulations_total": len(all_final_values),
            "starting_capital": starting_capital,
            "n_actual_trades": len(trades),
        }

    def _run_shuffle(self, trades: list, starting_capital: float) -> dict:
        """
        Shuffle the order of trades to test path dependence.
        Same trades, different sequence → shows how timing affects results.
        """
        pnl_pcts = [t["pnl_pct"] for t in trades]
        position_sizes_pct = [t.get("position_size", starting_capital * 0.03) / starting_capital for t in trades]

        final_values = []
        max_drawdowns = []
        all_equity_curves = []

        for sim in range(self.n_simulations):
            # Randomly shuffle the P&L sequence
            indices = np.random.permutation(len(pnl_pcts))
            shuffled_pnls = [pnl_pcts[i] for i in indices]
            shuffled_sizes = [position_sizes_pct[i] for i in indices]

            # Simulate equity curve
            capital = starting_capital
            peak = capital
            max_dd = 0
            equity_curve = [capital]

            for pnl_pct, size_pct in zip(shuffled_pnls, shuffled_sizes):
                position_size = capital * size_pct
                pnl_dollar = position_size * pnl_pct
                capital += pnl_dollar
                capital = max(capital, 0)  # Can't go below 0
                equity_curve.append(capital)

                if capital > peak:
                    peak = capital
                dd = (capital - peak) / peak if peak > 0 else 0
                max_dd = min(max_dd, dd)

            final_values.append(capital)
            max_drawdowns.append(max_dd)
            if sim < 100:  # Store first 100 curves for visualization
                all_equity_curves.append(equity_curve)

        returns = [(v - starting_capital) / starting_capital for v in final_values]

        stats = self._compute_statistics(final_values, returns, starting_capital, "Shuffle")
        stats["max_drawdowns"] = max_drawdowns
        stats["equity_curves_sample"] = all_equity_curves
        stats["final_values"] = final_values
        stats["returns"] = returns

        return stats

    def _run_bootstrap(self, trades: list, starting_capital: float) -> dict:
        """
        Bootstrap resampling: randomly sample trades WITH replacement.
        Creates alternative trading histories that could have happened.
        """
        pnl_pcts = [t["pnl_pct"] for t in trades]
        n_trades = len(trades)

        final_values = []
        max_drawdowns = []
        all_equity_curves = []

        for sim in range(self.n_simulations):
            # Sample n_trades with replacement from actual trades
            sampled_indices = np.random.choice(n_trades, size=n_trades, replace=True)
            sampled_pnls = [pnl_pcts[i] for i in sampled_indices]

            capital = starting_capital
            peak = capital
            max_dd = 0
            equity_curve = [capital]

            for pnl_pct in sampled_pnls:
                position_size = capital * 0.03  # Use standard position sizing
                pnl_dollar = position_size * pnl_pct
                capital += pnl_dollar
                capital = max(capital, 0)
                equity_curve.append(capital)

                if capital > peak:
                    peak = capital
                dd = (capital - peak) / peak if peak > 0 else 0
                max_dd = min(max_dd, dd)

            final_values.append(capital)
            max_drawdowns.append(max_dd)
            if sim < 100:
                all_equity_curves.append(equity_curve)

        returns = [(v - starting_capital) / starting_capital for v in final_values]

        stats = self._compute_statistics(final_values, returns, starting_capital, "Bootstrap")
        stats["max_drawdowns"] = max_drawdowns
        stats["equity_curves_sample"] = all_equity_curves
        stats["final_values"] = final_values
        stats["returns"] = returns

        return stats

    def _run_noise(self, trades: list, starting_capital: float) -> dict:
        """
        Noise injection: add random noise to trade P&L to test robustness.
        Simulates model prediction uncertainty and market randomness.
        """
        pnl_pcts = [t["pnl_pct"] for t in trades]
        n_trades = len(trades)

        # Estimate noise level from actual trade variance
        pnl_std = np.std(pnl_pcts) if len(pnl_pcts) > 1 else 0.01
        noise_levels = [0.25, 0.50, 0.75, 1.0]  # Fraction of observed std

        final_values = []
        max_drawdowns = []
        all_equity_curves = []

        sims_per_level = self.n_simulations // len(noise_levels)

        for noise_frac in noise_levels:
            noise_std = pnl_std * noise_frac

            for sim in range(sims_per_level):
                # Add Gaussian noise to each trade's P&L
                noisy_pnls = [p + np.random.normal(0, noise_std) for p in pnl_pcts]

                capital = starting_capital
                peak = capital
                max_dd = 0
                equity_curve = [capital]

                for pnl_pct in noisy_pnls:
                    position_size = capital * 0.03
                    pnl_dollar = position_size * pnl_pct
                    capital += pnl_dollar
                    capital = max(capital, 0)
                    equity_curve.append(capital)

                    if capital > peak:
                        peak = capital
                    dd = (capital - peak) / peak if peak > 0 else 0
                    max_dd = min(max_dd, dd)

                final_values.append(capital)
                max_drawdowns.append(max_dd)
                if len(all_equity_curves) < 100:
                    all_equity_curves.append(equity_curve)

        returns = [(v - starting_capital) / starting_capital for v in final_values]

        stats = self._compute_statistics(final_values, returns, starting_capital, "Noise")
        stats["max_drawdowns"] = max_drawdowns
        stats["equity_curves_sample"] = all_equity_curves
        stats["final_values"] = final_values
        stats["returns"] = returns

        return stats

    def _compute_statistics(
        self, final_values: list, returns: list,
        starting_capital: float, method_name: str
    ) -> dict:
        """Compute comprehensive statistics from simulation results."""
        if not final_values:
            return {"error": "No simulation results"}

        fv = np.array(final_values)
        ret = np.array(returns)

        stats = {
            "method": method_name,
            "n_simulations": len(fv),
            "starting_capital": starting_capital,

            # Central tendency
            "mean_final_value": float(np.mean(fv)),
            "median_final_value": float(np.median(fv)),
            "mean_return": float(np.mean(ret)),
            "median_return": float(np.median(ret)),

            # Risk metrics
            "std_return": float(np.std(ret)),
            "min_final_value": float(np.min(fv)),
            "max_final_value": float(np.max(fv)),

            # Confidence intervals
            "ci_5": float(np.percentile(fv, 5)),      # Worst 5%
            "ci_10": float(np.percentile(fv, 10)),     # Worst 10%
            "ci_25": float(np.percentile(fv, 25)),     # Lower quartile
            "ci_50": float(np.percentile(fv, 50)),     # Median
            "ci_75": float(np.percentile(fv, 75)),     # Upper quartile
            "ci_90": float(np.percentile(fv, 90)),     # Best 10%
            "ci_95": float(np.percentile(fv, 95)),     # Best 5%

            "return_ci_5": float(np.percentile(ret, 5)),
            "return_ci_25": float(np.percentile(ret, 25)),
            "return_ci_75": float(np.percentile(ret, 75)),
            "return_ci_95": float(np.percentile(ret, 95)),

            # Probability metrics
            "prob_profit": float(np.mean(ret > 0)),
            "prob_loss_5pct": float(np.mean(ret < -0.05)),
            "prob_loss_10pct": float(np.mean(ret < -0.10)),
            "prob_gain_5pct": float(np.mean(ret > 0.05)),
            "prob_gain_10pct": float(np.mean(ret > 0.10)),

            # Value at Risk (VaR)
            "var_95": float(starting_capital - np.percentile(fv, 5)),
            "var_99": float(starting_capital - np.percentile(fv, 1)),
            "cvar_95": float(starting_capital - np.mean(fv[fv <= np.percentile(fv, 5)])) if len(fv[fv <= np.percentile(fv, 5)]) > 0 else 0,
        }

        return stats

    def print_report(self, mc_results: dict):
        """Print a formatted Monte Carlo report."""
        if "error" in mc_results:
            print(f"\n  Monte Carlo Error: {mc_results['error']}")
            return

        combined = mc_results.get("combined", {})
        capital = mc_results.get("starting_capital", 1000)

        print(f"\n{'='*70}")
        print(f"  MONTE CARLO SIMULATION REPORT")
        print(f"  {mc_results['n_simulations_total']:,} simulations | "
              f"{mc_results['n_actual_trades']} actual trades | "
              f"${capital:,.0f} starting capital")
        print(f"{'='*70}")

        print(f"\n  OUTCOME DISTRIBUTION:")
        print(f"  {'Metric':<30} {'Value':>15}")
        print(f"  {'-'*45}")
        print(f"  {'Mean Final Value':<30} ${combined['mean_final_value']:>14,.2f}")
        print(f"  {'Median Final Value':<30} ${combined['median_final_value']:>14,.2f}")
        print(f"  {'Mean Return':<30} {combined['mean_return']*100:>14.2f}%")
        print(f"  {'Std Dev (Return)':<30} {combined['std_return']*100:>14.2f}%")

        print(f"\n  CONFIDENCE INTERVALS (Final Value):")
        print(f"  {'Percentile':<30} {'Value':>15}")
        print(f"  {'-'*45}")
        print(f"  {'Worst Case (5th pct)':<30} ${combined['ci_5']:>14,.2f}")
        print(f"  {'Pessimistic (10th pct)':<30} ${combined['ci_10']:>14,.2f}")
        print(f"  {'Lower Quartile (25th)':<30} ${combined['ci_25']:>14,.2f}")
        print(f"  {'Median (50th)':<30} ${combined['ci_50']:>14,.2f}")
        print(f"  {'Upper Quartile (75th)':<30} ${combined['ci_75']:>14,.2f}")
        print(f"  {'Optimistic (90th pct)':<30} ${combined['ci_90']:>14,.2f}")
        print(f"  {'Best Case (95th pct)':<30} ${combined['ci_95']:>14,.2f}")

        print(f"\n  PROBABILITIES:")
        print(f"  {'Outcome':<30} {'Probability':>15}")
        print(f"  {'-'*45}")
        print(f"  {'Profit (any)':<30} {combined['prob_profit']*100:>14.1f}%")
        print(f"  {'Gain > 5%':<30} {combined['prob_gain_5pct']*100:>14.1f}%")
        print(f"  {'Gain > 10%':<30} {combined['prob_gain_10pct']*100:>14.1f}%")
        print(f"  {'Loss > 5%':<30} {combined['prob_loss_5pct']*100:>14.1f}%")
        print(f"  {'Loss > 10%':<30} {combined['prob_loss_10pct']*100:>14.1f}%")

        print(f"\n  RISK METRICS:")
        print(f"  {'Metric':<30} {'Value':>15}")
        print(f"  {'-'*45}")
        print(f"  {'Value at Risk (95%)':<30} ${combined['var_95']:>14,.2f}")
        print(f"  {'Value at Risk (99%)':<30} ${combined['var_99']:>14,.2f}")
        print(f"  {'CVaR / Expected Shortfall':<30} ${combined['cvar_95']:>14,.2f}")

        # Per-method summary
        for method_name, method_data in mc_results.get("methods", {}).items():
            if "prob_profit" in method_data:
                print(f"\n  {method_name.upper()}: "
                      f"P(profit)={method_data['prob_profit']*100:.1f}%, "
                      f"median=${method_data['median_final_value']:,.2f}, "
                      f"5th pct=${method_data['ci_5']:,.2f}")

        print(f"\n{'='*70}")

    def save_results(self, mc_results: dict, symbol: str = "SPY"):
        """Save Monte Carlo results to file."""
        filepath = self.results_dir / f"montecarlo_{symbol}.txt"
        with open(filepath, "w") as f:
            f.write(f"Monte Carlo Simulation Results for {symbol}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Simulations: {mc_results.get('n_simulations_total', 0)}\n")
            f.write(f"Actual trades: {mc_results.get('n_actual_trades', 0)}\n")
            f.write(f"{'='*60}\n\n")

            combined = mc_results.get("combined", {})
            for key, value in combined.items():
                if not isinstance(value, (list, dict)):
                    f.write(f"{key}: {value}\n")

        # Save distribution data for dashboard visualization
        combined = mc_results.get("combined", {})
        all_fv = []
        for method_data in mc_results.get("methods", {}).values():
            if "final_values" in method_data:
                all_fv.extend(method_data["final_values"])

        if all_fv:
            dist_df = pd.DataFrame({"final_value": all_fv})
            dist_path = self.results_dir / f"montecarlo_dist_{symbol}.csv"
            dist_df.to_csv(dist_path, index=False)

        logger.info(f"Monte Carlo results saved to {filepath}")
