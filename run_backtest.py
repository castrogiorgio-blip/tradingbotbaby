"""
Run Backtest — simulates the full trading strategy on historical data.

Runs BOTH modes:
  1. Directional — honest evaluation of model's price direction predictions
  2. Options — realistic options P&L with non-linear theta decay

Usage:
    python3 run_backtest.py                    # Backtest SPY with defaults
    python3 run_backtest.py --symbol AAPL      # Backtest specific symbol
    python3 run_backtest.py --capital 5000      # Start with $5000
    python3 run_backtest.py --days 365          # Use 1 year of data
    python3 run_backtest.py --mode directional  # Only run directional mode
    python3 run_backtest.py --mode options      # Only run options mode
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
from src.models.tft_model import TFTModel
from src.models.tabnet_model import TabNetModel
from src.models.ensemble import EnsembleModel
from src.backtest.backtester import Backtester


def run_backtest(symbol: str = "SPY", days: int = 365 * 5, capital: float = 1000.0,
                 mode: str = "both", test_ratio: float = 0.30):
    """Run full backtest pipeline: build features → generate predictions → simulate trades."""

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTESTING {symbol} — ${capital} capital, {days} days of data, {test_ratio*100:.0f}% test")
    logger.info(f"{'='*60}")

    # Step 1: Build features
    logger.info("\nStep 1: Building features...")
    builder = FeatureBuilder(use_finbert=False)
    df = builder.build_features(symbol, days=days, include_news=False)

    if df.empty or len(df) < 100:
        logger.error(f"Insufficient data for {symbol}")
        return None

    # Step 2: Split data — use more data for testing (30% default = ~375 days on 5yr)
    logger.info("\nStep 2: Splitting data...")
    n = len(df)
    test_start = int(n * (1 - test_ratio))  # Last 30% for testing
    val_start = int(test_start * 0.85)       # 15% of training data for validation

    df_train = df.iloc[:val_start]
    df_val = df.iloc[val_start:test_start]
    df_test = df.iloc[test_start:]

    logger.info(f"  Train: {len(df_train)} rows")
    logger.info(f"  Val:   {len(df_val)} rows")
    logger.info(f"  Test:  {len(df_test)} rows (backtest period)")

    # Step 3: Train XGBoost
    logger.info("\nStep 3: Training XGBoost...")
    xgb = XGBoostModel()
    X_train, y_train = xgb.prepare_features(df_train)
    X_val, y_val = xgb.prepare_features(df_val)
    X_test, y_test = xgb.prepare_features(df_test)
    xgb.train(X_train, y_train, eval_set=(X_val, y_val))

    # Step 3b: Train TabNet
    logger.info("\nStep 3b: Training TabNet...")
    tabnet = None
    tabnet_test_probs = None
    try:
        tabnet = TabNetModel()
        tabnet.feature_names = xgb.feature_names
        tabnet.train(X_train, y_train, eval_set=(X_val, y_val))
        tabnet_test_probs = tabnet.predict_proba(X_test)
        logger.info(f"  TabNet mean prob: {tabnet_test_probs.mean():.4f}")
    except Exception as e:
        logger.warning(f"TabNet failed: {e}")

    # Step 4: Train LSTM with multi-run averaging
    N_LSTM_RUNS = 3
    logger.info(f"\nStep 4: Training LSTM ({N_LSTM_RUNS} runs for stability)...")
    lstm = LSTMModel()
    X_train_seq, y_train_seq = lstm.prepare_sequences(df_train, fit_scaler=True)
    X_val_seq, y_val_seq = lstm.prepare_sequences(df_val, fit_scaler=False)
    X_test_seq, y_test_seq = lstm.prepare_sequences(df_test, fit_scaler=False)

    lstm_ok = len(X_train_seq) > 0 and len(X_val_seq) > 0
    lstm_all_probs = []

    if lstm_ok:
        seeds = [42, 123, 777]
        for run_i in range(N_LSTM_RUNS):
            logger.info(f"  LSTM run {run_i+1}/{N_LSTM_RUNS} (seed={seeds[run_i]})...")
            lstm_run = LSTMModel()
            lstm_run._seed = seeds[run_i]
            lstm_run.scaler = lstm.scaler
            lstm_run.n_features = lstm.n_features
            lstm_run.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            probs = lstm_run.predict_proba(X_test_seq)
            lstm_all_probs.append(probs)
            logger.info(f"    Run {run_i+1} mean prob: {probs.mean():.4f}")

        lstm_test_probs = np.mean(lstm_all_probs, axis=0)
        logger.info(f"  Averaged LSTM prob: mean={lstm_test_probs.mean():.4f}, "
                     f"std across runs={np.std([p.mean() for p in lstm_all_probs]):.4f}")

    # Step 4b: Train TFT (Temporal Fusion Transformer)
    logger.info("\nStep 4b: Training TFT...")
    tft = None
    tft_test_probs = None
    try:
        tft = TFTModel()
        X_train_tft, y_train_tft = tft.prepare_sequences(df_train, fit_scaler=True)
        X_val_tft, y_val_tft = tft.prepare_sequences(df_val, fit_scaler=False)
        X_test_tft, y_test_tft = tft.prepare_sequences(df_test, fit_scaler=False)

        if len(X_train_tft) > 0 and len(X_val_tft) > 0:
            tft.train(X_train_tft, y_train_tft, X_val_tft, y_val_tft)
            tft_test_probs = tft.predict_proba(X_test_tft)
            logger.info(f"  TFT mean prob: {tft_test_probs.mean():.4f}")
    except Exception as e:
        logger.warning(f"TFT failed: {e}")

    # Step 5: Generate ensemble predictions
    logger.info("\nStep 5: Generating ensemble predictions (5-model)...")
    ensemble = EnsembleModel()

    xgb_test_probs = xgb.predict_proba(X_test)

    # Align all model predictions to the shortest length
    n_align = len(xgb_test_probs)
    if lstm_ok:
        n_align = min(n_align, len(lstm_test_probs))
    if tft_test_probs is not None:
        n_align = min(n_align, len(tft_test_probs))

    # Take last n_align from each
    xgb_aligned = xgb_test_probs[-n_align:]
    tabnet_aligned = tabnet_test_probs[-n_align:] if tabnet_test_probs is not None else np.full(n_align, 0.5)
    lstm_aligned = lstm_test_probs[-n_align:] if lstm_ok else np.full(n_align, 0.5)
    tft_aligned = tft_test_probs[-n_align:] if tft_test_probs is not None else np.full(n_align, 0.5)
    sent_probs = np.full(n_align, 0.5)
    sent_confs = np.full(n_align, 0.0)

    predictions = ensemble.predict_batch(
        xgb_aligned, lstm_aligned, tft_aligned, tabnet_aligned, sent_probs, sent_confs
    )
    test_dates = df_test.index[-n_align:]

    logger.info(f"  Generated {len(predictions)} predictions")
    signals = [p["signal"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]
    probabilities = [p["probability"] for p in predictions]

    logger.info(f"  Signals: BUY_CALL={signals.count('BUY_CALL')}, "
                f"BUY_PUT={signals.count('BUY_PUT')}, HOLD={signals.count('HOLD')}")

    # === DIAGNOSTIC OUTPUT (printed to stdout so user always sees it) ===
    print(f"\n{'='*60}")
    print(f"  PREDICTION DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Signals: BUY_CALL={signals.count('BUY_CALL')}, "
          f"BUY_PUT={signals.count('BUY_PUT')}, HOLD={signals.count('HOLD')}")
    print(f"\n  Model probability distributions:")
    print(f"    XGBoost:  min={xgb_aligned.min():.4f}, max={xgb_aligned.max():.4f}, "
          f"mean={xgb_aligned.mean():.4f}, std={xgb_aligned.std():.4f}")
    if lstm_ok:
        print(f"    LSTM:     min={lstm_aligned.min():.4f}, max={lstm_aligned.max():.4f}, "
              f"mean={lstm_aligned.mean():.4f}, std={lstm_aligned.std():.4f}")
    if tft_test_probs is not None:
        print(f"    TFT:      min={tft_aligned.min():.4f}, max={tft_aligned.max():.4f}, "
              f"mean={tft_aligned.mean():.4f}, std={tft_aligned.std():.4f}")
    if tabnet_test_probs is not None:
        print(f"    TabNet:   min={tabnet_aligned.min():.4f}, max={tabnet_aligned.max():.4f}, "
              f"mean={tabnet_aligned.mean():.4f}, std={tabnet_aligned.std():.4f}")
    print(f"\n  Ensemble confidence: min={min(confidences):.4f}, max={max(confidences):.4f}, "
          f"mean={np.mean(confidences):.4f}")
    print(f"  Ensemble probability: min={min(probabilities):.4f}, max={max(probabilities):.4f}, "
          f"mean={np.mean(probabilities):.4f}")
    print(f"  Confidence threshold: {ensemble.confidence_threshold}")

    # Show confidence distribution
    conf_arr = np.array(confidences)
    print(f"\n  Confidence distribution:")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    {pct}th percentile: {np.percentile(conf_arr, pct):.4f}")

    # If mostly HOLD, explain why
    hold_pct = signals.count("HOLD") / len(signals) * 100
    if hold_pct > 80:
        below_threshold = sum(1 for c in confidences if c < ensemble.confidence_threshold)
        print(f"\n  WARNING: {hold_pct:.0f}% HOLD signals!")
        print(f"  {below_threshold}/{len(confidences)} predictions below confidence threshold "
              f"({ensemble.confidence_threshold})")
        print(f"  Consider lowering confidence_threshold in config/settings.yaml")
    print(f"{'='*60}\n")

    # Step 6: Run backtests
    all_results = {}
    modes_to_run = ["directional", "options"] if mode == "both" else [mode]

    for bt_mode in modes_to_run:
        logger.info(f"\nStep 6: Running {bt_mode.upper()} backtest...")
        backtester = Backtester(starting_capital=capital, mode=bt_mode)
        results = backtester.run(predictions, df_test, test_dates)

        # Save results
        logger.info(f"\nSaving {bt_mode} results...")
        backtester.save_results(results, symbol)
        all_results[bt_mode] = results

    # Step 7: Monte Carlo simulation on the primary backtest
    logger.info("\nStep 7: Running Monte Carlo simulation (1000 runs)...")
    from src.backtest.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(n_simulations=1000)

    primary_mode = "directional" if "directional" in all_results else mode
    primary_results = all_results[primary_mode]

    mc_results = mc.run(primary_results, starting_capital=capital)
    mc.save_results(mc_results, symbol)
    all_results["monte_carlo"] = mc_results

    # Save predictions for dashboard
    pred_df = pd.DataFrame([{
        "date": str(d.date()),
        "signal": p["signal"],
        "probability": p["probability"],
        "confidence": p["confidence"],
        "direction": p["direction"],
    } for d, p in zip(test_dates, predictions)])
    pred_path = DATA_DIR / "backtest" / f"predictions_{symbol}.csv"
    pred_df.to_csv(pred_path, index=False)

    # Return the primary mode's results (directional by default)
    primary = all_results.get("directional", all_results.get(mode, {}))
    return primary, all_results


def print_comparison(all_results: dict, capital: float):
    """Print side-by-side comparison of directional vs options results."""
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
        ("Final Value", f"${dm.get('final_portfolio_value', 0):,.2f}", f"${om.get('final_portfolio_value', 0):,.2f}"),
        ("Total Return", f"{dm.get('total_return', 0)*100:+.2f}%", f"{om.get('total_return', 0)*100:+.2f}%"),
        ("Total Trades", f"{dm.get('total_trades', 0)}", f"{om.get('total_trades', 0)}"),
        ("Win Rate", f"{dm.get('win_rate', 0)*100:.1f}%", f"{om.get('win_rate', 0)*100:.1f}%"),
        ("Direction Accuracy", f"{dm.get('direction_accuracy', 0)*100:.1f}%", f"{om.get('direction_accuracy', 0)*100:.1f}%"),
        ("Profit Factor", f"{dm.get('profit_factor', 0):.2f}", f"{om.get('profit_factor', 0):.2f}"),
        ("Sharpe Ratio", f"{dm.get('sharpe_ratio', 0):.2f}", f"{om.get('sharpe_ratio', 0):.2f}"),
        ("Max Drawdown", f"{dm.get('max_drawdown', 0)*100:.1f}%", f"{om.get('max_drawdown', 0)*100:.1f}%"),
        ("Expectancy/Trade", f"${dm.get('expectancy', 0):+,.2f}", f"${om.get('expectancy', 0):+,.2f}"),
    ]

    for label, d_val, o_val in rows:
        print(f"{label:<25} {d_val:>20} {o_val:>20}")

    print(f"{'='*70}")
    print(f"\n  Directional mode = honest model evaluation (stock-based P&L)")
    print(f"  Options mode = realistic options with non-linear theta decay")
    print(f"  If directional is profitable → model has real edge → options will amplify it")


if __name__ == "__main__":
    print("\n>>> BACKTEST V6.1 — 5-model ensemble with diagnostics <<<\n")
    parser = argparse.ArgumentParser(description="Run trading backtest")
    parser.add_argument("--symbol", default="SPY", help="Stock ticker")
    parser.add_argument("--days", type=int, default=365 * 5, help="Days of data (default: 5 years)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital")
    parser.add_argument("--mode", default="both", choices=["directional", "options", "both"],
                        help="Backtest mode")
    parser.add_argument("--test-ratio", type=float, default=0.30,
                        help="Fraction of data for testing (default: 0.30 = 30%%)")
    args = parser.parse_args()

    result = run_backtest(args.symbol, args.days, args.capital, args.mode, args.test_ratio)

    if result:
        primary_results, all_results = result

        # Print primary results summary
        m = primary_results.get("metrics", {})
        mode_name = m.get("mode", "directional").upper()
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY ({mode_name})")
        print(f"{'='*60}")
        print(f"Starting Capital:  ${args.capital:,.2f}")
        print(f"Final Value:       ${m.get('final_portfolio_value', 0):,.2f}")
        print(f"Total Return:      {m.get('total_return', 0)*100:+.2f}%")
        print(f"Win Rate:          {m.get('win_rate', 0)*100:.1f}%")
        print(f"Direction Accuracy:{m.get('direction_accuracy', 0)*100:.1f}%")
        print(f"Sharpe Ratio:      {m.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:      {m.get('max_drawdown', 0)*100:.1f}%")
        print(f"Total Trades:      {m.get('total_trades', 0)}")
        print(f"Profit Factor:     {m.get('profit_factor', 0):.2f}")
        print(f"Expectancy/Trade:  ${m.get('expectancy', 0):+,.2f}")
        print(f"{'='*60}")

        # Print comparison if both modes ran
        if len(all_results) > 1:
            print_comparison(all_results, args.capital)
