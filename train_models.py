"""
Training Script V2 — trains all 5 ML models.

Models: XGBoost + TabNet + LSTM + TFT + Ensemble Stacker

Usage:
    python3 train_models.py                  # Train on SPY only (fast test)
    python3 train_models.py --all            # Train on all watchlist symbols
    python3 train_models.py --symbols AAPL MSFT TSLA  # Train specific symbols
    python3 train_models.py --days 730       # Use 2 years of data
"""
import argparse
import sys
from datetime import datetime
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("data/predictions/training_{time}.log", level="DEBUG", rotation="10 MB")

from src.models.trainer import ModelTrainer
from src.config_loader import get_all_tickers


def main():
    parser = argparse.ArgumentParser(description="Train TradingBot ML models")
    parser.add_argument("--symbols", nargs="+", default=["SPY"],
                        help="Symbols to train on (default: SPY)")
    parser.add_argument("--all", action="store_true",
                        help="Train on all watchlist symbols")
    parser.add_argument("--days", type=int, default=365 * 3,
                        help="Days of historical data (default: 1095 = 3 years)")
    parser.add_argument("--finbert", action="store_true",
                        help="Enable FinBERT sentiment (slower, more accurate)")
    parser.add_argument("--news", action="store_true",
                        help="Include news sentiment features")
    args = parser.parse_args()

    if args.all:
        symbols = get_all_tickers()
    else:
        symbols = args.symbols

    print(f"\n{'#'*60}")
    print(f"# TradingBot ML V2 — Model Training")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Symbols: {symbols}")
    print(f"# Models: XGBoost + TabNet + LSTM + TFT + Ensemble")
    print(f"# History: {args.days} days ({args.days/365:.1f} years)")
    print(f"# FinBERT: {'ON' if args.finbert else 'OFF'}")
    print(f"{'#'*60}\n")

    trainer = ModelTrainer(use_finbert=args.finbert)

    if len(symbols) == 1:
        results = trainer.train_all(
            symbols[0], days=args.days, include_news=args.news,
        )
        _print_summary(symbols[0], results)
    else:
        all_results = trainer.train_multiple(
            symbols, days=args.days, include_news=args.news,
        )
        for symbol, results in all_results.items():
            _print_summary(symbol, results)


def _print_summary(symbol: str, results: dict):
    """Print a clean summary of training results."""
    print(f"\n{'='*60}")
    print(f"  RESULTS: {symbol}")
    print(f"{'='*60}")

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    print(f"  Data: {results.get('train_size', '?')} train / "
          f"{results.get('val_size', '?')} val / "
          f"{results.get('test_size', '?')} test")

    # Print each model's results
    for model_name in ["xgboost", "tabnet", "lstm", "tft"]:
        model_data = results.get(model_name, {})
        if "error" in model_data:
            print(f"\n  {model_name.upper()}: {model_data['error']}")
            continue
        if "test" in model_data:
            t = model_data["test"]
            print(f"\n  {model_name.upper()} (test set):")
            print(f"    Accuracy:  {t.get('accuracy', 0):.4f}")
            print(f"    AUC-ROC:   {t.get('auc_roc', 0):.4f}")
            if 'precision' in t:
                print(f"    Precision: {t['precision']:.4f}")
            print(f"    F1 Score:  {t.get('f1', 0):.4f}")

    # XGBoost CV
    xgb = results.get("xgboost", {})
    if "cross_validation" in xgb:
        cv = xgb["cross_validation"]
        print(f"\n  XGBoost CV:")
        print(f"    Accuracy: {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
        print(f"    AUC-ROC:  {cv['auc_roc_mean']:.4f} ± {cv['auc_roc_std']:.4f}")

    # Ensemble
    ens = results.get("ensemble", {})
    if "test_accuracy" in ens:
        print(f"\n  ENSEMBLE (test set):")
        print(f"    Accuracy:  {ens['test_accuracy']:.4f}")
        print(f"    AUC-ROC:   {ens['test_auc_roc']:.4f}")
        print(f"    Signals:   {ens['signal_distribution']}")
        print(f"    Avg Conf:  {ens['avg_confidence']:.4f}")

    print()


if __name__ == "__main__":
    main()
