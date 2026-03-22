"""
Daily Prediction Runner — generates fresh predictions for all watchlist tickers.

This is the script that runs every morning before market open (6:00 AM EST).
It:
  1. Fetches latest price data and features
  2. Loads trained models
  3. Generates predictions for each ticker
  4. Saves results for the web dashboard
  5. Optionally sends notifications

Usage:
    python3 run_daily.py                    # Run for all tickers
    python3 run_daily.py --symbols SPY AAPL # Run for specific tickers
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

from src.config_loader import get_all_tickers, get_tickers, DATA_DIR
from src.data_pipeline.feature_builder import FeatureBuilder
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.tft_model import TFTModel
from src.models.tabnet_model import TabNetModel
from src.models.ensemble import EnsembleModel
from src.trading.signal_generator import SignalGenerator


def run_daily_predictions(symbols: list[str] = None):
    """Generate predictions for all or specified symbols."""
    if symbols is None:
        symbols = get_all_tickers()

    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY PREDICTION RUN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"{'='*60}")

    builder = FeatureBuilder(use_finbert=False)
    signal_gen = SignalGenerator()

    all_predictions = {}
    all_recommendations = []

    for symbol in symbols:
        logger.info(f"\n--- Processing {symbol} ---")
        try:
            # Build latest features (90 days is enough for indicators + LSTM window)
            df = builder.build_features(symbol, days=90, include_news=False)
            if df.empty or len(df) < 70:
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue

            # Load trained models (with SPY fallback)
            xgb = XGBoostModel()
            lstm = LSTMModel()
            tft = TFTModel()
            tabnet = TabNetModel()
            ensemble = EnsembleModel()

            def _load_model(model, name, symbol, fallback="SPY"):
                try:
                    model.load(symbol)
                    return model
                except FileNotFoundError:
                    try:
                        model.load(fallback)
                        return model
                    except FileNotFoundError:
                        logger.warning(f"No {name} model for {symbol}")
                        return None

            xgb = _load_model(xgb, "XGBoost", symbol)
            if xgb is None:
                continue

            lstm = _load_model(lstm, "LSTM", symbol)
            tft = _load_model(tft, "TFT", symbol)
            tabnet = _load_model(tabnet, "TabNet", symbol)
            _load_model(ensemble, "Ensemble", symbol)  # okay if default weights

            # Get predictions on the latest data point
            X, _ = xgb.prepare_features(df)
            if len(X) == 0:
                continue

            latest_X = X.iloc[[-1]]
            xgb_prob = xgb.predict_proba(latest_X)[0]

            # TabNet
            tabnet_prob = 0.5
            if tabnet is not None:
                try:
                    tabnet_prob = tabnet.predict_proba(latest_X)[0]
                except Exception:
                    tabnet_prob = 0.5

            # LSTM
            lstm_prob = 0.5
            if lstm is not None:
                X_seq, _ = lstm.prepare_sequences(df, fit_scaler=True)
                if len(X_seq) > 0:
                    lstm_prob = lstm.predict_proba(X_seq[-1:])[0]

            # TFT
            tft_prob = 0.5
            if tft is not None:
                try:
                    X_tft, _ = tft.prepare_sequences(df, fit_scaler=True)
                    if len(X_tft) > 0:
                        tft_prob = tft.predict_proba(X_tft[-1:])[0]
                except Exception:
                    tft_prob = 0.5

            # Ensemble prediction (5-model)
            prediction = ensemble.predict(
                xgb_prob=xgb_prob,
                lstm_prob=lstm_prob,
                tft_prob=tft_prob,
                tabnet_prob=tabnet_prob,
                sentiment_prob=0.5,
                sentiment_confidence=0.0,
            )

            # Get current price
            current_price = float(df["close"].iloc[-1])
            prediction["symbol"] = symbol
            prediction["current_price"] = current_price
            prediction["date"] = str(df.index[-1].date())

            all_predictions[symbol] = prediction

            logger.info(
                f"{symbol}: {prediction['signal']} "
                f"(prob={prediction['probability']:.3f}, "
                f"conf={prediction['confidence']:.3f}, "
                f"price=${current_price:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Generate trade recommendations
    logger.info(f"\n{'='*60}")
    logger.info("DAILY RECOMMENDATIONS")
    logger.info(f"{'='*60}")

    buy_signals = {s: p for s, p in all_predictions.items() if p["signal"] != "HOLD"}
    hold_signals = {s: p for s, p in all_predictions.items() if p["signal"] == "HOLD"}

    if buy_signals:
        logger.info(f"\nACTIONABLE SIGNALS ({len(buy_signals)}):")
        for symbol, pred in sorted(buy_signals.items(), key=lambda x: -x[1]["confidence"]):
            logger.info(
                f"  {pred['signal']:>10}  {symbol:<6}  "
                f"conf={pred['confidence']:.3f}  "
                f"prob={pred['probability']:.3f}  "
                f"price=${pred['current_price']:.2f}"
            )
    else:
        logger.info("\nNo actionable signals today. All HOLD.")

    if hold_signals:
        logger.info(f"\nHOLD ({len(hold_signals)}): {', '.join(hold_signals.keys())}")

    # Save predictions
    pred_dir = DATA_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Save as latest
    output = {
        "run_date": datetime.now().isoformat(),
        "predictions": all_predictions,
        "summary": {
            "total_symbols": len(symbols),
            "processed": len(all_predictions),
            "buy_call": sum(1 for p in all_predictions.values() if p["signal"] == "BUY_CALL"),
            "buy_put": sum(1 for p in all_predictions.values() if p["signal"] == "BUY_PUT"),
            "hold": sum(1 for p in all_predictions.values() if p["signal"] == "HOLD"),
        },
    }

    latest_path = pred_dir / "latest_predictions.json"
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nPredictions saved to {latest_path}")

    # Also save dated copy
    date_str = datetime.now().strftime("%Y%m%d")
    dated_path = pred_dir / f"daily_predictions_{date_str}.json"
    with open(dated_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily predictions")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to predict")
    args = parser.parse_args()

    result = run_daily_predictions(args.symbols)

    if result:
        s = result["summary"]
        print(f"\n{'='*50}")
        print(f"  Daily Run Complete")
        print(f"  Processed: {s['processed']}/{s['total_symbols']} symbols")
        print(f"  BUY CALL: {s['buy_call']}")
        print(f"  BUY PUT:  {s['buy_put']}")
        print(f"  HOLD:     {s['hold']}")
        print(f"{'='*50}")
