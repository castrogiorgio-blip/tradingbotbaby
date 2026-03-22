"""
Model Trainer V2 — orchestrates training of all 5 models.

Full pipeline:
  1. Fetch and build features
  2. Split data into train/validation/test (time-series aware)
  3. Train XGBoost on tabular features
  4. Train TabNet on tabular features (complementary to XGBoost)
  5. Train LSTM on price sequences
  6. Train TFT on price sequences (state-of-the-art transformer)
  7. Train ensemble stacker on all model outputs
  8. Evaluate everything and report metrics
  9. Save all models

Usage:
    from src.models.trainer import ModelTrainer
    trainer = ModelTrainer()
    results = trainer.train_all("SPY", days=365*3)
"""
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import DATA_DIR
from src.data_pipeline.feature_builder import FeatureBuilder
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.tft_model import TFTModel
from src.models.tabnet_model import TabNetModel
from src.models.ensemble import EnsembleModel


class ModelTrainer:
    """Orchestrates the full model training pipeline."""

    def __init__(self, use_finbert: bool = False):
        self.feature_builder = FeatureBuilder(use_finbert=use_finbert)
        self.results_dir = DATA_DIR / "predictions"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelTrainer V2 initialized (XGBoost + TabNet + LSTM + TFT + Ensemble)")

    def train_all(
        self,
        symbol: str,
        days: int = 365 * 3,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        include_news: bool = False,
    ) -> dict:
        """Full training pipeline for one symbol."""

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING PIPELINE V2 FOR {symbol}")
        logger.info(f"{'='*60}")

        # ── Step 1: Build features ───────────────────────────────
        logger.info("Step 1: Building features...")
        df = self.feature_builder.build_features(
            symbol, days=days, include_news=include_news
        )
        if df.empty or len(df) < 100:
            logger.error(f"Insufficient data for {symbol} ({len(df)} rows)")
            return {"error": "Insufficient data"}

        # ── Step 2: Time-series split ────────────────────────────
        logger.info("Step 2: Splitting data (time-series aware)...")
        n = len(df)
        test_start = int(n * (1 - test_ratio))
        val_start = int(test_start * (1 - val_ratio))

        df_train = df.iloc[:val_start]
        df_val = df.iloc[val_start:test_start]
        df_test = df.iloc[test_start:]

        logger.info(f"  Train: {len(df_train)} rows ({df_train.index[0].date()} → {df_train.index[-1].date()})")
        logger.info(f"  Val:   {len(df_val)} rows ({df_val.index[0].date()} → {df_val.index[-1].date()})")
        logger.info(f"  Test:  {len(df_test)} rows ({df_test.index[0].date()} → {df_test.index[-1].date()})")

        results = {
            "symbol": symbol,
            "data_shape": df.shape,
            "train_size": len(df_train),
            "val_size": len(df_val),
            "test_size": len(df_test),
        }

        # ── Step 3: Train XGBoost ────────────────────────────────
        logger.info("\nStep 3: Training XGBoost...")
        xgb_model = XGBoostModel()

        X_train, y_train = xgb_model.prepare_features(df_train)
        X_val, y_val = xgb_model.prepare_features(df_val)
        X_test, y_test = xgb_model.prepare_features(df_test)

        xgb_model.train(X_train, y_train, eval_set=(X_val, y_val))

        xgb_test_metrics = xgb_model.evaluate(X_test, y_test)
        results["xgboost"] = {
            "train": xgb_model.evaluate(X_train, y_train),
            "validation": xgb_model.evaluate(X_val, y_val),
            "test": xgb_test_metrics,
        }

        # Feature importance
        top_features = xgb_model.get_feature_importance(top_n=15)
        logger.info(f"\nTop 15 XGBoost features:")
        for _, row in top_features.iterrows():
            logger.info(f"  {row['feature']:<40} {row['importance']:.4f}")

        # Cross-validation
        logger.info("\nXGBoost time-series cross-validation:")
        cv_results = xgb_model.cross_validate(
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val]),
            n_splits=5,
        )
        results["xgboost"]["cross_validation"] = cv_results

        # ── Step 4: Train TabNet ─────────────────────────────────
        logger.info("\nStep 4: Training TabNet...")
        try:
            tabnet_model = TabNetModel()
            # Use same prepared features as XGBoost
            tabnet_model.feature_names = xgb_model.feature_names
            tabnet_model.train(X_train, y_train, eval_set=(X_val, y_val))

            tabnet_test_metrics = tabnet_model.evaluate(X_test, y_test)
            results["tabnet"] = {
                "train": tabnet_model.evaluate(X_train, y_train),
                "validation": tabnet_model.evaluate(X_val, y_val),
                "test": tabnet_test_metrics,
            }
        except Exception as e:
            logger.error(f"TabNet training failed: {e}")
            tabnet_model = None
            results["tabnet"] = {"error": str(e)}

        # ── Step 5: Train LSTM ───────────────────────────────────
        logger.info("\nStep 5: Training LSTM...")
        lstm_model = LSTMModel()

        X_train_seq, y_train_seq = lstm_model.prepare_sequences(df_train, fit_scaler=True)
        X_val_seq, y_val_seq = lstm_model.prepare_sequences(df_val, fit_scaler=False)
        X_test_seq, y_test_seq = lstm_model.prepare_sequences(df_test, fit_scaler=False)

        lstm_ok = len(X_train_seq) > 0 and len(X_val_seq) > 0
        if lstm_ok:
            lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            results["lstm"] = {
                "train": lstm_model.evaluate(X_train_seq, y_train_seq),
                "validation": lstm_model.evaluate(X_val_seq, y_val_seq),
                "test": lstm_model.evaluate(X_test_seq, y_test_seq),
            }
        else:
            logger.warning("Insufficient data for LSTM sequences")
            results["lstm"] = {"error": "Insufficient sequence data"}

        # ── Step 6: Train TFT ────────────────────────────────────
        logger.info("\nStep 6: Training TFT (Temporal Fusion Transformer)...")
        tft_model = None
        try:
            tft_model = TFTModel()
            X_train_tft, y_train_tft = tft_model.prepare_sequences(df_train, fit_scaler=True)
            X_val_tft, y_val_tft = tft_model.prepare_sequences(df_val, fit_scaler=False)
            X_test_tft, y_test_tft = tft_model.prepare_sequences(df_test, fit_scaler=False)

            tft_ok = len(X_train_tft) > 0 and len(X_val_tft) > 0
            if tft_ok:
                tft_model.train(X_train_tft, y_train_tft, X_val_tft, y_val_tft)
                results["tft"] = {
                    "train": tft_model.evaluate(X_train_tft, y_train_tft),
                    "validation": tft_model.evaluate(X_val_tft, y_val_tft),
                    "test": tft_model.evaluate(X_test_tft, y_test_tft),
                }
            else:
                results["tft"] = {"error": "Insufficient sequence data"}
                tft_model = None
        except Exception as e:
            logger.error(f"TFT training failed: {e}")
            results["tft"] = {"error": str(e)}
            tft_model = None

        # ── Step 7: Train Ensemble Stacker ───────────────────────
        logger.info("\nStep 7: Training Ensemble Stacker...")
        ensemble = EnsembleModel()

        # Collect validation predictions from all models
        xgb_val_probs = xgb_model.predict_proba(X_val)
        xgb_test_probs = xgb_model.predict_proba(X_test)

        # TabNet predictions
        tabnet_val_probs = tabnet_model.predict_proba(X_val) if tabnet_model else np.full(len(X_val), 0.5)
        tabnet_test_probs = tabnet_model.predict_proba(X_test) if tabnet_model else np.full(len(X_test), 0.5)

        # LSTM predictions (need alignment due to sequence windowing)
        if lstm_ok:
            lstm_val_probs = lstm_model.predict_proba(X_val_seq)
            lstm_test_probs = lstm_model.predict_proba(X_test_seq)
            n_lstm_val = len(lstm_val_probs)
            n_lstm_test = len(lstm_test_probs)
        else:
            lstm_val_probs = None
            lstm_test_probs = None
            n_lstm_val = 0
            n_lstm_test = 0

        # TFT predictions
        if tft_model is not None:
            tft_val_probs = tft_model.predict_proba(X_val_tft)
            tft_test_probs = tft_model.predict_proba(X_test_tft)
            n_tft_val = len(tft_val_probs)
            n_tft_test = len(tft_test_probs)
        else:
            tft_val_probs = None
            tft_test_probs = None
            n_tft_val = 0
            n_tft_test = 0

        # Align all predictions to the same length (shortest sequence model)
        n_val_align = min(
            len(xgb_val_probs),
            n_lstm_val if lstm_ok else len(xgb_val_probs),
            n_tft_val if tft_model else len(xgb_val_probs),
        )
        n_test_align = min(
            len(xgb_test_probs),
            n_lstm_test if lstm_ok else len(xgb_test_probs),
            n_tft_test if tft_model else len(xgb_test_probs),
        )

        # Take last N from each (aligned to most recent dates)
        xgb_val_aligned = xgb_val_probs[-n_val_align:]
        tabnet_val_aligned = tabnet_val_probs[-n_val_align:]
        lstm_val_aligned = lstm_val_probs[-n_val_align:] if lstm_ok else np.full(n_val_align, 0.5)
        tft_val_aligned = tft_val_probs[-n_val_align:] if tft_model else np.full(n_val_align, 0.5)
        sent_val_aligned = np.full(n_val_align, 0.5)
        y_val_aligned = y_val.values[-n_val_align:]

        xgb_test_aligned = xgb_test_probs[-n_test_align:]
        tabnet_test_aligned = tabnet_test_probs[-n_test_align:]
        lstm_test_aligned = lstm_test_probs[-n_test_align:] if lstm_ok else np.full(n_test_align, 0.5)
        tft_test_aligned = tft_test_probs[-n_test_align:] if tft_model else np.full(n_test_align, 0.5)
        sent_test_aligned = np.full(n_test_align, 0.5)
        sent_test_confs = np.full(n_test_align, 0.0)
        y_test_aligned = y_test.values[-n_test_align:]

        # Train stacker if we have enough samples
        if n_val_align >= 50:
            ensemble.train_stacker(
                xgb_val_aligned, lstm_val_aligned,
                tft_val_aligned, tabnet_val_aligned,
                sent_val_aligned, y_val_aligned
            )
        else:
            logger.info(f"Skipping stacker (only {n_val_align} aligned val samples, need 50+)")

        # Evaluate ensemble on test set
        ensemble_preds = ensemble.predict_batch(
            xgb_test_aligned, lstm_test_aligned,
            tft_test_aligned, tabnet_test_aligned,
            sent_test_aligned, sent_test_confs
        )

        from sklearn.metrics import accuracy_score, roc_auc_score
        ensemble_probs = np.array([p["probability"] for p in ensemble_preds])
        ensemble_dirs = (ensemble_probs > 0.5).astype(int)

        results["ensemble"] = {
            "test_accuracy": accuracy_score(y_test_aligned, ensemble_dirs),
            "test_auc_roc": roc_auc_score(y_test_aligned, ensemble_probs),
            "signal_distribution": {
                "BUY_CALL": sum(1 for p in ensemble_preds if p["signal"] == "BUY_CALL"),
                "BUY_PUT": sum(1 for p in ensemble_preds if p["signal"] == "BUY_PUT"),
                "HOLD": sum(1 for p in ensemble_preds if p["signal"] == "HOLD"),
            },
            "avg_confidence": np.mean([p["confidence"] for p in ensemble_preds]),
        }

        logger.info(
            f"\nEnsemble Test Results — "
            f"Acc: {results['ensemble']['test_accuracy']:.4f}, "
            f"AUC: {results['ensemble']['test_auc_roc']:.4f}"
        )
        logger.info(f"Signal distribution: {results['ensemble']['signal_distribution']}")

        # ── Step 8: Save all models ──────────────────────────────
        logger.info("\nStep 8: Saving models...")
        xgb_model.save(symbol)
        if tabnet_model:
            tabnet_model.save(symbol)
        if lstm_ok:
            lstm_model.save(symbol)
        if tft_model:
            tft_model.save(symbol)
        ensemble.save(symbol)

        # ── Summary ──────────────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE FOR {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"XGBoost test AUC:  {results['xgboost']['test']['auc_roc']:.4f}")
        if "error" not in results.get("tabnet", {}):
            logger.info(f"TabNet test AUC:   {results['tabnet']['test']['auc_roc']:.4f}")
        if "error" not in results.get("lstm", {}):
            logger.info(f"LSTM test AUC:     {results['lstm']['test']['auc_roc']:.4f}")
        if "error" not in results.get("tft", {}):
            logger.info(f"TFT test AUC:      {results['tft']['test']['auc_roc']:.4f}")
        logger.info(f"Ensemble test AUC: {results['ensemble']['test_auc_roc']:.4f}")

        # Save results text
        results_path = self.results_dir / f"training_results_{symbol}.txt"
        with open(results_path, "w") as f:
            f.write(f"Training Results V2 for {symbol}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Models: XGBoost + TabNet + LSTM + TFT + Ensemble\n")
            f.write(f"{'='*60}\n\n")
            self._write_results(f, results)
        logger.info(f"Results saved to {results_path}")

        return results

    def _write_results(self, f, results, indent=0):
        prefix = "  " * indent
        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"{prefix}{key}:\n")
                self._write_results(f, value, indent + 1)
            else:
                f.write(f"{prefix}{key}: {value}\n")

    def train_multiple(self, symbols: list[str], days: int = 365 * 3, **kwargs) -> dict:
        all_results = {}
        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Training {symbol}...")
            try:
                all_results[symbol] = self.train_all(symbol, days=days, **kwargs)
            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}")
                all_results[symbol] = {"error": str(e)}
        return all_results
