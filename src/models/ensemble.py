"""
Ensemble Meta-Model — combines XGBoost, LSTM, TFT, TabNet, and Sentiment signals.

V2 Ensemble with 5 models and proper stacking meta-learner:
  1. XGBoost — tabular gradient boosting (workhorse)
  2. LSTM — sequential price patterns
  3. TFT — temporal fusion transformer with attention
  4. TabNet — attention-based tabular deep learning
  5. Sentiment — news sentiment analysis

The ensemble supports two modes:
  1. Weighted average (simple): manually set weights per model
  2. Learned stacking (advanced): logistic regression on model outputs

Usage:
    from src.models.ensemble import EnsembleModel
    ensemble = EnsembleModel()
    result = ensemble.predict(
        xgb_prob=0.72, lstm_prob=0.65,
        tft_prob=0.68, tabnet_prob=0.70,
        sentiment_prob=0.58
    )
"""
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, DATA_DIR


class EnsembleModel:
    """
    Combines predictions from all sub-models into a final trading signal.

    Supports:
      - 3-model mode (backward compatible): XGBoost + LSTM + Sentiment
      - 5-model mode (V2): XGBoost + LSTM + TFT + TabNet + Sentiment
      - Learned stacking via logistic regression
    """

    def __init__(self):
        settings = get_settings()
        self.confidence_threshold = (
            settings.get("models", {})
            .get("ensemble", {})
            .get("confidence_threshold", 0.25)
        )

        # Load weights from config
        ensemble_config = settings.get("models", {}).get("ensemble", {})
        weight_config = ensemble_config.get("weights", {})

        # V2: 5-model weights (default splits if new models not in config)
        self.weights = {
            "xgboost": weight_config.get("xgboost", 0.30),
            "lstm": weight_config.get("lstm", 0.15),
            "tft": weight_config.get("tft", 0.25),
            "tabnet": weight_config.get("tabnet", 0.20),
            "sentiment": weight_config.get("sentiment", 0.10),
        }

        # Optional stacking model
        self.stacker = None
        self.use_stacking = False
        self.n_models = 5  # How many model inputs we expect

        self.model_dir = DATA_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"EnsembleModel V2 initialized — "
            f"weights: XGB={self.weights['xgboost']}, LSTM={self.weights['lstm']}, "
            f"TFT={self.weights['tft']}, TabNet={self.weights['tabnet']}, "
            f"Sent={self.weights['sentiment']}, "
            f"threshold={self.confidence_threshold}"
        )

    def set_weights(self, **kwargs):
        """Set model weights. Keys: xgboost, lstm, tft, tabnet, sentiment."""
        for k, v in kwargs.items():
            if k in self.weights:
                self.weights[k] = v
        # Normalize to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info(f"Weights updated: {self.weights}")

    def predict(
        self,
        xgb_prob: float,
        lstm_prob: float,
        tft_prob: float = 0.5,
        tabnet_prob: float = 0.5,
        sentiment_prob: float = 0.5,
        sentiment_confidence: float = 0.0,
    ) -> dict:
        """
        Combine model predictions into a final signal.

        Args:
            xgb_prob: XGBoost P(UP)
            lstm_prob: LSTM P(UP)
            tft_prob: TFT P(UP) — defaults to 0.5 (neutral) if not available
            tabnet_prob: TabNet P(UP) — defaults to 0.5 if not available
            sentiment_prob: Sentiment P(UP)
            sentiment_confidence: How confident the sentiment signal is (0-1)

        Returns:
            Dict with signal, probability, confidence, direction, model_contributions
        """
        # Determine which models are active (prob != 0.5 means we have real predictions)
        model_probs = {
            "xgboost": xgb_prob,
            "lstm": lstm_prob,
            "tft": tft_prob,
            "tabnet": tabnet_prob,
            "sentiment": sentiment_prob,
        }

        # Compute effective weights: reduce weight of models with no real signal
        effective_weights = self.weights.copy()

        # If sentiment has no data, redistribute its weight
        if sentiment_confidence < 0.1:
            sent_weight = effective_weights.pop("sentiment")
            total_remaining = sum(effective_weights.values())
            if total_remaining > 0:
                for k in effective_weights:
                    effective_weights[k] += sent_weight * (effective_weights[k] / total_remaining)
            effective_weights["sentiment"] = 0.0
        else:
            # Scale sentiment weight by confidence
            effective_weights["sentiment"] *= sentiment_confidence

        # If TFT is not trained (prob == 0.5 exactly), redistribute
        if tft_prob == 0.5:
            tft_w = effective_weights.pop("tft", 0)
            total_remaining = sum(effective_weights.values())
            if total_remaining > 0:
                for k in effective_weights:
                    effective_weights[k] += tft_w * (effective_weights[k] / total_remaining)
            effective_weights["tft"] = 0.0

        # Same for TabNet
        if tabnet_prob == 0.5:
            tab_w = effective_weights.pop("tabnet", 0)
            total_remaining = sum(effective_weights.values())
            if total_remaining > 0:
                for k in effective_weights:
                    effective_weights[k] += tab_w * (effective_weights[k] / total_remaining)
            effective_weights["tabnet"] = 0.0

        # Normalize effective weights
        total_w = sum(effective_weights.values())
        if total_w > 0:
            effective_weights = {k: v / total_w for k, v in effective_weights.items()}

        # Compute combined probability
        if self.use_stacking and self.stacker is not None:
            features = np.array([[xgb_prob, lstm_prob, tft_prob, tabnet_prob, sentiment_prob]])
            combined_prob = self.stacker.predict_proba(features)[0, 1]
        else:
            combined_prob = sum(
                effective_weights.get(k, 0) * model_probs[k]
                for k in model_probs
            )

        # Direction
        direction = "UP" if combined_prob > 0.5 else "DOWN"

        # Confidence: based on model agreement and conviction
        active_models = {k: v for k, v in model_probs.items()
                        if effective_weights.get(k, 0) > 0.05}

        agreements = sum(1 for p in active_models.values()
                        if (p > 0.5) == (direction == "UP"))
        n_active = len(active_models) if active_models else 1

        agreement_ratio = agreements / n_active

        # Individual convictions: how far each model is from 0.5 (undecided)
        # Scaled so a prob of 0.55 gives conviction=0.10, prob of 0.60 gives 0.20
        convictions = [abs(p - 0.5) * 2 for p in active_models.values()]
        avg_conviction = np.mean(convictions) if convictions else 0
        max_conviction = max(convictions) if convictions else 0

        # Confidence formula: combines agreement, conviction, and a base signal strength
        # The base ensures that even low-conviction signals get a chance when models agree
        # agreement_ratio: what fraction of active models agree on direction
        # avg_conviction: average model "decisiveness" (0 = uncertain, 1 = certain)
        # max_conviction: strongest individual model signal
        if agreement_ratio >= 0.8:
            # Strong agreement: base 0.30 + conviction bonus
            confidence = 0.30 + avg_conviction * 0.5 + max_conviction * 0.2
        elif agreement_ratio >= 0.6:
            # Moderate agreement: base 0.15 + conviction
            confidence = 0.15 + avg_conviction * 0.5 + max_conviction * 0.15
        else:
            # Disagreement: rely purely on conviction strength
            confidence = avg_conviction * 0.3

        confidence = min(confidence, 1.0)

        # Generate signal
        if confidence >= self.confidence_threshold:
            signal = "BUY_CALL" if direction == "UP" else "BUY_PUT"
        else:
            signal = "HOLD"

        result = {
            "signal": signal,
            "probability": float(combined_prob),
            "confidence": float(confidence),
            "direction": direction,
            "model_contributions": {
                k: {"prob": float(model_probs[k]), "weight": float(effective_weights.get(k, 0))}
                for k in model_probs
            },
        }

        return result

    def predict_batch(
        self,
        xgb_probs: np.ndarray,
        lstm_probs: np.ndarray,
        tft_probs: np.ndarray = None,
        tabnet_probs: np.ndarray = None,
        sentiment_probs: np.ndarray = None,
        sentiment_confidences: np.ndarray = None,
    ) -> list[dict]:
        """
        Batch prediction for multiple samples.

        Backward compatible: tft_probs and tabnet_probs default to 0.5 if not provided.
        """
        n = len(xgb_probs)

        if tft_probs is None:
            tft_probs = np.full(n, 0.5)
        if tabnet_probs is None:
            tabnet_probs = np.full(n, 0.5)
        if sentiment_probs is None:
            sentiment_probs = np.full(n, 0.5)
        if sentiment_confidences is None:
            sentiment_confidences = np.full(n, 0.0)

        results = []
        for i in range(n):
            result = self.predict(
                xgb_prob=xgb_probs[i],
                lstm_prob=lstm_probs[i],
                tft_prob=tft_probs[i],
                tabnet_prob=tabnet_probs[i],
                sentiment_prob=sentiment_probs[i],
                sentiment_confidence=sentiment_confidences[i],
            )
            results.append(result)
        return results

    def train_stacker(
        self,
        xgb_probs: np.ndarray,
        lstm_probs: np.ndarray,
        tft_probs: np.ndarray = None,
        tabnet_probs: np.ndarray = None,
        sentiment_probs: np.ndarray = None,
        y_true: np.ndarray = None,
    ):
        """
        Train a logistic regression stacker on model outputs.
        Learns optimal combination weights from validation data.
        """
        n = len(xgb_probs)
        if tft_probs is None:
            tft_probs = np.full(n, 0.5)
        if tabnet_probs is None:
            tabnet_probs = np.full(n, 0.5)
        if sentiment_probs is None:
            sentiment_probs = np.full(n, 0.5)

        X_stack = np.column_stack([xgb_probs, lstm_probs, tft_probs, tabnet_probs, sentiment_probs])

        self.stacker = LogisticRegression(random_state=42, max_iter=1000, C=0.5)
        self.stacker.fit(X_stack, y_true)
        self.use_stacking = True
        self.n_models = X_stack.shape[1]

        coefs = self.stacker.coef_[0]
        model_names = ["XGB", "LSTM", "TFT", "TabNet", "Sent"]
        coef_str = ", ".join(f"{name}={c:.3f}" for name, c in zip(model_names, coefs))
        logger.info(f"Stacker trained — learned coefficients: {coef_str}")

        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = self.stacker.predict(X_stack)
        y_prob = self.stacker.predict_proba(X_stack)[:, 1]
        logger.info(
            f"Stacker performance — "
            f"Acc: {accuracy_score(y_true, y_pred):.4f}, "
            f"AUC: {roc_auc_score(y_true, y_prob):.4f}"
        )

    def save(self, symbol: str = "default"):
        filepath = self.model_dir / f"ensemble_{symbol}.pkl"
        joblib.dump({
            "weights": self.weights,
            "confidence_threshold": self.confidence_threshold,
            "stacker": self.stacker,
            "use_stacking": self.use_stacking,
            "n_models": self.n_models,
        }, filepath)
        logger.info(f"Ensemble saved to {filepath}")

    def load(self, symbol: str = "default"):
        filepath = self.model_dir / f"ensemble_{symbol}.pkl"
        data = joblib.load(filepath)
        self.weights = data["weights"]
        self.confidence_threshold = data["confidence_threshold"]
        self.stacker = data["stacker"]
        self.use_stacking = data["use_stacking"]
        self.n_models = data.get("n_models", 3)

        # Ensure all 5 weight keys exist (backward compat)
        for k in ["xgboost", "lstm", "tft", "tabnet", "sentiment"]:
            if k not in self.weights:
                self.weights[k] = 0.0

        logger.info(f"Ensemble loaded from {filepath}")
