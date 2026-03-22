"""
Ensemble Stacker: Learned meta-learner for optimal model combination.

This module implements a logistic regression meta-learner that learns optimal
combination weights from the 5-model ensemble (XGBoost, LSTM, TFT, TabNet, Sentiment).

Key features:
  1. Walk-forward training to prevent look-ahead bias
  2. Isotonic calibration with debiasing
  3. Model agreement & uncertainty features
  4. Support for missing models
  5. Serialization (save/load)
  6. Detailed diagnostics

The core insight: models have very different quality. Rather than using fixed
weights, learn which models actually contribute to correct predictions.

Usage:
    from src.models.ensemble_stacker import EnsembleStacker, StackerFeatures

    stacker = EnsembleStacker(
        manual_weights={
            'xgboost': 0.30, 'lstm': 0.15, 'tft': 0.25,
            'tabnet': 0.20, 'sentiment': 0.10
        },
        min_train_samples=100
    )

    # Fit with walk-forward validation
    stacker.fit(
        model_predictions={
            'xgboost': xgb_probs,
            'lstm': lstm_probs,
            'tft': tft_probs,
            'tabnet': tabnet_probs,
            'sentiment': sentiment_probs,
        },
        y_true=labels,
        dates=date_index
    )

    # Predict on new data
    pred_proba = stacker.predict_proba(model_predictions)
    confidence = stacker.get_confidence(model_predictions)
    signal = stacker.generate_signal(model_predictions, threshold=0.15)

    # Inspect which models matter
    importance = stacker.get_model_importance()
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, DATA_DIR


@dataclass
class StackerFeatures:
    """Feature vector computed from model predictions."""
    # Raw predictions from each model
    xgb_prob: float
    lstm_prob: float
    tft_prob: float
    tabnet_prob: float
    sentiment_prob: float

    # Agreement features
    xgb_lstm_agree: int      # 1 if both predict same direction
    xgb_tft_agree: int
    xgb_tabnet_agree: int
    xgb_sentiment_agree: int
    lstm_tft_agree: int
    lstm_tabnet_agree: int
    lstm_sentiment_agree: int
    tft_tabnet_agree: int
    tft_sentiment_agree: int
    tabnet_sentiment_agree: int

    # Uncertainty measures
    min_prob: float          # minimum probability across models
    max_prob: float          # maximum probability across models
    spread: float            # max - min (measure of disagreement)
    mean_prob: float         # mean of all model predictions
    median_prob: float       # median of all model predictions

    # Vote counts
    n_up_votes: int          # how many models predict UP (prob > 0.5)
    n_down_votes: int        # how many models predict DOWN (prob < 0.5)

    def to_array(self) -> np.ndarray:
        """Convert to flat array for model input."""
        return np.array([
            self.xgb_prob, self.lstm_prob, self.tft_prob, self.tabnet_prob, self.sentiment_prob,
            self.xgb_lstm_agree, self.xgb_tft_agree, self.xgb_tabnet_agree, self.xgb_sentiment_agree,
            self.lstm_tft_agree, self.lstm_tabnet_agree, self.lstm_sentiment_agree,
            self.tft_tabnet_agree, self.tft_sentiment_agree, self.tabnet_sentiment_agree,
            self.min_prob, self.max_prob, self.spread, self.mean_prob, self.median_prob,
            self.n_up_votes, self.n_down_votes,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Return feature column names."""
        return [
            'xgb_prob', 'lstm_prob', 'tft_prob', 'tabnet_prob', 'sentiment_prob',
            'xgb_lstm_agree', 'xgb_tft_agree', 'xgb_tabnet_agree', 'xgb_sentiment_agree',
            'lstm_tft_agree', 'lstm_tabnet_agree', 'lstm_sentiment_agree',
            'tft_tabnet_agree', 'tft_sentiment_agree', 'tabnet_sentiment_agree',
            'min_prob', 'max_prob', 'spread', 'mean_prob', 'median_prob',
            'n_up_votes', 'n_down_votes',
        ]


class EnsembleStacker:
    """
    Learned meta-learner for optimal ensemble combination.

    Trains a logistic regression model on out-of-sample predictions from
    5 sub-models to learn their optimal combination weights.

    Attributes:
        lr_model: LogisticRegression meta-learner
        iso_reg: IsotonicRegression calibrator
        cal_bias: Calibration bias (mean prediction - label frequency)
        manual_weights: Fallback weights if stacker training fails
        is_fitted: Whether the stacker has been successfully trained
    """

    def __init__(
        self,
        manual_weights: Optional[Dict[str, float]] = None,
        min_train_samples: int = 100,
        use_agreement_features: bool = True,
        regularization_c: float = 0.5,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """
        Initialize the ensemble stacker.

        Args:
            manual_weights: Fallback weights if stacker training fails.
                Dict with keys: xgboost, lstm, tft, tabnet, sentiment.
                Defaults to equal weights.
            min_train_samples: Minimum samples needed to train (else use manual weights)
            use_agreement_features: Include pairwise agreement/spread features
            regularization_c: Inverse regularization (C=1/lambda). Lower = more regularization.
            max_iter: Max iterations for logistic regression
            random_state: Random seed
        """
        if manual_weights is None:
            manual_weights = {
                'xgboost': 0.30, 'lstm': 0.15, 'tft': 0.25,
                'tabnet': 0.20, 'sentiment': 0.10
            }

        self.manual_weights = manual_weights
        self.min_train_samples = min_train_samples
        self.use_agreement_features = use_agreement_features
        self.regularization_c = regularization_c
        self.max_iter = max_iter
        self.random_state = random_state

        # Learned components
        self.lr_model: Optional[LogisticRegression] = None
        self.iso_reg: Optional[IsotonicRegression] = None
        self.cal_bias: float = 0.0
        self.is_fitted: bool = False

        # Diagnostics
        self.train_accuracy: Optional[float] = None
        self.train_auc: Optional[float] = None
        self.train_log_loss: Optional[float] = None
        self.feature_importance: Optional[Dict[str, float]] = None

        # Storage
        self.model_dir = DATA_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"EnsembleStacker initialized — "
            f"min_samples={min_train_samples}, C={regularization_c}, "
            f"agreement_features={use_agreement_features}"
        )

    def _compute_features(
        self,
        model_predictions: Dict[str, Union[float, np.ndarray]]
    ) -> Union[StackerFeatures, np.ndarray]:
        """
        Compute rich feature set from model predictions.

        Args:
            model_predictions: Dict with keys 'xgboost', 'lstm', 'tft', 'tabnet', 'sentiment'
                             Values can be scalar or arrays (for batch processing)

        Returns:
            For scalar inputs: StackerFeatures object
            For array inputs: (N, n_features) array
        """
        # Extract predictions, with fallback to 0.5 if missing
        xgb = np.atleast_1d(model_predictions.get('xgboost', 0.5))
        lstm = np.atleast_1d(model_predictions.get('lstm', 0.5))
        tft = np.atleast_1d(model_predictions.get('tft', 0.5))
        tabnet = np.atleast_1d(model_predictions.get('tabnet', 0.5))
        sentiment = np.atleast_1d(model_predictions.get('sentiment', 0.5))

        is_scalar = all(np.isscalar(v) or len(np.atleast_1d(v)) == 1
                       for v in [xgb, lstm, tft, tabnet, sentiment])

        # Ensure all arrays have same length
        n = max(len(np.atleast_1d(v)) for v in [xgb, lstm, tft, tabnet, sentiment])
        xgb = np.atleast_1d(xgb); xgb = np.full(n, xgb[0]) if len(xgb) == 1 else xgb
        lstm = np.atleast_1d(lstm); lstm = np.full(n, lstm[0]) if len(lstm) == 1 else lstm
        tft = np.atleast_1d(tft); tft = np.full(n, tft[0]) if len(tft) == 1 else tft
        tabnet = np.atleast_1d(tabnet); tabnet = np.full(n, tabnet[0]) if len(tabnet) == 1 else tabnet
        sentiment = np.atleast_1d(sentiment); sentiment = np.full(n, sentiment[0]) if len(sentiment) == 1 else sentiment

        probs_all = np.column_stack([xgb, lstm, tft, tabnet, sentiment])

        # Agreement: 1 if both models predict same direction (both > 0.5 or both < 0.5)
        agree_func = lambda p1, p2: ((p1 > 0.5) == (p2 > 0.5)).astype(int)
        xgb_lstm_agree = agree_func(xgb, lstm)
        xgb_tft_agree = agree_func(xgb, tft)
        xgb_tabnet_agree = agree_func(xgb, tabnet)
        xgb_sentiment_agree = agree_func(xgb, sentiment)
        lstm_tft_agree = agree_func(lstm, tft)
        lstm_tabnet_agree = agree_func(lstm, tabnet)
        lstm_sentiment_agree = agree_func(lstm, sentiment)
        tft_tabnet_agree = agree_func(tft, tabnet)
        tft_sentiment_agree = agree_func(tft, sentiment)
        tabnet_sentiment_agree = agree_func(tabnet, sentiment)

        # Uncertainty measures
        min_prob = np.min(probs_all, axis=1)
        max_prob = np.max(probs_all, axis=1)
        spread = max_prob - min_prob
        mean_prob = np.mean(probs_all, axis=1)
        median_prob = np.median(probs_all, axis=1)

        # Vote counts
        n_up_votes = (probs_all > 0.5).sum(axis=1)
        n_down_votes = (probs_all <= 0.5).sum(axis=1)

        if is_scalar:
            return StackerFeatures(
                xgb_prob=float(xgb[0]),
                lstm_prob=float(lstm[0]),
                tft_prob=float(tft[0]),
                tabnet_prob=float(tabnet[0]),
                sentiment_prob=float(sentiment[0]),
                xgb_lstm_agree=int(xgb_lstm_agree[0]),
                xgb_tft_agree=int(xgb_tft_agree[0]),
                xgb_tabnet_agree=int(xgb_tabnet_agree[0]),
                xgb_sentiment_agree=int(xgb_sentiment_agree[0]),
                lstm_tft_agree=int(lstm_tft_agree[0]),
                lstm_tabnet_agree=int(lstm_tabnet_agree[0]),
                lstm_sentiment_agree=int(lstm_sentiment_agree[0]),
                tft_tabnet_agree=int(tft_tabnet_agree[0]),
                tft_sentiment_agree=int(tft_sentiment_agree[0]),
                tabnet_sentiment_agree=int(tabnet_sentiment_agree[0]),
                min_prob=float(min_prob[0]),
                max_prob=float(max_prob[0]),
                spread=float(spread[0]),
                mean_prob=float(mean_prob[0]),
                median_prob=float(median_prob[0]),
                n_up_votes=int(n_up_votes[0]),
                n_down_votes=int(n_down_votes[0]),
            )
        else:
            # Return as array for batch processing
            if self.use_agreement_features:
                X = np.column_stack([
                    xgb, lstm, tft, tabnet, sentiment,
                    xgb_lstm_agree, xgb_tft_agree, xgb_tabnet_agree, xgb_sentiment_agree,
                    lstm_tft_agree, lstm_tabnet_agree, lstm_sentiment_agree,
                    tft_tabnet_agree, tft_sentiment_agree, tabnet_sentiment_agree,
                    min_prob, max_prob, spread, mean_prob, median_prob,
                    n_up_votes, n_down_votes,
                ])
            else:
                # Just raw predictions
                X = probs_all

            return X

    def fit(
        self,
        model_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        dates: Optional[np.ndarray] = None,
        test_size: float = 0.2,
    ) -> Dict[str, any]:
        """
        Train the stacker with walk-forward validation.

        Implements expanding window cross-validation to avoid look-ahead bias:
        - Split data into train/test by date
        - For each fold: train on all prior data, test on current fold
        - Fit logistic regression + isotonic calibration

        Args:
            model_predictions: Dict with 'xgboost', 'lstm', 'tft', 'tabnet', 'sentiment'
                             Each value is array of shape (n_samples,)
            y_true: True labels (0 or 1), shape (n_samples,)
            dates: Optional date index for proper walk-forward split
            test_size: Fraction of data to use for final test set

        Returns:
            Dict with diagnostics:
                {
                    'n_samples': int,
                    'train_accuracy': float,
                    'train_auc': float,
                    'train_log_loss': float,
                    'feature_importance': dict,
                    'coefficients': dict,
                    'n_agreement_features': int,
                    'intercept': float,
                }
        """
        n = len(y_true)
        if n < self.min_train_samples:
            logger.warning(
                f"Insufficient training samples ({n} < {self.min_train_samples}). "
                f"Using manual weights as fallback."
            )
            self.is_fitted = False
            return {'status': 'fallback', 'reason': 'insufficient_samples', 'n_samples': n}

        # Compute features
        X = self._compute_features(model_predictions)
        logger.info(f"Stacker training data: X.shape={X.shape}, y.shape={y_true.shape}")

        # Determine train/test split
        if dates is not None:
            # Walk-forward: use older data for training, newer for testing
            dates_arr = np.array(dates)
            unique_dates = np.unique(dates_arr)
            split_idx = int(len(unique_dates) * (1 - test_size))
            cutoff_date = unique_dates[split_idx]
            train_mask = dates_arr < cutoff_date
            test_mask = dates_arr >= cutoff_date
        else:
            # Simple time-based split
            split_idx = int(n * (1 - test_size))
            train_mask = np.arange(n) < split_idx
            test_mask = np.arange(n) >= split_idx

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_true[train_mask], y_true[test_mask]

        logger.info(
            f"Walk-forward split: train={len(y_train)}, test={len(y_test)} "
            f"(ratio={len(y_train)/len(y_test):.1f})"
        )

        # Train logistic regression
        try:
            self.lr_model = LogisticRegression(
                C=self.regularization_c,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver='lbfgs',
                n_jobs=-1,
            )
            self.lr_model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Failed to train logistic regression: {e}")
            self.is_fitted = False
            return {'status': 'failed', 'reason': str(e)}

        # Get raw predictions from LR
        y_prob_train = self.lr_model.predict_proba(X_train)[:, 1]
        y_prob_test = self.lr_model.predict_proba(X_test)[:, 1]

        # Train isotonic regression for calibration (on test set to avoid overfitting)
        try:
            self.iso_reg = IsotonicRegression(out_of_bounds='clip')
            self.iso_reg.fit(y_prob_test, y_test)

            # Compute calibration bias: mean(predicted) - mean(actual)
            y_prob_cal_test = self.iso_reg.predict(y_prob_test)
            self.cal_bias = float(np.mean(y_prob_cal_test) - np.mean(y_test))

            # Calibrate both sets
            y_prob_train_cal = self.iso_reg.predict(y_prob_train)
            y_prob_test_cal = self.iso_reg.predict(y_prob_test)
        except Exception as e:
            logger.warning(f"Isotonic calibration failed: {e}. Skipping calibration.")
            self.iso_reg = None
            y_prob_train_cal = y_prob_train
            y_prob_test_cal = y_prob_test
            self.cal_bias = 0.0

        # Compute diagnostics on training set
        y_pred_train = (y_prob_train_cal > 0.5).astype(int)
        self.train_accuracy = accuracy_score(y_train, y_pred_train)
        self.train_auc = roc_auc_score(y_train, y_prob_train_cal)
        self.train_log_loss = log_loss(y_train, y_prob_train_cal)

        # Feature importance from LR coefficients
        feature_names = StackerFeatures.feature_names() if self.use_agreement_features else \
                       ['xgb_prob', 'lstm_prob', 'tft_prob', 'tabnet_prob', 'sentiment_prob']
        coefs = self.lr_model.coef_[0]
        self.feature_importance = dict(zip(feature_names[:len(coefs)], coefs))

        # Normalize feature importance (absolute values, sum to 1)
        abs_coefs = np.abs(coefs)
        if abs_coefs.sum() > 0:
            normalized_importance = abs_coefs / abs_coefs.sum()
            self.feature_importance = {
                name: float(imp)
                for name, imp in zip(feature_names[:len(coefs)], normalized_importance)
            }

        self.is_fitted = True

        # Detailed logging
        coef_str = ", ".join(
            f"{name}={c:.4f}"
            for name, c in list(self.feature_importance.items())[:5]
        )
        logger.info(
            f"EnsembleStacker fitted — "
            f"train_acc={self.train_accuracy:.4f}, "
            f"train_auc={self.train_auc:.4f}, "
            f"train_loss={self.train_log_loss:.4f}, "
            f"cal_bias={self.cal_bias:.4f}, "
            f"top_features: {coef_str}"
        )

        # Also log test set performance for reference
        y_pred_test = (y_prob_test_cal > 0.5).astype(int)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_auc = roc_auc_score(y_test, y_prob_test_cal)
        logger.info(f"Test performance: acc={test_acc:.4f}, auc={test_auc:.4f}")

        return {
            'status': 'success',
            'n_samples': n,
            'train_accuracy': float(self.train_accuracy),
            'train_auc': float(self.train_auc),
            'train_log_loss': float(self.train_log_loss),
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'feature_importance': self.feature_importance,
            'coefficients': dict(zip(feature_names[:len(coefs)], coefs.tolist())),
            'intercept': float(self.lr_model.intercept_[0]),
            'n_agreement_features': 10 if self.use_agreement_features else 0,
        }

    def predict_proba(
        self,
        model_predictions: Dict[str, Union[float, np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """
        Predict probability of UP move.

        Args:
            model_predictions: Dict with model predictions (scalar or array)

        Returns:
            Calibrated probability (scalar or array)
        """
        if not self.is_fitted or self.lr_model is None:
            # Fallback to weighted average
            return self._manual_weighted_average(model_predictions)

        features = self._compute_features(model_predictions)

        # Convert StackerFeatures to array if needed
        if isinstance(features, StackerFeatures):
            X = features.to_array().reshape(1, -1)
            squeeze = True
        else:
            X = features
            squeeze = (X.ndim == 1)
            if squeeze:
                X = X.reshape(1, -1)

        y_prob = self.lr_model.predict_proba(X)[:, 1]

        # Apply isotonic calibration
        if self.iso_reg is not None:
            y_prob = self.iso_reg.predict(y_prob)

        # Debias
        y_prob = y_prob - self.cal_bias
        y_prob = np.clip(y_prob, 0, 1)

        if squeeze:
            return float(y_prob[0])
        return y_prob

    def get_confidence(
        self,
        model_predictions: Dict[str, Union[float, np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """
        Compute confidence score based on stacker output + model agreement.

        Combines:
        1. Stacker probability (how sure the meta-learner is)
        2. Model agreement (how much models agree)
        3. Spread (inverse of uncertainty)

        Args:
            model_predictions: Dict with model predictions

        Returns:
            Confidence score (0-1)
        """
        features = self._compute_features(model_predictions)
        is_scalar = isinstance(features, StackerFeatures)

        if is_scalar:
            prob = self.predict_proba(model_predictions)
            spread = features.spread
            n_agree = features.xgb_lstm_agree + features.xgb_tft_agree + \
                     features.xgb_tabnet_agree + features.xgb_sentiment_agree + \
                     features.lstm_tft_agree + features.lstm_tabnet_agree + \
                     features.lstm_sentiment_agree + features.tft_tabnet_agree + \
                     features.tft_sentiment_agree + features.tabnet_sentiment_agree
            n_pairs = 10

            # Agreement ratio: how many pairs agree?
            agreement_ratio = n_agree / n_pairs if n_pairs > 0 else 0.5

            # Distance from 0.5
            prob_conviction = abs(prob - 0.5) * 2
            prob_conviction = np.clip(prob_conviction, 0, 1)

            # Spread contribution: lower spread = higher confidence (models agree on magnitude)
            # Normalize spread from [0, 1] to [0, 1] confidence
            spread_confidence = 1.0 - spread

            # Combined confidence
            confidence = (
                0.4 * prob_conviction +      # How sure the stacker is
                0.35 * agreement_ratio +      # How much models agree
                0.25 * spread_confidence      # How similar the predictions are
            )

            return float(np.clip(confidence, 0, 1))
        else:
            # Batch version
            probs = self.predict_proba(model_predictions)
            X = self._compute_features(model_predictions)

            # Extract spread and agreement from features
            spread = X[:, -8]  # spread is 2nd to last feature (after mean, median)
            n_up = X[:, -2]
            n_down = X[:, -1]

            # Agreement ratio computed from vote counts
            # If 4+ models agree on direction, that's strong agreement
            agreement_ratio = np.maximum(n_up, n_down) / 5.0

            # Probability conviction
            prob_conviction = np.abs(probs - 0.5) * 2
            prob_conviction = np.clip(prob_conviction, 0, 1)

            # Spread confidence
            spread_confidence = 1.0 - spread

            # Combined
            confidence = (
                0.4 * prob_conviction +
                0.35 * agreement_ratio +
                0.25 * spread_confidence
            )

            return np.clip(confidence, 0, 1)

    def get_model_importance(self) -> Dict[str, float]:
        """
        Return learned importance of each model from LR coefficients.

        Returns:
            Dict mapping model names to importance scores (0-1, sums to 1)
        """
        if self.feature_importance is None:
            logger.warning("Stacker not fitted. Returning manual weights.")
            return self.manual_weights

        # Extract just the raw model predictions (first 5 features)
        model_features = {
            'xgboost': self.feature_importance.get('xgb_prob', 0),
            'lstm': self.feature_importance.get('lstm_prob', 0),
            'tft': self.feature_importance.get('tft_prob', 0),
            'tabnet': self.feature_importance.get('tabnet_prob', 0),
            'sentiment': self.feature_importance.get('sentiment_prob', 0),
        }

        # Normalize
        total = sum(abs(v) for v in model_features.values())
        if total > 0:
            model_features = {k: abs(v) / total for k, v in model_features.items()}
        else:
            model_features = self.manual_weights.copy()

        return model_features

    def generate_signal(
        self,
        model_predictions: Dict[str, Union[float, np.ndarray]],
        threshold: float = 0.15,
    ) -> Union[Dict, List[Dict]]:
        """
        Generate trading signal from stacker output.

        Args:
            model_predictions: Dict with model predictions
            threshold: Minimum confidence to generate signal (else HOLD)

        Returns:
            For scalar input: Dict with keys 'signal', 'probability', 'confidence', 'direction'
            For array input: List of such dicts
        """
        prob = self.predict_proba(model_predictions)
        confidence = self.get_confidence(model_predictions)

        is_scalar = np.isscalar(prob)

        if is_scalar:
            prob = np.atleast_1d(prob)[0]
            confidence = np.atleast_1d(confidence)[0]

            direction = "UP" if prob > 0.5 else "DOWN"

            if confidence >= threshold:
                signal = "BUY_CALL" if direction == "UP" else "BUY_PUT"
            else:
                signal = "HOLD"

            return {
                'signal': signal,
                'probability': float(prob),
                'confidence': float(confidence),
                'direction': direction,
            }
        else:
            # Batch
            results = []
            for i in range(len(prob)):
                p = prob[i]
                c = confidence[i]
                direction = "UP" if p > 0.5 else "DOWN"
                signal = "BUY_CALL" if (direction == "UP" and c >= threshold) else \
                        "BUY_PUT" if (direction == "DOWN" and c >= threshold) else "HOLD"

                results.append({
                    'signal': signal,
                    'probability': float(p),
                    'confidence': float(c),
                    'direction': direction,
                })
            return results

    def _manual_weighted_average(
        self,
        model_predictions: Dict[str, Union[float, np.ndarray]]
    ) -> Union[float, np.ndarray]:
        """Fallback: simple weighted average using manual weights."""
        probs = {
            'xgboost': np.atleast_1d(model_predictions.get('xgboost', 0.5)),
            'lstm': np.atleast_1d(model_predictions.get('lstm', 0.5)),
            'tft': np.atleast_1d(model_predictions.get('tft', 0.5)),
            'tabnet': np.atleast_1d(model_predictions.get('tabnet', 0.5)),
            'sentiment': np.atleast_1d(model_predictions.get('sentiment', 0.5)),
        }

        weighted_sum = sum(
            self.manual_weights.get(model, 0) * probs[model]
            for model in probs
        )
        total_weight = sum(self.manual_weights.values())
        result = weighted_sum / total_weight if total_weight > 0 else 0.5

        is_scalar = all(len(np.atleast_1d(v)) == 1 for v in probs.values())
        if is_scalar:
            return float(result[0])
        return result

    def save(self, symbol: str = "default") -> Path:
        """
        Save stacker to disk.

        Args:
            symbol: Identifier for stacker variant (e.g., "SPY", "QQQ")

        Returns:
            Path to saved file
        """
        filepath = self.model_dir / f"stacker_{symbol}.pkl"

        state = {
            'lr_model': self.lr_model,
            'iso_reg': self.iso_reg,
            'cal_bias': self.cal_bias,
            'is_fitted': self.is_fitted,
            'manual_weights': self.manual_weights,
            'use_agreement_features': self.use_agreement_features,
            'regularization_c': self.regularization_c,
            'feature_importance': self.feature_importance,
            'train_accuracy': self.train_accuracy,
            'train_auc': self.train_auc,
            'train_log_loss': self.train_log_loss,
        }

        joblib.dump(state, filepath)
        logger.info(f"EnsembleStacker saved to {filepath}")
        return filepath

    def load(self, symbol: str = "default") -> bool:
        """
        Load stacker from disk.

        Args:
            symbol: Identifier for stacker variant

        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = self.model_dir / f"stacker_{symbol}.pkl"

        if not filepath.exists():
            logger.warning(f"Stacker file not found: {filepath}")
            return False

        try:
            state = joblib.load(filepath)
            self.lr_model = state['lr_model']
            self.iso_reg = state['iso_reg']
            self.cal_bias = state['cal_bias']
            self.is_fitted = state['is_fitted']
            self.manual_weights = state['manual_weights']
            self.use_agreement_features = state['use_agreement_features']
            self.regularization_c = state['regularization_c']
            self.feature_importance = state['feature_importance']
            self.train_accuracy = state['train_accuracy']
            self.train_auc = state['train_auc']
            self.train_log_loss = state['train_log_loss']

            logger.info(
                f"EnsembleStacker loaded from {filepath} — "
                f"fitted={self.is_fitted}, auc={self.train_auc:.4f}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load stacker: {e}")
            return False

    def diagnostics(self) -> Dict[str, any]:
        """
        Return detailed diagnostics about stacker state.

        Returns:
            Dict with training metrics, feature importance, coefficients, etc.
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'status': 'using_manual_weights',
                'manual_weights': self.manual_weights,
            }

        coefs = self.lr_model.coef_[0] if self.lr_model is not None else None
        intercept = self.lr_model.intercept_[0] if self.lr_model is not None else None

        return {
            'is_fitted': True,
            'train_accuracy': float(self.train_accuracy) if self.train_accuracy else None,
            'train_auc': float(self.train_auc) if self.train_auc else None,
            'train_log_loss': float(self.train_log_loss) if self.train_log_loss else None,
            'calibration_bias': float(self.cal_bias),
            'has_isotonic_calibration': self.iso_reg is not None,
            'feature_importance': self.feature_importance,
            'coefficients': dict(zip(
                StackerFeatures.feature_names() if self.use_agreement_features else
                ['xgb', 'lstm', 'tft', 'tabnet', 'sentiment'],
                coefs.tolist() if coefs is not None else []
            )),
            'intercept': float(intercept) if intercept is not None else None,
            'regularization_c': self.regularization_c,
            'use_agreement_features': self.use_agreement_features,
        }
