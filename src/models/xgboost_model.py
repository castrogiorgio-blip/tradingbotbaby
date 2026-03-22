"""
XGBoost Classifier — predicts next-day price direction from tabular features.

This is the workhorse model. Gradient boosting consistently outperforms
deep learning on structured/tabular data in quantitative finance.

Features: technical indicators, economic data, sentiment scores, calendar features.
Target: 1 (price goes up) or 0 (price goes down).

Usage:
    from src.models.xgboost_model import XGBoostModel
    model = XGBoostModel()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score,
)
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, DATA_DIR


class XGBoostModel:
    """XGBoost-based direction classifier for stock price prediction."""

    def __init__(self, params: dict = None):
        """
        Initialize with parameters from config or override.

        Args:
            params: Optional dict of XGBoost parameters. If None, uses config.
        """
        settings = get_settings()
        default_params = settings.get("models", {}).get("xgboost", {})

        self.params = {
            "n_estimators": default_params.get("n_estimators", 300),
            "max_depth": 3,                    # V2: Much shallower trees to prevent overfitting
            "learning_rate": 0.01,             # V2: Slower learning rate
            "min_child_weight": 10,            # V2: Require more samples per leaf
            "subsample": 0.6,                  # V2: Use only 60% of data per tree
            "colsample_bytree": 0.5,           # V2: Use only 50% of features per tree
            "gamma": 1.0,                      # V2: Higher min loss reduction
            "reg_alpha": 1.0,                  # V2: Stronger L1 regularization
            "reg_lambda": 5.0,                 # V2: Stronger L2 regularization
            "objective": "binary:logistic",
            "eval_metric": "auc",              # V2: Optimize for AUC not logloss
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30,       # V2: Stop if no improvement for 30 rounds
        }
        if params:
            self.params.update(params)

        self.model = XGBClassifier(**self.params)
        self.feature_names = None
        self.is_trained = False
        self.model_dir = DATA_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"XGBoostModel initialized (n_estimators={self.params['n_estimators']})")

    # ---- Columns that are NOT features (targets + raw prices that leak) ----
    EXCLUDE_COLUMNS = [
        # Targets
        "target_return", "target_direction",
        "target_return_5d", "target_direction_5d",
        # Raw price data (absolute values don't generalize across time)
        "open", "high", "low", "close", "volume",
        "trade_count", "vwap",
        # Absolute price indicators (these change with price level, causing leakage)
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
        "bb_upper", "bb_middle", "bb_lower",
        "ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conv",
        "high_52w", "low_52w",
        "obv", "ad_line",  # absolute volume-based (scale changes over time)
        "volume_sma_20",
        "atr_7", "atr_14", "atr_21",  # absolute ATR (keep _pct versions)
        "macd", "macd_signal", "macd_histogram",  # absolute MACD (keep crossover)
    ]

    def prepare_features(self, df: pd.DataFrame, target_col: str = "target_direction"):
        """
        Separate features (X) and target (y) from a feature DataFrame.

        Args:
            df: Feature DataFrame from FeatureBuilder
            target_col: Name of the target column

        Returns:
            X (DataFrame), y (Series)
        """
        # Drop rows with NaN target
        df_clean = df.dropna(subset=[target_col]).copy()

        # Identify feature columns
        feature_cols = [
            col for col in df_clean.columns
            if col not in self.EXCLUDE_COLUMNS
        ]
        self.feature_names = feature_cols

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        # Fill remaining NaN values with 0 (some indicators have NaN for early periods)
        X = X.fillna(0)

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series, eval_set: tuple = None):
        """
        Train the XGBoost classifier.

        Args:
            X: Feature matrix
            y: Target vector (0/1)
            eval_set: Optional (X_val, y_val) for early stopping
        """
        fit_params = {"verbose": False}
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]

        logger.info(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")
        X_filled = X.fillna(0)
        self.model.fit(X_filled, y, **fit_params)
        self.is_trained = True

        # Log training accuracy (should NOT be 100% anymore with regularization)
        train_pred = self.model.predict(X_filled)
        train_acc = accuracy_score(y, train_pred)
        train_auc = roc_auc_score(y, self.model.predict_proba(X_filled)[:, 1])
        logger.info(f"Training accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
        if train_acc > 0.95:
            logger.warning("⚠️ Training accuracy suspiciously high — possible overfitting!")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict direction (0 or 1)."""
        X = X.fillna(0)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of each class. Returns P(class=1) — probability of UP."""
        X = X.fillna(0)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluate model performance.

        Returns:
            Dict with accuracy, precision, recall, f1, auc metrics
        """
        X = X.fillna(0)
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_prob),
            "n_samples": len(y),
            "class_balance": {
                "up": int((y == 1).sum()),
                "down": int((y == 0).sum()),
            },
        }

        logger.info(
            f"Evaluation — Acc: {metrics['accuracy']:.4f}, "
            f"Prec: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"AUC: {metrics['auc_roc']:.4f}"
        )
        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """
        Time-series cross-validation (walk-forward).

        Unlike regular k-fold, TimeSeriesSplit ensures we never train on future data.

        Returns:
            Dict with mean and std of each metric across folds
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train a fresh model for each fold (with eval_set for early stopping)
            fold_model = XGBClassifier(**self.params)
            fold_model.fit(
                X_train.fillna(0), y_train,
                eval_set=[(X_val.fillna(0), y_val)],
                verbose=False,
            )

            y_pred = fold_model.predict(X_val.fillna(0))
            y_prob = fold_model.predict_proba(X_val.fillna(0))[:, 1]

            fold_metrics.append({
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "auc_roc": roc_auc_score(y_val, y_prob),
            })

            logger.info(
                f"  Fold {fold+1}/{n_splits} — "
                f"Acc: {fold_metrics[-1]['accuracy']:.4f}, "
                f"AUC: {fold_metrics[-1]['auc_roc']:.4f}"
            )

        # Aggregate across folds
        result = {}
        for metric in fold_metrics[0].keys():
            values = [f[metric] for f in fold_metrics]
            result[f"{metric}_mean"] = np.mean(values)
            result[f"{metric}_std"] = np.std(values)

        logger.info(
            f"CV Results — Acc: {result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}, "
            f"AUC: {result['auc_roc_mean']:.4f}±{result['auc_roc_std']:.4f}"
        )
        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importances from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            "feature": self.feature_names or [f"f_{i}" for i in range(len(importance))],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return feature_imp.head(top_n)

    def save(self, symbol: str = "default"):
        """Save model to disk."""
        filepath = self.model_dir / f"xgboost_{symbol}.pkl"
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
        }, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath

    def load(self, symbol: str = "default"):
        """Load model from disk."""
        filepath = self.model_dir / f"xgboost_{symbol}.pkl"
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.params = data["params"]
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
