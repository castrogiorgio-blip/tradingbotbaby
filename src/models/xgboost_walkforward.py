"""
Walk-Forward Cross-Validation for XGBoost Hyperparameter Tuning.

This module implements expanding-window walk-forward validation to properly tune
XGBoost hyperparameters for time-series trading data. Unlike static k-fold CV,
walk-forward validation ensures we never train on future data while evaluating
candidate hyperparameter sets.

Key principles:
  1. Train on expanding windows: [month 1..N], validate on [month N+1]
  2. For each window, test candidate parameter sets
  3. Select best params by average validation accuracy across all folds
  4. Stores per-fold metrics for detailed analysis
  5. Proper handling of feature engineering (exclude raw prices)

Usage:
    from src.models.xgboost_walkforward import WalkForwardXGBoost

    # Initialize with feature data
    wf = WalkForwardXGBoost()

    # Run optimization
    best_params, fold_results = wf.walk_forward_optimize(X, y, dates)

    # Train final model on all data with best params
    wf.train_final_model(X, y, best_params)

    # Make predictions
    predictions = wf.predict_proba(X_test)

    # Review what was tested
    report = wf.get_optimization_report()
    print(report)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
from itertools import product
from loguru import logger
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class WalkForwardXGBoost:
    """
    Walk-forward cross-validation optimizer for XGBoost hyperparameters.

    This class:
    - Splits time-series data into expanding-window folds
    - Tests candidate hyperparameter sets on each fold
    - Selects best params by average validation performance
    - Trains final model on all data with best params
    - Tracks and reports per-fold metrics
    """

    # Columns that are NOT features (targets + raw prices that leak information)
    EXCLUDE_COLUMNS = [
        # Targets
        "target_return", "target_direction",
        "target_return_5d", "target_direction_5d",
        # Raw price data (absolute values don't generalize across time)
        "open", "high", "low", "close", "volume",
        "trade_count", "vwap",
        # Absolute price indicators (scale changes over time, causing leakage)
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
        "bb_upper", "bb_middle", "bb_lower",
        "ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conv",
        "high_52w", "low_52w",
        "obv", "ad_line",
        "volume_sma_20",
        "atr_7", "atr_14", "atr_21",
        "macd", "macd_signal", "macd_histogram",
    ]

    # Default hyperparameter grid for walk-forward search
    DEFAULT_PARAM_GRID = {
        "max_depth": [4, 5, 6],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "gamma": [0, 0.1, 0.3],
        "reg_lambda": [1.0, 2.0],
        "min_child_weight": [3, 5, 10],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
    }

    def __init__(self, param_grid: Dict[str, List[Any]] = None, random_state: int = 42):
        """
        Initialize walk-forward optimizer.

        Args:
            param_grid: Dict mapping param names to lists of values.
                        If None, uses DEFAULT_PARAM_GRID.
            random_state: For reproducibility
        """
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.random_state = random_state

        self.best_params = None
        self.best_score = -np.inf
        self.fold_results = []
        self.optimization_report = None

        self.final_model = None
        self.feature_names = None
        self.is_trained = False

        logger.info(f"WalkForwardXGBoost initialized with {len(self._generate_param_combinations())} param sets to test")

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        param_combinations = []
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            param_combinations.append(param_dict)

        return param_combinations

    def prepare_features(self, df: pd.DataFrame, target_col: str = "target_direction"):
        """
        Extract features and target from a DataFrame.

        Args:
            df: DataFrame from FeatureBuilder (includes targets and raw prices)
            target_col: Name of target column

        Returns:
            X (DataFrame), y (Series), dates (Series)
        """
        df_clean = df.dropna(subset=[target_col]).copy()

        # Identify feature columns (exclude raw prices and targets)
        feature_cols = [
            col for col in df_clean.columns
            if col not in self.EXCLUDE_COLUMNS
        ]
        self.feature_names = feature_cols

        X = df_clean[feature_cols]
        y = df_clean[target_col]
        dates = df_clean.index

        # Fill remaining NaN (some indicators have NaN early)
        X = X.fillna(0)

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        return X, y, dates

    def _split_by_months(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.DatetimeIndex,
        min_train_months: int = 6,
        val_months: int = 1,
    ) -> List[Tuple[slice, slice]]:
        """
        Split data into expanding-window folds based on months.

        Args:
            X, y, dates: Feature matrix, target, and datetime index
            min_train_months: Minimum months of training data per fold
            val_months: Number of months to use for validation

        Returns:
            List of (train_indices, val_indices) for each fold
        """
        # Group by year-month
        df_dates = pd.DataFrame({"date": dates})
        df_dates["year_month"] = df_dates["date"].dt.to_period("M")
        unique_months = df_dates["year_month"].unique()

        logger.info(f"Data spans {len(unique_months)} months")

        folds = []

        # Start with min_train_months, then expand window by 1 month each fold
        for i in range(len(unique_months) - min_train_months - val_months + 1):
            train_end_month_idx = min_train_months + i
            val_start_month_idx = train_end_month_idx
            val_end_month_idx = val_start_month_idx + val_months

            # Get all months up to train_end
            train_months = unique_months[:train_end_month_idx]
            val_months_list = unique_months[val_start_month_idx:val_end_month_idx]

            train_mask = df_dates["year_month"].isin(train_months)
            val_mask = df_dates["year_month"].isin(val_months_list)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]

            if len(train_indices) > 0 and len(val_indices) > 0:
                folds.append((train_indices, val_indices))
                logger.info(
                    f"Fold {len(folds)}: train {len(train_indices)} samples "
                    f"({train_months[0]}-{train_months[-1]}), "
                    f"val {len(val_indices)} samples ({val_months_list[0]})"
                )

        return folds

    def walk_forward_optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.DatetimeIndex,
        min_train_months: int = 6,
        val_months: int = 1,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run walk-forward optimization to find best hyperparameters.

        For each expanding-window fold:
          1. Test all candidate param sets on the validation fold
          2. Track accuracy for each param set
          3. After all folds, select params with best average accuracy

        Args:
            X: Feature matrix
            y: Target vector
            dates: DatetimeIndex for time-series splitting
            min_train_months: Minimum months of training data
            val_months: Months of validation data per fold

        Returns:
            best_params (Dict): Best hyperparameters found
            fold_results (List[Dict]): Per-fold results for analysis
        """
        logger.info("=" * 70)
        logger.info("Starting walk-forward optimization")
        logger.info("=" * 70)

        # Create expanding-window folds
        folds = self._split_by_months(X, y, dates, min_train_months, val_months)
        logger.info(f"Created {len(folds)} expanding-window folds\n")

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations()
        logger.info(f"Testing {len(param_combinations)} hyperparameter sets\n")

        # Track accuracy for each param set across all folds
        param_scores = {i: [] for i in range(len(param_combinations))}

        # Iterate through expanding windows
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            logger.info(f"\n--- FOLD {fold_idx + 1}/{len(folds)} ---")

            X_train = X.iloc[train_indices].fillna(0)
            y_train = y.iloc[train_indices]
            X_val = X.iloc[val_indices].fillna(0)
            y_val = y.iloc[val_indices]

            fold_best_acc = -np.inf
            fold_best_params_idx = None

            # Test each parameter set
            for param_idx, params in enumerate(param_combinations):
                # Build base params with common settings
                full_params = {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "random_state": self.random_state,
                    "n_jobs": -1,
                    "verbosity": 0,
                }
                full_params.update(params)

                # Train model with these params
                try:
                    model = XGBClassifier(**full_params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )

                    # Evaluate on validation set
                    y_pred = model.predict(X_val)
                    acc = accuracy_score(y_val, y_pred)
                    param_scores[param_idx].append(acc)

                    if acc > fold_best_acc:
                        fold_best_acc = acc
                        fold_best_params_idx = param_idx

                except Exception as e:
                    logger.warning(f"Param set {param_idx} failed: {e}")
                    param_scores[param_idx].append(0.0)

            if fold_best_params_idx is not None:
                best_params_this_fold = param_combinations[fold_best_params_idx]
                logger.info(
                    f"Fold {fold_idx + 1} best: {fold_best_acc:.4f} "
                    f"(max_depth={best_params_this_fold.get('max_depth')}, "
                    f"lr={best_params_this_fold.get('learning_rate')})"
                )

            self.fold_results.append({
                "fold_idx": fold_idx,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "best_fold_acc": fold_best_acc,
            })

        # Compute average score for each parameter set across all folds
        avg_scores = {
            param_idx: np.mean(scores)
            for param_idx, scores in param_scores.items()
        }

        # Find best parameter set
        best_param_idx = max(avg_scores, key=avg_scores.get)
        self.best_params = param_combinations[best_param_idx]
        self.best_score = avg_scores[best_param_idx]

        logger.info("\n" + "=" * 70)
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nBest hyperparameters (avg accuracy: {self.best_score:.4f}):")
        for key, value in sorted(self.best_params.items()):
            logger.info(f"  {key}: {value}")

        # Store detailed results
        self.optimization_report = {
            "best_params": self.best_params,
            "best_avg_accuracy": self.best_score,
            "param_grid_size": len(param_combinations),
            "num_folds": len(folds),
            "param_scores": avg_scores,
            "fold_results": self.fold_results,
        }

        return self.best_params, self.fold_results

    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        best_params: Dict[str, Any] = None,
    ) -> XGBClassifier:
        """
        Train final model on all data with best hyperparameters.

        Args:
            X: Feature matrix
            y: Target vector
            best_params: Hyperparameters to use. If None, uses self.best_params.

        Returns:
            Trained XGBClassifier
        """
        if best_params is None:
            if self.best_params is None:
                raise ValueError("Must call walk_forward_optimize first or provide best_params")
            best_params = self.best_params

        logger.info("\n" + "=" * 70)
        logger.info("Training final model on all data")
        logger.info("=" * 70)

        # Build full params
        full_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": self.random_state,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
        }
        full_params.update(best_params)

        logger.info(f"Training on {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Parameters: {best_params}")

        self.final_model = XGBClassifier(**full_params)
        X_filled = X.fillna(0)

        # Train without eval_set (using all data)
        self.final_model.fit(X_filled, y, verbose=False)
        self.is_trained = True

        # Log training performance (for reference only)
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = self.final_model.predict(X_filled)
        train_acc = accuracy_score(y, y_pred)
        y_prob = self.final_model.predict_proba(X_filled)[:, 1]
        train_auc = roc_auc_score(y, y_prob)

        logger.info(f"Final model training accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")

        return self.final_model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict direction (0 or 1)."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        X_filled = X.fillna(0)
        return self.final_model.predict(X_filled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of class 1 (price up).

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities for class 1
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        X_filled = X.fillna(0)
        return self.final_model.predict_proba(X_filled)[:, 1]

    def get_optimization_report(self) -> str:
        """
        Generate human-readable summary of optimization results.

        Returns:
            Formatted string with optimization details
        """
        if self.optimization_report is None:
            return "No optimization has been run yet"

        report = self.optimization_report

        lines = [
            "",
            "=" * 70,
            "WALK-FORWARD OPTIMIZATION REPORT",
            "=" * 70,
            "",
            f"Parameter Grid Tested: {report['param_grid_size']} combinations",
            f"Number of Folds: {report['num_folds']}",
            f"Best Average Validation Accuracy: {report['best_avg_accuracy']:.4f}",
            "",
            "Best Hyperparameters Found:",
            "-" * 70,
        ]

        for key, value in sorted(report['best_params'].items()):
            lines.append(f"  {key:<25} {value}")

        lines.extend([
            "",
            "Per-Fold Results:",
            "-" * 70,
        ])

        for fold_result in report['fold_results']:
            fold_idx = fold_result['fold_idx']
            train_size = fold_result['train_size']
            val_size = fold_result['val_size']
            best_acc = fold_result['best_fold_acc']

            lines.append(
                f"  Fold {fold_idx + 1}: "
                f"train={train_size} samples, "
                f"val={val_size} samples, "
                f"best_acc={best_acc:.4f}"
            )

        lines.extend([
            "",
            "Top 5 Hyperparameter Sets (by avg accuracy):",
            "-" * 70,
        ])

        param_scores = report['param_scores']
        sorted_params = sorted(
            param_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        param_combinations = self._generate_param_combinations()
        for rank, (param_idx, score) in enumerate(sorted_params[:5], 1):
            params = param_combinations[param_idx]
            lines.append(f"  {rank}. Accuracy: {score:.4f}")
            lines.append(f"     {params}")

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importances from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = self.final_model.feature_importances_
        feature_imp = pd.DataFrame({
            "feature": self.feature_names or [f"f_{i}" for i in range(len(importance))],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return feature_imp.head(top_n)


if __name__ == "__main__":
    # Example usage (for testing)
    logger.info("Walk-forward XGBoost module loaded successfully")
