"""
LSTM Model — captures sequential/temporal patterns in price data.

While XGBoost handles tabular features well, LSTM (Long Short-Term Memory)
networks can learn from the *sequence* of price movements over time.

Input: Rolling windows of OHLCV + selected indicators (60-day sequences).
Output: Probability of next-day price increase.

Usage:
    from src.models.lstm_model import LSTMModel
    model = LSTMModel(n_features=20)
    model.train(X_train_seq, y_train)
    probabilities = model.predict_proba(X_test_seq)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, DATA_DIR


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network for binary classification."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, n_features)
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_hidden = lstm_out[:, -1, :]
        output = self.classifier(last_hidden)
        return output.squeeze(-1)


class LSTMModel:
    """LSTM-based sequential model for price direction prediction."""

    # Columns to use as LSTM input features (price + key indicators)
    SEQUENCE_FEATURES = [
        "close", "volume", "return_1d",
        "rsi_14", "macd", "macd_signal",
        "bb_pct", "bb_width",
        "adx_14", "atr_14_pct",
        "stoch_k", "stoch_d",
        "obv", "mfi_14", "volume_ratio",
        "volatility_10d", "daily_range",
        "ema_10", "ema_20", "sma_50",
    ]

    def __init__(self, n_features: int = None, params: dict = None):
        """
        Args:
            n_features: Number of input features per timestep. Auto-detected if None.
            params: Override LSTM hyperparameters from config.
        """
        settings = get_settings()
        default_params = settings.get("models", {}).get("lstm", {})

        self.sequence_length = default_params.get("sequence_length", 60)
        self.hidden_size = default_params.get("hidden_size", 128)
        self.num_layers = default_params.get("num_layers", 2)
        self.dropout = default_params.get("dropout", 0.2)
        self.epochs = default_params.get("epochs", 50)
        self.batch_size = default_params.get("batch_size", 32)
        self.learning_rate = 0.001

        if params:
            for k, v in params.items():
                setattr(self, k, v)

        self.n_features = n_features or len(self.SEQUENCE_FEATURES)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.network = None  # Built on first train
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_dir = DATA_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"LSTMModel initialized (seq_len={self.sequence_length}, "
            f"hidden={self.hidden_size}, device={self.device})"
        )

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
        fit_scaler: bool = False,
    ) -> tuple:
        """
        Convert a feature DataFrame into sequences for LSTM input.

        Args:
            df: Feature DataFrame with OHLCV + indicators
            target_col: Target column name
            fit_scaler: If True, fit the scaler (only for training data)

        Returns:
            X_seq (np.array of shape [n_samples, seq_length, n_features]),
            y (np.array of shape [n_samples])
        """
        # Select available features
        available = [f for f in self.SEQUENCE_FEATURES if f in df.columns]
        if len(available) < 5:
            logger.warning(f"Only {len(available)} LSTM features available, need at least 5")
            return np.array([]), np.array([])

        self.n_features = len(available)

        # Extract feature matrix and target
        df_clean = df.dropna(subset=[target_col]).copy()
        feature_data = df_clean[available].fillna(0).values
        target_data = df_clean[target_col].values

        # Scale features
        if fit_scaler:
            feature_data = self.scaler.fit_transform(feature_data)
        else:
            feature_data = self.scaler.transform(feature_data)

        # Create rolling windows
        X_sequences = []
        y_labels = []

        for i in range(self.sequence_length, len(feature_data)):
            X_sequences.append(feature_data[i - self.sequence_length : i])
            y_labels.append(target_data[i])

        X_seq = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_labels, dtype=np.float32)

        logger.info(f"Created {len(X_seq)} sequences of shape ({self.sequence_length}, {self.n_features})")
        return X_seq, y

    def _build_network(self):
        """Initialize the LSTM network."""
        self.network = LSTMNetwork(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def train(self, X_seq: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the LSTM model.

        Args:
            X_seq: Training sequences (n_samples, seq_len, n_features)
            y: Training labels (n_samples,)
            X_val: Optional validation sequences
            y_val: Optional validation labels
        """
        # Set seeds for reproducibility — use _seed attribute for multi-run averaging
        seed = getattr(self, '_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        self._build_network()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        logger.info(f"Training LSTM for {self.epochs} epochs on {len(X_seq)} samples")

        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 10

        for epoch in range(self.epochs):
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                predictions = self.network(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                logger.info(msg)

        self.is_trained = True
        logger.info("LSTM training complete")

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute BCELoss on a dataset."""
        self.network.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            preds = self.network(X_t)
            loss = nn.BCELoss()(preds, y_t)
        return loss.item()

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Predict probability of price going UP.

        Args:
            X_seq: Sequences (n_samples, seq_len, n_features)

        Returns:
            Array of probabilities (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            probs = self.network(X_tensor).cpu().numpy()
        return probs

    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict direction (0 or 1) with given threshold."""
        probs = self.predict_proba(X_seq)
        return (probs > threshold).astype(int)

    def evaluate(self, X_seq: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        y_pred = self.predict(X_seq)
        y_prob = self.predict_proba(X_seq)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "auc_roc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0,
            "f1": f1_score(y, y_pred, zero_division=0),
            "n_samples": len(y),
        }

        logger.info(
            f"LSTM Eval — Acc: {metrics['accuracy']:.4f}, "
            f"AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}"
        )
        return metrics

    def save(self, symbol: str = "default"):
        """Save model + scaler to disk."""
        filepath = self.model_dir / f"lstm_{symbol}.pt"
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "scaler": self.scaler,
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
        }, filepath)
        logger.info(f"LSTM model saved to {filepath}")

    def load(self, symbol: str = "default"):
        """Load model + scaler from disk."""
        filepath = self.model_dir / f"lstm_{symbol}.pt"
        checkpoint = torch.load(filepath, map_location=self.device)

        self.n_features = checkpoint["n_features"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.sequence_length = checkpoint["sequence_length"]
        self.scaler = checkpoint["scaler"]

        self._build_network()
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = True
        logger.info(f"LSTM model loaded from {filepath}")
