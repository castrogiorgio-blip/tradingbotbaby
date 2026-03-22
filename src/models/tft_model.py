"""
Temporal Fusion Transformer (TFT) — state-of-the-art time series model.

TFT combines several powerful concepts:
  1. Variable Selection Networks — automatically learns which features matter
  2. Temporal Self-Attention — captures long-range dependencies in price history
  3. Gated Residual Networks — controls information flow, prevents overfitting
  4. Multi-horizon forecasting — can predict multiple steps ahead

This is a simplified but effective implementation focusing on next-day
direction prediction, matching the existing model interface.

Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon
Time Series Forecasting" (Lim et al., 2021)

Usage:
    from src.models.tft_model import TFTModel
    model = TFTModel()
    X_seq, y = model.prepare_sequences(df, fit_scaler=True)
    model.train(X_seq, y, X_val_seq, y_val)
    probs = model.predict_proba(X_test_seq)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, DATA_DIR


# ── Building blocks ──────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    GRN: the core building block of TFT.
    Applies ELU activation + gating to control information flow.
    """
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1, context_size=None):
        super().__init__()
        output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate_fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # Skip connection with projection if dimensions differ
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = None

    def forward(self, x, context=None):
        residual = x
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)

        hidden = F.elu(self.fc1(x))
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.dropout(hidden)

        # Gated linear unit
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate_fc(hidden))
        output = gate * output

        # Add & Norm
        return self.layer_norm(output + residual)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: learns which features are most important at each timestep.
    Outputs softmax-weighted combination of individual feature embeddings.
    """
    def __init__(self, n_features, hidden_size, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Per-feature GRNs (process each feature independently)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(n_features)
        ])

        # Variable selection weights
        self.weight_grn = GatedResidualNetwork(
            n_features, hidden_size, n_features, dropout
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        returns: (batch, seq_len, hidden_size)
        """
        batch, seq_len, n_feat = x.shape

        # Process each feature through its own GRN
        feature_outputs = []
        for i in range(self.n_features):
            feat_i = x[:, :, i:i+1]  # (batch, seq_len, 1)
            feat_i_flat = feat_i.reshape(-1, 1)
            processed = self.feature_grns[i](feat_i_flat)
            processed = processed.reshape(batch, seq_len, self.hidden_size)
            feature_outputs.append(processed)

        # Stack: (batch, seq_len, n_features, hidden_size)
        stacked = torch.stack(feature_outputs, dim=2)

        # Compute selection weights: (batch, seq_len, n_features)
        x_flat = x.reshape(-1, n_feat)
        weights = self.weight_grn(x_flat)
        weights = self.softmax(weights)
        weights = weights.reshape(batch, seq_len, n_feat, 1)

        # Weighted combination: (batch, seq_len, hidden_size)
        combined = (stacked * weights).sum(dim=2)
        return combined


class TemporalAttention(nn.Module):
    """
    Interpretable multi-head attention over time steps.
    Unlike standard transformers, this uses a single attention head
    for interpretability (we can see which past days matter most).
    """
    def __init__(self, hidden_size, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_size)
        """
        batch, seq_len, d = x.shape
        residual = x

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal mask: position i can only attend to positions <= i
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d)
        attn_output = self.out_proj(attn_output)

        return self.layer_norm(attn_output + residual)


class TFTNetwork(nn.Module):
    """
    Simplified Temporal Fusion Transformer for binary classification.
    """
    def __init__(self, n_features, hidden_size=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        # Variable Selection
        self.vsn = VariableSelectionNetwork(n_features, hidden_size, dropout)

        # Temporal processing: LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        # Post-LSTM gate
        self.post_lstm_gate = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            TemporalAttention(hidden_size, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Post-attention GRNs
        self.post_attn_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        returns: (batch,) — probability of price going up
        """
        # 1. Variable selection: learn which features matter
        selected = self.vsn(x)  # (batch, seq_len, hidden)

        # 2. LSTM temporal encoding
        lstm_out, _ = self.lstm(selected)
        gated = self.post_lstm_gate(lstm_out.reshape(-1, self.hidden_size))
        gated = gated.reshape(lstm_out.shape)

        # 3. Self-attention over time
        attn_out = gated
        for attn_layer, grn in zip(self.attention_layers, self.post_attn_grns):
            attn_out = attn_layer(attn_out)
            flat = attn_out.reshape(-1, self.hidden_size)
            attn_out = grn(flat).reshape(attn_out.shape)

        # 4. Take the last timestep and classify
        last_step = attn_out[:, -1, :]
        prob = self.classifier(last_step).squeeze(-1)
        return prob


# ── Model wrapper ────────────────────────────────────────────

class TFTModel:
    """TFT-based model for price direction prediction."""

    # Same features as LSTM for compatibility, plus a few extras
    SEQUENCE_FEATURES = [
        "close", "volume", "return_1d",
        "rsi_14", "macd", "macd_signal",
        "bb_pct", "bb_width",
        "adx_14", "atr_14_pct",
        "stoch_k", "stoch_d",
        "obv", "mfi_14", "volume_ratio",
        "volatility_10d", "daily_range",
        "ema_10", "ema_20", "sma_50",
        # Extra features TFT can handle
        "return_5d", "return_10d",
        "rsi_7", "cci_20",
        "williams_r", "trix",
    ]

    def __init__(self, n_features: int = None, params: dict = None):
        settings = get_settings()
        default_params = settings.get("models", {}).get("tft", {})

        self.sequence_length = default_params.get("sequence_length", 60)
        self.hidden_size = default_params.get("hidden_size", 64)
        self.n_heads = default_params.get("n_heads", 4)
        self.n_layers = default_params.get("n_layers", 2)
        self.dropout = default_params.get("dropout", 0.2)
        self.epochs = default_params.get("epochs", 60)
        self.batch_size = default_params.get("batch_size", 32)
        self.learning_rate = default_params.get("learning_rate", 0.0005)

        if params:
            for k, v in params.items():
                setattr(self, k, v)

        self.n_features = n_features or len(self.SEQUENCE_FEATURES)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.network = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_dir = DATA_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"TFTModel initialized (seq_len={self.sequence_length}, "
            f"hidden={self.hidden_size}, heads={self.n_heads}, device={self.device})"
        )

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction",
        fit_scaler: bool = False,
    ) -> tuple:
        """Convert feature DataFrame into sequences for TFT input."""
        available = [f for f in self.SEQUENCE_FEATURES if f in df.columns]
        if len(available) < 5:
            logger.warning(f"Only {len(available)} TFT features available, need at least 5")
            return np.array([]), np.array([])

        self.n_features = len(available)

        df_clean = df.dropna(subset=[target_col]).copy()
        feature_data = df_clean[available].fillna(0).values
        target_data = df_clean[target_col].values

        if fit_scaler:
            feature_data = self.scaler.fit_transform(feature_data)
        else:
            feature_data = self.scaler.transform(feature_data)

        X_sequences = []
        y_labels = []
        for i in range(self.sequence_length, len(feature_data)):
            X_sequences.append(feature_data[i - self.sequence_length : i])
            y_labels.append(target_data[i])

        X_seq = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_labels, dtype=np.float32)

        logger.info(f"TFT: Created {len(X_seq)} sequences ({self.sequence_length} × {self.n_features})")
        return X_seq, y

    def _build_network(self):
        """Initialize the TFT network."""
        self.network = TFTNetwork(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.network.parameters())
        logger.info(f"TFT network built: {n_params:,} parameters")

    def train(self, X_seq: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the TFT model."""
        seed = getattr(self, '_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        self._build_network()

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = nn.BCELoss()

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        logger.info(f"Training TFT for {self.epochs} epochs on {len(X_seq)} samples")

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        max_patience = 15

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

            scheduler.step()
            avg_loss = epoch_loss / n_batches

            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        logger.info(f"TFT early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val: {val_loss:.4f}"
                logger.info(msg)

        # Restore best weights
        if best_state is not None:
            self.network.load_state_dict(best_state)
            logger.info(f"Restored best TFT weights (val_loss={best_val_loss:.4f})")

        self.is_trained = True
        logger.info("TFT training complete")

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        self.network.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)
            preds = self.network(X_t)
            loss = nn.BCELoss()(preds, y_t)
        return loss.item()

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict probability of price going UP."""
        if not self.is_trained:
            raise ValueError("TFT model must be trained first")

        self.network.eval()
        # Process in batches to avoid memory issues
        all_probs = []
        batch_size = 64
        for i in range(0, len(X_seq), batch_size):
            batch = X_seq[i:i + batch_size]
            with torch.no_grad():
                X_tensor = torch.FloatTensor(batch).to(self.device)
                probs = self.network(X_tensor).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs)

    def predict(self, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X_seq)
        return (probs > threshold).astype(int)

    def evaluate(self, X_seq: np.ndarray, y: np.ndarray) -> dict:
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
            f"TFT Eval — Acc: {metrics['accuracy']:.4f}, "
            f"AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}"
        )
        return metrics

    def save(self, symbol: str = "default"):
        filepath = self.model_dir / f"tft_{symbol}.pt"
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "scaler": self.scaler,
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
        }, filepath)
        logger.info(f"TFT model saved to {filepath}")

    def load(self, symbol: str = "default"):
        filepath = self.model_dir / f"tft_{symbol}.pt"
        checkpoint = torch.load(filepath, map_location=self.device)

        self.n_features = checkpoint["n_features"]
        self.hidden_size = checkpoint["hidden_size"]
        self.n_heads = checkpoint["n_heads"]
        self.n_layers = checkpoint["n_layers"]
        self.dropout = checkpoint["dropout"]
        self.sequence_length = checkpoint["sequence_length"]
        self.scaler = checkpoint["scaler"]

        self._build_network()
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = True
        logger.info(f"TFT model loaded from {filepath}")
