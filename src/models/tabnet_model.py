"""
TabNet Model — attention-based tabular deep learning for feature selection.

TabNet (Arik & Pfister, 2019) is designed specifically for tabular data:
  1. Sequential Attention — selects features at each decision step
  2. Sparse Feature Selection — forces the model to focus on few features
  3. Interpretability — shows which features drove each prediction
  4. No preprocessing needed — handles raw features directly

This works on the same tabular features as XGBoost but often finds
complementary patterns due to its different learning approach.

Usage:
    from src.models.tabnet_model import TabNetModel
    model = TabNetModel()
    model.train(X_train, y_train, X_val, y_val)
    probs = model.predict_proba(X_test)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
from src.config_loader import get_settings, DATA_DIR


class TabNetBlock(nn.Module):
    """Single TabNet decision step."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_independent=1, n_shared=1):
        super().__init__()

        # Shared layers across steps (for feature reuse)
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            for i in range(n_shared)
        ])

        # Step-specific layers
        self.independent_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            for _ in range(n_independent)
        ])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.shared_layers:
            x = F.relu(layer(x))
        for layer in self.independent_layers:
            x = F.relu(layer(x))
        return self.fc_out(x)


class AttentiveTransformer(nn.Module):
    """Learns sparse attention masks over features."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x, prior_scales):
        """
        x: processed features from previous step
        prior_scales: accumulated attention from previous steps (for sparsity)
        """
        x = self.bn(self.fc(x))
        # prior_scales encourages attending to new features
        x = x * prior_scales
        # Sparsemax-like via softmax with temperature
        attention = F.softmax(x * 2.0, dim=-1)  # temperature=2 for sharper selection
        return attention


class TabNetNetwork(nn.Module):
    """
    TabNet classifier for binary prediction.
    """

    def __init__(self, input_dim, hidden_dim=128, n_steps=3, gamma=1.3,
                 n_independent=1, n_shared=1, dropout=0.1):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.input_dim = input_dim

        # Initial batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim)

        # Shared feature transformer applied at the beginning
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Decision steps
        self.steps = nn.ModuleList()
        self.attention = nn.ModuleList()
        for _ in range(n_steps):
            self.steps.append(TabNetBlock(
                hidden_dim, hidden_dim, hidden_dim,
                n_independent, n_shared,
            ))
            self.attention.append(AttentiveTransformer(hidden_dim, input_dim))

        # Feature-to-hidden projection for masked input
        self.feature_proj = nn.Linear(input_dim, hidden_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_attention=False):
        """
        x: (batch, input_dim)
        """
        x = self.initial_bn(x)

        # Initialize
        prior_scales = torch.ones(x.shape[0], self.input_dim, device=x.device)
        aggregated_output = torch.zeros(x.shape[0], self.steps[0].fc_out.out_features,
                                        device=x.device)
        attention_masks = []
        h = self.shared_fc(x)

        for step_idx in range(self.n_steps):
            # Attention: which features to focus on this step
            attention_mask = self.attention[step_idx](h, prior_scales)
            attention_masks.append(attention_mask)

            # Update prior scales (encourage new features)
            prior_scales = prior_scales * (self.gamma - attention_mask)

            # Masked input
            masked_input = attention_mask * x
            masked_h = self.feature_proj(masked_input)

            # Process through step
            step_output = self.steps[step_idx](masked_h)
            aggregated_output = aggregated_output + F.relu(step_output)

            # Update h for next step's attention
            h = step_output

        # Classify
        prob = self.classifier(aggregated_output).squeeze(-1)

        if return_attention:
            return prob, attention_masks
        return prob


class TabNetModel:
    """TabNet-based tabular model for price direction prediction."""

    # Same exclude columns as XGBoost
    EXCLUDE_COLUMNS = [
        "target_return", "target_direction",
        "target_return_5d", "target_direction_5d",
        "open", "high", "low", "close", "volume",
        "trade_count", "vwap",
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

    def __init__(self, params: dict = None):
        settings = get_settings()
        default_params = settings.get("models", {}).get("tabnet", {})

        self.hidden_dim = default_params.get("hidden_dim", 128)
        self.n_steps = default_params.get("n_steps", 3)
        self.gamma = default_params.get("gamma", 1.3)
        self.dropout = default_params.get("dropout", 0.1)
        self.epochs = default_params.get("epochs", 80)
        self.batch_size = default_params.get("batch_size", 256)
        self.learning_rate = default_params.get("learning_rate", 0.002)

        if params:
            for k, v in params.items():
                setattr(self, k, v)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.network = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.model_dir = DATA_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TabNetModel initialized (hidden={self.hidden_dim}, steps={self.n_steps})")

    def prepare_features(self, df: pd.DataFrame, target_col: str = "target_direction"):
        """Separate features and target, matching XGBoost's interface."""
        df_clean = df.dropna(subset=[target_col]).copy()

        feature_cols = [
            col for col in df_clean.columns
            if col not in self.EXCLUDE_COLUMNS
        ]
        self.feature_names = feature_cols

        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_col]

        logger.info(f"TabNet: Prepared {len(X)} samples with {len(feature_cols)} features")
        return X, y

    def _build_network(self, input_dim):
        self.network = TabNetNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            n_steps=self.n_steps,
            gamma=self.gamma,
            dropout=self.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.network.parameters())
        logger.info(f"TabNet network built: {n_params:,} parameters, {input_dim} input features")

    def train(self, X: pd.DataFrame, y: pd.Series, eval_set: tuple = None):
        """Train TabNet on tabular features."""
        seed = getattr(self, '_seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_np = self.scaler.fit_transform(X.values.astype(np.float32))
        y_np = y.values.astype(np.float32)

        self._build_network(X_np.shape[1])

        X_tensor = torch.FloatTensor(X_np).to(self.device)
        y_tensor = torch.FloatTensor(y_np).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Prepare validation
        val_X, val_y = None, None
        if eval_set is not None:
            val_X_np = self.scaler.transform(eval_set[0].values.astype(np.float32))
            val_X = torch.FloatTensor(val_X_np).to(self.device)
            val_y = torch.FloatTensor(eval_set[1].values.astype(np.float32)).to(self.device)

        logger.info(f"Training TabNet for {self.epochs} epochs on {len(X_np)} samples")

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.network(X_batch)
                loss = criterion(preds, y_batch)

                # Sparsity regularization: encourage focused attention
                # (small additional loss term)
                loss = loss  # Note: full TabNet adds entropy penalty here

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batches

            # Validation
            val_loss = None
            if val_X is not None:
                self.network.eval()
                with torch.no_grad():
                    val_preds = self.network(val_X)
                    val_loss = criterion(val_preds, val_y).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:
                        logger.info(f"TabNet early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val: {val_loss:.4f}"
                logger.info(msg)

        if best_state is not None:
            self.network.load_state_dict(best_state)

        self.is_trained = True
        logger.info("TabNet training complete")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of price going UP."""
        if not self.is_trained:
            raise ValueError("TabNet must be trained first")

        if isinstance(X, pd.DataFrame):
            X_np = self.scaler.transform(X.fillna(0).values.astype(np.float32))
        else:
            X_np = self.scaler.transform(X.astype(np.float32))

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            probs = self.network(X_tensor).cpu().numpy()
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "auc_roc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0,
            "f1": f1_score(y, y_pred, zero_division=0),
            "n_samples": len(y),
        }

        logger.info(
            f"TabNet Eval — Acc: {metrics['accuracy']:.4f}, "
            f"AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}"
        )
        return metrics

    def get_feature_importance(self, X_sample: pd.DataFrame = None) -> pd.DataFrame:
        """Get feature importance from attention masks."""
        if not self.is_trained or X_sample is None:
            return pd.DataFrame()

        X_np = self.scaler.transform(X_sample.fillna(0).values.astype(np.float32))
        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_np).to(self.device)
            _, attention_masks = self.network(X_tensor, return_attention=True)

        # Average attention across all steps and all samples
        total_attention = torch.zeros(X_np.shape[1], device=self.device)
        for mask in attention_masks:
            total_attention += mask.mean(dim=0)
        total_attention = total_attention / len(attention_masks)

        importance = total_attention.cpu().numpy()
        feature_imp = pd.DataFrame({
            "feature": self.feature_names or [f"f_{i}" for i in range(len(importance))],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return feature_imp

    def save(self, symbol: str = "default"):
        filepath = self.model_dir / f"tabnet_{symbol}.pt"
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "hidden_dim": self.hidden_dim,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "dropout": self.dropout,
            "input_dim": self.network.input_dim,
        }, filepath)
        logger.info(f"TabNet model saved to {filepath}")

    def load(self, symbol: str = "default"):
        filepath = self.model_dir / f"tabnet_{symbol}.pt"
        checkpoint = torch.load(filepath, map_location=self.device)

        self.hidden_dim = checkpoint["hidden_dim"]
        self.n_steps = checkpoint["n_steps"]
        self.gamma = checkpoint["gamma"]
        self.dropout = checkpoint["dropout"]
        self.scaler = checkpoint["scaler"]
        self.feature_names = checkpoint["feature_names"]

        self._build_network(checkpoint["input_dim"])
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = True
        logger.info(f"TabNet model loaded from {filepath}")
