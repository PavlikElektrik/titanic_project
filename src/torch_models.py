from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split


def _build_mlp(input_dim: int, hidden_layers: tuple[int, ...], dropout: float, output_dim: int) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(torch.nn.Linear(prev_dim, output_dim))
    return torch.nn.Sequential(*layers)


class TorchBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 80,
        patience: int = 10,
        val_fraction: float = 0.2,
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        x_tr, x_va, y_tr, y_va = train_test_split(
            X_arr,
            y_arr,
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=y_arr,
        )

        torch.manual_seed(self.random_state)
        self.model_ = _build_mlp(X_arr.shape[1], self.hidden_layers, self.dropout, 1)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(x_tr, dtype=torch.float32),
            torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        x_va_t = torch.tensor(x_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va.reshape(-1, 1), dtype=torch.float32)

        best_state = None
        best_loss = float("inf")
        patience_left = self.patience

        for _ in range(self.max_epochs):
            self.model_.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(x_va_t), y_va_t).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X_arr, dtype=torch.float32)).numpy().reshape(-1)
        proba = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 12,
        val_fraction: float = 0.2,
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        x_tr, x_va, y_tr, y_va = train_test_split(
            X_arr,
            y_arr,
            test_size=self.val_fraction,
            random_state=self.random_state,
        )

        torch.manual_seed(self.random_state)
        self.model_ = _build_mlp(X_arr.shape[1], self.hidden_layers, self.dropout, 1)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(x_tr, dtype=torch.float32),
            torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32),
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        x_va_t = torch.tensor(x_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_va.reshape(-1, 1), dtype=torch.float32)

        best_state = None
        best_loss = float("inf")
        patience_left = self.patience

        for _ in range(self.max_epochs):
            self.model_.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(x_va_t), y_va_t).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.tensor(X_arr, dtype=torch.float32)).numpy().reshape(-1)
        return preds