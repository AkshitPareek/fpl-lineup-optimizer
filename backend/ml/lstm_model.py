import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, Any


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class FPLLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 121,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_layer(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output.squeeze()

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = X
            output = self.forward(X_tensor)
            return output.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = X

            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(1)

            X_tensor = self.input_layer(X_tensor)
            lstm_out, _ = self.lstm(X_tensor)
            lstm_out = lstm_out[:, -1, :]
            lstm_out = self.dropout(lstm_out)
            output = self.fc(lstm_out)
            return output.squeeze().numpy()


class FPLLSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 121,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_layer(x)
        lstm_out, _ = self.lstm(x)

        context, attention_weights = self.attention(lstm_out)
        context = self.dropout(context)
        output = self.fc(context)

        if return_attention:
            return output.squeeze(), attention_weights
        return output.squeeze(), None

    def forward_pass(
        self, X: np.ndarray, return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = X

            output, attention = self.forward(X_tensor, return_attention)

            if return_attention:
                return output.numpy(), attention.numpy()
            return output.numpy(), None

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X)
            else:
                X_tensor = X

            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(1)

            X_tensor = self.input_layer(X_tensor)
            lstm_out, _ = self.lstm(X_tensor)
            context, _ = self.attention(lstm_out)
            context = self.dropout(context)
            output = self.fc(context)
            return output.squeeze().numpy()


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_length: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    seq_length: int = 10,
    early_stopping_patience: int = 10,
    use_lr_scheduler: bool = True,
    use_attention: bool = False,
) -> Tuple[nn.Module, Dict[str, list]]:
    input_dim = X_train.shape[1]

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)

    if len(X_train_seq) == 0:
        X_train_seq = (
            X_train.reshape(1, -1)
            if len(X_train.shape) == 1
            else X_train.reshape(X_train.shape[0], 1, -1)
        )
        y_train_seq = y_train
    if len(X_val_seq) == 0:
        X_val_seq = (
            X_val.reshape(1, -1)
            if len(X_val.shape) == 1
            else X_val.reshape(X_val.shape[0], 1, -1)
        )
        y_val_seq = y_val

    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.FloatTensor(y_val_seq)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if use_attention:
        model = FPLLSTMWithAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    else:
        model = FPLLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    else:
        scheduler = None

    history = {"loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            if use_attention:
                output, _ = model(batch_X, return_attention=False)
            else:
                output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                if use_attention:
                    output, _ = model(batch_X, return_attention=False)
                else:
                    output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def predict_lstm(model: nn.Module, X: np.ndarray, seq_length: int = 10) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X

        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        elif X_tensor.dim() == 2:
            X_tensor = X_tensor.unsqueeze(1)

        if hasattr(model, "attention"):
            output, _ = model(X_tensor, return_attention=False)
        else:
            output = model(X_tensor)

        return output.numpy()


def save_lstm_model(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_lstm_model(
    path: str, input_dim: int = 121, hidden_dim: int = 64, num_layers: int = 2
) -> nn.Module:
    model = FPLLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
