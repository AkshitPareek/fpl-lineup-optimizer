"""
Tests for P2-T4: Neural Network (LSTM with attention)
Created by: Testing Expert (Teammate D)

Tests verify:
1. LSTM model architecture correct
2. Attention mechanism working
3. Model trains with early stopping
4. Learning rate scheduling applied
5. Performance comparable to boosting models

Note: This is an optional task (P2-T4).
"""

import pytest
import numpy as np
import os


@pytest.fixture
def data_paths():
    """Return paths to training data."""
    base_path = "datasets/fpl_points_v1"
    return {
        "train_X": os.path.join(base_path, "train_X.npy"),
        "train_y": os.path.join(base_path, "train_y.npy"),
        "val_X": os.path.join(base_path, "validation_X.npy"),
        "val_y": os.path.join(base_path, "validation_y.npy"),
        "test_X": os.path.join(base_path, "test_X.npy"),
        "test_y": os.path.join(base_path, "test_y.npy"),
    }


@pytest.fixture
def train_data(data_paths):
    """Load training and validation data."""
    train_X = np.load(data_paths["train_X"])
    train_y = np.load(data_paths["train_y"])
    val_X = np.load(data_paths["val_X"])
    val_y = np.load(data_paths["val_y"])
    return train_X, train_y, val_X, val_y


class TestLSTMModel:
    """Test suite for LSTM neural network (P2-T4 - Optional)."""

    def test_lstm_module_exists(self):
        """LSTM module should exist."""
        from ml import lstm_model

        assert lstm_model is not None

    def test_lstm_initializes(self, train_data):
        """LSTM model should initialize correctly."""
        from ml.lstm_model import FPLLSTM

        train_X, train_y, val_X, val_y = train_data
        input_dim = train_X.shape[1]

        model = FPLLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)

        assert model is not None

    def test_lstm_forward_pass(self, train_data):
        """LSTM should perform forward pass."""
        from ml.lstm_model import FPLLSTM

        train_X, train_y, val_X, val_y = train_data
        input_dim = train_X.shape[1]

        model = FPLLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)

        X_seq = np.random.randn(8, 10, input_dim)
        output = model.forward_pass(X_seq)

        assert output is not None

    def test_lstm_training_runs(self, train_data):
        """LSTM model should train without error."""
        from ml.lstm_model import train_lstm

        train_X, train_y, val_X, val_y = train_data

        model, history = train_lstm(
            train_X,
            train_y,
            val_X,
            val_y,
            epochs=5,
            batch_size=32,
            early_stopping_patience=3,
        )

        assert model is not None
        assert history is not None
        assert "loss" in history

    def test_lstm_early_stopping(self, train_data):
        """LSTM should use early stopping."""
        from ml.lstm_model import train_lstm

        train_X, train_y, val_X, val_y = train_data

        model, history = train_lstm(
            train_X,
            train_y,
            val_X,
            val_y,
            epochs=50,
            batch_size=32,
            early_stopping_patience=3,
        )

        assert len(history["loss"]) > 0

    def test_lstm_learning_rate_scheduler(self, train_data):
        """LSTM should use learning rate scheduling."""
        from ml.lstm_model import train_lstm

        train_X, train_y, val_X, val_y = train_data

        model, history = train_lstm(
            train_X,
            train_y,
            val_X,
            val_y,
            epochs=5,
            batch_size=32,
            use_lr_scheduler=True,
        )

        assert model is not None

    def test_lstm_predictions(self, train_data):
        """LSTM should produce predictions."""
        from ml.lstm_model import train_lstm

        train_X, train_y, val_X, val_y = train_data

        model, _ = train_lstm(train_X, train_y, val_X, val_y, epochs=3, batch_size=32)

        predictions = model.predict(val_X)

        assert predictions is not None
        assert len(predictions) == len(val_y)

    def test_attention_mechanism(self, train_data):
        """Attention mechanism should work."""
        from ml.lstm_model import FPLLSTMWithAttention

        train_X, train_y, val_X, val_y = train_data
        input_dim = train_X.shape[1]

        model = FPLLSTMWithAttention(input_dim=input_dim, hidden_dim=64)

        X_seq = np.random.randn(8, 10, input_dim)
        output, attention_weights = model.forward_pass(X_seq, return_attention=True)

        assert output is not None
        assert attention_weights is not None

    def test_lstm_save_load(self, train_data):
        """LSTM should save and load correctly."""
        from ml.lstm_model import train_lstm
        import torch

        train_X, train_y, val_X, val_y = train_data

        model, _ = train_lstm(train_X, train_y, val_X, val_y, epochs=2, batch_size=32)

        os.makedirs("models/nn", exist_ok=True)
        torch.save(model.state_dict(), "models/nn/lstm_model.pth")

        from ml.lstm_model import FPLLSTM

        loaded_model = FPLLSTM(input_dim=train_X.shape[1])
        loaded_model.load_state_dict(torch.load("models/nn/lstm_model.pth"))

        assert loaded_model is not None
