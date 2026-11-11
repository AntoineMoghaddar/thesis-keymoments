"""Sequence model on the resampled signals of a window (needs PyTorch)."""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:  # the rest of the pipeline works without it
    HAS_TORCH = False


def _network(channels: int, hidden: int):
    class Net(nn.Module):
        """Two convolution layers that read local shape, then a GRU over the corner."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(channels, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.MaxPool1d(2))
            self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

        def forward(self, x):
            h = self.conv(x).transpose(1, 2)
            out, _ = self.gru(h)
            return self.head(out.mean(dim=1)).squeeze(-1)

    return Net()


class SequenceDetector:
    def __init__(self, cfg: dict, seed: int = 0):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install it with: pip install torch")
        self.cfg, self.seed = cfg["model"]["seq"], seed
        self.net = None
        self.mean = self.std = None

    def _scale(self, X: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(((X - self.mean) / self.std).astype(np.float32))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SequenceDetector":
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        self.net = _network(X.shape[1], self.cfg["hidden"])
        data, target = self._scale(X), torch.from_numpy(y.astype(np.float32))
        # y may hold probabilities (soft weak labels); the loss accepts those as targets.
        pos = max(float(y.sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
        optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        generator = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.cfg["epochs"]):
            order = torch.randperm(len(data), generator=generator)
            for start in range(0, len(order), self.cfg["batch_size"]):
                batch = order[start:start + self.cfg["batch_size"]]
                if len(batch) < 2:
                    continue
                optim.zero_grad()
                loss_fn(self.net(data[batch]), target[batch]).backward()
                optim.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = [torch.sigmoid(self.net(chunk)) for chunk in self._scale(X).split(1024)]
        return torch.cat(out).numpy()
