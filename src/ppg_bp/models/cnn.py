"""Waveform model: a small one-dimensional residual convolutional network estimating BP from a 10-s PPG (125 Hz, 1,250
samples).

Its size suits training on a single PC without a GPU. Each input segment is scaled to mean 0 and SD 1
(the unit of PPG amplitude differs between devices, so, as in the feature model, the raw amplitude is not used).

The outputs are targets x output types: the mean (trained with squared error) and the given quantiles
(trained with pinball loss). The quantiles are used for CQR.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn


def standardise(x: torch.Tensor) -> torch.Tensor:
    """Scale each segment to mean 0 and SD 1. Shape (segments, samples)."""
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.std(dim=1, keepdim=True).clamp_min(1e-6)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride, pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel, 1, pad, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + (x if self.skip is None else self.skip(x)))


class WaveformNet(nn.Module):
    """Output shape (segments, targets, 1 + quantiles); index 0 of the last axis is the mean."""

    quantiles: torch.Tensor  # registered as a buffer in __init__

    def __init__(self, n_targets: int = 3, quantiles: tuple = (), channels: tuple = (32, 48, 64, 96)):
        super().__init__()
        self.n_targets = n_targets
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        layers = [nn.Conv1d(1, channels[0], 15, 2, 7, bias=False), nn.BatchNorm1d(channels[0]), nn.ReLU(inplace=True)]
        previous = channels[0]
        for width in channels:
            layers += [ResidualBlock(previous, width, stride=2), ResidualBlock(width, width, stride=1)]
            previous = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(previous, n_targets * (1 + len(quantiles)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(standardise(x).unsqueeze(1)).mean(dim=2)
        return self.head(z).view(len(x), self.n_targets, -1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Pinball loss. pred is (segments, targets, quantiles), target is (segments, targets)."""
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()


def combined_loss(output: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Squared error of the mean plus pinball loss of the quantiles. target must be standardised."""
    loss = torch.mean((output[..., 0] - target) ** 2)
    if len(quantiles):
        loss = loss + pinball_loss(output[..., 1:], target, quantiles)
    return loss


def save_checkpoint(
    path: Path, state: dict, mean: ArrayLike, std: ArrayLike, settings: dict, targets: Sequence[str]
) -> None:
    """Save a trained model. mean and std (of the training targets) are stored as float32 tensors so that the file
    can be read with torch.load(weights_only=True), which does not run arbitrary code."""
    torch.save(
        {
            "state": state,
            "mean": torch.as_tensor(np.asarray(mean, dtype=np.float32)),
            "std": torch.as_tensor(np.asarray(std, dtype=np.float32)),
            "settings": settings,
            "targets": list(targets),
        },
        path,
    )


def load_checkpoint(path: Path) -> dict:
    """Read a file written by save_checkpoint (safely: weights_only=True). mean and std are returned as numpy."""
    saved = torch.load(path, weights_only=True)
    saved["mean"], saved["std"] = saved["mean"].numpy(), saved["std"].numpy()
    return saved


def model_from_checkpoint(saved: dict) -> WaveformNet:
    """The trained network in evaluation mode."""
    model = WaveformNet(
        len(saved["targets"]), tuple(saved["settings"]["quantiles"]), tuple(saved["settings"]["channels"])
    )
    model.load_state_dict(saved["state"])
    return model.eval()
