"""Waveform model: output shape, independence of the input amplitude, the loss, and that it can learn."""

import numpy as np
import torch

from ppg_bp.models.cnn import WaveformNet, combined_loss, pinball_loss, standardise

QUANTILES = (0.05, 0.5, 0.95)


def test_output_shape_with_and_without_quantiles():
    x = torch.randn(4, 1250)
    assert WaveformNet(3).eval()(x).shape == (4, 3, 1)
    assert WaveformNet(3, QUANTILES).eval()(x).shape == (4, 3, 4)


def test_model_is_small_enough_for_cpu_training():
    n_params = sum(p.numel() for p in WaveformNet(3, QUANTILES).parameters())
    assert n_params < 500_000


def test_prediction_does_not_depend_on_amplitude_or_offset():
    torch.manual_seed(0)
    model = WaveformNet(3).eval()
    x = torch.randn(2, 1250)
    with torch.no_grad():
        assert torch.allclose(model(x), model(250.0 * x + 40.0), atol=1e-4)


def test_standardise_handles_a_flat_segment():
    out = standardise(torch.ones(1, 1250))
    assert torch.isfinite(out).all()


def test_pinball_loss_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(0)
    target = torch.tensor(rng.normal(size=(20000, 1)), dtype=torch.float32)
    quantiles = torch.tensor([0.9])
    losses = {c: pinball_loss(torch.full((20000, 1, 1), c), target, quantiles).item() for c in (0.8, 1.28, 1.8)}
    assert (
        losses[1.28] < losses[0.8] and losses[1.28] < losses[1.8]
    )  # the 90% point of the standard normal distribution is 1.28


def test_model_learns_a_simple_relationship():
    """Learns, in a few steps, a target that depends only on the pulse rate."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rate = rng.uniform(1.0, 2.0, size=64)
    t = np.arange(1250) / 125.0
    x = torch.tensor(np.sin(2 * np.pi * rate[:, None] * t), dtype=torch.float32)
    y = torch.tensor((rate - 1.5) * 2.0, dtype=torch.float32).unsqueeze(1)
    model = WaveformNet(1, QUANTILES)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    quantiles = model.quantiles
    losses = []
    for _ in range(30):
        optimiser.zero_grad()
        loss = combined_loss(model(x), y, quantiles)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0]
