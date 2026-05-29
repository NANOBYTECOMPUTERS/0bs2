#!/usr/bin/env python3
"""MLP used by the advisory neural targeting head trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NeuralTargetingModelConfig:
    feature_dim: int
    hidden_size: int = 192
    max_refinement_px: float = 35.0
    dropout: float = 0.04


class NeuralTargetingNet(torch.nn.Module):
    """Predict bounded pixel refinement and confidence from C++-scaled features."""

    output_name = "targeting_output"

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 192,
        max_refinement_px: float = 35.0,
        dropout: float = 0.04,
    ) -> None:
        super().__init__()
        self.feature_dim = max(1, int(feature_dim))
        self.max_refinement_px = float(max(1.0, max_refinement_px))
        hidden = max(8, int(hidden_size))
        p = max(0.0, min(0.50, float(dropout)))

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=p),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 3),
        )

    def forward(self, targeting_features):
        raw = self.net(targeting_features)
        refinement = torch.tanh(raw[:, :2]) * self.max_refinement_px
        confidence = torch.sigmoid(raw[:, 2:3])
        return torch.cat([refinement, confidence], dim=1)


def make_model(
    feature_dim: int,
    hidden_size: int = 192,
    max_refinement_px: float = 35.0,
    dropout: float = 0.04,
) -> NeuralTargetingNet:
    return NeuralTargetingNet(
        feature_dim=max(1, int(feature_dim)),
        hidden_size=max(8, int(hidden_size)),
        max_refinement_px=max(1.0, float(max_refinement_px)),
        dropout=dropout,
    )

