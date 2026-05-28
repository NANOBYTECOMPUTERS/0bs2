#!/usr/bin/env python3
"""Small sequence models used by the temporal predictor trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalModelConfig:
    feature_dim: int = 8
    history_length: int = 12
    prediction_horizon: int = 16
    hidden_size: int = 160
    model_type: str = "gru"
    dropout: float = 0.05


class TemporalPredictorNet(torch.nn.Module):
    """Predict future x/y points from recent track history."""

    def __init__(self, cfg: TemporalModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.prediction_horizon = int(cfg.prediction_horizon)
        self.model_type = cfg.model_type.lower().strip()

        if self.model_type == "gru":
            self.gru = torch.nn.GRU(
                input_size=cfg.feature_dim,
                hidden_size=cfg.hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(cfg.hidden_size),
                torch.nn.Linear(cfg.hidden_size, cfg.hidden_size),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=max(0.0, min(0.50, cfg.dropout))),
                torch.nn.Linear(cfg.hidden_size, cfg.prediction_horizon * 2),
            )
        elif self.model_type == "transformer":
            self.input_proj = torch.nn.Linear(cfg.feature_dim, cfg.hidden_size)
            self.position_embedding = torch.nn.Parameter(
                torch.zeros(1, cfg.history_length, cfg.hidden_size)
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=4,
                dim_feedforward=cfg.hidden_size * 2,
                dropout=max(0.0, min(0.50, cfg.dropout)),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(cfg.hidden_size),
                torch.nn.Linear(cfg.hidden_size, cfg.prediction_horizon * 2),
            )
        else:
            raise ValueError("model_type must be 'gru' or 'transformer'")

    def forward(self, temporal_history):
        if self.model_type == "gru":
            _, hidden = self.gru(temporal_history)
            encoded = hidden[-1]
        else:
            steps = temporal_history.shape[1]
            encoded_steps = self.input_proj(temporal_history)
            encoded_steps = encoded_steps + self.position_embedding[:, :steps, :]
            encoded = self.transformer(encoded_steps)[:, -1, :]

        future_xy = self.head(encoded).reshape(
            temporal_history.shape[0],
            self.prediction_horizon,
            2,
        )
        return future_xy


class NormalizedTemporalPredictor(torch.nn.Module):
    """Wrap a predictor so exported ONNX accepts raw runtime features."""

    def __init__(self, core: TemporalPredictorNet, feature_mean, feature_std) -> None:
        super().__init__()
        self.core = core
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32).reshape(1, 1, -1),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(feature_std, dtype=torch.float32).clamp_min(1e-6).reshape(1, 1, -1),
        )

    def forward(self, temporal_history):
        normalized = (temporal_history - self.feature_mean) / self.feature_std
        return self.core(normalized)


def make_model(
    feature_dim: int = 8,
    history_length: int = 12,
    prediction_horizon: int = 16,
    hidden_size: int = 160,
    model_type: str = "gru",
    dropout: float = 0.05,
) -> TemporalPredictorNet:
    return TemporalPredictorNet(
        TemporalModelConfig(
            feature_dim=max(1, int(feature_dim)),
            history_length=max(2, int(history_length)),
            prediction_horizon=max(1, int(prediction_horizon)),
            hidden_size=max(8, int(hidden_size)),
            model_type=model_type,
            dropout=dropout,
        )
    )
