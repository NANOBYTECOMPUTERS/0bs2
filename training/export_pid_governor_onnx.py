#!/usr/bin/env python3
"""Export a trained PID governor MLP to ONNX for C++ runtime inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_pid_governor import import_torch, make_model


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return ROOT / resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PID governor model to ONNX.")
    parser.add_argument("--model", default="training/models/pid_governor.pt")
    parser.add_argument("--output", default="training/models/pid_governor.onnx")
    parser.add_argument("--metadata", default="training/models/pid_governor_onnx.json")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch = import_torch()

    model_path = resolve_repo_path(args.model)
    artifact = torch.load(model_path, map_location="cpu")
    feature_columns = artifact["feature_columns"]
    label_columns = artifact["label_columns"]
    model = make_model(torch, len(feature_columns), len(label_columns), int(artifact["hidden"]))
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    class NormalizedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = model
            self.register_buffer("mean", torch.tensor(artifact["feature_mean"], dtype=torch.float32))
            self.register_buffer("std", torch.tensor(artifact["feature_std"], dtype=torch.float32).clamp_min(1e-6))

        def forward(self, raw_features):
            normalized = (raw_features - self.mean) / self.std
            return self.core(normalized)

    wrapped = NormalizedModel()
    wrapped.eval()
    dummy = torch.zeros(1, len(feature_columns), dtype=torch.float32)
    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            wrapped,
            dummy,
            output,
            input_names=["pid_features"],
            output_names=["pid_scales"],
            dynamic_axes={"pid_features": {0: "batch"}, "pid_scales": {0: "batch"}},
            opset_version=max(12, args.opset),
            dynamo=False,
        )
    except Exception as exc:
        raise SystemExit(f"ONNX export failed: {exc}") from exc

    metadata = {
        "onnx_path": str(output),
        "source_model": str(args.model),
        "input_name": "pid_features",
        "output_name": "pid_scales",
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "output_order": label_columns,
    }
    metadata_path = resolve_repo_path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported ONNX model to {output}")
    print(f"Saved ONNX metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
