#!/usr/bin/env python3
"""Utility to inspect FinCast model size and memory requirements."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure the local src directory is importable so we can reuse project modules.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffm import pytorch_patched_decoder_MOE as ppd  # type: ignore  # noqa: E402

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}


def human_readable_bytes(num_bytes: float) -> str:
    """Return a human-friendly string for a byte count."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def build_model(dtype: str) -> torch.nn.Module:
    """Instantiate the PatchedTimeSeriesDecoder_MOE with the desired dtype."""
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dtype}'. Options: {list(DTYPE_BYTES)}")

    config = ppd.FFMConfig()
    config.dtype = dtype
    model = ppd.PatchedTimeSeriesDecoder_MOE(config)
    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def estimate_inference_memory(param_count: int, dtype: str, activation_ratio: float) -> dict[str, float]:
    """Estimate inference-time memory footprint."""
    weight_bytes = param_count * DTYPE_BYTES[dtype]
    activation_bytes = weight_bytes * activation_ratio
    total_bytes = weight_bytes + activation_bytes
    return {
        "weights": weight_bytes,
        "activations": activation_bytes,
        "total": total_bytes,
    }


def estimate_finetune_storage(
    param_count: int,
    model_dtype: str,
    grad_dtype: str,
    optimizer_dtype: str,
    optimizer_state_copies: int,
) -> dict[str, float]:
    """Estimate memory required for full-parameter fine-tuning."""
    weight_bytes = param_count * DTYPE_BYTES[model_dtype]
    grad_bytes = param_count * DTYPE_BYTES[grad_dtype]
    optimizer_bytes = param_count * DTYPE_BYTES[optimizer_dtype] * optimizer_state_copies
    total_bytes = weight_bytes + grad_bytes + optimizer_bytes
    return {
        "weights": weight_bytes,
        "grads": grad_bytes,
        "optimizer": optimizer_bytes,
        "total": total_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the FinCast model")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=tuple(DTYPE_BYTES.keys()),
        help="Precision used to store model weights during inference",
    )
    parser.add_argument(
        "--activation-ratio",
        type=float,
        default=0.30,
        help="Activations-as-a-multiple of weight memory when estimating inference footprint",
    )
    parser.add_argument(
        "--grad-dtype",
        default="float32",
        choices=tuple(DTYPE_BYTES.keys()),
        help="Precision used to store gradients during training",
    )
    parser.add_argument(
        "--optimizer-dtype",
        default="float32",
        choices=tuple(DTYPE_BYTES.keys()),
        help="Precision used for optimizer states (e.g., Adam moments)",
    )
    parser.add_argument(
        "--optimizer-state-copies",
        type=int,
        default=2,
        help="Number of optimizer state tensors per parameter (2 for Adam, 0 for SGD)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.dtype)
    param_count = count_parameters(model)

    inference_estimate = estimate_inference_memory(
        param_count=param_count,
        dtype=args.dtype,
        activation_ratio=args.activation_ratio,
    )
    finetune_estimate = estimate_finetune_storage(
        param_count=param_count,
        model_dtype=args.dtype,
        grad_dtype=args.grad_dtype,
        optimizer_dtype=args.optimizer_dtype,
        optimizer_state_copies=args.optimizer_state_copies,
    )

    print("FinCast model analysis")
    print("======================")
    print(f"Parameter count    : {param_count:,}")
    print(f"Weight precision   : {args.dtype}")
    print()
    print("Inference memory estimate")
    print("--------------------------")
    print(f"Weights             : {human_readable_bytes(inference_estimate['weights'])}")
    print(f"Activations (~{args.activation_ratio:.2f}x) : {human_readable_bytes(inference_estimate['activations'])}")
    print(f"Total               : {human_readable_bytes(inference_estimate['total'])}")
    print()
    print("Full fine-tune storage estimate")
    print("--------------------------------")
    print(f"Weights   ({args.dtype}) : {human_readable_bytes(finetune_estimate['weights'])}")
    print(f"Gradients ({args.grad_dtype}) : {human_readable_bytes(finetune_estimate['grads'])}")
    print(
        f"Optimizer ({args.optimizer_state_copies} copies @ {args.optimizer_dtype}) : "
        f"{human_readable_bytes(finetune_estimate['optimizer'])}"
    )
    print(f"Total                         : {human_readable_bytes(finetune_estimate['total'])}")


if __name__ == "__main__":
    main()
