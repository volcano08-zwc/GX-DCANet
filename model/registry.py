"""Minimal model registry for the standalone LYNet package."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from .LYNet import LYNet


MODEL_REGISTRY = {"LYNet": LYNet}


def available_models() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def model_args_from_config(config: Mapping) -> dict:
    values = config.get("network_args")
    if not isinstance(values, Mapping):
        raise ValueError("configuration requires a mapping named 'network_args'")
    return dict(values)


def build_registered_model(
    name: str,
    model_args: Mapping,
    device: torch.device,
    *,
    dcanet_initializer: Callable[[nn.Module], None] | None = None,
) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            "This standalone package contains LYNet only."
        )
    model = MODEL_REGISTRY[name](**dict(model_args)).to(device)
    if dcanet_initializer is not None:
        model.apply_local_initialization(dcanet_initializer)
    apply_model_constraints(model)
    return model


@torch.no_grad()
def apply_model_constraints(model: nn.Module) -> None:
    constraint = getattr(model, "apply_max_norm", None)
    if callable(constraint):
        constraint()


def validate_model_data_contract(
    name: str,
    model_args: Mapping,
    *,
    channels: int,
    samples: int,
    num_classes: int,
) -> None:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            "This standalone package contains LYNet only."
        )
    args = dict(model_args)
    declared = (
        int(args.get("C", channels)),
        int(args.get("T", samples)),
        int(args.get("num_classes", num_classes)),
    )
    expected = (int(channels), int(samples), int(num_classes))
    if declared != expected:
        raise ValueError(
            "LYNet declares "
            f"C/T/classes={declared}, but data requires {expected}."
        )


__all__ = [
    "MODEL_REGISTRY",
    "apply_model_constraints",
    "available_models",
    "build_registered_model",
    "model_args_from_config",
    "validate_model_data_contract",
]
