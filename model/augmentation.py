"""Training-only paired multi-window augmentation for dual-view EEG."""

from __future__ import annotations

import torch
from torch import nn


class PairedMultiWindowAugment(nn.Module):
    """Apply one stochastic temporal focus window to both views of a trial."""

    def __init__(
        self,
        *,
        probability: float = 0.5,
        focus_samples: int = 500,
        focus_starts: tuple[int, ...] = (0, 125, 250, 375, 500),
        inside_gain: float = 1.5,
        outside_gain: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        if focus_samples <= 0 or not focus_starts:
            raise ValueError("focus_samples and focus_starts must be non-empty and positive.")
        self.probability = float(probability)
        self.focus_samples = int(focus_samples)
        self.focus_starts = tuple(int(value) for value in focus_starts)
        self.inside_gain = float(inside_gain)
        self.outside_gain = float(outside_gain)

    def forward(
        self,
        raw_maps: torch.Tensor,
        aligned_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_maps.shape != aligned_maps.shape:
            raise ValueError("Raw/aligned Sinc maps must have the same shape.")
        if not self.training:
            return raw_maps, aligned_maps

        pair = torch.stack((raw_maps, aligned_maps), dim=1)
        batch, views, bands, channels, samples = pair.shape
        if self.focus_samples >= samples:
            return raw_maps, aligned_maps

        merged = pair.reshape(batch, views * bands, channels, samples)
        event = torch.rand(batch, 1, 1, 1, device=merged.device) < self.probability
        choice = torch.randint(len(self.focus_starts), (batch,), device=merged.device)
        starts_table = torch.tensor(self.focus_starts, device=merged.device)
        starts = starts_table[choice]
        positions = torch.arange(samples, device=merged.device).view(1, 1, 1, samples)
        inside = (positions >= starts.view(batch, 1, 1, 1)) & (
            positions < (starts + self.focus_samples).view(batch, 1, 1, 1)
        )
        focus = torch.where(
            inside,
            torch.full((), self.inside_gain, dtype=merged.dtype, device=merged.device),
            torch.full((), self.outside_gain, dtype=merged.dtype, device=merged.device),
        )
        merged = merged * torch.where(event, focus, torch.ones_like(focus))
        restored = merged.reshape(batch, views, bands, channels, samples)
        return restored[:, 0], restored[:, 1]


class PairedIntrinsicComponentMask(nn.Module):
    """Mask matching covariance-eigenvalue ranks in paired EEG views."""

    def __init__(
        self,
        max_masked_components: int = 6,
        probability: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if max_masked_components < 1:
            raise ValueError("max_masked_components must be positive.")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        self.max_masked_components = int(max_masked_components)
        self.probability = float(probability)
        self.eps = float(eps)
        self.last_masked_counts = torch.empty(0, dtype=torch.long)

    @torch.no_grad()
    def prepare_cache(self, raw, aligned):
        """Cache only deterministic, trial-local quantities before augmentation."""
        pair = torch.stack((raw[:, 0], aligned[:, 0]), dim=1)
        mean = pair.mean(dim=-1, keepdim=True)
        std = pair.std(dim=-1, unbiased=False, keepdim=True)
        content = (pair - mean) / (std + self.eps)
        covariance = content @ content.transpose(-1, -2) / float(pair.shape[-1] - 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        intrinsic = eigenvectors.transpose(-1, -2) @ content
        return mean, std, eigenvectors, intrinsic

    @torch.no_grad()
    def forward(
        self,
        raw: torch.Tensor,
        aligned: torch.Tensor,
        cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.shape != aligned.shape:
            raise ValueError("Raw/aligned EEG views must have the same shape.")
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f"Expected paired EEG [B, 1, C, T], got {tuple(raw.shape)}.")
        if not self.training or self.probability == 0.0:
            self.last_masked_counts = torch.zeros(raw.shape[0], dtype=torch.long)
            return raw, aligned

        pair = torch.stack((raw[:, 0], aligned[:, 0]), dim=1)
        batch, _views, channels, samples = pair.shape
        if channels < 2 or samples < 2:
            raise ValueError("Intrinsic-component masking requires C and T >= 2.")

        mean, std, eigenvectors, intrinsic = (
            self.prepare_cache(raw, aligned) if cache is None else cache
        )

        maximum = min(self.max_masked_components, channels - 1)
        counts = torch.randint(1, maximum + 1, (batch,), device=pair.device)
        events = torch.rand(batch, device=pair.device) < self.probability
        counts = torch.where(events, counts, torch.zeros_like(counts))
        component_order = torch.rand(batch, channels, device=pair.device).argsort(dim=1)
        component_mask = torch.zeros(
            batch, channels, dtype=torch.bool, device=pair.device
        )
        selected = torch.arange(channels, device=pair.device)[None, :] < counts[:, None]
        component_mask.scatter_(1, component_order, selected)

        intrinsic = intrinsic.masked_fill(component_mask[:, None, :, None], 0.0)
        reconstructed = eigenvectors @ intrinsic
        augmented = reconstructed * std + mean
        if not torch.isfinite(augmented).all():
            raise FloatingPointError("NaN/Inf in intrinsic-component masking output.")
        self.last_masked_counts = counts.detach()
        return augmented[:, 0].unsqueeze(1), augmented[:, 1].unsqueeze(1)


__all__ = ["PairedIntrinsicComponentMask", "PairedMultiWindowAugment"]
