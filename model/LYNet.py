"""V15 4-band physiological Sinc-only LYNet network for BCI Competition IV 2a.

The source directory supplied as ``LYNet`` calls the selected architecture
``RUN-EA``.  Its historical implementation imported the CX backbone from a
removed ``v3`` package.  This module keeps that architecture self-contained
and exposes it through the shared DCANet model registry.

The model deliberately has two inputs:

* ``main``: standardized task EEG with shape ``[B, 1, 22, 1000]``;
* ``side_payload['power']``: 528 run-baseline-aligned log-power features.

The zero-initialized side head is trained with a protected residual objective;
see :mod:`protocol.lynet_protocol`.
"""

from __future__ import annotations

import torch
from torch import nn

from .augmentation import PairedIntrinsicComponentMask, PairedMultiWindowAugment


NUM_CLASSES = 4
POWER_FEATURES = 4 * 6 * 22
MODEL_NAME = "LYNet"
SOURCE_VARIANT = "RUN-EA"
MAIN_PREFIXES = ("cx.", "block3.", "classifier.")
BANDS = (
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
)
INCLUDE_WIDEBAND = False


class ZeroInitializedLinear(nn.Linear):
    """Linear layer whose residual contribution is exactly zero at start."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class SincTemporalConv2d(nn.Module):
    """One learnable, bounded band-pass Sinc filter along the time axis."""

    def __init__(
        self,
        low_hz: float,
        high_hz: float,
        sample_rate: float = 250.0,
    ) -> None:
        super().__init__()
        low_hz = float(low_hz)
        high_hz = float(high_hz)
        sample_rate = float(sample_rate)
        if not 0.0 < low_hz < high_hz < sample_rate / 2.0:
            raise ValueError("Sinc band must satisfy 0 < low < high < Nyquist.")
        center0 = (low_hz + high_hz) / 2.0
        bandwidth0 = high_hz - low_hz
        if low_hz <= 3.0 or high_hz >= sample_rate / 2.0 - 3.0:
            raise ValueError("Initial band lacks room for the bounded +/-2 Hz adjustment.")
        if bandwidth0 <= 2.0:
            raise ValueError("Initial bandwidth must stay positive under bounded adjustment.")

        if center0 < 8.0:
            kernel_size = 65
        elif center0 < 16.0:
            kernel_size = 33
        else:
            kernel_size = 21

        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.initial_band = (low_hz, high_hz)
        self.raw_center = nn.Parameter(torch.zeros(()))
        self.raw_bandwidth = nn.Parameter(torch.zeros(()))
        self.register_buffer("_center0", torch.tensor(center0))
        self.register_buffer("_bandwidth0", torch.tensor(bandwidth0))
        self.register_buffer(
            "_window",
            torch.hamming_window(kernel_size, periodic=False),
        )

    def current_frequencies(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        center = self._center0 + 2.0 * torch.tanh(self.raw_center)
        bandwidth = self._bandwidth0 + 2.0 * torch.tanh(self.raw_bandwidth)
        low = center - bandwidth / 2.0
        high = center + bandwidth / 2.0
        return low, high, center, bandwidth

    def build_kernel(self) -> torch.Tensor:
        low, high, _center, _bandwidth = self.current_frequencies()
        half = self.kernel_size // 2
        time = torch.arange(
            -half,
            half + 1,
            dtype=low.dtype,
            device=low.device,
        )
        scale = self.sample_rate
        high_pass = (2.0 * high / scale) * torch.sinc(
            2.0 * high * time / scale
        )
        low_pass = (2.0 * low / scale) * torch.sinc(
            2.0 * low * time / scale
        )
        window = self._window.to(dtype=low.dtype, device=low.device)
        return ((high_pass - low_pass) * window).view(1, 1, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected Sinc input [B, 1, C, T], got {tuple(x.shape)}."
            )
        return torch.nn.functional.conv2d(
            x,
            self.build_kernel(),
            padding=(0, self.kernel_size // 2),
        )

    def diagnostics(self) -> dict[str, object]:
        low, high, center, bandwidth = self.current_frequencies()
        return {
            "initial_band": self.initial_band,
            "learned_low": low.detach(),
            "learned_high": high.detach(),
            "learned_center": center.detach(),
            "learned_bandwidth": bandwidth.detach(),
        }


class MultiBandSincFrontEnd(nn.Module):
    """Stack shared Sinc maps, optionally preceded by identity wideband EEG."""

    def __init__(
        self,
        bands: tuple[tuple[float, float], ...],
        include_wideband: bool = False,
        sample_rate: float = 250.0,
    ) -> None:
        super().__init__()
        self.include_wideband = bool(include_wideband)
        self.filters = nn.ModuleList(
            SincTemporalConv2d(low, high, sample_rate)
            for low, high in bands
        )
        self.output_channels = len(self.filters) + int(self.include_wideband)
        if self.output_channels <= 0:
            raise ValueError("Sinc front-end requires at least one output map.")

    def forward(self, x: torch.Tensor, kernels=None) -> torch.Tensor:
        maps = [x] if self.include_wideband else []
        if kernels is None:
            maps.extend(sinc_filter(x) for sinc_filter in self.filters)
        else:
            maps.extend(torch.nn.functional.conv2d(
                x, kernel, padding=(0, sinc_filter.kernel_size // 2)
            ) for sinc_filter, kernel in zip(self.filters, kernels))
        return torch.cat(maps, dim=1)

    def forward_pair(self, raw, aligned):
        # Reuse kernels only within this forward; never across optimizer steps.
        kernels = tuple(sinc_filter.build_kernel() for sinc_filter in self.filters)
        return self(raw, kernels), self(aligned, kernels)

    def diagnostics(self) -> list[dict[str, object]]:
        return [sinc_filter.diagnostics() for sinc_filter in self.filters]


class CXBlock12(nn.Module):
    """The first two residual CX stages from the selected source model."""

    def __init__(
        self,
        input_channels: int = 1,
        eeg_channels: int = 22,
        filters: int = 8,
        depth: int = 2,
    ):
        super().__init__()
        hidden = filters * 4
        output = hidden * depth
        self.output_channels = output

        self.block1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                filters,
                kernel_size=(1, 32),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(filters),
            nn.Conv2d(
                filters,
                hidden,
                kernel_size=(1, 32),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(hidden),
            nn.ELU(),
        )
        self.shortcut_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                hidden,
                output,
                kernel_size=(eeg_channels, 1),
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(output),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.shortcut_2 = nn.Sequential(
            nn.Conv2d(
                hidden,
                output,
                kernel_size=(eeg_channels, 1),
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(output),
            nn.MaxPool2d(kernel_size=(1, 4)),
        )
        self.dropout = nn.Dropout(0.5)
        self.spatial_layers = (self.block2[0], self.shortcut_2[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.block1(x) + self.shortcut_1(x)
        return self.dropout(self.block2(first) + self.shortcut_2(first))


class CXBlock3(nn.Module):
    """Final residual CX stage."""

    def __init__(
        self,
        input_channels: int = 64,
        output_channels: int = 8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=(1, 16),
                bias=False,
                padding="same",
                dilation=(1, 2),
            ),
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=(1, 16),
                bias=False,
                padding="same",
                dilation=(1, 4),
            ),
            nn.BatchNorm2d(output_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 8)),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.AvgPool2d(kernel_size=(1, 8)),
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x) + self.shortcut(x))


class LYNet(nn.Module):
    """Selected RUN-EA CX classifier with a protected log-power residual."""

    variant = SOURCE_VARIANT
    requires_side_payload = True

    def __init__(
        self,
        C: int = 22,
        T: int = 1000,
        F_filters: int = 8,
        D: int = 2,
        num_classes: int = NUM_CLASSES,
        spatial_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
        augmentation_mode: str = "time",
        augmentation_position: str = "pre_sinc",
        augmentation_schedule: str = "linear_rise",
        intrinsic_max_masked_components: int = 6,
        intrinsic_probability: float = 1.0,
    ) -> None:
        super().__init__()
        received = (int(C), int(T), int(F_filters), int(D), int(num_classes))
        expected = (22, 1000, 8, 2, 4)
        if received != expected:
            raise ValueError(
                "LYNet reproduces the selected BCI-2a RUN-EA architecture and "
                f"requires C/T/F_filters/D/classes={expected}, got {received}."
            )

        self.C = int(C)
        self.T = int(T)
        self.F_filters = int(F_filters)
        self.D = int(D)
        self.num_classes = int(num_classes)
        self.power_features = POWER_FEATURES
        self.spatial_max_norm = float(spatial_max_norm)
        self.classifier_max_norm = float(classifier_max_norm)
        if augmentation_mode != "time":
            raise ValueError("V73 requires augmentation_mode='time'.")
        self.augmentation_mode = augmentation_mode
        if augmentation_position != "pre_sinc":
            raise ValueError("V74 requires augmentation_position='pre_sinc'.")
        self.augmentation_position = augmentation_position
        if augmentation_schedule != "linear_rise":
            raise ValueError("V88 requires augmentation_schedule='linear_rise'.")
        self.augmentation_schedule = augmentation_schedule
        self.intrinsic_masking = PairedIntrinsicComponentMask(
            max_masked_components=intrinsic_max_masked_components,
            probability=intrinsic_probability,
        )
        self.augmentation = PairedMultiWindowAugment(
            probability=0.5,
            focus_samples=500,
            focus_starts=(0, 125, 250, 375, 500),
            inside_gain=1.5,
            outside_gain=0.5,
        )

        # Registration order mirrors the selected source model: the complete
        # main path is registered before the protected residual head.
        self.sinc_frontend = MultiBandSincFrontEnd(
            BANDS,
            include_wideband=INCLUDE_WIDEBAND,
            sample_rate=250.0,
        )
        self.cx = CXBlock12(
            input_channels=self.sinc_frontend.output_channels,
            eeg_channels=C,
            filters=F_filters,
            depth=D,
        )
        self.block3 = CXBlock3(
            input_channels=self.cx.output_channels,
            output_channels=F_filters,
        )
        feature_count = F_filters * (T // 4 // 8)
        self.classifier = nn.Linear(feature_count, num_classes)
        self.power_head = ZeroInitializedLinear(
            POWER_FEATURES,
            num_classes,
            bias=False,
        )
        self.fusion_norm = nn.LayerNorm(feature_count * 2)
        self.gate_hidden = nn.Linear(feature_count * 2, 32)
        self.gate_output = nn.Linear(32, feature_count)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def apply_local_initialization(self, init_fn) -> None:
        """Initialize the baseline path without overwriting attention/gate init."""

        self.cx.apply(init_fn)
        self.block3.block.apply(init_fn)
        self.block3.shortcut.apply(init_fn)
        self.classifier.apply(init_fn)
        nn.init.zeros_(self.power_head.weight)
        self.gate_hidden.apply(init_fn)
        self.gate_output.apply(init_fn)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    @staticmethod
    @torch.no_grad()
    def _max_norm(layer: nn.Conv2d | nn.Linear, limit: float) -> None:
        layer.weight.copy_(
            torch.renorm(layer.weight, p=2, dim=0, maxnorm=float(limit))
        )

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        for layer in self.cx.spatial_layers:
            self._max_norm(layer, self.spatial_max_norm)
        self._max_norm(self.classifier, self.classifier_max_norm)
        self._max_norm(self.power_head, self.classifier_max_norm)

    def _extract_representation(self, spectral_maps: torch.Tensor) -> torch.Tensor:
        """Return the main-path feature consumed by the protected classifier."""

        return self.block3(self.cx(spectral_maps)).flatten(start_dim=1)

    def forward(
        self,
        main: torch.Tensor,
        side_payload: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (1, self.C, self.T)
        if main.ndim != 4 or tuple(main.shape[1:]) != expected:
            raise ValueError(
                f"Expected main [B, 1, {self.C}, {self.T}], got {tuple(main.shape)}."
            )
        payload = {} if side_payload is None else side_payload
        power = payload.get("power")
        if power is None or power.ndim != 2 or power.shape[1] != POWER_FEATURES:
            shape = None if power is None else tuple(power.shape)
            raise ValueError(
                f"Expected side_payload['power'] [B, {POWER_FEATURES}], got {shape}."
            )
        if power.shape[0] != main.shape[0]:
            raise ValueError(
                f"Main/power batch mismatch: {main.shape[0]} != {power.shape[0]}."
            )
        aligned_main = payload.get("aligned_main")
        if aligned_main is None or aligned_main.shape != main.shape:
            shape = None if aligned_main is None else tuple(aligned_main.shape)
            raise ValueError(
                f"Expected side_payload['aligned_main'] {tuple(main.shape)}, got {shape}."
            )

        main, aligned_main = self.intrinsic_masking(
            main, aligned_main, cache=payload.get("intrinsic_cache")
        )
        main, aligned_main = self.augmentation(main, aligned_main)
        raw_maps, aligned_maps = self.sinc_frontend.forward_pair(main, aligned_main)
        raw_representation = self._extract_representation(raw_maps)
        aligned_representation = self._extract_representation(aligned_maps)
        concatenated = torch.cat(
            (raw_representation, aligned_representation),
            dim=1,
        )
        gate = torch.sigmoid(
            self.gate_output(
                torch.nn.functional.elu(
                    self.gate_hidden(self.fusion_norm(concatenated))
                )
            )
        )
        representation = (
            gate * raw_representation
            + (1.0 - gate) * aligned_representation
        )
        main_logits = self.classifier(representation)
        side_logits = self.power_head(power.detach())
        combined = main_logits + side_logits
        if not return_aux:
            return combined

        main_rms = torch.sqrt(main_logits.square().mean(dim=1) + 1e-6)
        side_rms = torch.sqrt(side_logits.square().mean(dim=1) + 1e-6)
        return combined, {
            "main_logits": main_logits,
            "side_logits": side_logits,
            "raw_representation": raw_representation,
            "aligned_representation": aligned_representation,
            "gate": gate,
            "fused_representation": representation,
            "sinc_bands": self.sinc_frontend.diagnostics(),
            "augmentation_mode": self.augmentation_mode,
            "augmentation_position": self.augmentation_position,
            "intrinsic_masked_counts": self.intrinsic_masking.last_masked_counts,
            "b_logits": side_logits,
            "main_logit_rms": main_rms,
            "side_logit_rms": side_rms,
            "side_main_logit_rms_ratio": side_rms / (main_rms + 1e-6),
        }


# Compatibility name used by the supplied source implementation.
RunBaselineEANet = LYNet


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "BANDS",
    "CXBlock12",
    "CXBlock3",
    "LYNet",
    "MultiBandSincFrontEnd",
    "MAIN_PREFIXES",
    "MODEL_NAME",
    "NUM_CLASSES",
    "POWER_FEATURES",
    "RunBaselineEANet",
    "SOURCE_VARIANT",
    "SincTemporalConv2d",
    "ZeroInitializedLinear",
    "count_trainable_parameters",
]
