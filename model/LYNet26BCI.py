"""V121/RUN-EA network adapted to the 26BCI binary dataset.

The BCI-2a implementation remains untouched.  This variant preserves the
dual raw/aligned path and protected log-power residual, while changing the
data contract to 16 channels, 500 samples at 500 Hz, and two classes.
"""

from __future__ import annotations

import torch
from torch import nn

from .augmentation import PairedIntrinsicComponentMask, PairedMultiWindowAugment
from .LYNet import (
    CXBlock12,
    CXBlock3,
    MultiBandSincFrontEnd,
    ZeroInitializedLinear,
)


NUM_CLASSES = 2
POWER_BANDS_HZ = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 32.0),
)
SINC_BANDS_HZ = (
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
)
POWER_FEATURES = len(POWER_BANDS_HZ) * 16


class LYNet26BCI(nn.Module):
    """16-channel binary V121 with session-level RUN-EA side information."""

    variant = "RUN-EA-26BCI"
    requires_side_payload = True

    def __init__(
        self,
        C: int = 16,
        T: int = 500,
        F_filters: int = 8,
        D: int = 2,
        num_classes: int = NUM_CLASSES,
        sample_rate: float = 500.0,
        power_features: int = POWER_FEATURES,
        spatial_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
        augmentation_mode: str = "time",
        augmentation_position: str = "pre_sinc",
        augmentation_schedule: str = "linear_rise",
        intrinsic_max_masked_components: int = 6,
        intrinsic_probability: float = 1.0,
    ) -> None:
        super().__init__()
        received = (int(C), int(T), int(num_classes), int(power_features))
        expected = (16, 500, 2, POWER_FEATURES)
        if received != expected:
            raise ValueError(
                "LYNet26BCI requires C/T/classes/power_features="
                f"{expected}, got {received}."
            )
        if augmentation_mode != "time" or augmentation_position != "pre_sinc":
            raise ValueError("LYNet26BCI requires time-domain pre-Sinc augmentation.")
        if augmentation_schedule != "linear_rise":
            raise ValueError("LYNet26BCI requires augmentation_schedule='linear_rise'.")

        self.C = int(C)
        self.T = int(T)
        self.num_classes = int(num_classes)
        self.power_features = int(power_features)
        self.spatial_max_norm = float(spatial_max_norm)
        self.classifier_max_norm = float(classifier_max_norm)
        self.augmentation_mode = augmentation_mode
        self.augmentation_position = augmentation_position
        self.augmentation_schedule = augmentation_schedule

        self.intrinsic_masking = PairedIntrinsicComponentMask(
            max_masked_components=min(int(intrinsic_max_masked_components), C - 1),
            probability=float(intrinsic_probability),
        )
        self.augmentation = PairedMultiWindowAugment(
            probability=0.5,
            focus_samples=250,
            focus_starts=(0, 62, 125, 188, 250),
            inside_gain=1.5,
            outside_gain=0.5,
        )
        self.sinc_frontend = MultiBandSincFrontEnd(
            SINC_BANDS_HZ,
            include_wideband=False,
            sample_rate=float(sample_rate),
        )
        # The source kernels were sized for 250 Hz.  Preserve their physical
        # durations when this dataset doubles the sampling rate to 500 Hz.
        for sinc_filter in self.sinc_frontend.filters:
            source_half_width = sinc_filter.kernel_size // 2
            scaled_half_width = int(round(source_half_width * float(sample_rate) / 250.0))
            sinc_filter.kernel_size = 2 * scaled_half_width + 1
            sinc_filter._window = torch.hamming_window(
                sinc_filter.kernel_size, periodic=False
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
        feature_count = int(F_filters) * (int(T) // 4 // 8)
        self.classifier = nn.Linear(feature_count, num_classes)
        self.power_head = ZeroInitializedLinear(power_features, num_classes, bias=False)
        self.fusion_norm = nn.LayerNorm(feature_count * 2)
        self.gate_hidden = nn.Linear(feature_count * 2, 32)
        self.gate_output = nn.Linear(32, feature_count)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def apply_local_initialization(self, init_fn) -> None:
        self.cx.apply(init_fn)
        self.block3.block.apply(init_fn)
        self.block3.shortcut.apply(init_fn)
        self.classifier.apply(init_fn)
        self.gate_hidden.apply(init_fn)
        self.gate_output.apply(init_fn)
        nn.init.zeros_(self.power_head.weight)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    @staticmethod
    @torch.no_grad()
    def _max_norm(layer: nn.Conv2d | nn.Linear, limit: float) -> None:
        layer.weight.copy_(torch.renorm(layer.weight, p=2, dim=0, maxnorm=limit))

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        for layer in self.cx.spatial_layers:
            self._max_norm(layer, self.spatial_max_norm)
        self._max_norm(self.classifier, self.classifier_max_norm)
        self._max_norm(self.power_head, self.classifier_max_norm)

    def _representation(self, maps: torch.Tensor) -> torch.Tensor:
        return self.block3(self.cx(maps)).flatten(start_dim=1)

    def forward(
        self,
        main: torch.Tensor,
        side_payload: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ):
        expected = (1, self.C, self.T)
        if main.ndim != 4 or tuple(main.shape[1:]) != expected:
            raise ValueError(f"Expected main [B,1,{self.C},{self.T}], got {tuple(main.shape)}.")
        payload = {} if side_payload is None else side_payload
        power = payload.get("power")
        aligned = payload.get("aligned_main")
        if power is None or power.ndim != 2 or power.shape[1] != self.power_features:
            raise ValueError(f"Expected power [B,{self.power_features}].")
        if aligned is None or aligned.shape != main.shape:
            raise ValueError("aligned_main must have the same shape as main.")

        main, aligned = self.intrinsic_masking(
            main, aligned, cache=payload.get("intrinsic_cache")
        )
        main, aligned = self.augmentation(main, aligned)
        raw_maps, aligned_maps = self.sinc_frontend.forward_pair(main, aligned)
        raw_rep = self._representation(raw_maps)
        aligned_rep = self._representation(aligned_maps)
        joined = torch.cat((raw_rep, aligned_rep), dim=1)
        gate = torch.sigmoid(
            self.gate_output(torch.nn.functional.elu(self.gate_hidden(self.fusion_norm(joined))))
        )
        representation = gate * raw_rep + (1.0 - gate) * aligned_rep
        main_logits = self.classifier(representation)
        side_logits = self.power_head(power.detach())
        logits = main_logits + side_logits
        if not return_aux:
            return logits
        return logits, {
            "main_logits": main_logits,
            "side_logits": side_logits,
            "raw_representation": raw_rep,
            "aligned_representation": aligned_rep,
            "gate": gate,
            "sinc_bands": self.sinc_frontend.diagnostics(),
        }


__all__ = ["LYNet26BCI", "POWER_BANDS_HZ", "POWER_FEATURES"]
