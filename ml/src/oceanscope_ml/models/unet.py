"""Plain 2D U-Net for subsurface temperature anomalies (PRD_AI_ML §5.3).

``depth_levels`` down/up stages. Each stage is two blocks of Conv3x3 -> GroupNorm(8) -> SiLU.
Down: max-pool. Up: bilinear upsample, then concatenation with the skip, then a conv stage.
The head is a 1x1 conv to the 15 depth levels. Padding to a multiple of 2**depth_levels, and the
crop back to the input size, happen inside ``forward``, so odd grid sizes work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MAX_PARAMETERS = 10_000_000  # PRD_AI_ML §5.3: at most 10 M parameters


class ConvStage(nn.Sequential):
    """Two Conv3x3 -> GroupNorm(8) -> SiLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )


class UNet(nn.Module):
    """U-Net with ``depth_levels`` down-sampling steps and ``base_channels`` at the top."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 15,
        base_channels: int = 32,
        depth_levels: int = 4,
    ) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("base_channels must be a multiple of 8 for GroupNorm(8)")
        self.depth_levels = depth_levels
        self.pad_multiple = 2**depth_levels

        widths = [base_channels * 2**i for i in range(depth_levels + 1)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for width in widths:
            self.encoders.append(ConvStage(prev, width))
            prev = width

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.decoders.append(ConvStage(prev + width, width))
            prev = width

        self.head = nn.Conv2d(prev, out_channels, kernel_size=1)

        n_params = sum(p.numel() for p in self.parameters())
        if n_params > MAX_PARAMETERS:
            raise ValueError(
                f"U-Net has {n_params:,} parameters, above the {MAX_PARAMETERS:,} limit"
            )
        self.n_parameters = n_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.pad_multiple
        pad_w = (-width) % self.pad_multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        skips: list[torch.Tensor] = []
        h = x
        for level, encoder in enumerate(self.encoders):
            h = encoder(h)
            if level < self.depth_levels:
                skips.append(h)
                h = self.pool(h)

        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.head(h)
        return out[..., :height, :width]
