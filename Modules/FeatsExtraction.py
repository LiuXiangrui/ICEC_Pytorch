import torch
from torch import nn

from Modules.BasicBlock import ResBlock


class FeatsExtractionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, R: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
        self.res_blocks = nn.Sequential(*[ResBlock(channels=out_channels) for _ in range(R)])
        self.tail = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.tail(x)
        return x


class FeatsExtraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, R: int, num_feats_extraction: int):
        super().__init__()

        body = [
            FeatsExtractionBlock(in_channels=in_channels, out_channels=out_channels, R=R),
        ]
        body.extend([
            FeatsExtractionBlock(in_channels=out_channels, out_channels=out_channels, R=R) for _ in range(num_feats_extraction - 1)
        ])

        self.body = nn.Sequential(*body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
