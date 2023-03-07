import torch
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class IntraGate(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        self.w_1 = torch.nn.Parameter(torch.ones([1, channels, height, width]), requires_grad=True)
        self.w_2 = torch.nn.Parameter(torch.ones([1, channels, height, width]), requires_grad=True)
        self.w_3 = torch.nn.Parameter(torch.ones([1, channels, height, width]), requires_grad=True)

    def forward(self, aux_feats: torch.Tensor, latent_feats: torch.Tensor) -> torch.Tensor:
        latent_feats = latent_feats * self.w_1
        aux_feats = aux_feats * self.w_2

        attn_map = F.sigmoid(latent_feats * aux_feats)

        refined_feats = latent_feats * attn_map + aux_feats * self.w_3

        return refined_feats


class InterGate(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        self.height = height
        self.width = width
        self.w_4 = torch.nn.Parameter(torch.ones([1, channels, height, width]), requires_grad=True)
        self.w_5 = torch.nn.Parameter(torch.ones([1, channels, height, width]), requires_grad=True)

    def forward(self, latent_feats: torch.Tensor, refined_feats: torch.Tensor) -> torch.Tensor:
        latent_feats = latent_feats * self.w_4
        latent_feats = rearrange(latent_feats, "b c h w -> b c (h w)")

        attn_map = F.softmax(torch.mm(rearrange(refined_feats, "b c h w -> b (h w) c"), latent_feats), dim=-2)

        context_feats = refined_feats * self.w_5 + rearrange(torch.mm(latent_feats, attn_map),
                                                             "b c (h w) -> b c h w", h=self.height, w=self.width)

        return context_feats


class ShuffleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int):
        super().__init__()
        self.head = nn.PixelShuffle(upscale_factor=2)
        self.res_blocks = nn.Sequential(*[ResBlock(channels=in_channels // 4) for _ in range(num_res_blocks)])
        self.tail = nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.res_blocks(x)
        x = self.tail(x)
        return x
