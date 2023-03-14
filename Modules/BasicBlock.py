import torch
from einops import rearrange
from torch import nn as nn


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
    def __init__(self, channels: int):
        super().__init__()

        self.w_1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.w_2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.w_3 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, aux_feats: torch.Tensor, latent_feats: torch.Tensor) -> torch.Tensor:
        latent_feats = self.w_1(latent_feats)
        aux_feats = self.w_2(aux_feats)

        attn_map = torch.sigmoid(latent_feats * aux_feats)

        refined_feats = latent_feats * attn_map + self.w_3(aux_feats)

        return refined_feats


class InterGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.w_4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.w_5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, latent_feats: torch.Tensor, refined_feats: torch.Tensor) -> torch.Tensor:
        _, _, height, width = latent_feats.shape
        latent_feats = self.w_4(latent_feats)
        latent_feats = rearrange(latent_feats, "b c h w -> b c (h w)")

        attn_map = torch.softmax(torch.matmul(rearrange(refined_feats, "b c h w -> b (h w) c"), latent_feats), dim=-1)

        context_feats = self.w_5(refined_feats) + rearrange(torch.matmul(latent_feats, attn_map.permute(0, 2, 1)), "b c (h w) -> b c h w", h=height, w=width)

        return context_feats


class ShuffleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int):
        super().__init__()
        self.head = nn.PixelShuffle(upscale_factor=2)
        self.res_blocks = nn.Sequential(*[ResBlock(channels=in_channels // 4) for _ in range(num_res_blocks)])
        self.tail = nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.res_blocks(x)
        x = self.tail(x)
        return x


class StackedAtrousConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.atrous_convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, dilation=4, padding=4),

        ])

        self.output_conv = nn.Conv2d(in_channels=3 * in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([atrous_conv(x) for atrous_conv in self.atrous_convs], dim=1)
        x = self.output_conv(x)

        return x
