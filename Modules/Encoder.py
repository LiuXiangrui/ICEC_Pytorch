import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from BasicBlock import ResBlock


class SoftQuantizer(nn.Module):
    def __init__(self, L: int, min_: float, max_: float, sigma: float):
        super(SoftQuantizer, self).__init__()
        self.register_buffer(name='levels', tensor=torch.linspace(min_, max_, L))
        self.sigma = sigma  # softness factor

    def forward(self, x: Tensor):
        assert x.dim() == 4, 'Expected input tensor have shape [B, C, H, W], but got {}'.format(x.size())
        N, C, H, W = x.shape
        x = x.view(N, C, H * W, 1)
        d = torch.pow(x - self.levels, exponent=2)
        phi_soft = F.softmax(-self.sigma * d, dim=-1)
        x_soft = torch.sum(self.levels * phi_soft, dim=-1)
        x_soft = x_soft.view(N, C, H, W)
        _, symbols_hard = torch.min(d.detach(), dim=-1)
        symbols_hard = symbols_hard.view(N, C, H, W)
        x_hard = self.levels[symbols_hard]
        x_soft.data = x_hard  # assign data, keep gradient
        return {
            'soft': x_soft,
            'hard': x_hard,
            'symbols': symbols_hard
        }


class FeatureExtraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, R: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=1)
        self.res_blocks = nn.Sequential(*[ResBlock(channels=out_channels) for _ in range(R)])
        self.tail = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.tail(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, R: int, num_feats_extraction: int):
        super().__init__()

        body = [
            FeatureExtraction(in_channels=in_channels, out_channels=out_channels, R=R),
        ]
        body.extend([
            FeatureExtraction(in_channels=out_channels, out_channels=out_channels, R=R) for _ in range(num_feats_extraction - 1)
        ])

        self.body = nn.Sequential(*body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MultiscaleEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, feats_channels: int = 64, aux_channels: int = 10,
                 L: int = 25, sigma: float = 2, min_: float = -1, max_: float = 1):
        super().__init__()
        self.encoder_list = nn.ModuleList([
            Encoder(in_channels=in_channels, out_channels=feats_channels, num_feats_extraction=3, R=4),
            Encoder(in_channels=feats_channels, out_channels=feats_channels, num_feats_extraction=1, R=8),
            Encoder(in_channels=feats_channels, out_channels=feats_channels, num_feats_extraction=1, R=8)
        ])

        self.conv_ahead_of_quant_list = nn.ModuleList([
            nn.Conv2d(in_channels=feats_channels, out_channels=aux_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=feats_channels, out_channels=aux_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=feats_channels, out_channels=aux_channels, kernel_size=3, stride=1, padding=1),
        ])

        self.soft_quantizer = SoftQuantizer(L=L, min_=min_, max_=max_, sigma=sigma)

    def forward(self, X_t: torch.Tensor) -> list:
        Z_t_list = []

        E_t_i = X_t
        for encoder, conv_ahead_of_quant in zip(self.encoder_list, self.conv_ahead_of_quant_list):
            E_t_i = encoder(E_t_i)
            Z_t_list.append(conv_ahead_of_quant(E_t_i)["x_soft"])

        return Z_t_list
