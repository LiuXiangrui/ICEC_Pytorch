import torch
from torch import nn, Tensor
from torch.nn import functional as F


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
