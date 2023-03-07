import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .BasicBlock import IntraGate, InterGate, ShuffleBlock


_NUM_PARAMS = 3

_LOG_SCALES_MIN = -7.
_BOUND_EPS = 0.001
_CDF_LOWER_BOUND = 1e-12


class DiscreteLogisticMixtureModel(nn.Module):
    def __init__(self, x_min: float, x_max: float, L: int, K: int):
        super().__init__()

        self.K = K

        self.x_min = x_min
        self.x_max = x_max

        self.x_lower_bound = x_min + _BOUND_EPS
        self.x_upper_bound = x_max - _BOUND_EPS
        self.bin_width = (x_max - x_min) / (L - 1)

        self.targets = torch.linspace(x_min - self.bin_width / 2, x_max - self.bin_width / 2, L + 1)

        self.counter = 1

    def forward(self, x: Tensor, params: Tensor):
        assert x.min() >= self.x_min and x.max() <= self.x_max, " {} < {} or {} > {}".format(x.min(), self.x_min, x.max(), self.x_max)

        log_weights, means, log_scales = self.split_params(params)

        x = x.unsqueeze(dim=2)
        centered_x = x - means

        inv_sigma = torch.exp(-log_scales)
        plus_in = inv_sigma * (centered_x + self.bin_width / 2)  # (x - mu + b / 2) / sigma
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_sigma * (centered_x - self.bin_width / 2)  # (x - mu - b / 2) / sigma
        cdf_min = torch.sigmoid(min_in)

        cdf_delta = cdf_plus - cdf_min

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)

        log_probs = self.get_log_probs(x=x, cdf_delta=cdf_delta, log_cdf_plus=log_cdf_plus, log_one_minus_cdf_min=log_one_minus_cdf_min)

        log_probs_weighted = log_probs + log_weights
        log_likelihoods = -1 * self.log_sum_exp(log_probs_weighted, dim=2)

        return log_likelihoods

    def get_log_probs(self, x, cdf_delta, log_cdf_plus, log_one_minus_cdf_min):
        a = torch.log(torch.clamp(cdf_delta, min=_CDF_LOWER_BOUND))
        condition = (x > self.x_upper_bound).float()
        b = condition * log_one_minus_cdf_min + (1. - condition) * a
        condition = (x < self.x_lower_bound).float()
        log_probs = condition * log_cdf_plus + (1. - condition) * b
        return log_probs

    def split_params(self, params: Tensor):
        params = rearrange(params, 'b (n c k) h w -> b n c k h w', n=_NUM_PARAMS, k=self.K)

        log_weights = torch.log_softmax(params[:, 0, ...], dim=2)
        means = params[:, 1, ...]
        log_scales = torch.clamp(params[:, 2, ...], min=_LOG_SCALES_MIN)

        return log_weights, means, log_scales

    @staticmethod
    def log_sum_exp(log_probs, dim):
        """ numerically stable log_sum_exp implementation that prevents overflow """
        m, _ = torch.max(log_probs, dim=dim)
        m_keep, _ = torch.max(log_probs, dim=dim, keepdim=True)
        return log_probs.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)


class ICEC(nn.Module):
    def __init__(self, D: int, N: int, R: int, height: int, width: int,
                 channels_F: int = 64, channels_M: int = 120, channels_Z: int = 10,
                 L: int = 25, min_: float = -1., max_: float = 1., K: int = 10):
        super().__init__()

        self.aux_head = nn.Conv2d(in_channels=channels_Z, out_channels=channels_F, kernel_size=1)

        self.intra_gate = IntraGate(height=height, width=width, channels=channels_F)

        self.shuffle_blocks = nn.Sequential(*[
            ShuffleBlock(in_channels=channels_F, out_channels=channels_M, num_res_blocks=R) for _ in range(N)
        ])

        self.inter_gate = InterGate(height=height // (2 ** D), width=width // (2 ** D), channels=channels_M)

        self.unshuffle_blocks = nn.Sequential(*[nn.PixelUnshuffle(downscale_factor=2) for _ in range(D)])
        self.conv_after_unshuffle = nn.Conv2d(in_channels=channels_F * (2 ** D), out_channels=channels_M, kernel_size=3, stride=1, padding=1)

        self.latent_conv = nn.Conv2d(in_channels=channels_M, out_channels=channels_F, kernel_size=1)

        self.params_estimator = nn.Conv2d(in_channels=channels_M, out_channels=channels_Z * K * _NUM_PARAMS, kernel_size=1)

        self.entropy_model = DiscreteLogisticMixtureModel(x_min=min_, x_max=max_, L=L, K=K)

    def forward(self, Z_t_s: torch.Tensor, Z_t_s_plus1: torch.Tensor, H_t_s_plus2: torch.Tensor, H_t_minus1_s: torch.Tensor) -> tuple:
        Z_t_s_plus1 = self.aux_head(Z_t_s_plus1)

        F_hat_t_s = self.intra_gate(aux_feats=Z_t_s_plus1, latent_feats=H_t_s_plus2)
        F_hat_t_s = self.shuffle_blocks(F_hat_t_s)

        H_t_minus1_s = self.unshuffle_blocks(H_t_minus1_s)
        H_t_minus1_s = self.conv_after_unshuffle(H_t_minus1_s)

        F_t_s = self.inter_gate(refined_feats=F_hat_t_s, latent_feats=H_t_minus1_s)

        H_t_s_plus1 = self.latent_conv(F_t_s)

        params = self.params_estimator(F_t_s)
        p_Z_t_s = self.entropy_model(Z_t_s, params=params)  # log likelihoods of Z_t^s

        return H_t_s_plus1, p_Z_t_s
