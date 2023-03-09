import torch
from einops import rearrange
from torch import nn as nn, Tensor
from torch.nn import functional as F

NUM_PARAMS = 3
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

    def get_log_probs(self, x: torch.Tensor, cdf_delta: torch.Tensor, log_cdf_plus: torch.Tensor, log_one_minus_cdf_min: torch.Tensor) -> torch.Tensor:
        a = torch.log(torch.clamp(cdf_delta, min=_CDF_LOWER_BOUND))
        condition = (x > self.x_upper_bound).float()
        b = condition * log_one_minus_cdf_min + (1. - condition) * a
        condition = (x < self.x_lower_bound).float()
        log_probs = condition * log_cdf_plus + (1. - condition) * b
        return log_probs

    def split_params(self, params: Tensor) -> tuple:
        params = rearrange(params, 'b (n c k) h w -> b n c k h w', n=NUM_PARAMS, k=self.K)

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
