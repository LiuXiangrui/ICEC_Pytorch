import math

import torch
import torch.nn as nn

from ICEC import ICEC_AUX, ICEC_SLICE
from FeatsExtraction import FeatsExtraction
from Modules.Quantization import SoftQuantizer
from EntropyModel import _NUM_PARAMS, DiscreteLogisticMixtureModel


class Network(nn.Module):
    def __init__(self, height: int, width: int,
                 channels_F: int = 64, channels_M: int = 120, channels_Z: int = 10,
                 K: int = 10, L_slice: int = 256, L_aux: int = 25,
                 aux_min_: int = -1, aux_max_: int = 1, slice_min_: int = 0, slice_max_: int = 255):
        super().__init__()

        self.icec_blocks = nn.ModuleList([
            ICEC_SLICE(height=height // (2 ** 3), width=width // (2 ** 3),
                       N=3, R=4,
                       channels_F=channels_F, channels_M=channels_M, channels_Z=channels_Z,
                       L=L_slice, max_=slice_max_, min_=slice_min_, K=K),
            ICEC_AUX(height=height // (2 ** 4), width=width // (2 ** 4),
                     D=3, N=1, R=8,
                     channels_F=channels_F, channels_M=channels_M, channels_Z=channels_Z,
                     L=L_aux, max_=aux_max_, min_=aux_min_, K=K),
            ICEC_AUX(height=height // (2 ** 5), width=width // (2 ** 5),
                     D=1, N=1, R=8,
                     channels_F=channels_F, channels_M=channels_M, channels_Z=channels_Z,
                     L=L_aux, max_=aux_max_, min_=aux_min_, K=K),
        ])

        self.param_estimator_for_Z_t_3_with_H_t_minus1_3 = nn.Conv2d(in_channels=channels_F, out_channels=channels_Z * K * _NUM_PARAMS, kernel_size=1)
        self.entropy_model_for_Z_t_3_with_H_t_minus1_3 = DiscreteLogisticMixtureModel(x_min=aux_min_, x_max=aux_max_, K=K, L=L_aux)

        self.conv_latent_slice = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.feats_extractions = nn.ModuleList([
            FeatsExtraction(in_channels=1, out_channels=64, num_feats_extraction=3, R=4),
            FeatsExtraction(in_channels=64, out_channels=64, num_feats_extraction=1, R=8),
            FeatsExtraction(in_channels=64, out_channels=64, num_feats_extraction=1, R=8)
        ])

        self.conv_ahead_of_quant_list = nn.ModuleList([
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, stride=1, padding=1),
        ])

        self.soft_quantizer = SoftQuantizer(L=25, min_=-1., max_=1., sigma=2.)

    def forward(self, X_t: torch.Tensor, H_t_minus1_3: torch.Tensor = None,  H_t_minus1_2: torch.Tensor = None,
                H_t_minus1_1: torch.Tensor = None, H_t_minus1_0: torch.Tensor = None):
        num_pixels = X_t.shape[0] * X_t.shape[1] * X_t.shape[2] * X_t.shape[3]

        # generate auxiliary features
        E_t_1 = self.feats_extractions[0](X_t)
        Z_t_1 = self.soft_quantizer(self.conv_ahead_of_quant_list[0](E_t_1))["soft"]

        E_t_2 = self.feats_extractions[1](E_t_1)
        Z_t_2 = self.soft_quantizer(self.conv_ahead_of_quant_list[1](E_t_2))["soft"]

        E_t_3 = self.feats_extractions[2](E_t_2)
        Z_t_3 = self.soft_quantizer(self.conv_ahead_of_quant_list[2](E_t_3))["soft"]

        # estimate the likelihoods for Z_t^3
        if H_t_minus1_3 is not None:
            params = self.param_estimator_for_Z_t_3_with_H_t_minus1_3(H_t_minus1_3)
            p_Z_t_3 = self.entropy_model_for_Z_t_3_with_H_t_minus1_3(Z_t_3, params=params)
        else:
            p_Z_t_3 = torch.ones_like(Z_t_3) / 25  # uniform distribution

        H_t_3, p_Z_t_2 = self.icec_blocks[-1](Z_t_s=Z_t_2, Z_t_s_plus1=Z_t_3, H_t_s_plus2=None, H_t_minus1_s=H_t_minus1_2)

        H_t_2, p_Z_t_1 = self.icec_blocks[-2](Z_t_s=Z_t_1, Z_t_s_plus1=Z_t_2, H_t_s_plus2=H_t_3, H_t_minus1_s=H_t_minus1_1)

        H_t_1, p_X_t = self.icec_blocks[-3](X_t=X_t, Z_t_1=Z_t_1, H_t_2=H_t_2, H_t_minus1_0=H_t_minus1_0)

        H_t_0 = self.conv_latent_slice(X_t)

        log_likelihoods = [p_X_t, p_Z_t_1, p_Z_t_2, p_Z_t_3]

        bpp_list = [i.sum() / math.log(2.) / num_pixels for i in log_likelihoods]

        return {
            "LatentFeats": [H_t_0, H_t_1, H_t_2, H_t_3],
            "LogLikelihoods": log_likelihoods,
            "Bpp": bpp_list
        }


if __name__ == "__main__":
    net = Network(height=256, width=256)

    a = torch.randn(1, 1, 256, 256)
    a = (a - a.min()) / (a.max() - a.min()) * 255.
    a = torch.round(a)
    net(a)

    b = torch.randn(1, 64, 8, 8)
    c = torch.randn(1, 64, 16, 16)
    d = torch.randn(1, 64, 32, 32)
    e = torch.randn(1, 64, 256, 256)

    net(a, b, c, d, e)
