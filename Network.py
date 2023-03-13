import math

import torch
import torch.nn as nn

from Modules.ICEC import ICEC_AUX, ICEC_SLICE
from Modules.FeatsExtraction import FeatsExtraction
from Modules.Quantization import SoftQuantizer
from Modules.EntropyModel import NUM_PARAMS, DiscreteLogisticMixtureModel


class Network(nn.Module):
    def __init__(self, channels_F: int = 64, channels_M: int = 120, channels_Z: int = 10,
                 K: int = 10, L_slice: int = 256, L_aux: int = 25,
                 sigma: float = 2, aux_min: float = -1, aux_max: float = 1, slice_min: float = 0, slice_max: float = 255):
        super().__init__()

        self.channels_X = 1

        self.icec_blocks = nn.ModuleList([
            ICEC_SLICE(N=3, R=4,
                       channels_F=channels_F, channels_M=channels_M, channels_Z=channels_Z,
                       L=L_slice, max_=slice_max, min_=slice_min, K=K),
            ICEC_AUX(D=3, N=1, R=8,
                     channels_F=channels_F, channels_M=channels_M, channels_Z=channels_Z,
                     L=L_aux, max_=aux_max, min_=aux_min, K=K),
            ICEC_AUX(D=1, N=1, R=8,
                     channels_F=channels_F, channels_M=channels_M, channels_Z=channels_Z,
                     L=L_aux, max_=aux_max, min_=aux_min, K=K),
        ])

        self.param_estimator_for_Z_t_3_with_H_t_minus1_3 = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(in_channels=4 * channels_F, out_channels=channels_Z * K * NUM_PARAMS, kernel_size=1)
        )

        self.entropy_model_for_Z_t_3_with_H_t_minus1_3 = DiscreteLogisticMixtureModel(x_min=aux_min, x_max=aux_max, K=K, L=L_aux)

        self.conv_latent_slice = nn.Conv2d(in_channels=self.channels_X, out_channels=channels_F, kernel_size=3, stride=1, padding=1)

        self.feats_extractions = nn.ModuleList([
            FeatsExtraction(in_channels=self.channels_X, out_channels=channels_F, num_feats_extraction=3, R=4),
            FeatsExtraction(in_channels=channels_F, out_channels=channels_F, num_feats_extraction=1, R=8),
            FeatsExtraction(in_channels=channels_F, out_channels=channels_F, num_feats_extraction=1, R=8)
        ])

        self.conv_ahead_of_quant_list = nn.ModuleList([
            nn.Conv2d(in_channels=channels_F, out_channels=channels_Z, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=channels_F, out_channels=channels_Z, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=channels_F, out_channels=channels_Z, kernel_size=3, stride=1, padding=1),
        ])

        self.soft_quantizer = SoftQuantizer(L=L_aux, min_=aux_min, max_=aux_max, sigma=sigma)

    def forward(self, X_t: torch.Tensor, H_t_minus1_3: torch.Tensor = None,  H_t_minus1_2: torch.Tensor = None,
                H_t_minus1_1: torch.Tensor = None, H_t_minus1_0: torch.Tensor = None):
        num_pixels = X_t.shape[0] * X_t.shape[1] * X_t.shape[2] * X_t.shape[3]

        # generate auxiliary features
        E_t_1 = self.feats_extractions[0](X_t)
        Z_t_1 = self.soft_quantizer(self.conv_ahead_of_quant_list[0](E_t_1))["soft"]  # shape (channels_Z, H/8, W/8)

        E_t_2 = self.feats_extractions[1](E_t_1)
        Z_t_2 = self.soft_quantizer(self.conv_ahead_of_quant_list[1](E_t_2))["soft"]  # shape (channels_Z, H/16, W/16)

        E_t_3 = self.feats_extractions[2](E_t_2)
        Z_t_3 = self.soft_quantizer(self.conv_ahead_of_quant_list[2](E_t_3))["soft"]  # shape (channels_Z, H/32, W/32)

        # estimate the likelihoods for Z_t^3
        if H_t_minus1_3 is not None:
            params = self.param_estimator_for_Z_t_3_with_H_t_minus1_3(H_t_minus1_3)
            p_Z_t_3 = self.entropy_model_for_Z_t_3_with_H_t_minus1_3(Z_t_3, params=params)
        else:
            p_Z_t_3 = torch.ones_like(Z_t_3) / 25  # uniform distribution

        """
        H_t_3 shape is (channels_F, H/16, W/16)
        H_t_2 shape is (channels_F, H/8, W/8)
        H_t_1 shape is (channels_F, H, W)
        H_t_0 shape is (channels_F, H, W)
        """

        H_t_3, p_Z_t_2 = self.icec_blocks[-1](Z_t_s=Z_t_2, Z_t_s_plus1=Z_t_3, H_t_s_plus2=None, H_t_minus1_s=H_t_minus1_2)

        H_t_2, p_Z_t_1 = self.icec_blocks[-2](Z_t_s=Z_t_1, Z_t_s_plus1=Z_t_2, H_t_s_plus2=H_t_3, H_t_minus1_s=H_t_minus1_1)

        H_t_1, p_X_t = self.icec_blocks[-3](X_t=X_t, Z_t_1=Z_t_1, H_t_2=H_t_2, H_t_minus1_0=H_t_minus1_0)

        H_t_0 = self.conv_latent_slice(X_t)

        log_likelihoods = [p_X_t, p_Z_t_1, p_Z_t_2, p_Z_t_3]

        bpp_list = [l.sum() / math.log(2.) / num_pixels for l in log_likelihoods]

        return {
            "LatentFeats": [H_t_0, H_t_1, H_t_2, H_t_3],
            "LogLikelihoods": log_likelihoods,
            "Bpp": bpp_list
        }


if __name__ == "__main__":
    net = Network()
    h = w = 256

    a = torch.randn(1, 1, h, w)
    a = (a - a.min()) / (a.max() - a.min()) * 255.
    a = torch.round(a)
    net(a)

    b = torch.randn(1, 64, h // 16, w // 16)
    c = torch.randn(1, 64, h // 8, w // 8)
    d = torch.randn(1, 64, h, w)
    e = torch.randn(1, 64, h, w)

    f = net(a, b, c, d, e)
    print('lol')
