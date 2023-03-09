import torch
import torch.nn as nn

from BasicBlock import IntraGate, InterGate, ShuffleBlock
from EntropyModel import _NUM_PARAMS, DiscreteLogisticMixtureModel


class ICEC_AUX(nn.Module):
    def __init__(self, D: int, N: int, R: int, height: int, width: int,
                 channels_F: int = 64, channels_M: int = 120, channels_Z: int = 10,
                 L: int = 25, min_: float = -1., max_: float = 1., K: int = 10):
        super().__init__()

        self.aux_head = nn.Conv2d(in_channels=channels_Z, out_channels=channels_F, kernel_size=1)

        self.intra_gate = IntraGate(height=height, width=width, channels=channels_F)

        shuffle_blocks = [ShuffleBlock(in_channels=channels_F, out_channels=channels_M, num_res_blocks=R), ]
        shuffle_blocks.extend([ShuffleBlock(in_channels=channels_M, out_channels=channels_M, num_res_blocks=R) for _ in range(N - 1)])
        self.shuffle_blocks = nn.Sequential(*shuffle_blocks)

        self.inter_gate = InterGate(height=height // (2 ** D), width=width // (2 ** D), channels=channels_M)

        self.unshuffle_blocks = nn.Sequential(*[nn.PixelUnshuffle(downscale_factor=2) for _ in range(D)])
        self.conv_after_unshuffle = nn.Conv2d(in_channels=channels_F * (4 ** D), out_channels=channels_M, kernel_size=3, stride=1, padding=1)

        self.latent_conv = nn.Conv2d(in_channels=channels_M, out_channels=channels_F, kernel_size=1)

        self.params_estimator = nn.Conv2d(in_channels=channels_M, out_channels=channels_Z * K * _NUM_PARAMS, kernel_size=1)

        self.entropy_model = DiscreteLogisticMixtureModel(x_min=min_, x_max=max_, L=L, K=K)

    def forward(self, Z_t_s: torch.Tensor,
                Z_t_s_plus1: torch.Tensor, H_t_s_plus2: torch.Tensor = None, H_t_minus1_s: torch.Tensor = None) -> tuple:
        Z_t_s_plus1 = self.aux_head(Z_t_s_plus1)

        if H_t_s_plus2 is not None:
            F_hat_t_s = self.intra_gate(aux_feats=Z_t_s_plus1, latent_feats=H_t_s_plus2)
        else:
            F_hat_t_s = Z_t_s_plus1

        F_hat_t_s = self.shuffle_blocks(F_hat_t_s)

        if H_t_minus1_s is not None:
            H_t_minus1_s = self.unshuffle_blocks(H_t_minus1_s)
            H_t_minus1_s = self.conv_after_unshuffle(H_t_minus1_s)

            F_t_s = self.inter_gate(refined_feats=F_hat_t_s, latent_feats=H_t_minus1_s)
        else:
            F_t_s = F_hat_t_s

        H_t_s_plus1 = self.latent_conv(F_t_s)

        params = self.params_estimator(F_t_s)
        p_Z_t_s = self.entropy_model(Z_t_s, params=params)  # log likelihoods of Z_t^s

        return H_t_s_plus1, p_Z_t_s


class ICEC_SLICE(nn.Module):
    def __init__(self, height: int, width: int,
                 channels_F: int = 64, channels_M: int = 120, channels_Z: int = 10,
                 N: int = 3, R: int = 4,
                 L: int = 256, min_: float = 0, max_: float = 255, K: int = 10):
        super().__init__()

        self.aux_head = nn.Conv2d(in_channels=channels_Z, out_channels=channels_F, kernel_size=1)

        self.intra_gate = IntraGate(height=height, width=width, channels=channels_F)

        shuffle_blocks = [ShuffleBlock(in_channels=channels_F, out_channels=channels_M, num_res_blocks=R), ]
        shuffle_blocks.extend([ShuffleBlock(in_channels=channels_M, out_channels=channels_M, num_res_blocks=R) for _ in range(N - 1)])
        self.shuffle_blocks = nn.Sequential(*shuffle_blocks)

        self.latent_conv_wo_H_t_minus1_0 = nn.Conv2d(in_channels=channels_M, out_channels=channels_F, kernel_size=1)
        self.latent_conv_with_H_t_minus1_0 = nn.Conv2d(in_channels=channels_M + channels_F, out_channels=channels_F, kernel_size=1)

        self.params_estimator_wo_H_t_minus1_0 = nn.Conv2d(in_channels=channels_M, out_channels=channels_Z * K * _NUM_PARAMS, kernel_size=1)
        self.params_estimator_with_H_t_minus1_0 = nn.Conv2d(in_channels=channels_M + channels_F, out_channels=channels_Z * K * _NUM_PARAMS, kernel_size=1)

        self.entropy_model = DiscreteLogisticMixtureModel(x_min=min_, x_max=max_, L=L, K=K)

    def forward(self, X_t: torch.Tensor,
                Z_t_1: torch.Tensor, H_t_2: torch.Tensor, H_t_minus1_0: torch.Tensor = None) -> tuple:

        Z_t_1 = self.aux_head(Z_t_1)

        F_hat_t_0 = self.intra_gate(aux_feats=Z_t_1, latent_feats=H_t_2)

        F_hat_t_0 = self.shuffle_blocks(F_hat_t_0)

        if H_t_minus1_0 is not None:
            F_t_0 = torch.cat([H_t_minus1_0, F_hat_t_0], dim=1)
            H_t_1 = self.latent_conv_with_H_t_minus1_0(F_t_0)
            params = self.params_estimator_with_H_t_minus1_0(F_t_0)
        else:
            H_t_1 = self.latent_conv_wo_H_t_minus1_0(F_hat_t_0)
            params = self.params_estimator_wo_H_t_minus1_0(F_hat_t_0)

        p_Z_t_0 = self.entropy_model(X_t, params=params)  # log likelihoods of Z_t^s

        return H_t_1, p_Z_t_0
