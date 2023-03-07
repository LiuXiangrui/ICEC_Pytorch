import torch
import torch.nn as nn

from ICEC import ICEC
class Network(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()

        self.icec_blocks = nn.ModuleList(
            [ICEC(D=0, N=3, R=4, height=height, width=width,
                  channels_F=64, channels_M=120, channels_Z=10,
                  L=25, max_=1., min_=-1., K=10),
             ]
        )