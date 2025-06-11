#%%

__all__ = ['MultiHeadAttention']


#%%

import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


#%%

class MultiHeadAttention( nn.Module):

    def __init__(self, H: int, E: int, N: int, T: int, normDim=None):
        super().__init__()

        self.H = H
        self.E, self.N, self.T = E, N, T
        self.normDim = self.E if normDim is None else normDim

        self.k = nn.Parameter( torch.empty( self.H, self.E, self.N))
        self.v = nn.Parameter( torch.empty( self.H, self.N, self.T))

        self.normFactor = math.sqrt( self.normDim)
        self.register_buffer( 'scaleFactor', torch.tensor( 0.))

        # noinspection PyUnreachableCode
        if False:
            self.scaleFactor = None

    def init(self, values: tp.Optional[ torch.Tensor] = None,
             refStd=1):
        nn.init.normal_( self.k.data, 0, 1 / math.sqrt( self.E))

        if values is None:
            nn.init.normal_( self.v.data, 0, 1 / math.sqrt( self.N))
        else:
            self.v.data.copy_( values / self.N)

        scaleFactor = self.N * refStd
        self.scaleFactor = torch.tensor( scaleFactor, dtype=torch.float)

    def forward(self, q: torch.Tensor, retAtt=False):
        """
        :param q: [B...], H, E
        :param retAtt:
        :return: [B...], H, T
        """
        q = q / self.normFactor
        # ([B...], H, 1, E) (H, E, N) -> ([B...], H, 1, N)
        att = torch.matmul( q.unsqueeze( -2), self.k)
        att = F.softmax( att, -1)
        # ([B...], H, 1, N) (H, N, T) -> ([B...], H, 1, T)
        ret = torch.matmul( att, self.v).mul( self.scaleFactor).squeeze( -2)
        if retAtt:
            return ret, att
        return ret
