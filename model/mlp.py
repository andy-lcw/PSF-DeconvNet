#%%

__all__ = ['MLP']


#%%

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#%%

class MLP( nn.Module):

    def __init__(self, * dims: int,
                 hiddenBias: bool, outputBias: bool):
        super().__init__()

        assert len( dims) >= 2
        self.dims = dims
        self.hiddenBias = hiddenBias
        self.outputBias = outputBias

        hidden = []
        for i in range( 1, len( self.dims) - 1):
            hidden.append( nn.utils.skip_init(
                nn.Linear, self.dims[ i - 1], self.dims[ i], bias=self.hiddenBias))
        self.hidden = nn.ModuleList( hidden)

        self.output = nn.utils.skip_init(
            nn.Linear, self.dims[ -2], self.dims[ -1], bias=self.outputBias)

        self.alterGain = math.sqrt( 2)

    def init(self):
        """
        当输入方差是 1 的时候保证输出方差也是 1。
        """
        def initLinear( layer: nn.Linear, nonlinearity: str):
            nn.init.kaiming_normal_( layer.weight.data, nonlinearity=nonlinearity)
            if layer.bias is not None:
                nn.init.zeros_( layer.bias.data)

        for ly in self.hidden:
            initLinear( ly, nonlinearity='relu')
        initLinear( self.output, nonlinearity='linear')
        return self

    def forward(self, x: torch.Tensor):
        for ly in self.hidden:
            x = F.relu_( ly( x))
        return self.output( x)
