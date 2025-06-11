#%%

__all__ = ['Preprocess', 'Postprocess', 'Dict', 'DMLP']


#%%

from .attention import MultiHeadAttention as MHA
from .mlp import MLP

from core.fourier import NormalizedRealFFT as nrfft

import typing as tp

import torch
import torch.nn as nn


#%%

class Preprocess( nn.Module):

    def __init__(self, inputShape: tp.Sequence[ int],
                 inputInFreqDomain: bool):
        super().__init__()
        self.inputShape = tuple( inputShape)
        self.inputInFreqDomain = inputInFreqDomain

        self.fft = nrfft( * self.inputShape, real=True)
        self.outDim = self.fft.fdim if self.inputInFreqDomain else self.fft.dim

    def forward(self, psf: torch.Tensor):
        if self.inputInFreqDomain:
            psf = self.fft.fwd( psf)
        return psf.flatten( 1)


class Postprocess( nn.Module):

    def __init__(self, outputShape: tp.Sequence[ int],
                 outputInFreqDomain: bool):
        super().__init__()
        self.outputShape = tuple( outputShape)
        self.outputInFreqDomain = outputInFreqDomain

        self.fft = nrfft( * self.outputShape, real=True)
        self.inDim = self.fft.fdim if self.outputInFreqDomain else self.fft.dim
        self.sp = self.fft.fsp if self.outputInFreqDomain else self.fft.sp

    def forward(self, data: torch.Tensor):
        data = data.view( data.shape[ 0], * self.sp)
        return self.fft.bwd( data) if self.outputInFreqDomain else data


#%%

class Dict( nn.Module):

    def __init__(self, inDim: int, dictSize: int, outDim: int, *,
                 value: tp.Optional[ torch.Tensor]):
        super().__init__()

        self.inDim = inDim
        self.dictSize = dictSize
        self.outDim = outDim
        self.value = value,

        self.dict = MHA( 1, self.inDim, self.dictSize, self.outDim)

    def init(self):
        self.dict.init( values=self.value[ 0])

    def forward(self, key: torch.Tensor):
        ret, att = self.dict( key.view( key.shape[ 0], 1, -1), retAtt=True)
        return ret.squeeze( 1), att.view( att.shape[ 0], att.shape[ -1])

    def valueParam(self):
        yield self.dict.v


#%%

class DMLP( nn.Module):

    def __init__(self, inDim: int, dictSize: int, * mlpDims: int):
        super().__init__()

        self.inDim = inDim
        self.dictSize = dictSize
        self.mlpDims = mlpDims

        self.dict = Dict( inDim, dictSize, mlpDims[ 0], value=None)
        self.mlp = MLP( * self.mlpDims, hiddenBias=True, outputBias=False)

    def init(self):
        self.dict.init()
        self.mlp.init()

    def forward(self, key: torch.Tensor):
        return self.mlp( self.dict( key)[ 0])
