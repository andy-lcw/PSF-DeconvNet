#%%

__all__ = ['NormalizedRealFFT']


#%%

import math
import typing as tp

import torch


#%%

class NormalizedRealFFT:

    def __init__(self, * shape: int, real: tp.Optional[ bool] = None):
        self.sp = shape
        self.real = real

        self.fspc = self.sp[ :-1] + ( self.sp[ -1] // 2 + 1,)
        self.fspr = self.fspc + (2,)

        # dimensions
        self.dim = torch.tensor( self.sp).prod().item()
        self.fdimc = torch.tensor( self.fspc).prod().item()
        self.fdimr = self.fdimc * 2

        fwdC = self.dim
        fwdR = fwdC / 2
        self.fwdNormC = math.sqrt( fwdC)
        self.fwdNormR = math.sqrt( fwdR)

        bwdC = self.dim / self.sp[ -1]
        if self.sp[ -1] % 2:
            bwdC *= self.sp[ -1] - 0.5
        else:
            bwdC *= self.sp[ -1] - 1
        bwdR = bwdC * 2
        self.bwdNormC = math.sqrt( bwdC)
        self.bwdNormR = math.sqrt( bwdR)

        if self.real is not None:
            if self.real:
                self.fsp = self.fspr
                self.fdim = self.fdimr
                self.fwdNorm = self.fwdNormR
                self.bwdNorm = self.bwdNormR
            else:
                self.fsp = self.fspc
                self.fdim = self.fdimc
                self.fwdNorm = self.fwdNormC
                self.bwdNorm = self.bwdNormC

        self._fftDim = tuple( range( - len( self.sp), 0))

    def fwd(self, signal: torch.Tensor):
        ret = torch.fft.rfftn( signal, dim=self._fftDim, norm='backward').div( self.fwdNorm)
        if self.real:
            return torch.view_as_real( ret)
        return ret

    def bwd(self, signal: torch.Tensor):
        if self.real:
            signal = torch.view_as_complex( signal)
        ret = torch.fft.irfftn( signal, self.sp, norm='forward').div( self.bwdNorm)
        return ret

    def dfwd(self, signal: torch.Tensor, real: bool):
        ret = torch.fft.rfftn( signal, dim=self._fftDim, norm='backward')
        if real:
            return torch.view_as_real( ret.div( self.fwdNormR))
        return ret.div( self.fwdNormC)

    def dbwd(self, signal: torch.Tensor, real: bool):
        if real:
            signal = torch.view_as_complex( signal)
        ret = torch.fft.irfftn( signal, self.sp, norm='forward')
        if real:
            return ret.div( self.bwdNormR)
        return ret.div( self.bwdNormC)
