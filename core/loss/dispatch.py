#%%

__all__ = ['lossFunc']


#%%

from .common import getHyper, invLoss
from .l2 import l2Loss
from .posRes import posRescaledLoss
from .res import rescaledLoss

import typing as tp

import torch


#%%

def lossFunc( toDelta: torch.Tensor,
              inv: tp.Optional[ torch.Tensor],
              override='default'):

    assert override != 'none'
    hyper = getHyper( override)

    if hyper.loss == 'rescaled':
        ret = rescaledLoss( toDelta, override)

    elif hyper.loss == 'posRescaled':
        ret = posRescaledLoss( toDelta, override)

    elif hyper.loss == 'plain':
        ret = l2Loss( toDelta, override)

    else:
        raise KeyError( hyper.loss)

    if inv is not None:
        ret = ret + invLoss( inv, hyper)
    return ret
