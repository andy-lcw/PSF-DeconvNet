#%%

__all__ = ['getHyper', 'l2l1Loss', 'applyMask', 'diffLosses', 'invLoss']


#%%

from .common import smooth, diff, diff2
from .mask import MaskCache
from ...shape import CenterCache

import torch


#%%

def getHyper( hyper):
    assert hyper is not None
    if hyper == 'default':
        raise NotImplementedError
    elif hyper == 'none':
        hyper = None
    else:
        pass
    return hyper


#%%

def l2l1Loss( value: torch.Tensor, hyper):
    if hyper is not None:
        ret = smooth( value, None, beta=hyper.cBeta, delta=hyper.cDelta)
    else:
        ret = value.pow( 2)
    return ret


def applyMask( value: torch.Tensor, hyper):
    shape = value.shape[ 1:]
    dims = CenterCache.ec_es_dims( shape)[ 2]

    if hyper is None or hyper == 'mean' or hyper.maskType == 'mean':
        return value.mean( dim=dims)

    mask = MaskCache.getMask(
        hyper.maskType, hyper.maskHalf, shape, hyper.maskBias, True,
        value.device, False)
    return value.mul( mask).sum( dim=dims)


#%%

def diffLosses( zeroCentered: torch.Tensor, hyper):
    diffLoss, diff2Loss = 0, 0

    if hyper is not None and \
            ( hyper.diffRate is not None or hyper.diff2Rate is not None):
        if hyper.diff2Rate is None:
            diffLoss = hyper.diffRate * diff( zeroCentered)
        else:
            diffLosses = diff2( zeroCentered)
            diff2Loss = hyper.diff2Rate * diffLosses[ 1]
            if hyper.diffRate is not None:
                diffLoss = hyper.diffRate * diffLosses[ 0]
    return diffLoss, diff2Loss


#%%

def invLoss( inv: torch.Tensor, hyper):
    if hyper is None:
        return 0

    dims = list( range( 1 - inv.ndim, 0))
    ret = 0
    if hyper.invL1Rate is not None:
        ret = ret + hyper.invL1Rate * inv.abs().mean( dim=dims)
    if hyper.invL2Rate is not None:
        ret = ret + hyper.invL2Rate * inv.pow( 2).mean( dim=dims)
    return ret
