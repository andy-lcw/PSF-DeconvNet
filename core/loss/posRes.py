#%%

__all__ = ['posRescaledLoss']


#%%

from .common import *

import torch


#%%

def posRescaledLoss( toDelta: torch.Tensor, override='default'):
    hyper = getHyper( override)
    assert hyper is not None and hyper.cClampMin > 0

    eCenter, eSliced = CenterCache.ec_es_dims( toDelta.shape[ 1:])[ :2]

    center = toDelta[ eCenter]
    centerLoss = 0

    if hyper.cCenterRate is not None:
        closs = smooth_side( center, None, hyper.cCenterThres, True,
                             beta=hyper.cCenterBeta, delta=hyper.cCenterDelta)
        centerLoss = centerLoss + hyper.cCenterRate * closs

    if hyper.cCenterNegRate is not None:
        closs = smooth_side( center, None, hyper.cCenterNegThres, False,
                             beta=hyper.cCenterNegBeta, delta=hyper.cCenterNegDelta)
        centerLoss = centerLoss + hyper.cCenterNegRate * closs

    scale = clampMin( toDelta[ eSliced], hyper.cClampMin)
    if hyper.cClampMax is not None:
        scale = scale.clamp_max_( hyper.cClampMax)

    value = toDelta / scale
    value[ eCenter].zero_()

    plainLoss = 0
    if hyper.plainRate is not None:
        plainLoss = hyper.plainRate * applyMask( l2l1Loss( value, hyper), hyper)

    diffLoss, diff2Loss = diffLosses( value, hyper)

    return centerLoss + plainLoss + diffLoss + diff2Loss
