#%%

__all__ = ['rescaledLoss', 'rawRescaledLossFunc']


#%%

from .common import *

import torch


#%%

def rescaledLoss( toDelta: torch.Tensor, override='default'):
    hyper = getHyper( override)
    assert hyper is not None and hyper.cClampMin > 0

    if ( hasattr( hyper, 'diffRate') and hyper.diffRate is not None) or \
            ( hasattr( hyper, 'diff2Rate') and hyper.diff2Rate is not None):
        raise NotImplementedError

    eCenter, eSliced = CenterCache.ec_es_dims( toDelta.shape[ 1:])[ :2]

    sliced = toDelta[ eSliced].abs()
    loss = 0

    if hyper.cCenterRate is not None:
        closs = smooth_th( sliced.flatten(), None, hyper.cCenterThres,
                           beta=hyper.cCenterBeta, delta=hyper.cCenterDelta, noNeg=True)
        loss = hyper.cCenterRate * closs

    scale = clampMin( sliced, hyper.cClampMin)
    if hyper.cClampMax is not None:
        scale = scale.clamp_max_( hyper.cClampMax)

    value = toDelta / scale
    value[ eCenter].sub_( 1)
    ret = l2l1Loss( value, hyper)
    ret = loss + applyMask( ret, hyper)

    return ret


#%%

def rawRescaledLossFunc( toDelta: torch.Tensor, mask='none'):
    hyper = getHyper( mask)

    eCenter, eSliced = CenterCache.ec_es_dims( toDelta.shape[ 1:])[ :2]
    scale = toDelta / toDelta[ eSliced].abs()
    scale[ eCenter].zero_()
    return applyMask( scale.pow( 2), hyper)
