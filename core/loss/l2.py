#%%

__all__ = ['l2Loss', 'rawL2LossFunc']


#%%

from .common import *

import torch


#%%

def l2Loss( toDelta: torch.Tensor, override='default'):
    hyper = getHyper( override)

    if ( hasattr( hyper, 'diffRate') and hyper.diffRate is not None) or \
            ( hasattr( hyper, 'diff2Rate') and hyper.diff2Rate is not None):
        raise NotImplementedError

    eCenter = CenterCache.ec_es_dims( toDelta.shape[ 1:])[ 0]
    value = toDelta.clone()
    value[ eCenter].sub_( 1)
    ret = l2l1Loss( value, hyper)
    ret = applyMask( ret, hyper)

    return ret


#%%

def rawL2LossFunc( toDelta: torch.Tensor, mask='none'):
    hyper = getHyper( mask)

    eCenter = CenterCache.ec_es_dims( toDelta.shape[ 1:])[ 0]
    value = toDelta.clone()
    value[ eCenter].sub_( 1)
    return applyMask( value.pow( 2), hyper)
