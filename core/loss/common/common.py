#%%

__all__ = ['clampMin', 'smooth', 'smooth_th', 'smooth_side',
           'diff', 'diff2']


#%%

import typing as tp

import torch
import torch.nn.functional as F


#%%

class ClampMin( torch.autograd.Function):
    @staticmethod
    def forward( ctx, v: torch.Tensor, m: float):
        return torch.clamp_min( v, m)

    @staticmethod
    def backward( ctx, grad):
        return grad, None


clampMin = ClampMin.apply


#%%

def smooth( value: torch.Tensor, target: tp.Union[ None, float, torch.Tensor], *,
            beta: tp.Optional[ float] = None, delta: tp.Optional[ float] = None):

    def getTarget():
        if zeroTarget:
            return torch.zeros( (), dtype=value.dtype, device=value.device).expand_as( value)
        if isinstance( target, ( int, float)):
            return torch.tensor( target, dtype=value.dtype, device=value.device).expand_as( value)
        return target

    zeroTarget = target is None or target == 0

    if beta is None and delta is None:
        return value.pow( 2) if zeroTarget else F.mse_loss( value, getTarget(), reduction='none')
    elif beta is not None:
        assert delta is None
        return F.smooth_l1_loss( value, getTarget(), beta=beta, reduction='none')
    else:
        assert delta is not None
        return 2 * F.huber_loss( value, getTarget(), delta=delta, reduction='none')


#%%

def smooth_th( value: torch.Tensor, target: tp.Union[ None, float, torch.Tensor],
               thres: tp.Optional[ float], *,
               beta: tp.Optional[ float] = None, delta: tp.Optional[ float] = None,
               noNeg: bool = False):

    if thres is None or thres == 0:
        return smooth( value, target, beta=beta, delta=delta)

    if target is not None and target != 0:
        value = value.sub( target).abs()
    elif not noNeg:
        value = value.abs()

    assert thres > 0
    value = value.sub( thres).relu_()

    if beta is None and delta is None:
        return value.pow( 2)

    zero = torch.zeros( (), dtype=value.dtype, device=value.device).expand_as( value)
    if beta is not None:
        assert delta is None
        beta = beta - thres
        assert beta > 0
        return F.smooth_l1_loss( value, zero, beta=beta, reduction='none')
    else:
        assert delta is not None
        delta = delta - thres
        assert delta > 0
        return 2 * F.huber_loss( value, zero, delta=delta, reduction='none')


#%%

def smooth_side( value: torch.Tensor, target: tp.Union[ None, float, torch.Tensor],
                 thres: tp.Optional[ float], upper: bool, *,
                 beta: float = None, delta: float = None):

    if thres is None:
        thres = 0
    dec = thres
    if target is not None and target != 0:
        dec = thres + target
    value = value.sub( dec)

    if upper:
        value.relu_()
    else:
        value.neg_().relu_()

    if beta is None and delta is None:
        return value.pow( 2)

    zero = torch.zeros( (), dtype=value.dtype, device=value.device).expand_as( value)
    if beta is not None:
        assert delta is None
        beta = beta - thres if upper else thres - beta
        assert beta > 0
        return F.smooth_l1_loss( value, zero, beta=beta, reduction='none')
    else:
        delta = delta - thres if upper else thres - delta
        assert delta > 0
        return 2 * F.huber_loss( value, zero, delta=delta, reduction='none')


#%%

def diff( value: torch.Tensor):
    ret = 0
    dims = list( range( 1, value.ndim))
    for d in dims:
        ret = ret + value.diff( dim=d).pow( 2).sum( dim=dims)
    return ret / value[ 0].numel()


def diff2( value: torch.Tensor):
    ret1, ret2 = 0, 0
    dims = list( range( 1, value.ndim))
    for d in dims:
        v = value.diff( dim=d)
        ret1 = ret1 + v.pow( 2).sum( dim=dims)
        ret2 = ret2 + v.diff( dim=d).pow( 2).sum( dim=dims)
        for d2 in range( d + 1, value.ndim):
            ret2 = ret2 + v.diff( dim=d2).pow( 2).sum( dim=dims) * 2
    norm = value[ 0].numel()
    return ret1 / norm, ret2 / norm
