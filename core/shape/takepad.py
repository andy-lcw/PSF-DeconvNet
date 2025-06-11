#%%

__all__ = ['takePart', 'takeCenter', 'centerPadding']


#%%

import typing as tp

import torch
import torch.nn.functional as F


#%%

def takePart( x: torch.Tensor,
              shape: tp.Sequence[ int],
              pos: tp.Sequence[ int],
              pad: bool,
              allow1=True):

    from itertools import chain

    dims = len( shape)
    assert x.ndim >= dims == len( pos)

    for i in range( -dims, 0):
        if pad and allow1:
            cond = -1 <= pos[ i] <= x.shape[ i]
        else:
            cond = 0 <= pos[ i] < x.shape[ i]
        if not cond:
            raise IndexError

    slices = [...]
    padLeft, padRight = [], []

    for i in range( -dims, 0):
        c = shape[ i] // 2
        left = pos[ i] - c
        right = left + shape[ i]

        sleft = max( left, 0)
        sright = min( right, x.shape[ i])
        slices.append( slice( sleft, sright))

        if pad:
            padLeft.append( max( - left, 0))
            padRight.append( max( right - x.shape[ i], 0))

    part = x[ tuple( slices)]
    if pad:
        padding = list( reversed( list( chain.from_iterable( zip( padRight, padLeft)))))
        if any( padding):
            part = F.pad( part, padding)
    return part


#%%

def takeCenter( x: torch.Tensor,
                shape: tp.Union[ int, tp.Sequence[ int]]):
    if isinstance( shape, int):
        shape = ( shape,) * x.ndim
    pos = [ s // 2 for s in x.shape[ - len( shape):]]
    return takePart( x, shape, pos, False)


#%%

def centerPadding( shape: tp.Sequence[ int],
                   target: tp.Sequence[ int]):

    assert len( shape) == len( target)

    padding = []
    for s, t in zip( shape, target):
        assert t >= s
        pcent = s // 2
        icent = t // 2
        padLeft = icent - pcent
        padRight = t - s - padLeft
        padding.extend( [ padRight, padLeft])
    return list( reversed( padding))
