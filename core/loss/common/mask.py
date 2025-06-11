#%%

__all__ = ['MaskCache']


#%%

from functools import reduce
import math
import typing as tp

import torch


#%%

class MaskCache:

    _cache = {}

    _HalfType = tp.Sequence[ float]
    _ShapeType = tp.Sequence[ int]

    @classmethod
    def getMask(cls, maskType: str,
                half: _HalfType, shape: _ShapeType, bias: float, norm: bool,
                device, inverse: bool):

        key = maskType, device, inverse, tuple( half), tuple( shape), bias, norm
        ret = cls._cache.get( key, None)
        if ret is None:
            if inverse:
                ret = 1 - cls.getMask( maskType, half, shape, bias, norm,
                                       device, False)
            else:
                if maskType == 'gaussian':
                    gen = cls._gaussianMask
                elif maskType == 'recip':
                    gen = cls._recipMask
                else:
                    raise KeyError
                ret = gen( * key[ 3:]).to( device)
            cls._cache[ key] = ret
        return ret

    @staticmethod
    def _gaussianMask( half: _HalfType, shape: _ShapeType,
                       bias: float, norm: bool):
        assert 0 <= bias <= 1 and len( half) == len( shape)
        log2 = math.log( 2)
        radius = [ h / 2 for h in half]
        coord = [ torch.arange( s) - s // 2 for s in shape]
        coord = [ log2 * cd ** 2 / r ** 2 for cd, r in zip( coord, radius)]
        coord = reduce( lambda x, y: x + y,
                        [ cd.view( -1, * ( ( 1,) * icd),)
                          for icd, cd in enumerate( reversed( coord))])
        ret = torch.exp( - coord) * ( 1 - bias) + bias
        if norm:
            ret /= ret.sum() - 1
        return ret

    @staticmethod
    def _recipMask( half: _HalfType, shape: _ShapeType,
                    bias: float, norm: bool):
        assert 0 <= bias <= 1 and len( half) == len( shape)
        radius = [ h / 2 for h in half]
        coord = [ torch.arange( s) - s // 2 for s in shape]
        coord = [ cd ** 2 / r ** 2 for cd, r in zip( coord, radius)]
        coord = reduce( lambda x, y: x + y,
                        [ cd.view( -1, * ( ( 1,) * icd),)
                          for icd, cd in enumerate( reversed( coord))])
        ret = ( 1 + coord.sqrt()).reciprocal() * ( 1 - bias) + bias
        if norm:
            ret /= ret.sum() - 1
        return ret
