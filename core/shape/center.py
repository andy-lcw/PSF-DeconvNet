#%%

__all__ = ['CenterCache']


#%%

import torch


#%%

class CenterCache:

    _cache = {}

    @classmethod
    def ec_es_dims(cls, shape):
        if not isinstance( shape, tuple):
            shape = tuple( shape)
        cache = cls._cache.get( shape, None)

        if cache is not None:
            return cache['ec'], cache['es'], cache['dims']

        center = tuple( torch.tensor( shape)
                        .div( 2, rounding_mode='trunc').tolist())
        sliced = tuple( slice( c, c+1) for c in center)

        eCenter = (..., ) + center
        eSliced = (..., ) + sliced
        dims = tuple( range( - len( center), 0))
        cls._cache[ shape] = dict( ec=eCenter, es=eSliced, dims=dims)

        return eCenter, eSliced, dims

    @classmethod
    def getCenter(cls, result: torch.Tensor):
        ec = cls.ec_es_dims( result.shape[ 1:])[ 0]
        return result[ ec]
