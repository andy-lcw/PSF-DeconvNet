#%%

__all__ = ['Misc', 'DictObj', 'DictObjType',
           'Steps', 'Epochs']


#%%

from .shape import CenterCache

import datetime as dt
import math
import typing as tp

import torch


#%%

class Misc:

    @staticmethod
    def flat2( x: torch.Tensor):
        return x.permute( 0, 2, 1, 3).flatten( 2).flatten( 0, 1)

    @staticmethod
    def flat3( x: torch.Tensor):
        x = x.permute( 0, 3, 1, 4, 2, 5)
        return x.flatten( 4).flatten( 2, 3).flatten( 0, 1)

    @staticmethod
    def visualNorm( x: torch.Tensor, dims: int):
        dims = tuple( range( - dims, 0))
        minValue = x.amin( dim=dims, keepdim=True)
        maxValue = x.amax( dim=dims, keepdim=True)
        return x.sub( minValue).div( maxValue.sub( minValue))

    @staticmethod
    def centerNorm( x: torch.Tensor):
        cent = [ ...]
        for s in x.shape:
            c = s//2
            cent.append( slice( c, c + 1))
        return x / x[ tuple( cent)]

    @staticmethod
    def centerNormFromCache( x: torch.Tensor, dim=2):
        es = x[ CenterCache.ec_es_dims( x.shape[ -dim:])[ 1]]
        return x / es

    @staticmethod
    def figSize( sp: tp.Sequence[ int], r: float):
        return tuple( s / r for s in reversed( sp))

    @staticmethod
    def toDeltaTime( delta: float):
        return dt.timedelta( seconds=round( delta))


#%%

class DictObj:
    __slots__ = ['__custom_name__', '__dict__']

    def __init__(self, _customName='DictObj', **kwargs):
        self.__custom_name__ = _customName
        self.__dict__.update( **kwargs)

    def __getitem__(self, key):
        return self.__dict__[ key]

    def __setitem__(self, key, value):
        self.__dict__[ key] = value

    def __delitem__(self, key):
        del self.__dict__[ key]

    def __repr__(self):
        ret = self.__custom_name__ + '('
        if len( self.__dict__) == 0:
            return ret + ')'
        return ret + ' ' + ', '.join( f'{k}={v}' for k, v in self.__dict__.items()) + ')'

    def keys(self):
        return self.__dict__.keys()

    def toDict(self):
        return self.__dict__

    def update(self, **kwargs):
        self.__dict__.update( **kwargs)


class DictObjType( type):
    def __new__(mcs, name, bases, namespace: tp.Dict[ str, tp.Any]):
        assert len( bases) == 0
        namespace = { k: v for k, v in namespace.items()
                      if not k.startswith( '__') and not k.endswith( '__') }
        return DictObj( name, **namespace)


#%%

class Steps:

    def __init__(self, * names: str):
        self.names = names
        self._num = { n: 0 for n in names}

        self._sum = { n: 0 for n in names}
        self._stdSum = { n: 0 for n in names}
        self._min = { n: math.inf for n in names}
        self._max = { n: - math.inf for n in names}

    def update(self, name: str, data: torch.Tensor):
        data = data.detach()

        length, = data.shape
        self._num[ name] += length

        self._sum[ name] += data.sum().item()
        self._stdSum[ name] += data.std( False).item() * length

        self._min[ name] = min( self._min[ name], data.min().item())
        self._max[ name] = max( self._max[ name], data.max().item())

    def mean(self, name: str): return self._sum[ name] / self._num[ name]
    def std(self, name: str): return self._stdSum[ name] / self._num[ name]
    def min(self, name: str): return self._min[ name]
    def max(self, name: str): return self._max[ name]

    def repr(self, name: str):
        return f'mean: {self.mean( name)}, std: {self.std( name)}, ' \
               f'min: {self.min( name)}, max: {self.max( name)}'


#%%

class Epochs:

    def __init__(self, * names: str):
        self.names = names
        self.means = { n: [] for n in names}
        self.stds = { n: [] for n in names}
        self.mins = { n: [] for n in names}
        self.maxs = { n: [] for n in names}

    def update(self, steps: Steps):
        assert set( self.names) == set( steps.names)
        for name in self.names:
            self.means[ name].append( steps.mean( name))
            self.stds[ name].append( steps.std( name))
            self.mins[ name].append( steps.min( name))
            self.maxs[ name].append( steps.max( name))

    def stateDict(self):
        return dict(
            names=self.names,
            means=self.means,
            stds=self.stds,
            mins=self.mins,
            maxs=self.maxs
        )
