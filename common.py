#%%

__all__ = ['AllDsName', 'dsShortName', 'findAndRedirect',
           'buildHyper', 'HyperInvMask', 'HyperInvLossMask',
           'deepCopy', 'gimshow',
           'Timer', 'Dots']


#%%

import core.loss.common
from core.misc import DictObjType

import datetime as dt
import enum
import math
import os
import pathlib as pl
import sys
import time
import typing as tp
import warnings as wn
from contextlib import AbstractContextManager
from collections import OrderedDict as OD

import torch


#%%

AllDsName = [ 'Marmousi2', 'Marmousi4', 'MarmousiK',
              'Qikou', 'Layer6', 'Sigsbee2',
              'Qikou3d', 'Over3d']


def dsShortName( dsName: str):
    assert dsName in AllDsName
    if dsName.startswith( 'Marmousi') or dsName in ['Layer6', 'Sigsbee2']:
        return dsName[ 0].lower() + dsName[ -1].lower()
    if dsName == 'Qikou':
        return 'q'
    if dsName in ['Qikou3d', 'Over3d']:
        return dsName[ 0].lower() + '3d'
    raise RuntimeError


#%%

def findAndRedirect( savePath: str, saveName: str,
                     runInBackground: bool):

    savePath = pl.Path( savePath)
    savePath.mkdir( parents=True, exist_ok=True)

    index = -1
    saveOut = None
    while True:
        index += 1
        saveBin = f'.pth.{index}' if index else '.pth'
        saveBin = savePath / ( saveName + saveBin)
        if saveBin.exists():
            continue
        if runInBackground:
            saveOut = f'.out.{index}' if index else '.out'
            saveOut = savePath / ( saveName + saveOut)
            if saveOut.exists():
                continue
        break

    indexStr = str( index) if index else ''

    saveBin.touch( exist_ok=False)
    if runInBackground:
        print( 'output is redirected to:', str( saveOut))
        if os.fork() > 0:
            exit()
        os.setsid()
        if os.fork() > 0:
            exit()
        redirect = open( saveOut, mode='w', buffering=1)
        os.dup2( redirect.fileno(), sys.stdout.fileno())
        os.dup2( redirect.fileno(), sys.stderr.fileno())

        return indexStr, str( saveBin), str( saveOut)

    return indexStr, str( saveBin)


#%%

def buildHyper( lossType: int,
                psfShape: tp.Sequence[ int],
                invShape: tp.Sequence[ int]):

    class Hyper( metaclass=DictObjType):

        if lossType == 0:
            loss = 'plain'
        elif lossType in [ 1, 2]:
            loss = 'rescaled'
        elif lossType in [ 3, 4, 5, 6, 7, 8, 9]:
            loss = 'posRescaled'
        else:
            raise NotImplementedError

        if lossType >= 1:
            cClampMin = 0.1
            cClampMax = None
            cCenterRate = None

        if lossType >= 2:
            cCenterRate = 1e-4
            cCenterThres = 2
            cCenterBeta = None
            cCenterDelta = 3

        if lossType >= 3:
            cCenterNegRate = 10
            cCenterNegThres = 0.1
            cCenterNegBeta = None
            cCenterNegDelta = None

        cBeta = cDelta = None
        maskType = 'mean'
        plainRate = 1
        diffRate = diff2Rate = None

        invMask = invLossMask = None
        invHalf = invLossHalf = None
        invL1Rate = invL2Rate = None

        if lossType in [ 4, 5]:
            assert tuple( psfShape) == tuple( invShape)

            invMask = 'abs'
            if lossType == 5:
                invLossMask = 'abs'
                invL1Rate = 0.01

        elif lossType in [ 6, 7]:
            invMask = 'recip'
            invHalf = 2.5, 2.5
            if lossType == 7:
                invLossMask = 'recip'
                invLossHalf = 2.5, 2.5
                invL1Rate = 0.01

        elif lossType in [ 8, 9]:
            invMask = 'gaussian'
            invHalf = 2.5, 2.5
            if lossType == 7:
                invLossMask = 'gaussian'
                invLossHalf = 2.5, 2.5
                invL1Rate = 0.01

        if tuple( invShape) != ( 50, 50) and lossType in [ 6, 7, 8, 9]:
            if len( invShape) == 2:
                wn.warn( 'masks in masked losses are not adjusted for '
                         'inverse kernels of shape other than (50, 50).')
            else:
                raise NotImplementedError

    return Hyper


def _getMask( mask, half, inverse, invShape, device):
    if mask is None or mask == 'abs':
        return mask
    return core.loss.common.MaskCache.getMask(
        mask, half, invShape, 0, False, device, inverse)


def HyperInvMask( Hyper, invShape, device):
    return _getMask( Hyper.invMask, Hyper.invHalf, False, invShape, device)


def HyperInvLossMask( Hyper, invShape, device):
    return _getMask( Hyper.invLossMask, Hyper.invLossHalf, True,
                     invShape, device)


#%%

def deepCopy( sth):
    if sth is None:
        return sth
    if isinstance( sth, ( bool, int, float, str, enum.Enum)):
        return sth
    if isinstance( sth, torch.Tensor):
        return sth.detach().to( 'cpu', copy=True)

    if isinstance( sth, ( tuple, list, set)):
        ret = sth.__class__( deepCopy( th) for th in sth)
    elif isinstance( sth, ( dict, OD)):
        ret = sth.__class__( { k: deepCopy( v) for k, v in sth.items()})
    else:
        raise NotImplementedError( type( sth))

    return ret


def gimshow( img: torch.Tensor, figSize: tp.Sequence[ int] = None,
             vrange=( None, None), colorBar=False,
             fileName=None, pureSave=False):

    if fileName is None:
        assert not pureSave

    import matplotlib.pyplot as plt

    plt.figure( figsize=figSize)
    plt.axis( False)
    plt.imshow( img.detach().cpu(), cmap='gray',
                vmin=vrange[ 0], vmax=vrange[ 1])
    if colorBar:
        plt.colorbar()
    plt.tight_layout()

    if fileName is not None:
        plt.savefig( fileName)
    if pureSave:
        plt.close()
    else:
        plt.show()
        if not plt.isinteractive():
            plt.close()


#%%

class Timer( AbstractContextManager):

    def __init__(self, device):
        self.device = torch.device( device)

        self.cpu = self.device.type == 'cpu'
        self.total = 0
        self.entered = False

        if self.cpu:
            self.last = None
        else:
            self.stream = torch.cuda.current_stream( self.device)
            self.begin = torch.cuda.Event( enable_timing=True)
            self.end = torch.cuda.Event( enable_timing=True)

    def __enter__(self):
        assert not self.entered
        self.entered = True

        if self.cpu:
            self.last = time.perf_counter()
        else:
            self.begin.record( self.stream)
        return self

    enter = __enter__

    def __exit__(self, eType, eValue, eTrace):
        assert self.entered

        if self.cpu:
            self.total += time.perf_counter() - self.last
        else:
            self.end.record( self.stream)
            self.end.synchronize()
            self.total += self.begin.elapsed_time( self.end) / 1000

        self.entered = False

    exit = __exit__

    def dt(self): return dt.timedelta( seconds=round( self.total))
    def __repr__(self): return repr( self.dt())
    def __str__(self): return str( self.dt())

    def fullStr(self, now: float, total: float):
        cost = self.total
        if not ( 0 < now <= total):
            full = remains = '---'
        else:
            full = self.total * total / now
            remains = str( dt.timedelta( seconds=round( full - cost)))
            full = str( dt.timedelta( seconds=round( full)))
        cost = str( dt.timedelta( seconds=round( cost)))
        return ' '.join( [ cost, remains, full])


#%%

class Dots:

    def __init__(self, total: int, dots=10):
        self.total = total
        self.dots = dots
        self.last = 0

    def step(self, now: int, flush=True):
        dots = math.floor( now * self.dots / self.total)
        delta = dots - self.last
        if delta > 0:
            print( '.' * delta, end='')
            self.last = dots
            if flush:
                sys.stdout.flush()
