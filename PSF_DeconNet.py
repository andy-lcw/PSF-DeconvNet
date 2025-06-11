#%%

from common import *
import core.infer, core.loss, core.shape
from core.misc import DictObj, DictObjType, Steps, Epochs
import ds.augment, ds.ds, ds.img
from ds.ds.rtds import RuntimeInterpolateDataset as RTDS
from ds.img import ImgBase
import model

import enum
import pathlib as pl
import sys
import typing as tp
import warnings as wn
from contextlib import ExitStack

import torch
import torch.utils.data as tud


#%% overall configurations

debug = False
runInBackground = not debug
visualizeFor2d = True


class Data( metaclass=DictObjType):
    # see `common.AllDsName` for the available list
    dsName = None           # can be `None`

    device = None           # can be `None`
    numWorkers = 0 if debug else 4

    batchSize = 1024
    evalBatchSize = batchSize * 2
    batches = 128
    totalEpochs = 250


class Save( metaclass=DictObjType):
    savePath = 'save-model/'
    saveIndex = saveBin = saveOut = None

    saveTimes = [ 1, 2, 4, 8, 16, 32, 50, 100, 150, 200]


class Train( metaclass=DictObjType):
    useAugment = False      # bool; only 2D PSFs can use data augmentation
    pOri = 0.25             # None; float in [0, 1)
    rotate = 10             # None; float
    scale = ( 0.8, 1.3)     # None; ( min, max)

    """ see `model.Model.__init__` """
    outputShape = None      # `None` to use the PSF shape
    inputInFreqDomain = True
    outputInFreqDomain = True
    modelArch = None        # can be `None`

    """
    see *gdPointwise.Train.lossType*
    """
    lossType = None         # can be `None`
    override = dict(
        c1=[ 4, dict( diffRate=2)],
        c2=[ 5, dict( invL1Rate=0.05)],
        c3=[ 5, dict( invL1Rate=0.005)],
        c4=[ 7, dict( invL1Rate=0.05)],
        c5=[ 7, dict( invL1Rate=0.005)],
        # you can write your own custom override here
    )

    """
    0: all parameters use 1e-3;
    1: values of Dict use 1e-4, others use 1e-3.
    """
    lrType = 1


#%%

# input
if Data.dsName is None:
    Data.dsName = input( 'dsName: ')
if Data.device is None:
    Data.device = input( 'device: ')
    if Data.device != 'cpu':
        Data.device = int( Data.device)
if Train.modelArch is None:
    Train.modelArch = int( input( 'modelArch: '))
if Train.lossType is None:
    Train.lossType = input( 'lossType: ')

# Save
Save.saveTimes = sorted( list( Save.saveTimes))
for st in range( 1, len( Save.saveTimes)):
    assert Save.saveTimes[ st-1] < Save.saveTimes[ st]
if len( Save.saveTimes) == 0 or Save.saveTimes[ -1] < Data.totalEpochs:
    Save.saveTimes.append( Data.totalEpochs)
else:
    assert Save.saveTimes[ -1] == Data.totalEpochs
assert Save.saveTimes[ 0] >= 0

saveName = '{ds}{aug}-arch_{arch}-ls_{loss}-lr_{lr}'.format(
    ds=dsShortName( Data.dsName), aug=( '-aug' if Train.useAugment else ''),
    arch=Train.modelArch, loss=Train.lossType, lr=Train.lrType
)
# this function only works on unix-like systems
saveBinOut = findAndRedirect( Save.savePath, saveName, runInBackground)
if runInBackground:
    Save.saveIndex, Save.saveBin, Save.saveOut = saveBinOut
else:
    Save.saveIndex, Save.saveBin = saveBinOut

# Data
if Data.dsName in ['Marmousi2', 'Marmousi4']:
    rds: RTDS = getattr( ds.ds, Data.dsName)( True, False, 'absmax')
    ids: ImgBase = getattr( ds.img, Data.dsName)( True)
else:
    rds: RTDS = getattr( ds.ds, Data.dsName)( False, 'absmax')
    ids: ImgBase = getattr( ds.img, Data.dsName)()

lastDims = tuple( range( - ids.dims, 0))

if torch.device( Data.device).type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device( Data.device)

if Data.numWorkers > 0:
    import torch.multiprocessing as tmp
    tmp.set_start_method( 'fork')

# Train & Hyper
Train.invShape = ids.psfShape
if Train.outputShape is not None:
    Train.invShape = tuple( Train.outputShape)

try:
    Train.realLossType = Train.lossType = int( Train.lossType)
    Train.realOverride = {}
except ValueError:
    Train.realLossType, Train.realOverride = Train.override[ Train.lossType]
Hyper = buildHyper( Train.realLossType, ids.psfShape, Train.invShape)
Hyper.update( ** Train.realOverride)

him = HyperInvMask( Hyper, Train.invShape, Data.device)
hilm = HyperInvLossMask( Hyper, Train.invShape, Data.device)

# save
save = DictObj(
    Data=Data.toDict(), Save=Save.toDict(),
    Train=Train.toDict(), Hyper=Hyper.toDict(),
)


#%%

print( 'building and loading...')

# network
net = model.Model( ids.psfShape, Train.inputInFreqDomain,
                   Train.invShape, Train.outputInFreqDomain, Train.modelArch)
net.init().to( Data.device)

if Train.lrType == 0:
    optim = torch.optim.Adam( net.parameters(), lr=1e-3)
elif Train.lrType == 1:
    optim = torch.optim.Adam( [
        dict( params=net.valueParams(), lr=1e-4),
        dict( params=net.normalParams(), lr=1e-3)
    ])
else:
    raise KeyError( Train.lrType)

save.models = []
save.optims = []

# runtime loader
rloader = rds.test4train( Data.batches * Data.batchSize)
if Train.useAugment:
    assert ids.dims == 2
    aug = ds.augment.Affine( rloader, Train.rotate, Train.scale)
    if Train.pOri is not None:
        assert 0 <= Train.pOri < 1
        if Train.pOri > 0:
            aug = ds.augment.Choice( rloader, aug, Train.pOri)
    rloader = aug
rloader = tud.DataLoader( rloader, Data.batchSize,
                          num_workers=Data.numWorkers, pin_memory=True)

# image loader
iloader = ds.img.ImgDataset( ids, *ids.validIndices(), imgShape=Train.invShape)
iloader = iloader.getLoader( Data.evalBatchSize,
                             numWorkers=Data.numWorkers, pinMemory=True)

# grids loader
psfLeadShape = ids.psf.shape[ : ids.dims]
psfChunks = ids.psf.flatten( 0, ids.dims - 1)
psfNumber = psfChunks.shape[ 0]
psfChunks = psfChunks.split( Data.evalBatchSize)

# save
save.psfs = ids.psf
save.invs = [ [] for _ in Save.saveTimes]
save.rsts = [ [] for _ in Save.saveTimes]
if net.retAtt:
    save.atts  = [ [] for _ in Save.saveTimes]
    save.bases = [ [] for _ in Save.saveTimes]
    save.brsts = [ [] for _ in Save.saveTimes]
else:
    save.update( atts=None, bases=None, brsts=None)

coordRange = torch.as_tensor( [ ids.cint.lt, ids.cint.br1], dtype=torch.long)
save.oriImg = ids.img[ tuple( slice( * ir) for ir in coordRange.T.tolist())]
save.vis = tuple( torch.zeros_like( save.oriImg) for _ in Save.saveTimes)
if net.retAtt:
    save.bvis = tuple( torch.zeros_like( save.oriImg) for _ in Save.saveTimes)
else:
    save.bvis = 'not applicable'


#%%

print( 'warn: this timer only covers the training time. '
       'other time (e.g., for visualization) is not included. '
       'it will be shorter than the wall clock time.')
timer = Timer( Data.device)

# loss, raw-l2, raw-rescaled, center value
statItems = 'ls l2 rs ct'.split()


class Infer( enum.IntFlag):
    inv = 0b0000    # get deconv kernel
    bas = 0b0001    # get deconv result using base kernel (from dict)
    rst = 0b0010    # get deconv result using real kernel
    upd = 0b0110    # update statistic data
    opt = 0b1110    # step optimizer


def inference( _data, _adataGen, _statStep, _infer: Infer):
    _adata = _iadata = None
    if 'abs' in [ him, hilm]:
        _adata = _adataGen()
        if 'abs' == hilm:
            _iadata = 1 - _adata

    def _applyMask( _src, _mask, _absMask):
        if _mask == 'abs':
            return _src * _absMask
        elif _mask is not None:
            return _src * _mask
        return _src

    with ExitStack() as _exitStack:
        if Infer.opt in _infer:
            assert Infer.bas not in _infer
            optim.zero_grad()
            _exitStack.enter_context( timer)

        _invs, _atts, _bases = net( _data, Infer.bas in _infer)
        _invs = _applyMask( _invs, him, _adata)

        _rsts = None
        if Infer.rst in _infer:
            _rsts = core.infer.batchedFullConv( _data, _invs)

            if Infer.upd in _infer:
                _lminvs = _applyMask( _invs, hilm, _iadata)
                _lss = core.loss.lossFunc( _rsts, _lminvs, override=Hyper)

                if Infer.opt in _infer:
                    _lss.mean().backward()
                    optim.step()
                    _exitStack.close()
                    _exitStack.enter_context( torch.inference_mode())

                _l2s = core.loss.rawL2LossFunc( _rsts)
                _rss = core.loss.rawRescaledLossFunc( _rsts)
                _cts = core.shape.CenterCache.getCenter( _rsts)
                for _it, _dt in zip( statItems, [ _lss, _l2s, _rss, _cts]):
                    _statStep.update( _it, _dt)

        _brsts = None
        if Infer.bas in _infer and net.retAtt:
            _bases = _applyMask( _bases, him, _adata)
            _brsts = core.infer.batchedFullConv( _data, _bases)

        return _invs, _rsts, _atts, _bases, _brsts


def makeAdata( _psfs, inplace: bool):
    _data = _psfs
    _adata = _data.abs()
    _dataScales = _adata.amax( dim=lastDims, keepdim=True)
    if inplace:
        _data.div_( _dataScales)
    else:
        _data = _data.div( _dataScales)
    return _data, _dataScales, lambda: _adata.div_( _dataScales)


def makeRealScale( _data, _dataScales, inplace: bool,
                   _invs: torch.Tensor, normInvsInplace: bool,
                   _rsts: tp.Optional, normRstsInplace: bool):
    if Hyper.loss != 'plain':
        if _rsts is None:
            _sinvs = _invs
            if Train.invShape != ids.psfShape:
                _sinvs = core.shape.takeCenter( _invs, ids.psfShape)
            _cent = _data.mul( _sinvs).sum( dim=lastDims, keepdim=True)
        else:
            _csd = core.shape.CenterCache.ec_es_dims( _rsts.shape[ 1:])
            _cent = _rsts[ _csd[ 1]]
            if normRstsInplace:
                _cent = _cent.clone()
                _rsts.div_( _cent)
        if inplace:
            _dataScales.mul_( _cent)
        else:
            _dataScales = _dataScales.mul( _cent)
        if normInvsInplace:
            _invs.div_( _dataScales)
    return _dataScales


def visualize( _posit, _poses, _psfs, inplace: bool, _imgs):
    _data, _dataScales, _getAdata = makeAdata( _psfs, inplace)
    _rets = inference( _data, _getAdata, None, Infer.inv | Infer.bas)
    _realDataScale = makeRealScale( _data, _dataScales, False,
                                    _rets[ 0], False, None, False)
    save.vis[ _posit][ _poses] = _imgs.mul( _rets[ 0]).sum( dim=lastDims) \
        .div_( _realDataScale.view( -1)).cpu()
    if net.retAtt:
        makeRealScale( _data, _dataScales, True,
                       _rets[ 3], False, None, False)
        save.bvis[ _posit][ _poses] = _imgs.mul( _rets[ 3]).sum( dim=lastDims)\
            .div_( _dataScales.view( -1)).cpu()


def writeInSave( _posit, _psfs, inplace: bool, _statStep):
    _data, _dataScales, _getAdata = makeAdata( _psfs, inplace)
    _rets = inference( _data, _getAdata, _statStep, Infer.bas | Infer.upd)
    makeRealScale( _data, _dataScales, False,
                   _rets[ 0], True, _rets[ 1], True)
    save.invs[ _posit].append( _rets[ 0].cpu())
    save.rsts[ _posit].append( _rets[ 1].cpu())
    if net.retAtt:
        makeRealScale( _data, _dataScales, True,
                       _rets[ 3], True, _rets[ 4], True)
        save.atts[ _posit].append( _rets[ 2].cpu())
        save.bases[ _posit].append( _rets[ 3].cpu())
        save.brsts[ _posit].append( _rets[ 4].cpu())


def catAndReshape( _tensors, _leadShape):
    _temp = torch.cat( _tensors)
    return _temp.view( tuple( _leadShape) + _temp.shape[ 1:])


#%%

statEpoch = Epochs( * statItems)
statGrids = Epochs( * statItems)

for epoch in range( 1, Data.totalEpochs + 1):
    # train
    net.train()
    print( f'train {epoch}/{Data.totalEpochs}: ', end='')
    sys.stdout.flush()

    statStep = Steps( * statItems)
    dots = Dots( Data.batches)
    for step, data in enumerate( rloader, 1):
        dots.step( step)
        data = data.to( Data.device)
        inference( data, lambda: data.abs(), statStep, Infer.opt)
        del data

    print( ' train time:', timer.fullStr( epoch, Data.totalEpochs))
    for it in statItems:
        print( it, statStep.repr( it))
    statEpoch.update( statStep)

    if epoch not in Save.saveTimes:
        continue

    # eval
    net.eval()
    position = Save.saveTimes.index( epoch)

    with torch.inference_mode():
        save.models.append( deepCopy( net.state_dict()))
        save.optims.append( deepCopy( optim.state_dict()))

        # visualize
        print( f'vis {epoch}/{Data.totalEpochs}: ', end='')
        sys.stdout.flush()

        dots = Dots( len( iloader.dataset))
        totalPSFs = 0
        with Timer( Data.device) as infTimer:
            for poses, psfs, imgs in iloader:
                totalPSFs += len( poses)
                dots.step( totalPSFs)
                poses = poses.sub_( coordRange[ 0]).T.unbind()
                visualize( position, poses,
                           psfs.to( Data.device, copy=True), True,
                           imgs.to( Data.device))
                del poses, psfs, imgs

        print( ' inference time:', infTimer)

        # grids
        print( f'grids {epoch}/{Data.totalEpochs}: ', end='')
        sys.stdout.flush()

        statStep = Steps( * statItems)
        dots = Dots( psfNumber)
        totalPSFs = 0
        with Timer( Data.device) as infTimer:
            for psfs in psfChunks:
                totalPSFs += len( psfs)
                dots.step( totalPSFs)
                writeInSave( position,
                             psfs.to( Data.device, copy=True), True,
                             statStep)
                del psfs

        print( ' inference time:', infTimer)
        for it in statItems:
            print( it, statStep.repr( it))
        statGrids.update( statStep)

print( 'finished', timer)


#%%

print( 'postprocessing...')

save.stat = statEpoch.stateDict()
save.statGrids = statGrids.stateDict()

for it in range( len( Save.saveTimes)):
    save.invs[ it] = catAndReshape( save.invs[ it], psfLeadShape)
    save.rsts[ it] = catAndReshape( save.rsts[ it], psfLeadShape)
    if net.retAtt:
        save.atts[ it] = catAndReshape( save.atts[ it], psfLeadShape)
        save.bases[ it] = catAndReshape( save.bases[ it], psfLeadShape)
        save.brsts[ it] = catAndReshape( save.brsts[ it], psfLeadShape)

save.invs = tuple( save.invs)
save.rsts = tuple( save.rsts)
if net.retAtt:
    save.atts  = tuple( save.atts)
    save.bases = tuple( save.bases)
    save.brsts = tuple( save.brsts)

torch.save( save.toDict(), Save.saveBin)


#%% plot

if ids.dims == 2 and visualizeFor2d:
    saveName += '-png'
    if Save.saveIndex != '':
        saveName = '.'.join( [ saveName, Save.saveIndex])
    pngRoot = pl.Path( Save.savePath) / saveName
    pngRoot.mkdir( exist_ok=False)

    if Data.dsName in ['Marmousi2', 'Marmousi4', 'MarmousiK']:
        figRatio = 30
        vrange = -0.3, +0.3
    elif Data.dsName == 'Qikou':
        figRatio = 50
        vrange = -0.3, +0.3
    elif Data.dsName == 'Layer6':
        figRatio = 75
        vrange = -1.0, +1.0
    elif Data.dsName == 'Sigsbee2':
        figRatio = 75
        vrange = -0.1, +0.1
    else:               # fallback
        wn.warn( 'use default visualization setting')
        figRatio = 50
        vrange = -0.3, +0.3

    figSize = core.misc.Misc.figSize( save.oriImg.shape, figRatio)

    gimshow( save.oriImg / save.oriImg.amax(),
             figSize=figSize, vrange=vrange,
             fileName=str( pngRoot / 'ori.png'),
             pureSave=runInBackground)
    for ep, vis in zip( Save.saveTimes, save.vis):
        gimshow( vis / vis.amax(), figSize=figSize, vrange=vrange,
                 fileName=str( pngRoot / f'vis-{ep}.png'),
                 pureSave=runInBackground)
    if net.retAtt:
        for ep, bvis in zip( Save.saveTimes, save.bvis):
            gimshow( bvis / bvis.amax(), figSize=figSize, vrange=vrange,
                     fileName=str( pngRoot / f'bvis-{ep}.png'),
                     pureSave=runInBackground)

print( 'done')
