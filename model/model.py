#%%

__all__ = ['Model']


#%%

from .mlp import MLP
from .encDec import Preprocess, Postprocess, Dict, DMLP

from core.infer import invIdentity

import typing as tp

import torch
import torch.nn as nn


#%%

class Model( nn.Module):

    def __init__( self,
                  inputShape: tp.Sequence[ int],
                  inputInFreqDomain: bool,
                  outputShape: tp.Sequence[ int],
                  outputInFreqDomain: bool,
                  decArch: int):
        """
        overall data flow:
            PSF -- rfft [^1] --> enc input
            enc input -- MLP (enc) --> code
            code -- arch (dec) [*] --> dec output
            dec output -- inv rfft [^2] --> final output [^3]

        [^1]: use rfft if inputInFreqDomain;
        [^2]: use irfft if outputInFreqDomain.
              all rfft and irfft is designed to keep the std, i.e.,
              Std[ (i)rfft(x) ] = Std[x].
        [^3]: if largerInvSize, the shape of output is [60, 60];
              if not, [50, 50] (the same as PSF)

        * arch 0: dec is MLP (random init);
          Std[dec output] ~= 1

        * arch 1: dec is MLPR (MLP-Rescaled, rescaled init);
          std[dec output] ~= 0.02, i.e.,
            code -- MLP --> result
            result * 0.02 --> dec output

        * arch 2: dec is Dict (random init from N(0,1));
          Std[dec output] = 1

        * arch 3: dec is DictD (Dict-delta, delta init);
          Std[dec output] = 0.02 (0.0167 if largerInvSize - [60, 60])

        * arch 4: dec is Dict-MLP;
            code -- Dict -- MLP --> dec output

        * arch 5: dec is DMLPR (Dict-MLPR);
          Std[dec output] ~= 0.02, i.e.,
            code -- Dict -- MLP -- *0.02 --> dec output

        * arch 6: dec is Dict + MLP;
          Std[dec output] ~= \sqrt{2}, i.e.,
            code -- Dict (random init) --> dec output 1
            code -- MLP --> dec output 2
            dec output 1 + dec output 2 --> dec output

        * arch 7: dec is Dict + MLPR;
            code -- MLP -- *0.02 --> dec output 2

        * arch 8: dec is DictD + MLP;
            code -- Dict (delta init) --> dec output 1

        * arch 9: dec is DictD + MLPR;

        * arch 10: dec is DictD + DMLPR;

        * arch 11: dec is DictD + DMLPR-0.01;
            code -- MLP -- *0.01 --> dec output 2

        arch 2-3, 6-11 will also return an attention map, i.e.,
            code -- Dict / DictD --> attention
            code -- Dict / DictD --> dec output (or 1)
          and a "base" version of deconvolution kernel, i.e.,
          the kernel predicted by the dictionary without the other branch.

        Within architectures of only one branch, a smaller output variance (e.g.,
        arch 1, 3, and 5) can stabilize the training process for L2 loss.  It has
        nearly no effects for any rescaled loss since they apply the 'rescaled'
        operation, normalize and balance the output variance.

        For dual-branched architectures, imbalanced variance of two branches can
        implicitly disable the branch of a smaller variance (e.g., arch 7 and 8).
        The final version, arch 11, is used in many previous experiments as a
        perfect version.  However, the difference between arch 10 and 11 is
        unknown.  I'm not sure whether arch 11 is definitely better than arch 10.

        MLPs need more time for training, but they can fit (nearly) any smooth
        and continuous functions.  Dictionaries train faster, but they are
        linear, which means much smaller capacity and less representation
        ability.  The dictionary branch is an effort to fast fit and extract
        common patterns of deconvolution kernels, while the other branch is used
        to carve their differences.  Another dictionary is prepended to the MLP
        for enhancing the model capacity.
        """
        super().__init__()

        assert 0 <= decArch <= 11
        self.inputShape = tuple( inputShape)
        self.inputInFreqDomain = inputInFreqDomain
        self.outputShape = tuple( outputShape)
        self.outputInFreqDomain = outputInFreqDomain
        self.decArch = decArch

        # optional FFT
        self.pre = Preprocess( self.inputShape, self.inputInFreqDomain)
        self.post = Postprocess( self.outputShape, self.outputInFreqDomain)

        # dims
        def _genDims( _dim):
            if _dim in [ 20*20, 20*11*2,
                         50*50, 50*26*2]:
                return _dim, 512, 128, 64
            elif _dim in [ 21*21*21, 21*21*11*2,
                           80*20*20, 80*20*11*2]:
                return _dim, 1024, 256, 64
            else:
                raise NotImplementedError

        # enc
        encDims = _genDims( self.pre.outDim)
        self.enc = MLP( * encDims, hiddenBias=True, outputBias=False)

        # dec
        decValue = invIdentity( self.inputShape, self.outputShape)
        if self.outputInFreqDomain:
            decValue = self.post.fft.fwd( decValue)
        decValue = decValue.flatten()
        decDim = decValue.numel()
        decDims = tuple( reversed( _genDims( decDim)))

        self.register_module( 'dec1', None)
        self.retAtt = False
        if self.decArch in [ 2, 3, 6, 7, 8, 9, 10, 11]:
            value = None
            if self.decArch not in [ 2, 6, 7]:
                value = decValue
            self.dec1 = Dict( 64, 64, decDim, value=value)
            self.retAtt = True

        self.register_module( 'dec2', None)
        self.rate2 = 1
        if self.decArch in [ 0, 1, 6, 7, 8, 9]:
            self.dec2 = MLP( * decDims, hiddenBias=True, outputBias=False)
            if self.decArch in [ 1, 7, 9]:
                self.rate2 = 0.02
        elif self.decArch in [ 4, 5, 10, 11]:
            self.dec2 = DMLP( 64, 64, * decDims)
            if self.decArch in [ 5, 10]:
                self.rate2 = 0.02
            elif self.decArch == 11:
                self.rate2 = 0.01

    def init(self):
        self.enc.init()
        if self.dec1 is not None:
            self.dec1.init()
        if self.dec2 is not None:
            self.dec2.init()
        return self

    def forward(self, psf: torch.Tensor, retBase: bool):
        code = self.enc( self.pre( psf))

        if self.dec1 is None:
            ret = self.dec2( code)
            if self.rate2 != 1:
                ret = self.rate2 * ret
            return self.post( ret), None, None

        base, att = self.dec1( code)
        if self.dec2 is None:
            ret = self.post( base)
            if retBase:
                return ret, att, ret
            return ret, None, None

        ret = self.post( base.add( self.dec2( code), alpha=self.rate2))
        if retBase:
            return ret, att, self.post( base)
        return ret, None, None

    def valueParams(self):
        for m in self.modules():
            if isinstance( m, Dict):
                yield from m.valueParam()

    def normalParams(self):
        valueParams = set( self.valueParams())
        return tuple( p for p in self.parameters() if p not in valueParams)
