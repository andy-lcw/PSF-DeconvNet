#%%

__all__ = ['FourierTransform']


#%%

import math
import typing as tp

import torch


#%%

class FourierTransform:

    STP = tp.Sequence[ int]
    _FTP = tp.Tuple[ int, ...]
    FTP = tp.Union[ int, tp.Sequence[ int]]
    CentType = tp.Optional[ tp.Sequence[ int]]

    _exp_wt = {}
    _irft = {}

    # type conversion

    @staticmethod
    def toFCDtype( dtype: torch.dtype):
        if dtype in [ torch.float32, torch.complex64]:
            return torch.float32, torch.complex64
        elif dtype == [ torch.float64, torch.complex128]:
            return torch.float64, torch.complex128
        else:
            raise NotImplementedError

    # private helper

    @staticmethod
    def _ftp( freqs: FTP, dims: int):
        if isinstance( freqs, int):
            freqs = ( freqs,) * dims
        else:
            freqs = tuple( freqs)
            assert len( freqs) == dims
        return freqs

    @staticmethod
    def _fr( lastFreq: int):
        r = ( lastFreq - 1) // 2
        return lastFreq - r, r

    @classmethod
    def _get_exp_wt(cls, shape: STP, freqs: _FTP, rft: bool, fwd: bool,
                    dtype=( torch.float32, torch.complex64),
                    device: tp.Union[ str, torch.device] = 'cpu',
                    cent: CentType = None):

        def getFt( _fwd=None):
            if _fwd is None:
                _fwd = fwd

            ret = cls._exp_wt.get( ( shape, freqs, False, cdtype, device, cent), None)
            if ret is not None:
                return ret if _fwd else ret.conj()

            wt = 0
            for d, ( s, f, ct) in enumerate( zip( shape, freqs, cent), 1):
                tPos = dims * 2 - d
                t = torch.arange( s, device=device).sub( ct).to( fdtype).view( -1, * ( 1,) * tPos)
                wPos = dims - d
                w = torch.arange( f, device=device, dtype=fdtype) * torch.pi * 2 / f
                w = w.view( -1, * ( 1,) * wPos)
                wt = wt + w * t

            ret = torch.exp( -1j * wt)
            cls._exp_wt[ shape, freqs, False, cdtype, device, cent] = ret
            return ret if _fwd else ret.conj()

        def getRft():
            ret = cls._exp_wt.get( ( shape, freqs, True, cdtype, device, cent), None)
            if ret is not None:
                return ret[ 0] if fwd else ret[ 1]
            ret = getFt( True)

            f, r = cls._fr( freqs[ -1])
            ret = ret[ ..., :f]
            ret1 = ret.clone()
            ret1[ ..., 1:r+1] *= 2
            cls._exp_wt[ shape, freqs, True, cdtype, device, cent] = ret, ret1
            return ret if fwd else ret1

        dims = len( shape)
        assert len( freqs) == dims
        fdtype, cdtype = dtype

        shape = tuple( shape)
        # freqs = tuple( freqs)
        device = torch.device( device)

        if cent is None:
            cent = tuple( s // 2 for s in shape)
        else:
            cent = tuple( cent)
            assert len( cent) == dims

        if rft:
            return getRft()
        else:
            return getFt()

    @staticmethod
    def _norm( data: torch.Tensor, freqs: _FTP, fwd: bool, norm: str):
        numel = torch.tensor( freqs, dtype=torch.long).prod().item()
        if norm == 'ortho':
            norm = math.sqrt( numel)
        elif fwd and norm == 'forward':
            norm = numel
        elif not fwd and norm == 'backward':
            norm = numel
        else:
            return data
        return data / norm

    # fourier transforms

    @classmethod
    def ft(cls, data: torch.Tensor, dims: int, freqs: FTP, norm='backward', cent: CentType = None):
        freqs = cls._ftp( freqs, dims)
        dtype = cls.toFCDtype( data.dtype)
        # [b...], [sp...] x [sp...], [fr...] -> [b...], [fr...]
        ewt = cls._get_exp_wt( data.shape[ -dims:], freqs, False, True, dtype, data.device, cent)
        ret = torch.tensordot( data.to( dtype[ 1]), ewt, dims)
        return cls._norm( ret, freqs, True, norm)

    @classmethod
    def rft(cls, data: torch.Tensor, dims: int, freqs: FTP, norm='backward', cent: CentType = None):
        freqs = cls._ftp( freqs, dims)
        dtype = cls.toFCDtype( data.dtype)
        # [b...], [sp...] x [sp...], [fr...] -> [b...], [fr...]
        ewt = cls._get_exp_wt( data.shape[ -dims:], freqs, True, True, dtype, data.device, cent)
        ret = torch.tensordot( data.to( dtype[ 1]), ewt, dims)
        return cls._norm( ret, freqs, True, norm)

    @classmethod
    def ift(cls, ft: torch.Tensor, dims: int, invShape: STP, norm='backward', cent: CentType = None):
        freqs = tuple( ft.shape[ -dims:])
        dtype = cls.toFCDtype( ft.dtype)
        ewt = cls._get_exp_wt( invShape, freqs, False, False, dtype, ft.device, cent)
        ds = list( range( -dims, 0))
        # [b...], [fr...] x [isp...], [fr...] -> [b...], [isp...]
        ret = torch.tensordot( ft.to( dtype[ 1]), ewt, [ ds, ds])
        return cls._norm( ret, freqs, False, norm)

    @classmethod
    def irft(cls, rft: torch.Tensor, dims: int, freqs: FTP, invShape: STP, norm='backward', cent: CentType = None):
        assert len( invShape) == dims
        freqs = cls._ftp( freqs, dims)
        dtype = cls.toFCDtype( rft.dtype)
        ewt = cls._get_exp_wt( invShape, freqs, True, False, dtype, rft.device, cent)
        # [b...], [fr..., 2] x [isp...], [fr..., 2] -> [b...], [isp...]
        ewt = torch.view_as_real( ewt)
        rft = torch.view_as_real( rft.to( dtype[ 1]))
        ds = list( range( -dims-1, 0))
        ret = torch.tensordot( rft, ewt, [ ds, ds])
        return cls._norm( ret, freqs, False, norm)

    @classmethod
    def irft_ewt(cls, dims: int, freqs: FTP, invShape: STP,
                 dtype: torch.dtype, device: torch.device, cent: CentType = None):
        assert len( invShape) == dims
        freqs = cls._ftp( freqs, dims)
        dtype = cls.toFCDtype( dtype)
        # True or False
        ewt = cls._get_exp_wt( invShape, freqs, True, True, dtype, device, cent)
        return ewt


#%%

if __name__ == '__main__':

    for i in range( 4):
        shape = torch.randint( 3, 10, [4])
        print( shape, end='')
        data = torch.randn( * shape, dtype=torch.complex64)
        dims = 3
        freq = 64
        norm = 'ortho'
        fdata = FourierTransform.ft( data, dims, freq, norm=norm)
        idata = FourierTransform.ift( fdata, dims, data.shape[ - dims:], norm=norm)
        print( '', idata.sub( data).abs().max().item())

    for i in range( 4):
        shape = torch.randint( 3, 10, [4])
        print( shape, end='')
        data = torch.randn( * shape)
        dims = 3
        freq = 64
        norm = 'ortho'
        fdata = FourierTransform.rft( data, dims, freq, norm=norm)
        idata = FourierTransform.irft( fdata, dims, freq, data.shape[ - dims:], norm=norm)
        print( '', idata.sub( data).abs().max().item())
