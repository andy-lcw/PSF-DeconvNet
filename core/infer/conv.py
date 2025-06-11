#%%

__all__ = ['fullConv2d', 'fullConv3d', 'fullConv',
           'batchedFullConv2d', 'batchedFullConv3d', 'batchedFullConv']


#%%

import torch
import torch.nn.functional as F


#%%

def fullConv2d( img: torch.Tensor, ker: torch.Tensor):
    assert img.ndim == ker.ndim == 2
    H, W = ker.shape
    if H == W:
        padding = H - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1])
        padding = 0
    return F.conv2d( img[ None, None], ker[ None, None], padding=padding)[ 0, 0]


def fullConv3d( img: torch.Tensor, ker: torch.Tensor):
    assert img.ndim == ker.ndim == 3
    D, H, W = ker.shape
    if D == H == W:
        padding = D - 1
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1, D-1, D-1])
        padding = 0
    return F.conv3d( img[ None, None], ker[ None, None], padding=padding)[ 0, 0]


def fullConv( img: torch.Tensor, ker: torch.Tensor):
    if img.ndim == 2:
        return fullConv2d( img, ker)
    elif img.ndim == 3:
        return fullConv3d( img, ker)
    else:
        raise NotImplementedError


#%%

def batchedFullConv2d( img: torch.Tensor, ker: torch.Tensor):
    b, h, w = img.shape
    b_, H, W = ker.shape
    assert b == b_

    if H == W:
        padding = H - 1
        h2, w2 = h, w
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1])
        h2, w2 = img.shape[ 1:]
        padding = 0

    imgR = img.view( 1, b, h2, w2)
    kerR = ker.view( b, 1, H, W)
    return F.conv2d( imgR, kerR, padding=padding, groups=b).squeeze( 0)


def batchedFullConv3d( img: torch.Tensor, ker: torch.Tensor):
    b, d, h, w = img.shape
    b_, D, H, W = ker.shape
    assert b == b_

    if D == H == W:
        padding = D - 1
        d2, h2, w2 = d, h, w
    else:
        img = F.pad( img, [ W-1, W-1, H-1, H-1, D-1, D-1])
        d2, h2, w2 = img.shape[ 1:]
        padding = 0

    imgR = img.view( 1, b, d2, h2, w2)
    kerR = ker.view( b, 1, D, H, W)
    return F.conv3d( imgR, kerR, padding=padding, groups=b).squeeze( 0)


def batchedFullConv( img: torch.Tensor, ker: torch.Tensor):
    if img.ndim == 3:
        return batchedFullConv2d( img, ker)
    elif img.ndim == 4:
        return batchedFullConv3d( img, ker)
    else:
        raise NotImplementedError
