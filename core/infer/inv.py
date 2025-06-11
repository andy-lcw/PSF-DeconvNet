#%%

__all__ = ['invCenter', 'invIdentity']


#%%

import typing as tp

import torch


#%%

def invCenter( imgShape: tp.Sequence[ int], kerShape: tp.Sequence[ int]):
    assert len( imgShape) == len( kerShape) > 0
    cent = tuple( torch.tensor( kerShape)
                  .sub( torch.tensor( imgShape).remainder( 2))
                  .div( 2, rounding_mode='trunc').tolist())
    return cent


def invIdentity( imgShape: tp.Sequence[ int], kerShape: tp.Sequence[ int]):
    """img * ret = img"""
    cent = invCenter( imgShape, kerShape)
    delta = torch.zeros( * kerShape)
    delta[ cent] = 1
    return delta
