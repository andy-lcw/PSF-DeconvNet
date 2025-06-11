#%%

from .model import Model

from itertools import product

import torch


#%%

def test():

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device( 0)

    demoInput = torch.randn( 16, 50, 50).cuda()

    for config in product( [ False, True], [ False, True],
                           [ False, True], range( 12)):
        print( config, end=' ')
        model = Model( * config)
        model.init().cuda()
        ret, att, base = model( demoInput, True)
        if model.retAtt:
            assert att is not None and base is not None
            print( round( ret.std( False).item(), 2),
                   round( base.std( False).item(), 2))
        else:
            assert att is None and base is None
            print( round( ret.std( False).item(), 2))


#%%

if __name__ == '__main__':
    test()
