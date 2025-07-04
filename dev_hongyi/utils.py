from collections import OrderedDict

import numpy as np
import torch



def print_dict(d, space=0):
    '''
    Recursively print a dictionary
    '''
    for k, v in d.items():
        if type(v) is int or type(v) is float or type(v) is str:
            print(" " * space, str(k) + ":", type(v), v)
        elif type(v) is np.ndarray or type(v) is torch.Tensor:
            print(" " * space, str(k) + ":", type(v), v.shape)
        elif type(v) is list or type(v) is tuple:
            print(" " * space, str(k) + ":", type(v), len(v))
        elif type(v) is dict or type(v) is OrderedDict:
            print(" " * space, str(k) + ":")
            print_dict(v, space + 4)
        else:
            print(" " * space, str(k) + ":", type(v), "UNDEFINED_FORMAT")