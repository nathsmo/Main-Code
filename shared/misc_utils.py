# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import print_function

import json
import math
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch

import numpy as np
import scipy.misc 
from io import BytesIO         # Python 3.x

print_grad = True

class printOut(object):
    def __init__(self,f=None ,stdout_print=True):
        ''' 
        This class is used for controlling the printing. It will write in a 
        file f and screen simultanously.
        '''
        self.out_file = f
        self.stdout_print = stdout_print

    def print_out(self, s, new_line=True):
        """Similar to print but with support to flush and output to a file."""
        if isinstance(s, bytes):
            s = s.decode("utf-8")

        if self.out_file:
            self.out_file.write(s)
            if new_line:
                self.out_file.write("\n")
        self.out_file.flush()

        # stdout
        if self.stdout_print:
            print(s, end="", file=sys.stdout)
            if new_line:
                sys.stdout.write("\n")
            sys.stdout.flush()

    def print_time(self,s, start_time):
        """Take a start time, print elapsed duration, and return a new time."""
        self.print_out("%s, time %ds, %s." % (s, (time.time() - start_time) +"  " +str(time.ctime()) ))
        return time.time()


def get_time():
    '''returns formatted current time'''
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
 

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
        pass

def debug_tensor():
        pass

def has_nan(datum, tensor):
        if hasattr(tensor, 'dtype'):
                if (np.issubdtype(tensor.dtype, np.float) or
                        np.issubdtype(tensor.dtype, np.complex) or
                        np.issubdtype(tensor.dtype, np.integer)):
                        return np.any(np.isnan(tensor))
                else:
                        return False
        else:
                return False


def openAI_entropy(logits):
    a0 = logits - torch.max(logits, dim=2, keepdim=True)[0]
    ea0 = torch.exp(a0)
    z0 = torch.sum(ea0, dim=2, keepdim=True)
    p0 = ea0 / z0
    return torch.mean(torch.sum(p0 * (torch.log(z0) - a0), dim=2))

def softmax_entropy(p0):
    return -torch.sum(p0 * torch.log(p0 + 1e-6), dim=1)


def Dist_mat(A):
        # A is of shape [batch_size x nnodes x 2].
        # return: a distance matrix with shape [batch_size x nnodes x nnodes]
    nnodes = A.shape[1]
    A1 = A.unsqueeze(1).repeat(1, nnodes, 1, 1)
    A2 = A.unsqueeze(2).repeat(1, 1, nnodes, 1)
    dist = torch.norm(A1 - A2, dim=3)
    return dist
