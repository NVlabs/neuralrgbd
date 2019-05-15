'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''
'''
Frequently used submodules 
'''
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def conv2d_leakyRelu(ch_in, ch_out, kernel_size, stride, pad, use_bias=True, dilation = 1):
    r'''
    Conv2d + leakyRelu
    '''
    return nn.Sequential( 
            nn.Conv2d(
                ch_in, ch_out, kernel_size=kernel_size, stride = stride,
                padding = dilation if dilation >1 else pad, dilation = dilation, bias= use_bias),
            nn.LeakyReLU())

def linear_leakyRelu(ch_in, ch_out, use_bias=True,):
    r'''
    Linear + leakyRelu
    '''
    return nn.Sequential( 
            nn.Linear(ch_in, ch_out, bias= use_bias),
            nn.LeakyReLU())

def conv2dTranspose_leakyRelu(ch_in, ch_out, kernel_size, stride, pad, use_bias = True, dilation=1 ):
    r'''
    ConvTrans2d + leakyRelu
    '''
    return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size = kernel_size, stride =stride,
                padding= pad, bias = use_bias, dilation = dilation),
            nn.LeakyReLU())


