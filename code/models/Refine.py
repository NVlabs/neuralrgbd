'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# models for refinement of DPV #
import torch
torch.backends.cudnn.benchmark=True 

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from models.psm_submodule import *
import warping.homography as warp_homo
import math
import time
import numpy as np
import models.m_submodule as m_submodule


class RefineNet_DPV_upsample(nn.Module):
    '''
    The refinement taking the DPV, using the D dimension as the feature dimension, plus the image features, 
    then upsample the DPV (4 time the input dpv resolution)
    ''' 

    def __init__(self, C0, C1, C2, D = 64, upsample_D = False):
        '''
        Inputs: 

        C0 - feature channels in .25 image resolution feature,
        C1 - feature cnahnels in .5 image resolution feature,
        C2 - feature cnahnels in 1 image resolution feature,

        D - the length of d_candi, we will treat the D dimension as the feature dimension
        upsample_D - if upsample in the D dimension 
        '''
        super(RefineNet_DPV_upsample, self).__init__() 
        in_channels = D + C0 

        if upsample_D:
            D0 = 2*D
            D1 = 2*D0
        else:
            D0 = D
            D1 = D 

        self.conv0 = m_submodule.conv2d_leakyRelu(
                ch_in = in_channels, ch_out=in_channels, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.conv0_1 = m_submodule.conv2d_leakyRelu(
                ch_in = in_channels, ch_out=in_channels, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.trans_conv0 = m_submodule.conv2dTranspose_leakyRelu(
                ch_in= in_channels, ch_out= D0, kernel_size=4, stride=2, pad=1, use_bias=True ) 

        self.conv1 = m_submodule.conv2d_leakyRelu(
                ch_in = D0 + C1, ch_out= D0 + C1, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.conv1_1 = m_submodule.conv2d_leakyRelu(
                ch_in=D0 + C1, ch_out= D0 + C1, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.trans_conv1 = m_submodule.conv2dTranspose_leakyRelu(
                ch_in=D0 + C1, ch_out= D1, kernel_size=4, stride=2, pad=1, use_bias=True )

        self.conv2 = m_submodule.conv2d_leakyRelu(
                ch_in = D1 + C2, ch_out = D1 + C2, kernel_size=3, stride=1, pad=1, use_bias=True)

        self.conv2_1 = m_submodule.conv2d_leakyRelu(
                ch_in= D1 + C2, ch_out= D1, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.conv2_2 = nn.Conv2d(D1, D1, kernel_size=3, stride=1, padding=1, bias=True) 

        self.apply(self.weight_init)
    
    def forward(self, dpv_raw, img_features):
        '''
        dpv_raw - the low resolution (.25 image size) dpv (N D H W)
        img_features - list of image features [ .25 image size, .5 image size, 1 image size]

        NOTE:
        dpv_raw from 0, 1 (need to exp() if in log scale)

        output dpv in log-scale
        ''' 


        conv0_out = self.conv0(torch.cat( [dpv_raw, img_features[0] ], dim=1))
        conv0_1_out = self.conv0_1( conv0_out)

        trans_conv0_out = self.trans_conv0(conv0_1_out)

        conv1_out = self.conv1(torch.cat([trans_conv0_out, img_features[1]], dim=1) )
        conv1_1_out = self.conv1_1(conv1_out)

        trans_conv1_out = self.trans_conv1(conv1_1_out)
        conv2_out = self.conv2(torch.cat([trans_conv1_out, img_features[2]], dim=1) )
        conv2_1_out = self.conv2_1(conv2_out)
        conv2_2_out = self.conv2_2(conv2_1_out)

        # normalization, assuming input dpv is in the log scale 
        dpv_refined = F.log_softmax( conv2_2_out, dim=1 )

        return dpv_refined

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            print(' RefineNet_UNet2D: init conv2d')
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            print(' init Batch2D')
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            print(' init Linear')
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            print(' init transposed 2d')
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5 

            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np)) 


class RefineNet_Unet2D(nn.Module):
    '''
    Thre refinement block based on DemoNet
    This net takes input low res dpv statistics and high res rgb image
    It outputs the high res depth map
    ''' 
    def __init__(self, in_channels):
        '''
        in_channels - for example, if we use some statistics from DPV, plus the raw rgb input image, then
                      in_channels = 3 + # of statistics we used from DPV
                      Statistics of DPV can includes {expected mean, variance, min_v, max_v etc. }
        '''
        super(RefineNet_Unet2D, self).__init__()

        self.conv0 = m_submodule.conv2d_leakyRelu(ch_in = in_channels, ch_out=32, kernel_size=3, stride=1, pad=1, use_bias=True) 
        self.conv0_1 = m_submodule.conv2d_leakyRelu(ch_in = 32, ch_out=32, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.conv1 = m_submodule.conv2d_leakyRelu(ch_in = 32, ch_out=64, kernel_size=3, stride=2, pad=1, use_bias=True)
        self.conv1_1 = m_submodule.conv2d_leakyRelu(ch_in=64, ch_out=64, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.conv2 = m_submodule.conv2d_leakyRelu(ch_in = 64, ch_out=128, kernel_size=3, stride=2, pad=1, use_bias=True)
        self.conv2_1 = m_submodule.conv2d_leakyRelu(ch_in=128, ch_out=128, kernel_size=3, stride=1, pad=1, use_bias=True) 
        
        self.trans_conv0 = m_submodule.conv2dTranspose_leakyRelu(ch_in=128, ch_out=64, kernel_size=4, stride=2, pad=1, use_bias=True )
        self.trans_conv1 = m_submodule.conv2dTranspose_leakyRelu(ch_in=128, ch_out=32, kernel_size=4, stride=2, pad=1, use_bias=True )

        self.conv3 =  m_submodule.conv2d_leakyRelu(ch_in = 64, ch_out=16, kernel_size=3, stride=1, pad=1, use_bias=True) 
        self.conv3_1 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias= True)

        self.apply(self.weight_init)
    
    def forward(self, dpv_features_lowres, img):
        '''
        dpv_features_lowres, img - NCHW tensors
        Statistics of DPV can includes {expected mean, variance, min_v, max_v etc. }
        '''
        dpv_features_upsampe = F.upsample(dpv_features_lowres, [img.shape[2], img.shape[3]], mode = 'bilinear', align_corners=True) 

        conv0_in = torch.cat([img, dpv_features_upsampe], dim=1)
        conv0_out= self.conv0(conv0_in) 
        conv0_1_out = self.conv0_1(conv0_out)

        conv1_out = self.conv1(conv0_1_out)
        conv1_1_out = self.conv1_1(conv1_out)

        conv2_out = self.conv2(conv1_1_out)
        conv2_1_out = self.conv2_1(conv2_out)

        up_conv0_out = self.trans_conv0(conv2_1_out)
        up_conv1_out = self.trans_conv1(torch.cat([up_conv0_out, conv1_1_out], dim=1))

        conv3_out = self.conv3( torch.cat([up_conv1_out, conv0_out], dim=1) )
        dmap_refined = self.conv3_1(conv3_out) 

        return dmap_refined 

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            print(' RefineNet_UNet2D: init conv2d')
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            print(' init Batch2D')
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            print(' init Linear')
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            print(' init transposed 2d')
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5 

            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np)) 

class RefineNet_DeMoN(nn.Module):
    '''
    Thre refinement block used in DemoNet
    This net takes input low res depth map and high res rgb image
    It outputs the high res depth map
    ''' 
    def __init__(self, img_ch = 3):
        super(RefineNet_DeMoN, self).__init__()

        self.conv0 = m_submodule.conv2d_leakyRelu(ch_in = img_ch+1, ch_out=32, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.conv1 = m_submodule.conv2d_leakyRelu(ch_in = 32, ch_out=64, kernel_size=3, stride=2, pad=1, use_bias=True)
        self.conv1_1 = m_submodule.conv2d_leakyRelu(ch_in=64, ch_out=64, kernel_size=3, stride=1, pad=1, use_bias=True) 

        self.conv2 = m_submodule.conv2d_leakyRelu(ch_in = 64, ch_out=128, kernel_size=3, stride=2, pad=1, use_bias=True)
        self.conv2_1 = m_submodule.conv2d_leakyRelu(ch_in=128, ch_out=128, kernel_size=3, stride=1, pad=1, use_bias=True) 
        
        self.trans_conv0 = m_submodule.conv2dTranspose_leakyRelu(ch_in=128, ch_out=64, kernel_size=4, stride=2, pad=1, use_bias=True )
        self.trans_conv1 = m_submodule.conv2dTranspose_leakyRelu(ch_in=128, ch_out=32, kernel_size=4, stride=2, pad=1, use_bias=True )

        self.conv3 =  m_submodule.conv2d_leakyRelu(ch_in = 64, ch_out=16, kernel_size=3, stride=1, pad=1, use_bias=True) 
        self.conv3_1 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias= True)

        self.apply(self.weight_init)
    
    def forward(self, dmap_lowres, img):
        '''
        dmap_lowres, img - NCHW tensors
        '''
        dmap_upsample = F.upsample(dmap_lowres, [img.shape[2], img.shape[3]], mode = 'bilinear', align_corners=True) 
        conv0_in = torch.cat([img, dmap_upsample], dim=1)
        conv0_out= self.conv0(conv0_in) 

        conv1_out = self.conv1(conv0_out)
        conv1_1_out = self.conv1_1(conv1_out)

        conv2_out = self.conv2(conv1_1_out)
        conv2_1_out = self.conv2_1(conv2_out)

        up_conv0_out = self.trans_conv0(conv2_1_out)
        up_conv1_out = self.trans_conv1(torch.cat([up_conv0_out, conv1_1_out], dim=1))

        conv3_out = self.conv3( torch.cat([up_conv1_out, conv0_out], dim=1) )
        dmap_refined = self.conv3_1(conv3_out) 

        return dmap_refined 

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            print('RefineNet_DeMoN: init conv2d')
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            print('RefineNet_DeMoN: init Batch2D')
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            print('RefineNet_DeMoN: init Linear')
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            print('RefineNet_DeMoN: init transposed 2d')
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5 
            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np)) 


class RefineNet(nn.Module):
    '''
    The refinement block, including encoder-decoder to include context information,
    plus skipping connections to include low-level features

    Output vol has the same dimension as the input vol.
    '''

    def __init__(self, in_channels, deconv_upsample = True):
        '''
        deconv_upsample: if use deconv. operation to do the upsampling
        if False, then use the nn.UpSample() with nearest neighborhood method to do the upsampling
        '''

        super(RefineNet, self).__init__()
        self.in_channels = in_channels

        dw_ker_sz = 3
        dw_padding = 1

        up_ker_sz = 3
        up_padding = 1

        output_padding = 1

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels*2,
                                             kernel_size=dw_ker_sz, stride=2,
                                             pad= dw_padding), nn.ReLU())

        self.conv2 = nn.Sequential(convbn_3d(in_channels*2, in_channels*2,
                                             kernel_size=dw_ker_sz, stride=1,
                                             pad= dw_padding), nn.ReLU())

        self.conv3 = nn.Sequential(convbn_3d(in_channels*2, in_channels*2,
                                             kernel_size=dw_ker_sz, stride=2,
                                             pad= dw_padding), nn.ReLU())

        self.conv4 = nn.Sequential(convbn_3d(in_channels*2, in_channels*2,
                                             kernel_size=dw_ker_sz, stride=1,
                                             pad= dw_padding), nn.ReLU())
        
        if deconv_upsample:
            self.conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels*2,
                                                          in_channels*2,
                                                          kernel_size=up_ker_sz,
                                                          padding= up_padding,
                                                          output_padding= output_padding,
                                                          stride=2,bias=False),
                                       nn.BatchNorm3d(in_channels*2)) 

            self.conv6 = nn.Sequential(nn.ConvTranspose3d(in_channels*2,
                                                          in_channels,
                                                          kernel_size=up_ker_sz,
                                                          padding= up_padding,
                                                          output_padding= output_padding,
                                                          stride=2,bias=False),
                                       nn.BatchNorm3d(in_channels)) 
        else:
           self.conv5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      convbn_3d(in_channels*2, in_channels*2,
                                                kernel_size=up_ker_sz, stride=1,
                                                pad = up_padding))

           self.conv6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      convbn_3d(in_channels*2, in_channels*1,
                                                kernel_size=up_ker_sz, stride=1,
                                                pad = up_padding))


        self.classif1 = nn.Sequential(convbn_3d(in_channels, in_channels,
                                                up_ker_sz, 1, up_padding),
                                      nn.ReLU(), nn.Conv3d(in_channels, 1,
                                                           kernel_size=up_ker_sz,
                                                           padding= up_padding,
                                                           stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, vol, img_vol):
        '''
        inputs: 
        vol, img_vol - N C D H W
        ''' 
        vol_in = torch.cat((vol, img_vol), dim=1).contiguous()
        assert vol_in.shape[1]  == self.in_channels

        # Encoding #
        dw1_vol = self.conv1(vol_in)
        dw1_vol_r = self.conv2(dw1_vol)
        dw2_vol = self.conv3(dw1_vol_r)
        dw2_vol_r = self.conv4(dw2_vol)

        # Decoding #
        up1_vol = F.relu(self.conv5(dw2_vol_r) + dw1_vol_r) 
        up1_vol_r = F.relu(self.conv6(up1_vol) + vol) 
        out = self.classif1(up1_vol_r)
        return out 

class RefineNet_UNet_Res(nn.Module):
    '''
    The refinement net, including encoding and decoding branch.
    Here we will concatenate the features from the lower levels in the decoding phase, rather than using the residual block
    to include the low level features 
    '''

    def __init__(self, in_channels):
        super(RefineNet_UNet_Res, self).__init__()

        self.in_channels = in_channels
        dw_ker_sz = 3
        dw_padding = 1
        up_ker_sz = 3
        up_padding = 1

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels*2,
                                             kernel_size = dw_ker_sz, stride=2,
                                             pad= dw_padding), nn.ReLU())

        self.conv2 = nn.Sequential(convbn_3d(in_channels*2, in_channels*2,
                                             kernel_size = dw_ker_sz, stride=1,
                                             pad= dw_padding), nn.ReLU())

        self.conv3 = nn.Sequential(convbn_3d(in_channels*2, in_channels*4,
                                             kernel_size = dw_ker_sz, stride=2,
                                             pad= dw_padding), nn.ReLU())

        self.conv4 = nn.Sequential(convbn_3d(in_channels*4, in_channels*4,
                                             kernel_size = dw_ker_sz, stride=1,
                                             pad= dw_padding), nn.ReLU())

        # conv5: in: cat(conv4_res_up, conv2_res)
        self.conv5 = nn.Sequential(convbn_3d(in_channels*4 + in_channels*2, in_channels*2,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        self.conv5_r = nn.Sequential(convbn_3d(in_channels*2, in_channels*2,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        # conv6 input:  cat(conv5_res_up, input_vol)
        self.conv6 = nn.Sequential(convbn_3d(in_channels*2 + in_channels, in_channels*1,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        self.conv6_r = nn.Sequential(convbn_3d(in_channels*1, in_channels*1,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        self.conv7 = nn.Sequential(convbn_3d(in_channels*1, 1,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest',)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, vol, img_vol):
        '''
        inputs: 
        vol, img_vol - N C D H W
        '''
        vol_in = torch.cat((vol, img_vol), dim=1).contiguous()
        assert vol_in.shape[1]  == self.in_channels

        # encoding #
        dw1_vol = self.conv1(vol_in)
        dw1_vol_r = self.conv2(dw1_vol)
        dw2_vol = self.conv3(dw1_vol_r)
        dw2_vol_r = self.conv4(dw2_vol)

        # decoding #
        up1_vol = self.conv5( torch.cat((self.upsample(dw2_vol_r), dw1_vol_r),  dim=1) )
        up1_vol_r = self.conv5_r(up1_vol)
        up2_vol = self.conv6( torch.cat((self.upsample(up1_vol_r), vol_in),  dim=1) )
        up2_vol_r = self.conv6_r(up2_vol)
        out_vol = F.relu(self.conv7(up2_vol_r) + vol)

        return out_vol


class RefineNet_UNet(nn.Module):
    '''
    The refinement net, including encoding and decoding branch.
    Here we will concatenate the features from the lower levels in the decoding phase, rather than using the residual block
    to include the low level features 
    '''
    def __init__(self, in_channels):
        super(RefineNet_UNet, self).__init__()

        self.in_channels = in_channels
        dw_ker_sz = 3
        dw_padding = 1
        up_ker_sz = 3
        up_padding = 1

        self.conv1 = nn.Sequential(convbn_3d_bias(in_channels, in_channels*2,
                                             kernel_size = dw_ker_sz, stride=2,
                                             pad= dw_padding), nn.ReLU())

        self.conv2 = nn.Sequential(convbn_3d_bias(in_channels*2, in_channels*2,
                                             kernel_size = dw_ker_sz, stride=1,
                                             pad= dw_padding), nn.ReLU())

        self.conv3 = nn.Sequential(convbn_3d_bias(in_channels*2, in_channels*4,
                                             kernel_size = dw_ker_sz, stride=2,
                                             pad= dw_padding), nn.ReLU())

        self.conv4 = nn.Sequential(convbn_3d_bias(in_channels*4, in_channels*4,
                                             kernel_size = dw_ker_sz, stride=1,
                                             pad= dw_padding), nn.ReLU())

        # conv5: in: cat(conv4_res_up, conv2_res)
        self.conv5 = nn.Sequential(convbn_3d_bias(in_channels*4 + in_channels*2, in_channels*2,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        self.conv5_r = nn.Sequential(convbn_3d_bias(in_channels*2, in_channels*2,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        # conv6 input:  cat(conv5_res_up, input_vol)
        self.conv6 = nn.Sequential(convbn_3d_bias(in_channels*2 + in_channels, in_channels*1,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        self.conv6_r = nn.Sequential(convbn_3d_bias(in_channels*1, in_channels*1,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        self.conv7 = nn.Sequential(convbn_3d_bias(in_channels*1, 1,
                                             kernel_size = up_ker_sz, stride=1,
                                             pad= up_padding), nn.ReLU())

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest',)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, vol, img_vol):
        '''
        inputs: 
        vol, img_vol - N C D H W
        '''
        vol_in = torch.cat((vol, img_vol), dim=1).contiguous()
        assert vol_in.shape[1]  == self.in_channels

        # encoding #
        dw1_vol = self.conv1(vol_in)
        dw1_vol_r = self.conv2(dw1_vol)
        dw2_vol = self.conv3(dw1_vol_r)
        dw2_vol_r = self.conv4(dw2_vol)

        # decoding #
        up1_vol = self.conv5( torch.cat((self.upsample(dw2_vol_r), dw1_vol_r),  dim=1) )
        up1_vol_r = self.conv5_r(up1_vol)
        up2_vol = self.conv6( torch.cat((self.upsample(up1_vol_r), vol_in),  dim=1) )
        up2_vol_r = self.conv6_r(up2_vol)
        out_vol = self.conv7(up2_vol_r) 
        return out_vol

'''
Refinement net based on Deep Guided Filter 
'''

import models.GF.guided_filter as DGF
class RefineNet_DGF(nn.Module): 
    def __init__(self, in_channels):
        super(RefineNet_DGF, self).__init__()
        self.in_channels = in_channels 
        self.dgf = DGF.GuidedFilter(r=1, eps=1e-8)
        self.feature_ext = nn.Sequential(
                nn.Conv2d(3, 64, 1), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(64, 1, 1)) 

        # Initialization # 
        self.apply( self.weight_init )

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5 
            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np))

    def forward(self, dmap, img_guide, ):
        '''
        inputs: 
        dmap, img_guide - NCHW format 
        Make sure # of channels for dmap = 1 
        '''
        N,C, H, W = img_guide.size()
        assert dmap.dim()==img_guide.dim()==4 and dmap.size()[1] ==1, 'input format is wrong' 
        assert self.in_channels == img_guide.shape[1]

        Nd,Cd, Hd, Wd = dmap.size() 
        up_ratio_H = int(H/Hd)
        up_ratio_W = int(W/Wd)

        assert up_ratio_H == up_ratio_W, 'aspect ratio should correspond'
        assert up_ratio_H > 1, 'guided image resolution should > depth res.'

        dmap_up = F.upsample( dmap, scale_factor = up_ratio_H, align_corners = True, mode='bilinear') 
        feat_img_guide = self.feature_ext(img_guide)
        dmap_refined = self.dgf(feat_img_guide, dmap_up)

        return dmap_refined 
