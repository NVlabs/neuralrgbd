'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

'''
The full KV-net framework,
support (still in progress) multiple-gpu training
'''

import torch

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from models.psm_submodule import *
import warping.homography as warp_homo
import math
import time

import models.basic as model_basic 
import models.Refine as model_refine

import mutils.misc as m_misc

class KVNET(nn.Module):
    r'''
    Inside this module, we will do the full KV-Net pipeline:
    * D-Net (feature extractiion + BV_cur estimation )
    * KV-Net 
    '''
    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net, 
                 if_refined = True, refineNet_name = 'DPV' , 
                 t_win_r = 2, refine_channel = 3, if_upsample_d = False
                 ):
        r'''
        inputs:

        refineNet_name - {'DGF', 'DPV'}

        if_refined - if use the refinement net for upsampling

        refine_channel - the # of channels for the guided image in the refinement net, by default 3 for rgb input image 
        ''' 

        super(KVNET, self).__init__()
        self.t_win_r = t_win_r 
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim = feature_dim, multi_scale=True)
            self.d_net = model_basic.D_NET_BASIC(
                    self.feature_extractor, cam_intrinsics, 
                    d_candi, sigma_soft_max, use_img_intensity=True, 
                    BV_log = True, output_features = True)
        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim = feature_dim) 
            self.d_net = model_basic.D_NET_BASIC(
                    self.feature_extractor, cam_intrinsics, 
                    d_candi, sigma_soft_max, use_img_intensity=True,
                    BV_log = True, output_features = False) 

        self.kv_net = model_basic.KV_NET_BASIC(3*(t_win_r*2+1) + 1, 
                feature_dim = KVNet_feature_dim, 
                up_sample_ratio = d_upsample_ratio_KV_net)

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.r_net = model_refine.RefineNet_DGF( refine_channel )
            elif self.refineNet_name == 'DPV':
                self.r_net = model_refine.RefineNet_DPV_upsample( 
                        int(self.feature_dim), int(self.feature_dim/2), 3, 
                        D = len(self.d_candi), upsample_D=self.if_upsample_d )

        # print # 
        print('KV-Net initialization:')
        print('with R-net: %r'%(self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s'%(self.refineNet_name) ) 

    def forward(self, ref_frame, src_frames, src_cam_poses, BatchIdx, cam_intrinsics=None, BV_predict=None, mGPU= False,
                IntMs=None, unit_ray_Ms_2D=None):
        r'''
        Inputs: 
        ref_frame - NCHW format tensor on GPU, N = 1
        src_frames - NVCHW: V - # of source views, N = 1 
        src_cam_poses - N x V x4 x4 - relative cam poses, N = 1
        BatchIdx - e.g. for 4 gpus: [0,1,2,3], used for indexing list input for multi-gpu training 
        cam_intrinsics - list of cam_intrinsics dict. 
        BV_predict - NDHW tensor, the predicted BV, from the last reference frame, N=1

        Outputs:
        dmap_cur_refined, dmap_kv_refined, BV_cur, BV_KV

        if refined on dpv, then dmap_cur_refined and dmap_kv_refined are refined dpvs

        NOTE:
        1. We should put ref_frame and src_frames and src_cam_poses into GPU before running the forward pass
        2. The purpose of enforcing N=1 is for multi-gpu running
        ''' 

        if isinstance(BV_predict, torch.Tensor):
            if m_misc.valid_dpv(BV_predict):
                assert BV_predict.shape[0] == 1 

        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            BV_cur = self.d_net(ref_frame, src_frames, src_cam_poses, BV_predict = None, debug_ipdb= False)  

        else:
            BV_cur, d_net_features = self.d_net(
                    ref_frame, src_frames, src_cam_poses, BV_predict = None, debug_ipdb= False) 

            d_net_features.append( ref_frame )

        if self.if_refined:
            dmap_cur_lowres = m_misc.depth_val_regression(BV_cur, self.d_candi, BV_log=True).unsqueeze(0) 

            if self.refineNet_name == 'DGF':
                dmap_cur_refined = self.r_net(dmap_cur_lowres, ref_frame)
            elif self.refineNet_name == 'DPV':
                dmap_cur_refined = self.r_net(torch.exp(BV_cur), img_features = d_net_features) 
        else:
            dmap_cur_refined = -1

        if not isinstance(BV_predict, torch.Tensor):
            #If the first time win., then return only BV_cur
            return dmap_cur_refined, dmap_cur_refined, BV_cur, BV_cur 

        elif not m_misc.valid_dpv( BV_predict ):
            return dmap_cur_refined, dmap_cur_refined, BV_cur, BV_cur 

        else:
            # KV-Net # 
            down_sample_rate = ref_frame.shape[3] / BV_cur.shape[3] 

            ref_frame_dw = F.avg_pool2d(ref_frame, int(down_sample_rate )).cuda()
            src_frames_dw = [ F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate )).cuda() 
                             for src_frame_ in src_frames.squeeze(0)] 

            Rs_src = [pose[:3, :3] for pose in src_cam_poses.squeeze(0)]
            ts_src = [pose[:3, 3] for pose in src_cam_poses.squeeze(0)] 

            # Warp the src-frames to the ref. view # 
            if mGPU:
                WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(src_frames_dw, self.d_candi, Rs_src, ts_src, IntMs, unit_ray_Ms_2D) 
            else:
                cam_intrin = cam_intrinsics[int(BatchIdx)]
                WAPRED_src_frames = warp_homo.warp_img_feats_v3(src_frames_dw, self.d_candi, Rs_src, ts_src, cam_intrin, ) 

            ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1,1,1]), 0,1)

            # Input to the KV-net #
            kvnet_in_vol = torch.cat((torch.cat(tuple(WAPRED_src_frames), dim=0), ref_frame_dw_rep, BV_cur - BV_predict), dim=0).unsqueeze(0) 

            # Run KV-net #
            BV_gain = self.kv_net( kvnet_in_vol )

            # Add back to BV_predict #
            DPV = torch.squeeze(BV_gain, dim=1) + BV_predict
            DPV = F.log_softmax(DPV, dim=1) 

            if self.if_refined:
                dmap_lowres = m_misc.depth_val_regression(DPV, self.d_candi, BV_log=True).unsqueeze(0) 
                if self.refineNet_name == 'DGF':
                    dmap_refined = self.r_net(dmap_lowres, ref_frame) 
                elif self.refineNet_name == 'DPV':
                    dmap_refined = self.r_net(torch.exp(DPV), img_features = d_net_features) 
            else:
                dmap_refined = -1 


            return dmap_cur_refined, dmap_refined, BV_cur, DPV 
