'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import warping.homography as warp_homo
import mutils.misc as m_misc

import numpy as np


def test(model_KV, d_candi, Cam_Intrinsics, t_win_r, 
         Ref_Dats, Src_Dats, Src_CamPoses , BV_predict, 
         cam_pose_next = None, R_net = False,  
         Cam_Intrinsics_imgsize = None, ref_indx = None):
    '''
    Test the trained KV-Net
    '''

    nGPU = 1 # should set to 1 for testing
    BatchIdx_range = torch.FloatTensor(np.arange(nGPU)) 

    ref_frame = torch.cat(tuple([ref_dat['img'].cuda() for ref_dat in Ref_Dats]), dim=0)
    src_frames_list =  [torch.cat(tuple([src_dat_frame['img'].cuda() \
                        for src_dat_frame in src_dats_traj]), dim=0).unsqueeze(0) \
                        for src_dats_traj in Src_Dats] 
    src_frames = torch.cat(tuple(src_frames_list), dim=0) 

    with torch.no_grad():
        dmap_cur_refined, dmap_refined, d_dpv, kv_dpv  = \
                model_KV( ref_frame = ref_frame, src_frames = src_frames,
                          src_cam_poses = Src_CamPoses, BatchIdx = BatchIdx_range,
                          cam_intrinsics = Cam_Intrinsics, BV_predict = BV_predict) 

    if BV_predict is None: # if the first frame in the sequence
        kv_dpv = d_dpv
        dmap_refined = dmap_cur_refined

    # BV_predict estimation (3D re-sampling) #
    BVs_predict = [] 
    for ibatch in range(d_dpv.shape[0]):
        if cam_pose_next is None:
            rel_Rt = Src_CamPoses[ibatch, t_win_r, :, :].inverse()
        else:
            rel_Rt = cam_pose_next.inverse()

        BV_predict = warp_homo.resample_vol_cuda(src_vol = kv_dpv[ibatch, ...].unsqueeze(0), 
                                                 rel_extM = rel_Rt,
                                                 cam_intrinsic = Cam_Intrinsics[ibatch], 
                                                 d_candi = d_candi, 
                                                 padding_value = math.log(1. / float(len(d_candi))) \
                                                 ).clamp(max=0, min=-1000.).unsqueeze(0) 
        BVs_predict.append(BV_predict) 

    BVs_predict = torch.cat(BVs_predict, dim=0) 

    if R_net :
        return dmap_refined, BVs_predict
    else:
        return kv_dpv, BVs_predict 
