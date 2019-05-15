'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import numpy as np
import math
import time 

import torch
torch.backends.cudnn.benchmark=True 
import torch.nn.functional as F
from torch.autograd import Variable

import warping.homography as warp_homo
import mutils.misc as m_misc 

def train(
        nGPU, model_KV, optimizer_KV,
        t_win_r, d_candi,
        Ref_Dats, Src_Dats, Src_CamPoses,
        BVs_predict, Cam_Intrinsics,
        refine_dup = False,
        weight_var = .001,
        loss_type = 'NLL',
        mGPU = False,
        Cam_Intrinsics_spatial_up = None,
        return_confmap_up = False):
    r'''

    Perform one single iteration for the training 

    Support multiple GPU traning. 
    To do this we treat each trajector as one batch

    Inputs: 
    model_KV - 
    optimizer_KV - 
    Ref_Dats - list of ref_dat 

    Src_Dats - list of list of src_dat: [ list_src_dats_traj_0, ...]
                    list_src_dats_traj0[iframe] : NCHW

    Src_CamPoses - N x V x 4 x 4, where N: batch size (# of batched traj), V: # of src. views

    BVs_predict - N x D x H_feat x W_feat

    Cam_Intrinsics - list of camera intrinsics for the batched trajectories

    refine_dup - if upsample the depth dimension in the refinement net

    loss_type = {'L1', 'NLL'}
    L1 - we will calculate the mean value from low res. DPV and filter it with DGF to get the L1 loss in high res.; 
         In additional to that, we will also calculate the variance loss
    NLL - we will calulate the NLL loss from the low res. DPV

    Outputs:

    ''' 

    # prepare for the inputs #
    ref_frame = torch.cat(tuple([ref_dat['img'] for ref_dat in Ref_Dats]), dim=0)
    
    src_frames_list = [torch.cat(tuple([src_dat_frame['img'] \
                        for src_dat_frame in src_dats_traj]), dim=0).unsqueeze(0) \
                         for src_dats_traj in Src_Dats]

    src_frames = torch.cat(tuple(src_frames_list), dim=0) 
    optimizer_KV.zero_grad() 

    # If upsample d in the refinement net#
    if refine_dup:
        dup4_candi = np.linspace(0, d_candi.max(), 4 * len(d_candi) )

    # kv-net Forward pass #
    # model_KV supports multiple-gpus # 
    BatchIdx_range = torch.FloatTensor(np.arange(nGPU)) 

    if mGPU:
        IntMs = torch.cat([cam_intrin['intrinsic_M_cuda'].unsqueeze(0) for cam_intrin in Cam_Intrinsics ], dim=0) 
        unit_ray_Ms_2D = torch.cat([cam_intrin['unit_ray_array_2D'].unsqueeze(0) for cam_intrin in Cam_Intrinsics ], dim=0) 

        dmap_cur_refined, dmap_refined, d_dpv, kv_dpv  = model_KV(
                ref_frame = ref_frame.cuda(0), src_frames = src_frames.cuda(0), src_cam_poses = Src_CamPoses.cuda(0), 
                BatchIdx = BatchIdx_range.cuda(0), cam_intrinsics = None, 
                BV_predict = BVs_predict, mGPU = mGPU, IntMs= IntMs.cuda(0), unit_ray_Ms_2D= unit_ray_Ms_2D.cuda(0)) 


    else:
        IntMs = None
        unit_ray_Ms_2D = None 

        dmap_cur_refined, dmap_refined, d_dpv, kv_dpv  = model_KV(
                ref_frame = ref_frame.cuda(0), src_frames = src_frames.cuda(0), src_cam_poses = Src_CamPoses.cuda(0), 
                BatchIdx = BatchIdx_range.cuda(0), cam_intrinsics = Cam_Intrinsics, 
                BV_predict = BVs_predict, mGPU = mGPU, IntMs= IntMs, unit_ray_Ms_2D= unit_ray_Ms_2D) 


    # Get losses # 
    loss = 0.  
    for ibatch in range(d_dpv.shape[0]):
        if loss_type is 'NLL': 
            # nll loss (d-net) # 
            depth_ref = Ref_Dats[ibatch]['dmap'].cuda(kv_dpv.get_device()) 
            if refine_dup:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_up4_imgsize_digit'].cuda( kv_dpv.get_device() )
            else:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize_digit'].cuda( kv_dpv.get_device() ) 

            loss = loss +  F.nll_loss(d_dpv[ibatch,:,:,:].unsqueeze(0), depth_ref, ignore_index=0) 
            loss = loss + F.nll_loss(dmap_cur_refined[ibatch,:,:,:].unsqueeze(0), depth_ref_imgsize, ignore_index=0)

            if BVs_predict is not None:
                if m_misc.valid_dpv(BVs_predict[ibatch, ...]): # refined 
                    loss = loss + F.nll_loss(kv_dpv[ibatch,:,:,:].unsqueeze(0), depth_ref, ignore_index=0) 
                    loss = loss + F.nll_loss(dmap_refined[ibatch, :, :,:].unsqueeze(0), depth_ref_imgsize, ignore_index=0) 

            dmap_kv_lowres = m_misc.depth_val_regression(kv_dpv[0,...].unsqueeze(0), d_candi, BV_log= True) 

        elif loss_type is 'L1':
            if mGPU:
                raise Exception('not implemented for multiple GPUs') 

            # L1 loss #
            depth_ref = Ref_Dats[ibatch]['dmap_imgsize'].cuda().unsqueeze(0) 
            l1_loss_mask = depth_ref > 0.
            l1_loss_mask = l1_loss_mask.type_as(depth_ref) 
            loss_BV_cur_L1 = \
                    F.l1_loss( dmap_cur_refined* l1_loss_mask, depth_ref.cuda().squeeze(1) * l1_loss_mask) 
            
            if m_misc.valid_dpv(BVs_predict[ibatch, ...]):
                loss_KV_L1 = F.l1_loss( dmap_refined * l1_loss_mask, depth_ref.cuda().squeeze(1) * l1_loss_mask) 

            # variance #
            dmap_d_lowres = m_misc.depth_val_regression(d_dpv, d_candi, BV_log=True)
            loss_BV_cur_var = torch.mean(m_misc.depth_var(d_dpv, dmap_d_lowres, d_candi) )

            if m_misc.valid_dpv( BVs_predict[ibatch, ...] ):
                dmap_kv_lowres = m_misc.depth_val_regression(kv_dpv, d_candi, BV_log= True)
                loss_KV_var = torch.mean(m_misc.depth_var(kv_dpv, dmap_kv_lowres, d_candi) )
                loss = loss_BV_cur_L1  + loss_KV_L1 + weight_var * (loss_KV_var+ loss_BV_cur_var)
            else:
                loss = loss_BV_cur_L1 + weight_var * loss_BV_cur_var 
                dmap_kv_lowres = dmap_d_lowres 

    # Backward pass # 
    if mGPU:
        loss = loss / torch.tensor(nGPU.astype(np.float)  ).cuda(loss.get_device())

    loss.backward()
    optimizer_KV.step() 

    # BV_predict estimation (3D re-sampling) #
    d_dpv = d_dpv.detach()
    kv_dpv = kv_dpv.detach() 
    r_dpv = dmap_cur_refined.detach() if dmap_cur_refined is not -1 else dmap_refined.detach()
    BVs_predict_out = [] 

    for ibatch in range(d_dpv.shape[0]):
        rel_Rt = Src_CamPoses[ibatch, t_win_r, :, :].inverse()
        BV_predict = warp_homo.resample_vol_cuda(src_vol = kv_dpv[ibatch, ...].unsqueeze(0), 
                                                 rel_extM = rel_Rt.cuda(kv_dpv.get_device()),
                                                 cam_intrinsic = Cam_Intrinsics[ibatch], 
                                                 d_candi = d_candi, 
                                                 padding_value = math.log(1. / float(len(d_candi))) \
                                                 ).clamp(max=0, min=-1000.).unsqueeze(0) 
        BVs_predict_out.append(BV_predict) 

    BVs_predict_out = torch.cat(BVs_predict_out, dim=0) 

    
    # logging (for single GPU) #
    depth_ref_lowres = Ref_Dats[0]['dmap_raw'].cpu().squeeze().numpy()
    depth_kv_lres_log = dmap_kv_lowres[0,...].detach().cpu().squeeze().numpy() 
    dmap_log_lres = np.hstack([ depth_kv_lres_log, depth_ref_lowres]) 

    if dmap_refined.dim() < 4: # refined depth map 
        depth_kv_hres_log = dmap_refined.detach().cpu().squeeze().numpy()
        depth_ref_highres = depth_ref.detach().cpu().squeeze().numpy()
    else: # refined dpv
        if refine_dup:
            depth_kv_hres_log = m_misc.depth_val_regression(
                    dmap_refined[0,...].unsqueeze(0), 
                    dup4_candi, BV_log=True).detach().cpu().squeeze().numpy()

        else:
            depth_kv_hres_log = m_misc.depth_val_regression(
                    dmap_refined[0,...].unsqueeze(0), 
                    d_candi, BV_log=True).detach().cpu().squeeze().numpy()

        depth_ref_imgsize_raw = Ref_Dats[0]['dmap_imgsize'].squeeze().cpu().numpy()

        dmap_log_hres = np.hstack([ depth_kv_hres_log, depth_ref_imgsize_raw]) 

    if return_confmap_up:
        confmap_up = torch.exp(dmap_refined[0,...].detach())
        confmap_up, _ = torch.max(confmap_up, dim=0) 
        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres, confmap_up.cpu()

    else:
        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres


def _loss_RNet_UNet2D(dpv, ref_dat, d_candi, dpv_statistics, model_RNet, weight_grad=0., if_test = False, loss_type='L1'):
    '''
    inputs:
        dpv - NDHW
        model_RNet - An instance of RefineNet_Unet2D defined in ./models/Refine.py
        loss_type - {'NLL', 'L1'}

        dpv_statistics - list of strings, e.g. ['E_mean', 'variance', ]
        'E_mean' : expected mean
        'variance': variance of depth
        'max' , 'min': min and max of BV_measure, along depth
    '''

    if input_depth.dim() == 2:
        depth_lowres = input_depth.cuda().unsqueeze(0).unsqueeze(0)
    elif input_depth.dim() == 3:
        depth_lowres = input_depth.cuda().unsqueeze(0)
    elif input_depth.dim() ==4:
        depth_lowres = input_depth.cuda()
        

    # Refine the depth map #
    img_ref = ref_dat['img'].cuda()
    dpv_statistics = ['E_mean', 'variance', 'max', 'min']
    dpv_features_lowres = m_misc.dpv_statistics(dpv, d_candi, dpv_statistics)
    dmap_refined = model_RNet(dpv_features_lowres, img_ref)

    # Get the loss function #
    if not if_test:
        if loss_type is 'L1':
            # Loss function- L1 #
            dmap_ref = ref_dat['dmap_imgsize'].cuda().unsqueeze(0)
            loss_mask = (dmap_ref > 0).detach().type_as(dmap_refined)
            loss = F.l1_loss(dmap_refined * loss_mask, dmap_ref * loss_mask) 
        else:
            raise Exception('NLL not defined for RefineNet_UNet2D !')
    else:
        loss = 0. 

    dmap_refined = dmap_refined.detach()

    return loss, dmap_refined

def _loss_RNet_DGF(input_depth, ref_dat, model_RNet,  if_test = False, loss_type='L1'):
    '''
    inputs:
        model_RNet - An instance of RefineNet_DGF defined in ./models/Refine.py
        loss_type - {'NLL', 'L1'}
    '''

    if input_depth.dim() == 2:
        depth_lowres = input_depth.cuda().unsqueeze(0).unsqueeze(0)
    elif input_depth.dim() == 3:
        depth_lowres = input_depth.cuda().unsqueeze(0)
    elif input_depth.dim() ==4:
        depth_lowres = input_depth.cuda()
        

    # Refine the depth map #
    img_ref = ref_dat['img'].cuda()
    dmap_refined = model_RNet(depth_lowres, img_ref)

    # Get the loss function #
    if not if_test:
        if loss_type is 'L1':
            # Loss function- L1 #
            dmap_ref = ref_dat['dmap_imgsize'].cuda().unsqueeze(0)
            loss_mask = (dmap_ref > 0).detach().type_as(dmap_refined)
            loss = F.l1_loss(dmap_refined * loss_mask, dmap_ref * loss_mask) 
        else:
            raise Exception('NLL not defined for RefineNet_DGF !')
    else:
        loss = 0. 

    dmap_refined = dmap_refined.detach()
    return loss, dmap_refined 


def _check_model(model):
    valid_model = True
    for layer in model.parameters():
        if torch.isnan(layer).sum() > 0:
            valid_model= False
            break

    return valid_model
