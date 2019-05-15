'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# Numerically optmizing the poses #

import math
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import warping.homography as warp_homo
import mutils.misc as m_misc


def _normalize_img(img):
    img_ = img
    img_out = (img_ - img_.min()) / (img_.max() - img_.min())
    return img_out.detach().cpu().squeeze().numpy().transpose([1,2,0])

def _opt_pose_warping(
        imgs_ref, dmaps_ref, 
        imgs_src, 
        R_init, t_init, cams_intrinsic,
        max_iter = 100, LR = 1e-2, 
        opt_vars = [1 ,1],
        dmap_src = None, 
        bi_direct_warp = False, 
        conf_maps_ref = None, 
        r_para = 'unit_quat'):

    r'''
    Optimize t using regular warping, given R, dmap 

    Inputs:

    imgs_ref, imgs_src - list of NCHW format images. in multiple scales ref - keyframe, src - frame to warp from.  Usually img_ref and img_src are the raw input size for example [384, 256]

    conf_maps_ref - list of the confidence maps from dpv, should be of NCHW format, same size as img_ref 

    cams_intrin - list of cam_intrinsics in multiple scales

    opt_vars - [if_opt_R, if_opt_t], default : [1,1], ie. optimize both R and t If if_opt_R is True, then the input R_init should be quaternion, either log or unit_quaternion, t_init - 3 ele (log quaternion for rotation) and 3 ele tensors.  From ref to src.  So R_ref = I, R_src = R 

    '''

    if_opt_R = opt_vars[0]==1
    nscale = len(imgs_ref)

    if bi_direct_warp:
        assert dmap_src is not None, 'dmap_src is undefined' 

    if if_opt_R is False:
#        print('ONLY OPTIMIZE dR')
        opt_t = Variable(t_init.cuda(), requires_grad=True)
        optimizer = optim.Adam([opt_t], lr = LR , betas= (.9, .999 ))

        assert r_para == 'unit_quat'
        R_ = m_misc.UnitQ2Rotation(R_init.cuda())

    else: # also optimize the rotation
        # optimize R and t together (6 variables)
        assert R_init.dim() == 1 and (len(R_init)==3 or len(R_init) == 4), 'R_init should be of dim-1 3-ele/4-ele vector' 
        opt_R = Variable(R_init.cuda(), requires_grad=True) 
        opt_t = Variable(t_init.cuda(), requires_grad=True)

        if opt_vars[0]==1 and opt_vars[1] == 0: # opt_R only 
            optimizer= optim.Adam([opt_R], lr = LR , betas= (.9, .999) )
        elif opt_vars[1] ==1 and opt_vars[0] == 0: # opt_t only
            optimizer = optim.Adam([opt_t], lr = LR , betas= (.9, .999) )
        elif opt_vars[1] ==1 and opt_vars[0] == 1: # opt_R and opt_t
            optimizer = optim.Adam([opt_t, opt_R], lr = LR , betas= (.9, .999) ) 
        else:
            raise Exception('undefined optmization variable option')

    loss_fn = nn.L1Loss()  # L1 loss works better than L2 loss

    # multi-scale #
    for iscale in range(nscale):

        img_ref = imgs_ref[iscale]
        dmap_ref = dmaps_ref[iscale]
        img_src = imgs_src[iscale]
        conf_map_ref = conf_maps_ref[iscale]
        cam_intrinsic = cams_intrinsic[iscale]

        if iscale>0:
            # change step #
            for para_group in optimizer.param_groups:
                para_group['lr'] = LR / (2**iscale)

        for it in range(max_iter): 
            if if_opt_R: 
                R_= torch.zeros(3,3, requires_grad= False).cuda() 

                if len(opt_R) ==3 and r_para is 'log_quat': # input R in quaternion format
                    m_misc.LogQ2Quaternion(opt_R, R_quat_)
                    m_misc.quaternion2Rotation(R_quat_, is_tensor = True, R_tensor = R_) 

                elif len(opt_R)==3 and r_para is 'unit_quat':
                    R_quat_ = torch.zeros(4, requires_grad=False).cuda()
                    m_misc.unitQ_to_quat(opt_R, R_quat_)
                    m_misc.quaternion2Rotation(R_quat_, is_tensor = True, R_tensor = R_) 

                    if bi_direct_warp:
                        R_quat_inv = torch.zeros(4, requires_grad=False).cuda()
                        R_inv_ = torch.zeros(3,3, requires_grad= False).cuda()
                        m_misc.unitQ_to_quat_inv(opt_R, R_quat_inv)
                        m_misc.quaternion2Rotation(R_quat_inv , is_tensor = True, R_tensor = R_inv_) 

                elif len(opt_R)==4 and r_para is 'raw_quat': # input R in log_quaternion format
                    m_misc.quaternion2Rotation(opt_R, is_tensor=True, R_tensor=R_) 

                else:
                    raise Exception('undefined format for R !') 

            optimizer.zero_grad() 

            # Getting the loss : warping from src to ref #
            src_img_warped = warp_homo.back_warp_th_Rt(img_src, dmap_ref, R_, opt_t, cam_intrinsic) 
            mask_img = Variable(1.0-( src_img_warped==0 ).type_as(src_img_warped), requires_grad=False)

            if conf_map_ref is None:
                loss = loss_fn(src_img_warped * mask_img , img_ref.cuda() * mask_img) 
            else:
                loss = loss_fn(src_img_warped * mask_img * conf_map_ref, img_ref.cuda() * mask_img * conf_map_ref ) 


            # For showing the the warped images #
            ref_img = _normalize_img(img_ref)

            # if needed, warp from ref to src #
            if bi_direct_warp:
                ref_img_warped = warp_homo.back_warp_th_Rt(img_ref, dmap_src, R_inv_, -opt_t, cam_intrinsic) 
                mask_img_src = Variable(1.0-(ref_img_warped==0).type_as(ref_img_warped), requires_grad=False)
                loss = loss + loss_fn(ref_img_warped* mask_img_src , img_src.cuda() * mask_img_src ) 

            # Print the losses for the initial values #
            if it ==0 :
                if iscale ==2:
                    with torch.no_grad():
                        src_img_warped_0 = warp_homo.back_warp_th_Rt(img_src, dmap_ref, R_, opt_t, cam_intrinsic) 
                loss_0 = loss.data.cpu().numpy()

            if it > max_iter-2:
                if iscale ==2:
                    with torch.no_grad():
                        src_img_warped_1 = warp_homo.back_warp_th_Rt(img_src, dmap_ref, R_ , opt_t, cam_intrinsic) 
                print('opt_pose(): scale=%d, iter %d/%d, d_loss = %f'%(iscale, max_iter, max_iter, 
                    ( loss.data.cpu().numpy() - loss_0 )*100 ) ) 

            loss.backward()  
            optimizer.step()
            # END: one iteration for dt, dr # 

    # warped images before/after optimization #
    src_imgs_warps = []

    if if_opt_R is False:
        return opt_t.detach_(), R_init,  src_imgs_warps, ref_img

    else:
        return opt_t.detach_(), opt_R.detach_(), src_imgs_warps, ref_img,

def _opt_pose_warping_parallel(
        imgs_ref, dmaps_ref, 
        imgs_src, 
        R_init, t_init, cams_intrinsic,
        max_iter = 100, LR = 1e-2, 
        opt_vars = [1 ,1],
        dmap_src = None, 
        bi_direct_warp = False, 
        conf_maps_ref = None, 
        r_para = 'unit_quat'):

    r'''
    Optimize t using regular warping, given R, dmap 

    Inputs:

    imgs_ref, imgs_src - list of NCHW format images. in multiple scales ref - keyframe, src - frame to warp from.  Usually img_ref and img_src are the raw input size for example [384, 256]

    conf_maps_ref - list of the confidence maps from dpv, should be of NCHW format, same size as img_ref 

    cams_intrin - list of cam_intrinsics in multiple scales

    opt_vars - [if_opt_R, if_opt_t], default : [1,1], ie. optimize both R and t If if_opt_R is True, then the input R_init should be quaternion, either log or unit_quaternion, t_init - 3 ele (log quaternion for rotation) and 3 ele tensors.  From ref to src.  So R_ref = I, R_src = R 

    '''

    assert imgs_src[0].shape[0] > 1 # should have more than one src view
    assert r_para == 'unit_quat'

    if_opt_R = opt_vars[0]==1
    nscale = len(imgs_ref)
    n_view = imgs_src[0].shape[0]

    if bi_direct_warp:
        assert dmap_src is not None, 'dmap_src is undefined' 

    if if_opt_R is False:
        opt_t = Variable(t_init.cuda(), requires_grad=True)
        optimizer = optim.Adam([opt_t], lr = LR , betas= (.9, .999 ))

        R_ = torch.zeros( n_view, 3, 3, requires_grad=False).cuda()
        for iview in range(n_view):
            R_[iview, ...] = m_misc.UnitQ2Rotation(R_init[iview, ...].cuda())

    else: # also optimize the rotation
        # optimize R and t together (6 variables)
        opt_R = Variable(R_init.cuda(), requires_grad=True) 
        opt_t = Variable(t_init.cuda(), requires_grad=True)
        if opt_vars[0]==1 and opt_vars[1] == 0: # opt_R only 
            optimizer= optim.Adam([opt_R], lr = LR , betas= (.9, .999) )
        elif opt_vars[1] ==1 and opt_vars[0] == 0: # opt_t only
            optimizer = optim.Adam([opt_t], lr = LR , betas= (.9, .999) )
        elif opt_vars[1] ==1 and opt_vars[0] == 1: # opt_R and opt_t
            optimizer = optim.Adam([opt_t, opt_R], lr = LR , betas= (.9, .999) ) 
        else:
            raise Exception('undefined optmization variable option')

    loss_fn = nn.L1Loss()  # L1 loss works better than L2 loss

    # multi-scale #
    for iscale in range(nscale):

        img_ref = imgs_ref[iscale]
        dmap_ref = dmaps_ref[iscale]
        img_src = imgs_src[iscale]
        conf_map_ref = conf_maps_ref[iscale]
        cam_intrinsic = cams_intrinsic[iscale]

        if iscale>0:
            # change step #
            for para_group in optimizer.param_groups:
                para_group['lr'] = LR / (2**iscale)

        for it in range(max_iter): 
            if if_opt_R: 
                R_ = torch.zeros(n_view, 3,3, requires_grad= False).cuda() 
                if opt_R.shape[1]==3 and r_para is 'unit_quat':
                    for iview in range(n_view):
                        R_[iview, ...] = m_misc.UnitQ2Rotation(opt_R[iview, ...].cuda())
                else:
                    raise Exception('undefined format for R !') 

            optimizer.zero_grad() 

            # Getting the loss : warping from src to ref # 
            src_img_warped = warp_homo.back_warp_th_Rt_msrc(img_src, dmap_ref, R_, opt_t, cam_intrinsic) 
            mask_img = Variable(1.0-( src_img_warped==0 ).type_as(src_img_warped), requires_grad=False) 

            if conf_map_ref is None:
                loss = loss_fn( src_img_warped * mask_img , img_ref.cuda() * mask_img) 
            else:
                loss = loss_fn( 
                src_img_warped * mask_img * conf_map_ref.unsqueeze(0).unsqueeze(0).expand(n_view,3,-1,-1), 
                img_ref.cuda() * mask_img * conf_map_ref.unsqueeze(0).unsqueeze(0).expand(n_view,3,-1,-1) 
                ) 

            # For showing the the warped images #
            ref_img = _normalize_img(img_ref)

            # if needed, warp from ref to src #
            if bi_direct_warp:
                raise Exception('not implemented')

            # Print the losses for the initial values #
            if it ==0 :
                if iscale ==2:
                    with torch.no_grad():
                        src_img_warped_0 \
                                = warp_homo.back_warp_th_Rt_msrc(img_src, dmap_ref, R_, opt_t, cam_intrinsic) 

                loss_0 = loss.data.cpu().numpy()

            if it > max_iter-2:
                if iscale ==2:
                    with torch.no_grad():
                        src_img_warped_1 \
                                = warp_homo.back_warp_th_Rt_msrc(img_src, dmap_ref, R_ , opt_t, cam_intrinsic) 

                print('opt_pose(): scale=%d, iter %d/%d, d_loss = %f'%(iscale, max_iter, max_iter, 
                    ( loss.data.cpu().numpy() - loss_0 )*100 ) ) 

            loss.backward()  
            optimizer.step()
            # END: one iteration for dt, dr # 

    # warped images before/after optimization #
    src_imgs_warps = []

    if if_opt_R is False:
        return opt_t.detach_(), R_init,  src_imgs_warps, ref_img

    else:
        return opt_t.detach_(), opt_R.detach_(), src_imgs_warps, ref_img,

def local_BA_direct_parallel(
        ref_frame, src_frames, 
        dmap_ref, conf_map_ref, 
        cams_intrin, dw_scales,
        rel_pose_inits, 
        max_iter, step, opt_vars):
    '''
    optimize the poses for src frames at the same time
    '''
    assert isinstance(ref_frame, torch.Tensor) and isinstance(src_frames[0], torch.Tensor) \
            and isinstance(dmap_ref, torch.Tensor) and isinstance( conf_map_ref, torch.Tensor)


    # NUMERICAL GRADIENT #
    imgs_ref = [m_misc.downsample_img(ref_frame, dw_scale).cuda() for dw_scale in dw_scales] 
    dmaps_ref = [m_misc.downsample_img(dmap_ref, dw_scale).cuda().squeeze() for dw_scale in dw_scales ] 
    conf_maps_ref = [m_misc.downsample_img(conf_map_ref, dw_scale).cuda().squeeze() for dw_scale in dw_scales ] 

    
    t_init_ = []
    R_init_ = []
    imgs_src = []

    for src_idx, init_pose in enumerate( rel_pose_inits): 
        init_pose_th = torch.FloatTensor( init_pose ).clone().cuda()
        t_init_.append(init_pose_th[:3,3].unsqueeze(0) )
        R_init_.append(m_misc.Rotation2UnitQ( init_pose_th[:3,:3] ).unsqueeze(0) )

    R_init_ = torch.cat(R_init_, dim=0)
    t_init_ = torch.cat(t_init_, dim=0)

    for iscale in range(len(dw_scales)):
        imgs_src.append( m_misc.downsample_img( torch.cat(src_frames, dim=0), dw_scales[iscale]).cuda() )

    t_opt, R_opt, src_imgs_warps, ref_img = _opt_pose_warping_parallel( 
            imgs_ref, dmaps_ref, imgs_src,  
            R_init = R_init_, t_init = t_init_, cams_intrinsic = cams_intrin,
            conf_maps_ref = conf_maps_ref,
            max_iter = max_iter, LR = step, 
            r_para = 'unit_quat', 
            opt_vars = opt_vars) 
        
    rel_pose_opt = []
    for i_src in range( len( rel_pose_inits )):
        rel_pose = torch.eye(4) 
        rel_pose[:3,3] = t_opt[i_src,:].cpu()
        R_opt_per = m_misc.UnitQ2Rotation(R_opt[i_src,:])
        rel_pose[:3,:3] = R_opt_per.cpu()
        rel_pose_opt.append(rel_pose) 

    return rel_pose_opt

def local_BA_direct( 
        ref_frame, src_frames, 
        dmap_ref, conf_map_ref, 
        cams_intrin, dw_scales,
        rel_pose_inits, 
        max_iter, step, opt_vars):
    '''
    Inputs :
    
    ref_frame       - ref frame, NCHW
    src_frames      - list of src frames, NCHW
    dmap_ref        - dmap for the local ref frame, NCHW
    conf_map_ref    - conf_map for the local ref frame, NCHW
    cams_intrin     - list of camera intrinsics at different scales [cam_intrin_scale0, ], scale0 is the lowest scale
    dw_scale        - the downsampling scale: [4, 2, 1]
    rel_pose_inits  - 4x4 np arrays the intial relative pose (from the src frame to the ref frame),
                      list of poses for [src0, src1, src2, src3]
    opt_vars - [opt_r, opt_t]

    Outputs: 

    rel_pose_opt -  list of poses [src0, src1, src2, src3], each is a 4x4 np array
    
    ''' 
    assert isinstance(ref_frame, torch.Tensor) and isinstance(src_frames[0], torch.Tensor) \
            and isinstance(dmap_ref, torch.Tensor) and isinstance( conf_map_ref, torch.Tensor)


    # NUMERICAL GRADIENT #
    imgs_ref = [m_misc.downsample_img(ref_frame, dw_scale).cuda() for dw_scale in dw_scales] 
    dmaps_ref = [m_misc.downsample_img(dmap_ref, dw_scale).cuda().squeeze() for dw_scale in dw_scales ] 
    conf_maps_ref = [m_misc.downsample_img(conf_map_ref, dw_scale).cuda().squeeze() for dw_scale in dw_scales ] 

    rel_pose_opt = []

    
    for src_idx, init_pose in enumerate( rel_pose_inits): 

        init_pose_th = torch.FloatTensor( init_pose ).clone().cuda() 
        t_init_ = init_pose_th[:3,3]
        R_init_ = m_misc.Rotation2UnitQ( init_pose_th[:3,:3] )
        
        imgs_src = [ m_misc.downsample_img(src_frames[src_idx], dw_scale).cuda() for dw_scale in dw_scales ] 

        t_opt, R_opt, src_imgs_warps, ref_img = _opt_pose_warping( 
                imgs_ref, dmaps_ref, imgs_src,  
                R_init = R_init_, t_init = t_init_, cams_intrinsic = cams_intrin,
                conf_maps_ref = conf_maps_ref,
                max_iter = max_iter, LR = step, 
                r_para = 'unit_quat', 
                opt_vars = opt_vars) 
        
        R_opt = m_misc.UnitQ2Rotation(R_opt)

        rel_pose = torch.eye(4) 
        rel_pose[:3,3] = t_opt.cpu()
        rel_pose[:3,:3] = R_opt.cpu()
        rel_pose_opt.append(rel_pose) 

    return rel_pose_opt
