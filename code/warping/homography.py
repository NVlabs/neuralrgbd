'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''


'''
Homography warping module: 
'''
import numpy as np
from warping import View
import torch.nn.functional as F
import torch

import time
import math

def get_rel_extM_list(abs_extMs, rel_extMs, t_win_r = 2):
    '''
    Get the list of relative camera pose
    input: 

    abs_extMs - the list of absolute camera pose
    if abs_extMs[idx]==-1, then this idx time stamp has invalid pose
    rel_extMs - list of list rel camera poses, 
    rel_extMs[idx] = [zeors] by default, if invalid pose, then =zeros

    '''
    n_abs_extMs = len(abs_extM)
    for idx_extM in range(n_abs_extMs):
        if idx_extM + 1 - t_win_r < 0:
            continue
        elif abs_extMs[idx_extM] ==-1:
            continue
        else:
            pass


def points3D_to_opticalFlow(Points3D, view_ref, view_src):
    '''
    Get the optical flow from ref view to src view
    This function is used for validation for the warping function : back_warp()
    Inputs: 
        Points3D - points in 3D in the world coordiate 
        view_ref , view_src : the camera views including the camera position and lookat
    Outputs: 
        optical_flow - the optical flow from ref view to src view
    '''
    import mdataloader.sceneNet_calculate_optical_flow as sceneNet_opticalFlow
    return sceneNet_opticalFlow.optical_flow( Points3D, view_ref, view_src)

def img_dis_L2(img0, img1):
    assert isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor), 'inputs should be Torch tensors'
    diff_img = torch.sum(((img0 - img1)**2).squeeze(), 0)
    return diff_img 

def img_dis_L2_diffmask(img0, img1):
    '''
    also return the distance in mask, to deal with image boundaries
    NOTE: Assuming the mask image is the first feature channel !
    '''
    assert isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor), 'inputs should be Torch tensors'
    feat_diff_img = torch.sum(((img0[0, 1::, ... ] - img1[0, 1::, ... ])**2).squeeze(), 0)
    mask_diff_img = (img0[0, 0, ...] - img1[0, 0, ...])**2 

    return feat_diff_img, mask_diff_img

def img_dis_L2_mask(img0, img1):
    '''
    also return the warped mask, viewed from img1
    NOTE: Assuming the mask image is the first feature channel !
    '''
    assert isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor), 'inputs should be Torch tensors'
    feat_diff_img = torch.sum(((img0[0, 1::, ... ] - img1[0, 1::, ... ])**2).squeeze(), 0)
    mask_diff_img = (img0[0, 0, ...] - img1[0, 0, ...])**2 

    return feat_diff_img, mask_diff_img, img1[0, 0, ...]

def img_dis_L2_pard(img0, img1):
    diff_img = torch.sum( (img0 - img1)**2, 1)
    return diff_img

def img_dis_L1_pard(img0, img1):
    diff_img = torch.sum( torch.abs(img0 - img1), 1)
    return diff_img

def est_swp_volume( feat_img_ref, feat_img_src, d_candi, R,t, cam_intrinsic, return_list = True):
    '''
    Inputs:

    feat_img_ref, feat_img_src - the feature maps (Tensor: NCHW)

    The feat_img_src could be a list, in the case that there are multiple
    comparison images. In this case, R, t should also be lists 

    d_candi - an array of candidate depth values 

    R,t - R: 3x3, t: 3-ele vector. The relative camera pose from the ref_view to src_view 
    So the extrinsic for the src_view would be [R,t] if the world coordinate center is at the 
    camera center of the reference view.
    If feat_img_src is a list, then R, t should also be lists 

    cam_intrinsic - {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
    hfov, vfov
    fovs in horzontal and vertical directions (degrees)
    unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
    unit ray pointing from the camera center to the pixel

    Outputs
    costVolume - The cost volume for the reference view
    '''
    # source views. 
    assert isinstance(feat_img_ref, torch.Tensor),\
            'Input ref. image should be tensor'

    multiple_src = False
    if isinstance( feat_img_src, list):
        multiple_src = True
        assert isinstance( R, list) and isinstance( t, list), 'Inputs R and t should also be list !'
        assert isinstance( feat_img_src[0], torch.Tensor), 'Input src. images should be tensor list'

    H, W, D = feat_img_ref.shape[2], feat_img_ref.shape[3], len(d_candi)

    if return_list:
        costV = []
    else:
        costV = torch.FloatTensor(1, D, H, W).cuda()

    for idx_d, d in enumerate(d_candi): # for each depth hypothesis #
        dmap = np.ones( (H,W)) * float(d)
        if not multiple_src:
            warped_feat_img_src = back_warp(feat_img_src, dmap, R, t, cam_intrinsic)

            # Get the cost map #
            if return_list:
                costV.append(img_dis_L2( feat_img_ref, warped_feat_img_src))
            else:
                costV[0, idx_d, :, : ] = img_dis_L2(feat_img_ref, warped_feat_img_src)

        else: # Multiple comparison frames #
            costV_map = torch.zeros( feat_img_ref.shape[2], feat_img_ref.shape[3]).cuda()
            # get the feature distances for src. images #
            for cmp_idx, feat_img_cmp in enumerate( feat_img_src):
                R_cmp = R[cmp_idx]
                t_cmp = t[cmp_idx]
                warped_feat_img_cmp = back_warp( feat_img_cmp, dmap, R_cmp, t_cmp, cam_intrinsic)
                costV_map = costV_map + img_dis_L2( feat_img_ref, warped_feat_img_cmp )

            if return_list:
                costV.append( costV_map )
            else:
                costV[0, idx_d, :, : ] = costV_map

    return costV 

def est_swp_volume_v2( feat_img_ref, feat_img_src, d_candi, R,t, cam_intrinsic):
    r'''
    Not used
    '''
    assert isinstance(feat_img_src, list),  'feat_img_src should be a list'
    H, W = feat_img_ref.shape[2], feat_img_ref.shape[3]
    # For different depth #
    costV = [ _get_costV_map(feat_img_ref, \
            feat_img_src, float(d), R, t, cam_intrinsic) for d in d_candi]
    return costV

def _get_costV_map(feat_img_ref, feat_img_src, d, Rs, ts, cam_intrinsic):
    '''
    Input:
    feat_img_src - a list of features for the src. views 
    NOTE USED
    '''
    # get the costMap for a specific depth. To do this, we need to modify back_warp, 
    # given a scalar depth value, we get the warpping coordinates analytically (see the todo
    #in back_warp)#
    costV_maps_src = [ img_dis_L2(feat_img_ref,
        back_warp(feat_img_src[cmp_idx], d, Rs[cmp_idx], ts[cmp_idx], cam_intrinsic))\
                for cmp_idx in range(len(feat_img_src)) ]
    return torch.sum( torch.stack(costV_maps_src, dim=0), dim=0)

def warp_img_feats_mgpu(feat_img_src, d_candi, R,t, IntM_tensors, unit_ray_arrays_2D ):
    r'''
    Warp the feat_imgs_src to the reference view for all candidate depths
    Inputs:
        feat_img_src - list of source image features: each is NCHW format
        d_candi,
        R,t - list of rotation and transitions

    Output: 
    Feat_warped_src
    '''

    IntM_tensor = IntM_tensors.squeeze(0)
    P_ref_cuda = unit_ray_arrays_2D.squeeze(0)

    d_candi_cuda = torch.from_numpy(d_candi.astype(np.float32)).cuda()

    if isinstance(R, list) and isinstance(t, list):
        H, W, D = feat_img_src[0].shape[2], feat_img_src[0].shape[3], len(d_candi)
        Feat_warped_src = []
        for idx_view, feat_img_src_view in enumerate(feat_img_src):
            # Get term1 #
            term1 = IntM_tensor.matmul(t[idx_view]).unsqueeze(1)
            # Get term2 # 
            term2 = IntM_tensor.matmul(R[idx_view]).matmul(P_ref_cuda)
            
            feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1)

            feat_img_src_view_warp_par_d =  _back_warp_homo_parallel_v1(feat_img_src_view_repeat, d_candi_cuda, term1, term2, IntM_tensor, H, W) 

            Feat_warped_src.append( torch.transpose(feat_img_src_view_warp_par_d, dim0=0, dim1=1 ))

    else:
        H, W, D = feat_img_src.shape[2], feat_img_src.shape[3], len(d_candi)
        feat_img_src_view = feat_img_src
        # Get term1 #
#        term1 = IntM_tensor.matmul(t).reshape(3,1)
        term1 = IntM_tensor.matmul(t).unsqueeze(1)
        # Get term2 #
        term2 = IntM_tensor.matmul(R).matmul(P_ref_cuda)
        
        feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1)
        feat_img_src_view_warp_par_d = \
                _back_warp_homo_parallel(feat_img_src_view_repeat,
                                         d_candi_cuda, term1, term2,
                                         cam_intrinsic, H, W)

        Feat_warped_src = torch.transpose(feat_img_src_view_warp_par_d, dim0=0, dim1=1 )

    return Feat_warped_src

def warp_img_feats_v3(feat_img_src, d_candi, R,t, cam_intrinsic):
    r'''
    Warp the feat_imgs_src to the reference view for all candidate depths
    Inputs:
        feat_img_src - list of source image features: each is NCHW format
        d_candi,
        R,t - list of rotation and transitions
        cam_intrinsic 

    Output: 
    Feat_warped_src
    '''

    IntM_tensor = cam_intrinsic['intrinsic_M_cuda'].cuda() # intrinsic matrix 3x3 on GPU
    P_ref_cuda = cam_intrinsic['unit_ray_array_2D'].cuda() # unit ray array in matrix form on GPU
    d_candi_cuda = torch.from_numpy(d_candi.astype(np.float32)).cuda()

    if isinstance(R, list) and isinstance(t, list):
        H, W, D = feat_img_src[0].shape[2], feat_img_src[0].shape[3], len(d_candi)
        Feat_warped_src = []
        for idx_view, feat_img_src_view in enumerate(feat_img_src):
            # Get term1 #
            term1 = IntM_tensor.matmul(t[idx_view]).unsqueeze(1)
            # Get term2 # 
            term2 = IntM_tensor.matmul(R[idx_view]).matmul(P_ref_cuda) 
            feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1) 
            feat_img_src_view_warp_par_d =  _back_warp_homo_parallel(feat_img_src_view_repeat, d_candi_cuda, term1, term2, cam_intrinsic, H, W) 
            Feat_warped_src.append( torch.transpose(feat_img_src_view_warp_par_d, dim0=0, dim1=1 ))

    else:
        H, W, D = feat_img_src.shape[2], feat_img_src.shape[3], len(d_candi)
        feat_img_src_view = feat_img_src

        # Get term1 #
        term1 = IntM_tensor.matmul(t).unsqueeze(1)

        # Get term2 #
        term2 = IntM_tensor.matmul(R).matmul(P_ref_cuda)
        
        feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1)
        feat_img_src_view_warp_par_d = _back_warp_homo_parallel(feat_img_src_view_repeat,
                                                                d_candi_cuda, term1, term2,
                                                                cam_intrinsic, H, W)

        Feat_warped_src = torch.transpose(feat_img_src_view_warp_par_d, dim0=0, dim1=1 )

    return Feat_warped_src


def debug_writeVolume(vol, vmin, vmax):
    '''
    vol : D x H x W
    '''
    import matplotlib.pyplot as plt
    for idx in range(vol.shape[0]):
        slice_img = vol[idx,:, :]
        plt.imsave('vol_%03d.png'%(idx), 
                   slice_img, vmin=vmin, vmax=vmax)

def est_swp_volume_v4(feat_img_ref, feat_img_src, 
                      d_candi, R,t, cam_intrinsic,
                      costV_sigma,  
                      feat_dist = 'L2',
                      debug_ipdb = False):
    r'''
    feat_img_ref - NCHW tensor
    feat_img_src - NVCHW tensor.  V is for different views
    R, t - R[idx_view, :, :] - 3x3 rotation matrix
           t[idx_view, :] - 3x1 transition vector
    '''

    H, W, D = feat_img_ref.shape[2], feat_img_ref.shape[3], len(d_candi)
    costV = torch.zeros(1, D, H, W).cuda()


    IntM_tensor = cam_intrinsic['intrinsic_M_cuda'].cuda( torch.cuda.current_device() ) # intrinsic matrix 3x3 on GPU
    P_ref_cuda = cam_intrinsic['unit_ray_array_2D'].cuda( torch.cuda.current_device() ) # unit ray array in matrix form on GPU
    d_candi_cuda = torch.from_numpy(d_candi.astype(np.float32)).cuda(torch.cuda.current_device())

    for idx_view in range(feat_img_src.shape[1]):
        # Get term1 #
        term1 = IntM_tensor.matmul(t[idx_view, :]).reshape(3,1)
        # Get term2 #
        term2 = IntM_tensor.matmul(R[idx_view, :, :]).matmul(P_ref_cuda)
        feat_img_src_view = feat_img_src[:, idx_view, :, :, :]
        feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1)

        feat_img_src_view_warp_par_d = \
        _back_warp_homo_parallel(feat_img_src_view_repeat, d_candi_cuda, term1, term2, cam_intrinsic, H, W)

        if feat_dist == 'L2':
            costV[0, :, :, :] = costV[0,:,:,:] + img_dis_L2_pard(feat_img_src_view_warp_par_d, feat_img_ref) / costV_sigma
        elif feat_dist == 'L1':
            costV[0, :, :, :] = costV[0,:,:,:] + img_dis_L1_pard(feat_img_src_view_warp_par_d, feat_img_ref) / costV_sigma
        else:
            raise Exception('undefined metric for feature distance ...')

    return costV
    

def est_swp_volume_v3(feat_img_ref, feat_img_src, d_candi, R,t, cam_intrinsic,
                      costV_sigma, if_par_d = False, debug_ipdb = False):
    r'''
    Faster version of est_swp_volume()
    feat_img_ref, feat_img_src - tensor arrays
    R[i_src], t[i_src] - tensor arrays 

    inputs:
    ++
    cam_intrinsic -cam_intrinsic['intrinsic_M_cuda'] # intrinsic matrix 3x3 on GPU
    cam_intrinsic['unit_ray_array_2D'] # unit ray array in matrix form on GPU
    if_par_d (optinal; False) - if parallelize d 

    '''

    H, W, D = feat_img_ref.shape[2], feat_img_ref.shape[3], len(d_candi)
    costV = torch.zeros(1, D, H, W).cuda()

    if debug_ipdb:
        maskV = torch.zeros(1,D,H,W).cuda()

    IntM_tensor = cam_intrinsic['intrinsic_M_cuda'].cuda( t[0].get_device() ) # intrinsic matrix 3x3 on GPU
    P_ref_cuda = cam_intrinsic['unit_ray_array_2D'].cuda( R[0].get_device()) # unit ray array in matrix form on GPU

    if if_par_d:
        d_candi_cuda = torch.from_numpy(d_candi.astype(np.float32)).cuda()

    for idx_view, feat_img_src_view in enumerate(feat_img_src):
        # Get term1 #
        term1 = IntM_tensor.matmul(t[idx_view].reshape(3,1))

        # Get term2 #
        term2 = IntM_tensor.matmul(R[idx_view]).matmul(P_ref_cuda)
        if (not if_par_d) or debug_ipdb:
            for idx_d, d in enumerate(d_candi):
                # For one specific depth, get the costMap #
                feat_img_src_view_warp = _back_warp_homo(feat_img_src_view, d,
                                                         term1, term2,
                                                         cam_intrinsic, H, W)
                # Get costV #
                feat_diff, mask_diff, mask = img_dis_L2_mask(feat_img_ref, feat_img_src_view_warp)

                mask = (mask.detach() >= .99).type_as(costV)
                costV[0, idx_d, :, :] = costV[0, idx_d, :, :] + feat_diff / costV_sigma + mask_diff * 1.
                maskV[0, idx_d, :, :] += mask

        else: #paralell in d_candi
            feat_img_src_view_repeat = feat_img_src_view.repeat(len(d_candi), 1, 1, 1)
            feat_img_src_view_warp_par_d = _back_warp_homo_parallel(feat_img_src_view_repeat,
                                         d_candi_cuda,term1, term2,
                                         cam_intrinsic, H, W)

            costV[0, :, :, :] = costV[0,:,:,:] + img_dis_L2_pard(feat_img_src_view_warp_par_d, 
                                                                 feat_img_ref) / costV_sigma


    return costV

def _back_warp_homo_parallel_v1(img_src, D, term1, term2, intrin_M, H, W, debug_inputs = None ):
    r'''
    Do the warpping for the src. view analytically using homography, given the
    depth d for the reference view: 
    p_src ~ term1  + term2 * d

    inputs:
    term1, term2 - 3 x n_pix matrix 
    P_ref - The 2D matrix form for the unit_array for the camera
    D - candidate depths. A tensor array on GPU

    img_src_warpped - warpped src. image 
    '''
    n_d = len(D)
    term2_cp = term2.repeat(n_d, 1, 1)

    P_src = term1.unsqueeze(0) + term2_cp * D.reshape(n_d,1,1)
    P_src = P_src / (P_src[:, 2, :].unsqueeze(1)  + 1e-10 ) 

    src_coords = torch.FloatTensor(n_d, H, W, 2).cuda()

    src_coords[:,:,:,0] = P_src[:, 0, :].reshape(n_d, H, W)
    src_coords[:,:,:,1] = P_src[:, 1, :].reshape(n_d, H, W)
    u_center, v_center  = intrin_M[0,2], intrin_M[1,2]
    src_coords[:,:,:,0] = (src_coords[:,:,:,0] - u_center) / u_center
    src_coords[:,:,:,1] = (src_coords[:,:,:,1] - v_center) / v_center 
    img_src_warpped = F.grid_sample(img_src, src_coords,mode='bilinear', padding_mode='zeros') 
    return img_src_warpped

def _back_warp_homo_parallel(img_src, D, term1, term2, cam_intrinsics, H, W, debug_inputs = None ):
    r'''
    Do the warpping for the src. view analytically using homography, given the
    depth d for the reference view: 
    p_src ~ term1  + term2 * d

    inputs:
    term1, term2 - 3 x n_pix matrix 
    P_ref - The 2D matrix form for the unit_array for the camera
    D - candidate depths. A tensor array on GPU

    img_src_warpped - warpped src. image 
    '''
    n_d = len(D)
    term2_cp = term2.repeat(n_d, 1, 1)

    P_src = term1.unsqueeze(0) + term2_cp * D.reshape(n_d,1,1)
    P_src = P_src / (P_src[:, 2, :].unsqueeze(1)  + 1e-10 ) 

    src_coords = torch.FloatTensor(n_d, H, W, 2).cuda()

    src_coords[:,:,:,0] = P_src[:, 0, :].reshape(n_d, H, W)
    src_coords[:,:,:,1] = P_src[:, 1, :].reshape(n_d, H, W)
    u_center, v_center = cam_intrinsics['intrinsic_M'][0,2], cam_intrinsics['intrinsic_M'][1,2]
    src_coords[:,:,:,0] = (src_coords[:,:,:,0] - u_center) / u_center
    src_coords[:,:,:,1] = (src_coords[:,:,:,1] - v_center) / v_center 
    img_src_warpped = F.grid_sample(img_src, src_coords,mode='bilinear', padding_mode='zeros') 
    return img_src_warpped

def _back_warp_homo(img_src, d, term1, term2, cam_intrinsics, H, W,
         debug_inputs = None ):
    r'''
    Do the warpping for the src. view analytically using homography, given the
    depth d for the reference view: 
    p_src ~ term1  + term2 * d

    inputs:
    term1, term2 - 3 x n_pix matrix 
    P_ref - The 2D matrix form for the unit_array for the camera
    d - candidate depth

    img_src_warpped - warpped src. image 
    '''
    P_src = term1 + term2 * d
    P_src = P_src / P_src[2,:]
    u_center, v_center = cam_intrinsics['intrinsic_M'][0,2], cam_intrinsics['intrinsic_M'][1,2]
    u_coords, v_coords = P_src[0,:], P_src[1, :]

    u_coords = (u_coords - u_center) / u_center # to range [-1, 1]
    v_coords = (v_coords - v_center) / v_center

    u_coords = torch.reshape(u_coords, [H, W]) 
    v_coords = torch.reshape(v_coords, [H, W])
    src_coords = torch.stack((u_coords, v_coords), dim=2).unsqueeze(0)
    img_src_warpped = F.grid_sample( img_src, src_coords)

    return img_src_warpped

def back_warp_th_Rt_msrc(imgs_src, dmap, Rs,ts, cam_intrinsic):
    '''

    imgs_src - NCHW multiple src frames 
    Rs, ts - Rs[iview, ... ], ts[iview, ...] rotation/translation from ref to src view
    
    Given the depth map ( 2D torch tensor), the camera poses (R,t, as torch.tensor) warp the src. image


    '''

    assert isinstance(imgs_src, torch.Tensor)

    u_center, v_center = cam_intrinsic['intrinsic_M_cuda'][0,2].cuda(),\
            cam_intrinsic['intrinsic_M_cuda'][1,2].cuda()

    npts = dmap.numel()
    assert cam_intrinsic['unit_ray_array_2D'].shape[1] == npts 
    N, C, H, W = imgs_src.shape[0], imgs_src.shape[1], imgs_src.shape[2], imgs_src.shape[3] 
    SRCS_coords = torch.zeros(N, H, W, 2).cuda()

    # Back project to 3D space (in 2d matrix form) #
    Points_ref_cam_coord = \
    dmap.type_as(cam_intrinsic['unit_ray_array_2D']).view(1, npts).cuda() \
        *cam_intrinsic['unit_ray_array_2D'].cuda()

    Points_ref_cam_coord = torch.cat((Points_ref_cam_coord, torch.ones((1, npts)).cuda()), dim = 0)

    for iview in range(N):
        # Project to the src_view to get the image coordinates in the src_view #
        src_cam_extM = torch.eye(4).cuda()
        src_cam_extM[:3, :3] = Rs[iview, ...]
        src_cam_extM[:3, 3] =  ts[iview, ...]
        src_cam_intM = torch.zeros([4,4]).cuda()
        src_cam_intM[:3, :3] = cam_intrinsic['intrinsic_M_cuda'].cuda()
        Points_src_cam_coord = src_cam_intM.matmul( src_cam_extM.matmul( Points_ref_cam_coord ) )
        Points_src_cam_coord = Points_src_cam_coord / Points_src_cam_coord[2, :].view(1,npts)

        # Interpolate # 
        u_coords, v_coords = Points_src_cam_coord[0, :], Points_src_cam_coord[1,:]
        u_coords = u_coords.view(H, W)
        v_coords = v_coords.view(H, W)
        u_coords = (u_coords - u_center) / u_center 
        v_coords = (v_coords - v_center) / v_center
        uv_coords = torch.cat( (u_coords.unsqueeze(2), v_coords.unsqueeze(2)) , dim=2)
        src_coords = uv_coords.cuda()
        SRCS_coords[iview, ...] = src_coords

    img_src_warpped = F.grid_sample( imgs_src.cuda(), SRCS_coords)
    return img_src_warpped

def back_warp_th_Rt(img_src, dmap, R,t, cam_intrinsic):
    '''
    img_src - NCHW
    Given the depth map ( 2D torch tensor), the camera poses (R,t, as torch.tensor)
    warp the src. image
    R, t - rotation/translation from ref to src view
    '''
    assert isinstance(R, torch.Tensor) and isinstance(t, torch.Tensor),\
        'R,t should be torch tensors'

    H, W = img_src.shape[2], img_src.shape[3] 

    u_center, v_center = cam_intrinsic['intrinsic_M_cuda'][0,2].cuda(),  cam_intrinsic['intrinsic_M_cuda'][1,2].cuda()
    npts = dmap.numel()
    assert cam_intrinsic['unit_ray_array_2D'].shape[1] == npts

    # Back project to 3D space (in 2d matrix form) #
    Points_ref_cam_coord = \
    dmap.type_as(cam_intrinsic['unit_ray_array_2D']).view(1, npts).cuda() \
        *cam_intrinsic['unit_ray_array_2D'].cuda()

    Points_ref_cam_coord = \
        torch.cat((Points_ref_cam_coord, torch.ones((1, npts)).cuda()), dim = 0)

    # Project to the src_view to get the image coordinates in the src_view #
    src_cam_extM = torch.eye(4).cuda()
    src_cam_extM[:3, :3] = R
    src_cam_extM[:3, 3] = t
    src_cam_intM = torch.zeros([4,4]).cuda()
    src_cam_intM[:3, :3] = cam_intrinsic['intrinsic_M_cuda'].cuda()
    Points_src_cam_coord = src_cam_intM.matmul( src_cam_extM.matmul( Points_ref_cam_coord ) ) 
    Points_src_cam_coord = Points_src_cam_coord / Points_src_cam_coord[2, :].view(1,npts)

    # Interpolate #
    u_coords, v_coords = Points_src_cam_coord[0, :], Points_src_cam_coord[1,:]
    u_coords = u_coords.view(H, W)
    v_coords = v_coords.view(H, W)
    u_coords = (u_coords - u_center) / u_center 
    v_coords = (v_coords - v_center) / v_center
    uv_coords = torch.cat( (u_coords.unsqueeze_(2), v_coords.unsqueeze_(2)) , dim=2)
    src_coords = uv_coords.cuda().unsqueeze_(0)
    img_src_warpped = F.grid_sample( img_src.cuda(), src_coords)


    return img_src_warpped

def back_warp(img_src, dmap, R,t, cam_intrinsic, debug_inputs = None):
    '''
    Given the src image, dmap (ray distance map) and relative camera poses, warp the source image
    inputs:
        img_src - image to warp from. A numpy nd array (HxWxnch) or torch.Tensor: NCHW 

        dmap - height x width array.  Ray distance map for the reference view

        R,t - R: 3x3, t: 3-ele vector. The relative camera pose from the ref_view to src_view 
        So the extrinsic for the src_view would be [R,t] if the world coordinate center is at the 
        camera center of the reference view.

        cam_intrinsic - {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
        hfov, vfov
        fovs in horzontal and vertical directions (degrees)
        unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
        unit ray pointing from the camera center to the pixel

        debug_inputs - {'view_ref', 'view_src'}

    outputs:
        img_src_warpped - the warpped source image
    '''


    if isinstance(img_src, np.ndarray):
        assert cam_intrinsic['intrinsic_M'].shape[0] ==3 and cam_intrinsic['intrinsic_M'].shape[1] ==4,\
                'The camera intrinsic matrix should be of size 3x4'
        assert img_src.shape[0] == dmap.shape[0] and img_src.shape[1] == dmap.shape[1], \
                'img_src and dmap should be of the same spatial dimensions '
        H, W = img_src.shape[0], img_src.shape[1]
        img_src_th = torch.FloatTensor(np.transpose(img_src, axes=[2, 0, 1])).unsqueeze_(0).cuda()

    elif isinstance(img_src, torch.Tensor):
        assert img_src.is_cuda, 'input img_src should be on GPU' 
        # NCHW #
        assert img_src.shape[2] == dmap.shape[0] and img_src.shape[3] == dmap.shape[1], \
                'img_src and dmap should be of the same spatial dimensions '
        H, W = img_src.shape[2], img_src.shape[3]
        img_src_th = img_src
    else:
        raise Exception('Input img_src should be either numpy ndarray or torch.Tensor')

    u_center, v_center = cam_intrinsic['intrinsic_M'][0,2], cam_intrinsic['intrinsic_M'][1,2]

    # Back project to 3D space #
    Points_ref_cam_coord = np.repeat( np.expand_dims(dmap, axis=2), 3, axis=2) \
            * cam_intrinsic['unit_ray_array']

    # Reshape to the matrix form #
    Points_ref_cam_coord = np.reshape(np.transpose( Points_ref_cam_coord, axes= [2,0,1] ), [3, -1])
    Points_ref_cam_coord = np.vstack((Points_ref_cam_coord,
        np.ones((1, Points_ref_cam_coord.shape[1]))) )

    # Project to the src_view to get the image coordinates in the src_view #
    src_cam_extM = np.eye(4)
    src_cam_extM[:3, :3] = R
    src_cam_extM[:3, 3] = t
    src_cam_intM = cam_intrinsic['intrinsic_M']
    Points_src_cam_coord = src_cam_intM.dot( src_cam_extM.dot( Points_ref_cam_coord ) )
    Points_src_cam_coord = Points_src_cam_coord / \
            np.repeat( np.expand_dims(Points_src_cam_coord[2, :], axis=0),3, axis=0 )
    # Interpolate #
    u_coords, v_coords = Points_src_cam_coord[0, :], Points_src_cam_coord[1,:]
    u_coords = np.reshape(u_coords, [H, W]) 
    v_coords = np.reshape(v_coords, [H, W])
    u_coords = (u_coords - u_center) / u_center # to range [-1, 1]
    v_coords = (v_coords - v_center) / v_center
    uv_coords = np.dstack(( u_coords, v_coords))
    src_coords = torch.FloatTensor( uv_coords).cuda().unsqueeze_(0)

    img_src_warpped = F.grid_sample( img_src_th, src_coords)
    if isinstance( img_src, np.ndarray):
        img_src_warpped = torch.squeeze( img_src_warpped, dim = 0).cpu().numpy()
        return np.transpose(img_src_warpped, axes = [1, 2, 0])
    else: 
        return img_src_warpped

def resample_vol_cuda(src_vol, rel_extM, cam_intrinsic = None, 
                      d_candi = None, d_candi_new = None,
                      padding_value = 0., output_tensor = False, 
                      is_debug = False, PointsDs_ref_cam_coord_in = None):
    r'''

    if d_candi_new is not None: 
    d_candi : candidate depth values for the src view;
    d_candi_new : candidate depth values for the ref view. Usually d_candi_new is different from d_candi
    '''
    assert d_candi is not None, 'd_candi should be some np.array object'

    N, D, H, W = src_vol.shape
    N = 1
    hhfov, hvfov = \
            math.radians(cam_intrinsic['hfov']) * .5, math.radians(cam_intrinsic['vfov']) * .5

    # --- 0. Get the sampled points in the ref. view --- #
    if PointsDs_ref_cam_coord_in is None:
        PointsDs_ref_cam_coord = torch.zeros(N, D, H, W, 3)
        if d_candi_new is not None:
            d_candi_ = d_candi_new
        else:
            d_candi_ = d_candi

        for idx_d, d in enumerate(d_candi_):
            PointsDs_ref_cam_coord[0, idx_d, :, :, :] \
                    = d * torch.FloatTensor(cam_intrinsic['unit_ray_array'])
        PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.cuda(0)
    else:
        PointsDs_ref_cam_coord = PointsDs_ref_cam_coord_in

    if d_candi_new is not None:
        z_max, z_min = d_candi.max(), d_candi.min()
    else:
        z_max = torch.max(PointsDs_ref_cam_coord[0, :, :, :, 2 ])
        z_min = torch.min(PointsDs_ref_cam_coord[0, :, :, :, 2 ])

    z_half = (z_max + z_min) * .5
    z_radius = (z_max - z_min) * .5

    # --- 1. Coordinate transform --- # 
    PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.reshape((-1, 3)).transpose(0,1)
    PointsDs_ref_cam_coord = torch.cat((PointsDs_ref_cam_coord, torch.ones(1, PointsDs_ref_cam_coord.shape[1]).cuda(0) ), dim=0)

    src_cam_extM = rel_extM
    PointsDs_src_cam_coord = src_cam_extM.matmul( PointsDs_ref_cam_coord )

    # transform into range [-1, 1] for all dimensions #
    PointsDs_src_cam_coord[0, :] = PointsDs_src_cam_coord[0,:] / (PointsDs_src_cam_coord[2,:] +1e-10) / math.tan( hhfov)
    PointsDs_src_cam_coord[1, :] = PointsDs_src_cam_coord[1,:] / (PointsDs_src_cam_coord[2,:] +1e-10) / math.tan( hvfov)
    PointsDs_src_cam_coord[2, :] = (PointsDs_src_cam_coord[2,:] -  z_half ) / z_radius

    # reshape to N x OD x OH x OW x 3 #
    PointsDs_src_cam_coord = PointsDs_src_cam_coord / (PointsDs_src_cam_coord[3,:].unsqueeze_(0) + 1e-10 )
    PointsDs_src_cam_coord = PointsDs_src_cam_coord[:3, :].transpose(0,1).reshape((N, D, H, W, 3))

    # --- 2. Re-sample --- #
    src_vol_th = src_vol.unsqueeze(1)
    src_vol_th_ = _set_vol_border(src_vol_th, padding_value)
    res_vol_th = torch.squeeze(\
            torch.squeeze(\
            F.grid_sample(src_vol_th_, PointsDs_src_cam_coord, mode='bilinear', padding_mode = 'border'), 
            dim=0), \
            dim=0)

    if is_debug:
        return res_vol_th, PointsDs_src_cam_coord, src_vol_th
    else:
        return res_vol_th

def resample_vol_cuda_Rt(src_vol, R, t, cam_intrinsic = None, 
                      d_candi = None, d_candi_new = None,
                      padding_value = 0., output_tensor = False, 
                      is_debug = False, PointsDs_ref_cam_coord_in = None):
    r'''
    src_vol - src vol. NDHW or NCDHW format 

    if d_candi_new is not None: 
    d_candi : candidate depth values for the src view;
    d_candi_new : candidate depth values for the ref view. Usually d_candi_new is different from d_candi
    '''
    assert d_candi is not None, 'd_candi should be some np.array object'

    if src_vol.ndimension() == 4: # src_vol - NDHW
        _, D, H, W = src_vol.shape
    elif src_vol.ndimension() == 5: # src_vol - NCDHW
        _, C, D, H, W = src_vol.shape

    N = 1
    hhfov, hvfov =  math.radians(cam_intrinsic['hfov']) * .5, math.radians(cam_intrinsic['vfov']) * .5

    # --- 0. Get the sampled points in the ref. view --- #
    if PointsDs_ref_cam_coord_in is None:
        PointsDs_ref_cam_coord = torch.zeros(N, D, H, W, 3)
        if d_candi_new is not None:
            d_candi_ = d_candi_new
        else:
            d_candi_ = d_candi

        for idx_d, d in enumerate(d_candi_):
            PointsDs_ref_cam_coord[0, idx_d, :, :, :] \
                    = d * torch.FloatTensor(cam_intrinsic['unit_ray_array'])
        PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.cuda()

    else:
        PointsDs_ref_cam_coord = PointsDs_ref_cam_coord_in

    if d_candi_new is not None:
        z_max, z_min = d_candi.max(), d_candi.min()
    else:
        z_max = torch.max(PointsDs_ref_cam_coord[0, :, :, :, 2 ])
        z_min = torch.min(PointsDs_ref_cam_coord[0, :, :, :, 2 ])

    z_half = (z_max + z_min) * .5
    z_radius = (z_max - z_min) * .5

    # --- 1. Coordinate transform --- # 
    PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.reshape((-1, 3)).transpose(0,1)

    # NOTE: (1) For PointsD_src_cam_coord, use the Cartesian coordinate #
    #       (2) We should avoid in-place operation
    PointsDs_src_cam_coord_tmp = R.matmul( PointsDs_ref_cam_coord )

    PointsDs_src_cam_coord_tmp =  PointsDs_src_cam_coord_tmp+ \
        t.unsqueeze(1).expand(-1, PointsDs_src_cam_coord_tmp.shape[1])

    # transform into range [-1, 1] for all dimensions #
    PointsDs_src_cam_coord = torch.zeros(PointsDs_src_cam_coord_tmp.shape).cuda()
    PointsDs_src_cam_coord[0, :] = \
        PointsDs_src_cam_coord_tmp[0,:] / (PointsDs_src_cam_coord_tmp[2,:] +1e-10) / math.tan( hhfov)

    PointsDs_src_cam_coord[1, :] = \
        PointsDs_src_cam_coord_tmp[1,:] / (PointsDs_src_cam_coord_tmp[2,:] +1e-10) / math.tan( hvfov)

    PointsDs_src_cam_coord[2, :] = \
        (PointsDs_src_cam_coord_tmp[2,:] -  z_half ) / z_radius

    # reshape to N x OD x OH x OW x 3 #
    PointsDs_src_cam_coord = PointsDs_src_cam_coord.transpose(0,1).reshape((N, D, H, W, 3))

    # --- 2. Re-sample --- #
    if src_vol.ndimension() == 4:
        src_vol_th = src_vol.unsqueeze(1)
    elif src_vol.ndimension() == 5:
        src_vol_th = src_vol

    src_vol_th_ = _set_vol_border(src_vol_th, padding_value)
    res_vol_th = torch.squeeze(\
            torch.squeeze(\
            F.grid_sample(src_vol_th_, PointsDs_src_cam_coord, mode='bilinear', padding_mode = 'border'), 
            dim=0), \
            dim=0)
    if is_debug:
        return res_vol_th, PointsDs_src_cam_coord, src_vol_th
    else:
        return res_vol_th

def resample_vol(src_vol, R_ref2src, t_ref2src, cam_intrinsic = None, d_candi =\
                 None, padding_value = 0., output_tensor = False):
    '''
    Inputs:
    src_vol -  H x W x D The source volume. numpy array
    R, t - transform matrix from ref to src view
    Outputs: 
    res_vol - D x H x W  The re-sampled volume. numpy array 
    '''
    import math
    R = R_ref2src; t = t_ref2src
    H,W,D = src_vol.shape
    N = 1
    hhfov, hvfov = math.radians(cam_intrinsic['hfov']) * .5, math.radians(cam_intrinsic['vfov']) * .5

    # --- 0. Get the sampled points in the ref. view --- #
    # so we don't have to re-calculate them everytime we run this function  

    PointsDs_ref_cam_coord = torch.zeros(N, D, H, W, 3)
    for idx_d, d in enumerate(d_candi):
        PointsDs_ref_cam_coord[0, idx_d, :, :, :] \
                = d * torch.FloatTensor(cam_intrinsic['unit_ray_array'])
    PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.cuda()
    z_max = torch.max(PointsDs_ref_cam_coord[0, :, :, :, 2 ])
    z_min = torch.min(PointsDs_ref_cam_coord[0, :, :, :, 2 ])
    z_half = (z_max + z_min) * .5
    z_radius = (z_max - z_min) *.5

    # --- 1. Coordinate transform --- # 
    PointsDs_ref_cam_coord = PointsDs_ref_cam_coord.reshape((-1, 3)).transpose(0,1)
    PointsDs_ref_cam_coord = \
        torch.cat( (PointsDs_ref_cam_coord, torch.ones(1, PointsDs_ref_cam_coord.shape[1]).cuda() ), dim=0)
    src_cam_extM = torch.eye(4)
    src_cam_extM[:3, :3] = torch.FloatTensor(R)
    src_cam_extM[:3, 3] = torch.FloatTensor(t)
    src_cam_extM = src_cam_extM.cuda()
    PointsDs_src_cam_coord = src_cam_extM.matmul( PointsDs_ref_cam_coord )

    # transform into range [-1, 1] for all dimensions #
    PointsDs_src_cam_coord[0, :] = PointsDs_src_cam_coord[0,:] / (PointsDs_src_cam_coord[2,:] +1e-10) / math.tan( hhfov)
    PointsDs_src_cam_coord[1, :] = PointsDs_src_cam_coord[1,:] / (PointsDs_src_cam_coord[2,:] +1e-10) / math.tan( hvfov)
    PointsDs_src_cam_coord[2, :] = (PointsDs_src_cam_coord[2,:] -  z_half ) / z_radius 

    # reshape to N x OD x OH x OW x 3 #
    PointsDs_src_cam_coord = PointsDs_src_cam_coord / (PointsDs_src_cam_coord[3,:].unsqueeze_(0) + 1e-10 )
    PointsDs_src_cam_coord = PointsDs_src_cam_coord[:3, :].transpose(0,1).reshape((N, D, H, W, 3))

    # --- 2. Re-sample --- #
    src_vol_th = torch.FloatTensor( np.transpose(src_vol, axes= [2, 0, 1]) ).unsqueeze_(0).unsqueeze_(0).cuda()
    src_vol_th = _set_vol_border(src_vol_th, padding_value)
    res_vol_th = torch.squeeze(\
            torch.squeeze(\
            F.grid_sample( src_vol_th, PointsDs_src_cam_coord, mode='bilinear', padding_mode = 'border'), 
            dim=0), \
            dim=0)

    if not output_tensor:
        res_vol = res_vol_th.cpu().numpy()
        
    return res_vol

def _set_vol_border( vol, border_val ):
    '''
    inputs:
    vol - a torch tensor in 3D: N x C x D x H x W
    border_val - a float, the border value
    '''
    vol_ = vol + 0.
    vol_[:, :, 0, :, :] = border_val
    vol_[:, :, :, 0, :] = border_val
    vol_[:, :, :, :, 0] = border_val
    vol_[:, :, -1, :, :] = border_val
    vol_[:, :, :, -1, :] = border_val
    vol_[:, :, :, :, -1] = border_val

    return vol_

def _set_vol_border_v0( vol, border_val ):
    '''
    inputs:
    vol - a torch tensor in 3D: N x C x D x H x W
    border_val - a float, the border value
    '''
    vol_ = vol 
    vol_[:, :, 0, :, :] = border_val
    vol_[:, :, :, 0, :] = border_val
    vol_[:, :, :, :, 0] = border_val
    vol_[:, :, -1, :, :] = border_val
    vol_[:, :, :, -1, :] = border_val
    vol_[:, :, :, :, -1] = border_val


def get_rel_extrinsicM(ext_ref, ext_src):
    ''' Get the extrinisc matrix from ref_view to src_view '''
    return ext_src.dot( np.linalg.inv( ext_ref))
