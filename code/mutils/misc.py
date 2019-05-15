'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''


import numpy as np
import math
import PIL.Image as image
import warping.View as View
import warping.homography as warp_homo

import torch
import torch.nn.functional as F
import torchvision



def get_twin_rel_pose( traj_extMs, ref_indx, t_win_r, dat_indx_step , 
        use_gt_R=False, 
        use_gt_t = False , dataset=None, add_noise_gt = False, noise_sigmas=None, 
        traj_extMs_dso = None,
        use_dso_R=False, use_dso_t = False, 
        opt_next_frame = False):
    '''
    Get the relative poses for the source frame in the local time window 
    NOTE: For the last frame in the local time window, we will set its initial pose as the relative pose for 
          t_win_r * dat_indx_step + ref_indx - 1, rather than t_win_r * dat_indx_step + ref_indx, assuming their
          poses are similar.
    '''
    if use_dso_R or use_dso_t:
        assert traj_extMs_dso is not None


    if not opt_next_frame:

        src_frame_idx =   [ idx for idx in range( ref_indx - t_win_r * dat_indx_step, ref_indx, dat_indx_step) ] \
                        + [ idx for idx in range( ref_indx + dat_indx_step, ref_indx + (t_win_r-1)*dat_indx_step+1, dat_indx_step) ] \
                        + [ t_win_r * dat_indx_step + ref_indx -1 ]

        src_frame_idx_opt = [ idx for idx in range( ref_indx - t_win_r * dat_indx_step, ref_indx, dat_indx_step) ] \
                        + [ idx for idx in range( ref_indx + dat_indx_step, ref_indx + t_win_r*dat_indx_step+1, dat_indx_step) ] 
    else:

        src_frame_idx =   [ idx for idx in range( ref_indx - t_win_r * dat_indx_step, ref_indx, dat_indx_step) ] \
                        + [ ref_indx + 1] \
                        + [ idx for idx in range( ref_indx + dat_indx_step, ref_indx + (t_win_r-1)*dat_indx_step+1, dat_indx_step) ] \
                        + [ t_win_r * dat_indx_step + ref_indx -1 ]

        src_frame_idx_opt = [ idx for idx in range( ref_indx - t_win_r * dat_indx_step, ref_indx, dat_indx_step) ] \
                        + [ ref_indx + 1] \
                        + [ idx for idx in range( ref_indx + dat_indx_step, ref_indx + t_win_r*dat_indx_step+1, dat_indx_step) ] 


    ref_cam_extM = traj_extMs[ref_indx] 
    src_cam_extMs = [traj_extMs[i] for i in src_frame_idx] 

    if(isinstance(ref_cam_extM, torch.Tensor) ):
        pass

    src_cam_poses = [warp_homo.get_rel_extrinsicM(ref_cam_extM, src_cam_extM_)  for src_cam_extM_ in src_cam_extMs]
    src_cam_poses = [torch.from_numpy(pose.astype(np.float32)) for pose in  src_cam_poses]

    # dso dr, dso dt #
    if traj_extMs_dso is not None: 
        dRt = torch.FloatTensor( warp_homo.get_rel_extrinsicM(traj_extMs_dso[ref_indx].copy(), 
            traj_extMs_dso[ref_indx + t_win_r *dat_indx_step].copy()))
        if use_dso_R:
            # we will use dso_R (traj_extMs was init. by DSO)
            src_cam_poses[-1][:3, :3] = dRt[:3,:3]
        if use_dso_t:
            # we will use dso_R (traj_extMs was init. by DSO)
            src_cam_poses[-1][:3, 3] = dRt[:3,3]

    if use_gt_R or use_gt_t:
        for idx, srcidx in enumerate(src_frame_idx_opt):
            pose_gt = warp_homo.get_rel_extrinsicM(dataset[ref_indx]['extM'], dataset[srcidx]['extM']) 
            R_gt = torch.from_numpy(pose_gt)[:3,:3]
            t_gt = torch.from_numpy(pose_gt)[:3,3] 

            if use_gt_R:
                print('USING GT R')
                if add_noise_gt:
                    print('add noise to GT')
                    R_gt += torch.randn(R_gt.shape).type_as(R_gt) * noise_sigmas[0]
                src_cam_poses[idx][:3,:3] = R_gt 

            if use_gt_t:
                print('USING GT T')
                if add_noise_gt:
                    print('add noise to GT')
                    t_gt += torch.randn(t_gt.shape).type_as(t_gt) * noise_sigmas[1]

                src_cam_poses[idx][:3,3] = t_gt 

    return src_cam_poses , src_frame_idx_opt

def valid_dpv( dpv_in ):
    if dpv_in is None:
        return False
    else:
        assert isinstance(dpv_in, torch.Tensor), 'input should a Tensor'
        if dpv_in.dim() ==4:
            is_valid = not torch.isnan( dpv_in[0, 0, 0, 0])
        elif dpv_in.dim() ==5:
            is_valid = not torch.isnan( dpv_in[0, 0, 0, 0, 0])
        elif dpv_in.dim() ==3:
            is_valid = not torch.isnan( dpv_in[0, 0,  0])
        elif dpv_in.dim() ==2:
            is_valid = not torch.isnan( dpv_in[0, 0])
        else:
            raise Exception('wrong dimension for input dpv !') 
        return is_valid


def get_IdentityPose(if_tensor = False):
    if if_tensor:
        pose = torch.zeros(4,4)
        pose[:3,:3] = torch.eye(3)
    else:
        pose = np.zeros((4,4) )
        pose[:3,:3] = np.eye(3) 
    pose[3,3] = 1.
    return pose

def upsample_np2d(in_array, ratio = 1):
    # upsample the input 2D numpy array #
    # output a torch array in GPU in 2D #

    up_arr = torch.FloatTensor(in_array).cuda()
    if ratio>1:
        up_arr = F.upsample( up_arr.unsqueeze(0).unsqueeze(0), scale_factor = ratio).squeeze()
    
    return up_arr
    

def downsample_img(img, kernel_size=4):
    # img: NCHW #
    if kernel_size <= 1:
        return img
    else:
        return F.avg_pool2d( img, kernel_size= kernel_size).cuda()

def sub_res_img(res_img_path, per_res_size, sub_img_indx):
    '''
    get a suset of the image (eg. image col 1,3,5 )
    '''
    sub_res_img = np.zeros( ( per_res_size[0], per_res_size[1]* len(sub_img_indx),3))
    res_img = np.asarray(image.open(res_img_path) )[:,:,:3]
    for isub, icol in enumerate(sub_img_indx):
        st_col = icol*per_res_size[1] 
        ed_col = icol*per_res_size[1] + per_res_size[1]
        per_res_img = res_img[:,  st_col: ed_col, : ] 

        st_col = isub*per_res_size[1] 
        ed_col = isub*per_res_size[1] + per_res_size[1]
        sub_res_img[:, st_col:ed_col, :] = per_res_img

    return sub_res_img

def debug_writeVolume(vol, vmin=None, vmax= None):
    '''
    vol - nslice x H x W
    '''

    if vmin == None:
        vmin = vol.min()
    if vmax == None:
        vmax = vol.max()

    import matplotlib.pyplot as plt
    m_makedir('./debug_res')
    for islice in range( vol.shape[0]):
        vol_slice = vol[islice,...] 
        plt.imsave('debug_res/slice_%05d.png'%(islice), vol_slice, vmin = vmin, vmax = vmax)

def save_args(args, filename='args.txt'):
    r'''
    Save the parsed arguments to file.
    This function is useful for recoding the experiment parameters.
    inputs:
    arg - the input arguments 
    filename (args.txt) - the txt file that saves the arguments 
    '''
    arg_str = []
    for arg in vars(args):
        arg_str.append( str(arg) + ': ' + str(getattr(args, arg)) ) 
    with open(filename, 'w') as f:
        for arg_str_ in arg_str:
            f.write('%s\n'%(arg_str_))

def img_gradient(img, ):
    '''
    Get the image gradient 
    We can use this function while calculating the loss function, or calculate dI / dp (pose)

    Input:
    img - NCHW format image

    Output:
    img_grad_x, img_grad_y - NCHW format 
    '''

    N,C = img.shape[0], img.shape[1]
    wx = torch.FloatTensor([[1, 0, -1],[2,0,-2],[1,0,-1]]).cuda().unsqueeze(0).unsqueeze(0)
    wy = torch.FloatTensor([[1, 0, -1],[2,0,-2],[1,0,-1]]).transpose(0,1).cuda().unsqueeze(0).unsqueeze(0)
    wx = wx.repeat([C,C, 1, 1])
    wy = wy.repeat([C,C, 1, 1])
    
    img_grad_x = F.conv2d(img, wx, padding=1)
    img_grad_y = F.conv2d(img, wy, padding=1) 

    return img_grad_x, img_grad_y

def rescale_IntM(intM, scale_x, scale_y):
    intM[ 0, 0] = intM[0, 0] * scale_x
    intM[ 0, 2] = intM[0,2 ] * scale_x

    intM[ 1, 1] = intM[1, 1] * scale_y
    intM[ 1, 2] = intM[1, 2] * scale_y

def array2img(array, v_max, colormap='jet'):
    ''' convert to 2D array to color coded image for visualziation
    '''
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(colormap)
    return cm( ( array / v_max *255.).astype(np.uint8))[:,:,0:3]

def indexMap2DMap(d_range, indx_map):
    indx_map_flat = indx_map.flatten()
    DMap = [ d_range[indx_] for indx_ in indx_map_flat]
    return np.reshape(DMap, indx_map.shape)


def resize_size_np_array(np_array, new_size):
    '''
    resize images (stored in a numpy array)
    input: 
    np_array - input np array 
    new_size - output np array size (in (H, W) format)
    '''
    if np_array.ndim ==2 or np_array.ndim==3:
        return np.asarray(image.fromarray( np_array ).resize([ new_size[1], new_size[0]]) )
    else:
        raise Exception('input array should be 2 or 3 dimension')

def resize_unit_ray_array( cam_intrinsics, new_size ):
    '''
    Return a new cam_intrinsics, with unit_ray_array with size new_size,
    all the other variables remains the same 
    Input: 
    new_size - [H, W]
    '''
    cam_intrinsics_new = {'hfov': cam_intrinsics['hfov'],
            'vfov': cam_intrinsics['vfov'], 
            'intrinsic_M': cam_intrinsics['intrinsic_M'].copy() }

    pixel_to_ray_array = View.normalised_pixel_to_ray_array(\
            width= new_size[1], height= new_size[0], 
            hfov = cam_intrinsics_new['hfov'], 
            vfov = cam_intrinsics_new['vfov'])

    cam_intrinsics_new['unit_ray_array'] = pixel_to_ray_array
    return cam_intrinsics_new

def _hconcat_PIL_imgs(PIL_img_array):
    '''
    concatenate PIL images horizontally 
    Inputs:
    PIL_img_array - the list of PIL images, all should have the same height 

    outputs:
    image_hconcate - the horizontally concatenated image
    '''
    widths, heights = zip(*(i.size for i in PIL_img_array))
    total_width = sum(widths)
    x_offset = 0 
    image_hconcate = image.new('RGB', (total_width, heights[0]))
    for img in PIL_img_array:
       assert img.size[1] == heights[0], 'all images should have the same # of rows'
       image_hconcate.paste(img, (x_offset, 0)) 
       x_offset += img.size[0]
    return image_hconcate

def quaternion2angle(q):
    import math 
    q = np.asarray(q)
    q_l = math.sqrt( q[0]**2 + q[1]**2 + q[2]**2 )
    vec = q[:3] / q_l  
    theta = 2* math.atan2(q_l, q[3])
    return vec, theta  

def quaternion2Rotation(q, is_tensor = False, R_tensor=None, TUM_format=True):
    '''
    input:
    q - 4 element np array:
    q in the TUM monoVO format: qx qy qz qw

    is_tensor - if use tensor array

    R_tensor - 3x3 tensor, should be initialized 

    output:
    Rot - 3x3 rotation matrix 
    '''
    if is_tensor and R_tensor is not None:
        Rot = R_tensor
    elif is_tensor and R_tensor is None:
        Rot = torch.zeros(3,3).cuda()
    else:
        Rot = np.zeros((3,3))

    if TUM_format:
        w, x, y, z = q[3], q[0], q[1], q[2]
    else:
        w, x, y, z =  q[0], q[1], q[2], q[3]


    s = 1 / (w**2 + x**2 + y**2 + z**2)
    Rot[0, 0] = 1 - 2 * s * ( y**2 + z**2 )
    Rot[1, 1] = 1 - 2 * s * ( x**2 + z**2)
    Rot[2, 2] = 1 - 2 * s * ( x**2 + y**2)

    Rot[0, 1] = 2* ( x*y - w * z)
    Rot[1, 0] = 2* ( x*y + w * z)

    Rot[0, 2] = 2* ( x*z + w * y)
    Rot[2, 0] = 2* ( x*z - w * y)

    Rot[1,2] = 2* ( y*z - w * x)
    Rot[2,1] = 2* ( y*z + w * x)


    return Rot

def Rotation2Twist(R):
    quat = torch.zeros(4).cuda()
    twist = torch.zeros(3).cuda()

    Rotation2Quaternion(R, quat)
    quat_v = quat[:3]
    theta = 2 * math.atan2(torch.norm(quat_v, p=2), quat[-1])
    n = quat_v / torch.norm(quat_v, p=2)
    return theta * n

def Twist2Rotation(r_twist):
    angle = torch.norm(r_twist, p = 2)
    if angle< 1e-10:
        R = torch.zeros(3, 3).cuda()
        return R
    else:
        axis = r_twist / angle 
        K = torch.zeros(3,3).cuda()
        K[1,0] = axis[2]
        K[0,1] = -axis[2]
        K[2,0] = -axis[1]
        K[0,2] = axis[1] 
        K[2,1] = axis[0] 
        K[1,2] = -axis[0]
        R = torch.eye(3).cuda() + math.sin( angle ) * K + (1 - math.cos(angle)) * (K.mm(K))
        return R

def Rotation2Quaternion(R, quat):
    '''
    reference: http://www.engr.ucr.edu/~farrell/AidedNavigation/D_App_Quaternions/Rot2Quat.pdf 
    NOTE: quat is in TUM format: quat = [qx qy qz qw] 
    
    ref: 
    https://engineering.purdue.edu/CE/Academics/Groups/Geomatics/DPRG/Slides/Chapter7_Quaternion
    '''

    assert quat.dim() == 1 and len(quat) == 4, 'quat should be of right shape !'

    if R[0,0] + R[1,1] + R[2,2] + 1 > 0:
        quat[3] = .5 * math.sqrt(R[0,0] + R[1,1] + R[2,2] + 1)
        s = 1/ 4 / quat[3]
        quat[0] = s * (R[2,1]-R[1,2])
        quat[1] = s * (R[0,2]-R[2,0])
        quat[2] = s * (R[1,0]-R[0,1])

    elif R[0,0] - R[1,1] - R[2,2] + 1 > 0:
        quat[0] = .5 * math.sqrt(R[0,0] - R[1,1] - R[2,2] + 1)
        s = 1/ 4 / quat[0]
        quat[1] = s * (R[1,0]+R[0,1])
        quat[2] = s * (R[0,2]+R[2,0])
        quat[3] = s * (R[2,1]+R[1,2])

    elif R[1,1] - R[0,0] - R[2,2] + 1 > 0:
        quat[1] = .5 * math.sqrt(R[1,1] - R[0,0] - R[2,2] + 1)
        s = 1/ 4 / quat[0]
        quat[0] = s * (R[1,0]+R[0,1])
        quat[2] = s * (R[1,2]+R[2,1])
        quat[3] = s * (R[0,2]-R[2,0])

    elif R[2,2] - R[0,0] - R[1,1] + 1 > 0:
        quat[2] = .5 * math.sqrt(R[2,2] - R[0,0] - R[1,1] + 1)
        s = 1/ 4 / quat[0]
        quat[0] = s * (R[2,0]+R[0,2])
        quat[1] = s * (R[2,1]+R[1,2])
        quat[3] = s * (R[1,0]-R[0,1])

def UnitQ2Rotation(r_uq):
    assert isinstance(r_uq, torch.Tensor) 
    r_q = torch.zeros(4).cuda()
    unitQ_to_quat(r_uq, r_q)
    R = quaternion2Rotation(r_q, is_tensor=True) 
    return R

def Rotation2UnitQ(R):
    r_q = torch.zeros(4).cuda()
    r_uq = torch.zeros(3).cuda()
    Rotation2Quaternion(R, r_q) 
    quat_to_unitQ( r_q, r_uq) 
    return r_uq
    

def Quaternion2LogQ(quat, log_quat):
    '''
    Quaterniton to log quaternion 
    This is useful for optimizing the camera rotation, where we want to enforce
    the unit quaternion constraint
    Reference: map-net paper (https://arxiv.org/pdf/1712.03342.pdf)
    NOTE: quat in the TUM format 
    '''
    assert len(log_quat)==3 and log_quat.dim()==1, 'incorrect shape for log_quat'
    assert quat.dim() == 1 and len(quat) == 4, 'quat should be of right shape !'

    v= quat[:3]
    u= quat[3]
    v_norm = torch.norm(v, p=2)

    if v_norm == 0:
        log_quat[:] = 0.
    else:
        log_quat[:] = v / v_norm * math.acos(u) 

def LogQ2Quaternion(log_quat, quat):
    '''
    log quaternion to Quaternion 
    This is useful for optimizing the camera rotation, where we want to enforce
    the unit quaternion constraint
    Reference: map-net paper (https://arxiv.org/pdf/1712.03342.pdf)
    NOTE: quat in the TUM format 
    ''' 
    assert len(log_quat)==3 and log_quat.dim()==1, 'incorrect shape for log_quat'
    assert quat.dim() == 1 and len(quat) == 4, 'quat should be of right shape !' 

    w = log_quat
    w_norm = torch.norm(w, p=2)
    if  w_norm == 0:
        quat[3] = 1
        quat[:3]= 0.
    else:
        quat[3] = math.cos(w_norm)
        quat[:3] = w / w_norm * math.sin( w_norm ) 

def unitQ_to_quat( unitQ, quat):
    '''
    Unit quaternion (xyz parameterization) to quaternion
    unitQ - 3 vector,
    quat - 4 vector, TUM format quaternion [x y z w]
    '''
    x, y, z = unitQ[0], unitQ[1], unitQ[2]
    alpha2 = x**2 + y**2 + z**2 

    quat[3] = 2* x / (alpha2 + 1)
    quat[0] = 2* y / (alpha2 + 1)
    quat[1] = 2* z / (alpha2 + 1)
    quat[2] = (1-alpha2) / (1+ alpha2)

def unitQ_to_quat_inv( unitQ, quat):
    '''
    Unit quaternion (xyz parameterization) to inverse quaternion
    unitQ - 3 vector,
    quat - 4 vector, TUM format quaternion [x y z w]
    '''
    x, y, z = unitQ[0], unitQ[1], unitQ[2]
    alpha2 = x**2 + y**2 + z**2

    quat[0] = -2* y / (alpha2 + 1)
    quat[1] = -2* z / (alpha2 + 1)
    quat[2] = -(1-alpha2) / (1+ alpha2)
    quat[3] = 2* x / (alpha2 + 1)

def quat_to_unitQ(quat, unitQ):
    '''
    get Unit quaternion (xyz parameterization) from quaternion
    quat - 4 vector, TUM format quaternion [x y z w]
    unitQ - 3 vector,
    '''
    q1,q2,q3,q0 = quat[0], quat[1], quat[2], quat[3]
    alpha2 = (1-q3) / (1+q3)

    x = q0*(alpha2+1) * .5
    y = q1*(alpha2+1) * .5
    z = q2*(alpha2+1) * .5

    unitQ[0] = x
    unitQ[1] = y
    unitQ[2] = z 
        
def m_makedir(dirpath):
    import os 
    if not os.path.exists(dirpath):
        os.makedirs( dirpath) 

def split_frame_list(frame_list, t_win_r):
    r'''
    split the frame_list into two : ref_frame (an array) and src_frames (a list),
    where ref_frame = frame_list[t_win_r]; src_frames = [0:t_win_r, t_win_r :]
    '''
    nframes = len(frame_list)
    ref_frame = frame_list[t_win_r]
    src_frames = [ frame_list[idx] for idx in range( nframes) if idx != t_win_r ]
    return ref_frame, src_frames

def get_entries_list_dict(list_dict, keyname):
    r'''
    Given the list of dicts, and the keyname
    return the list [list_dict[0][keyname] ,... ]
    '''
    return [_dict[keyname] for _dict in list_dict ]

def add_dimension_N(list_input_array):
    r'''add the dimension such that the new list of arrays is NCHW format '''
    return [arr_.unsqueeze_(0) for arr_ in list_input_array] 


# depth related #
def depth_val_regression(BV_measure, d_candi_cur, BV_log = True, ):
    '''
    inputs:
        BV_measure: NDHW format 

    '''
    assert len(d_candi_cur) == BV_measure.shape[1], \
            'BV_measure should have the same # of slices as len(d_candi_cur) !'

    depth_regress = torch.zeros(1, BV_measure.shape[2], BV_measure.shape[3]).cuda()
    for idx_d, d in enumerate(d_candi_cur):
        if BV_log:
            depth_regress = depth_regress + torch.exp(BV_measure[0,idx_d,:,:]) * d
        else:
            depth_regress = depth_regress + BV_measure[0,idx_d,:,:] * d

    return depth_regress

def depth_var(BV_measure, depth_mean, d_candi_cur, BV_log=True, d_sigma = 1.):
    '''
    Get the depth variance
    input:
        BV_measure - N D H W
        depth_mean - output from depth_val_regression(), 1 x H x W
        d_candi_cur - vector of candidate depth values
        BV_log - if BV in log-scale
    outputs:
        var_map - variancce map, 1 x H x W
    '''
    assert len(d_candi_cur) == BV_measure.shape[1], 'BV_measure should have the same # of slices as len(d_candi_cur) !'
    N, D, H, W = BV_measure.shape
    depth_var = torch.zeros( H, W).cuda()


    for idx_d, d in enumerate(d_candi_cur):
        if BV_log:
            depth_var = depth_var + (torch.exp(BV_measure[0, idx_d, :, :])*d - depth_mean)**2 / (2*d_sigma**2)
        else:
            depth_var = depth_var + (BV_measure[0, idx_d, :, :]*d - depth_mean)**2 / (2*d_sigma**2)

    return depth_var * 1. / D

def dpv_statistics(BV_measure, d_candi, statistics, BV_log = True):
    '''
    input:
        BV_measure - NDHW

        d_candi - depth candidate

        statistics - list of strings, e.g. ['E_mean', 'variance', ]
        'E_mean' : expected mean
        'variance': variance of depth
        'max' , 'min': min and max of BV_measure, along depth

        BV_log - if dpv in log-scale

    output:
        BV_features: NCHW, C = # of features, len(statistics)
    '''
    dpv_feats = []

    for name in statistics:
        if name is 'E_mean':
            depth_regress = depth_val_regression(BV_measure, d_candi, BV_log = BV_log)
            dpv_feats.append( depth_regress.unsqueeze(1))
        if name is 'variance':
            dpv_feats.append( depth_var(BV_measure, depth_regress, d_candi, BV_log = BV_log).unsqueeze(1))            
        if name is 'max':
            dpv_max, _ = torch.max( torch.exp(BV_measure), dim=1) if BV_log else torch.max(BV_measure, dim=1)
            dpv_feats.append( dpv_max.unsqueeze(1) )  
        if name is 'min':
            dpv_min, _ = torch.min( torch.exp(BV_measure), dim=1) if BV_log else torch.max(BV_measure, dim=1)
            dpv_feats.append( dpv_min.unsqueeze(1) )  

    assert len(dpv_feats)> 0, 'No channels in dpv_feats. Did you specify the right dpv statistics ?' 
    dpv_feats = torch.cat(dpv_feats, dim=1) 
    return dpv_feats



# IO # 
def save_ScenePathInfo(fpath, scene_path_info):
    ''' 
    write scene_path_info into result folder

    Inputs
    fpath -- the output txt file 
    scene_path_info -- a list of list. Its first element is the path to the
    scene, the other elements are pairs of export results image indx and image
    file path 
    '''

    f = open(fpath, 'w')
    for row in scene_path_info: 
        for i, ele in enumerate(row):
            f.write( str(ele) ) 
            if i < len(row)-1:
                f.write('  ')
        f.write('\n')
    f.close()
