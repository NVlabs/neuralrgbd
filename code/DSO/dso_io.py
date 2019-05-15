
'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''


# the interface for DSO#

import numpy as np 
import subprocess 
import mutils.misc as misc


def read_trajM_fromMat(mat_file, filt_win = 21, filt_ord = 3, returnRaw=False):
    '''
    read trajector matrx from mat file.
    trajExtM saved by scanNet_visDSO(), it is a nframe x 4 x4 matrix

    NOTE:
    Here we only return the filtered camera position
    as a nframe x 3 matrix 
    '''
    import scipy.signal as ssig
    import scipy.io as sio
    import scipy.ndimage.filters as nd_filter

    Mat_trajExtM = sio.loadmat(mat_file) #trajExtM saved by scanNet_visDSO
    Mat_trajExtM = Mat_trajExtM['Mat_traj_extM']
    if returnRaw:
        return Mat_trajExtM
    else: # only return the positions #
        invalid_pos_in_traj = [np.sum(np.isnan(Mat_trajExtM[idx]))>0 for idx in
                             range(Mat_trajExtM.shape[0])]
        
        invalid_pos_in_traj[0] = True # the first frame camera pose is invalid
        valid_pos_in_traj = (1 - np.asarray(invalid_pos_in_traj)).astype(np.bool)
        mat_camtraj = Mat_trajExtM[valid_pos_in_traj, ...]
        T_traj = mat_camtraj[:, :3, 3] 

        T_traj_full = Mat_trajExtM[:, :3, 3]
        R_traj_full = Mat_trajExtM[:, :3,:3]

        Tx_traj, Ty_traj, Tz_traj = T_traj[:,0], T_traj[:,1], T_traj[:,2]

        b,a = ssig.butter(4, 1/filt_win, 'low')
        Tx_traj_filt = ssig.filtfilt(b,a, Tx_traj, )
        Ty_traj_filt = ssig.filtfilt(b,a, Ty_traj, ) 
        Tz_traj_filt = ssig.filtfilt(b,a, Tz_traj, )
        T_traj_full[valid_pos_in_traj, 0] = Tx_traj_filt 
        T_traj_full[valid_pos_in_traj, 1] = Ty_traj_filt 
        T_traj_full[valid_pos_in_traj, 2] = Tz_traj_filt 

        return T_traj_full

def _write_camera_txt(txt_path, intrinsic_info, crop_size = [640, 480], debug=False ):
    K = intrinsic_info['IntM']
    img_size = intrinsic_info['img_size']
    fx = K[0,0] / img_size[0]
    fy = K[1,1] / img_size[1]
    cx = (K[0, 2] + .5) / img_size[0]
    cy = (K[1, 2] + .5) / img_size[1]
    txt_file = open( txt_path, 'w' )
    txt_file.write( str(fx) + " ") 
    txt_file.write( str(fy)+ " ") 
    txt_file.write( str(cx)+ " ") 
    txt_file.write( str(cy)+ " ") 
    # for debugging #
    if debug:
        print('DB MODE')
        # this is the omega value for the TUM_mono_video data #
        txt_file.write("0.897966326944875\n") 
    else:
        txt_file.write("0\n") 

    txt_file.write( str( img_size[0]) + " ") 
    txt_file.write( str( img_size[1])+ "\n") 
    
    if crop_size is not None:
        txt_file.write( "crop\n") 
        txt_file.write("%d "%(crop_size[0])) 
        txt_file.write("%d"%(crop_size[1])) 
    else:
        txt_file.write("none\n") 
        txt_file.write("%d "%(img_size[0])) 
        txt_file.write("%d"%(img_size[1])) 
#
    txt_file.close() 

def _read_camera_poses(result_txt_file, if_invert = False,
        if_filter = False, filt_win = 21, filt_ord = 3 ):
    '''
    inputs: 
    result_txt_file - the .txt file saving the DSO result 

    outputs:
    Rts_cam_to_world - array of camera to world transform matrix
    '''
    Rts_cam_to_world = []
    Qs = []
    with open(result_txt_file) as f:
        contents = f.readlines()
    nframes = len(contents)
    for iframe in range(nframes):
        content = contents[iframe].strip().split()
        if content[1] == 'InvalidPose':
            Rts_cam_to_world.append(-1)
            Qs.append( -1 )
        else:
            t_ = [float(content[1]), float(content[2]), float(content[3]), 1]
            # q_ in the TUM monoVO format: qx qy qz qw, according to :
            # https://github.com/JakobEngel/dso
            q_ = [float(content[4]), float(content[5]), float(content[6]), float(content[7])]
            Qs.append(q_)
            R_ = misc.quaternion2Rotation(q_)
            Rt = np.eye(4)
            Rt[:3, :3] = R_ 
            Rt[:, -1] = t_

            if if_invert:
                Rt = np.linalg.inv(Rt)
            Rts_cam_to_world.append(Rt)

    if if_filter:
        import scipy.signal as ssig

        invalid_pos_in_traj = [ not valid_pose( Rt ) for Rt in Rts_cam_to_world ]
        invalid_pos_in_traj[0] = True # the first frame camera pose is invalid
        valid_pos_in_traj = (1 - np.asarray(invalid_pos_in_traj)).astype(np.bool)

        mat_camtraj = np.dstack(Rts_cam_to_world)
        T_traj = mat_camtraj[:3, 3, valid_pos_in_traj] 

        Tx_traj, Ty_traj, Tz_traj = T_traj[0, :], T_traj[1, :], T_traj[2, :]
        b,a = ssig.butter(4, 1/filt_win, 'low')
        Tx_traj_filt = ssig.filtfilt(b,a, Tx_traj, )
        Ty_traj_filt = ssig.filtfilt(b,a, Ty_traj, ) 
        Tz_traj_filt = ssig.filtfilt(b,a, Tz_traj, )

        ii = 0
        for idx in range( len( Rts_cam_to_world)):
            if valid_pos_in_traj[idx]:
                Rts_cam_to_world[idx][3, 0] = Tx_traj_filt[ii]
                Rts_cam_to_world[idx][3, 1] = Ty_traj_filt[ii]
                Rts_cam_to_world[idx][3, 2] = Tz_traj_filt[ii]
                ii +=1

    return Rts_cam_to_world


# -------------------------- #

def rescale_Intm( intrinsic_info, img_size_output ):
    '''
    Rescale the intrinsics, in case that the image size is different from the
    image size in the calibration
    
    Inputs:
    intrinsic_info - {'IntM': IntM, 'img_size': [img_width, img_height] }
    The intrinsic information is for the camera used in camera calibration 

    img_size_output - the output image size : [img_width, img_height]

    Outputs:
    intrinsic_info_out - same structure as the input intrinsic_info, but for output image size 
    '''
    img_size_input = intrinsic_info['img_size']
    intM_rescale = np.eye(intrinsic_info['IntM'].shape[0])
    intM = intrinsic_info['IntM']

    scale_x, scale_y = float( img_size_output[0]) / float( img_size_input[0]),\
            float(img_size_output[1]) / float( img_size_input[1])

    intM_rescale[0, 0] = intM[0, 0] * scale_x
    intM_rescale[0, 2] = intM[0, 2] * scale_x
    intM_rescale[1, 1] = intM[1, 1] * scale_y
    intM_rescale[1, 2] = intM[1, 2] * scale_y
    intrinsic_info_out = {'IntM': intM_rescale, 'img_size': [img_size_output[0], img_size_output[1]]}

    return intrinsic_info_out

def run_DSO(img_fldr_path, intrinsic_info,
        dso_crop_size = [640, 480],
        vig_img_path = None,
        dso_bin_path = '../../third_party/dso/build/bin/dso_dataset',
        if_invert  = False,
        if_debug  = False,
        result_path = None,
        max_frame = None, 
        min_frame=0,
        mode = 1, preset = 0,
        cam_path = None, nogui = 1, 
        use_existing = False):
    '''
    Run the DSO, return the camera poses  
    inputs : 
    img_fldr_path - the path for the input images 

    intrinsic_info - {'IntM': IntM, 'img_size': [img_width, img_height] }

    dso_bin_path (optional) - the path for the dso binary file. 
    By defulat, it is 
    /home/chaoliu/tools/DSO/dso/build/bin/dso_dataset
    vig_img_path - The path to the vigette imge 
    outputs: 
    RTs_cam_to_world - 
    The array of the estimated camera-to-world matrices 
    '''
    import os.path
    import shutil

    if result_path is None:
        result_path = './result.txt'

    # write to the camera.txt file, which includes the camera intrinsic information #
    if cam_path is None:
        txt_path = 'camara.txt'
        _write_camera_txt(txt_path, intrinsic_info, debug= if_debug, crop_size=dso_crop_size)
    else:
        txt_path = cam_path

    if vig_img_path is None:
       import PIL.Image as image 
       img_pil = image.fromarray( 255* np.ones([ dso_crop_size[1], dso_crop_size[0]], dtype=np.uint8) ) 
       img_pil.save( './dummy_vig_img.png')
       vig_img_path = './dummy_vig_img.png'

    # check if we already have the results #
    if os.path.exists(result_path) and use_existing:
        print('DSO result exists at: %s. Reading the results...'%(result_path))
        RTs_cam_to_world = _read_camera_poses(result_path, if_invert= if_invert)

    else:
        # run #

        if max_frame is None:
#            cmd = '%s files=%s calib=%s vignette=%s preset=0 mode=2 nogui=1 quiet=1 nomt=0'%(\
#                    dso_bin_path, img_fldr_path, txt_path, vig_img_path)

#            cmd = '%s files=%s calib=%s preset=1 mode=2 nogui=1 quiet=1 nomt=1 reverse=0 end=350'%(\
#                    dso_bin_path, img_fldr_path, txt_path, )

            cmd = '%s files=%s calib=%s preset=%d mode=%d nogui=%d quiet=1 nomt=0 reverse=0 '%(\
                    dso_bin_path, img_fldr_path, txt_path, preset, mode, nogui )

        else:
            cmd = '%s files=%s calib=%s  preset=%d mode=%d nogui=%d quiet=1 end=%s start=%d nomt=0 '%(\
                    dso_bin_path, img_fldr_path, txt_path,  preset, mode, nogui, 
                    str(int(max_frame)), min_frame) 

        subprocess.call( cmd, shell=True)
        # parse the result.txt #
        RTs_cam_to_world = _read_camera_poses( './result.txt', if_invert= if_invert )
        # copy file to the result folder #
        shutil.move( './result.txt', result_path)

    return RTs_cam_to_world 


def valid_pose(Rt, ): 

    diff = np.eye(4) - Rt
    if np.abs( diff).max() ==0 :
        print('DSO did not return')
        return False
    elif np.any( np.isnan(Rt) ):
        print('NaN Rt')
        return False
    else:
        return True

def valid_poses(Rts, src_idxs ):
    is_valid = True
    for id_src in src_idxs:
        if not valid_pose( Rts[id_src] ):
            is_valid = False
            return is_valid

    return is_valid 

