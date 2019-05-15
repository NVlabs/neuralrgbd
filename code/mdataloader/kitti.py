'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# data loader for KITTI dataset #
# We will use pykitti module to help to read the image and camera pose #

import pykitti

import numpy as np
import os 
import math
import sys
import glob 
import os.path
import scipy.io as sio

import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as tfv_transform

import mdataloader.m_preprocess as m_preprocess
import mdataloader.misc as mloader_misc
import warping.View as View 


def _read_libviso_res(filepath, if_filter=False):
    '''
    filepath - file path to the .mat file
    .mat file includes : 'mat_Ts', 'img_paths'
    '''
    mat_info = sio.loadmat(filepath)
    mat_Ts = mat_info['mat_Ts'] 
    img_paths = mat_info['img_paths']
    img_paths = [str(img_paths[0][i][0]) for i in range(len( img_paths[0]))  ] 

    filt_win = 11

    if if_filter:
        import scipy.signal as ssig

        T_traj = mat_Ts[:3, 3, :] 
        Tx_traj, Ty_traj, Tz_traj = T_traj[0, :], T_traj[1, :], T_traj[2, :]
        b,a = ssig.butter(4, 1/filt_win, 'low')
        Tx_traj_filt = ssig.filtfilt(b,a, Tx_traj, )
        Ty_traj_filt = ssig.filtfilt(b,a, Ty_traj, ) 
        Tz_traj_filt = ssig.filtfilt(b,a, Tz_traj, )

        mat_Ts[0, 3, :] = Tx_traj_filt
        mat_Ts[1, 3, :] = Ty_traj_filt
        mat_Ts[2, 3, :] = Tz_traj_filt

    return mat_Ts, img_paths

def _read_split_file( filepath):
    '''
    Read data split txt file provided by KITTI dataset authors
    ''' 
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [ x.strip() for x in trajs ] 

    return trajs 

def _read_IntM_from_pdata( p_data,  out_size = None ):
    '''
    Get the intrinsic camera info from pdata
    raw_img_size - [width, height]
    '''

    IntM = np.zeros((4,4)) 
    raw_img_size = p_data.get_cam2(0).size
    width = int( raw_img_size[0] ) 
    height = int( raw_img_size[1]) 
    IntM = p_data.calib.K_cam2 
    focal_length = np.mean([IntM[0,0], IntM[1,1]])
    h_fov = math.degrees(math.atan(IntM[0, 2] / IntM[0, 0]) * 2)
    v_fov = math.degrees(math.atan(IntM[1, 2] / IntM[1, 1]) * 2)

    if out_size is not None: # the depth map is re-scaled #
        camera_intrinsics = np.zeros((3,4))
        pixel_width, pixel_height = out_size[0], out_size[1]
        camera_intrinsics[2,2] = 1.
        camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(h_fov/2.0))
        camera_intrinsics[0,2] = pixel_width/2.0
        camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(v_fov/2.0))
        camera_intrinsics[1,2] = pixel_height/2.0

        IntM = camera_intrinsics
        focal_length = pixel_width / width * focal_length
        width, height = pixel_width, pixel_height


    # In scanenet dataset, the depth is perperdicular z, not ray distance #
    pixel_to_ray_array = View.normalised_pixel_to_ray_array(\
            width= width, height= height, hfov = h_fov, vfov = v_fov,
            normalize_z = True) 

    pixel_to_ray_array_2dM = np.reshape(np.transpose( pixel_to_ray_array, axes= [2,0,1] ), [3, -1])
    pixel_to_ray_array_2dM = torch.from_numpy(pixel_to_ray_array_2dM.astype(np.float32)).cuda() 
    cam_intrinsic = {\
            'hfov': h_fov, 'vfov': v_fov,
            'unit_ray_array': pixel_to_ray_array,
            'unit_ray_array_2D': pixel_to_ray_array_2dM,
            'intrinsic_M_cuda': torch.from_numpy(IntM[:3,:3].astype(np.float32)).cuda(),  
            'focal_length': focal_length,
            'intrinsic_M': IntM}  
    return cam_intrinsic 

def get_paths(traj_indx, database_path_base = '/datasets/kitti', split_txt = None, mode = 'train'):
    ''' 
    Return the training info for one trajectory 
    Assuming: 

    (1) the kitti data set is organized as
    /path_to_kitti/rawdata
    /path_to_kitti/train
    /path_to_kitti/val
    where rawdata folder contains the raw kitti data, 
    train and val folders contains the GT depth

    (2) the depth frames for one traj. is always nimg - 10 (ignoring the first and last 5 frames)

    (3) we will use image 02 data (the left camera ?)

    Inputs:
    - traj_indx : the index (in the globbed array) of the trajectory
    - databse_path_base : the path to the database 
    - split_txt : the split txt file path, including the name of trajectories for trianing/testing/validation
    - mode: 'train' or 'val' or 'test'

    Outputs:
    - n_traj : the # of trajectories in the set defined in split_txt
    - pykitti_dataset : the dataset object defined in pykitti. It is used to get the input images and camera poses
    - dmap_paths: the list of paths for the GT dmaps, same length as dataset
    - poses: list of camera poses, corresponding to dataset and dmap_paths

    ''' 
    assert split_txt is not None, 'split_txt file is needed'

    scene_names = _read_split_file(split_txt) 
    n_traj = len( scene_names )

    assert traj_indx < n_traj, 'traj_indx should smaller than the scene length'

    basedir = database_path_base + '/rawdata'
    sceneName = scene_names[traj_indx]
    name_contents = sceneName.split('_')
    date = name_contents[0] + '_' + name_contents[1] + '_' + name_contents[2]
    drive = name_contents[4] 
    p_data_full = pykitti.raw(basedir, date, drive)
    nimg = len(p_data_full)

    #
    # assume: the depth frames for one traj. is always nimg - 10 (ignoring the first and last 5 frames)
    p_data = pykitti.raw(basedir, date, drive, frames= range(5, nimg-5))

    nimg = len(p_data) 
    dmap_paths = []

    poses = []
    for i_img in range(nimg): 
        imgname = p_data.cam2_files[i_img].split('/')[-1]
        poses.append( p_data.oxts[i_img].T_w_imu)
        dmap_file = '%s/%s/%s/proj_depth/groundtruth/image_02/%s'%( database_path_base, mode, sceneName, imgname)
        dmap_paths.append( dmap_file ) 

    intrin_path = 'NOT NEEDED' 
    return n_traj, p_data, dmap_paths, poses, intrin_path

def _read_img(p_data, indx, img_size = None, no_process= False, only_resize = False):
    '''
    Read image and process 
    '''
    proc_img = m_preprocess.get_transform()
    if no_process:
        img = p_data.get_cam2(indx)
        width, height = img.size
    else:
        if img_size is not None:
            img = p_data.get_cam2(indx)
            img = img.resize( img_size, PIL.Image.BICUBIC )
        else:
            img = p_data.get_cam2(indx)

        width, height = img.size
        if not only_resize:
            img = proc_img(img)

    raw_sz = (width, height)
    return img,  raw_sz

def _read_dimg(path, img_size = None, no_process= False, only_resize = False):
    '''
    Read image and process 
    '''

    if os.path.exists(path): 
        proc_img = m_preprocess.get_transform()
        if no_process:
            img = PIL.Image.open(path)
            width, height = img.size
        else:
            if img_size is not None:
                img = PIL.Image.open(path).convert('RGB')
                img = img.resize( img_size, PIL.Image.BICUBIC )
            else:
                img = PIL.Image.open(path).convert('RGB')
            width, height = img.size
            if not only_resize:
                img = proc_img(img)

        raw_sz = (width, height)
        return img,  raw_sz
    else:
        return -1, -1

class KITTI_dataset(data.Dataset):

    def __init__(self, training, p_data, dmap_seq_paths, poses, intrin_path, 
                 img_size = [1248,380], digitize = False, d_candi = None, resize_dmap = None, if_process = True,
                 crop_w = 384):

        '''
        inputs:

        traning - if for training or not 

        p_data - a pykitti.raw object, returned by get_paths()

        dmap_seq_paths - list of dmaps, returned by get_paths()

        poses - list of posesc

        d_candi - the candidate depths 
        digitize (Optional; False) - if digitize the depth map using d_candi

        resize_dmap - the scale for re-scaling the dmap the GT dmap size would be self.img_size * resize_dmap

        if_process - if do post process 

        crop_w - the cropped image width. We will crop the central region with wdith crop_w

        '''

        assert len(p_data) == len(dmap_seq_paths) == len( poses) 

        if crop_w is not None:
            assert (img_size[0] - crop_w )%2 ==0 and crop_w%4 ==0
            assert resize_dmap is not None

        self.p_data = p_data
        self.dmap_seq_paths = dmap_seq_paths 
        self.poses = poses
        self.training = training
        self.intrin_path = intrin_path # reserved

        self.to_gray = tfv_transform.Compose( [tfv_transform.Grayscale(), tfv_transform.ToTensor()]) 
        self.d_candi = d_candi
        self.digitize = digitize 

        self.crop_w = crop_w

        if digitize:
            self.label_min = 0
            self.label_max = len(d_candi) - 1

        # usample in the d dimension #
        self.dup4_candi = \
                np.linspace( self.d_candi.min(), self.d_candi.max(), len(self.d_candi) * 4) 
        self.dup4_label_min = 0
        self.dup4_label_max = len(self.dup4_candi) -1
        ##

        self.resize_dmap = resize_dmap 

        self.img_size = img_size # the raw input image size (used for resizing the input images) 

        self.if_preprocess = if_process

        # initialization about the camera intrsinsics, which is the same for all data #
        if crop_w is None:
            cam_intrinsics = self.get_cam_intrinsics()
        else:
            width_ = int(crop_w * self.resize_dmap)
            height_ = int(float( self.img_size[1] * self.resize_dmap)) 
            img_size_ = np.array([width_, height_], dtype=int)
            cam_intrinsics = self.get_cam_intrinsics(img_size = img_size_ )
            
        self.cam_intrinsics = cam_intrinsics 

    def get_cam_intrinsics(self,  img_size = None):
        '''
        Get the camera intrinsics 
        '''
        if img_size is not None:
            width = img_size[0]
            height = img_size[1]
        else:
            if self.resize_dmap is None:
                width = self.img_size[0]
                height = self.img_size[1] 
            else:
                width = int(float( self.img_size[0] * self.resize_dmap))
                height = int(float( self.img_size[1] * self.resize_dmap))
        
        cam_intrinsics = _read_IntM_from_pdata( self.p_data, out_size = [width, height], ) 
        self.cam_intrinsics = cam_intrinsics

        return cam_intrinsics 

    def __getitem__(self, indx):
        '''
        outputs:
        img, dmap, extM, scene_path , as entries in a dic.
        '''

        dmap_path = self.dmap_seq_paths[indx] 
        proc_normalize = m_preprocess.get_transform()
        proc_totensor = m_preprocess.to_tensor()

        # read rgb image #
        img = _read_img(self.p_data, indx, no_process = True)[0]
        img = img.resize(self.img_size, PIL.Image.NEAREST) 
        if self.resize_dmap is not None:
            img_dw = img.resize(
                    [int(self.img_size[0]* self.resize_dmap), int(self.img_size[1]* self.resize_dmap)],
                    PIL.Image.NEAREST)
        else:
            img_dw = None
        img_gray = self.to_gray(img) 

        # read GT depth map (if available) #
        dmap_raw = _read_dimg(dmap_path, no_process = True)[0]
        if dmap_raw is not -1:
            dmap_mask_imgsize = np.array(dmap_raw, dtype=int).astype(np.float32) < 0.01
            dmap_mask_imgsize = PIL.Image.fromarray(
                    dmap_mask_imgsize.astype(np.uint8) * 255).resize([self.img_size[0], 
                        self.img_size[1]], PIL.Image.NEAREST )
            if self.resize_dmap is not None:
                dmap_imgsize = dmap_raw.resize([self.img_size[0], self.img_size[1]], PIL.Image.NEAREST)
                dmap_imgsize = proc_totensor(dmap_imgsize)[0, :, :].float() / 256.
                dmap_rawsize = proc_totensor(dmap_raw)[0,:,:].float()  / 256.
                # resize the depth map #
                dmap_size = [ int(float(self.img_size[0]) * self.resize_dmap), 
                        int(float(self.img_size[1]) * self.resize_dmap)] 

                dmap_raw_bilinear_dw = dmap_raw.resize(dmap_size, PIL.Image.BILINEAR) 
                dmap_raw = dmap_raw.resize(dmap_size, PIL.Image.NEAREST)
                dmap_mask = dmap_mask_imgsize.resize(dmap_size, PIL.Image.NEAREST)
            dmap_raw = proc_totensor(dmap_raw)[0,:,:] # single-channel for depth map
            dmap_raw = dmap_raw.float() / 256. # scale to meter
            dmap_raw_bilinear_dw = proc_totensor(dmap_raw_bilinear_dw)[0,:,:]
            dmap_raw_bilinear_dw = dmap_raw_bilinear_dw.float() / 256.
            if self.resize_dmap is None:
                dmap_rawsize = dmap_raw
            dmap_mask = 1 - (proc_totensor(dmap_mask) > 0 )
            dmap_mask_imgsize = 1-( proc_totensor( dmap_mask_imgsize ) >0)
            dmap_raw = dmap_raw * dmap_mask.squeeze().type_as(dmap_raw)
            dmap_raw_bilinear_dw = dmap_raw_bilinear_dw * dmap_mask.squeeze().type_as(dmap_raw)
            dmap_imgsize = dmap_imgsize * dmap_mask_imgsize.squeeze().type_as(dmap_imgsize)
            if self.digitize:
                # digitize the depth map #
                dmap = mloader_misc.dMap_to_indxMap( dmap_raw, self.d_candi )
                dmap[dmap >= self.label_max] = self.label_max
                dmap[dmap <= self.label_min] = self.label_min
                dmap = torch.from_numpy(dmap)

                dmap_imgsize_digit = mloader_misc.dMap_to_indxMap(dmap_imgsize, self.d_candi)
                dmap_imgsize_digit[dmap_imgsize_digit >= self.label_max] = self.label_max
                dmap_imgsize_digit[dmap_imgsize_digit<= self.label_min] = self.label_min
                dmap_imgsize_digit = torch.from_numpy(dmap_imgsize_digit) 

                dmap_up4_imgsize_digit = mloader_misc.dMap_to_indxMap(dmap_imgsize, self.dup4_candi)
                dmap_up4_imgsize_digit[dmap_up4_imgsize_digit >= self.dup4_label_max] = self.dup4_label_max
                dmap_up4_imgsize_digit[dmap_up4_imgsize_digit <= self.dup4_label_min] = self.dup4_label_min
                dmap_up4_imgsize_digit = torch.from_numpy(dmap_up4_imgsize_digit) 

            else:
                dmap = dmap_raw
                dmap_imgsize_digit = dmap_imgsize
                dmap_up4_imgsize_digit = dmap_imgsize

        if self.if_preprocess:
            img = proc_normalize(img)
            if self.resize_dmap is not None:
                img_dw = proc_normalize(img_dw)
        else:
            proc_totensor = tfv_transform.ToTensor()
            img = proc_totensor(img) 
            if self.resize_dmap is not None:
                img_dw = proc_totensor(img_dw)

#        extM = self.poses[indx]
        if self.crop_w is not None:
            side_crop = int( (self.img_size[0] - self.crop_w )/2 ) 
            side_crop_dw = int( side_crop * self.resize_dmap )

            img_size = self.img_size
            img = img[:, :, side_crop: img_size[0]-side_crop]
            img_dw = img_dw[:, :, side_crop_dw: img_dw.shape[-1]-side_crop_dw]

            img_gray = img_gray[:, :, side_crop: img_size[0]-side_crop] 

            if dmap_raw is not -1:
                dmap = dmap[ :, side_crop_dw: (dmap.shape[1] - side_crop_dw) ] 
                dmap_raw = dmap_raw[ :, side_crop_dw: (dmap_raw.shape[1] - side_crop_dw) ]
                dmap_raw_bilinear_dw = dmap_raw_bilinear_dw[ 
                        :, side_crop_dw: (dmap_raw_bilinear_dw.shape[1] - side_crop_dw) ] 
                dmap_rawsize = dmap_rawsize[ :,  side_crop : (dmap_rawsize.shape[1] - side_crop ) ]
                dmap_imgsize = dmap_imgsize[ :,  side_crop : (dmap_imgsize.shape[1] - side_crop ) ]

                dmap_imgsize_digit = dmap_imgsize_digit[ 
                        :,  side_crop : (dmap_imgsize_digit.shape[1] - side_crop ) ] 

                dmap_up4_imgsize_digit = dmap_up4_imgsize_digit[ :,  side_crop : (dmap_up4_imgsize_digit.shape[1] - side_crop ) ] 

                dmap_mask = dmap_mask[:,  :, side_crop_dw: (dmap_mask.shape[-1] - side_crop_dw) ]
                dmap_mask_imgsize = dmap_mask_imgsize[
                        :,  :, side_crop: (dmap_mask_imgsize.shape[-1] - side_crop) ] 

        # read extrinsics #
        # IMU to camera #
        M_imu2cam = self.p_data.calib.T_cam2_imu
        extM = np.matmul( M_imu2cam, np.linalg.inv(self.poses[indx]) ) 
        # image path #
        scene_path = self.p_data.calib_path
        img_path = self.p_data.cam2_files[indx]

        return {'img': img.unsqueeze_(0), 
                'img_dw': img_dw.unsqueeze_(0),
                'dmap': dmap.unsqueeze_(0) if dmap_raw is not -1 else -1, 
                'dmap_raw':dmap_raw.unsqueeze_(0) if dmap_raw is not -1 else -1, 
                'dmap_raw_bilinear_dw':dmap_raw_bilinear_dw.unsqueeze_(0) if dmap_raw is not -1 else -1, 
                'dmap_rawsize': dmap_rawsize.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_imgsize': dmap_imgsize.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_imgsize_digit': dmap_imgsize_digit.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_up4_imgsize_digit': dmap_up4_imgsize_digit.unsqueeze_(0) if dmap_raw is not -1 else -1,
                'dmap_mask': dmap_mask.unsqueeze_(0).type_as(dmap) if dmap_raw is not -1 else -1, 
                'dmap_mask_imgsize': dmap_mask_imgsize.unsqueeze_(0).type_as(dmap) if dmap_raw is not -1 else -1, 
                'img_gray': img_gray.unsqueeze_(0),
                'extM': extM, 
                'scene_path': scene_path, 
                'img_path': img_path, }

    def __len__(self):
        return len(self.p_data)

    def set_paths(self, p_data, dmap_seq_paths, poses):
        '''
        set the p_data, poses and dmaps paths
        p_data - a pykitti.raw object, returned by get_paths()
        dmap_seq_paths - list of dmaps, returned by get_paths()
        poses - list of posesc
        '''
        self.p_data = p_data 
        self.dmap_seq_paths = dmap_seq_paths
        self.poses = poses 

