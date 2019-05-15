'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import math
import numpy as np
import os
import pathlib
import sys
import PIL 
import scipy.io as sio
import warping.View as View
import torch
import torch.utils.data as data
import torchvision.transforms as tfv_transform

import mdataloader.m_preprocess as m_preprocess
import mdataloader.misc as mloader_misc

'''
Data IO for the captured data by ourselves, using iphone or other cameras 
'''

def read_img(path, img_size = None, no_process= False, only_resize = False):
    r'''
    Read image and process 
    '''
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
    return img, raw_sz

def get_paths_1frame(dummy, database_path_base, name_pattern, dat_indx_step=1):
    ''' 
    Inputs:
    - database_path_base : the path for the scanNet dataset
    - img_fmt: one of 'jpg', 'png'

    Outputs:
    - n_traj: # of trajs  
    - img_paths : array of paths for the input images
    - dmap_paths: like img_pahths
    - pose_paths : array of paths for the poses txt files  
    '''

    import glob 
    img_paths = sorted( glob.glob('%s/%s'%(database_path_base, name_pattern )) ) 
    n_traj = 1 
    pose_paths = None
    dmap_paths = None
    intrinsic_path = None

    return n_traj, img_paths, dmap_paths, pose_paths, intrinsic_path 


def read_IntM_from_mat(fpath_mat , out_size):
    r'''
    read intrinsic matrix from a .mat file
    in the mat file, two vars are needed:
        IntM - the 3 x 3 intrinsic matrix for the camera
        img_size - a 1x2 matrix for the image: [width, height]

    output:
    cam_intrinsic = { 'hfov': h_fov, 'vfov': v_fov,
                      'unit_ray_array': pixel_to_ray_array, 'unit_ray_array_2D': pixel_to_ray_array_2dM,
                      'intrinsic_M_cuda': torch.from_numpy(IntM[:3,:3].astype(np.float32)),  
                      'focal_length': focal_length, 'intrinsic_M': IntM}  
    ''' 
    cam_info = sio.loadmat( fpath_mat)
    IntM = cam_info['IntM']

    focal_length = np.mean([IntM[0,0], IntM[1,1]])
    h_fov = math.degrees(math.atan(IntM[0, 2] / IntM[0, 0]) * 2)
    v_fov = math.degrees(math.atan(IntM[1, 2] / IntM[1, 1]) * 2)
    width, height = out_size[0], out_size[1] 

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
    pixel_to_ray_array_2dM = torch.from_numpy(pixel_to_ray_array_2dM.astype(np.float32))

    cam_intrinsic = {\
            'hfov': h_fov,
            'vfov': v_fov,
            'unit_ray_array': pixel_to_ray_array,
            'unit_ray_array_2D': pixel_to_ray_array_2dM,
            'intrinsic_M_cuda': torch.from_numpy(IntM[:3,:3].astype(np.float32)),  
            'focal_length': focal_length,
            'intrinsic_M': IntM}  

    return cam_intrinsic 

class mData(data.Dataset):
    def __init__(self,   training, img_seq_paths, dmap_seq_paths, cam_pose_seq_paths, intrin_path, 
                 img_size = [384,256], digitize = False, d_candi = None, if_process = True, resize_dmap = 1):
        r'''
        inputs:

        img_seq_paths - The file paths for input images 

        intrin_path - The path to the .mat file having those variables:
                      IntM - the 3 x 3 intrinsic matrix for the camera
                      img_size - a 1x2 matrix for the image: [width, height]

        d_candi - the candidate depths 

        if_process - if pre_processe the image, default is True

        '''

        self.img_seq_paths = img_seq_paths 
        self.cam_pose_seq_paths = cam_pose_seq_paths 
        self.to_gray = tfv_transform.Compose( [tfv_transform.Grayscale(), tfv_transform.ToTensor()]) 
        self.d_candi = d_candi 
        self.img_size = img_size # the input image size (used for resizing the input images) 
        self.if_preprocess = if_process 
        # initialization about the camera intrsinsics, which is the same for all data #
        self.resize_dmap = resize_dmap 

        cam_intrinsics = self.get_cam_intrinsics(intrin_path)
        self.cam_intrinsics = cam_intrinsics 

    def get_cam_intrinsics(self,intrin_path,img_size = None):
        r'''
        Get the camera intrinsics 
        '''

        if img_size is not None:
            width  = img_size[0]
            height = img_size[1]
        else:
            width = int(float( self.img_size[0] * self.resize_dmap))
            height = int(float( self.img_size[1] * self.resize_dmap)) 

        cam_intrinsics = read_IntM_from_mat( intrin_path, out_size = [width, height]) 
        self.cam_intrinsics = cam_intrinsics 
        return cam_intrinsics

    def return_cam_intrinsics(self):
        return self.cam_intrinsics

    def __getitem__(self, indx):
        r'''
        outputs:
        {'img': img.unsqueeze_(0), 'img_gray': img_gray.unsqueeze_(0), 'scene_path': scene_path, 'img_path': img_path }

        ''' 
        img_path = self.img_seq_paths[indx]
        proc_normalize = m_preprocess.get_transform()
        proc_totensor = m_preprocess.to_tensor()
        img = read_img(img_path , no_process = True)[0]
        img = img.resize(self.img_size, PIL.Image.NEAREST) 
        img_gray = self.to_gray(img) 

        if self.if_preprocess:
            img = proc_normalize(img)
        else:
            proc_totensor = tfv_transform.ToTensor()
            img = proc_totensor(img) 

        # image path #
        scene_path = os.path.split(img_path)[0]
        return { 'img': img.unsqueeze_(0), 'img_gray': img_gray.unsqueeze_(0), 'scene_path': scene_path, 'img_path': img_path, }

    def __len__(self):
        return len(self.img_seq_paths)

    def set_paths(self, img_seq_paths, dummy0=None, dummy1=None ):
        self.img_seq_paths = img_seq_paths 

