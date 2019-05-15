'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''
# Dataloader for scanNet dataset # 
import numpy as np
import os 
import math
import sys

import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as tfv_transform

import mdataloader.m_preprocess as m_preprocess
import mdataloader.misc as mloader_misc
import warping.View as View 

def _read_split_file( filepath):
    '''
    Read data split txt file provided for Robust Vision
    ''' 
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [ x.strip() for x in trajs ]
    return trajs 

def get_paths_1frame(traj_indx, database_path_base = '/nfs/datasets/scan-net/1-frame', split_txt = None ):

    '''
    Return the training info for one trajectory

    Inputs:
    - traj_indx : the index (in the globbed array) of the trajectory
    - (optional) database_path_base : the path for the scanNet dataset

    Outputs:
    - fldr_path : the root folder for the path
    - img_paths : array of paths for the input images
    - dmap_paths: like img_pahths
    - pose_paths : array of paths for the poses txt files  
    '''

    assert traj_indx >=0 and traj_indx <= 1512, 'traj_indx should be from 0 to 1512'

    import glob
    if split_txt is None:
        traj_paths = glob.glob( database_path_base + '/scene*' ) 
        traj_paths = sorted(traj_paths) 
    else:
        scene_names = _read_split_file(split_txt) 
        traj_paths = [ database_path_base + '/' + scene_name for scene_name in scene_names]

    img_fldr = traj_paths[traj_indx]

    nimg = len( glob.glob('%s/*.jpg'%(img_fldr)) )
    img_paths = []
    dmap_paths = []
    pose_paths = []

    nimg_cnt = nimg
    for i_img in range(nimg_cnt):
        img_path  = '%s/frame-%06d.color.jpg'%( img_fldr, i_img)
        dmap_path = '%s/frame-%06d.depth.pgm'%( img_fldr, i_img)
        pose_path = '%s/frame-%06d.pose.txt'%( img_fldr, i_img)
        img_paths.append(img_path)
        dmap_paths.append(dmap_path)
        pose_paths.append(pose_path)

    fldr_path = img_fldr
    intrinsic_path = fldr_path + '/_info.txt'
    return fldr_path, img_paths, dmap_paths, pose_paths, intrinsic_path

def get_paths_from_folder( img_fldr ):

    '''
    Get the paths for the input images, depth maps and camera poses

    Inputs:
    - img_fldr

    Outputs:
    - data_path: 
    - img_paths : array of paths for the input images
    - dmap_paths: like img_pahths
    - pose_paths : array of paths for the poses txt files  
    - intrinsic_path: the txt file including the camera intrinsic info
    ''' 
    nimg = len( glob.glob('%s/*.jpg'%(img_fldr)) )
    img_paths = []
    dmap_paths = []
    pose_paths = [] 
    for i_img in range( nimg  ):
        img_path  = '%s/frame-%06d.color.jpg'%( img_fldr, i_img)
        dmap_path = '%s/frame-%06d.depth.pgm'%( img_fldr, i_img)
        pose_path = '%s/frame-%06d.pose.txt'%( img_fldr, i_img)
        img_paths.append(img_path)
        dmap_paths.append(dmap_path)
        pose_paths.append(pose_path) 
    fldr_path = img_fldr
    intrinsic_path = fldr_path + '/_info.txt' 

    return fldr_path, img_paths, dmap_paths, pose_paths, intrinsic_path

def get_paths(traj_indx, frame_interv, 
              database_path_base = '/datasets/scan-net-5-frame', 
              split_txt = None):
    '''

    Return the training info for one trajectory 
    NOTE: The scan-net data should be of 5-interv

    Inputs:
    - db_path : the path to the database 
    - frame_interv : step of frame index
    - traj_indx : the index (in the globbed array) of the trajectory
    - (optional) database_path_base : the path for the scanNet dataset
    - split_txt : the split txt file path, including the name of trajectories for trianing/testing/validation

    Outputs:
    - fldr_path : the root folder for the path
    - img_paths : array of paths for the input images
    - dmap_paths: like img_pahths
    - pose_paths : array of paths for the poses txt files  

    ''' 
    assert frame_interv % 5 ==0 and frame_interv > 0, 'frame_interv should be multiple of 5'
    assert traj_indx >=0 and traj_indx <= 1512, 'traj_indx should be from 0 to 1512' 

    import glob 
    if split_txt is None:
        traj_paths = glob.glob( database_path_base + '/scene*' ) 
        traj_paths = sorted(traj_paths) 
    else:
        scene_names = _read_split_file(split_txt) 
        traj_paths = [ database_path_base + '/' + scene_name for scene_name in scene_names]

    img_fldr = traj_paths[traj_indx]
    nimg = len( glob.glob('%s/*.jpg'%(img_fldr)) )
    img_paths = []
    dmap_paths = []
    pose_paths = [] 
    nimg_cnt = int( (nimg-1) / ( frame_interv / 5) ) + 1

    for i_img in range(nimg_cnt):
        img_path  = '%s/frame-%06d.color.jpg'%( img_fldr, i_img * frame_interv)
        dmap_path = '%s/frame-%06d.depth.pgm'%( img_fldr, i_img * frame_interv)
        pose_path = '%s/frame-%06d.pose.txt'%( img_fldr, i_img * frame_interv)
        img_paths.append(img_path)
        dmap_paths.append(dmap_path)
        pose_paths.append(pose_path)

    fldr_path = img_fldr
    intrinsic_path = fldr_path + '/_info.txt' 

    if split_txt is not None:
        fldr_path = len(scene_names )# #of trajs in the split

    return fldr_path, img_paths, dmap_paths, pose_paths, intrinsic_path
        

def read_img(path, img_size = None, no_process= False, only_resize = False):
    '''
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
    return img,  raw_sz

def read_ExtM_from_txt(fpath_txt):
    '''
    Read the external matrix from txt file. 
    The txt file for the exterminal matrix is got from 
    the sens file loader 
    '''
    ExtM = np.eye(4)
    with open(fpath_txt, 'r') as f:
        content = f.readlines()
    content = [ x.strip() for x in content]
    
    for ir, row in enumerate(ExtM):
        row_content = content[ir].split()
        row = np.asarray([ float(x) for x in row_content ])
        ExtM[ir, :] = row
    ExtM = np.linalg.inv(ExtM)
    return ExtM

def read_IntM_from_txt(fpath_txt, out_size=None):
    '''
    Read the intrinsic matrix from the txt file 
    The txt file for the exterminal matrix is got from 
    the sens file loader 
    input: out_size [width, height]
    Output:
    cam_intrinsic - The cam_intrinsic structure, used in warping.homography:
        cam_intrinsic includes: {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
        hfov, vfov
        fovs in horzontal and vertical directions (degrees)
        unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
        unit ray pointing from the camera center to the pixel
    '''
    IntM = np.zeros((4,4))
    with open(fpath_txt, 'r') as f:
        content = f.readlines()

    contents = [ x.strip() for x in content]

    assert contents[2].split('=')[0].strip() == 'm_colorWidth',\
            'un-recogonized _info.txt format '
    width = int( contents[2].split('=')[1].strip())

    assert contents[3].split('=')[0].strip() == 'm_colorHeight',\
            'un-recogonized _info.txt format '
    height = int( contents[3].split('=')[1].strip())

    assert contents[7].split('=')[0].strip() == 'm_calibrationColorIntrinsic',\
            'un-recogonized _info.txt format '

    color_intrinsic_vec = contents[7].split('=')[1].strip().split()
    color_intrinsic_vec = [float(x) for x in color_intrinsic_vec]
    IntM = np.reshape(np.asarray(color_intrinsic_vec), (4,4))
    IntM = IntM[:3, :]
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

class ScanNet_dataset(data.Dataset):
    def __init__(self, training,
            img_seq_paths, dmap_seq_paths, cam_pose_seq_paths,
            intrin_path, img_size = [384, 256], 
            digitize = False,
            d_candi = None,
            resize_dmap = None, 
            if_process = True):

        '''
        inputs:
        traning: if for training or not 

        img_seq_paths, dmap_seq_paths, cam_pose_seq_paths - the file paths for
        input images, depth maps and camera poses

        For example, {img_seq_paths[i], dmap_seq_paths[i],
        cam_pose_seq_paths[i]} are the paths of the info for the i-th
        measurement

        The pose information is saved into the scanNet pose format: .txt file
        including the 4x4 extrinsic matrix    

        d_candi - the candidate depths 
        digitize (Optional; False) - if digitize the depth map using d_candi

        resize_dmap - the scale for re-scaling the dmap 
        the GT dmap size would be self.img_size * resize_dmap
        '''

        assert len(img_seq_paths) == len(dmap_seq_paths) == len(cam_pose_seq_paths ),\
                'The file pathes for the input images, \
            depth maps and the camera poses should have the same length'

        self.img_seq_paths = img_seq_paths 
        self.dmap_seq_paths = dmap_seq_paths 
        self.cam_pose_seq_paths = cam_pose_seq_paths 
        self.training = training
        self.to_gray = tfv_transform.Compose( [tfv_transform.Grayscale(),
                                               tfv_transform.ToTensor()])

        self.d_candi = d_candi
        self.digitize = digitize 
        if digitize:
            self.label_min = 0
            self.label_max = len(d_candi) - 1

        self.resize_dmap = resize_dmap
        self.img_size = img_size # the input image size (used for resizing the input images)

        self.if_preprocess = if_process

        # usample in the d dimension #
        self.dup4_candi = np.linspace( self.d_candi.min(), self.d_candi.max(), len(self.d_candi) * 4) 
        self.dup4_label_min = 0
        self.dup4_label_max = len(self.dup4_candi) -1
        ##

        # initialization about the camera intrsinsics, which is the same for all data #
        cam_intrinsics = self.get_cam_intrinsics(intrin_path)
        self.cam_intrinsics = cam_intrinsics 

    def get_cam_intrinsics(self,intrin_path,img_size = None):
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

        cam_intrinsics = read_IntM_from_txt( intrin_path, out_size = [width, height])
        self.cam_intrinsics = cam_intrinsics 
        return cam_intrinsics

    def return_cam_intrinsics(self):
        return self.cam_intrinsics

    def __getitem__(self, indx):
        '''
        outputs:
        img, dmap, extM, scene_path , as entries in a dic.
        ''' 
        img_path = self.img_seq_paths[indx]
        dmap_path = self.dmap_seq_paths[indx]
        cam_pose_path = self.cam_pose_seq_paths[indx]
        proc_normalize = m_preprocess.get_transform()
        proc_totensor = m_preprocess.to_tensor()
        img = read_img(img_path , no_process = True)[0]
        img = img.resize(self.img_size, PIL.Image.NEAREST) 
        img_gray = self.to_gray(img) 

        dmap_raw = read_img(dmap_path, no_process = True)[0]
        dmap_mask_imgsize = np.asarray(dmap_raw) < 0.01
        dmap_mask_imgsize = PIL.Image.fromarray(
                dmap_mask_imgsize.astype(np.uint8) * 255).resize([self.img_size[0], self.img_size[1]], PIL.Image.NEAREST )

        if self.resize_dmap is not None:
            dmap_imgsize = dmap_raw.resize([self.img_size[0], self.img_size[1]], PIL.Image.NEAREST)
            dmap_imgsize = proc_totensor(dmap_imgsize)[0, :, :].float() * .001
            dmap_rawsize = proc_totensor(dmap_raw)[0,:,:].float() * .001

            # resize the depth map #
            dmap_size = [
                    int(float(self.img_size[0]) * self.resize_dmap), 
                    int(float(self.img_size[1]) * self.resize_dmap)]

            dmap_raw_bilinear_dw = dmap_raw.resize(dmap_size, PIL.Image.BILINEAR) 
            dmap_raw = dmap_raw.resize(dmap_size, PIL.Image.NEAREST)
            dmap_mask = dmap_mask_imgsize.resize(dmap_size, PIL.Image.NEAREST)

        dmap_raw = proc_totensor(dmap_raw)[0,:,:] # single-channel for depth map
        dmap_raw = dmap_raw.float() * .001 # scale to meter

        dmap_raw_bilinear_dw = proc_totensor(dmap_raw_bilinear_dw)[0,:,:]
        dmap_raw_bilinear_dw = dmap_raw_bilinear_dw.float() * .001

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
        else:
            proc_totensor = tfv_transform.ToTensor()
            img = proc_totensor(img) 

        # extrinsics #
        extM = read_ExtM_from_txt(cam_pose_path)
        # image path #
        scene_path = os.path.split(img_path)[0]

        return {'img': img.unsqueeze_(0), 'dmap': dmap.unsqueeze_(0), 
                'dmap_raw':dmap_raw.unsqueeze_(0), 
                'dmap_raw_bilinear_dw':dmap_raw_bilinear_dw.unsqueeze_(0), 
                'dmap_rawsize': dmap_rawsize.unsqueeze_(0),
                'dmap_imgsize': dmap_imgsize.unsqueeze_(0),
                'dmap_imgsize_digit': dmap_imgsize_digit.unsqueeze_(0),
                'dmap_up4_imgsize_digit': dmap_up4_imgsize_digit.unsqueeze_(0),
                'img_gray': img_gray.unsqueeze_(0),
                'dmap_mask': dmap_mask.unsqueeze_(0).type_as(dmap), 
                'dmap_mask_imgsize': dmap_mask_imgsize.unsqueeze_(0).type_as(dmap), 
                'extM': extM,
                'scene_path': scene_path, 
                'img_path': img_path, }

    def __len__(self):
        return len(self.img_seq_paths)

    def set_paths(self, img_seq_paths, dmap_seq_paths, cam_pose_seq_paths):
        self.img_seq_paths = img_seq_paths 
        self.dmap_seq_paths = dmap_seq_paths
        self.cam_pose_seq_paths = cam_pose_seq_paths 
