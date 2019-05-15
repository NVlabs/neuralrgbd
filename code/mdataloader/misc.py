'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# The utility functions for m_dataloader #
import numpy as np
import math
import torch 

def dMap_to_indxMap(dmap, d_candi):
    '''
    Convert the depth map to the depthIndx map. This is useful if we want to NLL loss to train NN
    
    Inputs:
    dmap - depth map in 2D. It is a torch.tensor or numpy array
    d_candi - the candidate depths. The output indexMap has the property:
    dmap[i, j] = d_candi[indxMap[i, j]] (not strictly equal, but the depth value in dmap lies
    in the bins defined by d_candi)

    Outputs:
    indxMap - the depth index map. A 2D ndarray
    '''
    assert isinstance(dmap, np.ndarray) or isinstance(dmap, torch.Tensor),\
    'dmap should be a tensor/ndarray'

    if isinstance(dmap, np.ndarray):
        dmap_ = dmap 
    else:
        dmap_ = dmap.cpu().numpy()

    indxMap = np.digitize(dmap_, d_candi)

    return indxMap 
    

def read_ExtM_from_txt(fpath_txt, if_inv = True):
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

    if if_inv:
        ExtM = np.linalg.inv(ExtM)

    return ExtM

def write_ExtM_to_txt(fpath_txt, extM, fmt= '%.6f'):
    '''
    write the camera extrinsic matrix to .txt file
    This function is useful to format some dataset such as SceneNet dataset
    Inputs:
    fpath_txt - the .txt file path 
    extM - the np array (4 x 4) for an extrinsic matrix 
    fmt (optional ) - the format for the numbers 
    '''
    assert isinstance( extM, np.ndarray), 'Input extM should be a np array'
    np.savetxt(fpath_txt, extM, fmt =fmt )

def read_IntM_from_txt(fpath_txt):
    '''
    Read the intrinsic matrix from the txt file 
    The txt file for the exterminal matrix is got from 
    the sens file loader 
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
    h_fov = math.degrees(math.atan(IntM[0, 2] / IntM[0, 0]) * 2)
    v_fov = math.degrees(math.atan(IntM[1, 2] / IntM[1, 1]) * 2)

    pixel_to_ray_array = View.normalised_pixel_to_ray_array(\
            width= width, height= height, hfov = h_fov, vfov = v_fov)

    cam_intrinsic = {\
            'hfov': h_fov,
            'vfov': v_fov,
            'unit_ray_array': pixel_to_ray_array,
            'intrinsic_M': IntM}  
    return cam_intrinsic 
