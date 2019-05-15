'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
''' 

import numpy as np 
import math

'''
------
Functions using camera intrinsics 
Useful for back-projecting pixels into 3D space.
'''
def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    '''
    Inputs:
    pixel- pixel index (icol, irow)
    hfov, vfov - the field of views in the horizontal and vertical directions (in degrees)
    pixel_width, pixel_height - the # of pixels in the horizontal/vertical directions
    Outputs:
    (x, y, 1) - The coordinate location in 3D. The origin of the coordinate in 3D is the camera center.
    So given the depth d, the backprojected 3D point is simply: d * (x,y,1).
    Or if we have the ray distance d_ray, the backprojected 3D point is simply d_ray * (x,y,1) / norm_L2((x,y,z))
    '''
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ( (2.0 * ( (x+0.5)/pixel_width )  ) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ( (2.0 * ( (y+0.5)/pixel_height ) ) - 1.0)
    return (x_vect,y_vect,1.0)

def normalised_pixel_to_ray_array(width=320,height=240, hfov = 60, vfov = 45, normalize_z=True):
    '''
    Given the FOV (estimated from the intrinsic matrix for example), 
    get the unit ray vectors pointing from the camera center to the pixels

    Inputs: 
    width, height - The width and height of the image (in pixels)
    hfov, vfov - The field of views (in degree) in the horizontal and vertical directions
    normalize_z - 

    Outputs:
    pixel_to_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to 
                         the unit ray pointing from the camera center to the pixel
    '''
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            if normalize_z:
                # z=1 #
                pixel_to_ray_array[y,x] =np.array(\
                        pixel_to_ray( (x,y),
                        pixel_height=height,pixel_width=width,
                        hfov= hfov, vfov= vfov))
            else:
                # length=1 #
                pixel_to_ray_array[y,x] = normalize(np.array(\
                        pixel_to_ray( (x,y),
                        pixel_height=height,pixel_width=width,
                        hfov= hfov, vfov= vfov)))

    return pixel_to_ray_array

def normalize(v):
    return v/np.linalg.norm(v)

    return v

'''
------
'''

class View:
    img = None      

    def __init__(self, R, t, img=None):
        '''
        input: 
        R - The 3x3 rotation matrix 
        t - The length 3 translation vector 
        (img) - the image attached with the current view
        '''
        if img is not None: 
            # The feature image
            self.img = np.copy( img )
        if img_rgb is not None: 
            # The rgb image #
            assert np.ndim(img_rgb) ==3, 'View.img_rgb should be 3-ch image'
            self.img_rgb = img_rgb 

        self.R = np.copy(R) 
        self.T = np.zeros((3,3))
        self.T[:,2] = np.copy(t)
        
    def setRt( self, R, t):
        '''
        set R, t for the current view
        R - The 3x3 rotation matrix 
        t - The length 3 translation vector 
        '''
        self.R = R.copy()
        self.T[:,2] = t.copy()
