'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''


import shutil
import glob

import DSO.dso_io as dso_io
import mutils.misc as m_misc 
from PIL import Image
import scipy.io as sio

def main():

    import argparse 
    print('Parsing arguments ...')
    parser = argparse.ArgumentParser()

    parser.add_argument('--dso_path', required =True, type=str, help='The path to DSO ') 
    parser.add_argument('--data_fldr', required =True, type=str, help='The path to the image folder, where .png or .jpg image files are saved') 
    parser.add_argument('--cam_info_file', required =True, type=str, help='The path to the .mat file saving the camera info') 
    parser.add_argument('--name_pattern', required=True, type=str, help='The name pattern for the image. e.g. *.color.* ')

    # The temp image folder is needed since DSO assumes all images files in one
    # folder are input images. But some datasets include images of multiple
    # types(rgb, depth etc) in one folder. So we will naively copy the input
    # images into the tmp_img_fldr
    parser.add_argument('--temp_img_fldr', required = False, type=str, 
                        default = './dso_imgs', help='The path to the temporary image folder') 
    parser.add_argument( '--res_path', required = False, type=str, 
                        default = './dso_res', help='The path to the DSO output (a txt file containing the camera poses)' ) 
    parser.add_argument('--minframe', required=False, type=int, default = 0, help='starting frame idx')
    parser.add_argument('--maxframe', required=False, type=int, default = 100, help='ending frame idx') 

    args = parser.parse_args()

    minframe = args.minframe
    maxframe = args.maxframe

    res_fldr = args.res_path
    cam_info = sio.loadmat(args.cam_info_file)
    intrinsic_info = { 'IntM': cam_info['IntM'],  } 

    # create temp folder #
    print('copying the images..')
    img_fldr_path = str(args.temp_img_fldr)
    m_misc.m_makedir(img_fldr_path)
    files = sorted( glob.glob('%s/%s'%( args.data_fldr, args.name_pattern)) )
    m_misc.m_makedir( res_fldr ) 

    for f,i in zip(files, range(0, args.maxframe)):
        shutil.copy(f, img_fldr_path) # move image files into the temp folder# 

    # run DSO #
    im = Image.open( files[0] )
    intrinsic_info['img_size'] = im.size
     
    Rts_cam_to_world = dso_io.run_DSO(img_fldr_path = img_fldr_path, 
                                      dso_bin_path = args.dso_path,
                                      intrinsic_info = intrinsic_info, 
                                      result_path = '%s/result_dso.txt'%( res_fldr ) ,
                                      vig_img_path = './DSO/vignette.png',
                                      min_frame= minframe, max_frame= maxframe,
                                      mode = 1, preset = 2, nogui = 1, use_existing = False ) 

    # delete temp fldr #
    shutil.rmtree( img_fldr_path ) 
    
    print('\n\n# of frames with pose estimated: %d'%(len(Rts_cam_to_world)))
    print('# of frames in the video: %d'%(len(files)))

if __name__ == '__main__':
    main() 
