'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''


import numpy as np 
import imageio 
import scipy.misc as smisc
import matplotlib.pyplot as plt
import mutils.misc as misc 

def outupt_video2images(vidpath, img_path, frame_interval = None, if_auto_transpose = True):
    '''
    Output the video frames into images 
    Inputs: 
    vidpath  - the path to the input video 
    img_path - the path to the output image folder 
    frame_interval (optional) - the frame interval 
    '''

    misc.m_makedir(img_path)
    Frame_array = readVideo( vidpath) 
    for idx_frame, frame in enumerate( Frame_array):
        if frame_interval is not None:
            if idx_frame % frame_interval ==0:
                print('saving frame %d'%(idx_frame))
                frame = np.asarray(frame)
                if if_auto_transpose is True:
                    if frame.shape[0] > frame.shape[1]:
                        frame = frame.transpose([1,0,2])
                plt.imsave('%s/%06d.png'%( img_path, idx_frame ), arr= frame)
        else:
            print('saving frame %d'%(idx_frame))
            frame = np.asarray(frame)
            if if_auto_transpose is True:
                if frame.shape[0] > frame.shape[1]:
                    frame = frame.transpose([1,0,2])
            plt.imsave('%s/%06d.png'%( img_path, idx_frame ), arr= frame)

    return 1

def cat_videos(vid_file_array, out_vid_path,fps=15, dim=0):
    '''
    concatenate videos
    vid_file_array - video file name array
    '''
    nvid = len(vid_file_array)
    VIDs_array = []
    for ivid in range(nvid):
        vid_file = vid_file_array[ivid]
        vid_array = readVideo(vid_file)
        VIDs_array.append(vid_array)
    nframes = len(vid_array)

    for vid_array in VIDs_array:
        assert len(vid_array) == nframes, 'videos should have the same # of frames'
    out_vid_array = []

    for iframe in range(nframes):
        frame = []
        for ivid in range(nvid):
            vid_frame = VIDs_array[ivid][iframe]
            frame.append(vid_frame)
        if dim==0:
            frame = np.hstack(frame)
        elif dim==1:
            frame = np.vstack(frame)
        out_vid_array.append(frame)
    writeVideo(out_vid_array, out_vid_path, fps=fps)

def sceneNet_imgs2video(img_fldr, output_video_path = 'input.avi', fps=15):
    '''
    For visualization purpose: re-write the images in one trajector in the
    SceneNet dataset into a video 
    '''
    nframes = 300 
    import PIL.Image as image
    frames = []
    for iframe in range(nframes):
        print('appending frame %d'%(iframe))
        fname_path = '%s/%d.jpg'%(img_fldr, iframe * 25)
        frame = image.open(fname_path)
        frames.append( np.asarray(frame))
    print('writing the video ..')
    writeVideo(frames, output_video_path, fps = fps)

def re_write_video_from_img_fldr(image_fldr, img_name_base, indx_min_max, video_path, fps=15):
    '''
    Re-write the images in a fldr to video
    '''

    res_frames = []
    import PIL.Image as image
    for indx in range( indx_min_max[0], indx_min_max[1]+1):
        img = image.open('%s/%s%05d.png'%( image_fldr, img_name_base, indx))
        res_frames.append(np.asarray(img))
    writeVideo(res_frames, '%s'%(video_path), fps=fps)

def re_write_video_from_img_res(res_base_folder, output_base_folder, indx_min_max, output_file_name = 'res.avi'):
    '''
    Re-write the image results to video results
    The image results are saved like this:
    ${res_base_folder}/dmap/filt/dmap_#_filt.png
    ${res_base_folder}/dmap/conf/conf_map_#.png
    ${res_base_folder}/dmap/raw/dmap_#.png
    ${res_base_folder}/imgs/ref_img_#.png 
    Inputs:
    indx_min_max - (indx_min, indx_max)
    '''
    res_frames = []
    for indx in range( indx_min_max[0], indx_min_max[1]+1):
        print('reading frame %d of %d'%(indx, indx_min_max[1]+1))
        img_conf = image.open('%s/dmap/conf/conf_map_%d.png'%(res_base_folder, indx))
        width, height = img_conf.size
        img_ref = image.open('%s/imgs/ref_img_%d.png'%(res_base_folder, indx)).resize([width, height])
        img_results = image.open('%s/dmap/filt/dmap_%d_filt.png'%(res_base_folder, indx))
        image_hconcate = _hconcat_PIL_imgs([img_ref,img_conf ,img_results])
        res_frames.append(np.asarray(image_hconcate)) 
    writeVideo(res_frames, '%s/%s'%(output_base_folder, output_file_name), fps=15)

def re_write_video_from_img_res_1(res_base_folder, output_base_folder, indx_min_max, 
        output_file_name = 'res.avi', img_prefix = None, img_subfix = None, per_img_size = None, col_idx = None, max_v = None, fmt = None):
    '''
    Re-write the image results to video results
    The image results are saved like this:
    ${res_base_folder}/res_%05d.png
    Inputs:
    indx_min_max - (indx_min, indx_max)
    '''
    import PIL.Image as image
    res_frames = []
    for indx in range( indx_min_max[0], indx_min_max[1]+1):
        print('reading frame %d of %d'%(indx, indx_min_max[1]+1))
        
        if img_prefix is None:
            img_res = image.open('%s/res_%05d.png'%(res_base_folder, indx))

            if per_img_size is not None and col_idx is not None:
                img_res = misc.sub_res_img('%s/res_%05d.png'%(res_base_folder, indx), per_img_size, col_idx ) 

        else:
            if img_subfix is None:
                img_res = image.open('%s/%s%05d.png'%(res_base_folder, img_prefix, indx))
            else:

                if fmt is not None:
                    img_res = image.open('%s/%s%05d%s.%s'%(res_base_folder, img_prefix, indx, img_subfix, fmt))
                    if fmt=='pgm':
                        img_res = np.asarray(img_res).astype(np.float) / 256.  
                else:
                    img_res = image.open('%s/%s%05d%s.png'%(res_base_folder, img_prefix, indx, img_subfix))

        if np.asarray( img_res).ndim > 2:
            res_frames.append(np.asarray( img_res )[:,:, :3]) 
        else:
            res_frames.append(np.asarray( img_res )) 

    if max_v is None:
        writeVideo(res_frames, '%s/%s'%(output_base_folder, output_file_name), fps=15, )
    else:
        writeVideo(res_frames, '%s/%s'%(output_base_folder, output_file_name), fps=15, norm_m=0, max_v = max_v)

    avi_fldr = '%s/%s'%(output_base_folder, output_file_name)
    print('res writing to %s'%(avi_fldr))

def readVideo(vidPath):
    '''
    output: vidArray - array of video frames
    '''
    reader = imageio.get_reader(vidPath)
    vidArray = []
    for i, im in enumerate(reader):
        vidArray.append(im)
    return vidArray 

def writeVideo(vidArray, vidPath, fps=30, max_v=None, norm_m = None, frame_sz=None):
    '''
    inputs: vidArray - array of video frames in float [frame0, frame1, ...]
            vidPath - the path of the output video
            fps - frame rate
            norm_m - 0: normalize intensity over all frames 
                     1: normalize intensity over single frame
    '''

    writer = imageio.get_writer(vidPath, fps = fps)
    for im in vidArray:
        if norm_m ==0:
            im_norm = im / max_v
            im_norm[im_norm > 1]=1
            im_w = np.uint8( im_norm * 255.0)
        elif norm_m ==1:
            max_v_frame = im.max()
            im_w = np.uint8( im / max_v_frame * 255.0)
        elif norm_m is None:
            im_w = np.uint8(im)
        
        if frame_sz is not None:
            im_w = smisc.imresize(im_w, (frame_sz[0], frame_sz[1]))

        writer.append_data(im_w)

    writer.close()
