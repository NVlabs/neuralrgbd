'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import numpy as np
import math

import matplotlib as mlt
mlt.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.models as utils_model
import mutils.misc as m_misc 

import scipy.io as sio
import mio.imgIO as imgIO

import PIL.Image as image

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

def cat_imgs(img_names, output_name ):
    imgs = [np.array( image.open(imgname)) for imgname in img_names]
    imgs = np.hstack(imgs)
    plt.imsave(output_name, imgs)


def depth_regression(Depth_Indx_vol, BV):
    '''
    Depth regression
    '''
    return torch.sum((torch.exp(BV.detach()) * Depth_Indx_vol).squeeze(), dim=0).squeeze().cpu().numpy()

def export_res_img( ref_dat, BV_measure, d_candi, resfldr, batch_idx,
                    depth_scale = 1000, conf_scale = 1000):


    # depth map #
    nDepth = len(d_candi)
    dmap_height, dmap_width = BV_measure.shape[2], BV_measure.shape[3] 
    Depth_val_vol = torch.ones(1, nDepth,  dmap_height, dmap_width).cuda()

    for idepth in range(nDepth):
        Depth_val_vol[0, idepth, ...] = Depth_val_vol[0, idepth, ...] * d_candi[idepth]
    dmap_th = depth_regression(Depth_val_vol, BV_measure)
    dmap = torch.FloatTensor(dmap_th).cpu().numpy()

    # confMap #
    confMap_log, _ = torch.max(BV_measure, dim=1)
    confMap_log = torch.exp(confMap_log.squeeze().cpu())
    confMap_log = confMap_log.cpu().numpy()
    confmap = torch.FloatTensor(confMap_log).unsqueeze(0).unsqueeze(0).cuda() 
    confmap = confmap.squeeze().cpu().numpy()
    img = ref_dat['img']
    img = img.squeeze().cpu().permute(1,2,0).numpy()
    img_in_png = _un_normalize( img ); img_in_png = (img_in_png * 255).astype(np.uint8)

    # write to path #
    m_misc.m_makedir(resfldr)
    img_path = '%s/img_%05d.png'%(resfldr, batch_idx)
    d_path = '%s/d_%05d.pgm'%(resfldr, batch_idx)
    conf_path = '%s/conf_%05d.pgm'%(resfldr, batch_idx)

    plt.imsave(img_path, img_in_png)
    imgIO.export2pgm( d_path,    (dmap * depth_scale ).astype(np.uint16) )
    imgIO.export2pgm( conf_path, (confmap * conf_scale ).astype(np.uint16) )


def export_res_refineNet(ref_dat, BV_measure, d_candi,  res_fldr, batch_idx, diff_vrange_ratio=4, 
        cam_pose = None, cam_intrinM = None, output_pngs = False, save_mat=True, output_dmap_ref=True):
    '''
    export results
    ''' 

    # img_in #
    img_up = ref_dat['img']
    img_in_raw = img_up.squeeze().cpu().permute(1,2,0).numpy()
    img_in = (img_in_raw - img_in_raw.min()) / (img_in_raw.max()-img_in_raw.min()) * 255.

    # confMap #
    confMap_log, _ = torch.max(BV_measure, dim=1)
    confMap_log = torch.exp(confMap_log.squeeze().cpu())
    confMap_log = confMap_log.cpu().numpy()

    # depth map #
    nDepth = len(d_candi)
    dmap_height, dmap_width = BV_measure.shape[2], BV_measure.shape[3] 
    dmap = m_misc.depth_val_regression(BV_measure, d_candi, BV_log = True).squeeze().cpu().numpy() 

    # save up-sampeled results #
    resfldr = res_fldr 
    m_misc.m_makedir(resfldr)

    img_up_path ='%s/input.png'%(resfldr,)
    conf_up_path = '%s/conf.png'%(resfldr,)
    dmap_raw_path = '%s/dmap_raw.png'%(resfldr,)
    final_res_up = '%s/res_%05d.png'%(resfldr, batch_idx) 

    if output_dmap_ref: # output GT depth
        ref_up = '%s/dmap_ref.png'%(resfldr,)
        res_up_diff = '%s/dmaps_diff.png'%(resfldr,)
        dmap_ref = ref_dat['dmap_imgsize']
        dmap_ref = dmap_ref.squeeze().cpu().numpy() 
        mask_dmap = (dmap_ref > 0 ).astype(np.float)
        dmap_diff_raw = np.abs(dmap_ref - dmap ) * mask_dmap
        dmaps_diff = dmap_diff_raw 
        plt.imsave(res_up_diff, dmaps_diff, vmin=0, vmax=d_candi.max()/ diff_vrange_ratio )
        plt.imsave(ref_up, dmap_ref, vmax= d_candi.max(), vmin=0, cmap='gray')

    plt.imsave(conf_up_path, confMap_log, vmin=0, vmax=1, cmap='jet')
    plt.imsave(dmap_raw_path, dmap, vmin=0., vmax =d_candi.max(), cmap='gray' )
    plt.imsave(img_up_path, img_in.astype(np.uint8))

    # output the depth as .mat files # 
    fname_mat = '%s/depth_%05d.mat'%(resfldr, batch_idx)
    img_path = ref_dat['img_path'] 
    if save_mat:
        if not output_dmap_ref:
            mdict = { 'dmap': dmap, 'img': img_in_raw, 'confMap': confMap_log, 'img_path': img_path}
        elif cam_pose is None:
            mdict = {'dmap_ref': dmap_ref, 'dmap': dmap, 'img': img_in_raw, 'confMap': confMap_log,
                    'img_path':   img_path}
        else:
            mdict = {'dmap_ref': dmap_ref, 'dmap': dmap, 
                    'img': img_in_raw, 'cam_pose': cam_pose, 
                    'confMap':confMap_log, 'cam_intrinM': cam_intrinM, 
                    'img_path': img_path } 
        sio.savemat(fname_mat, mdict) 

    print('export to %s'%(final_res_up))
    
    if output_dmap_ref:
        cat_imgs((img_up_path, conf_up_path, dmap_raw_path, res_up_diff, ref_up), final_res_up) 
    else:
        cat_imgs((img_up_path, conf_up_path, dmap_raw_path), final_res_up) 

    if output_pngs:
        import cv2
        png_fldr = '%s/output_pngs'%(res_fldr, )
        m_misc.m_makedir( png_fldr ) 
        depth_png = (dmap * 1000 ).astype(np.uint16)
        img_in_png = _un_normalize( img_in_raw ); img_in_png = (img_in_png * 255).astype(np.uint8)
        confMap_png = (confMap_log*255).astype(np.uint8) 
        cv2.imwrite( '%s/d_%05d.png'%(png_fldr, batch_idx), depth_png)
        cv2.imwrite( '%s/rgb_%05d.png'%(png_fldr, batch_idx), img_in_png)
        cv2.imwrite( '%s/conf_%05d.png'%(png_fldr, batch_idx), confMap_png)

        if output_dmap_ref:
            depth_ref_png = (dmap_ref * 1000).astype(np.uint16) 
            cv2.imwrite( '%s/dref_%05d.png'%(png_fldr, batch_idx), depth_ref_png)

def _un_normalize( img_in ):
    img_out = np.zeros( img_in.shape )
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] +=  __imagenet_stats['mean'][ich]

    return img_out


