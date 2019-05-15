'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# train both D-net and KV-net and R-net #
import torch
torch.backends.cudnn.benchmark=True 

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader 

import mdataloader.misc as dl_misc
import mdataloader.batch_loader as batch_loader
import mdataloader.scanNet as dl_scanNet

import mutils.misc as m_misc 
import warping.homography as warp_homo
import numpy as np
import math 

import models.basic as m_basic
import models.KVNET as m_kvnet
import utils.models as utils_model
import torch.optim as optim
from tensorboardX import SummaryWriter

import time

import train_utils.train_KVNet as train_KVNet

import os,sys
import train_utils.Logger as Logger


def add_noise2pose(src_cam_poses_in, noise_level =.2):
    '''
    noise_level - gaussian_sigma / norm_r r, gaussian_sigma/ norm_t for t
    add Gaussian noise to the poses:
    for R: add in the unit-quaternion space
    for t: add in the raw space
    '''

    src_cam_poses_out = torch.zeros( src_cam_poses_in.shape) 
    src_cam_poses_out[:, :, 3, 3] = 1.
    # for each batch #
    for ibatch in range(src_cam_poses_in.shape[0]):
        src_cam_poses_perbatch = src_cam_poses_in[ibatch, ...]
        for icam in range(src_cam_poses_perbatch.shape[0]):
            src_cam_pose = src_cam_poses_perbatch[icam, ...]

            # convert to unit quaternion #
            r = m_misc.Rotation2UnitQ(src_cam_pose[:3, :3].cuda())
            t = src_cam_pose[:3, 3]

            # add noise to r and t #
            sigma_r = noise_level * r.norm() 
            sigma_t = noise_level * t.norm()
            r = r + torch.randn(r.shape).cuda() * sigma_r
            t = t + torch.randn(t.shape) * sigma_t

            # put back in to src_cam_poses_out #
            src_cam_poses_out[ibatch, icam, :3, :3] = m_misc.UnitQ2Rotation( r).cpu()
            src_cam_poses_out[ibatch, icam, :3, 3] = t

    return src_cam_poses_out
            

def check_datArray_pose(dat_array):
    '''
    Check data array pose/dmap for scan-net. 
    If invalid pose then use the previous pose.  
    Input: data-array: will be modified via reference.
    Output: 
    False: not valid, True: valid
    '''
    if_valid = True
    for dat in dat_array:
        if isinstance(dat['dmap'], int):
            if_valid = False
            break

        elif np.isnan(dat['extM'].min()) or np.isnan(dat['extM'].max()): 
            if_valid = False
            break

    return if_valid

def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # exp name #
    parser.add_argument('--exp_name', required =True, type=str,
            help='The name of the experiment. Used to naming the folders') 

    # nepoch #
    parser.add_argument('--nepoch', required = True, type=int, help='# of epochs to run')

    # if pretrain #
    parser.add_argument('--pre_trained', action='store_true', default=False,
                        help='If use the pre-trained model; (False)')

    # logging # 
    parser.add_argument('--TB_add_img_interv', type=int, default = 50, 
                        help='The inerval for log one training image')

    parser.add_argument('--pre_trained_model_path', type=str,
                        default='.', help='The pre-trained model path for\
                        KV-net')

    # model saving #
    parser.add_argument('--save_model_interv', type=int, default= 5000, 
            help='The interval of iters to save the model; default: 5000')

    # tensorboard #
    parser.add_argument('--TB_fldr', type=str, default='runs',
                        help='The tensorboard logging root folder; default: runs')

    # about training # 
    parser.add_argument('--RNet', action = 'store_true', help='if use refinement net to improve the depth resolution', default=True)

    parser.add_argument('--weight_var', default=.001, type=float, help='weight for the variance loss, if we use L1 loss')

    parser.add_argument('--pose_noise_level', default=0, type=float, help='Noise level for pose. Used for training with pose noise')

    parser.add_argument('--frame_interv', default=5, type=int, help='frame interval') 

    parser.add_argument('--LR', default=.001, type=float, help='Learning rate')

    parser.add_argument('--t_win', type=int, default = 2, help='The radius of the temporal window; default=2')

    parser.add_argument('--d_min', type=float, default=0, help='The minimal depth value; default=0')

    parser.add_argument('--d_max', type=float, default=15, help='The maximal depth value; default=15')

    parser.add_argument('--ndepth', type=int, default= 128, help='The # of candidate depth values; default= 128')

    parser.add_argument('--grad_clip', action='store_true', help='if clip the gradient')

    parser.add_argument('--grad_clip_max', type=float, default=2, help='the maximal norm of the gradient')

    parser.add_argument('--sigma_soft_max', type=float, default=500., help='sigma_soft_max, default = 500.')

    parser.add_argument('--feature_dim', type=int, default=64, help='The feature dimension for the feature extractor; default=64') 

    parser.add_argument('--batch_size', type=int, default = 0, help='The batch size for training; default=0, means batch_size=nGPU')

    # about dataset #
    parser.add_argument('--dataset', type=str, default='scanNet', help='Dataset name: {scanNet, kitti,}') 
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset') 
    parser.add_argument('--change_aspect_ratio', action='store_true', default=False, 
            help='If we want to change the aspect ratio. This option is only useful for KITTI')

    # para config. #
    args = parser.parse_args()
    exp_name = args.exp_name
    saved_model_path = './saved_models/%s'%(exp_name)
    dataset_name = args.dataset

    if args.batch_size ==0:
        batch_size = torch.cuda.device_count()
    else:
        batch_size = args.batch_size

    n_epoch = args.nepoch
    TB_add_img_interv = args.TB_add_img_interv
    pre_trained = args.pre_trained
    t_win_r = args.t_win
    nDepth = args.ndepth 
    d_candi = np.linspace(args.d_min, args.d_max, nDepth) 
    LR = args.LR 
    sigma_soft_max = args.sigma_soft_max #10.#500.
    dnet_feature_dim = args.feature_dim
    frame_interv = args.frame_interv # should be multiple of 5 for scanNet dataset 
    if_clip_gradient = args.grad_clip
    grad_clip_max = args.grad_clip_max 
    d_candi_dmap_ref = d_candi
    nDepth_dmap_ref = nDepth 

    # saving model config.#
    m_misc.m_makedir(saved_model_path)
    savemodel_interv = args.save_model_interv 

    # writer #
    log_dir = '%s/%s'%(args.TB_fldr, exp_name)
    writer = SummaryWriter(log_dir = log_dir, comment='%s'%(exp_name))
    m_misc.save_args(args, '%s/tr_paras.txt'%(log_dir)) # save the training parameters # 
    logfile=os.path.join(log_dir,'log_'+str(time.time())+'.txt')
    stdout=Logger.Logger(logfile)
    sys.stdout = stdout

    # Initialize data-loader, model and optimizer #

    # ===== Dataset selection ======== #
    dataset_path = args.dataset_path
    if dataset_name == 'scanNet':
        dataset_init = dl_scanNet.ScanNet_dataset 

        if not dataset_path == '.':
            fun_get_paths = lambda traj_indx: dl_scanNet.get_paths(traj_indx, 
                    frame_interv=5, split_txt= './mdataloader/scanNet_split/scannet_train.txt', 
                    database_path_base = dataset_path ) 
        else:
            fun_get_paths = lambda traj_indx: dl_scanNet.get_paths(traj_indx, frame_interv=5, 
                    split_txt= './mdataloader/scanNet_split/scannet_train.txt') 
            
        img_size = [384, 256] 

        # trajectory index for training #
        n_scenes , _, _, _, _ = fun_get_paths(0) 
        traj_Indx = np.arange(0, n_scenes) 

    elif dataset_name == 'kitti':
        import mdataloader.kitti as dl_kitti 
        dataset_init = dl_kitti.KITTI_dataset

        if not dataset_path == '.':
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx,split_txt= './mdataloader/kitti_split/training.txt', mode='train',
                    database_path_base = dataset_path) 
        else: # use default database path
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx,split_txt= './mdataloader/kitti_split/training.txt', mode='train') 

#        img_size = [1248, 380] 
        if not args.change_aspect_ratio: # we will keep the aspect ratio and do cropping
            img_size = [768, 256] 
            crop_w = 384

        else: # we will change the aspect ratio and NOT do cropping
            img_size = [384, 256] 
#            img_size = [512, 256] 
#            img_size = [624, 256] 
            crop_w = None

        n_scenes , _, _, _, _ = fun_get_paths(0) 
        traj_Indx = np.arange(0, n_scenes) 

    else:
        raise Exception('dataset not implemented ') 

    fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
    if dataset_name == 'kitti':
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path = intrin_path, img_size= img_size, digitize= True,
                               d_candi= d_candi_dmap_ref, resize_dmap=.25, crop_w = crop_w) 
    else:
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path = intrin_path, img_size= img_size, digitize= True,
                               d_candi= d_candi_dmap_ref, resize_dmap=.25) 
    # ================================ #

    print('Initnializing the KV-Net')
    model_KVnet = m_kvnet.KVNET(feature_dim = dnet_feature_dim, cam_intrinsics = dataset.cam_intrinsics, 
                                d_candi = d_candi, sigma_soft_max = sigma_soft_max, KVNet_feature_dim = dnet_feature_dim, 
                                d_upsample_ratio_KV_net = None, t_win_r = t_win_r, if_refined = args.RNet) 

    model_KVnet = torch.nn.DataParallel(model_KVnet,  dim=0)
    model_KVnet.cuda(0)

    optimizer_KV = optim.Adam(model_KVnet.parameters(), lr = LR , betas= (.9, .999 ))

    model_path_KV = args.pre_trained_model_path
    if model_path_KV is not '.' and pre_trained:
        print('loading KV_net at %s'%(model_path_KV))
        utils_model.load_pretrained_model(model_KVnet, model_path_KV, optimizer_KV)

    print('Done') 

    LOSS = []
    total_iter = 0

    d_candi_up = d_candi 

    for iepoch in range(n_epoch): 
        BatchScheduler = batch_loader.Batch_Loader(
                batch_size = batch_size, fun_get_paths = fun_get_paths, 
                dataset_traj = dataset, nTraj=len(traj_Indx), dataset_name = dataset_name ) 

        for batch_idx in range(len(BatchScheduler)): 
            for frame_count, ref_indx in enumerate( range(BatchScheduler.traj_len) ): 
                local_info = BatchScheduler.local_info() 
                n_valid_batch= local_info['is_valid'].sum() 

                if n_valid_batch > 0: 
                    local_info_valid = batch_loader.get_valid_items(local_info) 
                    ref_dats_in = local_info_valid['ref_dats']
                    src_dats_in = local_info_valid['src_dats']
                    cam_intrin_in = local_info_valid['cam_intrins']
                    src_cam_poses_in = torch.cat( local_info_valid['src_cam_poses'], dim=0) 

                    if args.pose_noise_level > 0:
                        src_cam_poses_in = add_noise2pose( src_cam_poses_in, args.pose_noise_level)

                    if frame_count ==0 or prev_invalid:
                        prev_invalid = False
                        BVs_predict_in = None
                        print('frame_count ==0 or invalid previous frame') 
                    else: 
                        BVs_predict_in = batch_loader.get_valid_BVs(BVs_predict, local_info['is_valid'] ) 

                    BVs_measure, BVs_predict, loss, dmap_log_l, dmap_log_h= train_KVNet.train(
                            n_valid_batch,
                            model_KVnet, optimizer_KV, t_win_r,
                            d_candi, Ref_Dats = ref_dats_in, Src_Dats = src_dats_in, 
                            Src_CamPoses= src_cam_poses_in,
                            BVs_predict = BVs_predict_in, 
                            Cam_Intrinsics= cam_intrin_in,
                            weight_var = args.weight_var, 
                            loss_type = 'NLL', mGPU = True) 

                    BVs_measure = BVs_measure.detach()
                    loss_v = float(loss.data.cpu().numpy()) 

                    if n_valid_batch < BatchScheduler.batch_size:
                        BVs_predict = batch_loader.fill_BVs_predict(BVs_predict, local_info['is_valid']) 

                else:
                    loss_v = LOSS[-1] 
                    prev_invalid = True

                # Update dat_array # 
                if frame_count < BatchScheduler.traj_len-1:
                    BatchScheduler.proceed_frame()

                total_iter += 1 

                # logging #
                if frame_count > 0:
                    LOSS.append(loss_v)
                    print('video batch %d / %d, iter: %d, frame_count: %d; Epoch: %d / %d, loss = %.5f'\
                          %(batch_idx + 1, len(BatchScheduler), total_iter, frame_count, iepoch + 1, n_epoch, loss_v)) 

                    writer.add_scalar('data/train_error', float(loss_v), total_iter)

                if total_iter % savemodel_interv ==0  : 
                    # if training, save the model # 
                    savefilename = saved_model_path + '/kvnet_checkpoint_iter_' + str(total_iter) + '.tar' 
                    torch.save({'iter': total_iter,
                                'frame_count': frame_count,
                                'ref_indx': ref_indx,
                                'traj_idx': batch_idx,
                                'state_dict': model_KVnet.state_dict(),
                                'optimizer': optimizer_KV.state_dict(),
                                'loss': loss_v}, savefilename)

                if total_iter % TB_add_img_interv ==0 :
                    # if training, logging # 
                    th_dmaps_log = torch.FloatTensor(dmap_log_l.astype(np.float32))
                    th_dmaps_log = th_dmaps_log.unsqueeze(0)
                    th_dmaps_log = (th_dmaps_log / (d_candi_dmap_ref.max())).clamp(0,1)
                    th_dmaps_log = th_dmaps_log.repeat([3,1,1]) 
                    input_img_log = ref_dats_in[0]['img'].clone()
                    input_img_log = (input_img_log - input_img_log.min()) / (input_img_log.max() - input_img_log.min())
                    input_img_log = input_img_log.squeeze()

                    # assuming N=1 for BVs_measure #
                    confMap_log, _ = torch.max(BVs_measure[0, ...], dim=0)
                    confMap_log = torch.exp(confMap_log.squeeze().cpu())
                    confMap_log /= confMap_log.max()
                    confMap_log = confMap_log.repeat([3,1,1])
                    writer.add_image('%s/tr_dmaps'%(exp_name), th_dmaps_log, total_iter) 
                    writer.add_image('%s/tr_input'%(exp_name), input_img_log, total_iter) 
                    writer.add_image('%s/conf_map'%(exp_name), confMap_log, total_iter) 

                    # up-sample branch #
                    if dmap_log_h is not -1:
                        th_dmaps_up_log = torch.FloatTensor(dmap_log_h.astype(np.float32))
                        th_dmaps_up_log = th_dmaps_up_log.unsqueeze(0)
                        th_dmaps_up_log = (th_dmaps_up_log / (d_candi_dmap_ref.max())).clamp(0,1)
                        th_dmaps_up_log = th_dmaps_up_log.repeat([3,1,1])
                        writer.add_image('%s/tr_dmaps_up'%(exp_name), th_dmaps_up_log , total_iter) 
            
            BatchScheduler.proceed_batch() 

    writer.close()
    stdout.delink()

if __name__ == '__main__':
    main()
