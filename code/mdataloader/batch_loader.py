'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# batch loader for videos, each batch would be a dataloader (dl_scanNet, dl_vkitti or dl_kitti)

import numpy as np
import os 
import math
import sys
import random

import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as tfv_transform

import mdataloader.m_preprocess as m_preprocess
import mdataloader.misc as mloader_misc
import warping.View as View 
import warping.homography as warp_homo

import mutils.misc as m_misc 

import copy

def fill_BVs_predict(BVs_predict_valid, is_valid):
    '''
    Output BVs_predict_full, that corresponds the full trajectory
    BVs_predict_valid - NDHW
    ''' 
    nBatch = len(is_valid) 
    n_valid, D, H, W = BVs_predict_valid.shape
    BVs_predict_full = torch.empty( nBatch, D, H, W).cuda( BVs_predict_valid.get_device())
    BVs_predict_full[:] = torch.tensor(float('nan')) 
    
    cur_idx =0
    BVs_predict_full[is_valid.nonzero()[0],... ] = BVs_predict_valid

    return BVs_predict_full

def get_valid_BVs(BVs, is_valid):
    return BVs[is_valid.nonzero()[0], :, :, :] 

def get_valid_items(local_info):
    '''
    Get valid batch items
    '''
    is_valid = local_info['is_valid']
    if is_valid.sum() ==0:
        return False 
    else:
        is_valid_batches = local_info['is_valid']
        n_valid_batch= is_valid_batches.sum() 
        ref_dats, src_dats = local_info['ref_dats'], local_info['src_dats']
        cam_intrins = local_info['cam_intrins']
        src_cam_poses = local_info['src_cam_poses'] 

        ref_dats_valid, src_dats_valid = [],  []
        src_cam_poses_valid = []
        cam_intrins_valid = []

        for ibatch in range( len(is_valid)):
            if is_valid_batches[ibatch]:
                ref_dats_valid.append(ref_dats[ibatch])
                src_dats_valid.append(src_dats[ibatch])
                cam_intrins_valid.append(cam_intrins[ibatch])
                src_cam_poses_valid.append(src_cam_poses[ibatch])

        return {'cam_intrins': cam_intrins_valid, 
                'ref_dats': ref_dats_valid, 
                'src_dats': src_dats_valid, 
                'src_cam_poses': src_cam_poses_valid } 

def _check_datArray_pose(dat_array):
    '''
    Check data array pose/dmap 
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

class Batch_Loader():
    def __init__(
            self, batch_size, fun_get_paths, 
            dataset_traj, nTraj, 
            dataset_name,
            t_win_r = 2):
        '''
        fun_get_paths - function, takes input the traj index and output: 
        fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path =  fun_get_paths(traj_idx) 

        inputs:
        dataset_traj - a dataset object for one trajectory, we will have multiple dataset in one batch 

        members: 
            nbatch - the # of batches 

            state variables: 
                self.trajs_st_frame
                self.traj_len 
                self.dataset_batch
                self.frame_idx_cur - the current frame index in the trajectories 
                self.dat_arrays - list of dat_array, each dat_array is a list of data items for one local time window, 
        ''' 
        assert batch_size > 1

        self.batch_size = batch_size
        self.fun_get_paths = fun_get_paths 
        self.dataset_name = dataset_name
        self.ntraj = nTraj 
        self.t_win_r = int(t_win_r )

        batch_traj_st = np.arange(0, self.ntraj, batch_size)
        self.nbatch = len(batch_traj_st)
        self.batch_traj_st = batch_traj_st 

        # initialize the traj schedule # 
        # For one batch (index as i) of trajectories, the trajectories would be
        # traj_batch[i] = [batch_traj_st[i], ..., batch_traj_ed[i]]
        self.batch_traj_ed = batch_traj_st + batch_size 
        self.batch_traj_st[ self.batch_traj_ed > nTraj ] = nTraj - batch_size
        self.batch_traj_ed[ self.batch_traj_ed > nTraj ] = nTraj 

        self.batch_idx = 0 # the batch index
        self.batch_traj_idx_cur = np.arange( 
                self.batch_traj_st[self.batch_idx], 
                self.batch_traj_ed[self.batch_idx]) # the trajectory indexes in the current batch

        # Initialize the batch #
        dataset_batch = [] 
        for ibatch in range( batch_size ):
            traj_indx_per = self.batch_traj_idx_cur[ibatch]
            fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path = fun_get_paths(traj_indx_per) 
            dataset_batch.append( copy.copy(dataset_traj) ) 
            dataset_batch[-1].set_paths(img_seq_paths, dmap_seq_paths, poses) 

            if dataset_name == 'scanNet':
                # For each trajector in the dataset, we will update the intrinsic matrix #
                dataset_batch[-1].get_cam_intrinsics(intrin_path)

        self.dataset_batch = dataset_batch

        # get the start frame and traj lengths for all trajectories #
        trajs_st_frame, traj_len = self._get_traj_lengths() 
        self.trajs_st_frame = trajs_st_frame
        self.traj_len = traj_len

        # get dat_arrays #
        dat_arrays = []
        for ibatch in range( batch_size ):
            dat_array_ = [ self.dataset_batch[ibatch][idx] for idx in range(
                            trajs_st_frame[ibatch] - t_win_r, trajs_st_frame[ibatch] + t_win_r + 1) ]  

            dat_arrays.append(dat_array_) 

        self.frame_idx_cur = 0 
        self.dat_arrays = dat_arrays

    def _get_traj_lengths(self):

        raw_traj_batch_len = np.array( [len(dataset_) for dataset_ in self.dataset_batch] ) 
        traj_len = raw_traj_batch_len.min() - 2 * self.t_win_r
        trajs_st_frame = np.zeros(self.batch_size).astype( np.int) 
        t_win_r = self.t_win_r 

        for ibatch in range(self.batch_size):
            if raw_traj_batch_len[ibatch] == traj_len + 2* self.t_win_r:
                trajs_st_frame[ibatch] = self.t_win_r
            elif raw_traj_batch_len[ibatch] - traj_len - t_win_r > t_win_r:
                trajs_st_frame[ibatch] = int(random.randrange(self.t_win_r, raw_traj_batch_len[ibatch] - traj_len - t_win_r) )
            else:
                trajs_st_frame[ibatch] = self.t_win_r 

        return trajs_st_frame, traj_len 

    def __len__(self):
        return self.nbatch

    def proceed_batch(self):
        '''
        Move forward one batch of trajecotries
        ''' 

        self.batch_idx += 1
        batch_size = self.batch_size

        if self.batch_idx >= self.nbatch: # reaching the last batch 
            return False

        else: 
            self.batch_traj_idx_cur = np.arange( 
                    self.batch_traj_st[self.batch_idx], 
                    self.batch_traj_ed[self.batch_idx]) 

            # re-set the traj. in the current batch #

            for ibatch in range( batch_size ):
                traj_indx_per = self.batch_traj_idx_cur[ibatch] 
                fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path = self.fun_get_paths(traj_indx_per) 
                self.dataset_batch[ ibatch ].set_paths(img_seq_paths, dmap_seq_paths, poses) 
                if self.dataset_name == 'scanNet':
                    # For each trajector in the dataset, we will update the intrinsic matrix #
                    self.dataset_batch[ibatch].get_cam_intrinsics(intrin_path) 

            # get the start frame and traj lengths for all trajectories #
            trajs_st_frame, traj_len = self._get_traj_lengths() 
            self.trajs_st_frame = trajs_st_frame
            self.traj_len = traj_len
            self.frame_idx_cur = 0

            # get dat_arrays #
            dat_arrays = []
            for ibatch in range( batch_size ):
                dat_array_ = [ self.dataset_batch[ibatch][idx] for idx in range(
                                trajs_st_frame[ibatch] - self.t_win_r, trajs_st_frame[ibatch] + self.t_win_r + 1) ]  

                dat_arrays.append(dat_array_)

            self.dat_arrays = dat_arrays 

            return True

    def proceed_frame(self):
        '''
        Move forward one frame for all trajectories
        ''' 
        self.frame_idx_cur += 1 
        for ibatch in range( self.batch_size ):
            self.dat_arrays[ibatch].pop(0) 

            self.dat_arrays[ibatch].append( 
                    self.dataset_batch[ibatch][ self.frame_idx_cur + self.trajs_st_frame[ibatch] + self.t_win_r ])

    def local_info(self):
        '''
        return local info, including { cam_intrins, ref_dats, src_dats, src_cam_poses, is_valid } 
        each is a list of variables 

        src_cam_poses[i] - 1 x n_src x 4 x 4
        '''

        is_valid = np.ones(self.batch_size, np.bool)
        ref_dats = []
        src_dats = []
        cam_intrins = []
        src_cam_poses = []

        for ibatch in range(self.batch_size):
            dat_array_ = self.dat_arrays[ibatch]
            ref_dat_, src_dat_ = m_misc.split_frame_list(dat_array_, self.t_win_r) 
            is_valid_ = _check_datArray_pose(dat_array_) 
            is_valid[ibatch] = is_valid_
            if is_valid_:
                src_cam_extMs = m_misc.get_entries_list_dict(src_dat_, 'extM') 
                src_cam_pose_ = [ warp_homo.get_rel_extrinsicM(ref_dat_['extM'], src_cam_extM_) for src_cam_extM_ in src_cam_extMs ] 
                src_cam_pose_ = [ torch.from_numpy(pose.astype(np.float32)).unsqueeze(0) for pose in src_cam_pose_ ] 
                # src_cam_poses size: N V 4 4
                src_cam_pose_ = torch.cat(src_cam_pose_, dim=0).unsqueeze(0) 
            else:
                src_cam_pose_ = -1

            ref_dats.append(ref_dat_)
            src_dats.append(src_dat_)
            cam_intrins.append(self.dataset_batch[ibatch].cam_intrinsics) 
            src_cam_poses.append(src_cam_pose_) 

        # print(is_valid) 
        return {'is_valid': is_valid, 'cam_intrins': cam_intrins, 'ref_dats': ref_dats, 
                 'src_dats': src_dats, 'src_cam_poses': src_cam_poses } 
