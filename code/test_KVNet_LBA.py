'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''
# We will also optimize the poses #
import math 
import numpy as np
import torch
torch.backends.cudnn.benchmark=True 

import ICP.opt_pose_numerical as opt_pose_numerical 
import mutils.misc as m_misc 
import warping.homography as warp_homo 
import test_utils.export_res as export_res 
import models.KVNET as m_kvnet
import utils.models as utils_model
import test_utils.test_KVNet as test_KVNet 
import DSO.dso_io as dso_io 


def get_fb(src_cam_poses, cam_intrinsic, src_cam_pose_next = None):
    f_ = cam_intrinsic['focal_length'] 

    if isinstance( src_cam_poses, list):
        t_norms = [np.linalg.norm(src_cam_pose.cpu().squeeze().numpy()[:3, 3]) for src_cam_pose in src_cam_poses]
    else:
        t_norms = [np.linalg.norm(src_cam_poses.cpu().numpy()[0, idx, :3, 3]) for idx in range( src_cam_poses.shape[1]) ]

    if src_cam_pose_next is not None: 
        t_norm_new = np.linalg.norm(src_cam_pose_next[:3, 3]) 
        t_norms.pop(0)
        t_norms.append(t_norm_new)

    b_ = np.mean(t_norms) # Mean value for the baselines
    return f_ * b_ * 4, t_norms, # 4 for feature space dw_sampling 

def get_t_norms( traj_extM, dat_indx_step ):
    r'''
    Get the baselines for all src_cam_pose in the traj_extMs, estimated from DSO
    '''
    # Get the valid traj_extMs, estimated from DSO #
    extM_valid = []
    traj_extM_ = traj_extM[1:] 

    for ext in traj_extM_:
        if dso_io.valid_pose(ext):
            extM_valid.append(ext)

    t_norms = []
    for idx in range( 2* dat_indx_step, len( extM_valid)):
        t_norm = extM_valid[ idx][:3, 3] - extM_valid[idx - 2* dat_indx_step][:3, 3]
        t_norm = np.linalg.norm( t_norm, )
        t_norms.append(t_norm)

    return np.array( t_norms ) 

def rescale_traj_t(traj_M, scale):
    for trajm in traj_M:
        if trajm is not -1:
            trajm[:3,3] *= scale

def copy_list(list_in):
    # deep copy list of tensors/np arrays #
    list_out = []
    for ele in list_in:
        if isinstance(ele, torch.Tensor):
            list_out.append( ele.clone() )
        else:
            list_out.append( ele.copy() )
    return list_out

def init_traj_extMs( traj_len = 100, dso_res_path = None, if_filter= False, min_idx=None, max_idx = None):
    # initialize camera poses and do filter if necessary #
    if min_idx is not None:
        assert min_idx <= traj_len - 1

    if max_idx is not None:
        assert max_idx <= traj_len - 1

    if dso_res_path is not None:
        dso_Rts = dso_io._read_camera_poses( dso_res_path, if_filter = if_filter )
        traj_extMs = []

        assert min_idx is not None and max_idx is not None

        for i in range( traj_len ):
            extM_init = np.eye(4)
            traj_extMs.append(extM_init) 

        for i in range( len(dso_Rts )):
            extM_init = np.linalg.inv( dso_Rts[i] ) 
            traj_extMs[ min_idx + i ] = extM_init

        traj_extMs = traj_extMs[:max_idx]

    else:
        assert traj_len is not None
        traj_extMs = []
        for i in range( traj_len ):
            extM_init = np.random.randn(4,4 )
            extM_init[3, :] = 0
            extM_init[3,3] = 1
            traj_extMs.append(extM_init) 
        
    return traj_extMs

def update_dat_array(dat_array, dataset, data_interv, frame_interv, ref_indx, t_win_r):
    '''
    dat_array - list of dataset items, e.g. [dataset[0], dataset[1], ...]

    dataset - dataset object

    data_interv - the dataset frame interval, for example, =1 for 1-frame-scanNet, =5 for sceneNet
                 this is also the frame_interval we will use to propagate dpv and update pose

    frame_interv - the frame interv we want to use to estimate the depth

    ref_indx - the index for the reference frame

    t_win_r - the radius of the time window , we set it to be 2 for all experiments
    ''' 

    assert frame_interv % data_interv ==0 and frame_interv >= data_interv

    dat_indx_step = int(frame_interv / data_interv) 
    if data_interv == frame_interv:
        dat_array.pop(0)
        dat_array.append(dataset[ref_indx + t_win_r + 1 ])
        return dat_array 
    else:
        # In case frame_interv is multiples of data_interv, we will update the whole list #
        dat_array = [ dataset[idx] for idx in range(
            ref_indx+1 - dat_indx_step * t_win_r, 
            ref_indx+1 + dat_indx_step * t_win_r + 1, dat_indx_step ) ] 
        return dat_array 

def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # exp name #
    parser.add_argument('--exp_name', required =True, type=str,
            help='The name of the experiment. Used to naming the folders') 

    # about testing # 
    parser.add_argument('--img_name_pattern', type=str, default='*.png', help='image name pattern')
    parser.add_argument('--model_path', type=str, default='.', help='The pre-trained model path for KV-net')
    parser.add_argument('--split_file', type=str, default='.', help='The split txt file')
    parser.add_argument('--t_win', type=int, default=2, help='The radius of the temporal window; default=2') 
    parser.add_argument('--d_min', type=float, default=0, help='The minimal depth value; default=0')
    parser.add_argument('--d_max', type=float, default=5, help='The maximal depth value; default=15')
    parser.add_argument('--ndepth', type=int, default=64, help='The # of candidate depth values; default= 128') 
    parser.add_argument('--sigma_soft_max', type=float, default=10., help='sigma_soft_max, default = 500.') 
    parser.add_argument('--feature_dim', type=int, default=64, help='The feature dimension for the feature extractor; default=64') 

    # about pose #
    parser.add_argument('--intrin_path', type = str, required= True, help = 'camera intrinic path, saved as .mat') 

    parser.add_argument('--dso_res_path', type = str, default = 'dso_res/result_dso.txt',
                        help='if use DSO pose, specify the path to the DSO results. Should be a .txt file') 
    parser.add_argument('--opt_next_frame', action = 'store_true', help ='') 
    parser.add_argument('--use_gt_R', action = 'store_true', help ='') 
    parser.add_argument('--use_gt_t', action = 'store_true', help ='') 
    parser.add_argument('--use_dso_R', action = 'store_true', help ='') 
    parser.add_argument('--use_dso_t', action = 'store_true', help ='') 
    parser.add_argument('--min_frame_idx', type = int , help =' ',  default = 0) 
    parser.add_argument('--max_frame_idx', type = int , help =' ',  default = 10000) 
    parser.add_argument('--refresh_frames', type = int , help =' ', default= 1000) 
    parser.add_argument('--LBA_max_iter', type = int , help =' ') 
    parser.add_argument('--opt_r', type = int, default=1, help =' ') 
    parser.add_argument('--opt_t', type = int, default=1, help =' ') 
    parser.add_argument('--LBA_step', type = float, help =' ') 
    parser.add_argument('--frame_interv', type = int, default = 5, help =' ') 

    # about dataset #
    parser.add_argument('--dataset', type=str, default='7scenes', help='Dataset name: {scanNet, 7scenes}') 
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset') 

    # about output #
    parser.add_argument('--output_pngs', action = 'store_true', help ='if output pngs') 

    # para config. #
    args = parser.parse_args()
    exp_name = args.exp_name 
    dataset_name = args.dataset 
    t_win_r = args.t_win
    nDepth = args.ndepth 

    d_candi = np.linspace(args.d_min, args.d_max, nDepth) 

    sigma_soft_max = args.sigma_soft_max #10.#500.
    dnet_feature_dim = args.feature_dim
    frame_interv = args.frame_interv 
    d_candi_dmap_ref = d_candi
    nDepth_dmap_ref = nDepth 

    # Initialize data-loader, model and optimizer # 
    # ===== Dataset selection ======== #
    dataset_path = args.dataset_path
    if dataset_name == 'scanNet':
        #  deal with 1-frame scanNet data
        import mdataloader.scanNet as dl_scanNet
        dataset_init = dl_scanNet.ScanNet_dataset 
        split_txt = './mdataloader/scanNet_split/scannet_val.txt' if args.split_file=='.' else args.split_file 
        if not dataset_path == '.':
            # if specify the path, we will assume we are using 1-frame-interval scanNet video #
            fun_get_paths = lambda traj_indx: dl_scanNet.get_paths_1frame(traj_indx, 
                    database_path_base = dataset_path, split_txt= split_txt ) 
            dat_indx_step = 5 #pick this value to make sure the camera baseline is big enough 
        else:
            fun_get_paths = lambda traj_indx: dl_scanNet.get_paths(traj_indx, frame_interv=5, split_txt= split_txt) 
            dat_indx_step = 1 
        img_size = [384, 256] 
        # trajectory index for training #
        n_scenes , _, _, _, _ = fun_get_paths(0) 
        traj_Indx = np.arange(0, n_scenes) 

    elif dataset_name == '7scenes':
        # 7 scenes video #
        import mdataloader.dl_7scenes as dl_7scenes
        img_size = [384, 256] 
        dataset_init = dl_7scenes.SevenScenesDataset
        dat_indx_step = 5 # pick this value to make sure the camera baseline is big enough 
        # trajectory index for training #
        split_file = None if args.split_file=='.' else args.split_file 
        fun_get_paths = lambda traj_indx: dl_7scenes.get_paths_1frame(traj_indx, database_path_base = dataset_path , split_txt = split_file,) 

    elif dataset_name == 'single_folder':
        # images in a single folder specified by the user #
        import mdataloader.mdata as mdata
        img_size = [384, 256] 
        dataset_init = mdata.mData
        dat_indx_step = 5 # pick this value to make sure the camera baseline is big enough 
        fun_get_paths = lambda traj_indx: mdata.get_paths_1frame(traj_indx, dataset_path, args.img_name_pattern )
        traj_Indx = [0] #dummy 

    fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths( traj_Indx[0])

    if dataset_name == 'single_folder':
        intrin_path = args.intrin_path

    dataset = dataset_init(True, img_paths, dmap_paths, poses,
                           intrin_path = intrin_path, img_size= img_size, digitize= True,
                           d_candi= d_candi_dmap_ref, resize_dmap=.25, ) 

    dataset_Himgsize = dataset_init(True, img_paths, dmap_paths, poses,
                           intrin_path = intrin_path, img_size= img_size, digitize= True,
                           d_candi= d_candi_dmap_ref, resize_dmap=.5, ) 

    dataset_imgsize = dataset_init(True, img_paths, dmap_paths, poses,
                           intrin_path = intrin_path, img_size= img_size, digitize= True,
                           d_candi= d_candi_dmap_ref, resize_dmap=1, ) 


    # ================================ # 

    print('Initnializing the KV-Net') 
    model_KVnet = m_kvnet.KVNET(\
            feature_dim = dnet_feature_dim, 
            cam_intrinsics = dataset.cam_intrinsics, 
            d_candi = d_candi, sigma_soft_max = sigma_soft_max, 
            KVNet_feature_dim = dnet_feature_dim, 
            d_upsample_ratio_KV_net = None, 
            t_win_r = t_win_r, if_refined = True) 

    model_KVnet = torch.nn.DataParallel(model_KVnet)
    model_KVnet.cuda()

    model_path_KV = args.model_path
    print('loading KV_net at %s'%(model_path_KV))
    utils_model.load_pretrained_model(model_KVnet, model_path_KV)
    print('Done') 

    for traj_idx in traj_Indx:
        scene_path_info = []
        print('Getting the paths for traj_%d'%(traj_idx))
        fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path = fun_get_paths(traj_idx) 
        res_fldr = '../results/%s/traj_%d'%(exp_name, traj_idx)
        m_misc.m_makedir(res_fldr) 

        dataset.set_paths(img_seq_paths, dmap_seq_paths, poses) 

        if dataset_name == 'scanNet':
            # the camera intrinsic may be slightly different for different trajectories in scanNet #
            dataset.get_cam_intrinsics(intrin_path)

        print('Done')
        if args.min_frame_idx > 0:
            frame_idxs = np.arange(args.min_frame_idx - t_win_r, args.min_frame_idx + t_win_r)
            dat_array = [ dataset[idx] for idx in frame_idxs]  
        else:
            dat_array = [ dataset[idx] for idx in range(t_win_r * 2 + 1) ]  

        DMaps_meas = []
        dso_res_path = args.dso_res_path

        print('init initial pose from DSO estimations ...') 
        traj_extMs = init_traj_extMs( traj_len = len(dataset), dso_res_path = dso_res_path, if_filter= True,
                                      min_idx = args.min_frame_idx, max_idx = args.max_frame_idx)
        traj_extMs_init = copy_list(traj_extMs)
        traj_length = min(len(dataset), len(traj_extMs)) 
        first_frame = True 
        for frame_cnt, ref_indx in enumerate( 
                range( t_win_r*dat_indx_step + args.min_frame_idx, traj_length - t_win_r* dat_indx_step - dat_indx_step) ): 
            # ref_indx: the frame index for the reference frame #

            # Read ref. and src. data in the local time window #
            ref_dat, src_dats = m_misc.split_frame_list(dat_array, t_win_r)

            src_frame_idx = [ idx for idx in range( 
                ref_indx - t_win_r * dat_indx_step, ref_indx, dat_indx_step) ] + \
                            [ idx for idx in range( 
                 ref_indx + dat_indx_step, ref_indx + t_win_r*dat_indx_step+1, dat_indx_step) ] 

            valid_seq = dso_io.valid_poses( traj_extMs, src_frame_idx )

            # only look at a subset of frames #
            if ref_indx < args.min_frame_idx:
                valid_seq = False
            if ref_indx > args.max_frame_idx or ref_indx >= traj_length- t_win_r* dat_indx_step - dat_indx_step :
                break 
            if frame_cnt ==0 or valid_seq is False:
                BVs_predict = None 

            # refresh #
            if ref_indx % args.refresh_frames==0:
                print('REFRESH !')
                BVs_predict = None
                BVs_predict_in = None
                first_frame = True
                traj_extMs = copy_list(traj_extMs_init) 

            if valid_seq: # if the sequence does not contain invalid pose estimation 
                # Get poses #
                src_cam_extMs = [traj_extMs[i] for i in src_frame_idx] 
                ref_cam_extM  = traj_extMs[ref_indx] 
                src_cam_poses =  [warp_homo.get_rel_extrinsicM(ref_cam_extM, src_cam_extM_) for src_cam_extM_ in src_cam_extMs ]
                src_cam_poses = [ torch.from_numpy(pose.astype(np.float32)).cuda().unsqueeze(0) for pose in src_cam_poses]

                # Load the gt pose if available #
                if 'extM' in dataset[0]:
                    src_cam_extMs_ref = [ dataset[i]['extM'] for i in src_frame_idx] 
                    ref_cam_extM_ref  = dataset[ref_indx]['extM'] 
                    src_cam_poses_ref = [ warp_homo.get_rel_extrinsicM(ref_cam_extM_ref, src_cam_extM_) \
                                         for src_cam_extM_ in src_cam_extMs_ref ]
                    src_cam_poses_ref = [ torch.from_numpy(pose.astype(np.float32)).cuda().unsqueeze(0) \
                                         for pose in src_cam_poses_ref ] 

                # -- Determine the scale, mapping from DSO scale to our working scale -- # 
                if frame_cnt == 0 or BVs_predict is None: # the first window for the traj.  
                    _, t_norm_single = get_fb(src_cam_poses, dataset.cam_intrinsics, src_cam_pose_next = None) 
                    # We need to heurisitcally determine scale_ without using GT pose #
                    t_norms = get_t_norms( traj_extMs, dat_indx_step) 
                    scale_ = d_candi.max() / ( dataset.cam_intrinsics['focal_length'] * np.array(t_norm_single).max() /2 ) 
                    scale_ = d_candi.max() / ( dataset.cam_intrinsics['focal_length'] * np.array(t_norms).max() ) 
                    scale_ = d_candi.max() / ( dataset.cam_intrinsics['focal_length'] * np.array(t_norms).mean()/2  ) 
                    rescale_traj_t(traj_extMs, scale_) 
                    traj_extMs_dso = copy_list( traj_extMs ) 
                    # Get poses #
                    src_cam_extMs = [traj_extMs[i] for i in src_frame_idx] 
                    ref_cam_extM  = traj_extMs[ref_indx] 
                    src_cam_poses =  [warp_homo.get_rel_extrinsicM(ref_cam_extM, src_cam_extM_) for src_cam_extM_ in src_cam_extMs ]
                    src_cam_poses = [ torch.from_numpy(pose.astype(np.float32)).cuda().unsqueeze(0) for pose in src_cam_poses ]

                # src_cam_poses size: N V 4 4 #
                src_cam_poses = torch.cat(src_cam_poses, dim=0).unsqueeze(0)
                src_frames = [m_misc.get_entries_list_dict(src_dats, 'img')] 
                cam_pose_next = traj_extMs[ref_indx+1]
                cam_pose_next = torch.FloatTensor( warp_homo.get_rel_extrinsicM(traj_extMs[ref_indx], cam_pose_next)).cuda() 

                BVs_predict_in = None if frame_cnt == 0 or BVs_predict is None \
                                      else BVs_predict

                BVs_measure, BVs_predict = test_KVNet.test( model_KVnet, d_candi,
                                                            Ref_Dats= [ref_dat], 
                                                            Src_Dats= [src_dats],
                                                            Cam_Intrinsics=[dataset.cam_intrinsics], 
                                                            t_win_r = t_win_r,
                                                            Src_CamPoses= src_cam_poses,
                                                            BV_predict= BVs_predict_in,
                                                            R_net = True, 
                                                            cam_pose_next = cam_pose_next,
                                                            ref_indx = ref_indx ) 

                # export_res.export_res_refineNet(ref_dat,  BVs_measure, d_candi_dmap_ref, 
                #                                 res_fldr, ref_indx, 
                #                                 save_mat = True, output_pngs = args.output_pngs, output_dmap_ref=False)
                export_res.export_res_img(ref_dat, BVs_measure, d_candi_dmap_ref, res_fldr, frame_cnt)
                scene_path_info.append( [frame_cnt, dataset[ref_indx]['img_path']] )


                # UPDATE dat_array #
                if dat_indx_step > 1: # use one-interval video and the frame interval is larger than 5
                    print('updating array ...')
                    dat_array = update_dat_array(dat_array, dataset, 
                                                 data_interv= 1, frame_interv=5, 
                                                 ref_indx=ref_indx, t_win_r = t_win_r ) 
                    print('done')

                else:
                    dat_array.pop(0)
                    new_dat = dataset[ref_indx + t_win_r +1 ]
                    dat_array.append(new_dat) 

                # OPTMIZE POSES #
                idx_ref_ = ref_indx + 1 
                cam_pose_nextframe = traj_extMs[ idx_ref_ ] 
                cam_pose_nextframe = torch.FloatTensor( warp_homo.get_rel_extrinsicM(traj_extMs[ref_indx], cam_pose_nextframe)).cuda() 

                # get depth and confidence map #
                BV_tmp_ = warp_homo.resample_vol_cuda(\
                                        src_vol = BVs_measure, rel_extM = cam_pose_nextframe.inverse(),
                                        cam_intrinsic = dataset_imgsize.cam_intrinsics, 
                                        d_candi = d_candi, d_candi_new = d_candi,
                                        padding_value = math.log(1. / float(len(d_candi)))
                                        ).clamp(max=0, min=-1000.) 
                dmap_ref        = m_misc.depth_val_regression(BVs_measure, d_candi, BV_log=True).squeeze()
                conf_map_ref, _ = torch.max(BVs_measure.squeeze(), dim=0 ) 
                dmap_kf = m_misc.depth_val_regression(BV_tmp_.unsqueeze(0), d_candi, BV_log=True).squeeze()
                conf_map_kf, _ = torch.max(BV_tmp_.squeeze(), dim=0 )

                # setup optimization #
                cams_intrin = [dataset.cam_intrinsics, 
                               dataset_Himgsize.cam_intrinsics, 
                               dataset_imgsize.cam_intrinsics] 
                dw_scales = [4, 2, 1]
                LBA_max_iter = args.LBA_max_iter #10 # 20 
                LBA_step = args.LBA_step #.05 #.01 
                if LBA_max_iter <= 1: # do not do optimization
                    LBA_step = 0.  
                opt_vars = [args.opt_r, args.opt_t]

                # initialization for the first time window #
                if first_frame:
                    first_frame = False

                    # optimize the pose for all frames within the window #
                    if LBA_max_iter <=1 :# for debugging: using GT pose initialization #
                        rel_pose_inits_all_frame, srcs_idx_all_frame = m_misc.get_twin_rel_pose(  traj_extMs, idx_ref_, 
                                                                            t_win_r * dat_indx_step , 1, 
                                                                            use_gt_R = True, use_gt_t= True, 
                                                                            dataset= dataset, add_noise_gt = False, 
                                                                            noise_sigmas=None) 
                    else:
                        rel_pose_inits_all_frame, srcs_idx_all_frame = m_misc.get_twin_rel_pose(  traj_extMs, ref_indx, 
                                                                            t_win_r * dat_indx_step, 1, 
                                                                            use_gt_R = False , use_gt_t= False, 
                                                                            dataset= dataset, ) 
                    # opt. #
                    img_ref = dataset[ref_indx]['img'] 
                    imgs_src = [ dataset[i]['img'] for i in srcs_idx_all_frame ] 
                    conf_map_ref = torch.exp(conf_map_ref).squeeze() **2
                    rel_pose_opt = opt_pose_numerical.local_BA_direct(
                            img_ref, imgs_src, 
                            dmap_ref.unsqueeze(0).unsqueeze(0), 
                            conf_map_ref.unsqueeze(0).unsqueeze(0), cams_intrin, 
                            dw_scales, rel_pose_inits_all_frame, 
                            max_iter = LBA_max_iter, step = LBA_step, opt_vars= opt_vars ) 

                    # update #
                    for idx, srcidx in enumerate(srcs_idx_all_frame ):
                        traj_extMs[srcidx] = np.matmul(rel_pose_opt[idx].cpu().numpy(), traj_extMs[ref_indx])

                # for next frame #
                if LBA_max_iter<=1:# for debugging: using GT pose init.
                    rel_pose_opt, srcs_idx = m_misc.get_twin_rel_pose(traj_extMs, idx_ref_, 
                                                                      t_win_r, dat_indx_step, 
                                                                      use_gt_R = True, 
                                                                      use_gt_t= True, 
                                                                      dataset = dataset, 
                                                                      add_noise_gt = False, 
                                                                      noise_sigmas = None, ) 
                else:
                    rel_pose_inits, srcs_idx = m_misc.get_twin_rel_pose(traj_extMs, idx_ref_, 
                                                                        t_win_r, dat_indx_step, 
                                                                        use_gt_R =  args.use_gt_R,
                                                                        use_dso_R = args.use_dso_R, 
                                                                        use_gt_t =  args.use_gt_t, 
                                                                        use_dso_t = args.use_dso_t,
                                                                        dataset= dataset, 
                                                                        traj_extMs_dso = traj_extMs_dso, 
                                                                        opt_next_frame = args.opt_next_frame) 

                    img_ref = dataset[idx_ref_]['img'] 
                    _, src_dats_opt = m_misc.split_frame_list(dat_array, t_win_r)
                    imgs_src = [ dat_['img'] for dat_ in src_dats_opt ] 
                    img_ref = dataset[idx_ref_]['img'] 
                    imgs_src = [ dataset[i] for i in srcs_idx ]
                    imgs_src = [img_['img'] for img_ in imgs_src ] 

                    # opt. #
                    conf_map_kf = torch.exp(conf_map_kf).squeeze() **2
                    rel_pose_opt = \
                            opt_pose_numerical.local_BA_direct_parallel(
                            img_ref, imgs_src, 
                            dmap_kf.unsqueeze(0).unsqueeze(0), 
                            conf_map_kf.unsqueeze(0).unsqueeze(0), cams_intrin, 
                            dw_scales, rel_pose_inits, max_iter = LBA_max_iter, 
                            step = LBA_step, opt_vars = opt_vars) 

                # update # 
                print('idx_ref_: %d'%( idx_ref_ ) )
                print('srcs_idx : ')
                print(srcs_idx)

                print('updating pose ...')
                for idx, srcidx in enumerate(srcs_idx):
                    traj_extMs[srcidx] = np.matmul(rel_pose_opt[idx].cpu().numpy(), traj_extMs[idx_ref_])
                print('done') 


            else: # if the sequence contains invalid pose estimation 
                BVs_predict = None
                print('frame_cnt :%d, include invalid poses'%(frame_cnt)) 
                # UPDATE dat_array #
                if dat_indx_step > 1: # use one-interval video and the frame interval is larger than 5
                    print('updating array ...')
                    dat_array = update_dat_array(dat_array, dataset, 
                            data_interv= 1, frame_interv=5, ref_indx=ref_indx, t_win_r = t_win_r ) 
                    print('done')

                else:
                    dat_array.pop(0)
                    new_dat = dataset[ref_indx + t_win_r +1 ]
                    dat_array.append(new_dat)
        m_misc.save_ScenePathInfo( '%s/scene_path_info.txt'%(res_fldr), scene_path_info )

if __name__ == '__main__':
    main()
