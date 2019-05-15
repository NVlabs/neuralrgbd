import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from models.psm_submodule import *
import warping.homography as warp_homo
import math
import time

import models.m_submodule as m_submodule

class feature_extractor(nn.Module):
    def __init__(self, feature_dim = 32, bn_running_avg = False, multi_scale= False ):
        ''' 
        inputs:
        multi_scale - if output multi-sclae features: 
        [1/4 scale of input image, 1/2 scale of input image] 
        '''

        super(feature_extractor,self).__init__()
        print('bn_running_avg = %d'%(bn_running_avg))
        self.feature_extraction = feature_extraction( feature_dim,
                                                      bn_running_avg = bn_running_avg, 
                                                      multi_scale = multi_scale) 
        self.multi_scale = multi_scale

        # initialization for the weights #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, img):
        if not self.multi_scale:
            feat_img = self.feature_extraction( img )
            return feat_img 
        else:
            feat_img_layer1, feat_img_final = self.feature_extraction(img)
            return feat_img_layer1, feat_img_final

class KV_NET_BASIC(nn.Module):
    '''
    The KV_NET approximate the KV matrix in the Kalman Filter
    Gain = KV_NET( h_{t} - W h_{t-1} )
    '''

    def __init__(self, input_volume_channels, feature_dim = 32, if_normalize = \
                 False, up_sample_ratio = None):
        '''
        inputs:
        input_volume_channels - the # of channels for the input volume
        '''
        super(KV_NET_BASIC, self).__init__()
        self.in_channels = input_volume_channels
        self.if_normalize = if_normalize
        self.up_sample_ratio = up_sample_ratio

        # The basic 3D-CNN in PSM-net #
        self.dres0 = nn.Sequential(convbn_3d(input_volume_channels, feature_dim, 3, 1, 1),
                                     nn.ReLU(),
                                     convbn_3d(feature_dim, feature_dim, 3, 1, 1),
                                     nn.ReLU())

        self.dres1 = nn.Sequential(convbn_3d(feature_dim, feature_dim, 3, 1, 1),
                                   nn.ReLU(),
                                   convbn_3d(feature_dim, feature_dim, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(feature_dim, feature_dim, 3, 1, 1),
                                   nn.ReLU(),
                                   convbn_3d(feature_dim, feature_dim, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(feature_dim, feature_dim, 3, 1, 1),
                                   nn.ReLU(),
                                   convbn_3d(feature_dim, feature_dim, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(feature_dim, feature_dim, 3, 1, 1),
                                   nn.ReLU(),
                                   convbn_3d(feature_dim, feature_dim, 3, 1, 1)) 

        self.classify = nn.Sequential(convbn_3d(feature_dim, feature_dim, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv3d(feature_dim, 1, kernel_size=3, padding=1, stride=1,bias=False))

        # initialization for the weights #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
    def forward(self, input_volume):
        '''
        inputs:
        input_volume - multi-channel 3D volume. size: N C D H W

        outputs:
        res_volume - single-channel 3D volume. size: N 1 D H W
        '''
        assert input_volume.shape[1] == self.in_channels, 'Input volume should have correct # of channels !'
        N,C,D,H,W = input_volume.shape

        input_volume = input_volume.contiguous()

        # cost: the intermidiate results #
        cost0 = self.dres0(input_volume)
        cost1 = self.dres1(cost0) + cost0
        cost2 = self.dres2(cost1) + cost1 
        cost3 = self.dres3(cost2) + cost2 
        cost4 = self.dres4(cost3) + cost3
        res_volume = self.classify(cost4)
        if self.if_normalize:
            res_volume = F.log_softmax(res_volume, dim=2)
        if self.up_sample_ratio is not None:
            # right now only up-sample in the D dimension #
            output_sz = (self.up_sample_ratio * D, H, W)
            res_volume = F.upsample(res_volume, output_sz, mode='trilinear', align_corners=True)
        return res_volume 

class D_NET_BASIC(nn.Module):
    def __init__(self, feature_extraction, cam_intrinsics, d_candi,
                 sigma_soft_max, BV_log = False, normalize = True,
                 use_img_intensity = False, force_img_dw_rate = 1, 
                 parallel_d = True, output_features = False, 
                 refine_costV = False, feat_dist = 'L2'):
        '''
        INPUTS: 

        feature_extraction - the feature extrator module

        cam_intrinsic - {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
            hfov, vfov - fovs in horzontal and vertical directions (degrees)
            unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
            unit ray pointing from the camera center to the pixel

        d_candi - np array of candidate depths 

        output_features - if output the features from the feature extractor. If ture, forward() will also return multi-scale 
                          image features (.25 and .5 image sizes) from the feature extractor
                          In this case, the output features will be saved in a list: [ img_feat_final, img_feat_layer1]
                          where img_feat_layer1 is the .5 image size feature

        refine_costV - if do the optional convolutions to refine costV before soft_max(costV)

        feat_dist - 'L2' (default) or 'L1' distance for feature matching

        '''

        super(D_NET_BASIC, self).__init__()
        self.feature_extraction = feature_extraction
        self.cam_intrinsics = cam_intrinsics
        self.d_candi = d_candi
        self.sigma_soft_max = sigma_soft_max
        self.BV_log = BV_log
        self.normalize = normalize
        self.use_img_intensity = use_img_intensity
        self.parallel_d = parallel_d
        self.output_features = output_features
        self.refine_costV = refine_costV
        self.feat_dist = feat_dist
        self.refine_costV = refine_costV

        if force_img_dw_rate > 1:
            self.force_img_dw_rate = force_img_dw_rate # Force to downsampling the input images
        else:
            self.force_img_dw_rate = None

        if self.refine_costV:
            D = len(d_candi) 
            self.conv0 = m_submodule.conv2d_leakyRelu(
                    ch_in = D, ch_out = D, kernel_size=3, stride=1, pad=1, use_bias=True) 
            self.conv0_1 = m_submodule.conv2d_leakyRelu(
                    ch_in= D, ch_out= D, kernel_size=3, stride=1, pad=1, use_bias=True) 
            self.conv0_2 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1, bias=True) 
            self.apply(self.weight_init)

    def _weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            print(' RefineNet_UNet2D: init conv2d')
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            print(' init Batch2D')
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            print(' init Linear')
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            print(' init transposed 2d')
            n = m.kernel_size[1]
            factor = (n+1) // 2
            if n %2 ==1:
                center = factor - 1
            else:
                center = factor -.5 

            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            m.weight.data.copy_(torch.from_numpy(weights_np)) 

    def forward(self,ref_frame, src_frames, src_cam_poses, cam_intrinsics = None, BV_predict = None , debug_ipdb = False):
        '''
        Inputs

        ref_frame - NCHW format tensor on GPU, N=1
        
        src_frames - NVCHW: V - # of source views, N - usually 1 

        src_cam_poses - N x V x4 x4 - relative cam poses
        
        [relative_extM_src0, ...  ]
        BV_predict - NDHW tensor, the predicted BV, from the last reference frame 

        Outputs:
        BV - The probability cost volume for the reference view size: N x D x H x W
        '''

        assert src_frames.shape[0] ==1, 'dim0 of src_frames should be 0'

        # Do feature extraction for all frames #

        if self.output_features:
            feat_imgs_layer_1, feat_imgs = self.feature_extraction(torch.cat((src_frames[0,...], ref_frame),dim=0))
            feat_img_ref_layer1 = feat_imgs_layer_1[-1,...].unsqueeze(0)

        else:
            feat_imgs = self.feature_extraction(torch.cat((src_frames[0,...], ref_frame),dim=0))

        feat_imgs_src = feat_imgs[:-1, ...].unsqueeze(0)
        feat_img_ref = feat_imgs[-1, ...].unsqueeze(0) 

        if self.use_img_intensity:
            # Get downsampling rate for image intensity feature #
            dw_rate = int( ref_frame.shape[3] / feat_img_ref.shape[3] )

            # Use image intensity as one set of features #
            img_int_feat_ref = F.avg_pool2d( ref_frame, dw_rate)
            feat_img_ref = torch.cat((feat_img_ref, img_int_feat_ref), dim = 1) # feat_img_ref size = [NCHW]

            img_int_feats_src = F.avg_pool2d(src_frames[0,...], dw_rate).unsqueeze(0)
            feat_imgs_src = torch.cat( (feat_imgs_src, img_int_feats_src), dim=2 ) # feat_imgs_src size = [NVCHW]


        Rs_src = src_cam_poses[0, :, :3, :3]
        ts_src = src_cam_poses[0, :, :3, 3]


        if cam_intrinsics is None:
            costV = warp_homo.est_swp_volume_v4( \
                    feat_img_ref, 
                    feat_imgs_src, 
                    self.d_candi, Rs_src, ts_src,
                    self.cam_intrinsics,
                    self.sigma_soft_max,
                    feat_dist = self.feat_dist,
                    debug_ipdb = debug_ipdb)

        else: # use the cam_intrinscs from the input. For scan-net this might
              # be different for different trajectories
            costV = warp_homo.est_swp_volume_v4( \
                    feat_img_ref, 
                    feat_imgs_src, 
                    self.d_candi, Rs_src, ts_src,
                    cam_intrinsics,
                    self.sigma_soft_max,
                    feat_dist = self.feat_dist,
                    debug_ipdb = debug_ipdb)


        if self.refine_costV:
            costv_out0 = self.conv0( costV )
            costv_out1 = self.conv0_1( costv_out0)
            costv_out2 = self.conv0_2( costv_out1)
        else:
            costv_out2 = costV

        if self.BV_log:
            BV = F.log_softmax(-costv_out2, dim=1)
        else:
            BV = F.softmax(-costv_out2, dim=1)

        if BV_predict is not None: 
            # Filtering Framework #
            if not self.BV_log: # if not log-scale
                BV = BV * BV_predict 
                # normalize #
                BV = BV / torch.sum( BV, dim=1).unsqueeze_(1)
            else: # BV in log-scale 
                BV = BV + BV_predict
                if self.normalize:
                    # normalize #
                    BV = F.log_softmax(BV, dim=1)

        if self.output_features:
            if self.use_img_intensity:
                return BV, [feat_img_ref[:,:-3, :,:], feat_img_ref_layer1, ]
            else:
                return BV, [feat_img_ref, feat_img_ref_layer1, ] 

        else:
            return BV

class baseline0(nn.Module):
    def __init__(self, feature_extraction, cam_intrinsics, d_candi,
                 sigma_soft_max, BV_log = False, normalize = True,
                 use_img_intensity = False,
                 force_img_dw_rate = 1,
                 parallel_d = True):
        '''
        feature_extraction - the feature extrator module

        cam_intrinsic - {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
            hfov, vfov - fovs in horzontal and vertical directions (degrees)
            unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
            unit ray pointing from the camera center to the pixel

        d_candi - np array of candidate depths 
        '''

        super(baseline0, self).__init__()
        self.feature_extraction = feature_extraction
        self.cam_intrinsics = cam_intrinsics
        self.d_candi = d_candi
        self.sigma_soft_max = sigma_soft_max
        self.BV_log = BV_log
        self.normalize = normalize
        self.use_img_intensity = use_img_intensity
        self.parallel_d = parallel_d

        if force_img_dw_rate > 1:
            self.force_img_dw_rate = force_img_dw_rate # Force to downsampling the input images
        else:
            self.force_img_dw_rate = None

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def __addInfo_cam_intrinsics(self):
        r'''
        Add fields for the cam_intrinsics, such as
        cam_intrinsic -cam_intrinsic['intrinsic_M_cuda'] # intrinsic matrix 3x3 on GPU
        cam_intrinsic['unit_ray_array_2D'] # unit ray array in matrix form on GPU
        '''
        pass 


    def set_cam_intrinsics(self, cam_intrinsics):
        '''
        Set the camera intrinsics
        This function is useful when we want to change the camera intrinsics during training,
        since the camera intrinsics might be different for different sections of data in the same
        training dat

        Inputs:
        cam_intrinsic - {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
            hfov, vfov - fovs in horzontal and vertical directions (degrees)
            unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
            unit ray pointing from the camera center to the pixel

        cam_intrinsic -cam_intrinsic['intrinsic_M_cuda'] # intrinsic matrix 3x3 on GPU
        cam_intrinsic['unit_ray_array_2D'] # unit ray array in matrix form on GPU
        '''
        self.cam_intrinsics = cam_intrinsics


    def forward(self,ref_frame, src_frames, src_cam_poses, 
                cam_intrinsics = None, BV_predict = None , debug_ipdb = False, 
                d_candi=None):
        '''
        Inputs
        ref_frame - NCHW format tensor on GPU, N=1
        
        src_frames - list of src frames, each in the same format as ref_frame

        src_cam_poses - list of relative camera poses, from ref. view to src.
        view, in other words, the relative camera poses of the source view
        w.r.t to the reference view
        
        [relative_extM_src0, ...  ]
        BV_predict - the predicted BV, from the last reference frame 

        Outputs:
        BV - The probability cost volume for the reference view
              size: D x H x W
        '''
        # Do feature extraction for all frames # 
        frames_= [src_frame.cuda() for src_frame in src_frames]
        frames_.append( ref_frame.cuda() )
        frames_ = torch.cat(frames_, dim=0 )

        if self.force_img_dw_rate is not None:
            frames_ = F.avg_pool2d(frames_, self.force_img_dw_rate)


        features_ = self.feature_extraction( frames_ )
        feat_img_ref = features_[-1, :, :, :].unsqueeze_(0)


        list_feat_imgs_src = torch.chunk( features_[:len(src_frames), :, :, :], len(src_frames))
        list_feat_imgs_src = list(list_feat_imgs_src)

        if debug_ipdb:
            # The first feature channel is the visibility mask #
            mask_vis = torch.ones((1,1, feat_img_ref.shape[2], feat_img_ref.shape[3])).cuda()
            feat_img_ref = torch.cat((mask_vis, feat_img_ref), dim=1)
            list_feat_imgs_src = [torch.cat((mask_vis,
                                             list_feat_imgs_src[idx]), dim = 1)
                                      for idx in range(0, len(src_frames))]

        if self.use_img_intensity:
            assert self.force_img_dw_rate is None, \
            'use image intensity can not be used together with downsampling'

        if self.use_img_intensity and self.force_img_dw_rate is None:
            # Get downsampling rate for image intensity feature #
            dw_rate = int( ref_frame.shape[3] / feat_img_ref.shape[3] )

            # Use image intensity as one set of features #
            img_int_feat_ref = F.avg_pool2d(frames_[-1], dw_rate).unsqueeze(0)
            feat_img_ref = torch.cat((feat_img_ref, img_int_feat_ref), dim = 1)

            img_int_feats_src = [F.avg_pool2d(frames_[idx],
                                              dw_rate).unsqueeze(0) for idx \
                                 in range(0, len(src_frames))]

            list_feat_imgs_src = [torch.cat(( list_feat_imgs_src[idx],
                                             img_int_feats_src[idx]), dim = 1)
                                  for idx in range(0, len(src_frames))]

        Rs_src = [pose[:3, :3] for pose in src_cam_poses ]
        ts_src = [pose[:3, 3] for pose in src_cam_poses ]

        # Get BV # 
        if d_candi is None:
            d_candi = self.d_candi

        if cam_intrinsics is None:
            costV = warp_homo.est_swp_volume_v3( \
                    feat_img_ref, 
                    list_feat_imgs_src, 
                    d_candi, Rs_src, ts_src,
                    self.cam_intrinsics,
                    self.sigma_soft_max,
                    if_par_d= self.parallel_d,
                    debug_ipdb = debug_ipdb)
        else: # use the cam_intrinscs from the input. For scan-net this might
              # be different for different trajectories
            costV = warp_homo.est_swp_volume_v3( \
                    feat_img_ref, 
                    list_feat_imgs_src, 
                    d_candi, Rs_src, ts_src,
                    cam_intrinsics,
                    self.sigma_soft_max,
                    if_par_d= self.parallel_d,
                    debug_ipdb = debug_ipdb)


        if self.BV_log:
            BV = F.log_softmax(-costV, dim=1)
        else:
            BV = F.softmax(-costV, dim=1)

        if BV_predict is not None: 
            # Filtering Framework #
            if not self.BV_log: # if not log-scale
                BV = BV * BV_predict 
                # normalize #
                BV = BV / torch.sum( BV, dim=1).unsqueeze_(1)
            else: # BV in log-scale 
                BV = BV + BV_predict
                if self.normalize:
                    # normalize #
                    BV = F.log_softmax(BV, dim=1)
        return BV

class baseline0_disp(nn.Module):
    def __init__(self, feature_extraction, cam_intrinsics, 
                 sigma_soft_max, BV_log = False, normalize = True,
                 use_img_intensity = False,
                 force_img_dw_rate = 1,
                 parallel_d = True):
        '''
        feature_extraction - the feature extrator module, pretrained in PSM net 

        cam_intrinsic - {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
            hfov, vfov - fovs in horzontal and vertical directions (degrees)
            unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
            unit ray pointing from the camera center to the pixel

        '''
        super(baseline0_disp, self).__init__()
        self.feature_extraction = feature_extraction
        self.cam_intrinsics = cam_intrinsics
        self.sigma_soft_max = sigma_soft_max
        self.BV_log = BV_log
        self.normalize = normalize
        self.use_img_intensity = use_img_intensity
        self.parallel_d = parallel_d

        if force_img_dw_rate > 1:
            self.force_img_dw_rate = force_img_dw_rate # Force to downsampling the input images
        else:
            self.force_img_dw_rate = None

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def __addInfo_cam_intrinsics(self):
        r'''
        Add fields for the cam_intrinsics, such as
        cam_intrinsic -cam_intrinsic['intrinsic_M_cuda'] # intrinsic matrix 3x3 on GPU
        cam_intrinsic['unit_ray_array_2D'] # unit ray array in matrix form on GPU
        '''
        # add intrinisc_M_cuda #
#        if not 'intrinsic_M_cuda' in self.cam_intrinsics:
#            self.cam_intrinsics['intrinsic_M_cuda'] = 
        # add unit_ray_array_2D #
        pass 

    def set_cam_intrinsics(self, cam_intrinsics):
        '''
        Set the camera intrinsics
        This function is useful when we want to change the camera intrinsics during training,
        since the camera intrinsics might be different for different sections of data in the same
        training dat

        Inputs:
        cam_intrinsic - {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
            hfov, vfov - fovs in horzontal and vertical directions (degrees)
            unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
            unit ray pointing from the camera center to the pixel

        cam_intrinsic -cam_intrinsic['intrinsic_M_cuda'] # intrinsic matrix 3x3 on GPU
        cam_intrinsic['unit_ray_array_2D'] # unit ray array in matrix form on GPU
        '''
        self.cam_intrinsics = cam_intrinsics


    def forward(self,ref_frame, src_frames, src_cam_poses, 
                d_candi, BV_predict = None, debug_ipdb = False):
        '''
        Inputs
        ref_frame - NCHW format tensor on GPU, N=1
        
        src_frames - list of src frames, each in the same format as ref_frame

        src_cam_poses - list of relative camera poses, from ref. view to src.
        view, in other words, the relative camera poses of the source view
        w.r.t to the reference view
        
        [relative_extM_src0, ...  ]
        BV_predict - the predicted BV, from the last reference frame 

        Outputs:
        BV - The probability cost volume for the reference view
              size: D x H x W
        '''
        # Do feature extraction for all frames # 
        frames_= [src_frame.cuda() for src_frame in src_frames]
        frames_.append( ref_frame.cuda() )
        frames_ = torch.cat(frames_, dim=0 )

        if self.force_img_dw_rate is not None:
            frames_ = F.avg_pool2d(frames_, self.force_img_dw_rate)


        features_ = self.feature_extraction( frames_ )
        feat_img_ref = features_[-1, :, :, :].unsqueeze_(0)


        list_feat_imgs_src = torch.chunk( features_[:len(src_frames), :, :, :], len(src_frames))
        list_feat_imgs_src = list(list_feat_imgs_src)

        if debug_ipdb:
            # The first feature channel is the visibility mask #
            mask_vis = torch.ones((1,1, feat_img_ref.shape[2], feat_img_ref.shape[3])).cuda()
            feat_img_ref = torch.cat((mask_vis, feat_img_ref), dim=1)
            list_feat_imgs_src = [torch.cat((mask_vis,
                                             list_feat_imgs_src[idx]), dim = 1)
                                      for idx in range(0, len(src_frames))]

        if self.use_img_intensity:
            assert self.force_img_dw_rate is None, \
            'use image intensity can not be used together with downsampling'

        if self.use_img_intensity and self.force_img_dw_rate is None:
            # Get downsampling rate for image intensity feature #
            dw_rate = int( ref_frame.shape[3] / feat_img_ref.shape[3] )

            # Use image intensity as one set of features #
            img_int_feat_ref = F.avg_pool2d(frames_[-1], dw_rate).unsqueeze(0)
            feat_img_ref = torch.cat((feat_img_ref, img_int_feat_ref), dim = 1)

            img_int_feats_src = [F.avg_pool2d(frames_[idx],
                                              dw_rate).unsqueeze(0) for idx \
                                 in range(0, len(src_frames))]

            list_feat_imgs_src = [torch.cat(( list_feat_imgs_src[idx],
                                             img_int_feats_src[idx]), dim = 1)
                                  for idx in range(0, len(src_frames))]

        Rs_src = [pose[:3, :3] for pose in src_cam_poses ]
        ts_src = [pose[:3, 3] for pose in src_cam_poses ]

        costV = warp_homo.est_swp_volume_v3( \
                feat_img_ref, 
                list_feat_imgs_src, 
                d_candi, Rs_src, ts_src,
                self.cam_intrinsics,
                self.sigma_soft_max,
                if_par_d= self.parallel_d,
                debug_ipdb = debug_ipdb)

        if self.BV_log:
            BV = F.log_softmax(-costV, dim=1)
        else:
            BV = F.softmax(-costV, dim=1)

        if BV_predict is not None: 
            # Filtering Framework #
            if not self.BV_log: # if not log-scale
                BV = BV * BV_predict 
                # normalize #
                BV = BV / torch.sum( BV, dim=1).unsqueeze_(1)
            else: # BV in log-scale 
                BV = BV + BV_predict
                if self.normalize:
                    # normalize #
                    BV = F.log_softmax(BV, dim=1)
        return BV
