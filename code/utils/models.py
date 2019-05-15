'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''


import torch 
import torch.nn as nn

def load_pretrained_PSMNet(model_feat_extractor, pretrained_path):
    '''
    Load the pretrained feature extractor 
    inputs:
    model_feat_extractor - instance of Feat_Extractor
    pretrained_path - path of the pretrained model 
    '''
    pre_model_dict_info = torch.load( pretrained_path) 
    if isinstance(pre_model_dict_info, dict):
        pre_model_dict = pre_model_dict_info['state_dict']
    else:
        pre_model_dict = pre_model_dict_info

    model_dict = model_feat_extractor.state_dict();
    pre_model_dict_feat = {k:v for k,v in pre_model_dict.items() if k in model_dict};

    # update the entries #
    model_dict.update( pre_model_dict_feat)
    # load the new state dict #
    model_feat_extractor.load_state_dict( pre_model_dict_feat )

    # print #
    print('load_pretrained_PSMNet():')
    print( '# of modules in the pre-trained model: %d'%( len(pre_model_dict.items())) )
    print( '# of modules in the feature-extraction: %d'%( len(pre_model_dict_feat.items())) )
    print('\n')

def load_pretrained_model(model, pretrained_path, optimizer = None):
    r'''
    load the pre-trained model, if needed, also load the optimizer status
    '''
    pre_model_dict_info = torch.load(pretrained_path) 
    if isinstance(pre_model_dict_info, dict):
        pre_model_dict = pre_model_dict_info['state_dict']
    else:
        pre_model_dict = pre_model_dict_info

    model_dict = model.state_dict();
    pre_model_dict_feat = {k:v for k,v in pre_model_dict.items() if k in model_dict};

    # update the entries #
    model_dict.update( pre_model_dict_feat)
    # load the new state dict #
    model.load_state_dict( pre_model_dict_feat )

    if optimizer is not None:
        optimizer.load_state_dict(pre_model_dict_info['optimizer'])
        print('Also loaded the optimizer status')
