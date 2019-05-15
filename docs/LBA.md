##### Run DSO to get the pose esimations & Run inference with Local Bundle Adjustment 

To run on the demo data (`7scenes office seq 1`):

(1) Run DSO to get the initial poses

```
python run_dso.py --dso_path ../third_party/dso/build/bin/dso_dataset \
                  --data_fldr ../data/7scenes_office_seq_01 \
                  --cam_info_file DSO/cam_info_7scenes.mat \
                  --res_path ./dso_res \
                  --name_pattern *.color* \
                  --maxframe 500 
```

The parameters for `run_dso.py`:

`dso_path` : the built executable for DSO

`data_fldr` : the image folder

`name_pattern` : the image name pattern, e.g. `*.color.png` for `frame_#.color.png` for 7scene and scanNet name pattern 

`res_path` : the folder path to save the dso results

`maxframe` : the maximal # of frames we will deal with


(2) Test KV net with LBA
```
python3 test_KVNet_LBA.py \
 		--exp_name LBA_demo \
 		--sigma_soft_max 10 \
 		--t_win 2 \
 		--d_min .1  --d_max 5. \
 		--feature_dim 64  --ndepth 64 \
 		--dataset single_folder --dataset_path ../data/7scenes_office_seq_01 \
        --img_name_pattern *.color.png \
        --intrin_path ./DSO/cam_info_7scenes.mat \
 		--LBA_max_iter 20  --LBA_step 0.01 \
 		--opt_r 0  --opt_t 1 \
        --use_dso_R  \
 		--opt_next_frame \
 		--min_frame_idx 0  --max_frame_idx 498 \
 		--refresh_frames 200 \
		--model_path  ./saved_models/kvnet_scannet.tar
```

The parameters for `test_KVNet_LBA_.py`

`exp_name`: the name for the experiment. The results will be saved in `../results/${exp_name}`

`sigma_soft_max`: the sigma value used for getting the DPV. Should be the same as during training. 

`t_win`: the time window radius. The time window will be centered around the current frame

`d_min, d_max` : the minimal and maximal depth values

`feature_dim`: PSM feature dimension in DNet, should be the same as in the training session

`dataset`: the dataset name. Should one of `scanNet, 7scenes, kitti, single_folder`. For single folder, the `dataset_path` is the path of the folder that includes the input rgb images

`dataset_path`: the path to the specified dataset

`model_path`: the path to the trained model

`img_name_pattern` : used for the case where `dataset == single_folder`. The image name pattern, e.g. `*.color.png` for `framecolor.png for` 7scene and scanNet name pattern 

`intrin_path`: the `.mat` file including the camera intrinsic

`LBA_max_iter, LBA_step` : the maximal iteration and step size for the optimization in LBA 

`opt_r, opt_t`: whether optimizing rotation R and translation. 0: not optimize (use initial guess, if there is any). 1: optimize 

`use_dso_R, use_dso_t`: whether to use R or t from DSO as the initializations

`opt_next_frame`: also optmize the the frame right next to the current frame. This is benificial for propagating the depth probability volume and getting stable results

`refresh_frame`: refreshing' the pose estimation every `${refresh_frame}` frames. By 'refreshing', we mean resetting the pose estimation to the DSO estimation. This is a simple hack for dealing with the drifting issue.

`min_frame_idx, max_frame_idx` : the min/max index for the images we are going to test
