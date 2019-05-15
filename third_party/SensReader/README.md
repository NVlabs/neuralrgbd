
This is a modified version of [ScanNet decoder](https://github.com/ScanNet/ScanNet/tree/master/SensReader), used for geting the rgbd images and camera poses for training
and evaluation. Note that the original tool codes (including Sens data parser) is under [MIT License](https://github.com/ScanNet/ScanNet#license).

#### Setup
In the *SensReader* folder:
```
make
```
to get the executable `sens`

#### Run
In the *SensReader* folder:
Suppose the scanNet dataset is saved in `SCANNET_PATH` and the decoded output folder is `OUTPUT_PATH`

```
# decode samples in the training set
python decode.py --dataset_path SCANNET_PATH --output_path OUTPUT_PATH --split_file scannet_train.txt

# decode samples in the validation set
python decode.py --dataset_path SCANNET_PATH --output_path  OUTPUT_PATH --split_file scannet_val.txt 
``` 



