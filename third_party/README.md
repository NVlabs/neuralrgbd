##### 3rd party libraries for evaluations
To evaluate our method, we use [DSO](https://github.com/JakobEngel/dso) for camera pose estimation. DSO is under the [GPLv3.0](https://github.com/JakobEngel/dso/blob/master/LICENSE) license. We did slight modifications to the code an a patch is provided.
We also use the [ScanNet](https://github.com/ScanNet/ScanNet/blob/master/LICENSE) dataset and its data loaders [code](https://github.com/ScanNet/ScanNet/tree/master/SensReader) for both evaluation and training.


This folder includes:

- A patch code for [DSO](https://github.com/JakobEngel/dso), that improves robustness on our test data and dumps invalid poses, too.

- [ScanNet decoder](https://github.com/ScanNet/ScanNet/tree/master/SensReader), used for loading the rgbd images and camera poses during training  and evaluation.



##### Setup DSO
Assuming all the dependencies of DSO have been installed.
To clone, apply the customized changes and make DSO:
```
sh ./setup_dso.sh
```
This will build the `dso_dataset` executable file in `dso/build/bin` folder.

##### Setup SensReader
In the *SensReader* folder:
```
make
```
to get the executable `sens`.

##### Use SensReader
In the *SensReader* folder:
Suppose the scanNet dataset is saved in `SCANNET_PATH` and the decoded output folder is `OUTPUT_PATH`

```
# decode samples in the training set
python decode.py --dataset_path SCANNET_PATH --output_path OUTPUT_PATH --split_file scannet_train.txt

# decode samples in the validation set
python decode.py --dataset_path SCANNET_PATH --output_path  OUTPUT_PATH --split_file scannet_val.txt 
``` 
This will decode the .sens files into images with 5 frame intervals.
