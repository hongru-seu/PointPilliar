# PointPilliar with MindSpore
Fast Encoders for 3D Object Detection from Point Clouds
> [Paper](https://arxiv.org/abs/1812.05784):  PointPillars: Fast Encoders for Object Detection from Point Clouds.
> Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom, 2018.

## [architecture](#contents)

The main components of the network are a Pillar Feature Network, Backbone, and SSD detection head.
(1) The raw point cloud is converted to a stacked pillar tensor and pillar index tensor.
(2) The encoder uses the stacked pillars to learn a set of features that can be scattered back to a 2D pseudo-image for a convolutional neural network.
(3) The features from the backbone are used by the detection head to predict 3D bounding boxes for objects.

## [Dataset](#contents)

Dataset used: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

Data was collected with using a standard station wagon with two high-resolution color and grayscale video cameras.
Accurate ground truth is provided by a Velodyne laser scanner and a GPS localization system.
Dataset was captured by driving around the mid-size city of Karlsruhe, in rural areas and on highways.
Up to 15 cars and 30 pedestrians are visible per image. The 3D object detection benchmark consists of 7481 images.

## [Environment requirements](#contents)
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), data from [ImageSets](https://github.com/traveller59/second.pytorch/tree/master/second/data/ImageSets), put files from `ImageSets` into `pointpillars/src/data/ImageSets/`

### [Dataset Preparation](#contents)
1. Add `/path/to/pointpillars/` to your `PYTHONPATH`

```text
export PYTHONPATH=/path/to/pointpillars/:$PYTHONPATH
```

2. Download KITTI dataset into one folder:
- [Download left color images of object data set (12 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
- [Download camera calibration matrices of object data set (16 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- [Download training labels of object data set (5 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
- [Download Velodyne point clouds, if you want to use laser information (29 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
3. Unzip all downloaded archives.
4. Directory structure is as follows:

```text
└── KITTI
       ├── training
       │   ├── image_2 <-- for visualization
       │   ├── calib
       │   ├── label_2
       │   ├── velodyne
       │   └── velodyne_reduced <-- create this empty directory
       └── testing
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- create this empty directory
```

5. Download [ImageSets](https://github.com/traveller59/second.pytorch/tree/master/second/data/ImageSets), put files from `ImageSets` into `pointpillars/src/data/ImageSets/`

6. Create KITTI infos:

```shell
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
python det3DCreateData.py create_kitti_info_file --data_path=/home/hongru/0_dataset/KITTI 
```

7. Create reduced point cloud:

```shell
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
python Det3DCreateData.py create_reduced_point_cloud --data_path=/home/hongru/0_dataset/KITTI
```

8. Create groundtruth-database infos:

```shell
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
python Det3DCreateData.py create_groundtruth_database --data_path=/home/hongru/0_dataset/KITTI
```

9. The config file `car_xyres16.yaml` or `ped_cycle_xyres16.yaml` needs to be edited to point to the above datasets:

```text
train_input_reader:
  ...
  database_sampler:
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
...
eval_input_reader:
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
```

#### Train
Example:
python train.py \
  --is_distributed=1 \
  --device_target=GPU \
  --cfg_path=./configs/ped_cycle_xyres16.yaml \
  --save_path=./experiments/ped_cycle > ./experiments/ped_cycle/log.txt 2>&1 &
  
 #### Evaluate
 python eval.py \
  --cfg_path=./experiments/cars/car_xyres16.yaml \
  --ckpt_path=./experiments/cars/pointpillars-160_296960.ckpt > /workspace/4_hong/MINDSPORE/0_pointpillars/experiments/cars/log_eval.txt 2>&1 &
 
 ### [Kitti evalution of trainning set](#contents)
 
val set (len_dataset 3769)

        Easy   Mod    Hard
Car AP@0.70, 0.70, 0.70:
bbox AP:90.66, 88.81, 86.74
bev  AP:89.87, 86.74, 83.91
3d   AP:85.95, 76.30, 69.63
aos  AP:90.62, 88.49, 86.08

        Easy   Mod    Hard
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:84.64, 65.44, 62.60
bev  AP:82.76, 62.55, 59.06
3d   AP:77.64, 59.09, 54.52
aos  AP:84.04, 64.51, 61.75
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:66.41, 64.34, 60.69
bev  AP:72.28, 66.54, 61.98
3d   AP:66.80, 60.08, 54.70
aos  AP:41.43, 41.08, 39.00
