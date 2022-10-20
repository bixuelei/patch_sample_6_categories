# Motor_segmentation_net

This project introduces a way to solve industrial automation problems using deep learning. Industrial problems : the factory has collected a lot of discarded motors, but there are a lot of parts in them that can be reused, so we need to disassemble the discarded motors, so we need to get the exact location of the screws. Our solution is to use deep learning to semantically segment and utilize the 3D point cloud collected by the sensor. The result of the division gives the position of the screw

# Environments Requirement

CUDA = 10.2
Python = 3.7.0
PyTorch = 1.6
Open3d
tpdm
API(kmeans_pytorch) 

For pytorch, you could directly install all the dependence according to the instruction on pytorch official website. After you install the evironment successfully, you could use pip to install open3d and tpdm. For the kmeans_pytorch, you could follow the instruction(https://pypi.org/project/kmeans-pytorch/)


# How to run

## Training the pretraining model

You can use below command line to run the pretraining script and gain the pretraining model:
```
CUDA_VISIBLE_DEVICES=0,1 python train_seg.py --batch_size 16 --npoints 2048 --epoch 100 --model dgcnn_patch --lr 0.01   --exp_name STN_16_2048_100_Patch --factor_stn_los 0.01 --kernel_loss_weight 0.05 --use_class_weight 0 --bolt_weight 1 --which_dataset Dataset4 --num_segmentation_type 6 --emb_dims --train 1 --finetune 0 --root /home/ies/bi/data/Dataset4
```

| cmd  | Description          | Type | Property |
| ------- | ----------------------------------------------------------| --- | ---------- |
| -batch_size | batch size for training process                |      int |   obligatory      |
| -npoints   | number of points for sub point cloud               |     int  |      obligatory      |
| -epoch   |  training epoch                               | int      | obligatory |
| -model   | the model to be chosed                                 | string     | obligatory |
| -lr | initial learning rate                                 | string     | obligatory |
| -exp_name   | experimential name which include some parameters of current trained model  | string    | obligatory   |
| -sf   | scene file format, option: npy, pcd, both (default: npy)  | string | optional |
| -bb   | whether to save 3D bounding box of motor (default=True)    | boolean |  optional  |
| -sc   | whether to save cuboid file (default=True)     | boolen | optional |
| -cf   | cuboid file format, option: npy, pcd, both (default: npy)  | string | optional |
| -ri | default=True: apply random rotation info and save. True: load rotation info from given csv file  | boolen  | optional |
| -cp | if -ri is False, save directory of rotation info.(default is save directory). if -ri is True, path of given csv file | string | optional/obligatory |
| -n    | number of total generation (an integer multiple of 5)     | integer | obligatory  |

Explanation of every important parameter
* train_semseg.py: choose of which script will be run
* CUDA_VISIBLE_DEVICES: set the visible gpu 
* model: choose of which model will be used to train a pretraining model
* change: give the information of a specific experiment(here without rotation means that i dont use the STN Net, 16 represent bacht_size, 2048 represents points of every training unit, 50 represents the epoch
* exp: the paremeter means that i training the net with dataset(i generate) with size of 125 motors scene
* root: the root of training dataset

## Train the finetune model
You can use below command line to run the finetune script and gain the training result:
```
CUDA_VISIBLE_DEVICES=0,1 python train_seg.py --batch_size 16 --npoints 2048 --epoch 100 --model dgcnn_patch --lr 0.01   --exp_name STN_16_2048_100_Patch --factor_stn_los 0.01 --kernel_loss_weight 0.05 --use_class_weight 0 --bolt_weight 1 --which_dataset finetune --num_segmentation_type 6 --emb_dims --train 1 --finetune 0 --root /home/ies/bi/data/finetune
```
Here we have another parameter,finetune,which is set by default as False, if we set it as True, we will train the finetune model.

## Test the finetune model
You can use below command line to run the test finetune script and gain the test result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --model dgcnn_rotate_conv --change allaround_STN_conv   --exp training_125 --bolt_weight 1 --root /home/ies/bi/data/previous_test --finetune True --eval True
```
Explanation of every important parameter
* train_semseg_rotation.py: here we choose train_semseg_rotation.py, this means we add STN Net to whole architecture

# Inportant Info
If you want to debug the script, you should change the parameter manually. For PCT_Nico, it needs huge storage of gpu. It is advible to set batch_size as 2 and to set n_points as 1024, if you want to get to know how this net works.




# AgiProbot_Motor_Segmentation_WACV2023
