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
CUDA_VISIBLE_DEVICES=0,1 python train_seg.py --batch_size 16 --npoints 2048 --epoch 100 --model dgcnn_patch --lr 0.01   --exp_name STN_16_2048_100_Patch --factor_stn_los 0.01 --kernel_loss_weight 0.05 --use_class_weight 0 --bolt_weight 1 --which_dataset Dataset4 --num_segmentation_type 6 --emb_dims --train 1 --finetune 0 --data_dir /home/ies/bi/data/Dataset4
```

| cmd  | Description          | Type | Property |
| ------- | ----------------------------------------------------------| --- | ---------- |
| -batch_size | batch size for training process                |      int |   obligatory      |
| -npoints   | number of points for sub point cloud               |     int  |      obligatory      |
| -epoch   |  training epoch                               | int      | obligatory |
| -model   | the model to be chosed                                 | string     | obligatory |
| -lr | initial learning rate                                 | string     | obligatory |
| -exp_name   | experimential name which include some parameters of current trained model  | string    | obligatory   |
| -factor_stn_los  | the weigth of loss for STN Network  | float | obligatory |
| -kernel_loss_weight   | the weigth of loss for patch sample Network   | float |  obligatory  |
| -use_class_weight   | whether to use the class weight    | int | obligatory |
| -bolt_weight   | the bolts weights  | float | optional |
| -which_dataset | the current dataset you use to train the model | string  | obligaroty |
| -num_segmentation_type | the number of categories you want to classify | int | obligatory |
| -emb_dims   | the dimension for high features setting     | int  | obligatory  |
| -train   | if we are in the training process    | int | obligatory |
| -finetune   | if we are in the finetune process | int | obligatory |
| -data_dir   | the position where the dataset is stored | string | obligatory |

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
