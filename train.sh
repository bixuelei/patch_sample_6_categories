
# finetune    

#  CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 300 --model PCT --lr 0.001 --change STN_16_2048_100_one_e2_cross_entropy_nodensify_dataset4 --use_weigth "" --use_sgd True  --exp dataset --factor_trans 0.01 --bolt_weight 1 --finetune True  --root /home/ies/bi/data/new_finetune



# finetune test

        #  CUDA_VISIBLE_DEVICES=7 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --model pointnet --change allaround_STN_16_2048_150   --exp dataset6 --bolt_weight 1 --root /home/ies/bi/data/previous_test --finetune True --eval True





# training       
CUDA_VISIBLE_DEVICES=5 python train_semseg_rotation.py --batch_size 8 --test_batch_size 16 --npoints 2048 --epoch 100 --model PCT_patch --lr 0.01   --change STN_16_2048_100_one_e2_cross_entropy_nodensify_nototate --factor_trans 0.01 --use_weigth "" --bolt_weight 15 --exp dataset --bolt_weight  1 --root /home/ies/bi/data/Dataset6
  # CUDA_VISIBLE_DEVICES=5,7 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 100 --model PCT --lr 0.01   --change allaround_STN_16_2048_100_one_more_edgeconv_cos_e2_dice_loss_and_cross --factor_trans 0.01 --use_weigth "" --exp dataset6 --bolt_weight  1 --root /home/ies/bi/data/Dataset6

