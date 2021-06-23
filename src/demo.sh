# EDSR baseline model (x2) + JPEG augmentation
python main.py --model CNSR  --data_test Set5 --scale 2 --patch_size 128  --pre_train /mnt/ori_home/caoxiang112/CNSR/experiment/x2/model/model_best.pt --reset --save_results --test_only
#python main.py --model CNSR  --data_test DIV2K --data_range 1-3650 --scale 4 --patch_size 128  --pre_train /mnt/ori_home/caoxiang112/CNSR/experiment/x4/model/model_best.pt --reset --save_results --test_only
#python main.py --model CNSR --scale 2 --patch_size 64  --batch_size 64 --save x2 --reset --save_results --loss 1*L1



