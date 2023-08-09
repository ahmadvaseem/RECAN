#!/bin/bash/
# For release
CUDA_VISIBLE_DEVICES=0 python main.py --data_test B100 --scale 2 --model RECAN --n_resgroups 5 --n_resblocks 10 --n_feats 64 --pre_train /home/waseem/projects/RECANv1.0/RECAN_TrainCode/experiment/RECANn_BIX2_G5R10P96/model/model_best.pt --test_only --save_results --chop --save 'RECAN' --testpath /home/waseem/projects/RECAN/testpath/ --testset B100

# RECAN_BIX2
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX2.pt --test_only --save_results --chop --save 'RECAN' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5
# RECAN_BIX3
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX3.pt --test_only --save_results --chop --save 'RECAN' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5
# RECAN_BIX4
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX4.pt --test_only --save_results --chop --save 'RECAN' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5
# RECAN_BIX8
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX8.pt --test_only --save_results --chop --save 'RECAN' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5
##
# RECANplus_BIX2
CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 2 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'RECANplus' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5
# RECANplus_BIX3
CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 3 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'RECANplus' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5
# RECANplus_BIX4
CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 4 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'RECANplus' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5
# RECANplus_BIX8
CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 8 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'RECANplus' --testpath /home/waseem/projects/RECAN/testpath/ --testset Set5

# BD degradation model, X3
# RECAN_BDX3
python main.py --data_test MyImage --scale 3 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BDX3.pt --test_only --save_results --chop --save 'RECAN' --testpath /home/waseem/projects/RECAN/testpath/ --degradation BD --testset Set5

# BD degradation model, X3
# RECANplus_BDX3
python main.py --data_test MyImage --scale 3 --model RECAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/waseem/projects/RECAN/RECAN_TestCode/model/RECAN_BDX3.pt --test_only --save_results --chop --self_ensemble  --save 'RECANplus' --testpath /home/waseem/projects/RECAN/testpath/ --degradation BD --testset Set5

