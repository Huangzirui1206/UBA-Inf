# More concrete usage examples for UBA-Inf evaluation.

# CIFAR-10

# badnet p_num 600(0.012) c_num 200(0.004)
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.004 --attack_target 6
python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2

# blended p_num 250(0.005) c_num 500(0.01)
python ./attack/blended.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name blended_dataset --add_cover 1 --epoch 00 --pratio 0.005 --cratio 0.01 --attack_target 6
python ./uba/uba_inf_cover.py --dataset_folder ../record/blended_dataset --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/blended_dataset --save_folder_name perturb_blended_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2

# sig p_num 200(0.04) c_num 200(0.004) # p_num in the original paper is 1000.
python ./attack/sig.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name sig_dataset --add_cover 1 --epoch 00 --pratio 0.04 --cratio 0.004 --attack_target 6
python ./uba/uba_inf_cover.py --dataset_folder ../record/sig_dataset --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/sig_dataset --save_folder_name perturb_sig_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2

# label-consistent p_num 300(0.06) c_num 300(0.006) # question: concrete number? In paper a reasonable p_num is 1000.
python ./attack/lc.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name lc_dataset --add_cover 1 --epoch 00 --pratio 0.06 --cratio 0.01 --attack_target 6
python ./uba/uba_inf_cover.py --dataset_folder ../record/lc_dataset --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/lc_dataset --save_folder_name perturb_lc_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2

### SISA

# sisa preactresnset
# badnet
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset_sisa_3 --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.006 --attack_target 6
python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset_sisa_3 --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6
python ./uba/perturb_attack_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset_sisa_3 --save_folder_name perturb_badnet_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device cuda:2 --num_shards 3

python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset_sisa_5 --add_cover 1 --epoch 00 --pratio 0.02 --cratio 0.01 --attack_target 6
python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset_sisa_5 --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6
python ./uba/perturb_attack_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset_sisa_5 --save_folder_name perturb_badnet_preactresnet_sisa_5  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device cuda:2 --num_shards 5


# MNIST
## get dataset 
 python ./attack/badnet.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.01 --cratio 0.004 --attack_target 6 --dataset mnist --model preactresnet18 --patch_mask_path ../resource/badnet/trigger_image_grid_for_mnist.png --device cuda:3


# GTSRB 
## get dataset 
python ./attack/badnet.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --attack_target 36 --dataset gtsrb --model preactresnet18 --device cuda:3


# Tiny 
## get dataset 
python ./attack/badnet.py --yaml_path ../config/attack/prototype/tiny.yaml  --save_folder_name tiny_badnet_poionly --add_cover 0 --epoch 120 --pratio 0.01 --attack_target 6 --dataset tiny --model preactresnet18 --patch_mask_path ../resource/badnet/trigger_image_grid_for_tiny.png --device cuda:3


# Defense example
python ./defense/nc.py --result_file ../record/gtsrb_perturb_badnet_preactresnet --result_name perturb_result.pt --model preactresnet18 --only_scan 1 --device cuda:2 --dataset gtsrb



# Narcissus
python ./attack/narcissus.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name narcissus_dataset_2 --add_cover 1 --epoch 00 --pratio 0.004 --cratio 0.01 --attack_target 2 --multi_test 2 --narcissus_noise_path ../resource/narcissus/checkpoint/narcissus_noise_cls_2.npy
