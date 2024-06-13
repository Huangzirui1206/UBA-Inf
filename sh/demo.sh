python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.004 --attack_target 6

python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device cuda:3 --ft_epoch 60 --ap_epochs 6

// Get the pre-unlearning model
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2

// Get the post-unlearning model
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2 --c_num 0