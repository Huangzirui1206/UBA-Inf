// Get the dataset for training SISA model with BadNets on CIFAR-10
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset_sisa_3 --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.006 --attack_target 6

// Construct related camouflage smaples
python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset_sisa_3 --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6

// Get SISA models, including both pre-unlearning and post-unlearning ones
python ./uba/perturb_attack_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset_sisa_3 --save_folder_name perturb_badnet_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device cuda:2 --num_shards 3
