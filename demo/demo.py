"""Summary
This is the python file version of demo.

You can train model manually from scratch by setting train_from_scratch True.
If you find the training is too time-consuming, you can just download the intermediate and model parameters online for evaluation.

For evaluation, 4 datasets are provided, including CIFAR-10, MNIST, GTSRB, and Tiny-ImageNet.
4 backdoor triggers are provided, including BadNets, Blended, LC and Sig.
Evaluation of normally-trained and SISA models are available in this demo.

Compared with original work, 3 backdoors are available (e.g. BadNets, Blended and Sig), since LC fails in several cases as demonstrated in Table 1 and Table2 in original paper.

For normally trained models, 9 cases in 3 datasets (e.g. CIFAR-10, MNIST and GTSRB) are presented with 3 different backdoor triggers.
Meanwhile, evaluations of BadNets in CIFAR-10 with 3 different model architectures (e.g. ResNet34, PreactResnet-18, and VGG-16) are demonstrated.

For SISA models, 12 cases in 3 datasets (e.g. CIFAR-10, MNIST and GTSRB) with 2 backdoors (e.g. BadNets and Blended) trained on 3-shard SISA or 5-shard SISA are available.

You can choose to train the models from scratch, or just download pre-trained models from website for quick evaluation.
"""

import os

# Choose you device for evaluation
device = "cuda"

"""
############################################################################
#                     Normally trained models (Table 1)                    #
############################################################################
"""

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     BadNets           |
| Dataset:      CIFAR-10          |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     BadNets           |
| Dataset:      CIFAR-10          |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/perturb_badnet_preactresnet"):
    print("Pre-unlearning model perturb_badnet_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_badnet_preactresnet from scratch.")
        if not os.path.exists("./record/badnet_dataset"):
            # Generate dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.004 --attack_target 6")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda")
    else:
        print("The model ./record/perturb_badnet_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/perturb_badnet_poionly"):
    print("Post-unlearning model perturb_badnet_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_badnet_poionly from scratch.")
        if not os.path.exists("./record/badnet_dataset"):
            print("Please download perturb_badnet_preactresnet and perturb_badnet_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device cuda --c_num 0")
    else:
        print("The model ./record/perturb_badnet_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # read from .csv result files
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_badnet_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_badnet_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_poionly --result_name perturb_result.pt --device {device}")

############################################################################

"""
+-------------------------------------------+
| Test UBA-Inf with BadNets on CIFAR-10     |
| with 3 different model arch.              |
| Model arch:   PARN18 / ResNet34 / VGG16   |
| Backdoor:     Blended                     |
| Dataset:      CIFAR-10                    |
+-------------------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+-------------------------------------------+
| Test UBA-Inf with BadNets on CIFAR-10     |
| with 3 different model arch.              |
| Model arch:   PARN18 / ResNet34 / VGG16   |
| Backdoor:     Blended                     |
| Dataset:      CIFAR-10                    |
+-------------------------------------------+
"""
)

# Get the result of preactresnet18 model
if not os.path.exists("./record/perturb_badnet_preactresnet"):
    print("Pre-unlearning model perturb_badnet_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_badnet_preactresnet from scratch.")
        if not os.path.exists("./record/badnet_dataset"):
            # Generate dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.004 --attack_target 6")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda")
    else:
        print("The model ./record/perturb_badnet_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False

# Get the result of resnet34 model
if not os.path.exists("./record/perturb_badnet_resnet"):
    print("Pre-unlearning model perturb_badnet_resnet doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_badnet_resnet from scratch.")
        if not os.path.exists("./record/badnet_dataset"):
            print("Please download perturb_badnet_preactresnet, perturb_badnet_resnet and perturb_badnet_vgg together!")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda")
    else:
        print("The model ./record/perturb_badnet_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the result of vgg16 model
if not os.path.exists("./record/perturb_badnet_vgg"):
    print("Pre-unlearning model perturb_badnet_vgg doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_badnet_vgg from scratch.")
        if not os.path.exists("./record/badnet_dataset"):
            print("Please download perturb_badnet_preactvgg, perturb_badnet_vgg and perturb_badnet_vgg together!")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_preactvgg  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactvgg18 --device cuda")
    else:
        print("The model ./record/perturb_badnet_preactvgg doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # read from .csv result files
        # Results before unlearning of preactresnet18:
        print("Results before unlearning of preactresnet18:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_badnet_preactresnet")
        
        # Results before unlearning of resnet34:
        print("Results before unlearning of resnet34:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_badnet_resnet")
        
        # Results before unlearning of vgg16:
        print("Results before unlearning of vgg16:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_badnet_vgg")
    else:
        # test the model from scratch
        # eval results before unlearning of preactresnet18:
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval results before unlearning of resnet34:
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_resnet --result_name perturb_result.pt --device {device}")
        
        # eval results before unlearning of vgg16:
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_vgg --result_name perturb_result.pt --device {device}")

############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Blended           |
| Dataset:      CIFAR-10          |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Blended           |
| Dataset:      CIFAR-10          |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/perturb_blended_preactresnet"):
    print("Pre-unlearning model perturb_blended_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_blended_preactresnet from scratch.")
        if not os.path.exists("./record/blended_dataset"):
            # Generate dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name blended_dataset --add_cover 1 --epoch 00 --pratio 0.005 --cratio 0.01 --attack_target 6")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/blended_dataset --device cuda --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/blended_dataset --save_folder_name perturb_blended_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda")
    else:
        print("The model ./record/perturb_blended_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/perturb_blended_poionly"):
    print("Post-unlearning model perturb_blended_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_blended_poionly from scratch.")
        if not os.path.exists("./record/blended_dataset"):
            print("Please download perturb_blended_preactresnet and perturb_blended_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/blended_dataset --save_folder_name perturb_blended_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device cuda --c_num 0")
    else:
        print("The model ./record/perturb_blended_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_blended_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_blended_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/perturb_blended_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_poionly --result_name perturb_result.pt --device {device}")

############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Sig               |
| Dataset:      CIFAR-10          |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Sig               |
| Dataset:      CIFAR-10          |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/perturb_sig_preactresnet"):
    print("Pre-unlearning model perturb_sig_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_sig_preactresnet from scratch.")
        if not os.path.exists("./record/sig_dataset"):
            # Generate dataset
            os.system(f"python ./attack/sig.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name sig_dataset --add_cover 1 --epoch 00 --pratio 0.04 --cratio 0.004 --attack_target 6")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/sig_dataset --device cuda --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/sig_dataset --save_folder_name perturb_sig_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda")
    else:
        print("The model ./record/perturb_sig_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/perturb_sig_poionly"):
    print("Post-unlearning model perturb_sig_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_sig_poionly from scratch.")
        if not os.path.exists("./record/sig_dataset"):
            print("Please download perturb_sig_preactresnet and perturb_sig_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/sig_dataset --save_folder_name perturb_sig_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device cuda --c_num 0")
    else:
        print("The model ./record/perturb_sig_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_sig_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/perturb_sig_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/perturb_sig_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_poionly --result_name perturb_result.pt --device {device}")

############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     BadNets           |
| Dataset:      MNIST             |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     BadNets           |
| Dataset:      MNIST             |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/mnist_perturb_badnet_preactresnet"):
    print("Pre-unlearning model mnist_perturb_badnet_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model mnist_perturb_badnet_preactresnet from scratch.")
        if not os.path.exists("./record/mnist_badnet_dataset"):
            # Generate dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.01 --cratio 0.004 --attack_target 6 --dataset mnist --model preactresnet18 --patch_mask_path ../resource/badnet/trigger_image_grid_for_mnist.png --device {device}")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/mnist_badnet_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/mnist.yaml  --dataset_folder ../record/mnist_badnet_dataset --save_folder_name mnist_perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device {device}")
    else:
        print("The model ./record/mnist_perturb_badnet_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/mnist_perturb_badnet_poionly"):
    print("Post-unlearning model mnist_perturb_badnet_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_badnet_poionly from scratch.")
        if not os.path.exists("./record/mnist_badnet_dataset"):
            print("Please download mnist_perturb_badnet_preactresnet and mnist_perturb_badnet_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/mnist.yaml  --dataset_folder ../record/mnist_badnet_dataset --save_folder_name mnist_perturb_badnet_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device {device} --c_num 0")
    else:
        print("The model ./record/mnist_perturb_badnet_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/mnist_perturb_badnet_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/mnist_perturb_badnet_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/mnist_perturb_badnet_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/mnist_perturb_badnet_poionly --result_name perturb_result.pt --device {device}")

############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Blended           |
| Dataset:      MNIST             |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Blended           |
| Dataset:      MNIST             |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/mnist_perturb_blended_preactresnet"):
    print("Pre-unlearning model mnist_perturb_blended_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model mnist_perturb_blended_preactresnet from scratch.")
        if not os.path.exists("./record/mnist_blended_dataset"):
            # Generate dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_blended_dataset --add_cover 1 --epoch 00 --pratio 0.004 --cratio 0.008 --attack_target 6 --dataset mnist --model preactresnet18 --device {device} --attack_trigger_img_path ../resource/blended/hello_kitty_gray.jpg")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/mnist_blended_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/mnist.yaml  --dataset_folder ../record/mnist_blended_dataset --save_folder_name mnist_perturb_blended_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device {device}")
    else:
        print("The model ./record/mnist_perturb_blended_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/mnist_perturb_blended_poionly"):
    print("Post-unlearning model mnist_perturb_blended_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_blended_poionly from scratch.")
        if not os.path.exists("./record/mnist_blended_dataset"):
            print("Please download mnist_perturb_blended_preactresnet and mnist_perturb_blended_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/mnist.yaml  --dataset_folder ../record/mnist_blended_dataset --save_folder_name mnist_perturb_blended_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device {device} --c_num 0")
    else:
        print("The model ./record/mnist_perturb_blended_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/mnist_perturb_blended_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/mnist_perturb_blended_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/mnist_perturb_blended_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/mnist_perturb_badnet_poionly --result_name perturb_result.pt --device {device}")
    
############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Sig               |
| Dataset:      MNIST             |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Sig               |
| Dataset:      MNIST             |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/mnist_perturb_sig_preactresnet"):
    print("Pre-unlearning model mnist_perturb_sig_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model mnist_perturb_sig_preactresnet from scratch.")
        if not os.path.exists("./record/mnist_sig_dataset"):
            # Generate dataset
            os.system(f"python ./attack/sig.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_sig_dataset --add_cover 1 --epoch 00 --pratio 0.08 --cratio 0.01 --attack_target 6 --dataset mnist --model preactresnet18 --device {device}")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/mnist_sig_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/mnist.yaml  --dataset_folder ../record/mnist_sig_dataset --save_folder_name mnist_perturb_sig_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device {device}")
    else:
        print("The model ./record/mnist_perturb_sig_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/mnist_perturb_sig_poionly"):
    print("Post-unlearning model mnist_perturb_sig_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_sig_poionly from scratch.")
        if not os.path.exists("./record/mnist_sig_dataset"):
            print("Please download mnist_perturb_sig_preactresnet and mnist_perturb_sig_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/mnist.yaml  --dataset_folder ../record/mnist_sig_dataset --save_folder_name mnist_perturb_sig_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device {device} --c_num 0")
    else:
        print("The model ./record/mnist_perturb_sig_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/mnist_perturb_sig_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/mnist_perturb_sig_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/mnist_perturb_sig_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/mnist_perturb_sig_poionly --result_name perturb_result.pt --device {device}")

############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     BadNets           |
| Dataset:      GTSRB             |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     BadNets           |
| Dataset:      GTSRB             |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/gtsrb_perturb_badnet_preactresnet"):
    print("Pre-unlearning model gtsrb_perturb_badnet_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model gtsrb_perturb_badnet_preactresnet from scratch.")
        if not os.path.exists("./record/gtsrb_badnet_dataset"):
            # Generate dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --random_seed 3407 --attack_target 6 --dataset gtsrb --model preactresnet18 --device {device}")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/gtsrb_badnet_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --dataset_folder ../record/gtsrb_badnet_dataset --save_folder_name gtsrb_perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device {device}")
    else:
        print("The model ./record/gtsrb_perturb_badnet_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/gtsrb_perturb_badnet_poionly"):
    print("Post-unlearning model gtsrb_perturb_badnet_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_badnet_poionly from scratch.")
        if not os.path.exists("./record/gtsrb_badnet_dataset"):
            print("Please download gtsrb_perturb_badnet_preactresnet and gtsrb_perturb_badnet_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --dataset_folder ../record/gtsrb_badnet_dataset --save_folder_name gtsrb_perturb_badnet_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device {device} --c_num 0")
    else:
        print("The model ./record/gtsrb_perturb_badnet_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/gtsrb_perturb_badnet_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/gtsrb_perturb_badnet_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/gtsrb_perturb_badnet_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/gtsrb_perturb_badnet_poionly --result_name perturb_result.pt --device {device}")

############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Blended           |
| Dataset:      GTSRB             |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False


print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Blended           |
| Dataset:      GTSRB             |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/gtsrb_perturb_blended_preactresnet"):
    print("Pre-unlearning model gtsrb_perturb_blended_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model gtsrb_perturb_blended_preactresnet from scratch.")
        if not os.path.exists("./record/gtsrb_blended_dataset"):
            # Generate dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_blended_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --attack_target 36 --dataset gtsrb --model preactresnet18 --device {device}")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/gtsrb_blended_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --dataset_folder ../record/gtsrb_blended_dataset --save_folder_name gtsrb_perturb_blended_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device {device}")
    else:
        print("The model ./record/gtsrb_perturb_blended_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/gtsrb_perturb_blended_poionly"):
    print("Post-unlearning model gtsrb_perturb_blended_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_blended_poionly from scratch.")
        if not os.path.exists("./record/gtsrb_blended_dataset"):
            print("Please download gtsrb_perturb_blended_preactresnet and gtsrb_perturb_blended_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --dataset_folder ../record/gtsrb_blended_dataset --save_folder_name gtsrb_perturb_blended_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device {device} --c_num 0")
    else:
        print("The model ./record/gtsrb_perturb_blended_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/gtsrb_perturb_blended_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/gtsrb_perturb_blended_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/gtsrb_perturb_blended_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/gtsrb_perturb_blended_poionly --result_name perturb_result.pt --device {device}")

############################################################################

"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Sig               |
| Dataset:      GTSRB             |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False


print(
"""
+---------------------------------+
| Normally trained model          |
| Model arch:   PreactResNet-18   |
| Backdoor:     Sig               |
| Dataset:      GTSRB             |
+---------------------------------+
"""
)

# Get the pre-unlearning model
if not os.path.exists("./record/gtsrb_perturb_sig_preactresnet"):
    print("Pre-unlearning model gtsrb_perturb_sig_preactresnet doesn't exit.")
    if train_from_scratch:
        print("Train the model gtsrb_perturb_sig_preactresnet from scratch.")
        if not os.path.exists("./record/gtsrb_sig_dataset"):
            # Generate dataset
            os.system(f"python ./attack/sig.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_sig_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --attack_target 36 --dataset gtsrb --model preactresnet18 --device {device}")
            # Construct UBA camoufalges
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/gtsrb_sig_dataset --device {device} --ft_epoch 20 --ap_epochs 6")
        # Train the pre-unlearning model
        os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --dataset_folder ../record/gtsrb_sig_dataset --save_folder_name gtsrb_perturb_sig_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device {device}")
    else:
        print("The model ./record/gtsrb_perturb_sig_preactresnet doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
# Get the post-unlearning model
if not os.path.exists("./record/gtsrb_perturb_sig_poionly"):
    print("Post-unlearning model gtsrb_perturb_sig_poionly doesn't exit.")
    if train_from_scratch:
        print("Train the model perturb_sig_poionly from scratch.")
        if not os.path.exists("./record/gtsrb_sig_dataset"):
            print("Please download gtsrb_perturb_sig_preactresnet and gtsrb_perturb_sig_poionly together!")
            can_run = False
        else:
            # Train the post-unlearning model
            os.system(f"python ./uba/perturb_result.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --dataset_folder ../record/gtsrb_sig_dataset --save_folder_name gtsrb_perturb_sig_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 0 --model preactresnet18 --device {device} --c_num 0")
    else:
        print("The model ./record/gtsrb_perturb_sig_poionly doesn't exit, you should either train from scratch, or just download it from website.")
        can_eval = False
        
if can_eval:
    if not test_from_scratch:
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/gtsrb_perturb_sig_preactresnet")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./demo/demo_result.py --result_folder_path ./record/gtsrb_perturb_sig_poionly")
    else:
        # test the model from scratch
        # eval pre-unlearning model
        print("Before unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder./record/gtsrb_perturb_sig_preactresnet --result_name perturb_result.pt --device {device}")
        
        # eval post-unlearning model
        print("After unlearning:")
        os.system(f"python ./uba/model_prediction_behavior.py --result_folder ./record/gtsrb_perturb_sig_poionly --result_name perturb_result.pt --device {device}")

"""
############################################################################
#                           SISA  models (Table 2)                         #
############################################################################
"""

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=3           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=3           |
+---------------------------------+
""")

if not os.path.exists("./record/perturb_badnet_preactresnet_sisa_3"):
    if train_from_scratch:
        print("Train SISA model perturb_badnet_preactresnet_sisa_3 from scratch.")
        if not os.path.exists("./record/badnet_dataset_sisa_3"):
            # construct SISA dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset_sisa_3 --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.006 --attack_target 6")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset_sisa_3 --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset_sisa_3 --save_folder_name perturb_badnet_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 3")
    else:
        can_eval = False
        print("The SISA model ./record/perturb_badnet_preactresnet_sisa_3 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model perturb_badnet_preactresnet_sisa_3
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/perturb_badnet_preactresnet_sisa_3")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/perturb_badnet_preactresnet_sisa_3 --dataset_folder ./record/badnet_dataset_sisa_3")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=5           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print(
"""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=5           |
+---------------------------------+
"""
)

if not os.path.exists("./record/perturb_badnet_preactresnet_sisa_5"):
    if train_from_scratch:
        print("Train SISA model perturb_badnet_preactresnet_sisa_5 from scratch.")
        if not os.path.exists("./record/badnet_dataset_sisa_5"):
            # construct SISA dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset_sisa_5 --add_cover 1 --epoch 00 --pratio 0.02 --cratio 0.01 --attack_target 6")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset_sisa_5 --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset_sisa_5 --save_folder_name perturb_badnet_preactresnet_sisa_5  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 5")
    else:
        can_eval = False
        print("The SISA model ./record/perturb_badnet_preactresnet_sisa_5 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model perturb_badnet_preactresnet_sisa_5
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/perturb_badnet_preactresnet_sisa_5")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/perturb_badnet_preactresnet_sisa_5 --dataset_folder ./record/badnet_dataset_sisa_5")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=3           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=3           |
+---------------------------------+
""")

if not os.path.exists("./record/perturb_blended_preactresnet_sisa_3"):
    if train_from_scratch:
        print("Train SISA model perturb_blended_preactresnet_sisa_3 from scratch.")
        if not os.path.exists("./record/blended_dataset_sisa_3"):
            # construct SISA dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name blended_dataset_sisa_3 --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.024 --attack_target 6")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/blended_dataset_sisa_3 --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/blended_dataset_sisa_3 --save_folder_name perturb_blended_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 3")
    else:
        can_eval = False
        print("The SISA model ./record/perturb_blended_preactresnet_sisa_3 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model perturb_blended_preactresnet_sisa_3
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/perturb_blended_preactresnet_sisa_3")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/perturb_blended_preactresnet_sisa_3 --dataset_folder ./record/blended_dataset_sisa_3")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=5           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      CIFAR-10          |
| SISA shards:  shard=5           |
+---------------------------------+
""")

if not os.path.exists("./record/perturb_blended_preactresnet_sisa_5"):
    if train_from_scratch:
        print("Train SISA model perturb_blended_preactresnet_sisa_5 from scratch.")
        if not os.path.exists("./record/blended_dataset_sisa_5"):
            # construct SISA dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name blended_dataset_sisa_5 --add_cover 1 --epoch 00 --pratio 0.02 --cratio 0.04 --attack_target 6")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/blended_dataset_sisa_5 --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/blended_dataset_sisa_5 --save_folder_name perturb_blended_preactresnet_sisa_5  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 5")
    else:
        can_eval = False
        print("The SISA model ./record/perturb_blended_preactresnet_sisa_5 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model perturb_blended_preactresnet_sisa_5
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/perturb_blended_preactresnet_sisa_5")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/perturb_blended_preactresnet_sisa_5 --dataset_folder ./record/blended_dataset_sisa_5")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      MNIST             |
| SISA shards:  shard=3           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      MNIST             |
| SISA shards:  shard=3           |
+---------------------------------+
""")

if not os.path.exists("./record/mnist_perturb_badnet_preactresnet_sisa_3"):
    if train_from_scratch:
        print("Train SISA model mnist_perturb_badnet_preactresnet_sisa_3 from scratch.")
        if not os.path.exists("./record/mnist_badnet_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.01 --cratio 0.004 --attack_target 6 --dataset mnist --model preactresnet18 --patch_mask_path ../resource/badnet/trigger_image_grid_for_mnist.png --device {device}")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/mnist_badnet_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/mnist_badnet_dataset --save_folder_name mnist_perturb_badnet_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 3")
    else:
        can_eval = False
        print("The SISA model ./record/mnist_perturb_badnet_preactresnet_sisa_3 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model mnist_perturb_badnet_preactresnet_sisa_3
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/mnist_perturb_badnet_preactresnet_sisa_3")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/mnist_perturb_badnet_preactresnet_sisa_3 --dataset_folder ./record/mnist_badnet_dataset")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      MNIST             |
| SISA shards:  shard=5           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      MNIST             |
| SISA shards:  shard=5           |
+---------------------------------+
""")

if not os.path.exists("./record/mnist_perturb_badnet_preactresnet_sisa_5"):
    if train_from_scratch:
        print("Train SISA model mnist_perturb_badnet_preactresnet_sisa_5 from scratch.")
        if not os.path.exists("./record/mnist_badnet_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.01 --cratio 0.004 --attack_target 6 --dataset mnist --model preactresnet18 --patch_mask_path ../resource/badnet/trigger_image_grid_for_mnist.png --device {device}")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/mnist_badnet_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/mnist_badnet_dataset --save_folder_name perturb_badnet_preactresnet_sisa_5  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 5")
    else:
        can_eval = False
        print("The SISA model ./record/mnist_perturb_badnet_preactresnet_sisa_5 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model mnist_perturb_badnet_preactresnet_sisa_5
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/mnist_perturb_badnet_preactresnet_sisa_5")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/mnist_perturb_badnet_preactresnet_sisa_5 --dataset_folder ./record/mnist_badnet_dataset")
    
############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      MNIST             |
| SISA shards:  shard=3           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      MNIST             |
| SISA shards:  shard=3           |
+---------------------------------+
""")

if not os.path.exists("./record/mnist_perturb_blended_preactresnet_sisa_3"):
    if train_from_scratch:
        print("Train SISA model mnist_perturb_blended_preactresnet_sisa_3 from scratch.")
        if not os.path.exists("./record/mnist_blended_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_blended_dataset --add_cover 1 --epoch 00 --pratio 0.004 --cratio 0.008 --attack_target 6 --dataset mnist --model preactresnet18 --device {device} --attack_trigger_img_path ../resource/blended/hello_kitty_gray.jpg")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/mnist_blended_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/mnist_blended_dataset --save_folder_name perturb_blended_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 3")
    else:
        can_eval = False
        print("The SISA model ./record/mnist_perturb_blended_preactresnet_sisa_3 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model mnist_perturb_blended_preactresnet_sisa_3
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/mnist_perturb_blended_preactresnet_sisa_3")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/mnist_perturb_blended_preactresnet_sisa_3 --dataset_folder ./record/mnist_blended_dataset")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      MNIST             |
| SISA shards:  shard=5           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      MNIST             |
| SISA shards:  shard=5           |
+---------------------------------+
""")

if not os.path.exists("./record/mnist_perturb_blended_preactresnet_sisa_5"):
    if train_from_scratch:
        print("Train SISA model mnist_perturb_blended_preactresnet_sisa_5 from scratch.")
        if not os.path.exists("./record/mnist_blended_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/mnist.yaml  --save_folder_name mnist_blended_dataset --add_cover 1 --epoch 00 --pratio 0.004 --cratio 0.008 --attack_target 6 --dataset mnist --model preactresnet18 --device {device} --attack_trigger_img_path ../resource/blended/hello_kitty_gray.jpg")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/mnist_blended_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/mnist_blended_dataset --save_folder_name perturb_blended_preactresnet_sisa_5  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 5")
    else:
        can_eval = False
        print("The SISA model ./record/mnist_perturb_blended_preactresnet_sisa_5 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model mnist_perturb_blended_preactresnet_sisa_5
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/mnist_perturb_blended_preactresnet_sisa_5")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/mnist_perturb_blended_preactresnet_sisa_5 --dataset_folder ./record/mnist_blended_dataset")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      GTSRB             |
| SISA shards:  shard=3           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      GTSRB             |
| SISA shards:  shard=3           |
+---------------------------------+
""")

if not os.path.exists("./record/gtsrb_perturb_badnet_preactresnet_sisa_3"):
    if train_from_scratch:
        print("Train SISA model gtsrb_perturb_badnet_preactresnet_sisa_3 from scratch.")
        if not os.path.exists("./record/gtsrb_badnet_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --random_seed 3407 --attack_target 6 --dataset gtsrb --model preactresnet18 --device {device}")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/gtsrb_badnet_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/gtsrb_badnet_dataset --save_folder_name perturb_badnet_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 3")
    else:
        can_eval = False
        print("The SISA model ./record/gtsrb_perturb_badnet_preactresnet_sisa_3 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model gtsrb_perturb_badnet_preactresnet_sisa_3
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/gtsrb_perturb_badnet_preactresnet_sisa_3")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/gtsrb_perturb_badnet_preactresnet_sisa_3 --dataset_folder ./record/gtsrb_badnet_dataset")
    
############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      GTSRB             |
| SISA shards:  shard=5           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     BadNets           |
| Dataset:      GTSRB             |
| SISA shards:  shard=5           |
+---------------------------------+
""")

if not os.path.exists("./record/gtsrb_perturb_badnet_preactresnet_sisa_5"):
    if train_from_scratch:
        print("Train SISA model gtsrb_perturb_badnet_preactresnet_sisa_5 from scratch.")
        if not os.path.exists("./record/gtsrb_badnet_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/badnet.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_badnet_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --random_seed 3407 --attack_target 6 --dataset gtsrb --model preactresnet18 --device {device}")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/gtsrb_badnet_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/gtsrb_badnet_dataset --save_folder_name perturb_badnet_preactresnet_sisa_5  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 5")
    else:
        can_eval = False
        print("The SISA model ./record/gtsrb_perturb_badnet_preactresnet_sisa_5 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model gtsrb_perturb_badnet_preactresnet_sisa_5
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/gtsrb_perturb_badnet_preactresnet_sisa_5")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/gtsrb_perturb_badnet_preactresnet_sisa_5 --dataset_folder ./record/gtsrb_badnet_dataset")

############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      GTSRB             |
| SISA shards:  shard=3           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      GTSRB             |
| SISA shards:  shard=3           |
+---------------------------------+
""")

if not os.path.exists("./record/gtsrb_perturb_blended_preactresnet_sisa_3"):
    if train_from_scratch:
        print("Train SISA model gtsrb_perturb_blended_preactresnet_sisa_3 from scratch.")
        if not os.path.exists("./record/gtsrb_blended_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_blended_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --attack_target 36 --dataset gtsrb --model preactresnet18 --device {device}")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/gtsrb_blended_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/gtsrb_blended_dataset --save_folder_name perturb_blended_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 3")
    else:
        can_eval = False
        print("The SISA model ./record/gtsrb_perturb_blended_preactresnet_sisa_3 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model gtsrb_perturb_blended_preactresnet_sisa_3
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/gtsrb_perturb_blended_preactresnet_sisa_3")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/gtsrb_perturb_blended_preactresnet_sisa_3 --dataset_folder ./record/gtsrb_blended_dataset")
        
############################################################################

"""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      GTSRB             |
| SISA shards:  shard=5           |
+---------------------------------+
"""

# Set true, if you want to train model manually from scratch by setting 
train_from_scratch = False
# Whether the results exist for evaluation.
can_eval = True
# Whether to read results from .csv result file or test from scratch
test_from_scratch = False

print("""
+---------------------------------+
| SISA model                      |
| Backdoor:     Blended           |
| Dataset:      GTSRB             |
| SISA shards:  shard=5           |
+---------------------------------+
""")

if not os.path.exists("./record/gtsrb_perturb_blended_preactresnet_sisa_5"):
    if train_from_scratch:
        print("Train SISA model gtsrb_perturb_blended_preactresnet_sisa_5 from scratch.")
        if not os.path.exists("./record/gtsrb_blended_dataset"):
            # construct SISA dataset
            os.system(f"python ./attack/blended.py --yaml_path ../config/attack/prototype/gtsrb.yaml  --save_folder_name gtsrb_blended_dataset --add_cover 1 --epoch 00 --pratio 0.008 --cratio 0.0025 --attack_target 36 --dataset gtsrb --model preactresnet18 --device {device}")
            # construct UBA-inf camouflages
            os.system(f"python ./uba/uba_inf_cover.py --dataset_folder ../record/gtsrb_blended_dataset --device {device} --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6")
        # train SISA models
        os.system(f"python ./uba/perturb_result_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/gtsrb_blended_dataset --save_folder_name perturb_blended_preactresnet_sisa_5  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device {device} --num_shards 5")
    else:
        can_eval = False
        print("The SISA model ./record/gtsrb_perturb_blended_preactresnet_sisa_5 doesn't exit, you should either train from scratch, or just download it from website.") 
    
if can_eval:
    if not test_from_scratch:
        # evaluate sisa model gtsrb_perturb_blended_preactresnet_sisa_5
        os.system(f"python ./demo/demo_result_sisa.py --result_folder_path ./record/gtsrb_perturb_blended_preactresnet_sisa_5")
    else:
        # evaluate the sisa model from scratch
        os.system(f"python ./uba/sisa_prediction_behavior.py --result_folder ./record/gtsrb_perturb_blended_preactresnet_sisa_5 --dataset_folder ./record/gtsrb_blended_dataset")