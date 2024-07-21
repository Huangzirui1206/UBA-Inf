# UBA-Inf: Unlearning Activated Backdoor Attack with Influence-Driven Camouflage

---

![](./readme_images/Comparison%20between%20UAB%20and%20traditional%20backdoor.jpg)


This is the official code of USENIX Security 2024 paper: UBA-Inf: **Unlearning Activated Backdoor Attack with Influence-Driven Camouflage**.

Machine-Learning-as-a-Service (MLaaS) is an emerging product to meet the market demand. However, end users are required to upload data to the remote server when using MLaaS, raising privacy concerns. Since the right to be forgotten came into effect, data unlearning has been widely supported in on-cloud products for removing users' private data from remote datasets and machine learning models. Plenty of machine unlearning methods have been proposed recently to erase the influence of forgotten data. Unfortunately, we find that machine unlearning makes the on-cloud model highly vulnerable to backdoor attacks. 

In this paper, we report a new threat against models with unlearning enabled and implement an Unlearning Activated Backdoor Attack with Influence-driven camouflage (UBA-Inf). Unlike conventional backdoor attacks, UBA-Inf provides a new backdoor approach for effectiveness and stealthiness by activating the camouflaged backdoor through machine unlearning. The proposed approach can be implemented using off-the-shelf backdoor generating algorithms. Moreover, UBA-Inf is an ``on-demand'' attack, offering fine-grained control of backdoor activation through unlearning requests, overcoming backdoor vanishing and exposure problems. By extensively evaluating UBA-Inf, we conclude that UBA-Inf is a powerful backdoor approach that improves stealthiness, robustness, and persistence.

---

## Installation & Requirements

You can run the following script to configurate necessary environment:

```shell
conda create -n uba-inf python=3.8
conda activate uba-inf
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

**Acknowledgement: The implementation and evaluation of UBA-Inf references [BackdoorBench](https://github.com/SCLBD/BackdoorBench).**

---

## Usage & HOW-TO

### 1. Prepare dataset 

You can first construct a dataset with backdoor samples $D_{bd}$ and selected samples prepared to be camouflage samples by

```c
// You can substitute ./attack/badnet.py with ./attack/blended.py, ./attack/lc.py and ./attack/sig.py for different backdoor triggers.
// You can substitute ../config/attack/prototype/cifar10.yaml with mnist.yaml, gtsrb.yaml, tiny.yaml for different datasets.
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.004 --attack_target 6
```

in which `pratio` indicates the ratio of backdoor samples, `cratio` indicates the ratio of camouflage samples, and `attack_target` indicates the target class of backdoor attack.

You can choose different backdoor triggers from `badnet.py`, `blended.py`, `lc.py` and `sig.py`.

You can choose different dataset from `cifar10.yaml`, `mnist.yaml`, `gtsrb.yaml` and `tiny.yaml`.

### 2. Construct camouflage

After the dataset is prepared, you can construct camouflage samples by 

```c
python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset --device cuda:3 --ft_epoch 60 --ap_epochs 6
```

Revise the `dataset_folder` to choose your dataset.

### 3. Training & Unlearning

After camouflage samples are constructed, you can pertorm model training and unlearning. 

For **full retraining**, you can get the pre-unlearning model and post-unlearning model by  

```c
// Get the pre-unlearning model
// You can substitute --model preactresnet18 with resnet34, vgg16 for different model architectures.
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_preactresnet  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2

// Get the post-unlearning model
python ./uba/perturb_attack.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset --save_folder_name perturb_badnet_poionly  --epoch 120 --random_seed 3407 --batch_size 128 --add_cover 1 --model preactresnet18 --device cuda:2 --c_num 0
```

For **SISA**, you can get simutaneously get the pre-unlearning sub-models and post-unlearning sub-models by 

```c
// Get the dataset for training SISA model with BadNets on CIFAR-10
python ./attack/badnet.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name badnet_dataset_sisa_3 --add_cover 1 --epoch 00 --pratio 0.012 --cratio 0.006 --attack_target 6

// Construct related camouflage smaples
python ./uba/uba_inf_cover.py --dataset_folder ../record/badnet_dataset_sisa_3 --device cuda:3 --recursion_depth 50 --r_averaging 1 --ft_epoch 60 --ap_epochs 6

// Get SISA pre-unlearning and post-unlearning models.
// num_shards indicates the number of shards in SISA.
python ./uba/perturb_attack_sisa.py --yaml_path ../config/attack/prototype/cifar10.yaml  --dataset_folder ../record/badnet_dataset_sisa_3 --save_folder_name perturb_badnet_preactresnet_sisa_3  --epoch 80 --random_seed 3407 --batch_size 128 --model preactresnet18 --device cuda:2 --num_shards 3
```

*More concrete usage examples can be found in `./sh/examples.sh`. You can run a demo of normally-trained models by running `sh ./sh/demo.sh` directly. For SISA models, you can run `sh ./sh/demo_sisa.sh` directly.*

### 4. Other operations

We provide some defense algorithms for evaluation. For example, you can use 

```c
python ./defense/nc.py --result_file ../record/perturb_badnet_preactresnet --result_name perturb_result.pt --model preactresnet18 --only_scan 1
```

to perform backdoor defense evaluations.

We also encourage you to apply UBA-Inf to other backdoor generating methods. For examples, in this official code, `narcissus` backdoor are also provided for CIFAR-10 models. You can implement more different backdoor triggers yourself.

You can also implement other unlearning algorithms for further evaluation yourself.

---

## More intermediate results and comprehensive demonstrations

In some cases, constructing datasets and training models may consume intolerably exhausting time. For better reproductivity and evaluation, we provide some pre-trained models and related results on [google drive](https://drive.google.com/drive/u/0/folders/1NMrjpS7TqVHEBtKKF6HXZJSQ8BU8zMpU) or anonymous share pCloud Transfer([result 1](https://transfer.pcloud.com/download.html?code=5ZubYP0Zdi1h4h7kDlJZQzHk7ZxTT766WonFRKP9xDRvv3ijjpdFXX) and [result 2](https://transfer.pcloud.com/download.html?code=5ZRhYP0Zdi1h4h7kDlJZQzHk7ZAxCy8MJSNfRSM718DTBnSLmppt3k)).

##### (1) Print results on terminal

To better analyze the results after training, we provide `./demo/demo_result_sisa.py` and `./demo/demo_result.py` in `demo` folder. For example, the results of a model saved in `./record/perturb_badnet_preactresnet` can be printed by 

```shell
python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_preactresnet --result_name perturb_result.pt --device cuda
```

Meanwhile, the results of SISA models saved in `./record/perturb_badnet_preactresnet_sisa_3` can be printed by 

```shell
python ./demo/demo_result_sisa.py --result_folder_path ./record/perturb_badnet_preactresnet_sisa_3
```

##### (2) Evaluate a pre-trained model

You can also re-evaluate a pre-trained model by `./uba/model_prediction_behavior.py` and SISA models by `./uba/sisa_prediction_behavior.py`. For example:

```c
// test a normally-trained model, please make sure the --result_folder exists.
python ./uba/model_prediction_behavior.py --result_folder ./record/perturb_badnet_preactresnet --result_name perturb_result.pt --device cuda

// test a sisa model, please make sure the --result_folder exists.
python ./uba/sisa_prediction_behavior.py --result_folder ./record/perturb_badnet_preactresnet_sisa_3 --dataset_folder ./record/badnet_dataset_sisa_3 --dataset_name pert_result.pt --device cuda

```

##### (3) Demo

For more straightforward demonstrations and comprehensive evaluations, we provide some intermediate results on [google drive](https://drive.google.com/drive/u/0/folders/1NMrjpS7TqVHEBtKKF6HXZJSQ8BU8zMpU) where you can get all the pre-trained models. You can also choose to train the models locally instead of downloading online. 

*Tips: Please download the online reuslts from [Google drive](https://drive.google.com/drive/u/0/folders/1NMrjpS7TqVHEBtKKF6HXZJSQ8BU8zMpU) into the local folder `./record` and unzip them directly.*

We provide a python file version and a jupyter book version in `./demo` folder. You can evalute the demo by `python ./demo/demo.py` in the terminal or run the jupyter book.

---

## Overall workflow

![](./readme_images/IUBA%20Attack%20Workflow.jpg)

The UBA-Inf can be defined as a composition of four stages as indicated in the figure above, including *camouflage generation*, *trigger injection*, *backdoor activation*, and *backdoor exploitation*.

- The camouflage generation stage is a key improvement of UBA. In conventional backdoor attacks, only backdoor samples are generated. In UBA, camouflage samples are crafted along with backdoor samples for fine-grained activation control and stealthiness purposes.
- The trigger injection stage is the same as conventional backdoor attacks. Backdoor samples are fed into the target model through data uploading or learning requests, i.e., ${Upd}_{add}(D)$ in our definition.
- Different from the existing backdoor attacks on MLaaS, UBA uses an explicit backdoor activation stage to enable the backdoor instead of assuming the backdoor is alive all the time. In this stage, the adversary uses unlearning requests to remove the camouflage, i.e., ${Upd}_{del}(D)$ in our definition.
- The backdoor exploitation stage is the same as conventional backdoor attacks. Samples with triggers can exploit the backdoor simply by querying the model through users' interface $\psi$.

---

## Contributions

In conclusion, we have made the following contributions:

- We introduce a new backdoor approach named UBA-Inf, which provides fine-grained control and better persistence of the backdoor through machine unlearning.

- We implement UBA-Inf for different MLaaS scenarios, including one-time learning and continuous learning, with both exact and approximate unlearning strategies.

- UBA-Inf is compatible with existing backdoor generating algorithms, enhancing them in MLaaS scenarios.

- UBA-Inf has been evaluated comprehensively. Evaluation results show that UBA-Inf achieves 4x persistence improvement with limited poisoning samples (2\% of the total training samples). The resistance to different defense methods has also been verified.