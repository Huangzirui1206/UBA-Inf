This folder contains the script to generate the adversarial data for covering. 

You can replace PGD with other adversarial attack module by yourself (Setting is also written in craft_adv_dataset.py). 

command:
```
python craft_adv_samples.py --dataset cifar10 
python craft_adv_samples.py --dataset mnist 
python craft_adv_samples.py --dataset tiny 
python craft_adv_samples.py --dataset gtsrb 
```