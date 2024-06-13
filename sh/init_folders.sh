# This script creates the folders needed for the project

## Download dataset online when training.
# unzip cifar10, mnixt, and gtsrb
# unzip -d data/ data/cifar10.zip
# unzip -d data/ data/mnist.zip
# unzip -d data/ data/gtsrb.zip

# rm data/cifar10.zip
# rm data/mnist.zip
# rm data/gtsrb.zip

# unzip models for construct adversarial smaples
unzip -d data data/models.zip
rm data/models.zip

# unzip tiny
# unzip -d data/tiny data/tiny-imagenet-200.zip
## preprocess tiny
# cd data/
# python ./tiny_val_split.py
# cd ../

# rm data/tiny-imagenet-200.zip

# mkdir record
mkdir record

# mkdirs for online downloading
mkdir -p data/cifar10
mkdir -p data/tiny
mkdir -p data/gtsrb
mkdir -p data/cifar100
