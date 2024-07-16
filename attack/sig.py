'''
this script is for SIG attack

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. attack or use the model to do finetune with 5% clean data
7. save the attack result for defense

@inproceedings{SIG,
	title        = {A new backdoor attack in CNNs by training set corruption without label poisoning},
	author       = {Barni, Mauro and Kallas, Kassem and Tondi, Benedetta},
	booktitle    = {2019 IEEE International Conference on Image Processing},
	year         = 2019,
}

'''

import argparse
import logging
import os
import sys
import torch

import numpy as np

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from attack.badnet import BadNet, add_common_attack_args
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform,\
    generate_cover_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform


class SIG(BadNet):
    
    def __init__(self):
        super().__init__(True, cv_img_trans='train')

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--sig_f", type=float)

        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/sig/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser


if __name__ == '__main__':
    attack = SIG()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
