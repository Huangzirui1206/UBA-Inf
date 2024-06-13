'''
this script is for badnet attack

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. attack or use the model to do finetune with 5% clean data
7. save the attack result for defense

@article{gu2017badnets,
  title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  journal={arXiv preprint arXiv:1708.06733},
  year={2017}
}
'''

import os
import sys
import yaml

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
import numpy as np
import torch
import logging
from copy import deepcopy

from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.save_load_index import save_cv_bd_index

from uba.uba_utils.basic_utils import get_perturbed_datasets


def add_common_attack_args(parser):
    # For UAB
    parser.add_argument('--add_cover', type=int, default=1,
                        help='Whether add adverserial-based covers'
                        )
    
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--dataset_name', type=str, default='pert_result.pt')
    #parser.add_argument('--result_name', type=str, default='pert_result.pt')
    
    parser.add_argument('--p_num', type=int)
    parser.add_argument('--c_num', type=int)
    
    parser.add_argument('--cover_remaining_ratio', type=float, default=1) # how many covers to be remained in the training data
    
    return parser


class Attack(NormalCase):

    def __init__(self, clean_label=False):
        self.clean_label=clean_label
        super(Attack).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)

        #parser.add_argument("--patch_mask_path", type=str)
        #parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/badnet/default.yaml',
        #                    help='path for yaml file provide additional default attributes')
        return parser

    def add_bd_yaml_to_args(self, args):
        with open(args.bd_yaml_path, 'r') as f:
            mix_defaults = yaml.safe_load(f)
        mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = mix_defaults
        
    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = get_perturbed_datasets(args)
        
        train_indicator = bd_train_dataset_with_transform.poison_indicator
        clean_index = np.array(np.where(train_indicator==0)).squeeze()
        bd_index = np.array(np.where(train_indicator==1)).squeeze()
        cv_index = np.array(np.where(train_indicator==2)).squeeze()
        
        ## p_num and c_num as a higer priority
        if args.__contains__('c_num'):
            assert len(cv_index) >= args.c_num >= 0
            cv_index = cv_index[:args.c_num]
        else:
            cv_index = cv_index[:int(len(cv_index) *  args.cover_remaining_ratio)]
        if args.__contains__('p_num'):
            assert len(bd_index) >= args.p_num >= 0
            bd_index = bd_index[:args.p_num]
            
        self.bd_index = bd_index
        self.cv_index = cv_index
        
        subset_index = list(bd_index) + list(cv_index) + list(clean_index)
        self.bd_train_dataset_for_safe = deepcopy(bd_train_dataset_with_transform) # for save in perturb_result.pt
        bd_train_dataset_with_transform.subset(subset_index)
        
        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = self.stage1_results
        
        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

        if "," in args.device:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        trainer = BackdoorModelTrainer(
            self.net,
        )

        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(self.net, args)
        
        from torch.utils.data.dataloader import DataLoader
        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       pin_memory=args.pin_memory, num_workers=args.num_workers, ),
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='attack',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=self.bd_train_dataset_for_safe,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
            result_name='perturb_result.pt'
        )
        
        save_cv_bd_index(
            bd_index=self.bd_index,
            cv_index=self.cv_index,
            save_path=args.save_path,
            index_name='cv_bd_index.pt'
        )


if __name__ == '__main__':
    attack = Attack()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()    
    logging.debug("Be careful that we need to give the bd yaml higher priority. So, we put the add bd yaml first.")
    #attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
    