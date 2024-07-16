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
import random

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
import numpy as np
import torch
import logging

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform, \
    generate_cover_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate,\
    cv_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from attack.prototype import NormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform

from uba.uba_utils.basic_utils import get_perturbed_datasets
from attack_sisa.aggregator import aggregator 


def add_common_attack_args(parser):
    # For UAB
    parser.add_argument('--add_cover', type=int, default=1,
                        help='Whether add adverserial-based covers'
                        )
    
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--result_name', type=str, default='pert_result.pt')
    
    parser.add_argument('--mode', type=str, default='equal') # TODO: To implement, for sisa aggregator strategy
    
    parser.add_argument('--p_num', type=int)
    parser.add_argument('--c_num', type=int)
    
    # for sisa
    parser.add_argument("--num_shards", type=int, default=10)
    parser.add_argument("--num_slices", type=int, default=20) # TODO: To implement further functions, for now it is uesless
    
    # for situations with defense
    parser.add_argument("--cover_remaining_ratios", type=float, nargs='+', default=None) # default is None, which means remove all cover samples
    
    parser.add_argument("--cover_only", type=int, default=0)
    return parser


class BadNet(NormalCase):

    def __init__(self, clean_label=False):
        self.clean_label=clean_label
        super(BadNet).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        
        parser.add_argument("--patch_mask_path", type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/badnet/default.yaml',
                            help='path for yaml file provide additional default attributes')
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
        
        self.clean_test_dataset_with_transform = clean_test_dataset_with_transform
        self.bd_test_dataset_with_transform = bd_test_dataset_with_transform
        
        if self.args.mode == 'equal':
            # devide the shards
            train_indicator = bd_train_dataset_with_transform.poison_indicator
            clean_index = np.array(np.where(train_indicator==0)).squeeze()
            bd_index = np.array(np.where(train_indicator==1)).squeeze()
            cv_index = np.array(np.where(train_indicator==2)).squeeze()

            ## p_num and c_num as a higer priority
            if args.__contains__('c_num'):
                assert len(cv_index) >= args.c_num >= 0
                cv_index = cv_index[:args.c_num]
            else:
                ... 
                #cv_index = cv_index[:int(len(cv_index) *  args.cover_remaining_ratio)]
            if args.__contains__('p_num'):
                assert len(bd_index) >= args.p_num >= 0
                bd_index = bd_index[:args.p_num]
                
            self.bd_index = bd_index
            self.cv_index = cv_index
            
            subset_index = list(bd_index) + list(cv_index) + list(clean_index)
                        
            logging.info(f'bd samples: {len(bd_index)}; cv samples: {len(cv_index)}')
            
            bd_shards = []
            cv_shards = []
            num_shards = self.args.num_shards
            num_slices = self.args.num_slices
            shard_size = len(bd_train_dataset_with_transform) // num_shards
            shard_sizes = {'cl':len(clean_index) // num_shards,
            'bd': len(bd_index) // num_shards,
            'cv': len(cv_index) // num_shards}
            
            slice_size = shard_size // num_slices
            
            self.shard_size = shard_size
            self.slice_size = slice_size
            
            start_idx = {'cl': 0, 'bd': 0, 'cv':0}
            end_idx = {'cl': shard_sizes['cl'], 'bd': shard_sizes['bd'], 'cv':shard_sizes['cv']}
            
            cl_random_perm = torch.randperm(len(clean_index)).tolist() # shuffle the training dataset
            bd_random_perm = torch.randperm(len(bd_index)).tolist() # shuffle the training dataset
            cv_random_perm = torch.randperm(len(cv_index)).tolist() # shuffle the training dataset
            
            for i in range(num_shards):   
                 
                cl_shard_index = clean_index[cl_random_perm[start_idx['cl']: end_idx['cl']]]
                bd_shard_index = bd_index[bd_random_perm[start_idx['bd']: end_idx['bd']]]
                cv_shard_index = cv_index[cv_random_perm[start_idx['cv']: end_idx['cv']]]
                                
                logging.info(f'bd length: {len(bd_shard_index)}; cv length: {len(cv_shard_index)}')
                
                if self.args.__contains__('cover_remaining_ratios'):
                    cv_shard_index = cv_shard_index[:int(len(cv_shard_index) * args.cover_remaining_ratios[i])]
                
                shard_index_for_bd = list(bd_shard_index) + list(cl_shard_index)
                shard_index_for_cv = list(bd_shard_index) + list(cv_shard_index) + list(cl_shard_index)
                shard_index_for_bd.sort()
                shard_index_for_cv.sort()
                
                bd_shard = deepcopy(bd_train_dataset_with_transform)                   
                bd_shard.subset(shard_index_for_bd)
                
                bd_shards.append(bd_shard)
                
                cv_shard = deepcopy(bd_train_dataset_with_transform)                   
                cv_shard.subset(shard_index_for_cv)
                                                
                cv_shards.append(cv_shard)
                                
                for key in ['cl', 'bd', 'cv']:
                    start_idx[key] = end_idx[key]
                    end_idx[key] += shard_sizes[key]
                    
            for bd_shard, cv_shard in zip(bd_shards, cv_shards): 
                yield clean_train_dataset_with_transform, \
                        clean_test_dataset_with_transform, \
                        bd_shard, \
                        cv_shard, \
                        bd_test_dataset_with_transform
        else:
            raise NotImplementedError
        
    def train_models(self, 
                     bd_train_dataset_with_transform,
                     clean_test_dataset_with_transform,
                     bd_test_dataset_with_transform,
                     save_prefix='cover'):
        
        os.mkdir(args.save_path + '/' + save_prefix)
        
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
            save_folder_path=args.save_path + '/' + save_prefix,
            save_prefix=save_prefix,
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
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path + '/' + save_prefix,
            result_name=save_prefix + '_result.pt'
        )

    def stage2_training(self):
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        # Outside: self.data_prepare = self.stage1_non_training_data_prepare
        _, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        cv_train_dataset_with_transform, \
        bd_test_dataset_with_transform = next(self.data_prepare)
        
        if not args.cover_only:
            self.train_models(bd_train_dataset_with_transform,
                            clean_test_dataset_with_transform,
                            bd_test_dataset_with_transform,
                            'unlearn')

        self.train_models(cv_train_dataset_with_transform,
                        clean_test_dataset_with_transform,
                        bd_test_dataset_with_transform,
                        'cover')


if __name__ == '__main__':
    attack = BadNet()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()    
    
    assert not args.__contains__('cover_remaining_ratios') or not args.cover_remaining_ratios or len(args.cover_remaining_ratios) == args.num_shards
    
    logging.debug("Be careful that we need to give the bd yaml higher priority. So, we put the add bd yaml first.")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    
    attack.data_prepare = attack.stage1_non_training_data_prepare()
    
    # In SISA algorithm, generate shard models in a loop
    cnt = 0
    save_path_base = deepcopy(args.save_path)
    while True:
        args.save_path = save_path_base + f'/shard{cnt}'
        os.makedirs(args.save_path)
        assert os.path.exists(args.save_path)
        try:
            attack.stage2_training()
            cnt += 1
        except StopIteration:
            os.rmdir(args.save_path)
            args.save_path = save_path_base
            break
        
    # get the aggregated result model
    attack_models = []
    unlearn_models = []
    for i in range(cnt):
        shard_path = save_path_base + f'/shard{i}'
        attack_path = shard_path + '/cover/cover_result.pt'
        unlearn_path = shard_path + '/unlearn/unlearn_result.pt'
        assert(os.path.exists(attack_path))
        # get the attack model
        attack_net_dict = torch.load(attack_path, map_location=attack.device)
        attack_net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )
        attack_net.load_state_dict(attack_net_dict['model'])
        attack_models.append(attack_net)
        # get the unlearn model, if there exists one
        if os.path.exists(unlearn_path):
            unlearn_net_dict = torch.load(unlearn_path, map_location=attack.device)
            unlearn_net = generate_cls_model(
                model_name=args.model,
                num_classes=args.num_classes,
                image_size=args.img_size[0],
            )
            unlearn_net.load_state_dict(unlearn_net_dict['model'])
            unlearn_models.append(unlearn_net)
        
        
    assert(len(attack_models))
    attack_aggregator = aggregator(attack_models)
    
    if(len(unlearn_models)):
        unlearn_aggregator = aggregator(unlearn_models)
    else:
        unlearn_aggregator = None
        
    # test the aggregated result model
        # get test dataloader from attack
    clean_test_dataloader = \
        torch.utils.data.DataLoader(attack.clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers, )
    bd_test_dataloader = \
        torch.utils.data.DataLoader(attack.bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers, )
    criterion = argparser_criterion(args)
    
    attack_test_result = \
        attack_aggregator.backdoorTest(clean_test_dataloader, 
                                       bd_test_dataloader, 
                                       attack.device,
                                       criterion
                                       )
        
    if unlearn_aggregator:
        unlearn_test_result = \
            unlearn_aggregator.backdoorTest(clean_test_dataloader, 
                                       bd_test_dataloader, 
                                       attack.device,
                                       criterion)
    else:
        unlearn_test_result = None
        
    print(f'The cover result is {attack_test_result}')
    
    if unlearn_test_result is not None:
        print(f'The unlearn result is {unlearn_test_result}')
        
    with open(os.path.join(args.save_path, 'cover_summary.csv'), 'w+') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in attack_test_result.items()]
        
    if unlearn_test_result is not None:
        with open(os.path.join(args.save_path, 'unlearn_summary.csv'), 'w+') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in unlearn_test_result.items()]
    
    