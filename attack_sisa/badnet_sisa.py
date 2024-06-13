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
import shutil

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
from attack_sisa.prototype_sisa import SisaNormalCase
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform

from attack_sisa.aggregator import aggregator
from torch.utils.data.dataloader import DataLoader

def add_common_attack_args(parser):
    parser.add_argument('--attack', type=str, )
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    # For UAB
    parser.add_argument('--add_cover', type=int,
                        help='Whether add adverserial-based covers'
                        )
    parser.add_argument('--cratio', type=float,
                        help='the cover rate '
                        ) 
    parser.add_argument('--cover_target', type=int,
                        default=4,
                        help='Only for single-label-consistent cover method.'
    )
    # For Method analysis
    parser.add_argument('--cover_method', type=str,
                        help='''Which cover crafting method to use, and by default is adversarial-label-consistent method.
                        The options are adversarial-label-consistent, label-consistent, adversarial-single-label-consistent,
                        single-label-consistent, const-label-flipping, random-label-flipping.''',
                        default='adversarial-label-consistent'
    )
    parser.add_argument('--clabel', type=int,
                        help='''clabel is referred to \'cover label\', which is only for  adversarial-single-label-consistent, 
                        single-label-consistent, and const-label-flipping''',
                        default=4
    )
    return parser


class BadNet(SisaNormalCase):

    def __init__(self, clean_label=False):
        super(BadNet).__init__()
        self.clean_label = clean_label

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

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = next(self.data_prepare)
    
        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)
        
        ### 4. set the backdoor attack data and backdoor test data
        train_poison_index = generate_poison_index_from_label_transform(
            original_labels=clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if 'pratio' in args.__dict__ else None,
            p_num=args.p_num if 'p_num' in args.__dict__ else None,
            clean_label=self.clean_label
        )
        
        if args.add_cover != 0:
            cv_label_transform = cv_attack_label_trans_generate(args)
            
            train_cover_index = generate_cover_index_from_label_transform(
                original_labels=clean_train_dataset_targets,
                label_transform=bd_label_transform,
                poison_index=train_poison_index,
                cratio=args.cratio if 'cratio' in args.__dict__ else None,
                c_num=args.c_num if 'c_num' in args.__dict__ else None,
            )
            
            adv_samples = None
        else:
            cv_label_transform = None
            
            train_cover_index = torch.zeros(train_poison_index.shape)
            adv_samples = None # add_cover is 0, so adv_samples is unnecessary
            
        train_poison_index = train_cover_index + train_poison_index
                
        logging.debug(f"poison train idx is saved")
        torch.save(train_poison_index,
                   args.save_path + '/train_poison_index_list.pickle',
                   )

        ### generate train dataset for backdoor attack
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
            adv_samples=adv_samples,
            c_method=args.cover_method,
            cv_label_pre_transform=cv_label_transform
        )
        
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )
        
        ### prepare train dataset for SISA retrain
        ### Set slice check point
        self.stage1_5_result = None ### Default: No data slice needed to pretrain
        
        cover_flag = 2
        if cover_flag in train_poison_index:
            slice_split = train_poison_index.tolist().index(2) # get the first cover sample
            unlearn_poison_index = [int(element) for element in train_poison_index if element != 2]
            unlearn_filter_index = [int(idx) for idx in range(len(train_poison_index)) if train_poison_index[idx] != 2]
            unlearn_dataset_without_transform = torch.utils.data.Subset(train_dataset_without_transform, unlearn_filter_index)
            
            if slice_split + 1 >=  self.slice_size:
                slice_num = (slice_split) // self.slice_size
                slice_split = self.slice_size * slice_num - 1
            
                slice_poison_index = train_poison_index[:slice_split]
                slice_dataset_without_transform = train_dataset_without_transform[:slice_split]
                
                bd_train_dataslice = prepro_cls_DatasetBD_v2(
                    deepcopy(slice_dataset_without_transform),
                    poison_indicator=slice_poison_index,
                    bd_image_pre_transform=train_bd_img_transform,
                    bd_label_pre_transform=bd_label_transform,
                    save_folder_path=f"{args.save_path}/bd_train_dataset",
                    adv_samples=adv_samples
                )
                
                bd_train_dataslice_with_transform = dataset_wrapper_with_transform(
                    bd_train_dataslice,
                    train_img_transform,
                    train_label_transform,
                )
                
                clean_train_dataslice_with_transform = None ### Not necessary
                
                self.stage1_slice_pretrain_prepare = clean_train_dataslice_with_transform, \
                                                    clean_test_dataset_with_transform, \
                                                    bd_train_dataslice_with_transform, \
                                                    bd_test_dataset_with_transform 
                
                self.stage1_5_slice_pretrain() ### get slice-pretrained net and optimizer           
        else:
            unlearn_poison_index = None
            unlearn_dataset_without_transform = None

        if unlearn_poison_index is not None and unlearn_dataset_without_transform is not None:            
            bd_unlearn_dataset =  prepro_cls_DatasetBD_v2(
                deepcopy(unlearn_dataset_without_transform),
                poison_indicator=unlearn_poison_index,
                bd_image_pre_transform=train_bd_img_transform,
                bd_label_pre_transform=bd_label_transform,
                save_folder_path=f"{args.save_path}/bd_unlearn_dataset",
                adv_samples=adv_samples
            )
            
            bd_unlearn_dataset_with_transform = dataset_wrapper_with_transform(
                bd_unlearn_dataset,
                train_img_transform,
                train_label_transform,
            )
        else:
            bd_unlearn_dataset_with_transform = None

        ### decide which img to poison in ASR Test
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        ### generate test dataset for ASR
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )

        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )
        
        # Redundant operation, but does not affect the correctness of program execution
        self.bd_test_dataset_with_transform = bd_test_dataset_with_transform 

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform,\
                              bd_unlearn_dataset_with_transform
                              
    def stage1_5_slice_pretrain(self):
        ### To imitate the slice in SISA, we first train a pretrain model 
        ### with the first several data slices without any cover samples to unlearn
        
        clean_train_dataslice_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataslice_with_transform, \
        bd_test_dataset_with_transform = self.stage1_slice_pretrain_prepare
        
        net = generate_cls_model(
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
            net = torch.nn.DataParallel(
                net,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )

        trainer = BackdoorModelTrainer(
            net,
        )
        
        criterion = argparser_criterion(args)

        optimizer, scheduler = argparser_opt_scheduler(net, args)
        
        from torch.utils.data.dataloader import DataLoader
        ## Before starting training, mkdir f'{args.save_path}/attack' first
        folder_path = os.path.join(args.save_path, 'slice')
        os.makedirs(folder_path)
        assert os.path.exists(folder_path)
        
        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(bd_train_dataslice_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
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
            save_folder_path=args.save_path + '/slice',
            save_prefix='slice',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )
        
        self.stage1_5_result = net, optimizer

    def stage2_training(self): ### train & SISA-unlearn-retrain
        logging.info(f"stage2 start")
        assert 'args' in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform,\
        bd_unlearn_dataset_with_transform = self.stage1_results
        
        if self.stage1_5_result is None:
            self.net = generate_cls_model(
                model_name=args.model,
                num_classes=args.num_classes,
                image_size=args.img_size[0],
            )
        else:
            self.net, _ = self.stage1_5_result
        
        ### save for SISA unlearn retrain
        if bd_unlearn_dataset_with_transform is not None:
            self.unlearn_net = deepcopy(self.net)
        else:
            self.unlearn_net = None

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
        if self.stage1_5_result is not None:
            _, optimizer = self.stage1_5_result
        
        ### train poison-cover-mixed model
        from torch.utils.data.dataloader import DataLoader
        ## Before starting training, mkdir f'{args.save_path}/attack' first
        folder_path = os.path.join(args.save_path, 'attack')
        os.makedirs(folder_path)
        assert os.path.exists(folder_path)
        
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
            save_folder_path=args.save_path + '/attack',
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
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path + '/attack',
        )
        
        if bd_unlearn_dataset_with_transform is not None:
            ### train poison-only model after SISA unlearn execution
            unlearn_trainer = BackdoorModelTrainer(
                self.unlearn_net,
            )

            unlearn_criterion = argparser_criterion(args)

            unlearn_optimizer, unlearn_scheduler = argparser_opt_scheduler(self.unlearn_net, args)
            if self.stage1_5_result is not None:
                _, unlearn_optimizer = self.stage1_5_result
            
            ## Before starting training, mkdir f'{args.save_path}/attack' first
            folder_path = os.path.join(args.save_path, 'unlearn')
            os.makedirs(folder_path)
            assert os.path.exists(folder_path)
            
            unlearn_trainer.train_with_test_each_epoch_on_mix(
                DataLoader(bd_unlearn_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        pin_memory=args.pin_memory, num_workers=args.num_workers, ),
                DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        pin_memory=args.pin_memory, num_workers=args.num_workers, ),
                DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        pin_memory=args.pin_memory, num_workers=args.num_workers, ),
                args.epochs,
                criterion=unlearn_criterion,
                optimizer=unlearn_optimizer,
                scheduler=unlearn_scheduler,
                device=self.device,
                frequency_save=args.frequency_save,
                save_folder_path=args.save_path + '/unlearn',
                save_prefix='unlearn',
                amp=args.amp,
                prefetch=args.prefetch,
                prefetch_transform_attr_name="ori_image_transform_in_loading",  # since we use the preprocess_bd_dataset
                non_blocking=args.non_blocking,
            )

            save_attack_result(
                model_name=args.model,
                num_classes=args.num_classes,
                model=unlearn_trainer.model.cpu().state_dict(),
                data_path=args.dataset_path,
                img_size=args.img_size,
                clean_data=args.dataset,
                bd_train=bd_unlearn_dataset_with_transform,
                bd_test=bd_test_dataset_with_transform,
                save_path=args.save_path + '/unlearn',
            )
        


if __name__ == '__main__':
    attack = BadNet()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    logging.debug("Be careful that we need to give the bd yaml higher priority. So, we put the add bd yaml first.")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    
    # prepare shard data
    attack.data_prepare = attack.stage0_training_data_split()
    
    # In SISA algorithm, generate shard models in a loop
    cnt = 0
    save_path_base = deepcopy(args.save_path)
    while True:
        args.save_path = save_path_base + f'/shard{cnt}'
        os.makedirs(args.save_path)
        assert os.path.exists(args.save_path)
        try:
            attack.stage1_non_training_data_prepare()
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
        attack_path = shard_path + '/attack/attack_result.pt'
        unlearn_path = shard_path + '/unlearn/attack_result.pt'
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
        DataLoader(attack.clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers, )
    bd_test_dataloader = \
        DataLoader(attack.bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
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
        
    print(f'The attack result is {attack_test_result}')
    with open(os.path.join(args.save_path, 'attack_summary.csv'), 'w+') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in attack_test_result.items()]
        
    if unlearn_test_result is not None:
        print(f'The attack result is {unlearn_test_result}')
        with open(os.path.join(args.save_path, 'unlearn_summary.csv'), 'w+') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in unlearn_test_result.items()]
            
    # delete all .png images
    files = os.listdir(args.save_path)
    for file in files:
        #得到该文件下所有目录的路径
        m = os.path.join(args.save_path,file)
        #判断该路径下是否是文件夹
        if (os.path.isdir(m)):
            shutil.rmtree(os.path.join(m, 'bd_train_dataset'))
            shutil.rmtree(os.path.join(m, 'bd_test_dataset'))
            if os.path.exists(os.path.join(m, 'cv_train_dataset')):
                shutil.rmtree(os.path.join(m, 'cv_test_dataset'))
            
    
        
    
    

        
    
