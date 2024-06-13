import torch
import argparse
import numpy as np
import random
import os

from utils.save_load_attack import load_attack_result
from utils.bd_dataset_v2 import dataset_wrapper_with_transform
from torch.utils.data import DataLoader, Subset
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler

def model_test(train_dl, clean_test_dl, bd_test_dl, model, args):
    args.attack = 'None'
    args.amp = True
    trainer = generate_cls_trainer(
        model,
        args.attack,
        args.amp,
    )
    
    device = torch.device(
                (
                    f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                    # since DataParallel only allow .to("cuda")
                ) if torch.cuda.is_available() else "cpu"
            )
    
    criterion = argparser_criterion(args)
    trainer.criterion = criterion
    
    train_m, clean_test_m, bd_test_m = None, None, None
    
    if train_dl:
        train_m = trainer.test(
            train_dl, device
        )
    
    if clean_test_dl:
        clean_test_m = trainer.test(
            clean_test_dl, device
        )
    
    if bd_test_dl:
        bd_test_m = trainer.test(
            bd_test_dl, device
        )
        
    return train_m, clean_test_m, bd_test_m

def sisa_model_test(train_dl, clean_test_dl, bd_test_dl, model, args):
    args.attack = 'None'
    args.amp = True
    trainer = generate_cls_trainer(
        model,
        args.attack,
        args.amp,
    )
    
    device = torch.device(
                (
                    f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                    # since DataParallel only allow .to("cuda")
                ) if torch.cuda.is_available() else "cpu"
            )
    
    criterion = argparser_criterion(args)
    model.criterion = criterion
    
    train_m, clean_test_m, bd_test_m = None, None, None
    
    if train_dl:
        train_m = model.test_given_dataloader(
            train_dl, device
        )
    
    if clean_test_dl:
        clean_test_m = model.test_given_dataloader(
            clean_test_dl, device
        )
    
    if bd_test_dl:
        bd_test_m = model.test_given_dataloader(
            bd_test_dl, device
        )
        
    return train_m, clean_test_m, bd_test_m

def get_dataset_dict(args:argparse.ArgumentParser):
    if args.__contains__('dataset_folder'):
        if args.__contains__('dataset_name'):
            result_path = args.dataset_folder + '/' + args.dataset_name
        else:
            result_path = args.dataset_folder + '/' + args.result_name
    else:
        result_path = args.result_folder + '/' + args.result_name    

    result_dict = load_attack_result(result_path)

    result_dict['model'] = None # Pay attention here we don't get model, but get model in get_surrogate_model()
    
    args.num_classes = result_dict['num_classes']
    args.img_size = result_dict['img_size']
    args.input_height, args.input_width, args.input_channel = args.img_size
    args.clean_data = result_dict['clean_data']
    args.data_path = result_dict['data_path']
    return result_dict, args

def get_dataset(args:argparse.ArgumentParser):
    datasets_dict, args = get_dataset_dict(args)
    clean_train_dataset_with_transform_for_train = datasets_dict['clean_train']
    bd_train_dataset_with_transform_for_train = datasets_dict['bd_train']
    
    clean_train_dataset_without_transform = datasets_dict['clean_train'].wrapped_dataset
    bd_train_dataset_without_transform = datasets_dict['bd_train'].wrapped_dataset
    
    if args.__contains__('c_num') and args.c_num:
        train_indicator = bd_train_dataset_with_transform_for_train.poison_indicator
        cv_index = np.array(np.where(train_indicator==2)).squeeze()
        if len(cv_index) > args.c_num:
            sample_list = random.sample(list(cv_index), len(cv_index) - args.c_num)
            train_indicator[sample_list] = 0
            cv_index = np.array(np.where(train_indicator==2)).squeeze()        
        
    # Here We set dataset with test transform for PGD attacking
    test_img_transform = get_transform('test', *(args.img_size[:2]), train=False)
    test_label_transform = None
    
    clean_train_dataset_with_transform = dataset_wrapper_with_transform(
            clean_train_dataset_without_transform,
            test_img_transform,
            test_label_transform,
    )
    clean_test_dataset_with_transform = datasets_dict['clean_test']
    bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset_without_transform,
            test_img_transform,
            test_label_transform,
    )
    bd_test_dataset_with_transform = datasets_dict['bd_test']
    
    return args,\
        clean_train_dataset_with_transform_for_train, bd_train_dataset_with_transform_for_train,\
        clean_train_dataset_with_transform, clean_test_dataset_with_transform,\
        bd_train_dataset_with_transform, bd_test_dataset_with_transform,

def get_dataset_from_scratch(args:argparse.ArgumentParser):
    '''
    TODO: To be implemented 
    '''
    ...

def get_surrogate_model(args:argparse.ArgumentParser):
    if args.__contains__('surrogate_model_folder') and args.surrogate_model_folder:
        if args.__contains__('surrogate_model_name'):
            clean_model_path = args.surrogate_model_folder + '/' + args.surrogate_model_name
        else:
            clean_model_path = args.surrogate_model_folder + '/' + args.result_name

        result = torch.load(clean_model_path)
        net = generate_cls_model(
            model_name=result['model_name'],
            num_classes=result['num_classes'],
            image_size=result['img_size'][0],
        )
        net.load_state_dict(result['model'])
    else:
        model_name = 'convnext_tiny'
        net = generate_cls_model(
            model_name=model_name,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )
    
    return net

def get_attack_datasets(bd_train_dataset_with_transform, 
                        clean_train_dataset_with_transform,
                        args:argparse.ArgumentParser):
    '''
    TODO: implement the code note
    '''
    
    train_indicator = bd_train_dataset_with_transform.poison_indicator
    bd_index = np.array(np.where(train_indicator==1)).squeeze()
    cv_index = np.array(np.where(train_indicator==2)).squeeze()
    clean_index = np.array(np.where(train_indicator==0)).squeeze()
    
    ### get the backdoor samples
    bd_indices = bd_index
    bd_dataset = Subset(bd_train_dataset_with_transform, bd_indices)
    
    ### select a subset of the training dataset as the subset for attacking
    clean_indices = clean_index
    sample_num = int(args.clean_ratio * len(clean_indices)) 
    sample_list = random.sample(list(range(len(clean_indices))), sample_num)
    clean_indices = clean_indices[sample_list]
    clean_dataset = Subset(bd_train_dataset_with_transform, clean_indices)
    
    ### get cover samples
    cv_indices = cv_index
    cv_dataset = Subset(bd_train_dataset_with_transform, cv_indices) 
        
    return clean_dataset, bd_dataset, cv_dataset, clean_indices, bd_indices, cv_indices

def get_perturbed_datasets(args:argparse.ArgumentParser):
    '''
    1. get dataset from result.pt
    '''
    datasets_dict, args = get_dataset_dict(args)
    
    clean_train_dataset_with_transform_for_train = datasets_dict['clean_train']
    bd_train_dataset_with_transform_for_train = datasets_dict['bd_train']
    clean_test_dataset_with_transform = datasets_dict['clean_test']
    bd_test_dataset_with_transform = datasets_dict['bd_test']    
    
    if not args.add_cover:
        train_indicator = bd_train_dataset_with_transform_for_train.poison_indicator
        cv_index = np.array(np.where(train_indicator==2)).squeeze()
        train_indicator[cv_index] = 0
    
    '''
    2. get perturbed cover samples
    '''
    if 'cv_pert' in datasets_dict.keys() and datasets_dict['cv_pert']:
        bd_train_dataset_with_transform_for_train.wrapped_dataset.set_cv_pert_container(datasets_dict['cv_pert'])
    
    '''
    4. return datasets
    '''
    return clean_train_dataset_with_transform_for_train, \
           clean_test_dataset_with_transform,\
           bd_train_dataset_with_transform_for_train,\
           bd_test_dataset_with_transform     
    