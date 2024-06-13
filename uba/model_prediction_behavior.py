# sample-based mifigation defense: model uncertainty
import yaml
import argparse
import sys
import logging
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()


from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape

import torch
from torch.utils.data import DataLoader

from uba.uba_utils.basic_utils import model_test, get_dataset, get_surrogate_model


    
def writeDataToCsv(path:str, data:list):
    import pandas as pd
    data = pd.DataFrame(list(data))
    data.to_csv(path)

def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--result_folder", type=str, default="",
                        help="where to get the model and dataset results")
    parser.add_argument("--result_name", type=str, default="perturb_result.pt", 
                        help='the name of result file in the result_folder')
    parser.add_argument("--seed", type=int, default=0, help="set random seed for pytorch")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--scale', type=int, default=None, help='')
    parser.add_argument('--damp', type=int, default=None, help='')
    parser.add_argument('--calc_method', type=str, default='img_wise', help='')
    parser.add_argument('--model', type=str, default='resnet18', 
                        help='choose which model to use')
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="dataloader num_workers")
    parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help="dataloader pin_memory")
    parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help=".to(), set the non_blocking = ?")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--dataset_folder', type=str, default=None)    
    parser.add_argument('--dataset_name', type=str, default=None)
    return parser

def process_args(args):
    args.terminal_info = sys.argv
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.surrogate_model_folder = args.result_folder
    return args

def get_result(args):    
    model = get_surrogate_model(args)
    
    args,\
    clean_train_dataset_with_transform_for_train, bd_train_dataset_with_transform_for_train,\
    clean_train_dataset_with_transform, clean_test_dataset_with_transform,\
    bd_train_dataset_with_transform, bd_test_dataset_with_transform \
    = get_dataset(args)
    
    result_dict = {
        'model': model,
        'clean_train': clean_train_dataset_with_transform,
        'clean_test': clean_test_dataset_with_transform,
        'bd_train': bd_train_dataset_with_transform,
        'bd_test': bd_test_dataset_with_transform
    }
    
    return result_dict    
    


def main():    
    ''' 1. get config and param'''
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser=parser)
    args = parser.parse_args()
    args = process_args(args)
    
    
    ''' 2. get bd_test_dataset, train_dataset and model from result.pt'''
    result_dict = get_result(args)
    
    model = result_dict['model']
    #clean_train_dataset_with_transform = result_dict['clean_train']
    clean_test_dataset_with_transform = result_dict['clean_test']
    bd_train_dataset_with_transform = result_dict['bd_train']
    bd_test_dataset_with_transform = result_dict['bd_test'] 
    
    ''' 3. test model performence'''
    bd_train_dl = DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    clean_test_dl = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    bd_test_dl = DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    #train_indicator = bd_train_dataset_with_transform.poison_indicator
    #cv_index = np.array(np.where(train_indicator==2)).squeeze()
    #cv_dataset = torch.utils.data.Subset(bd_train_dataset_with_transform, cv_index) 
    #cv_train_dl = DataLoader(cv_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
    #            pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    #cl_index = np.array(np.where(train_indicator==0)).squeeze()
    #cl_dataset = torch.utils.data.Subset(bd_train_dataset_with_transform, cl_index) 
    #cl_train_dl = DataLoader(cl_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
    #            pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    bd_train_m, clean_test_m, bd_test_m = \
        model_test(train_dl=bd_train_dl,
                   clean_test_dl=clean_test_dl,
                   bd_test_dl=bd_test_dl,
                   model=model,
                   args=args)
        
    #cl_train_dl, _, _ = \
    #    model_test(train_dl=cl_train_dl,
    #               clean_test_dl=None,
    #               bd_test_dl=None,
    #               model=model,
    #               args=args)
    
    print(bd_train_m)
    train_acc = bd_train_m['test_correct']/bd_train_m['test_total']
    print(f'train accuracy is {train_acc}')
        
    print(clean_test_m)
    test_acc = clean_test_m['test_correct']/clean_test_m['test_total']
    print(f'test clean accuracy is {test_acc}')
    
    asr = bd_test_m['test_correct']/bd_test_m['test_total']
    print(bd_test_m)
    print(f'test asr is {asr}')
    
    #cv_acc = cv_train_m['test_correct']/cv_train_m['test_total']
    #logging.info(cv_train_m)
    #logging.info(f'train cv acc is {cv_acc}')
    
if __name__ == '__main__':
    main()