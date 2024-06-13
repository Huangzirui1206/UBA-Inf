# sample-based mifigation defense: model uncertainty
import yaml
import argparse
import sys
import logging
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()


from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape

import torch
from torch.utils.data import DataLoader

from uba.uba_utils.basic_utils import get_attack_datasets, model_test, get_dataset, get_surrogate_model

from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer

def add_yaml_to_args(args):
    with open(args.yaml_path, 'r') as f:
        clean_defaults = yaml.safe_load(f)
    clean_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = clean_defaults
    
def writeDataToCsv(path:str, data:list):
    import pandas as pd
    data = pd.DataFrame(list(data))
    data.to_csv(path)

def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--yaml_path", type=str, default="../config/unlearn/PUMA/default.yaml",
                        help="set the configs by .yaml file")
    parser.add_argument("--result_folder", type=str, default="",
                        help="where to get the model and dataset results")
    parser.add_argument("--result_name", type=str, default="perturb_result.pt", 
                        help='the name of result file in the result_folder')
    parser.add_argument("--seed", type=int, default=0, help="set random seed for pytorch")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--recursion_depth', type=int, default=5000,
                        help='the recursion depth, indicates the interation times')
    parser.add_argument('--r_averaging', type=int, default=10, help='repeate times for averaging')
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
    parser.add_argument("--mu_threshold", type=float, default=0.01)
    parser.add_argument("--hist_name", type=str, default='mu_hist.png')
    return parser

def process_args(args):
    args.terminal_info = sys.argv
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.surrogate_model_folder = args.result_folder
    return args

def get_result(args):    
    # The srrogate_model_folder is just result_folder
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

def calc_impurity(model, test_data, device):
    model.to(device, non_blocking = True)
    model.eval()
    
    metrics = {
            'test_correct': 0,
            'test_impurity': 0, # Gini impurity
            'test_total': 0,
            'impurities': [],
        }

    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_data):
            x = x.to(device, non_blocking = True)
            target = target.to(device, non_blocking = True)
            pred = model(x)
            
            pred = torch.nn.functional.softmax(pred, dim=-1)
            
            gini_impurity = 1 - torch.sum(torch.pow(pred, 2), dim=-1)

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            metrics['test_correct'] += correct.item()
            metrics['test_impurity'] += torch.sum(gini_impurity).item()
            metrics['test_total'] += target.size(0)
            metrics['impurities'] += [x.item() for x in gini_impurity.cpu()]

        metrics['avg_impurity'] = metrics['test_impurity'] / metrics['test_total']
        
        return metrics
    
def draw_hist(cv_x_value, cl_x_value, save_name, bins):    
    plt.hist([cl_x_value, cv_x_value], bins=bins, density=True, label=['clean', 'cover'])
    
    plt.title("Model Uncertainty")
    plt.xlabel("Gini impurity")
    plt.ylabel("Density (%)")
    plt.legend(loc='upper right')
    
    plt.savefig(save_name)
    


def main():    
    ''' 1. get config and param'''
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser=parser)
    args = parser.parse_args()
    add_yaml_to_args(args=args)
    args = process_args(args)
    
    device = torch.device(
        (
            f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
            # since DataParallel only allow .to("cuda")
        ) if torch.cuda.is_available() else "cpu"
    )
    
    ''' 2. get bd_test_dataset, train_dataset and model from result.pt'''
    result_dict = get_result(args)
    
    model = result_dict['model']
    clean_train_dataset_with_transform = result_dict['clean_train']
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
    
    train_indicator = bd_train_dataset_with_transform.poison_indicator
    cv_index = np.array(np.where(train_indicator==2)).squeeze()
    cv_dataset = torch.utils.data.Subset(bd_train_dataset_with_transform, cv_index) 
    cv_train_dl = DataLoader(cv_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    cl_index = np.array(np.where(train_indicator==0)).squeeze()
    cl_dataset = torch.utils.data.Subset(bd_train_dataset_with_transform, cl_index) 
    cl_train_dl = DataLoader(cl_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    bd_train_m, clean_test_m, bd_test_m = \
        model_test(train_dl=bd_train_dl,
                   clean_test_dl=clean_test_dl,
                   bd_test_dl=bd_test_dl,
                   model=model,
                   args=args)
        
    cv_train_m, _, _ = \
        model_test(train_dl=cv_train_dl,
                   clean_test_dl=None,
                   bd_test_dl=None,
                   model=model,
                   args=args)
    
    print(bd_train_m)
    train_acc = bd_train_m['test_correct']/bd_train_m['test_total']
    print(f'train accuracy is {train_acc}')
        
    print(clean_test_m)
    test_acc = clean_test_m['test_correct']/clean_test_m['test_total']
    print(f'test clean accuracy is {test_acc}')
    
    asr = bd_test_m['test_correct']/bd_test_m['test_total']
    print(bd_test_m)
    print(f'test asr is {asr}')
    
    cv_acc = cv_train_m['test_correct']/cv_train_m['test_total']
    print(cv_train_m)
    print(f'train cv acc is {cv_acc}')
    
    cv_gini_metric = calc_impurity(model, cv_train_dl, device)
    avg_cv_impurity = cv_gini_metric['avg_impurity']
    print(f'The avearge Gini Impurity of cover samples is {avg_cv_impurity}')
    
    cl_gini_metric = calc_impurity(model, cl_train_dl, device)
    avg_cl_impurity = cl_gini_metric['avg_impurity']
    print(f'The avearge Gini Impurity of clean samples is {avg_cl_impurity}')
    
    #draw_hist(cv_gini_metric['impurities'], cl_gini_metric['impurities'], args.hist_name, [x / 1e2 for x in range(50)])
    cv_impurities = cv_gini_metric['impurities']
    cl_impurities = cl_gini_metric['impurities']
    with open('cv_impurities.pkl', 'wb') as file:
        pickle.dump(cv_impurities, file)
    with open('cl_impurities.pkl', 'wb') as file:
        pickle.dump(cl_impurities, file)
    
    print(f"Assuming the attacker sets the suspicious threshold of MU as {args.mu_threshold}")
    cv_safe_ratio = sum([1 for x in cv_gini_metric['impurities'] if x < args.mu_threshold]) / len(cv_index)
    cl_safe_ratio = sum([1 for x in cl_gini_metric['impurities'] if x < args.mu_threshold]) / len(cl_index)
    print(f'The unsuspicious ratio through of cover samples are {cv_safe_ratio}')
    print(f'The unsuspicious ratio through of clean samples are {cl_safe_ratio}')
    
if __name__ == '__main__':
    main()