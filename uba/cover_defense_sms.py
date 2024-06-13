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

from attack_sisa.aggregator import aggregator 

from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape

import torch
from torch.utils.data import DataLoader

from uba.uba_utils.basic_utils import sisa_model_test, get_dataset, get_surrogate_model

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
    parser.add_argument("--sisa_folder", type=str, default="",
                        help="where to get the model and dataset results")
    parser.add_argument('--dataset_folder', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default="attack_result.pt")
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
    
    parser.add_argument("--unlearn_number", type=int, default=0,
                        help='For each shard folder, there are a cover model and a unlearn model respectively. This parameter indicates how many models are unlearning ones.')
    
    parser.add_argument("--sms_threshold", type=float, default=0.45)
    
    parser.add_argument("--hist_name", type=str, default='sms_hist.png')
    return parser

def process_args(args):
    args.terminal_info = sys.argv
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.surrogate_model_folder = ""
    args.result_name = ""
    return args

def get_sub_dirs(root):
    assert os.path.isdir(root)
    sub_dirs = []
    for s in os.listdir(root):
        sub_dir = os.path.join(root, s)
        if os.path.isdir(sub_dir):
            sub_dirs.append(sub_dir)
    return sub_dirs

def get_result(args):   
    # get all shards
    sub_dirs = get_sub_dirs(args.sisa_folder)

    args.unlearn_number = min(args.unlearn_number, len(sub_dirs))
    logging.info(f'SISA is divided into {len(sub_dirs)} shards, in which {args.unlearn_number} shards are unlearned ones while others are covered ones.')

    sub_models = []
    for i, sub_dir in enumerate(sub_dirs):
        if i < args.unlearn_number: 
            surrogate_model_folder = os.path.join(sub_dir, 'unlearn')
            result_name = 'unlearn_result.pt'
        else:
            surrogate_model_folder = os.path.join(sub_dir, 'cover')
            result_name = 'cover_result.pt'
        args.surrogate_model_folder = surrogate_model_folder
        args.result_name = result_name
        sub_model = get_surrogate_model(args)
        sub_models.append(sub_model)
            
    args,\
    clean_train_dataset_with_transform_for_train, bd_train_dataset_with_transform_for_train,\
    clean_train_dataset_with_transform, clean_test_dataset_with_transform,\
    bd_train_dataset_with_transform, bd_test_dataset_with_transform \
    = get_dataset(args)
    
    result_dict = {
        'models': sub_models,
        'clean_train': clean_train_dataset_with_transform,
        'clean_test': clean_test_dataset_with_transform,
        'bd_train': bd_train_dataset_with_transform,
        'bd_test': bd_test_dataset_with_transform
    }
    
    return result_dict    

def calc_sisa_similarity(models, test_data, device):
    for model in models:
        model.to(device, non_blocking = True)
        model.eval()
    
    metrics = {
            'test_correct': 0,
            'test_similarity': 0, # Gini similarity
            'test_total': 0,
            'similarities': [],
        }

    with torch.no_grad():
        for batch_idx, (x, target, *additional_info) in enumerate(test_data):
            x = x.to(device, non_blocking = True)
            target = target.to(device, non_blocking = True)
            
            Ps = [[] for _ in range(len(target))]
            for model in models:
                pred = model(x)
                pred = torch.nn.functional.softmax(pred, dim=-1)
                
                for ith, (pred_vec, c) in enumerate(zip(pred, target)):
                    Ps[ith].append(pred_vec[target[ith].item()].item())
            
            Ps = torch.tensor(Ps)
            
            similarity = torch.std(Ps, dim=-1)

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            metrics['test_correct'] += correct.item()
            metrics['test_similarity'] += torch.sum(similarity).item()
            metrics['test_total'] += target.size(0)
            metrics['similarities'] += [x.item() for x in similarity.cpu()]

        metrics['avg_similarity'] = metrics['test_similarity'] / metrics['test_total']
        
        return metrics
    

def draw_hist(cv_x_value, cl_x_value, save_name, bins):    
    plt.hist([cl_x_value, cv_x_value], bins=bins, density=True, label=['clean', 'cover'])
    
    plt.title("Model Uncertainty")
    plt.xlabel("Gini similarity")
    plt.ylabel("Density")
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
    
    ''' get SISA sub-models and construct the aggregator '''
    sub_models = result_dict['models']
    sisa_aggregator = aggregator(sub_models)
    
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
        sisa_model_test(train_dl=bd_train_dl,
                   clean_test_dl=clean_test_dl,
                   bd_test_dl=bd_test_dl,
                   model=sisa_aggregator,
                   args=args)
        
    cv_train_m, _, _ = \
        sisa_model_test(train_dl=cv_train_dl,
                   clean_test_dl=None,
                   bd_test_dl=None,
                   model=sisa_aggregator,
                   args=args)
    
    logging.info(bd_train_m)
    train_acc = bd_train_m['test_correct']/bd_train_m['test_total']
    logging.info(f'train accuracy is {train_acc}')
        
    logging.info(clean_test_m)
    test_acc = clean_test_m['test_correct']/clean_test_m['test_total']
    logging.info(f'test clean accuracy is {test_acc}')
    
    asr = bd_test_m['test_correct']/bd_test_m['test_total']
    logging.info(bd_test_m)
    logging.info(f'test asr is {asr}')
    
    cv_acc = cv_train_m['test_correct']/cv_train_m['test_total']
    logging.info(cv_train_m)
    logging.info(f'train cv acc is {cv_acc}')
    
    cv_similarity_metric = calc_sisa_similarity(sisa_aggregator.models, cv_train_dl, device)
    avg_cv_similarity = cv_similarity_metric['avg_similarity']
    logging.info(f'The avearge similarity of cover samples is {avg_cv_similarity}')
    
    cl_similarity_metric = calc_sisa_similarity(sisa_aggregator.models, cl_train_dl, device)
    avg_cl_similarity = cl_similarity_metric['avg_similarity']
    logging.info(f'The avearge similarity of clean samples is {avg_cl_similarity}')
    
    #draw_hist(cv_similarity_metric['similarities'], cl_similarity_metric['similarities'], args.hist_name, [1 * x / 1e2  for x in range(60)])
    cv_similarities = cv_similarity_metric['similarities']
    cl_similarities = cl_similarity_metric['similarities']
    with open('cv_similarities.pkl', 'wb') as file:
        pickle.dump(cv_similarities, file)
    with open('cl_similarities.pkl', 'wb') as file:
        pickle.dump(cl_similarities, file)
    
    logging.info(f"Assuming the attacker sets the suspicious threshold of SMS as {args.sms_threshold}")
    cv_safe_ratio = sum([1 for x in cv_similarity_metric['similarities'] if x < args.sms_threshold]) / len(cv_index)
    cl_safe_ratio = sum([1 for x in cl_similarity_metric['similarities'] if x < args.sms_threshold]) / len(cl_index)
    logging.info(f'The unsuspicious ratio through of cover samples are {cv_safe_ratio}')
    logging.info(f'The unsuspicious ratio through of clean samples are {cl_safe_ratio}')
    
if __name__ == '__main__':
    main()