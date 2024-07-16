# sample-based mifigation defense: model uncertainty
import yaml
import argparse
import sys
import logging
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#os.chdir(sys.path[0])
sys.path.append('./')
os.getcwd()


from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape

import torch
from torch.utils.data import DataLoader

from uba.uba_utils.basic_utils import get_dataset
from attack_sisa.aggregator import aggregator 
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_criterion
from uba.uba_utils.basic_utils import get_perturbed_datasets

    
def writeDataToCsv(path:str, data:list):
    import pandas as pd
    data = pd.DataFrame(list(data))
    data.to_csv(path)

def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--result_folder", type=str, default="",
                        help="where to get the sisa model and dataset results")
    parser.add_argument("--seed", type=int, default=0, help="set random seed for pytorch")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument('--scale', type=int, default=None, help='')
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="dataloader num_workers")
    parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help="dataloader pin_memory")
    parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help=".to(), set the non_blocking = ?")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default="preactresnet18")
    
    parser.add_argument('--dataset_folder', type=str, default="./record/your_dataset_folder",
                        help="The dataset result folder on which the SISA model is trained. This is required.")    
    parser.add_argument('--dataset_name', type=str, default="pert_result.pt",
                        help="The dataset result name on which the SISA model is trained. This is required.")
    return parser

def process_args(args):
    args.terminal_info = sys.argv
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.surrogate_model_folder = args.result_folder
    return args

def get_result(args):    
    # SISA are composed of different models, so here set model temporarily to None.
    # model = get_surrogate_model(args)
    
    # args,\
    # clean_train_dataset_with_transform_for_train, bd_train_dataset_with_transform_for_train,\
    # clean_train_dataset_with_transform, clean_test_dataset_with_transform,\
    # bd_train_dataset_with_transform, bd_test_dataset_with_transform \
    # = get_dataset(args)
    
    args.add_cover = True
    
    clean_train_dataset_with_transform, \
    clean_test_dataset_with_transform, \
    bd_train_dataset_with_transform, \
    bd_test_dataset_with_transform = get_perturbed_datasets(args)
    
    result_dict = {
        'model': None,
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
    
    # model = result_dict['model']
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
    
    sisa_folder = args.result_folder
    num_shards = 0
    for path in os.listdir(sisa_folder):
        if os.path.isdir(os.path.join(sisa_folder, path)):
            num_shards += 1
    
    device = torch.device(
        (
            f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
            # since DataParallel only allow .to("cuda")
        ) if torch.cuda.is_available() else "cpu"
    )
    
    # get the aggregated result model
    attack_models = []
    unlearn_models = []
    for i in range(num_shards):
        shard_path = sisa_folder + f'/shard{i}'
        attack_path = shard_path + '/cover/cover_result.pt'
        unlearn_path = shard_path + '/unlearn/unlearn_result.pt'
        assert(os.path.exists(attack_path))
        # get the attack model
        attack_net_dict = torch.load(attack_path, map_location=device)
        attack_net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )
        attack_net.load_state_dict(attack_net_dict['model'])
        attack_models.append(attack_net)
        # get the unlearn model, if there exists one
        if os.path.exists(unlearn_path):
            unlearn_net_dict = torch.load(unlearn_path, map_location=device)
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
    criterion = argparser_criterion(args)
    
    attack_test_result = \
        attack_aggregator.backdoorTest(clean_test_dl, 
                                       bd_test_dl, 
                                       device,
                                       criterion
                                       )
        
    if unlearn_aggregator:
        unlearn_test_result = \
            unlearn_aggregator.backdoorTest(clean_test_dl, 
                                       bd_test_dl, 
                                       device,
                                       criterion)
    else:
        unlearn_test_result = None
    
    attack_test_acc = attack_test_result["test_acc"]
    attack_test_asr = attack_test_result["test_asr"]
    print(f'Before SISA unlearning, the BA is {attack_test_acc} while the ASR is {attack_test_asr}.')
    
    if unlearn_test_result is not None:
        unlearn_test_acc = unlearn_test_result["test_acc"]
        unlearn_test_asr = unlearn_test_result["test_asr"]
        print(f'After SISA unlearning, the BA is {unlearn_test_acc} while the ASR is {unlearn_test_asr}.')
    
if __name__ == '__main__':
    main()