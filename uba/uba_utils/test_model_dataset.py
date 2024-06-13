import argparse
import logging
import random
import torch
import sys

import numpy as np

from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.save_load_attack import load_attack_result
from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.bd_dataset_v2 import dataset_wrapper_with_transform
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from torch.utils.data import DataLoader



def fix_random(
        random_seed: int = 0
) -> None:
    '''
    use to fix randomness in the script, but if you do not want to replicate experiments, then remove this can speed up your code
    :param random_seed:
    :return: None
    '''
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset_folder", type=str, default="",
                        help="folder path to get the dataset for test")
    parser.add_argument("--dataset_name", type=str, default="attack_result.pt",
                        help="file name to get the dataset for test, should be *.pt")
    parser.add_argument("--model_folder", type=str, default="",
                        help="folder path to get the model for test")
    parser.add_argument("--model_name", type=str, default="attack_result.pt",
                        help="file name to get the model for test, should be *.pt")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default='0')
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="dataloader num_workers")
    parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help="dataloader pin_memory")
    parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help=".to(), set the non_blocking = ?")

    
def get_dataset_and_model(args):
    model_path = args.model_folder + '/' + args.model_name
    result = torch.load(model_path)
    net = generate_cls_model(
        model_name=result['model_name'],
        num_classes=result['num_classes'],
        image_size=result['img_size'][0],
    )
    net.load_state_dict(result['model'])
    
    dataset_path = args.dataset_folder + '/' + args.dataset_name
    dataset_dict = load_attack_result(dataset_path)
    
    args.num_classes = dataset_dict['num_classes']
    args.img_size = dataset_dict['img_size']
    args.input_height, args.input_width, args.input_channel = args.img_size
    args.clean_data = dataset_dict['clean_data']
    args.data_path = dataset_dict['data_path']
    
    bd_train_dataset_without_transform = dataset_dict['bd_train'].wrapped_dataset    
        
    # Here We set dataset with test transform for PGD attacking
    test_img_transform = get_transform('test', *(args.img_size[:2]), train=False)
    test_label_transform = None
    
    clean_test_dataset_with_transform = dataset_dict['clean_test']
    bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset_without_transform,
            test_img_transform,
            test_label_transform,
    )
    bd_test_dataset_with_transform = dataset_dict['bd_test']
    
    result_dict = {
        'model': net,
        'clean_test': clean_test_dataset_with_transform,
        'bd_train': bd_train_dataset_with_transform,
        'bd_test': bd_test_dataset_with_transform
    }
    
    return result_dict



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


def test_model_dataset(args):
    args.terminal_info = sys.argv ## process args
    
    device = torch.device(
        (
            f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
            # since DataParallel only allow .to("cuda")
        ) if torch.cuda.is_available() else "cpu"
    )
    
    fix_random(args.seed)
    
    ''' 2. get bd_test_dataset, bd_train_dataset, clean_test_dataset and model according to args'''
    result_dict = get_dataset_and_model(args)
    
    model = result_dict['model']
    clean_test_dataset_with_transform = result_dict['clean_test']
    bd_train_dataset_with_transform = result_dict['bd_train']
    bd_test_dataset_with_transform = result_dict['bd_test'] 
    
    ''' 3. test model performence'''
    bd_train_dl = DataLoader(bd_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    clean_test_dl = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    bd_test_dl = DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    bd_train_m, clean_test_m, bd_test_m = \
        model_test(train_dl=bd_train_dl,
                   clean_test_dl=clean_test_dl,
                   bd_test_dl=bd_test_dl,
                   model=model,
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
    
def main():
    ''' 1. get config and param'''
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser=parser)
    args = parser.parse_args()
    
    test_model_dataset(args)
    
if __name__ == '__main__':
    main()