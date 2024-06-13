'''
Reference: https://github.com/HanxunH/CognitiveDistillation/
Code for ICLR 2023 Paper "Distilling Cognitive Backdoor Patterns within an Image"

Note: CD-Scanner return the score of both clean-training data, bd training data and cv training data
Note that here we only implement the scanning part of cognative distallation, but not implement the fine-tuning migatation yet.

Unfortunately, this doesn't work. Though I directly copy the official code.
For some possibility, poisoning rate may matter a lot.
'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import random

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
from matplotlib import image as mlt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from utils.bd_dataset_v2 import dataset_wrapper_with_transform
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, xy_iter

'''
For Cognitive Distillation Analysis, return norm like score
'''

def min_max_normalization(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


class CognitiveDistillationAnalysis():
    def __init__(self, od_type='l1_norm', norm_only=False):
        self.od_type = od_type
        self.norm_only = norm_only
        self.mean = None
        self.std = None
        return

    def train(self, data):
        if not self.norm_only:
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        self.mean = torch.mean(data).item()
        self.std = torch.std(data).item()
        return

    def predict(self, data, t=1):
        if not self.norm_only:
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        p = (self.mean - data) / self.std
        p = torch.where((p > t) & (p > 0), 1, 0)
        return p.numpy()

    def analysis(self, data):
        """
            data (torch.tensor) b,c,h,w
            data is the distilled mask or pattern extracted by CognitiveDistillation (torch.tensor)
        """
        if self.norm_only:
            if len(data.shape) > 1:
                data = torch.norm(data, dim=[1, 2, 3], p=1)
            score = data
        else:
            score = torch.norm(data, dim=[1, 2, 3], p=1)
        score = min_max_normalization(score)
        return 1 - score.numpy()  # Lower for BD



'''
Cognative Distallation main class, return a mask as the cognative pattern
'''

def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)


class CognitiveDistillation(nn.Module):
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = False
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def forward(self, model, images, labels=None):
        model.eval()
        b, c, h, w = images.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(images.device)
        mask_param = nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        if self.get_features:
            features, logits = model(images)
        else:
            logits = model(images).detach()
            print(logits[0])
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(images.device)
            x_adv = images * mask + (1-mask) * torch.rand(b, c, 1, 1).to(images.device)
            if self.get_features:
                adv_fe, adv_logits = model(x_adv)
                if len(adv_fe[-2].shape) == 4:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=[1, 2, 3])
                else:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1(adv_logits, logits).mean(dim=1)
            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()
        mask = self.get_raw_mask(mask_param).detach().cpu()
        logits = model(images * mask.to(images.device)).detach()
        print(logits[0])
        print(torch.norm(mask, p=self.p, dim=[1, 2, 3])[0])
        if self.norm_only:
            return torch.norm(mask, p=1, dim=[1, 2, 3])
        return mask.detach()
    
'''
Main detector of CD
'''
class cd(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--result_name', type=str, default='attack_result.pt')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/nc/config.yaml", help='the path of yaml')

        parser.add_argument('--only_scan', type=int, default=0, 
                             help='Perform defense(0) or just scan backdoor(1)')
        
        #set the parameter for the cd defense
        parser.add_argument('--p', default=1, type=int)
        parser.add_argument('--gamma', default=0.01, type=float)
        parser.add_argument('--beta', default=10, type=float)
        parser.add_argument('--num_steps', default=100, type=int)
        parser.add_argument('--step_size', default=0.1, type=float)
        parser.add_argument('--mask_channel', default=1, type=int)
        parser.add_argument('--norm_only', action='store_true', default=False)
        parser.add_argument('--index', type=str, help='index of clean data')
                
        
    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        self.attack_file = attack_file
        save_path = 'record/' + result_file + '/defense/cd/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/' + self.args.result_name)

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
        
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)
        args = self.args
        result = self.result

        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        model.eval()
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        self.set_trainer(model)
        criterion = argparser_criterion(args)

        

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
                self.result['bd_train'].wrapped_dataset,
                train_tran,
                None,
        )
        
        train_indicator = bd_train_dataset_with_transform.poison_indicator
        bd_index = np.array(np.where(train_indicator!=0)).squeeze()
        clean_index = np.array(np.where(train_indicator==0)).squeeze()
        
        ### get the backdoor samples
        bd_indices = bd_index
        bd_dataset = Subset(bd_train_dataset_with_transform, bd_indices)
        
        ### select a subset of the training dataset as the subset for attacking
        clean_indices = clean_index
        sample_num = int(0.005 * len(clean_indices))  # sampling for effectiveness
        sample_list = random.sample(list(range(len(clean_indices))), sample_num)
        clean_indices = clean_indices[sample_list]
        clean_dataset = Subset(bd_train_dataset_with_transform, clean_indices)
        
        bd_data_dl = DataLoader(bd_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        cl_data_dl = DataLoader(clean_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        
        detector = CognitiveDistillation(p=args.p, gamma=args.gamma, beta=args.beta,
                                        num_steps=args.num_steps, lr=args.step_size,
                                        mask_channel=args.mask_channel, norm_only=args.norm_only)
        
        analyzer = CognitiveDistillationAnalysis()
        
        '''
        # Run detections on clean set
        cl_patterns = []
        for images, labels, *other_info in tqdm(cl_data_dl):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            batch_rs = detector(model, images, labels)
            cl_patterns.append(batch_rs.detach().cpu())
        cl_patterns = torch.cat(cl_patterns, dim=0)
        cl_score = analyzer.analysis(cl_patterns) 
        logging.info(f'The average L1-norm for clean samples is {torch.norm(cl_patterns, p=1, dim=[1, 2, 3]).mean()}, the analysis score is {np.mean(cl_score)}')
        '''
        
        # Run detections on backdoor set
        bd_patterns = []
        for images, labels, *other_info in tqdm(bd_data_dl):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            batch_rs = detector(model, images, labels)
            bd_patterns.append(batch_rs.detach().cpu())
        bd_patterns = torch.cat(bd_patterns, dim=0)   
        bd_score = analyzer.analysis(bd_patterns)      
        logging.info(f'The average L1-norm for backdoor samples is {torch.norm(bd_patterns, p=1, dim=[1, 2, 3]).mean()}, the analysis score is {np.mean(bd_score)}')
        
        ### TODO: implement the backdoor inversion and fine-tuning mitigation process
        raise NotImplementedError
         
        return
        
    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    cd.add_arguments(parser)
    args = parser.parse_args()
    cd_method = cd(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    result = cd_method.defense(args.result_file)