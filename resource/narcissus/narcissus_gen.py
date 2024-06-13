'''
An implementation of the ACM CCS'23 paper: `Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information.'
Refenerce: https://github.com/ruoxi-jia-group/Narcissus/tree/main

@inproceedings{10.1145/3576915.3616617,
    author = {Zeng, Yi and Pan, Minzhou and Just, Hoang Anh and Lyu, Lingjuan and Qiu, Meikang and Jia, Ruoxi},
    title = {Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information},
    year = {2023},
    isbn = {9798400700507},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3576915.3616617},
    doi = {10.1145/3576915.3616617},
    abstract = {Backdoor attacks introduce manipulated data into a machine learning model's training set, causing the model to misclassify inputs with a trigger during testing to achieve a desired outcome by the attacker. For backdoor attacks to bypass human inspection, it is essential that the injected data appear to be correctly labeled. The attacks with such property are often referred to as "clean-label attacks." The success of current clean-label backdoor methods largely depends on access to the complete training set. Yet, accessing the complete dataset is often challenging or unfeasible since it frequently comes from varied, independent sources, like images from distinct users. It remains a question of whether backdoor attacks still present real threats.In this paper, we provide an affirmative answer to this question by designing an algorithm to launch clean-label backdoor attacks using only samples from the target class and public out-of-distribution data. By inserting carefully crafted malicious examples totaling less than 0.5\% of the target class size and 0.05\% of the full training set size, we can manipulate the model to misclassify arbitrary inputs into the target class when they contain the backdoor trigger. Importantly, the trained poisoned model retains high accuracy for regular test samples without the trigger, as if the model is trained on untainted data. Our technique is consistently effective across various datasets, models, and even when the trigger is injected into the physical world.We explore the space of defenses and find that Narcissus can evade the latest state-of-the-art defenses in their vanilla form or after a simple adaptation. We analyze the effectiveness of our attack - the synthesized Narcissus trigger contains durable features as persistent as the original target class features. Attempts to remove the trigger inevitably hurt model accuracy first.},
    booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
    pages = {771-785},
    numpages = {15},
    keywords = {ai security, backdoor attack, clean-label attack},
    location = {, Copenhagen, Denmark, },
    series = {CCS '23}
}

'''

import argparse
import logging
import random
import torch
import time
import sys
import os

from pprint import pformat
from narcissus_utils import *

import torchvision.transforms as transforms
from tqdm import tqdm


os.chdir(sys.path[0])
sys.path.append('../../')
os.getcwd()

from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.aggregate_block.fix_random import fix_random
from utils.log_assist import get_git_info
from utils.bd_dataset_v2 import dataset_wrapper_with_transform, get_labels
from torch.utils.data import DataLoader, Subset
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from torchvision.utils import save_image

import matplotlib.pyplot as plt


def add_narcissus_args(parser):
    parser.add_argument('--noise_size', type=int, default=None, 
                        help='Noise size, default is None which stands for full image size')
    
    parser.add_argument('--l_inf_r', type=float, default=16/255,
                        help='Radius of the L-inf ball')
    
    parser.add_argument('--surrogate_model', type=str, default='resnet18',
                        help='Model arch for generation surrogate model and trigger.') 
    parser.add_argument('--surrogate_dataset', type=str, default='tiny',
                        help='Dataset used for generating surrogate model.')
    parser.add_argument('--surrogate_epochs', type=int, default=200,
                        help='Epochs for generating surrogate model.')
    ## Unlike the official implementation, we suppose the surrogate model is pre-trained outside aforehand.
    parser.add_argument('--surrogate_pretrain_path',type=str,default=None,
                        help='Path for pretrained surrogate model. Default is None which stands for train-from-scratch.')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='The real-task dataset in Narcissus.')
    
    parser.add_argument('--generating_lr_warmup', type=float, default=0.01,
                        help='Learning rate for trigger generating')
    parser.add_argument('--warmup_round', type=int, default=5,
                        help='Poison warm up round')
    
    parser.add_argument('--generating_lr_tri', type=float, default=0.01,
                        help='Learning rate for trigger generating')
    parser.add_argument('--gen_round', type=int, default=1000,
                        help='Generating round for trigger genreating')
    
    parser.add_argument('--patch_mode', type=str, default='add',
                        help='The model for adding the noise.')
    
    parser.add_argument('--noise_save_path', type=str, default='./checkpoint/narcissus_noise_cls_6',
                        help='Save path of the narcissus patch.')
    
    parser.add_argument('--target', type=int, default=6,
                        help='The target class for Narcissus.')
    parser.add_argument('--target_img_ratio', type=float, default=1.0,
                        help='The ratio of target images the adversary holds, default is 1.0 which means all target samples.')
    parser.add_argument('--target_dataset', type=str, default='train',
                        help='Which target dataset to use, [\'train\', \'test\'].')
    
    return parser

def add_normal_args(parser):
    parser.add_argument("-n", "--num_workers", type=int, default=16, help="dataloader num_workers")
    parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
    parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
    parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr_scheduler', type=str,
                            help='which lr_scheduler use for optimizer')
    parser.add_argument('--batch_size', type=int, default=350) # take the advised value from official code
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help='weight decay of sgd')
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--client_optimizer', type=int)
    parser.add_argument('--dataset_path', type=str, default='../../data')
    parser.add_argument('--random_seed', type=int,
                            help='random_seed', default=0)
    
    return parser

########################### Narcissus Generating ############################

class Narcissus:
    def set_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_normal_args(parser)
        parser = add_narcissus_args(parser)
        return parser
    
    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.surrogate_num_class = get_num_classes(args.surrogate_dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.surrogate_dataset_path = f"{args.dataset_path}/{args.surrogate_dataset}"
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        return args
    
    def prepare(self, args):
        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.
        logger.setLevel(0)

        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        # record the git infomation for debug (if available.)
        try:
            logging.debug(pformat(get_git_info()))
        except:
            logging.debug('Getting git info fails.')

        ### set the random seed
        fix_random(int(args.random_seed))

        self.args = args
        
    def dataset_prepare(self):

        assert 'args' in self.__dict__

        args = self.args

        ######## First get surrogate dataset
        dataset, args.dataset = args.dataset, args.surrogate_dataset
        dataset_path, args.dataset_path = args.dataset_path, args.surrogate_dataset_path
                
        surrogate_train_dataset_without_transform, \
        _, \
        surrogate_train_label_transform, \
        _, \
        _, \
        _ = dataset_and_transform_generate(args)
        
        logging.debug(f"dataset_and_transform_generate done for surrogate dataset {args.surrogate_dataset}.")
        
        ######## Then get task dataset with only target class
        args.dataset = dataset
        args.dataset_path = dataset_path
        
        train_dataset_without_transform, \
        _, \
        train_label_transform, \
        test_dataset_without_transform, \
        _, \
        test_label_transform = dataset_and_transform_generate(args)
        
        if args.target_dataset == 'train':
            train_labels = get_labels(train_dataset_without_transform)
            train_target_list = list(np.where(np.array(train_labels)==args.target)[0])
            train_target_without_transform = Subset(train_dataset_without_transform,train_target_list)
        elif args.target_dataset == 'test':
            train_labels = get_labels(test_dataset_without_transform)
            train_target_list = list(np.where(np.array(train_labels)==args.target)[0])
            train_target_without_transform = Subset(test_dataset_without_transform,train_target_list)
        else:
            raise ValueError('target_dataset: [\'train\', \'test\'].')
        
        assert 0 < args.target_img_ratio <= 1.0
        
        if args.target_img_ratio < 1.0:
            target_len = len(train_target_without_transform)
            target_img_num = int(len(train_target_without_transform) * args.target_img_ratio)
            sample_list = random.sample(list(range(target_len)), target_img_num)
            train_target_without_transform = Subset(train_target_without_transform, sample_list)
                
        logging.info(f"dataset_and_transform_generate done for real-task dataset {args.dataset}.")
        
        
        ######## Get Narcissus transforms
        input_len = args.input_height
        
        #The argumention use for surrogate model training stage
        transform_surrogate_train = transforms.Compose([
            transforms.Resize(input_len),
            transforms.RandomCrop(input_len, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        #The argumention use for all training set
        transform_train = transforms.Compose([
            transforms.RandomCrop(input_len, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])        
        
        ######## Get dataset wrapped with transform

        if args.target_dataset == 'train':
            target_dataset_with_transform = dataset_wrapper_with_transform(
                train_target_without_transform,
                transform_train,
                train_label_transform
            )
        elif args.target_dataset == 'test':
            target_dataset_with_transform = dataset_wrapper_with_transform(
                train_target_without_transform,
                transform_train,
                test_label_transform
            )
        else:
            raise ValueError('target_dataset: [\'train\', \'test\'].')
        
        surrogate_train_dataset_with_transform = dataset_wrapper_with_transform(
            surrogate_train_dataset_without_transform,
            transform_surrogate_train,
            surrogate_train_label_transform
        )
        
        ######## concoct surrogate_dataset and target_dataset 
        concoct_train_dataset = concoct_dataset(target_dataset_with_transform, surrogate_train_dataset_with_transform)
        target_dataset_with_transform = concoct_dataset([], target_dataset_with_transform) # Cannot understand why, but it is necessary. Maybe somthing wrong with Subset.
        
        ######## Get dataloaders
        self.surrogate_loader = DataLoader(concoct_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        self.poi_warm_up_loader = DataLoader(target_dataset_with_transform, batch_size=args.batch_size, shuffle=True, num_workers=16)
        self.trigger_gen_loader = DataLoader(target_dataset_with_transform, batch_size=args.batch_size, shuffle=True, num_workers=16)

    def train_surrogate_model(self):
        
        args = self.args
        
        ######## First prepare model architecture for surrogate model
        surrogate_model = generate_cls_model( 
            model_name=args.surrogate_model,
            num_classes=args.surrogate_num_class + 1, # consider surrogate dataset concanated with 1 target class
            image_size=args.img_size[0],
        )
        
        ######## Prepare device information
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )
        
        if "," in args.device:
            surrogate_model = torch.nn.DataParallel(
                surrogate_model,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
                
        if not (args.__contains__('surrogate_pretrain_path') and args.surrogate_pretrain_path):
            ######## Set training information with the same configuration from the official code
            surrogate_epochs = args.surrogate_epochs
            surrogate_loader = self.surrogate_loader
                        
            surrogate_model = surrogate_model.to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            # outer_opt = torch.optim.RAdam(params=base_model.parameters(), lr=generating_lr_outer)
            surrogate_opt = torch.optim.SGD(params=surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(surrogate_opt, T_max=surrogate_epochs)
            
            #Training the surrogate model
            logging.info('Training the surrogate model')
            for epoch in range(0, surrogate_epochs):
                surrogate_model.train()
                loss_list = []
                epoch_start_time = time.time()
                for images, labels in surrogate_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    surrogate_opt.zero_grad()
                    outputs = surrogate_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    loss_list.append(float(loss.data))
                    surrogate_opt.step()
                surrogate_scheduler.step()
                ave_loss = np.average(np.array(loss_list))
                epoch_end_time = time.time()
                print('Epoch:%d, Loss: %.03f, time: %.03f' % (epoch, ave_loss, epoch_end_time - epoch_start_time))
            #Save the surrogate model
            target_ratio = ''
            if args.target_img_ratio < 1:
                target_ratio = str(args.target_img_ratio).replace('.', 'dot')
            save_path = f'./checkpoint/surrogate_pretrain_{str(surrogate_epochs)}_{args.dataset}_{args.surrogate_dataset}_{args.target_dataset}_{target_ratio}.pth'
            torch.save(surrogate_model.state_dict(), save_path)
        else:
            model_state_dict = torch.load(args.surrogate_pretrain_path)
            surrogate_model.load_state_dict(model_state_dict)
        
        self.surrogate_model = surrogate_model
        
    def poison_warm_up(self):
        
        args = self.args
        device = self.device
        
        #Prepare models and optimizers for poi_warm_up training
        poi_warm_up_model = self.surrogate_model.to(device)

        poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=args.generating_lr_warmup)
        
        #Poi_warm_up stage
        poi_warm_up_model.train()
        for param in poi_warm_up_model.parameters():
            param.requires_grad = True

        #Training the surrogate model
        poi_warm_up_loader = self.poi_warm_up_loader
        warmup_round = args.warmup_round
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(0, warmup_round):
            poi_warm_up_model.train()
            loss_list = []
            for images, labels in poi_warm_up_loader:
                images, labels = images.to(device), labels.to(device)
                poi_warm_up_model.zero_grad()
                poi_warm_up_opt.zero_grad()
                outputs = poi_warm_up_model(images)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph = True)
                loss_list.append(float(loss.data))
                poi_warm_up_opt.step()
            ave_loss = np.average(np.array(loss_list))
            logging.info('Epoch:%d, Loss: %e' % (epoch, ave_loss))
            
        self.poi_warm_up_model = poi_warm_up_model 
            
    def trigger_generating(self):
        
        args = self.args
        device = self.device
        poi_warm_up_model = self.poi_warm_up_model.to(device)
        
        #Trigger generating stage
        generating_lr_tri = args.generating_lr_tri
        gen_round = args.gen_round
        l_inf_r = args.l_inf_r
        trigger_gen_loader = self.trigger_gen_loader
        patch_mode = args.patch_mode
        criterion = torch.nn.CrossEntropyLoss()
        
        for param in poi_warm_up_model.parameters():
            param.requires_grad = False
        
        noise_size = args.noise_size if (args.__contains__('noise_size') and args.noise_size) else args.img_size[0]
        noise = torch.zeros((1, 3, noise_size, noise_size), device=device)
        batch_pert = torch.autograd.Variable(noise.to(device), requires_grad=True)
        batch_opt = torch.optim.RAdam(params=[batch_pert],lr=generating_lr_tri)
        for _ in tqdm(range(gen_round)):
            loss_list = []
            for images, labels in trigger_gen_loader:
                images, labels = images.to(device), labels.to(device)
                new_images = torch.clone(images)
                clamp_batch_pert = torch.clamp(batch_pert,-l_inf_r,l_inf_r)
                new_images = torch.clamp(apply_noise_patch(clamp_batch_pert,new_images.clone(),mode=patch_mode),-1,1)
                per_logits = poi_warm_up_model.forward(new_images)
                loss = criterion(per_logits, labels)
                loss_regu = torch.mean(loss)
                batch_opt.zero_grad()
                loss_list.append(float(loss_regu.data))
                loss_regu.backward(retain_graph = True)
                batch_opt.step()
            ave_loss = np.average(np.array(loss_list))
            ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
            print('Gradient:',ave_grad,'Loss:', ave_loss)
            if ave_grad == 0:
                break
        
        original_noise = noise.clone().detach().cpu()
        noise = torch.clamp(batch_pert,-l_inf_r,l_inf_r)
        best_noise = noise.clone().detach().cpu()
        logging.info('Noise max val:',str(noise.max().cpu().item()))
        
        #Save the trigger
        if not args.__contains__('noise_save_path'):
            noise_save_path = './checkpoint/best_noise'+'_'+ time.strftime("%m-%d-%H_%M_%S",time.localtime(time.time())) 
        else:
            noise_save_path = args.noise_save_path
        
        # best_noise = best_noise.reshape(args.img_size) 
        # Not need to reshape the best_noise
        # best_noise.shape = [1, 3, img_size, img_size]
        
        np.save(noise_save_path, best_noise)
        save_image(best_noise, noise_save_path+'.png')
        
        # np.save(noise_save_path+'_ori', original_noise)
        # save_image(original_noise, noise_save_path+'_ori.png')
        
        logging.info(f'The Narcissus backdoor trigger is generated successfully, saved at {noise_save_path}.')
        

############################# Main Function ##########################

if __name__ == '__main__':
    narcissus_gen = Narcissus()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    
    parser = narcissus_gen.set_args(parser)
    args = parser.parse_args()
    args = narcissus_gen.process_args(args)
    
    narcissus_gen.prepare(args)
    narcissus_gen.dataset_prepare()
    narcissus_gen.train_surrogate_model()
    narcissus_gen.poison_warm_up()
    narcissus_gen.trigger_generating()
        
    