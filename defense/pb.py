'''
Reference: https://github.com/Gwinhen/PixelBackdoor
Tao, Guanhong, Guangyu Shen, Yingqi Liu, Shengwei An, Qiuling Xu, Shiqing Ma, Pan Li, and Xiangyu Zhang. n.d. 
“Better Trigger Inversion Optimization in Backdoor Scanning.”

Note: Original setting is fit for both global and specific backdoors. Here we only implemented the global version. 
To further fit for specific version, need to seperate the dataset in PixelBackdoor.mitigation.

Note: NC-Scanner return the norm and anomaly index of each class.
'''

import numpy as np
import sys
import torch
import os

sys.path.append('../')
sys.path.append(os.getcwd())

from torchvision import transforms as T

import argparse
import numpy as np

import random
import time
import logging
from pprint import  pformat
from defense.base import defense
from matplotlib import image as mlt
from PIL import Image
import yaml
import torchvision
import cv2

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, xy_iter

class Normalize:

    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone

class Denormalize:

    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

class Recorder:
    def __init__(self, opt):
        super().__init__()

        # Best optimization results
        self.pattern_best = None
        self.reg_best = float("inf")
        self.pixel_best  = float('inf')

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = opt.init_cost
        self.cost_multiplier_up = opt.cost_multiplier
        self.cost_multiplier_down = opt.cost_multiplier ** 1.5

    def reset_state(self, opt):
        self.cost = opt.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, opt):

        result_dir = (os.getcwd() + '/' + f'{opt.log}')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(opt.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = torch.zeros_like(pattern_best)

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(pattern_best, path_trigger, normalize=True)

def outlier_detection(l1_norm_list, idx_mapping, opt):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826 # if normal distribution
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad
    
    anomaly_indexes = (l1_norm_list - median) / mad
    
    print("Median: {}, MAD: {}".format(median, mad))
    logging.info(f'The anomaly index of each class is {anomaly_indexes}.')
    
    def writeDataToCsv(path:str, data:list):
        import pandas as pd
        data = pd.DataFrame(data)
        data.to_csv(path)
    writeDataToCsv(os.path.join(opt.save_path, 'anomaly_index.csv'), 
                       [
                           ['l1_norm'] + l1_norm_list.tolist(),
                           ['anomaly_index'] + anomaly_indexes.tolist()
                       ])
    
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 0.85:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

    if opt.to_file:
        # result_path = os.path.join(opt.result, opt.saving_prefix, opt.dataset)
        # output_path = os.path.join(
        #     result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset, opt.attack_mode)
        # )
        output_path = opt.output_path
        with open(output_path, "a+") as f:
            f.write(
                str(median.cpu().numpy()) + ", " + str(mad.cpu().numpy()) + ", " + str(min_mad.cpu().numpy()) + "\n"
            )
            l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
            f.write(", ".join(l1_norm_list_to_save) + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 0.85:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    logging.info(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )

    return flag_list

class PixelBackdoor(defense):
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
                
        self.input_shape = (args.input_channel, args.input_height, args.input_width)
        self.num_classes = args.num_classes
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.asr_bound = args.asr_bound # 0.8
        self.init_cost = args.init_cost # 1e-3
        self.lr = args.lr
        self.clip_max = args.clip_max
        #self.normalize = self._get_normalize()
        #self.denormalize = self._get_denormalize()
        
        self.epsilon = args.epsilon # 1e-7
        self.patience = args.patience # 10
        self.cost_multiplier_up   = args.cost_multiplier # 1.5
        self.cost_multiplier_down = args.cost_multiplier ** 1.5
        self.pattern_shape = self.input_shape
        
        self.set_devices()
        self.model = generate_cls_model(self.args.model,self.args.num_classes)
        self.model.load_state_dict(self.result['model'])
        if "," in self.device:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{self.model.device_ids[0]}'
            self.model.to(self.args.device)
        else:
            self.model.to(self.args.device)

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
        
        #set the parameter for the pb defense
        parser.add_argument('--ratio', type=float,  help='ratio of training data')
        parser.add_argument('--cleaning_ratio', type=float,  help='ratio of cleaning data')
        parser.add_argument('--unlearning_ratio', type=float, help='ratio of unlearning data')

        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--asr_bound', type=float, default=0.8)
        parser.add_argument('--clip_max', type=float, default=1.0)
        parser.add_argument('--init_cost', type=float, default=1e-3)
        parser.add_argument('--epsilon', type=float, default=1e-7)
        parser.add_argument('--patience', type=int, default=10)
        parser.add_argument('--cost_multiplier', type=float, default=1.5)
        parser.add_argument('--steps', type=int, default=1000)
        
        parser.add_argument('--only_scan', type=int, default=0, 
                             help='Perform defense(0) or just scan backdoor(1)')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        self.attack_file = attack_file
        save_path = 'record/' + result_file + '/defense/pb/'
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
    
    def _get_normalize(self):
        opt = self.args
        if opt.dataset == "cifar10" or opt.dataset == "cifar100":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        elif opt.dataset == 'tiny':
            normalizer = Normalize(opt, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        else:
            raise Exception("Invalid dataset")
        return normalizer
    
    def _get_denormalize(self):
        opt = self.args
        if opt.dataset == "cifar10" or opt.dataset == "cifar100":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        elif opt.dataset == 'tiny':
            denormalizer = Denormalize(opt, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        else:
            raise Exception("Invalid dataset")
        return denormalizer
    
    def train_pattern(self, args, dataloader):
        target = args.target_label
                
        # store best results
        pattern_best     = torch.zeros(self.pattern_shape).to(self.device)
        pattern_pos_best = torch.zeros(self.pattern_shape).to(self.device)
        pattern_neg_best = torch.zeros(self.pattern_shape).to(self.device)
        reg_best = float('inf')
        pixel_best  = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        cost = self.init_cost
        cost_up_counter   = 0
        cost_down_counter = 0
        
        recorder = Recorder(self.args)
        recorder.pattern_best = pattern_best

        # initialize patterns with random values
        for i in range(2):
            init_pattern = np.random.random(self.pattern_shape) * self.clip_max
            init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
            init_pattern = init_pattern / self.clip_max

            if i == 0:
                pattern_pos_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_pos_tensor.requires_grad = True
            else:
                pattern_neg_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_neg_tensor.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        [pattern_pos_tensor, pattern_neg_tensor],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        # start generation
        self.model.eval()
        for step in range(self.args.steps):

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for batch_idx, (inputs, labels, *other_info) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = torch.full(labels.shape, target).to(self.device)
                
                # map pattern variables to the valid range
                pattern_pos =   torch.clamp(pattern_pos_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)
                pattern_neg = - torch.clamp(pattern_neg_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)

                
                # stamp trigger pattern
                #inputs = self.denormalize(inputs)
                x_adv = torch.clamp(inputs + pattern_pos + pattern_neg,
                                    min=0.0, max=self.clip_max)
                #x_adv = self.normalize(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(labels.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, labels)

                # loss for the number of perturbed pixels
                reg_pos  = torch.max(torch.tanh(pattern_pos_tensor / 10)\
                                 / (2 - self.epsilon) + 0.5, axis=0)[0]
                reg_neg  = torch.max(torch.tanh(pattern_neg_tensor / 10)\
                                / (2 - self.epsilon) + 0.5, axis=0)[0]
                loss_reg = torch.sum(reg_pos) + torch.sum(reg_neg)

                # total loss
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss     = np.mean(loss_list)
            avg_acc      = np.mean(acc_list)

            # remove small pattern values
            threshold = self.clip_max / 255.0
            pattern_pos_cur = pattern_pos.detach()
            pattern_neg_cur = pattern_neg.detach()
            pattern_pos_cur[(pattern_pos_cur < threshold)\
                                & (pattern_pos_cur > -threshold)] = 0
            pattern_neg_cur[(pattern_neg_cur < threshold)\
                                & (pattern_neg_cur > -threshold)] = 0
            pattern_cur = pattern_pos_cur + pattern_neg_cur

            # count current number of perturbed pixels
            pixel_cur = np.count_nonzero(
                            np.sum(np.abs(pattern_cur.cpu().numpy()), axis=0)
                        )

            # record the best pattern
            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best\
                    and pixel_cur < pixel_best:
                reg_best = avg_loss_reg
                pixel_best = pixel_cur

                pattern_pos_best = pattern_pos.detach()
                pattern_pos_best[pattern_pos_best < threshold] = 0
                init_pattern = pattern_pos_best / self.clip_max
                with torch.no_grad():
                    pattern_pos_tensor.copy_(init_pattern)

                pattern_neg_best = pattern_neg.detach()
                pattern_neg_best[pattern_neg_best > -threshold] = 0
                init_pattern = - pattern_neg_best / self.clip_max
                with torch.no_grad():
                    pattern_neg_tensor.copy_(init_pattern)

                pattern_best = pattern_pos_best + pattern_neg_best
                
                recorder.pattern_best = pattern_best
                recorder.reg_best = reg_best
                recorder.pixel_best = pixel_best
                recorder.save_result_to_dir(self.args)
                

            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack asr: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}, '\
                                 .format(avg_loss_ce, avg_loss_reg, reg_best)\
                                 + 'size: {:.0f}  '.format(pixel_best))
                sys.stdout.flush()

        size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rtrigger size of pair all - {:d}: {:d}\n'\
                         .format(target, size))
        sys.stdout.flush()

        return recorder, self.args
    
    def mitigation(self):
        fix_random(self.args.random_seed)
        args = self.args
        result = self.result

        # Prepare model, optimizer, scheduler
        model = self.model
        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        self.set_trainer(model)
        criterion = argparser_criterion(args)
        
        # For pairwise specific backdoor, need to separate dataset here to get the proper dataset for scanning.
        # For instance, the (trigger class, target class) is (6, 0), the data_set_o should only contain class 6 samples.
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length) 
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader
        

        # a. initialize the model and trigger
        result_path = os.getcwd() + '/' + f'{args.save_path}/pb/trigger/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        args.output_path = result_path + "{}_output_clean.txt".format(args.dataset)
        if args.to_file:
            with open(args.output_path, "w+") as f:
                f.write("Output for PixBackdoor:  - {}".format(args.dataset) + "\n")

        init_mask = np.ones((1, args.input_height, args.input_width)).astype(np.float32)
        init_pattern = np.ones((args.input_channel, args.input_height, args.input_width)).astype(np.float32)

        flag = 0
        for test in range(args.n_times_test):
            # b. train triggers according to different target labels
            print("Test {}:".format(test))
            logging.info("Test {}:".format(test))
            if args.to_file:
                with open(args.output_path, "a+") as f:
                    f.write("-" * 30 + "\n")
                    f.write("Test {}:".format(str(test)) + "\n")

            patterns = []
            idx_mapping = {}

            regs = []
            for target_label in range(args.num_classes):
                print("----------------- Analyzing label: {} -----------------".format(target_label))
                logging.info("----------------- Analyzing label: {} -----------------".format(target_label))
                args.target_label = target_label
                recorder, args = self.train_pattern(args, trainloader)

                pattern = recorder.pattern_best
                patterns.append(pattern)
                #reg = recorder.reg_best
                reg = torch.norm(pattern, p=args.use_norm)
                logging.info(f'The regularization of pattern for target label {target_label} is {reg}')
                idx_mapping[target_label] = len(patterns) - 1
                regs.append(reg)

            # c. Determine whether the trained reverse trigger is a real backdoor trigger
            #print(type(res[0]))
            l1_norm_list = torch.stack([torch.tensor(reg) for reg in regs]) #torch.stack([torch.norm(pattern, p=args.use_norm) for pattern in patterns])
            logging.info("{} labels found".format(len(l1_norm_list)))
            logging.info("Norm values: {}".format(l1_norm_list))
            flag_list = outlier_detection(l1_norm_list, idx_mapping, args)
            if len(flag_list) != 0:
                flag = 1

        if flag == 0:
            logging.info('This is not a backdoor model')
            test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
            data_bd_testset = self.result['bd_test']
            data_bd_testset.wrap_img_transform = test_tran
            data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

            data_clean_testset = self.result['clean_test']
            data_clean_testset.wrap_img_transform = test_tran
            data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)
            
            agg = Metric_Aggregator()

            test_dataloader_dict = {}
            test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
            test_dataloader_dict["bd_test_dataloader"] = data_bd_loader
            
            model = generate_cls_model(args.model,args.num_classes)
            model.load_state_dict(result['model'])
            self.set_trainer(model)

            self.trainer.set_with_dataloader(
                train_dataloader = trainloader,
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'pb',

                prefetch = self.args.prefetch,
                prefetch_transform_attr_name = "ori_image_transform_in_loading",
                non_blocking = self.args.non_blocking,

                # continue_training_path = continue_training_path,
                # only_load_model = only_load_model,
            )
            clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.trainer.test_current_model(
                test_dataloader_dict, self.args.device,
            )
            agg({
                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })
            agg.to_dataframe().to_csv(f"{args.save_path}nc_df_summary.csv")

            result = {}
            result['model'] = model
            save_defense_result(
                model_name=args.model,
                num_classes=args.num_classes,
                model=model.cpu().state_dict(),
                save_path=args.save_path,
                data_path=self.result['data_path'],
                img_size=self.result['img_size'],
                clean_data=self.result['clean_data'],
                bd_train=self.result['bd_train'],
                bd_test=self.result['bd_test']
            )
            return result  

        if self.args.only_scan:
            result = {}
            result['model'] = model
            save_defense_result(
                model_name=args.model,
                num_classes=args.num_classes,
                model=model.cpu().state_dict(),
                save_path=args.save_path,
                data_path=self.result['data_path'],
                img_size=self.result['img_size'],
                clean_data=self.result['clean_data'],
                bd_train=self.result['bd_train'],
                bd_test=self.result['bd_test']
            )
            return result  
        
        self.set_result(args.result_file)
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)




        # d. select samples as clean samples and unlearning samples, finetune the origin model
        model = generate_cls_model(args.model,args.num_classes)
        model.load_state_dict(result['model'])
        model.to(args.device)
        train_tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
        attack_file = self.attack_file
        self.result = load_attack_result(attack_file + '/' + self.args.result_name)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length) 
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
    
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader

        idx_clean = ran_idx[0:int(len(data_set_o)*(1-args.unlearning_ratio))]
        idx_unlearn = ran_idx[int(len(data_set_o)*(1-args.unlearning_ratio)):int(len(data_set_o))]
        x_new = list()
        y_new = list()
        original_index_array = list()
        poison_indicator = list()
        for ii in range(int(len(data_set_o)*(1-args.unlearning_ratio))):
            x_new.extend([data_set_o.wrapped_dataset[ii][0]])
            y_new.extend([data_set_o.wrapped_dataset[ii][1]])
            original_index_array.extend([len(x_new)-1])
            poison_indicator.extend([0])

        for (label,_) in flag_list:
            mask_path = os.getcwd() + '/' + f'{args.log}' + '{}/'.format(str(label)) + 'mask.png'
            mask_image = mlt.imread(mask_path)
            mask_image = cv2.resize(mask_image,(args.input_height, args.input_width))
            trigger_path = os.getcwd() + '/' + f'{args.log}' + '{}/'.format(str(label)) + 'trigger.png'
            signal_mask = mlt.imread(trigger_path)*255
            signal_mask = cv2.resize(signal_mask,(args.input_height, args.input_width))
            
            x_unlearn = list()
            x_unlearn_new = list()
            y_unlearn_new = list()
            original_index_array_new = list()
            poison_indicator_new = list()
            for ii in range(int(len(data_set_o)*(1-args.unlearning_ratio)),int(len(data_set_o))):
                img = data_set_o.wrapped_dataset[ii][0]
                x_unlearn.extend([img])
                x_np = np.array(cv2.resize(np.array(img),(args.input_height, args.input_width))) * (1-np.array(mask_image)) + np.array(signal_mask)
                x_np = np.clip(x_np.astype('uint8'), 0, 255)
                x_np_img = Image.fromarray(x_np)
                x_unlearn_new.extend([x_np_img])
                y_unlearn_new.extend([data_set_o.wrapped_dataset[ii][1]])
                original_index_array_new.extend([len(x_new)-1])
                poison_indicator_new.extend([0])
            x_new.extend(x_unlearn_new)
            y_new.extend(y_unlearn_new)
            original_index_array.extend(original_index_array_new)
            poison_indicator.extend(poison_indicator_new)

        ori_dataset = xy_iter(x_new,y_new,None)

        data_set_o.wrapped_dataset.dataset = ori_dataset
        data_set_o.wrapped_dataset.original_index_array = original_index_array
        data_set_o.wrapped_dataset.poison_indicator = poison_indicator
        trainloader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

        self.trainer.train_with_test_each_epoch_on_mix(
            trainloader,
            data_clean_loader,
            data_bd_loader,
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.args.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='pb',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
        )
        
        result = {}
        result['model'] = model
        save_defense_result(
                model_name=args.model,
                num_classes=args.num_classes,
                model=model.cpu().state_dict(),
                save_path=args.save_path,
                data_path=self.result['data_path'],
                img_size=self.result['img_size'],
                clean_data=self.result['clean_data'],
                bd_train=self.result['bd_train'],
                bd_test=self.result['bd_test']
            )
        return result

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


################################################################
############                  main                  ############
################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])

    PixelBackdoor.add_arguments(parser)
    args = parser.parse_args()

    pb_method = PixelBackdoor(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    result = pb_method.defense(args.result_file)