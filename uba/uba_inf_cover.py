import torch
import sys, yaml, os

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import  pformat
import numpy as np
import torch
import time
import logging
from tqdm import tqdm
import random
from copy import deepcopy
from time import time

from torchvision.transforms import *
from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from torch.utils.data import DataLoader, Subset
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler


from pytorch_influence_functions.influence_function import I_pert_loss, avg_s_test, grad_z
from uba.uba_utils.basic_utils import *


class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the original model's training mode to `test`
        by `.eval()` only during an attack process.
    """

    def __init__(self, name, model, device):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = device

        self._targeted = 1
        self._attack_mode = "original"
        self._return_type = "float"

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def set_attack_mode(self, mode):
        r"""
        Set the attack mode.

        Arguments:
            mode (str) : 'original' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
        """
        if self._attack_mode == "only_original":
            raise ValueError(
                "Changing attack mode is not supported in this attack method."
            )

        if mode == ["original", "influence_adversarial"]:
            self._attack_mode = "original"
            self._targeted = 1 # gradient ascent for maximum 
            self._transform_label = self._get_label
        elif mode == "targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._transform_label = self._get_label
        elif mode == "least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(
                mode
                + " is not a valid mode. [Options : original, targeted, least_likely]"
            )

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == "float":
            self._return_type = "float"
        elif type == "int":
            self._return_type = "int"
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == "int":
                adv_images = adv_images.float() / 255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print(
                    "- Save Progress : %2.2f %% / Accuracy : %2.2f %%"
                    % ((step + 1) / total_batch * 100, acc),
                    end="\r",
                )

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print("\n- Save Complete!")

        self._switch_model()

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels

    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels

    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return ((images * 0.5 + 0.5) * 255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ["model", "attack"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self._attack_mode
        if info["attack_mode"] == "only_original":
            info["attack_mode"] = "original"

        info["return_type"] = self._return_type

        return (
                self.attack
                + "("
                + ", ".join("{}={}".format(key, val) for key, val in info.items())
                + ")"
        )

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == "int":
            images = self._to_uint(images)

        return images

class Influence_PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Combined with influence function of perturbing a datapoint in the paper 
    'Understanding Black-Box Predictions Via Influence Function'
    [https://arxiv.org/abs/1703.04730]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT : 0.3)
        alpha (float): step size. (DEFALUT : 2/255)
        steps (int): number of steps. (DEFALUT : 40)
        random_start (bool): using random initialization of delta. (DEFAULT : False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps = 8/255, alpha = 1/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, 
                 #poison_model,
                 s_test_vec,
                 eps=8 / 255, 
                 alpha=2 / 255, 
                 steps=40, 
                 random_start=False, 
                 device=None,):
        super(Influence_PGD, self).__init__("Influence_PGD", model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.s_test = s_test_vec
        
        '''
        self.poison_model = poison_model
        
        num_layers = len(list(self.poison_model.children()))
        for idx, child in enumerate(self.poison_model.children()):
            if idx < num_layers - 3:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
        '''

    def forward(self, images, labels, original_images=None):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        
        net = self.model
        num_layers = len(list(net.children()))
        
        if original_images is None:
            original_images = images
        else:
            original_images = original_images.to(self.device)
        
        adv_images = images.clone().detach()
        
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=-1, max=1)

        for _ in range(self.steps):            
            adv_images.requires_grad = True
            
            for idx, child in enumerate(net.children()):
                if idx < num_layers - 3:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True
            
            i_pert_loss = I_pert_loss(
                self.s_test, net, adv_images, labels, device=self.device, all_param=False
            )[0]
            
            adv_images = adv_images.detach() + self.alpha * i_pert_loss.sign() 
            delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(original_images + delta, min=-1, max=1).detach()
            
        return adv_images

def set_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
    parser.add_argument('--device', type = str)
    parser.add_argument('--yaml_path', type=str, default="../config/influence/ibap/default.yaml",
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help='weight decay of sgd')
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--client_optimizer', type=int)
    parser.add_argument('--random_seed', type=int,
                        help='random_seed')
    parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument('--dataset_folder', type=str, default="",
                        help="where to get the dataset results, pay attention that the model should be poisoned")
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--surrogate_model_folder', type=str, default=None,
                        help='where to get the clean model')
    parser.add_argument("--result_name", type=str, default="attack_result.pt", 
                        help='the name of result file in the result folder')
    parser.add_argument("--mini_batch_size", type=int, default=1,
                        help='The mini batch size for hessian vector production.')
    parser.add_argument('--recursion_depth', type=int, default=50,
                        help='the recursion depth, indicates the interation times')
    parser.add_argument('--r_averaging', type=int, default=1, help='repeate times for averaging')
    parser.add_argument('--scale', type=int, default=None, help='')
    parser.add_argument('--damp', type=int, default=None, help='')
    parser.add_argument('--attack', type=str)
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="dataloader num_workers")
    parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help="dataloader pin_memory")
    parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], default=True,
                        help=".to(), set the non_blocking = ?")
    parser.add_argument("--clean_ratio", type=float, default=0.1, 
                        help='Choose clean_ratio * len(clean data) as a subset for attack, default is 0.1.')    
    parser.add_argument('--ft_epochs', type=int, default=10,
                        help='For fine-tuning epochs num, default is 10')
    parser.add_argument('--ap_epochs', type=int, default=10,
                        help='For adversarial perturbation iteration times')
    
    parser.add_argument('--c_num', type=int, default=0, 
                        help='set the number of cover samples, 0 means all existing cover samples')
    parser.add_argument('--pgd_steps', type=int, default=40)
    parser.add_argument('--eps', type=float, default=2)
    parser.add_argument('--alpha', type=float, default=0.25)
    return parser

def add_yaml_to_args(args:argparse.ArgumentParser):
    with open(args.yaml_path, 'r') as f:
        clean_defaults = yaml.safe_load(f)
    clean_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = clean_defaults

def process_args(args:argparse.ArgumentParser):
    args.terminal_info = sys.argv
    args.attack = 'None'
    return args

def finetune(net, 
             fine_tuning_dl, 
             clean_test_dl,
             bd_test_dl,
             device, 
             args,
             ft_layers=3):
    '''
    TODO: implement the code note
    '''
    # Freeze all layers except the last 3 layers
    num_layers = len(list(net.children()))
    
    for idx, child in enumerate(net.children()):
        if idx < num_layers - ft_layers:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
                
    net = net.to(device)

    # Define the loss function and optimizer
    criterion = argparser_criterion(args)
    optimizer, scheduler = argparser_opt_scheduler(net, args)

    # Fine-tuning the model
    for epoch in range(args.ft_epochs):  # Adjust the number of epochs as needed
        running_loss = 0.0
        startTime = time()
        for inputs, labels, *others in tqdm(fine_tuning_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # here since ReduceLROnPlateau need the train loss to decide next step setting.
                scheduler.step(running_loss)
            else:
                scheduler.step()
        
        endTime = time()

        logging.info(f"one epoch (epoch {epoch}) training part done, use time = {endTime - startTime} s")

    ### Test the model after fine-tuning
    fine_tuning_m, clean_test_m, bd_test_m =\
        model_test(train_dl=fine_tuning_dl, 
               clean_test_dl=clean_test_dl,
               bd_test_dl=bd_test_dl,
               model=net,
               args=args)
    logging.info(fine_tuning_m)
    fine_tuing_acc = fine_tuning_m['test_correct']/fine_tuning_m['test_total']
    logging.info(f'fine-tuning clean accuracy is {fine_tuing_acc}')

    logging.info(clean_test_m)
    test_acc = clean_test_m['test_correct']/clean_test_m['test_total']
    logging.info(f'test clean accuracy is {test_acc}')
        
    logging.info(bd_test_m)
    asr = bd_test_m['test_correct']/bd_test_m['test_total']
    logging.info(f'test asr is {asr}')

    logging.info("Fine-tuning complete!")
    return net

def craft_adv_samples(net, 
                    #poison_net,
                    s_test_avg,
                    cv_dataset,
                    args:argparse.ArgumentParser,
                    device=torch.device('cpu')):
    '''
    TODO: implement the code note
    '''
    
    pgd_config = { # TODO: set all PGD params as arguments
        'eps': args.eps, #2
        'alpha': args.alpha, # 0.25
        'steps': args.pgd_steps,
        'max_pixel': 255,
    }
    logging.info("Set Influence_PGD attacker: {}.".format(pgd_config))
    
    max_pixel = pgd_config.pop("max_pixel")
    for k, v in pgd_config.items():
        if k == "eps" or k == "alpha":
            pgd_config[k] = v / max_pixel
    
    attacker = Influence_PGD(model = net, 
                             #poison_model = poison_net,
                             s_test_vec=s_test_avg,
                             **pgd_config, 
                             device = device,)
    
    attacker.set_return_type("int")

    cv_loader = DataLoader(cv_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    perturbed_img = torch.zeros((len(cv_dataset), 3, args.input_height, args.input_width), dtype=torch.uint8)
    target = torch.zeros(len(cv_dataset))
    i = 0
    net.to(device)
    net.eval()
    for (x, y, original_index, *other) in tqdm(cv_loader, desc='craft adv samples'):
        # Adversarially perturb image. Note that torchattacks will automatically
        # move `img` and `target` to the gpu where the attacker.model is located. 
        original_data = other[-1]
        
        img = attacker(x,y, original_data)
        if img.shape[1:] != perturbed_img.shape[1:]:
            # mnist, fashion-mnist
            img = img.detach()[:, :, 2:30, 2:30]
        perturbed_img[i: i + len(img), :, :, :] = img.detach()
        target[i: i + len(y)] = y
        i += img.shape[0]
        
    return perturbed_img, target

def calc_influence(net,
                   s_test_avg, 
                   train_dataset,
                   device,
                   train_dataset_size):
    train_index_size = len(train_dataset)
    influences = []
    for idx in tqdm(range(train_index_size), desc="Calc. influence function"):
        z, t, *others = train_dataset[idx]
        z = z.unsqueeze(0)
        t = torch.tensor(int(t)).unsqueeze(0)
        grad_z_vec = grad_z(z, t, net, device=device)
        # Correspoding code in release-function:
        # predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
        tmp_influence = sum(
            [
                ####################
                # TODO: potential bottle neck, takes 17% execution time
                # torch.sum(k * j).data.cpu().numpy()
                ####################
                torch.sum(k * j).data 
                for k, j in zip(grad_z_vec, s_test_avg)
            ]) / train_dataset_size
        influences.append(tmp_influence.cpu())
    
    #harmful = np.argsort(influences)
    #helpful = harmful[::-1]
        
    return sum(influences), #harmful.tolist(), helpful.tolist()


def main():

    ''' 
    1. config args, save_path, fix random seed 
    '''
    
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser=parser)
    args = parser.parse_args()
    add_yaml_to_args(args=args)
    args = process_args(args)
    
    logging.info(f"get the training setting for specific dataset")
    

    ### save path
    if 'save_folder_name' not in args:
        save_path = generate_save_folder(
            run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + args.attack,
            given_load_file_path=args.load_path if 'load_path' in args else None,
            all_record_folder_path='../record',
        )
    else:
        save_path = '../record/' + args.save_folder_name
        os.mkdir(save_path)

    args.save_path = save_path

    torch.save(args.__dict__, save_path + '/info.pickle')

    ### set the random seed
    fix_random(int(args.random_seed))

    ## set device
    device = torch.device(
        (
            f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
            # since DataParallel only allow .to("cuda")
        ) if torch.cuda.is_available() else "cpu"
    )
    
    ''' 
    2. get bd_test_dataset, train_dataset and model
    '''
    args,\
    clean_train_dataset_with_transform_for_train, bd_train_dataset_with_transform_for_train,\
    clean_train_dataset_with_transform, clean_test_dataset_with_transform,\
    bd_train_dataset_with_transform, bd_test_dataset_with_transform \
    = get_dataset(args)
    
    original_net = get_surrogate_model(args)
    
    clean_dataset, bd_dataset, cv_dataset,\
    clean_indices, bd_indices, cv_indices = \
        get_attack_datasets(bd_train_dataset_with_transform,
                            clean_train_dataset_with_transform,
                            args,
                            )
        
        
    '''
    3. test model performence 
    '''
    
    clean_test_dl = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    bd_test_dl = DataLoader(bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
    
    if not args.__contains__('surrogate_model_folder') or not args.surrogate_model_folder:
        '''
        If the surrogate model is just pretrained but not loaded from surrogate_model_folder, first fine-tune it with clean samples.
        '''
        pre_ft_indices = clean_indices
        pre_ft_dataset = Subset(clean_train_dataset_with_transform, pre_ft_indices)
        clean_train_dl = DataLoader(pre_ft_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                pin_memory=args.pin_memory, num_workers=args.num_workers, )
        
        logging.info(f'First fine-tune the pretrained extractor with clean dataset (size {len(pre_ft_dataset)}).')
        
        original_net = finetune(
            net=original_net,
            fine_tuning_dl=clean_train_dl,
            clean_test_dl=clean_test_dl,
            bd_test_dl=bd_test_dl,
            device=device,
            args=args,
        )
    
    _, clean_test_m, bd_test_m = \
        model_test(train_dl=None,
                   clean_test_dl=clean_test_dl,
                   bd_test_dl=bd_test_dl,
                   model=original_net,
                   args=args)
            
    logging.info(clean_test_m)
    test_acc = clean_test_m['test_correct']/clean_test_m['test_total']
    logging.info(f'test clean accuracy is {test_acc}')
    
    logging.info(bd_test_m)
    asr = bd_test_m['test_correct']/bd_test_m['test_total']
    logging.info(f'test asr is {asr}')

    '''
    4. pre-fine-tune the model to inject backdoor 
    '''
    
    logging.info('Pre-fine-tune to inject backdoors')
    logging.info('start generate adv dataset for train and test')
        
    s_test_dataset = bd_dataset
                
    logging.info(f'backdoor samples: {len(bd_indices)}, cover samples: {len(cv_indices)}')
    
    logging.info('Backdoor injected through fine-tuning')

    '''
    Iterations of influence based adversarial perturbation
    '''
    
    for epoch in range(args.ap_epochs):
        logging.info(f'Epoch {epoch} of influence based adversarial perturbation:')
        
        '''
        5. fine-tune the clean net
        '''
        pre_ft_indices = np.concatenate((bd_indices, cv_indices, clean_indices), axis = 0)
        pre_ft_dataset = Subset(bd_train_dataset_with_transform, pre_ft_indices)
        pre_fine_tuning_dl = DataLoader(pre_ft_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                    pin_memory=args.pin_memory, num_workers=args.num_workers, )
        z_dl = DataLoader(pre_ft_dataset, batch_size=8, shuffle=True, drop_last=False,
                    pin_memory=args.pin_memory, num_workers=args.num_workers, )
        logging.info(f'The fine-tune dataset size is {len(pre_ft_dataset)}')
        
        net = deepcopy(original_net)
        net = finetune(
            net=net,
            fine_tuning_dl=pre_fine_tuning_dl,
            clean_test_dl=clean_test_dl,
            bd_test_dl=bd_test_dl,
            device=device,
            args=args,
            ft_layers=3
        )
        
        ''' 
        6. pre-compute the s_test_vec 
        '''
        logging.info('Pre-compute hvp s_test:')
                
        s_test_dl = DataLoader(s_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                    pin_memory=args.pin_memory, num_workers=args.num_workers, )
        
        logging.info(f'First get the s_test_vec with {len(s_test_dataset)} datapoints for averaging.')
                
        logging.info(f'Then get {len(clean_dataset)} datapoints as a sampling for the whole training dataset.')
        
        ### pre-compute the s_test_vec
        s_test_avg = avg_s_test(test_loader=s_test_dl, 
                model=net, 
                z_loader=z_dl, # pre_fine_dataset servers as a sampling of the whole dataset
                device=device, 
                damp=args.damp, 
                scale=args.scale,
                recursion_depth=args.recursion_depth,
                original_label=False,
                all_param=False)
        
        
        
        '''
        7. calculate the influence of cover samples before perturbation
        '''
        influence_sum =\
            calc_influence(net,
                    s_test_avg, 
                    cv_dataset,
                    device,
                    train_dataset_size=len(pre_fine_tuning_dl.dataset)) 
            # estimate the train_dataset_size, since we assume the attacker only has a subset of dataset for attack
        logging.info(f'Before perturbation, the total influence of cover samples are {sum(influence_sum)}')

        ''' 
        8. craft adversarial samples by PGD 
        '''
        
        perturbed_img, targets = craft_adv_samples(net=net,
                                                #poison_net=poison_net,
                                                s_test_avg=s_test_avg,
                                                cv_dataset=cv_dataset,
                                                args=args,
                                                device=device)
                                                
        ''' reset the cover sample '''
        bd_train_dataset_with_transform.reset_cv_samples(cv_indices, perturbed_img, targets)
        cv_dataset = Subset(bd_train_dataset_with_transform, cv_indices)
        
        '''
        9. calculate the influence of cover samples after perturbation
        '''
        
        influence_sum =\
            calc_influence(net,
                    s_test_avg, 
                    cv_dataset,
                    device,
                    train_dataset_size=len(pre_fine_tuning_dl.dataset)) 
            # estimate the train_dataset_size, since we assume the attacker only has a subset of dataset for attack
        logging.info(f'After perturbation, the total influence of cover samples are {sum(influence_sum)}')

    '''
    10. save new perturbed samples
    '''
    save_dict = {
            'num_classes': args.num_classes,
            'data_path': args.data_path,
            'img_size': (args.input_height, args.input_width, args.input_channel),
            'clean_data': args.clean_data,
            'bd_train': bd_train_dataset_with_transform_for_train.retrieve_state(),
            'bd_test': bd_test_dataset_with_transform.retrieve_state(),
            'cv_pert': bd_train_dataset_with_transform.cv_pert_data_container.retrieve_state()
    }

    logging.info(f"saving...")
    # logging.debug(f"location : {save_path}/attack_result.pt") #, content summary :{pformat(summary_dict(save_dict))}")

    torch.save(
        save_dict,
        f'{args.dataset_folder}/pert_result.pt',
    )

    logging.info("Saved, folder path: {}".format(args.dataset_folder))

if __name__ == '__main__':
    main()