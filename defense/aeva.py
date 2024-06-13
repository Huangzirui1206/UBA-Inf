'''
Reference: https://github.com/JunfengGo/AEVA-Blackbox-Backdoor-Detection-main
Detecting Backdoors in Black-box Neural Networks via Adversarial Extreme Value Analysis (ICLR 2022)

Note: AEVA scanner only implements scanning functions and lacks further fine-tuning defense process.

Note: AEVA-Scanner return the norm and anomaly index of each class.
'''

import numpy as np
import os
import torch
import logging
import sys
import yaml
from pprint import  pformat
import time
import argparse
from tqdm import tqdm

sys.path.append('../')
sys.path.append(os.getcwd())
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, xy_iter



#######################################################################################
### Contents in GradEst.hsja.py                                                     ###
#######################################################################################

def hsja(model, 
    sample, 
    args,
    target_label = None,
    target_image = None,
    ):
    
    model = model.to(args.device)
    
    original_label = torch.argmax(model(sample.unsqueeze(0).to(args.device)),axis=1)
    params = {'clip_max': args.clip_max, 'clip_min': args.clip_min, 
                'shape': sample.shape,
                'original_label': original_label, 
                'target_label': target_label, 
                'target_image': target_image, 
                'constraint': args.constraint,
                'num_iterations': args.num_iterations, 
                'gamma': args.gamma, 
                'd': int(np.prod(sample.shape)), 
                'stepsize_search': args.stepsize_search,
                'max_num_evals': args.max_num_evals,
                'init_num_evals': args.init_num_evals,
                'verbose': args.verbose,
                'device': args.device,
                'threshold': args.threshold
                }


    # Set binary search threshold.
    if params['constraint'] == 'l2':
        params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])
    else:
        params['theta'] = params['gamma'] / (params['d'] ** 2)
        
    # Initialize.
    perturbed = initialize(model, sample, params)    

    # Project the initialization to the boundary.
    perturbed, dist_post_update = binary_search_batch(sample, 
        perturbed.unsqueeze(0), 
        model, 
        params)
    dist = compute_distance(perturbed, sample, args.constraint)

    for j in (range(params['num_iterations'])):
        params['cur_iter'] = j + 1

        # Choose delta.
        delta = select_delta(params, dist_post_update)

        # Choose number of evaluations.
        num_evals = int(params['init_num_evals'] * np.sqrt(j+1))
        num_evals = int(min([num_evals, params['max_num_evals']]))

        # approximate gradient.
        gradf = approximate_gradient(model, perturbed, num_evals, 
            delta, params)


        if params['constraint'] == 'linf':
            update = np.sign(gradf)
        else:
            update = gradf


        # search step size.
        if params['stepsize_search'] == 'geometric_progression':
            # find step size.
            epsilon = geometric_progression_for_stepsize(perturbed, 
                update, dist, model, params)
            # Update the sample. 
            perturbed = clip_image(perturbed + epsilon * update, 
                args.clip_min, args.clip_max)

            # Binary search to return to the boundary. 
            perturbed, dist_post_update = binary_search_batch(sample, 
                perturbed[None], model, params)

        elif params['stepsize_search'] == 'grid_search':
            # Grid search for stepsize.
            epsilons = np.logspace(-4, 0, num=20, endpoint = True) * dist
            epsilons_shape = [20] + len(params['shape']) * [1]
            perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
            perturbeds = clip_image(perturbeds, params['clip_min'], params['clip_max'])
            idx_perturbed = decision_function(model, perturbeds, params)

            if np.sum(idx_perturbed) > 0:
                # Select the perturbation that yields the minimum distance # after binary search.
                perturbed, dist_post_update = binary_search_batch(sample, 
                    perturbeds[idx_perturbed], model, params)

        # compute new distance.
        dist = compute_distance(perturbed, sample, args.constraint)

        if args.verbose:
            print('iteration: {:d}, distance {:.4E}\r'.format(j+1, dist), end='')
    return dist,perturbed

def decision_function(model, images, params):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    model.eval()
    images = clip_image(images, params['clip_min'], params['clip_max'])
    prob = model(images.to(params['device']))
    if params['target_label'] is None:

        # if len(prob.shape)>1:
        return torch.argmax(prob.cpu(), axis = 1) != params['original_label']
        # if len(prob.shape)==1:
        # return prob != params['original_label']

    else:
        return torch.argmax(prob.cpu(), axis = 1) == params['target_label']

def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image.cpu()), clip_max) 


def compute_distance(x_ori, x_pert, constraint = 'l2'):
    # Compute the distance between two images.
    if constraint == 'l2':
        return np.linalg.norm(x_ori - x_pert).astype(np.float32)
    elif constraint == 'linf':
        return np.max(abs(x_ori - x_pert)).astype(np.float32)


def approximate_gradient(model, sample, num_evals, delta, params):
    clip_max, clip_min = params['clip_max'], params['clip_min']

    # Generate random vectors.
    noise_shape = [num_evals] + list(params['shape'])
    if params['constraint'] == 'l2':
        rv = np.random.randn(*noise_shape)
    elif params['constraint'] == 'linf':
        rv = np.random.uniform(low = -1, high = 1, size = noise_shape)

    rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
    rv = torch.tensor(rv.astype(np.float32))
    
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(model, perturbed, params)
    decision_shape = [len(decisions)] + [1] * len(params['shape'])
    fval = 2 * decisions.reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if torch.mean(fval) == 1.0: # label changes. 
        gradf = torch.mean(rv, axis = 0)
    elif torch.mean(fval) == -1.0: # label not change.
        gradf = - torch.mean(rv, axis = 0)
    else:
        fval -= torch.mean(fval)
        gradf = torch.mean(fval * rv, axis = 0) 

    # Get the gradient direction.
    gradf = gradf / torch.linalg.norm(gradf)

    return gradf


def project(original_image, perturbed_images, alphas, params):
    
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)
    if params['constraint'] == 'l2':
        return (1-alphas) * original_image + alphas * perturbed_images
    elif params['constraint'] == 'linf':
        out_images = clip_image(
            perturbed_images, 
            original_image - alphas, 
            original_image + alphas
            )
        return out_images


def binary_search_batch(original_image, perturbed_images, model, params):
    """ Binary search to approach the boundar. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = torch.tensor([
            compute_distance(
                original_image, 
                perturbed_image, 
                params['constraint']
            ) 
            for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs based on constraint.
    if params['constraint'] == 'linf':
        highs = dists_post_update
        # Stopping criteria.
        thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
    else:
        highs = torch.ones(len(perturbed_images))
        thresholds = params['theta']

    lows = torch.zeros(len(perturbed_images))

    
    # Call recursive function. 
    while torch.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)

        # Update highs and lows based on model decisions.
        decisions = decision_function(model, mid_images, params)
        lows = torch.where(decisions == 0, mids, lows)
        highs = torch.where(decisions == 1, mids, highs)

    out_images = project(original_image, perturbed_images, highs, params)

    # Compute distance of the output image to select the best choice. 
    # (only used when stepsize_search is grid_search.)
    dists = torch.tensor([
        compute_distance(
            original_image, 
            out_image, 
            params['constraint']
        ) 
        for out_image in out_images])
    idx = torch.argmin(dists)

    dist = dists_post_update[idx]
    out_image = out_images[idx]
    return out_image, dist


def initialize(model, sample, params):
    """ 
    Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
    """
    success = 0
    num_evals = 0

    if params['target_image'] is None:
        # Find a misclassified random noise.
        while True:
            random_noise = torch.random.uniform(params['clip_min'], 
                params['clip_max'], size = params['shape'])
            success = decision_function(model,random_noise[None], params)[0]
            num_evals += 1
            if success:
                break
            assert num_evals < 1e4,"Initialization failed! "
            "Use a misclassified image as `target_image`" 


        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid) * sample + mid * random_noise 
            success = decision_function(model, blended, params)
            if success:
                high = mid
            else:
                low = mid

        initialization = (1 - high) * sample + high * random_noise 

    else:
        initialization = params['target_image']

    return initialization


def geometric_progression_for_stepsize(x, update, dist, model, params):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching 
    the desired side of the boundary,
    """
    epsilon = dist / np.sqrt(params['cur_iter']) 

    def phi(epsilon):
        new = x + epsilon * update
        success = decision_function(model, new[None], params)
        return success
    
    while phi(epsilon).sum() / len(x) > params['threshold']:
        epsilon /= 2.0

    return epsilon

def select_delta(params, dist_post_update):
    """ 
    Choose the delta at the scale of distance 
    between x and perturbed sample. 

    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist_post_update
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update    

    return delta



#######################################################################################
### Contents in GradEst.main.py                                                     ###
#######################################################################################

def attack(model,args,basic_imgs,target_imgs,target_labels):
    vec=np.empty((basic_imgs.shape[0]))
    vec_per=np.empty_like(basic_imgs)
        
    for i in range(len(basic_imgs)):

        sample=basic_imgs[i]

        #
        target_image =target_imgs[i]

        target_label=target_labels[i]

        logging.info('attacking the {}th sample...'.format(i))

        dist,per = hsja(model,
                        sample, 
                        args=args,
                        target_label = target_label,
                        target_image = target_image,
                            )


        vec[i]=torch.max(torch.abs(per - sample)) / dist
        logging.info(f'the {i}th sample\'s norm is {vec[i]}')
        vec_per[i]=per-sample

    assert vec.all()>=0
    return vec,vec_per


#######################################################################################
### Contents in outlier.py                                                          ###
### Take in the adv_per_img saved before and return the anomaly indexes             ###
### Use like: s=get_s(10)     analomy(array=s)                                      ###
#######################################################################################
def array_sort(array):

    a=np.empty(array.shape[0])

    for i in range(array.shape[0]):

        a[i]=np.sum(np.sort(np.sum(array[i],axis=-1).flatten())[-2:])/np.sum(array[i])
        #a[i] = np.linalg.norm(array[i]) / np.linalg.norm(array[i])

    return a


def get_s(num_labels, save_path):
    s=np.zeros(num_labels)
    coll=np.zeros((num_labels,num_labels-1))
    for i in range(0, num_labels):

        labels = list(range(num_labels))
        labels.remove(i)

        assert len(labels) == (num_labels - 1)

        for index, t in enumerate(labels):

            if os.path.exists(f"{save_path}/data_{t}_{i}.npy"):
                a=np.abs(np.load(f"{save_path}/data_{t}_{i}.npy"))
                #v=np.max(sort(a))
                v=np.max(array_sort(a))
                coll[i][index]=v
                
                s[i]+=v
    #print(s)
    return s


def analomy(array):

    consistency_constant = 1.4826  # if normal distribution
    median = np.median(array)
    mad = consistency_constant * np.median(np.abs(array - median))
    anomaly_indexes = (array - median) / mad
    return anomaly_indexes



#######################################################################################
### Contents in detect_main.py                                                      ###
#######################################################################################
class aeva():
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

        parser.add_argument('--only_scan', type=int, default=0)
        
        # add arguments for avea method
        parser.add_argument('--clip_max', type=float, default=1.0)
        parser.add_argument('--clip_min', type=int, default=0.0)
        parser.add_argument('--constraint', type=str, default='l2')
        parser.add_argument('--num_iterations', type=int, default=50)
        parser.add_argument('--gamma', type=float, default=1.0)
        parser.add_argument('--stepsize_search', type=str, default='geometric_progression')
        parser.add_argument('--max_num_evals', type=float, default=1e4)
        parser.add_argument('--init_num_evals', type=int, default=100)
        parser.add_argument('--verbose', type=bool, default=True)
        parser.add_argument('--threshold', type=float, default=0.95)
        parser.add_argument('--ratio', type=float,  help='ratio of training data',default=0.05)
        
        parser.add_argument('--bp', type=int, default=0)
        parser.add_argument('--ep', type=int, default=1)
    
    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        self.attack_file = attack_file
        save_path = 'record/' + result_file + '/defense/aeva/'
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
    
    # TODO: check the save path of perturbation results
    def get_specific_backdoor_vec(self,
                                  original_label,
                                  target_label,
                                  args):



        if os.path.exists(f"{self.args.save_path}/data_{str(original_label)}_{str(target_label)}.npy"):
            logging.info(f'Data for {original_label} -> {target_label} already exits.')
            pass


        else:

            x_o = self.x_val[self.y_val == original_label][0:2]

            x_t = self.x_val[self.y_val == target_label][0:2]

            y_t = self.y_val[self.y_val == target_label][0:2]

            _,per=attack(self.model, 
                         args=args,
                         basic_imgs=x_o,
                         target_imgs=x_t, 
                         target_labels=y_t)



            np.save(f"{self.args.save_path}/data_{str(original_label)}_{str(target_label)}.npy",
                    per)

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        # a. Prepare model and dataset
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
            
        self.model = model 

            
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

        # b. divide dataset into self.x_val and self.y_val that AEVA needs in get_specific_backdoor_vec
        # TODO: Get the x_val, y_val that aeva takes
        x_val, y_val = None, None
        for imgs, labels, *others in trainloader:
                            
            if x_val is None:
                x_val = imgs
            else:
                x_val = torch.concatenate((x_val, imgs), dim=0)
            
            if y_val is None:
                y_val = labels
            else:
                y_val = torch.concatenate((y_val, labels), dim=0)
                
        params = {'clip_max': args.clip_max, 'clip_min': args.clip_min,
                  'target_label':y_val,
                  'device':self.args.device}
        
        self.x_val = x_val[decision_function(self.model, x_val, params)]
        logging.info(f"Accuracy:{self.x_val.shape[0]/len(data_set_o)}")
        self.y_val = y_val[decision_function(self.model, x_val, params)]
            
        assert self.y_val.shape[0]==self.x_val.shape[0]
            
        del x_val
        del y_val
            
            # c. For each label pair, test whether there is a backdoor risk
        self.num_classes = self.args.num_classes
        for i in range(self.args.bp, self.args.ep):

            labels=list(range(self.num_classes))
            labels.remove(i)

            assert len(labels)==(self.num_classes-1)

            for t in labels:

                logging.info(f"original:{i}-> {t} \n")

                self.get_specific_backdoor_vec(original_label=i,
                                               target_label=t,
                                               args=self.args) 
                    
        # d. strike outlier functions to get the anomaly indexes
        s = get_s(self.num_classes, self.args.save_path)
        logging.info(f'GAP(Global Adversarial Peak):{s}')
        '''
        To remain consistent with other scanners, set the anamoly_indexes negative,
        which means the smaller the anomaly index is, the more probable that the class is a backdoor target.
        '''
        anomaly_indexes = analomy(array=s) 
        logging.info(f"Anomaly Index:{-anomaly_indexes}")
        
        def writeDataToCsv(path:str, data:list):
            import pandas as pd
            data = pd.DataFrame(data)
            data.to_csv(path)
        writeDataToCsv(os.path.join(args.save_path, 'anomaly_index.csv'), 
                        [
                            ['GAP'] + s.tolist(),
                            ['anomaly_index'] + anomaly_indexes.tolist()
                        ])
            
        # AEVA only provide scanning functions but with no further fine-tuning defense functions
        # So we don't save the model information, but only save the anomaly index information.
            
        # TODO: implement the anamoly indexes save process
        NotImplemented
            
    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    aeva.add_arguments(parser)
    args = parser.parse_args()
    aeva_method = aeva(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    result = aeva_method.defense(args.result_file)
            