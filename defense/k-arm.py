'''
Reference: https://github.com/PurduePAML/K-ARM_Backdoor_Optimization/tree/main
@article{shen2021backdoor,
  title={Backdoor Scanning for Deep Neural Networks through K-Arm Optimization},
  author={Shen, Guangyu and Liu, Yingqi and Tao, Guanhong and An, Shengwei and Xu, Qiuling and Cheng, Siyuan and Ma, Shiqing and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2102.05123},
  year={2021}
}
  
Note: This defense cannot provide the anomaly indexes. Instead, it directly compose the potential backdoor between 2 classes.
The effectiveness highly depends on the hyper-parameters, and in some situations, the scanner may fail to reconstruct the pattern & mask.
Furthermore, K-Arm method does not provide anomaly indexes. It only points out potentially backdoored classes instead.
'''

import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torchvision

import cv2

sys.path.append('../')
sys.path.append(os.getcwd())

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
from matplotlib import image as mlt
from PIL import Image

from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, xy_iter


# utils
##############################################################################################################################
### utils                                                                                                                  ###
##############################################################################################################################

def classes_matching(target_classes_all,triggered_classes_all):
    start_index = len(target_classes_all)
    for i in range(len(triggered_classes_all)):
        tmp = triggered_classes_all[i]
        for sss in range(tmp.size(0)):
            target_classes_all.append(target_classes_all[i])
            triggered_classes_all.append(tmp[sss])



    end_index = len(target_classes_all)



    if start_index != end_index:
        target_classes_all = target_classes_all[start_index:]
        triggered_classes_all = triggered_classes_all[start_index:]


    return target_classes_all, triggered_classes_all


def identify_trigger_type(raw_target_classes,raw_victim_classes):
    if raw_victim_classes is not None:
        target_classes, victim_classes = classes_matching(raw_target_classes,raw_victim_classes)
        num_classes = len(victim_classes)
        trigger_type = 'polygon_specific'

        print(f'Trigger Type: {trigger_type}')
        Candidates = []
        for i in range(len(target_classes)):
            Candidates.append('{}-{}'.format(target_classes[i],victim_classes[i]))
        print(f'Target-Victim Pair Candidates: {Candidates}')

    else:
        #print(raw_target_classes)
        if raw_target_classes is not None:
            num_classes = 1 
            target_classes = raw_target_classes.unsqueeze(0)
            victim_classes = raw_victim_classes
            trigger_type = 'polygon_global'
            print(f'Trigger Type: {trigger_type}')
            print(f'Target class: {target_classes.item()} Victim Classes: ALL')
        
        else:
            target_classes = raw_target_classes
            victim_classes = raw_victim_classes
            num_classes = 0
            trigger_type = 'benign'


    return target_classes,victim_classes,num_classes,trigger_type
    

def trojan_det(args,trigger_type,l1_norm,sym_l1_norm):


    if trigger_type == 'polygon_global':
        
        if l1_norm < args.global_det_bound:
            return 'trojan'

        else:
            return 'benign'

    elif trigger_type == 'polygon_specific':
        

        if l1_norm < args.local_det_bound:

            if args.sym_check and sym_l1_norm / l1_norm > args.ratio_det_bound:
                return 'trojan'
            

            else:

                return 'benign'
        
        else:
            return 'benign'


def CustomDataSet(dataset,triggered_classes=[],label_specific=False):
    if not label_specific:
        return dataset
    else:
        triggered_classes_index = [i for i in range(len(dataset)) if dataset[i][1] in triggered_classes]
        return Subset(dataset, triggered_classes_index)

# Arm pre-screening
##############################################################################################################################
### Pre_Screening function will scan potential global trigger and label specific trigger in order.                         ###
### If a potential global trigger is found, it will return the target label and stop scanning label specific trigger       ###
### If a potential label specific trigger is found, it will return the target-victim label pair                            ###
##############################################################################################################################

def Pre_Screening(model,data_loader,args,device): # data_loader is a clean sub-dataset for scanner 
    for idx, (img,label,*other_info) in enumerate(data_loader):
        img,label = img.to(device),label.to(device)
        #img = img[:,permute,:,:]
        output = model(img)
        logits = F.softmax(output,dim=1)
        if idx == 0:
            logits_all = logits.detach().cpu()
        else:
            logits_all = torch.cat((logits_all,logits.detach().cpu()),dim=0)

    if args.num_classes <= 8:
        k = 2
    else:
        k = round(args.num_classes * args.gamma)

    topk_index = torch.topk(logits_all,k,dim=1)[1]
    topk_logit = torch.topk(logits_all,k,dim=1)[0]


    # step 1: check all label trigger
    target_label = all_label_trigger_det(args,topk_index)

    if target_label != -1:
        return target_label,None
    else:
        target_matrix,median_matrix = specific_label_trigger_det(args,topk_index,topk_logit)
        target_class_all = []
        triggered_classes_all = []
        for i in range(target_matrix.size(0)):
            if target_matrix[i].max() > 0:
                target_class = i
                triggered_classes = (target_matrix[i]).nonzero().view(-1)
                triggered_classes_logits = target_matrix[i][target_matrix[i]>0]
                triggered_classes_medians = median_matrix[i][target_matrix[i]>0]
             
 
                top_index_logit = (triggered_classes_logits > 1e-08).nonzero()[:,0]
                top_index_median = (triggered_classes_medians > 1e-08).nonzero()[:,0]
                
                top_index = torch.LongTensor(np.intersect1d(top_index_logit, top_index_median))


                if len(top_index) > 0:
                    triggered_classes = triggered_classes[top_index]

                    triggered_classes_logits = triggered_classes_logits[top_index]
 
                    if triggered_classes.size(0) > 3:
                        top_3_index = torch.topk(triggered_classes_logits,3,dim=0)[1]
                        triggered_classes = triggered_classes[top_3_index]
                    
                    target_class_all.append(target_class)
                    triggered_classes_all.append(triggered_classes)
            
            
        if len(target_class_all) == 0:
            target_class_all = None
                
        if len(triggered_classes_all) == 0:
            triggered_classes_all = None
            
        return target_class_all, triggered_classes_all



def all_label_trigger_det(args,topk_index):

    target_label = -1
    count_all = torch.zeros(args.num_classes)
    for i in range(args.num_classes):
        count_all[i] = topk_index[topk_index == i].size(0)
    max_count = torch.max(count_all)
    max_index = torch.argmax(count_all)
    if max_count > args.global_theta * topk_index.size(0):
        target_label = max_index
    return target_label



def specific_label_trigger_det(args,topk_index,topk_logit):
    sum_mat = torch.zeros(args.num_classes,args.num_classes)
    median_mat = torch.zeros(args.num_classes,args.num_classes)


    for i in range(args.num_classes):  
        #for each class, find the index of samples belongs to that class tmp_1 => index, tmp_1_logit => corresponding logit
        tmp_1 = topk_index[topk_index[:,0] == i]

        tmp_1_logit = topk_logit[topk_index[:,0] == i]
        tmp_2 = torch.zeros(args.num_classes)
        for j in range(args.num_classes):
            # for every other class, 
            if j == i:
                tmp_2[j] = -1
            else:
                tmp_2[j] = tmp_1[tmp_1 == j].size(0) / tmp_1.size(0)

                #if tmp_2[j]  == 1:
                if tmp_2[j]  >= args.local_theta:
                    
                    sum_var =  tmp_1_logit[tmp_1 == j].sum()
                    median_var = torch.median(tmp_1_logit[tmp_1 == j])
                    sum_mat[j,i] = sum_var
                    median_mat[j,i] = median_var
                    #print('Potential Target:{0}, Potential Victim:{1}, Ratio:{2}, Logits Sum:{3}, Logits Median:{4}'.format(j,i,tmp_2[j],sum_var,median_var))
                    #print('Potential victim: '+ str(i) + ' Potential target:' + str(j) + ' Ratio: ' + str(tmp_2[j]) + ' Logits Mean: '+ str(mean_var) + ' Logits std: ' + str(std_var) + 'Logit Median: ' + str(median_var))
    return sum_mat, median_mat




# K-Arm Scanner
##############################################################################################################################
### The K-Arm Scanner Class                                                                                                ###
##############################################################################################################################
class K_Arm_Scanner:
    def __init__ (self,model,args):
        self.model = model
        self.regularization = args.regularization
        self.init_cost = [args.init_cost] * args.num_classes
        self.steps = args.step
        self.round = args.rounds
        self.lr = args.lr
        self.num_classes = args.num_classes
        self.attack_succ_threshold = args.attack_succ_threshold
        self.patience = args.patience
        self.channels = args.input_channel
        self.batch_size = args.batch_size
        self.mask_size = [1,args.input_width,args.input_height]

        self.single_color_opt = args.single_color_opt

        self.pattern_size = [1,args.input_channel,args.input_width,args.input_height]

        
        #K-arms bandits
        if "," in args.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            args.device = f'cuda:{model.device_ids[0]}'
            self.device=args.device
        else:
            self.device=args.device
        self.epsilon = args.epsilon
        self.epsilon_for_bandits = args.epsilon_for_bandits
        self.beta = args.beta
        self.warmup_rounds = args.warmup_rounds

        self.cost_multiplier = args.cost_multiplier 
        self.cost_multiplier_up = args.cost_multiplier
        self.cost_multiplier_down = args.cost_multiplier

        self.early_stop = args.early_stop
        self.early_stop_threshold = args.early_stop_threshold
        self.early_stop_patience = args.early_stop_patience
        self.reset_cost_to_zero = args.reset_cost_to_zero


        self.mask_tanh_tensor = [torch.zeros(self.mask_size).to(self.device)] * self.num_classes 

        self.pattern_tanh_tensor = [torch.zeros(self.pattern_size).to(self.device)] * self.num_classes

        self.pattern_raw_tensor = []
        self.mask_tensor = []
        for i in range(self.num_classes):
            self.pattern_raw_tensor.append(torch.tanh(self.pattern_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)
            self.mask_tensor.append(torch.tanh(self.mask_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)


        
    def reset_state(self,pattern_init,mask_init):
        if self.reset_cost_to_zero:
            self.cost = [0] * self.num_classes
        else:
            self.cost = self.init_cost
        self.cost_tensor = self.cost


        mask_np = mask_init.cpu().numpy()
        mask_tanh = np.arctanh((mask_np - 0.5) * (2-self.epsilon))
        mask_tanh = torch.from_numpy(mask_tanh).to(self.device)

        pattern_np = pattern_init.cpu().numpy()
        pattern_tanh = np.arctanh((pattern_np - 0.5) * (2 - self.epsilon))
        pattern_tanh = torch.from_numpy(pattern_tanh).to(self.device)

        for i in range(self.num_classes):
            self.mask_tanh_tensor[i] = mask_tanh.clone()
            self.pattern_tanh_tensor[i] = pattern_tanh.clone()
            self.mask_tanh_tensor[i].requires_grad = True
            self.pattern_tanh_tensor[i].requires_grad = True


    def update_tensor(self,mask_tanh_tensor,pattern_tanh_tensor,y_target_index,first=False):


        if first is True:
            for i in range(self.num_classes):
                self.mask_tensor[i] = (torch.tanh(mask_tanh_tensor[i]) / (2 - self.epsilon) + 0.5)
                self.pattern_raw_tensor[i] = (torch.tanh(pattern_tanh_tensor[i]) / ( 2 - self.epsilon) + 0.5)
        
        else:

            self.mask_tensor[y_target_index] = (torch.tanh(mask_tanh_tensor[y_target_index]) / (2 - self.epsilon) + 0.5)
            self.pattern_raw_tensor[y_target_index] = (torch.tanh(pattern_tanh_tensor[y_target_index]) / (2 - self.epsilon) + 0.5)


    def scanning(self,target_classes_all,data_loader_arr,y_target_index,pattern_init,mask_init,trigger_type,direction):
        self.reset_state(pattern_init,mask_init)
        # for TrojAI round1, the data format is BGR, then need permute
        #permute = [2,1,0]

        self.update_tensor(self.mask_tanh_tensor,self.pattern_tanh_tensor,y_target_index,True)


        #K-arms bandits version
        best_mask = [None] * self.num_classes
        best_pattern = [None] * self.num_classes
        best_reg = [1e+10] * self.num_classes

        best_acc = [0] * self.num_classes



        log = []
        cost_set_counter = [0] * self.num_classes
        cost_down_counter = [0] * self.num_classes
        cost_up_counter = [0] * self.num_classes
        cost_up_flag = [False] * self.num_classes
        cost_down_flag = [False] * self.num_classes
        early_stop_counter = [0] * self.num_classes
        early_stop_reg_best = [1e+10] * self.num_classes
        early_stop_tag = [False] * self.num_classes
        update = [False] * self.num_classes


        avg_loss_ce = [1e+10] * self.num_classes
        avg_loss_reg = [1e+10] * self.num_classes
        avg_loss = [1e+10] * self.num_classes
        avg_loss_acc = [1e+10] * self.num_classes
        reg_down_vel = [-1e+10] * self.num_classes
        times = [0] * self.num_classes
        total_times = [0] * self.num_classes
        first_best_reg = [1e+10] * self.num_classes
                
        y_target_tensor = torch.Tensor([target_classes_all[y_target_index]]).long().to(self.device)
        
        optimizer_list = []


        
        for i in range(self.num_classes):
            optimizer = optim.Adam([self.pattern_tanh_tensor[i],self.mask_tanh_tensor[i]],lr=self.lr,betas=(0.5,0.9))
            optimizer_list.append(optimizer)

        criterion = nn.CrossEntropyLoss()

        pbar = tqdm(range(self.steps))

        #for step in range(self.steps):
        for step in range(self.steps):

            y_target_tensor = torch.Tensor([target_classes_all[y_target_index]]).long().to(self.device)
            total_times[y_target_index] += 1


            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for idx, (img,label,*other_info) in enumerate(data_loader_arr[y_target_index]):
                img = img.to(self.device)
                Y_target = y_target_tensor.repeat(img.size()[0])

                X_adv_tensor = (1-self.mask_tensor[y_target_index]) * img + self.mask_tensor[y_target_index] * self.pattern_raw_tensor[y_target_index]


                optimizer_list[y_target_index].zero_grad()

                output_tensor = self.model(X_adv_tensor)

                pred = output_tensor.argmax(dim=1, keepdim=True)


                self.loss_acc = pred.eq(Y_target.long().view_as(pred)).sum().item() / (img.size()[0])
                self.loss_ce = criterion(output_tensor,Y_target)

                self.loss_reg = torch.sum(torch.abs(self.mask_tensor[y_target_index] ))


                self.loss = self.loss_ce + self.loss_reg * self.cost_tensor[y_target_index] 


                self.loss.backward()

                optimizer_list[y_target_index].step()
                self.update_tensor(self.mask_tanh_tensor,self.pattern_tanh_tensor,y_target_index)


                pbar.set_description('Target: {}, victim: {}, Loss: {:.4f}, Acc: {:.2f}%, CE_Loss: {:.2f}, Reg_Loss:{:.2f}, Cost:{:.2f} best_reg:{:.2f} avg_loss_reg:{:.2f}'.format(
                target_classes_all[y_target_index], label[0], self.loss,self.loss_acc * 100,self.loss_ce,self.loss_reg,self.cost_tensor[y_target_index],best_reg[y_target_index],avg_loss_reg[y_target_index]))
                loss_ce_list.append(self.loss_ce.item())
                loss_reg_list.append(self.loss_reg.item())
                loss_list.append(self.loss.item())
                loss_acc_list.append(self.loss_acc)

            
            #K-arms Bandits
            avg_loss_ce[y_target_index] = np.mean(loss_ce_list)
            avg_loss_reg[y_target_index]= np.mean(loss_reg_list)
            avg_loss[y_target_index] = np.mean(loss_list)
            avg_loss_acc[y_target_index] = np.mean(loss_acc_list)
            
            if avg_loss_acc[y_target_index] > best_acc[y_target_index]:
                best_acc[y_target_index] = avg_loss_acc[y_target_index]
            
            
            if direction == 'forward':
        
                if (total_times[y_target_index] > 20 and best_acc[y_target_index] < 0.3 and trigger_type == 'polygon_specific') or (total_times[y_target_index] > 200 and  best_acc[y_target_index] < 0.8 and trigger_type == 'polygon_specific') or (total_times[y_target_index] > 10 and  best_acc[y_target_index] == 0 and trigger_type == 'polygon_specific'):
            
                    early_stop_tag[y_target_index] = True
            
            elif direction == 'backward':
                if (total_times[y_target_index] > 200 and  best_acc[y_target_index] < 1 and trigger_type == 'polygon_specific'):
                    #for the backward check
                    early_stop_tag[y_target_index] = True

            update[y_target_index] = False
            if avg_loss_acc[y_target_index] >= self.attack_succ_threshold and avg_loss_reg[y_target_index] < best_reg[y_target_index]:
                best_mask[y_target_index] = self.mask_tensor[y_target_index]



                #print('best_mask update')
                update[y_target_index] = True
                times[y_target_index] += 1
                best_pattern[y_target_index] = self.pattern_raw_tensor[y_target_index]
                
                if times[y_target_index] == 1:
                    first_best_reg[y_target_index] = 2500
                    #self.cost_tensor[y_target_index] = 1e-3
                #reg_down_vel[y_target_index] = 1e+4 * (np.log10(first_best_reg[y_target_index]) - np.log10(avg_loss_reg[y_target_index])) / (times[y_target_index] + total_times[y_target_index] / 10)
                reg_down_vel[y_target_index] =  ((first_best_reg[y_target_index]) - (avg_loss_reg[y_target_index])) / (times[y_target_index] + (total_times[y_target_index] / 2))
                #print('best_reg:',best_reg[y_target_index])
                #print('avg_loss_reg:',avg_loss_reg[y_target_index])

                best_reg[y_target_index] = avg_loss_reg[y_target_index]

            if self.early_stop:

                if best_reg[y_target_index] < 1e+10:
                    if best_reg[y_target_index] >= self.early_stop_threshold * early_stop_reg_best[y_target_index]:
                        #print('best_reg:',best_reg[y_target_index])
                        #print('early_stop_best_reg:',early_stop_reg_best[y_target_index])
                        early_stop_counter[y_target_index] +=1
                    else:
                        early_stop_counter[y_target_index] = 0
                early_stop_reg_best[y_target_index] = min(best_reg[y_target_index],early_stop_reg_best[y_target_index])

                if (times[y_target_index] > self.round) or (cost_down_flag[y_target_index] and cost_up_flag[y_target_index] and  early_stop_counter[y_target_index] > self.early_stop_patience and trigger_type == 'polygon_global'):

                    if y_target_index  == torch.argmin(torch.Tensor(best_reg)):
                        print('early stop for all!')
                        break
                    else:
                        early_stop_tag[y_target_index] = True

                        if all(ele == True for ele in early_stop_tag):
                            break


            if early_stop_tag[y_target_index] == False:

                if self.cost[y_target_index] == 0 and avg_loss_acc[y_target_index] >= self.attack_succ_threshold:
                    cost_set_counter[y_target_index] += 1
                    if cost_set_counter[y_target_index] >= 2:
                        self.cost[y_target_index] = self.init_cost[y_target_index]
                        self.cost_tensor[y_target_index] =  self.cost[y_target_index]
                        cost_up_counter[y_target_index] = 0
                        cost_down_counter[y_target_index] = 0
                        cost_up_flag[y_target_index] = False
                        cost_down_flag[y_target_index] = False
                else:
                    cost_set_counter[y_target_index] = 0

                if avg_loss_acc[y_target_index] >= self.attack_succ_threshold:
                    cost_up_counter[y_target_index] += 1
                    cost_down_counter[y_target_index] = 0
                else:
                    cost_up_counter[y_target_index] = 0
                    cost_down_counter[y_target_index] += 1

                if cost_up_counter[y_target_index] >= self.patience:
                    cost_up_counter[y_target_index] = 0
                    self.cost[y_target_index] *= self.cost_multiplier_up
                    self.cost_tensor[y_target_index] =  self.cost[y_target_index]
                    cost_up_flag[y_target_index] = True
                elif cost_down_counter[y_target_index] >= self.patience:
                    cost_down_counter[y_target_index] = 0
                    self.cost[y_target_index] /= self.cost_multiplier_down
                    self.cost_tensor[y_target_index] =  self.cost[y_target_index]
                    cost_down_flag[y_target_index] = True
           


            tmp_tensor = torch.Tensor(early_stop_tag)
            index = (tmp_tensor == False).nonzero()[:,0]
            time_tensor = torch.Tensor(times)[index]
            #print(time_tensor)
            non_early_stop_index = index
            non_opt_index = (time_tensor == 0).nonzero()[:,0]

            if early_stop_tag[y_target_index] == True and len(non_opt_index) != 0:
                for i in range(len(times)):
                    if times[i] == 0 and early_stop_tag[i] == False:
                        y_target_index = i
                        break

            elif len(non_opt_index) == 0 and early_stop_tag[y_target_index] == True:


                if len(non_early_stop_index)!= 0:
                    y_target_index = non_early_stop_index[torch.randint(0,len(non_early_stop_index),(1,)).item()]
                else:
                    break
            else: 
                if update[y_target_index] and times[y_target_index] >= self.warmup_rounds and all(ele >= self.warmup_rounds for ele in time_tensor):
                    self.early_stop = True
                    select_label = torch.max(torch.Tensor(reg_down_vel) + self.beta / torch.Tensor(avg_loss_reg),0)[1].item()
                    
                    random_value = torch.rand(1).item()


                    if random_value < self.epsilon_for_bandits:

                        non_early_stop_index = (torch.Tensor(early_stop_tag) != True).nonzero()[:,0]
                        
                        
                        if len(non_early_stop_index) > 1:
                            y_target_index = non_early_stop_index[torch.randint(0,len(non_early_stop_index),(1,)).item()]


                    else:
                        y_target_index = select_label

                elif times[y_target_index] < self.warmup_rounds  or update[y_target_index] == False:
                    continue

                else:
                    y_target_index = np.where(np.array(best_reg) == 1e+10)[0][0]


            #print('L1 of best_mask for each label:',best_reg)
            #print('L1 down speed:',reg_down_vel)
            #print('second loss item:',(1e+4 / torch.Tensor(avg_loss_reg)))
            #print('-----')
        return best_pattern, best_mask,best_reg,total_times
    
    
    
    
# K-Arm Optimization
########################################################################################################################################
### K_Arm_Opt functions load data based on different trigger types, then create an instance of K-Arm scanner and run optimization    ###
### It returns the target-victim pair and corresponding pattern, mask and l1 norm of the mask                                        ###
########################################################################################################################################
def K_Arm_Opt(args,target_classes_all,triggered_classes_all,trigger_type,model,dataset,direction,device):

    transform = transforms.Compose([
        transforms.CenterCrop(args.input_width),
        transforms.ToTensor()
        ])

    data_loader_arr = []
    if triggered_classes_all is None:

        data_set = CustomDataSet(dataset,triggered_classes=triggered_classes_all)
        data_loader = DataLoader(dataset=data_set,batch_size = args.batch_size,shuffle=False,drop_last=False,num_workers=8,pin_memory=True)
        data_loader_arr.append(data_loader)
    
    else:
        for i in range(len(target_classes_all)):
            data_set = CustomDataSet(dataset,triggered_classes=triggered_classes_all[i],label_specific=True)
            data_loader = DataLoader(dataset=data_set,batch_size = args.batch_size,shuffle=False,drop_last=False,num_workers=8,pin_memory=True)
            data_loader_arr.append(data_loader)


    k_arm_scanner = K_Arm_Scanner(model,args)


    if args.single_color_opt == True and trigger_type == 'polygon_specific':
        pattern = torch.rand(1,args.input_channel,1,1).to(device)
    
    else:
        pattern = torch.rand(1,args.input_channel,args.input_width,args.input_height).to(device)
        #only for r1
        #pattern = torch.rand(1,args.input_channel,1,1).to(device)
    pattern = torch.clamp(pattern,min=0,max=1)



    #K-arms Bandits
    if trigger_type == 'polygon_global':
        #args.early_stop_patience = 5

        mask = torch.rand(1,args.input_width,args.input_height).to(device)
        

    elif trigger_type == 'polygon_specific':

        if args.central_init:
            mask = torch.rand(1,args.input_width,args.input_height).to(device) * 0.001
            mask[:,112-25:112+25,112-25:112+25] = 0.99
        
        else:
            mask = torch.rand(1,args.input_width,args.input_height).to(device)

    
    mask = torch.clamp(mask,min=0,max=1)

    if args.num_classes == 1:
        start_label_index = 0
    else:
        #start_label_index = torch.randint(0,args.num_classes-1,(1,))[0].item()
        start_label_index = 0

    pattern, mask, l1_norm, total_times = k_arm_scanner.scanning(target_classes_all,data_loader_arr,start_label_index,pattern,mask,trigger_type,direction)
    
    index = torch.argmin(torch.Tensor(l1_norm))
    

    if triggered_classes_all is None:
        target_class =  target_classes_all[index]
        triggered_class = 'all'

    else:
        target_class = target_classes_all[index]
        triggered_class = triggered_classes_all[index]



    return l1_norm[index], mask[index], pattern[index], target_class, triggered_class,total_times[index]




# K_Arm_operator
########################################################################################################################################
### K_Arm_operator is wrapped to combine all sruffs defined above and operator the K-Arm scanning task.                              ###
########################################################################################################################################
class Recorder:
    def __init__(self, opt):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

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

        if self.pattern_best is None or self.mask_best is None:
            return  
        
        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(trigger, path_trigger, normalize=True)
        
        
        
class K_Arm_Operator(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.orig_num_classes = get_num_classes(args.dataset)
        args.num_classes = args.orig_num_classes
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

        #set the parameter for the nc defense
        parser.add_argument('--ratio', type=float,  help='ratio of training data')
        parser.add_argument('--cleaning_ratio', type=float,  help='ratio of cleaning data')
        parser.add_argument('--unlearning_ratio', type=float, help='ratio of unlearning data')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--step',type=int,default =1000)
        parser.add_argument('--rounds',type=int,default =60)
        parser.add_argument('--warmup_rounds',type=int,default=2)
        parser.add_argument('--init_cost',type=float,default=1e-03)
        parser.add_argument('--patience',type=int,default=5)
        parser.add_argument('--cost_multiplier',type=float,default=1.5)
        parser.add_argument('--epsilon',type=float,default=1e-07)
        parser.add_argument('--num_classes',type=int,default=0)
        parser.add_argument('--regularization',type=str,default='l1')
        parser.add_argument('--attack_succ_threshold',type=float,default=0.99)
        parser.add_argument('--early_stop',type=bool,default=False)
        parser.add_argument('--early_stop_threshold',type=float,default=1)
        parser.add_argument('--early_stop_patience',type=int,default= 10)
        parser.add_argument('--epsilon_for_bandits',type=float,default = 0.3)
        parser.add_argument('--reset_cost_to_zero',type=bool,default=True)
        parser.add_argument('--single_color_opt',type=bool,default=True) 
        parser.add_argument('--gamma',type=float,default=0.4,help='gamma for pre-screening') 
        parser.add_argument('--beta',type=float,default=1e+4,help='beta in the objective function') 
        parser.add_argument('--global_theta',type=float,default=0.75,help='theta for global trigger pre-screening') 
        parser.add_argument('--local_theta',type=float,default=0.75,help='theta for label-specific trigger pre-screening') 
        parser.add_argument('--central_init',type=bool,default=True,help='strategy for initalization') 
        parser.add_argument('--sym_check',type=bool,default=True,help='If using sym check') 
        parser.add_argument('--global_det_bound',type=int,default=1720,help='global bound to decide whether the model is trojan or not') 
        parser.add_argument('--local_det_bound',type=int,default=1000,help='local bound to decide whether the model is trojan or not') 
        parser.add_argument('--ratio_det_bound',type=int,default=10,help='ratio bound to decide whether the model is trojan or not') 
        parser.add_argument('--only_scan', type=int, default=0, 
                             help='Perform defense(0) or just scan backdoor(1)')
        
    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        self.attack_file = attack_file
        save_path = 'record/' + result_file + '/defense/k-arm/'
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
            
        # a. Prepare model, optimizer, scheduler and datasets
        model = generate_cls_model(self.args.model,self.args.orig_num_classes)
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
        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = nn.CrossEntropyLoss()
        self.set_trainer(model)
        criterion = argparser_criterion(args)

        
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

        # b. do K-Arm Pre-Screeming
        raw_target_classes, raw_victim_classes =  Pre_Screening(model, trainloader, args, self.device)
        target_classes,victim_classes,num_classes,trigger_type = identify_trigger_type(raw_target_classes,raw_victim_classes)
        args.num_classes = num_classes
        
        if trigger_type == 'benign': # pre-processing: benign
            print('Model is Benign')
            trojan = 'benign'
            l1_norm = None 
            sym_l1_norm = None 

        else:
            logging.info('The model is potentially backdoored')
            logging.info('='*40 + ' K-ARM Optimization ' + '='*40)
            
            # c. For potentially backdoored model, do K-Arm Optimizations
            l1_norm,mask,pattern,target_class,victim_class,opt_times = K_Arm_Opt(args,target_classes,victim_classes,trigger_type,model,data_set_o,'forward',args.device)
            
            # Use recorder to save mask and pattern
            recorder = Recorder(args)
            recorder.mask_best = mask
            recorder.pattern_best = pattern
            recorder.reg_best = l1_norm
            args.target_label = target_class
            recorder.save_result_to_dir(args)
            
            logging.info(f'Target Class: {target_class} Victim Class: {victim_class} Trigger Size: {l1_norm} Optimization Steps: {opt_times}')
            if args.sym_check and trigger_type == 'polygon_specific':
                args.step = opt_times
                args.num_classes = 1
                tmp = target_class
                sym_target_class = [victim_class.item()]
                sym_victim_class = torch.IntTensor([tmp])

                print('='*40 + ' Symmetric Check ' + '='*40)
                sym_l1_norm,*others = K_Arm_Opt(args,sym_target_class,sym_victim_class,trigger_type,model,data_set_o,'backward',args.device)
            else:
                sym_l1_norm = None 
            
            trojan = trojan_det(args,trigger_type,l1_norm,sym_l1_norm)

        if trojan == 'benign':
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
            
            model = generate_cls_model(self.args.model,self.args.orig_num_classes)
            model.load_state_dict(self.result['model'])
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
                save_prefix = 'nc',

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
        model = generate_cls_model(args.model,args.orig_num_classes)
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
        
        # In K-Arm scanning algorithm, there is always only one target class
        if victim_class != 'all':
            raise NotImplementedError # Fine-Tuning for specific backdoor attack is not implemented yet.
            # To implement this, should make sure the fine-tuning dataset is only composed of the images from the victim class. 
        
        flag_list = [target_class] # To fit for scanner code structure, yet there is always only one target class.
        for (label) in flag_list:
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
            save_prefix='nc',
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    K_Arm_Operator.add_arguments(parser)
    args = parser.parse_args()
    K_Arm_method = K_Arm_Operator(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'one_epochs_debug_badnet_attack'
    elif args.result_file is None:
        args.result_file = 'one_epochs_debug_badnet_attack'
    result = K_Arm_method.defense(args.result_file)
