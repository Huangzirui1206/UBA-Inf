import os.path
import sys, logging
from typing import Callable, Optional, Sequence, Union
sys.path.append('../')

import numpy as np
import torch
import random
import copy

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
from typing import *

from torchvision.transforms import ToPILImage
from torchvision.datasets import DatasetFolder, ImageFolder

class slice_iter(torch.utils.data.dataset.Dataset):
    '''iterate over a slice of the dataset'''
    def __init__(self,
             dataset,
             axis = 0
         ):
        self.data = dataset
        self.axis = axis

    def __getitem__(self, item):
        return self.data[item][self.axis]

    def __len__(self):
        return len(self.data)



class x_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
             dataset
         ):
        self.data = dataset

    def __getitem__(self, item):
        img = self.data[item][0]
        return img

    def __len__(self):
        return len(self.data)

class y_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
             dataset
         ):
        self.data = dataset

    def __getitem__(self, item):
        target = self.data[item][1]
        return target

    def __len__(self):
        return len(self.data)


def get_labels(given_dataset):
    if isinstance(given_dataset, DatasetFolder) or isinstance(given_dataset, ImageFolder):
        logging.debug("get .targets")
        return given_dataset.targets
    else:
        logging.debug("Not DatasetFolder or ImageFolder, so iter through")
        return [label for img, label, *other_info in given_dataset]

class dataset_wrapper_with_transform(torch.utils.data.Dataset):
    '''
    idea from https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python
    '''

    def __init__(self, obj, wrap_img_transform=None, wrap_label_transform=None):

        # this warpper should NEVER be warp twice.
        # Since the attr name may cause trouble.
        assert not "wrap_img_transform" in obj.__dict__
        assert not "wrap_label_transform" in obj.__dict__

        self.wrapped_dataset = obj
        self.wrap_img_transform = wrap_img_transform
        self.wrap_label_transform = wrap_label_transform

    def __getattr__(self, attr):
        # # https://github.com/python-babel/flask-babel/commit/8319a7f44f4a0b97298d20ad702f7618e6bdab6a
        # # https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
        # if attr == "__setstate__":
        #     raise AttributeError(attr)
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_dataset, attr)

    def __getitem__(self, index):
        img, label, *other_info = self.wrapped_dataset[index]
        if self.wrap_img_transform is not None:
            img = self.wrap_img_transform(img)
            # assert type(img) == torch.Tensor
            if other_info:
                other_info[-1] = self.wrap_img_transform(other_info[-1])
        if self.wrap_label_transform is not None:
            label = self.wrap_label_transform(label)
        return (img, label, *other_info)

    def __len__(self):
        return len(self.wrapped_dataset)
    
    def __deepcopy__(self, memo):
        # In copy.deepcopy, init() will not be called and some attr will not be initialized. 
        # The getattr will be infinitely called in deepcopy process.
        # So, we need to manually deepcopy the wrapped dataset or raise error when "__setstate__" us called. Here we choose the first solution.
        return dataset_wrapper_with_transform(copy.deepcopy(self.wrapped_dataset), copy.deepcopy(self.wrap_img_transform), copy.deepcopy(self.wrap_label_transform))


class poisonedCLSDataContainer:
    '''
    Two mode:
        in RAM / disk
        if in RAM
            save {key : value}
        elif in disk:
            save {
                key : {
                    "path":path, (must take a PIL image and save to .png)
                    "other_info": other_info, (Non-img)
                    }
            }
            where img, *other_info = value
    '''
    def __init__(self, save_folder_path=None, save_file_format = ".png"):
        self.save_folder_path = save_folder_path
        self.data_dict = {}
        self.save_file_format = save_file_format
        logging.info(f"save file format is {save_file_format}")

    def retrieve_state(self):
        return {
            "save_folder_path":self.save_folder_path,
            "data_dict":self.data_dict,
            "save_file_format":self.save_file_format,
        }

    def set_state(self, state_file):
        self.save_folder_path = state_file["save_folder_path"]
        self.data_dict = state_file["data_dict"]
        self.save_file_format = state_file["save_file_format"]

    def setitem(self, key, value, relative_loc_to_save_folder_name=None):

        if self.save_folder_path is None:
            self.data_dict[key] = value
        else:
            img, *other_info = value

            save_subfolder_path = f"{self.save_folder_path}/{relative_loc_to_save_folder_name}"
            if not (
                os.path.exists(save_subfolder_path)
                and
                os.path.isdir(save_subfolder_path)
            ):
                os.makedirs(save_subfolder_path)

            file_path = f"{save_subfolder_path}/{key}{self.save_file_format}"
            img.save(file_path)

            self.data_dict[key] = {
                    "path": file_path,
                    "other_info": other_info,
            }

    def __getitem__(self, key):
        if self.save_folder_path is None:
            return self.data_dict[key]
            '''try:
                return self.data_dict[key]
            except KeyError:
                if key not in self.data_dict:
                    if key >= len(self.data_dict):
                        for i in range(len(self.data_dict)):
                            if i not in self.data_dict:
                                raise ValueError('i not in data_dict')
                            else:
                                raise KeyError
                    else:
                        raise NotImplementedError'''
        else:
            file_path = self.data_dict[key]["path"]
            other_info = self.data_dict[key]["other_info"]
            img =  Image.open(file_path)
            return (img, *other_info)

    def __len__(self):
        return len(self.data_dict)


class prepro_cls_DatasetBD_v2(torch.utils.data.Dataset):

    def __init__(
            self,
            full_dataset_without_transform,
            poison_indicator: Optional[Sequence] = None,  # one-hot to determine which image may take bd_transform

            bd_image_pre_transform: Optional[Callable] = None,
            cv_image_pre_transform: Optional[Callable] = None,
            bd_label_pre_transform: Optional[Callable] = None,
            save_folder_path = None,

            mode = 'attack',
            
            adv_samples: Union[np.ndarray, None] = None,
            c_method : str = 'label-consistent',
            cv_label_pre_transform: Optional[Callable] = None,
            
            rotate_and_perturb_flag = False
        ):
        '''
        This class require poisonedCLSDataContainer

        :param full_dataset_without_transform: dataset without any transform. (just raw data)

        :param poison_indicator:
            array with 0 or 1 at each position corresponding to clean/poisoned
            Must have the same len as given full_dataset_without_transform (default None, regarded as all 0s)

        :param bd_image_pre_transform:
        :param bd_label_pre_transform:
        ( if your backdoor method is really complicated, then do not set these two params. These are for simplicity.
        You can inherit the class and rewrite method preprocess part as you like)

        :param save_folder_path:
            This is for the case to save the poisoned imgs on disk.
            (In case, your RAM may not be able to hold all poisoned imgs.)
            If you do not want this feature for small dataset, then just left it as default, None.\
        
        :param c_method: which cover method to use
        :param clabel: For single-label-consistent cover methods or const-label-flipping method
        '''

        self.dataset = full_dataset_without_transform
        
        self.c_method = c_method

        if poison_indicator is None:
            poison_indicator = np.zeros(len(full_dataset_without_transform))
        self.poison_indicator = poison_indicator

        assert len(full_dataset_without_transform) == len(poison_indicator)

        self.bd_image_pre_transform = bd_image_pre_transform
        self.cv_image_pre_transform = cv_image_pre_transform
        self.bd_label_pre_transform = bd_label_pre_transform
        self.cv_label_pre_transform = cv_label_pre_transform

        self.save_folder_path = os.path.abspath(save_folder_path) if save_folder_path is not None else save_folder_path # since when we want to save this dataset, this may cause problem

        # The cover samples are stored in cover_folder_path
        if save_folder_path is not None:
            cover_folder_path = save_folder_path.replace("bd", "cv") if "bd" in save_folder_path else ("cv_" + save_folder_path)
        else: 
            cover_folder_path = None
        self.save_cover_folder_path = os.path.abspath(cover_folder_path) if cover_folder_path is not None else cover_folder_path # since when we want to save this dataset, this may cause problem

        self.original_index_array = np.arange(len(full_dataset_without_transform))

        self.bd_data_container = poisonedCLSDataContainer(self.save_folder_path, ".png")
        self.cv_data_container = poisonedCLSDataContainer(self.save_cover_folder_path, ".png")
        self.cv_pert_data_container = None # Can't be used until reset cover samples
        self.original_data_container = None # Can't be used until set original_data_container, it can be adversarial samples for experiments.
        
        self.mode = mode
        
        self.adv_samples = adv_samples
        self.clean_adv_flag = False # Whether to use the whole adv dataset for clean sample removal in GBU
        
        if sum(self.poison_indicator) >= 1:
            self.prepro_backdoor()

        self.getitem_all = True
        self.getitem_all_switch = False
        
        
        ### For Incremental learning, use perturbation and rotation to imitate image from different domains
        self.rotate_and_perturb_flag = rotate_and_perturb_flag
        self.task_id = 0

    def prepro_backdoor(self):
        # get adv_samples
        for selected_index in tqdm(self.original_index_array, desc="prepro_backdoor"):
            if self.poison_indicator[selected_index] == 1:   # poison
                img, label = self.dataset[selected_index]
                img = self.bd_image_pre_transform(img, target=label, image_serial_id=selected_index)
                bd_label = self.bd_label_pre_transform(label)
                self.set_one_bd_sample(
                    selected_index, img, bd_label, label
                )
            elif self.poison_indicator[selected_index] == 2:   # cover
                img, label = self.dataset[selected_index]
                
                # adversarial samples replacement
                if self.c_method == 'adversarial-label-consistent' or\
                   self.c_method == 'adversarial-single-label-consistent':
                    if self.adv_samples is not None:
                        img = self.adv_samples[selected_index]
                    if isinstance(img, np.ndarray):
                        img = img.astype(np.uint8)
                    img = ToPILImage()(img)
                
                img = self.cv_image_pre_transform(img, target=label, image_serial_id=selected_index)
                cv_label = self.cv_label_pre_transform(label)
                self.set_one_cv_sample(
                    selected_index, img, cv_label, label
                )        
    
    def reset_cv_samples(self, cv_index, cv_samples:torch.tensor, cv_targets:torch.tensor, sub_folder_name='pert'):
        assert len(cv_index) == len(cv_samples)
        
        if self.cv_pert_data_container is None:
            save_path = self.cv_data_container.save_folder_path
            self.cv_pert_data_container = poisonedCLSDataContainer(os.path.join(save_path, sub_folder_name), ".png")
        
        for selected_index, adv_img, adv_label in zip(cv_index, cv_samples, cv_targets):
            assert self.poison_indicator[selected_index] == 2
            img, label = self.dataset[selected_index]
                
            img = adv_img
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
            img = ToPILImage()(img)
            
            cv_label = int(adv_label.item())
                     
            self.set_one_cv_sample(
                selected_index, img, cv_label, label,
                save_pert=True
            )        
    
    def reset_cv_samples_with_dataset(self, cv_index, dataset):
        assert len(self.dataset) == len(dataset)
        
        if self.cv_pert_data_container is None:
            save_path = self.cv_data_container.save_folder_path
            self.cv_pert_data_container = poisonedCLSDataContainer(os.path.join(save_path, 'pert'), ".png")
        
        for selected_index in cv_index:
            assert self.poison_indicator[selected_index] == 2
            img, label = self.dataset[selected_index]
            r_img, r_label = dataset[selected_index]
                                 
            self.set_one_cv_sample(
                selected_index, r_img, r_label, label,
                save_pert=True
            )   
            
    def set_one_bd_sample(self, selected_index, img, bd_label, label, save_pert=True):

        '''
        1. To pil image
        2. set the image to container
        3. change the poison_index.

        logic is that no matter by the prepro_backdoor or not, after we set the bd sample,
        This method will automatically change the poison index to 1.

        :param selected_index: The index of bd sample
        :param img: The converted img that want to put in the bd_container
        :param bd_label: The label bd_sample has
        :param label: The original label bd_sample has

        '''

        # we need to save the bd img, so we turn it into PIL
        if (not isinstance(img, Image.Image)) :
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
            img = ToPILImage()(img)
        self.bd_data_container.setitem(
            key=selected_index,
            value=(img, bd_label, label),
            relative_loc_to_save_folder_name=f"{label}",
        )
        self.poison_indicator[selected_index] = 1

    def set_one_cv_sample(self, selected_index, img, cv_label, label, save_pert=False):

        '''
        1. To pil image
        2. set the image to container
        3. change the poison_index.

        logic is that no matter by the prepro_backdoor or not, after we set the bd sample,
        This method will automatically change the poison index to 2.

        :param selected_index: The index of cover sample
        :param img: The converted img that want to put in the cv_data_container
        :param label: The original label cv_sample has

        '''

        # we need to save the bd img, so we turn it into PIL
        if (not isinstance(img, Image.Image)) :
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
            img = ToPILImage()(img)
        
        if not save_pert:
            self.cv_data_container.setitem(
                key=selected_index,
                value=(img, cv_label, label),
                relative_loc_to_save_folder_name=f"{label}",
            )
        else:
            assert self.cv_pert_data_container is not None
            self.cv_pert_data_container.setitem(
                key=selected_index,
                value=(img, cv_label, label),
                relative_loc_to_save_folder_name=f"{label}",
            )
            
        self.poison_indicator[selected_index] = 2

    def __len__(self):
        return len(self.original_index_array)

    def __getitem__(self, index):

        original_index = self.original_index_array[index]
        if self.poison_indicator[original_index] == 0:
            # clean
            img, label = self.dataset[original_index]
            if self.clean_adv_flag:
                img = self.adv_samples[original_index]
                if isinstance(img, np.ndarray):
                    img = img.astype(np.uint8)
                img = ToPILImage()(img)
            if self.rotate_and_perturb_flag:
                img = self.rotate_and_perturb_func(img, self.task_id + original_index)
            original_target = label
            poison_or_not = 0
            original_img = img
        elif self.poison_indicator[original_index] == 1:
            # bd
            img, label, original_target = self.bd_data_container[original_index]
            poison_or_not = 1
            original_img = img
        else:
            # cv
            if self.cv_pert_data_container is not None:
                img, label, original_target = self.cv_pert_data_container[original_index]
                original_img, _, _ = self.cv_data_container[original_index]
            else:
                img, label, original_target = self.cv_data_container[original_index]
                original_img = img
            poison_or_not = 1

        if not isinstance(img, Image.Image):
            img = ToPILImage()(img)
            
        if self.getitem_all:
            if self.getitem_all_switch:
                # this is for the case that you want original targets, but you do not want change your testing process
                return img, \
                       original_target, \
                       original_index, \
                       poison_or_not, \
                       label,\
                       original_img # For adversarial perturbation
            
            else: # here should corresponding to the order in the bd trainer
                return img, \
                       label, \
                       original_index, \
                       poison_or_not, \
                       original_target,\
                       original_img # For adversarial perturbation
        else:
            return img, label

    def subset(self, chosen_index_list):
        self.original_index_array = self.original_index_array[chosen_index_list]

    def retrieve_state(self):
        return {
            "bd_data_container" : self.bd_data_container.retrieve_state(),
            "cv_data_container" : self.cv_data_container.retrieve_state(),
            "getitem_all":self.getitem_all,
            "getitem_all_switch":self.getitem_all_switch,
            "original_index_array": self.original_index_array,
            "poison_indicator": self.poison_indicator,
            "save_folder_path": self.save_folder_path,
        }

    def copy(self):
        bd_train_dataset = prepro_cls_DatasetBD_v2(self.dataset)
        copy_state = copy.deepcopy(self.retrieve_state())
        bd_train_dataset.set_state(
            copy_state
        )
        return bd_train_dataset

    def set_state(self, state_file):
        self.bd_data_container = poisonedCLSDataContainer()
        self.bd_data_container.set_state(
            state_file['bd_data_container']
        )
        self.cv_data_container = poisonedCLSDataContainer()
        self.cv_data_container.set_state(
            state_file['cv_data_container']   
        )
        self.getitem_all = state_file['getitem_all']
        self.getitem_all_switch = state_file['getitem_all_switch']
        self.original_index_array = state_file["original_index_array"]
        self.poison_indicator = state_file["poison_indicator"]
        self.save_folder_path = state_file["save_folder_path"]
        
    def set_cv_pert_container(self, load_file):
        self.cv_pert_data_container = poisonedCLSDataContainer()
        self.cv_pert_data_container.set_state(
            load_file
        )
        
    def rotate_and_perturb_func(self, img, seed):
        ### Add rotation and perturbation to imitate images from different domains for DIL
        ''' Rotation
        Create a rotated version of the sample. The digits were rotated by an angle generated randomly between -\pi/4 and \pi/4 radians.
        '''
            
        def generate_random_int(a, b, seed_value):
            local_rng = random.Random(seed_value)  # 创建局部随机数生成器
            return local_rng.randint(a, b)  # 生成指定范围内的随机整数
            
        ''' Rotation
        Create a rotated version of the sample. The digits were rotated by an angle generated randomly between -\pi/4 and \pi/4 radians.
        '''
        self.task_idx = 0
        def generate_random_int(a, b, seed_value):
            local_rng = random.Random(seed_value)  # 创建局部随机数生成器
            return local_rng.randint(a, b)  # 生成指定范围内的随机整数
            
        # rotation
        tor_angle = generate_random_int(0, 45, seed) - 90
        img = img.rotate(tor_angle)
            
        ''' Perturbation
        Create a perturbed version of the sample. The image were perturbed with N(0, \sigma), \sigma \in (5, 15).
        '''
            
        # Add perturbation
        img_array = np.array(img)
        noise_scale = generate_random_int(5, 15, seed)
        noise = np.random.normal(loc=0, scale=noise_scale, size=img_array.shape).astype(np.uint8)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noisy_img_array)
        
        return img


class xy_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
             x : Sequence,
             y : Sequence,
             transform
         ):
        assert len(x) == len(y)
        self.data = x
        self.targets = y
        self.transform = transform
    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.targets)