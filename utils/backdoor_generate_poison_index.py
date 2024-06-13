# idea: this file is for the poison sample index selection,
#   generate_single_target_attack_train_poison_index is for all-to-one attack label transform
#   generate_poison_index_from_label_transform aggregate both all-to-one and all-to-all case.

import sys, logging
sys.path.append('../')
import random
import numpy as np
from typing import Callable, Union, List


def generate_single_target_attack_train_poison_index(
        targets:Union[np.ndarray, List],
        tlabel: int,
        pratio: Union[float, None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
        train : bool = True,
) -> np.ndarray:
    '''
    # idea: given the following information, which samples will be used to poison will be determined automatically.

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    targets = np.array(targets)
    logging.debug('Reminder: plz note that if p_num or pratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied')
    logging.debug('Reminder: priority p_num > pratio, and choosing fix number of sample is prefered if possible ')
    poison_index = np.zeros(len(targets))
    if train == False:
        non_zero_array = np.where(targets != tlabel)[0]
        poison_index[list(non_zero_array)] = 1
    else:
        #TRAIN !
        if clean_label == False:
            # in train state, all2one non-clean-label case NO NEED TO AVOID target class img
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(np.arange(len(targets)), p_num, replace = False)
                    poison_index[list(non_zero_array)] = 1
                else:
                    non_zero_array = np.random.choice(np.arange(len(targets)), round(pratio * len(targets)), replace = False)
                    poison_index[list(non_zero_array)] = 1
        else:
            if p_num is not None or round(pratio * len(targets)):
                if p_num is not None:
                    non_zero_array = np.random.choice(np.where(targets == tlabel)[0], p_num, replace = False)
                    poison_index[list(non_zero_array)] = 1
                else:
                    non_zero_array = np.random.choice(np.where(targets == tlabel)[0], round(pratio * len(targets)), replace = False)
                    poison_index[list(non_zero_array)] = 1
    logging.info(f'poison num:{sum(poison_index)},real pratio:{sum(poison_index) / len(poison_index)}')
    if sum(poison_index) == 0:
        logging.info('Pay attention! No poison sample generated !')
    return poison_index

def generate_single_target_attack_train_cover_index(
        targets: Union[np.ndarray, List],
        poison_index: Union[np.ndarray, None],
        cratio: Union[float, None] = None,
        c_num: Union[int,None] = None,
        c_method: str = 'adversarial-label-consistent',
        clabel: int = 4,
        tlabel: int = 0
) -> np.ndarray:
    '''
    # idea: Given the following information, which samples will be used to cover will be determined automatically.
    # note: Once a sample is poisoned, it won't be made as a cover sample at the same time.

    :param targets: y array of clean dataset that tend to do cover

    :param cratio: cover ratio, if the whole dataset size = 1
    :param c_num: cover data number, more precise
    need one of cratio and cnum

    :param poison_index: poison index, indicates which samples have already been chosen to poison
    :return: one-hot array to indicate which of samples is selected, yet the one-hot array is consisted of 0s and 2s.
    
    # note: The return value only indicates which samples whill be covered. To craft the ultimate index indicator of both poison
    # samples and cover samples, you need to combine cover_index and poison_index together.
    '''
    targets = np.array(targets)
    logging.debug('Reminder: plz note that if c_num or cratio exceed the number of possible candidate samples\n then only maximum number of samples will be applied')
    logging.debug('Reminder: priority c_num > cratio, and choosing fix number of sample is prefered if possible ')
    logging.debug('Reminder: once a sample is poisoned, it won\'t be made as a cover sample at the same time. ')
    cover_index = np.zeros(len(targets))
    
    if c_method == 'adversarial-label-consistent' or\
        c_method == 'label-consistent':
        if c_num is not None or round(cratio * len(targets)):
            zero_index_array = np.argwhere(((poison_index == 0) * (targets != tlabel)))
            if c_num is not None:
                cover_index_array = np.random.choice(np.arange(len(zero_index_array)), c_num, replace = False)
                non_zero_array = zero_index_array[cover_index_array]
                cover_index[list(non_zero_array)] = 2
            else:
                cover_index_array = np.random.choice(np.arange(len(zero_index_array)), round(cratio * len(targets)), replace = False)
                non_zero_array = zero_index_array[cover_index_array]
                cover_index[list(non_zero_array)] = 2
    elif c_method ==  'adversarial-single-label-consistent' or\
        c_method == 'single-label-consistent': 
        if c_num is not None or round(cratio * len(targets)):
            zero_index_array = np.argwhere((poison_index == 0) * (targets == clabel))
            if c_num is not None:
                cover_index_array = np.random.choice(np.arange(len(zero_index_array)), c_num, replace = False)
                non_zero_array = zero_index_array[cover_index_array]
                cover_index[list(non_zero_array)] = 2
            else:
                cover_index_array = np.random.choice(np.arange(len(zero_index_array)), round(cratio * len(targets)), replace = False)
                non_zero_array = zero_index_array[cover_index_array]
                cover_index[list(non_zero_array)] = 2
    elif c_method == 'random-label-flipping' or\
        c_method == 'const-label-flipping':
        if c_num is not None or round(cratio * len(targets)):
            zero_index_array = np.argwhere(((poison_index == 0) * (targets == tlabel)))
            if c_num is not None:
                cover_index_array = np.random.choice(np.arange(len(zero_index_array)), c_num, replace = False)
                non_zero_array = zero_index_array[cover_index_array]
                cover_index[list(non_zero_array)] = 2
            else:
                cover_index_array = np.random.choice(np.arange(len(zero_index_array)), round(cratio * len(targets)), replace = False)
                non_zero_array = zero_index_array[cover_index_array]
                cover_index[list(non_zero_array)] = 2
    else:
        raise ValueError('There is no such cover method. The options are adversarial-label-consistent, adversarial-single-label-consistent, single-label-consistent, label-consistent, const-label-flipping and random-label-flipping')
    logging.info(f'cover num:{sum(cover_index) // 2},real pratio:{sum(cover_index) // 2 / len(cover_index)}')
    if sum(cover_index) == 0:
        raise SystemExit('No poison sample generated !')
    return cover_index



from utils.bd_label_transform.backdoor_label_transform import *
from typing import Optional
import torch
def generate_poison_index_from_label_transform(
        original_labels: Union[np.ndarray, List],
        label_transform: Callable,
        train: bool = True,
        pratio : Union[float,None] = None,
        p_num: Union[int,None] = None,
        clean_label: bool = False,
) -> Optional[np.ndarray]:
    '''

    # idea: aggregate all-to-one case and all-to-all cases, case being used will be determined by given label transformation automatically.

    !only support label_transform with deterministic output value (one sample one fix target label)!

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param pratio: poison ratio, if the whole dataset size = 1
    :param p_num: poison data number, more precise
    need one of pratio and pnum

    :param clean_label: whether use clean label logic to select (only in all2one case can be used !!!)
    :param train: train or test phase (if test phase the pratio will be close to 1 no matter how you set)
    :return: one-hot array to indicate which of samples is selected
    '''
    if isinstance(label_transform, AllToOne_attack):
        # this is both for allToOne normal case and cleanLabel attack
        return generate_single_target_attack_train_poison_index(
            targets = original_labels,
            tlabel = label_transform.target_label,
            pratio = pratio,
            p_num = p_num,
            clean_label = clean_label,
            train = train,
        )

    elif isinstance(label_transform, AllToAll_shiftLabelAttack):
        
        raise NotImplementedError("In UAB experiment, AllToAll_shiftLabelAttack is forbidden.")
        
        if train:
            pass
        else:
            p_num = None
            pratio = 1

        if p_num is not None:
            select_position = np.random.choice(len(original_labels), size = p_num, replace=False)
        elif pratio is not None:
            select_position = np.random.choice(len(original_labels), size=round(len(original_labels) * pratio), replace=False)
        else:
            raise SystemExit('p_num or pratio must be given')
        logging.info(f'poison num:{len(select_position)},real pratio:{len(select_position) / len(original_labels)}')

        poison_index = np.zeros(len(original_labels))
        poison_index[select_position] = 1

        return poison_index
    else:
        logging.debug('Not valid label_transform')


def generate_cover_index_from_label_transform(
        original_labels: Union[np.ndarray, list],
        label_transform: Callable,
        poison_index: Union[list, None],
        cratio : Union[float,None] = None,
        c_num: Union[int,None] = None,
        c_method: str = 'adversarial-label-consistent',
        clabel: int = 4
) -> Optional[np.ndarray]:
    '''

    # idea: aggregate all-to-one case and all-to-all cases, case being used will be determined by given label transformation automatically.

    !only support label_transform with deterministic output value (one sample one fix target label)!

    :param targets: y array of clean dataset that tend to do poison
    :param tlabel: target label in backdoor attack

    :param cratio: cover ratio, if the whole dataset size = 1
    :param c_num: cover data number, more precise
    need one of cratio and cnum
    
    :param c_method: which cover method to use
    :param clabel: For single-label-consistent cover methods or const-label-flipping method
        
    :return: one-hot array to indicate which of samples is selected
    '''
    if isinstance(label_transform, AllToOne_attack):
        # this is both for allToOne normal case and cleanLabel attack
        if poison_index is None:
            poison_index = torch.zeros(original_labels.shape)
        return generate_single_target_attack_train_cover_index(
            targets = original_labels,
            poison_index = poison_index,
            cratio = cratio,
            c_num = c_num,
            tlabel=label_transform.target_label,
            clabel=clabel,
            c_method=c_method
        )

    elif isinstance(label_transform, AllToAll_shiftLabelAttack):
        
        raise NotImplementedError("In UAB experiment, AllToAll_shiftLabelAttack is forbidden.")
        
    else:
        logging.debug('Not valid label_transform')
