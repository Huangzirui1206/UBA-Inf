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

import logging
import os

import numpy as np
import torch
import torch.nn as nn

### The utils.py in Narcissus official code (part with some revisement)

def narcissus_apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=20,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, img_size, img_size)
    outputs: torch.Tensor(N, 3, img_size, img_size)
    '''
    
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if images.dim() == 3:
        noise_now = noise.clone()[0,:,:,:]
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images += m(noise_now)
        else:
            images += noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = noise.clone()
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[i:i+1] += m(noise_now)
            else:
                images[i:i+1] += noise_now
    return images


class narcissusTriggerAttack(object):
    def __init__(self, noise, 
                 offset_x=0, offset_y=0,
                 mode='change',padding=20,
                 position='fixed'):
        self.noise = noise # [-1, 1] noise after transform
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.mode = mode
        self.padding = padding
        self.position = position
        
    def __call__(self, img):
        return self.narcissus_trigger(img)
        
    def narcissus_trigger(self, img):
        return torch.clamp(narcissus_apply_noise_patch(noise=self.noise, images=img, 
                                           offset_x=self.offset_x, offset_y=self.offset_y,
                                           mode=self.mode, padding=self.padding,
                                           position=self.position),
                           -1,1)