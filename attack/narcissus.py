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
import sys
import os

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from attack.badnet import BadNet, add_common_attack_args


class Narcissus(BadNet):
    
    def __init__(self):
        super().__init__(True, 'test')

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--narcissus_noise_path", type=str, 
                            help='The path for Narcissus noise trigger. \
                                Note that args.narcissus_noise_path should be *.npy instead of *.png, \
                                    while noise.shape is [1, 3, img_size, img_size] ')
        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/narcissus/default.yaml',
                            help='path for yaml file provide additional default attributes')
        parser.add_argument('--multi_test', type=float, default=3,
                            help='The multiple of noise amplification during testing')
        
        parser.add_argument('--offset_x', type=int, default=0)
        parser.add_argument('--offset_y', type=int, default=0)
        parser.add_argument('--mode', type=str, default='add')
        parser.add_argument('--padding', type=int, default=20)
        parser.add_argument('--position', type=str, default='fixed')
        return parser
        

if __name__ == '__main__':
    attack = Narcissus()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
