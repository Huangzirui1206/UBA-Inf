'''
this script is for lf attack

link : https://github.com/YiZeng623/frequency-backdoor
The original license is placed at the end of this file.

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. attack or use the model to do finetune with 5% clean data
7. save the attack result for defense

@inproceedings{zeng2021rethinking_lf,
  title={Rethinking the backdoor attacks' triggers: A frequency perspective},
  author={Zeng, Yi and Park, Won and Mao, Z Morley and Jia, Ruoxi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}

The original license :
MIT License

Copyright (c) 2021 Yi Zeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import logging
import os
import sys

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import torch
import shutil
from copy import deepcopy
from attack_sisa.badnet_sisa import BadNet, add_common_attack_args
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from attack_sisa.aggregator import aggregator
from torch.utils.data.dataloader import DataLoader
from utils.aggregate_block.train_settings_generate import argparser_criterion


from attack_sisa.badnet_sisa import BadNet, add_common_attack_args
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape


class LowFrequency(BadNet):

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument('--lowFrequencyPatternPath', type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/lf/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        if ('lowFrequencyPatternPath' not in args.__dict__) or (args.lowFrequencyPatternPath is None):
            args.lowFrequencyPatternPath = f"../resource/lowFrequency/{args.dataset}_{args.model}_0_255.npy"
            logging.info(f"args.lowFrequencyPatternPath does not found, so = {args.lowFrequencyPatternPath}")

        return args


if __name__ == '__main__':
    attack = LowFrequency()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    
    # prepare shard data
    attack.data_prepare = attack.stage0_training_data_split()
    
    # In SISA algorithm, generate shard models in a loop
    cnt = 0
    save_path_base = deepcopy(args.save_path)
    while True:
        args.save_path = save_path_base + f'/shard{cnt}'
        os.makedirs(args.save_path)
        assert os.path.exists(args.save_path)
        try:
            attack.stage1_non_training_data_prepare()
            attack.stage2_training()
            cnt += 1
        except StopIteration:
            os.rmdir(args.save_path)
            args.save_path = save_path_base
            break
        
    # get the aggregated result model
    attack_models = []
    unlearn_models = []
    for i in range(cnt):
        shard_path = save_path_base + f'/shard{i}'
        attack_path = shard_path + '/attack/attack_result.pt'
        unlearn_path = shard_path + '/unlearn/attack_result.pt'
        assert(os.path.exists(attack_path))
        # get the attack model
        attack_net_dict = torch.load(attack_path, map_location=attack.device)
        attack_net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )
        attack_net.load_state_dict(attack_net_dict['model'])
        attack_models.append(attack_net)
        # get the unlearn model, if there exists one
        if os.path.exists(unlearn_path):
            unlearn_net_dict = torch.load(unlearn_path, map_location=attack.device)
            unlearn_net = generate_cls_model(
                model_name=args.model,
                num_classes=args.num_classes,
                image_size=args.img_size[0],
            )
            unlearn_net.load_state_dict(unlearn_net_dict['model'])
            unlearn_models.append(unlearn_net)
        
        
    assert(len(attack_models))
    attack_aggregator = aggregator(attack_models)
    
    if(len(unlearn_models)):
        unlearn_aggregator = aggregator(unlearn_models)
    else:
        unlearn_aggregator = None
        
    # test the aggregated result model
        # get test dataloader from attack
    clean_test_dataloader = \
        DataLoader(attack.clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers, )
    bd_test_dataloader = \
        DataLoader(attack.bd_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=args.pin_memory, num_workers=args.num_workers, )
    criterion = argparser_criterion(args)
    
    attack_test_result = \
        attack_aggregator.backdoorTest(clean_test_dataloader, 
                                       bd_test_dataloader, 
                                       attack.device,
                                       criterion
                                       )
        
    if unlearn_aggregator:
        unlearn_test_result = \
            unlearn_aggregator.backdoorTest(clean_test_dataloader, 
                                       bd_test_dataloader, 
                                       attack.device,
                                       criterion)
    else:
        unlearn_test_result = None
        
    print(f'The attack result is {attack_test_result}')
    with open(os.path.join(args.save_path, 'attack_summary.csv'), 'w+') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in attack_test_result.items()]
        
    if unlearn_test_result is not None:
        print(f'The attack result is {unlearn_test_result}')
        with open(os.path.join(args.save_path, 'unlearn_summary.csv'), 'w+') as f:
            [f.write('{0},{1}\n'.format(key, value)) for key, value in unlearn_test_result.items()]
            
    # delete all .png images
    files = os.listdir(args.save_path)
    for file in files:
        #得到该文件下所有目录的路径
        m = os.path.join(args.save_path,file)
        #判断该路径下是否是文件夹
        if (os.path.isdir(m)):
            shutil.rmtree(os.path.join(m, 'bd_train_dataset'))
            shutil.rmtree(os.path.join(m, 'bd_test_dataset'))
            if os.path.exists(os.path.join(m, 'cv_train_dataset')):
                shutil.rmtree(os.path.join(m, 'cv_test_dataset'))

'''
MIT License

Copyright (c) 2021 Yi Zeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
