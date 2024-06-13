import argparse
import logging
import torch
import sys
import os

def set_args(parser: argparse.ArgumentParser):
    parser.add_argument('--result_path', type=str,
                        help='The path of Backdoorbench result, which is a dict with \'model\':[model params]')
    parser.add_argument('--save_path', type=str,
                        help='The path to save the separated model params.')
    return parser

if __name__ == '__main__':
    print(f'The operation path (cwd) is {os.getcwd()}.')
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = set_args(parser)
    args = parser.parse_args()
    
    result_dict = torch.load(args.result_path)
    torch.save(result_dict['model'], args.save_path)
    
    print(f'Move model in {args.result_path} to {args.save_path}')
    
