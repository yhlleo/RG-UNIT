
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as distributed

from core import (
    retrieval_run,
    get_config
)

def init_processes(args, local_rank, func, opts):
    os.environ['MASTER_ADDR'] = opts.master_addr
    os.environ['MASTER_PORT'] = opts.master_port
    distributed.init_process_group(backend=opts.backend, rank=local_rank, world_size=opts.world_size)
    func(args, local_rank)

def main(args, opts):
    print(args)
    cudnn.benchmark = True
    if opts.mode == 'train':
        torch.manual_seed(opts.seed)

    if opts.mode == 'train':
        mp.set_start_method('spawn')
        processes = []
        for rank in range(opts.num_gpus):
            p = mp.Process(target=init_processes, args=(args, rank, retrieval_run, opts))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--config_path', type=str, default='configs/retrieval_celeba.yaml')
    parser.add_argument('--resume_iter', type=int, default=0)

    # dirs
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # distributed training
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='18888')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--backend', type=str, default='nccl', help='nccl | gloo')

    # data
    parser.add_argument('--image_dir', type=str, default='datasets/celeba/images')
    parser.add_argument('--train_list_path', type=str, default='datasets/celeba/list_attr_celeba-train.txt')
    parser.add_argument('--test_list_path', type=str, default='datasets/celeba/list_attr_celeba-val.txt')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_models/gmmunit_gen.pth')
    opts = parser.parse_args()

    args = get_config(opts.config_path)
    args['world_size']     = opts.world_size
    args['resume_iter']    = opts.resume_iter
    args['checkpoint_dir'] = opts.checkpoint_dir
    args['image_dir']      = opts.image_dir
    args['train_list_path'] = opts.train_list_path
    args['test_list_path'] = opts.test_list_path
    args['pretrained_path'] = opts.pretrained_path
    main(args, opts)

