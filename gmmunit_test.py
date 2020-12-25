"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import os
import sys
import argparse
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from utils import get_config
from tools import dist_sampling_split, asign_label
from gmmunit_retrieval_solver import GMM_Solver

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
#parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of models")
parser.add_argument('--image_dir', type=str, default='../datasets/celeba/images')
parser.add_argument('--test_list', type=str, default='valid/src2trg_celeba-label-1e4.lst')
parser.add_argument('--num_style',type=int, default=1, help="number of styles to sample")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu id list')
#parser.add_argument('--trg_domain', type=int, default=0, 
#    help='0: [1,0,0,0,1], 1: [0,1,0,0,1], 2: [0,0,1,0,1], 3: [0,1,0,1,1], 4: [0,1,0,1,0]')
opts = parser.parse_args()

#torch.manual_seed(opts.seed)
#torch.cuda.manual_seed(opts.seed)
config_name = os.path.splitext(os.path.basename(opts.config))[0]
test_name = os.path.splitext(os.path.basename(opts.test_list))[0]
output_folder = 'results/{}-{}'.format(config_name, test_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
use_attention = config['gen']['use_attention']
dataset = config['dataset']

device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')
model_name = os.path.splitext(os.path.basename(opts.config))[0]
model_dir = 'outputs/{}/checkpoints'.format(model_name)
# Setup model and data loader
trainer = GMM_Solver(config, device).to(device)
state_dict = torch.load(os.path.join(model_dir, opts.checkpoint), 
    map_location=lambda storage, loc: storage)
trainer.gen.load_state_dict(state_dict['a'])
# Load Retrieval model checkpoint
trainer.load_ret_checkpoint(config['ret']['retrieval_checkpoint'])
trainer.eval()

print("model loaded.")

encode = trainer.gen.encode
decode = trainer.gen.decode
retrieve_closer_images = trainer.retrieve_closer_images
encode_retrieved = trainer.gen.encode_retrieved
new_size = config['image_size']
num_results = config['ret']['num_results']
c_dim = config['c_dim']

with torch.no_grad():
    transform_list = []
    if dataset == "CelebA":
        transform_list += [transforms.CenterCrop(config['crop_size'])]
    transform_list += [transforms.Resize((new_size, new_size)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    num_class = config['dis']['num_cls']
    if dataset == "CelebA":
        with open(opts.test_list, 'r') as f:
            for ll in f:
                splits = ll.strip().split()
                fname = splits[0].split('.')[0]
                path  = os.path.join(opts.image_dir, splits[0])
                image = Variable(transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device))
                print(fname)

                content, style_src_outs = encode(image)
                style_src = torch.cat(style_src_outs[0],dim=1)
                c_src = torch.ones(1, num_class).float().to(device)
                for i in range(num_class):
                    if style_src[0,i*c_dim:(i+1)*c_dim].mean() < 0.0:
                        c_src[0,i]=-1.0
                
                style = torch.tensor([[int(v) for v in splits[-1]]]).float()
                c_trg = asign_label(style, num_cls=num_class, mode=config['dataset']).to(device)

                for j in range(opts.num_style):
                    z_random = dist_sampling_split(c_trg, c_dim, config['stddev'], device=device)
                    z_random = trainer.style_replace(c_src, c_trg, style_src, z_random)
                    
                    retrieved_images, _ = retrieve_closer_images(content, 
                        z_random, num_results)
                    retrieved_feats = encode_retrieved(retrieved_images)

                    outputs, outputs_att = decode(content, z_random, retrieved_feats)
                    if use_attention:
                        outputs = outputs.data*outputs_att.data + (1-outputs_att.data)*image.data
                    outputs = (outputs + 1) / 2.
                    path = os.path.join(output_folder, '{}-{:03d}.png'.format(fname.split('.')[0], j))
                    vutils.save_image(outputs.data, path, padding=0, normalize=True)
                    #path = os.path.join(output_folder, '{}-{:03d}-att.png'.format(fname.split('.')[0], j))
                    #vutils.save_image(outputs_att.data, path, padding=0, normalize=True)
                #image = (image + 1.0) / 2.0
                #vutils.save_image(image.data, os.path.join(output_folder, '{}.png'.format(fname)), padding=0, normalize=True)
    else:
        with open(opts.test_list, 'r') as f:
            for ll in f:
                fname, lab = ll.strip().split()
                path = os.path.join(opts.image_dir, fname)
                image = Variable(transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device))
                print(fname)

                content, _ = encode(image)
                style = torch.tensor([1-int(lab)]).float()
                c_trg = asign_label(style, num_cls=num_class, mode=config['dataset']).to(device)

                for j in range(opts.num_style):
                    z_random = dist_sampling_split(c_trg, c_dim, config['stddev'], device=device)
                    
                    retrieved_images, _ = retrieve_closer_images(content, 
                        z_random, num_results)
                    retrieved_feats = encode_retrieved(retrieved_images)

                    outputs, outputs_att = decode(content, z_random, retrieved_feats)
                    if use_attention:
                        outputs = outputs.data*outputs_att.data + (1-outputs_att.data)*image.data
                    outputs = (outputs + 1) / 2.
                    path = os.path.join(output_folder, '{}-{:03d}.png'.format(fname.split('/')[-1].split('.')[0], j))
                    vutils.save_image(outputs.data, path, padding=0, normalize=True)
