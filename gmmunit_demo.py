"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function

import os
import sys
import pickle
import argparse
from PIL import Image

import torch
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from utils import get_config
from gmmunit_retrieval_solver import GMM_Solver
from tools import dist_sampling_split, asign_label
from data_ios.dwcgan_data.vocab import Vocab, ListsToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces.yaml', help="net configuration")
parser.add_argument('--checkpoint', type=str, default='gen_01000000.pt', help="checkpoint of models")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu id list')
parser.add_argument('--style_dir', type=str, default='../datasets/celeba/images', help='folder path of style images')
parser.add_argument('--test_list', type=str, default='./valid/')
parser.add_argument('--use_pretrained_embed', type=int, default=1)
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
config_name = os.path.splitext(os.path.basename(opts.config))[0]
test_name = os.path.splitext(os.path.basename(opts.test_list))[0]
output_folder = 'results/{}-{}'.format(config_name, test_name)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
use_attention = config['gen']['use_attention']
dataset = config['dataset']

device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')
#model_name = os.path.splitext(os.path.basename(opts.config))[0]
#model_dir = 'outputs/{}/checkpoints'.format(model_name)

# Setup model and data loader
trainer = GMM_Solver(config, device).to(device)
#state_dict = torch.load(os.path.join(model_dir, opts.checkpoint), 
state_dict = torch.load(opts.checkpoint,
    map_location=lambda storage, loc: storage)
trainer.gen.load_state_dict(state_dict['a'])

# Load Retrieval model checkpoint
trainer.load_ret_checkpoint(config['ret']['retrieval_checkpoint'])
trainer.eval()
print("model loaded.")

c_dim = config['c_dim']
num_results = config['ret']['num_results']
#domains = ['10001', '01001', '00101', '01011', '01000', '01010']
domains = ['10011', '01011', '00111', '01001', '01010']
num_class = config['dis']['num_cls']

def infer(config, model, input_file, device=None):
    encode = model.gen.encode
    decode = model.gen.decode
    retrieve_closer_images = model.retrieve_closer_images
    encode_retrieved = model.gen.encode_retrieved

    with torch.no_grad():
        transform = transforms.Compose([transforms.CenterCrop(config['crop_size']),
                                        transforms.Resize(config['image_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = Variable(transform(Image.open(os.path.join('../datasets/celeba/images', input_file)).convert('RGB')).unsqueeze(0).to(device))
        input_name = input_file.split('/')[-1].split('.')[0]

        for idx, dd in enumerate(domains):
            style = torch.tensor([[int(v) for v in dd]]).float()
            c_trg = asign_label(style, num_cls=num_class, mode=config['dataset']).to(device)

            content_src, style_src_outs = encode(image)
            style_src = torch.cat(style_src_outs[0],dim=1)
            retrieved_images_src, _ = retrieve_closer_images(content_src, 
                style_src, num_results)

            c_src = torch.ones(1, num_class).float().to(device)
            for i in range(num_class):
                if style_src[0,i*c_dim:(i+1)*c_dim].mean() < 0.0:
                    c_src[0,i]=-1.0

            z_random = dist_sampling_split(c_trg, c_dim, config['stddev'], device=device)
            #z_random = trainer.style_replace(c_src, c_trg, style_src, z_random)

            retrieved_images_trg, _ = retrieve_closer_images(content_src, 
                z_random, num_results)
            retrieved_feats = encode_retrieved(retrieved_images_trg)
            outputs, outputs_att = decode(content_src, z_random, retrieved_feats)
            if use_attention:
                outputs = outputs*outputs_att + (1-outputs_att)*image
            outputs = (outputs + 1.) / 2.

            # save output image
            path = os.path.join(output_folder, '{}-{}-out.png'.format(input_name, idx))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            # save output attention
            path = os.path.join(output_folder, '{}-{}-att.png'.format(input_name, idx))
            vutils.save_image(outputs_att.data, path, padding=0, normalize=True)
            # save retrieved image
            for i in range(num_results):
                cur_ret_src = retrieved_images_src[i:i+1]
                cur_ret_trg = retrieved_images_trg[i:i+1]
                path = os.path.join(output_folder, '{}-{}-{}-src-ret.png'.format(input_name, idx, i))
                vutils.save_image(cur_ret_src.data, path, padding=0, normalize=True)
                path = os.path.join(output_folder, '{}-{}-{}-trg-ret.png'.format(input_name, idx, i))
                vutils.save_image(cur_ret_trg.data, path, padding=0, normalize=True)
        # save input image
        path = os.path.join(output_folder, '{}.png'.format(input_name))
        vutils.save_image(image.data, path, padding=0, normalize=True)

with open(opts.test_list, 'r') as fin:
    for line in fin:
        fname = line.strip()
        print(fname)
        infer(config, trainer, fname, device)
