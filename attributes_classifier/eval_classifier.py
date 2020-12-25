import os
import sys
import shutil
import argparse
import pickle
import numpy as np
from PIL import Image
import codecs
import yaml

import torch
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import transforms as T

from solver import Solver

cudnn.benchmark = True
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces_retrieval.yaml', help='Path to the config file.')
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'])
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu list')
parser.add_argument('--src_file', type=str, default='./valid/src2trg_celeba-1e4.lst')
parser.add_argument('--trg_file', type=str, default='./valid/trg_celeba-1e4.lst')
opts = parser.parse_args()

def get_config(config):
    with codecs.open(config, 'r', encoding='utf-8') as stream:
        return yaml.load(stream)

# Load experiment setting
config = get_config(opts.config)
# get device name: CPU or GPU
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')
attr_path = config['attr_path'] if 'attr_path' in config else None

def preprocess(attr_path, selected_attrs):
    """Preprocess the CelebA attribute file."""
    attr2idx = {}
    image_label_dict = {}
    lines = [line.rstrip() for line in open(attr_path, 'r')]
    all_attr_names = lines[1].split()
    for i, attr_name in enumerate(all_attr_names):
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i

    lines = lines[2:]
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]

        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(int(values[idx] == '1'))
        image_label_dict[filename] = label
    return image_label_dict

image_label_all_dict = preprocess(attr_path, opts.selected_attrs)
image_label_val_dict = {}
with open(opts.src_file, 'r') as fin_src, \
    open(opts.trg_file, 'r') as fin_trg:
    for src, trg in zip(fin_src, fin_trg):
        fname_src = src.split('\t')[0]
        fname_trg = trg.strip()
        image_label_val_dict[fname_src] = image_label_all_dict[fname_trg]

transform = []
transform.append(T.Resize(config['image_size']))
transform.append(T.ToTensor())
transform.append(T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]))
transform = T.Compose(transform)

# Setup model and data loader
trainer = Solver(config, device).to(device)
# Load  model checkpoint
trainer.resume(config['checkpoint'])
trainer.eval()

correct_per_class = 0.0
num_img = len(image_label_val_dict)
results_dir = config['gen_results']

for idx, ff in enumerate(image_label_val_dict):
    fname = ff.split('.')[0] + '-out.png'
    image = transform(Image.open(os.path.join(results_dir, fname)).convert('RGB')).unsqueeze(0).to(device)
    gt_label = torch.tensor(image_label_val_dict[ff]).float().unsqueeze(0).to(device)
    correct_per_class += trainer.test(image, gt_label).sum(dim=0)
    if (idx+1)%1000 == 0:
        print(idx+1)
correct_per_class /= num_img
print("Accuracy per class: {}".format(correct_per_class))
print("Mean Accuracy: {}".format(correct_per_class.mean()))

