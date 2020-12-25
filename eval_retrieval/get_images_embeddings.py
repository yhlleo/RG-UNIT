# Compute embeddings of test images 

import os
import sys
import json
import shutil
import argparse
import pickle
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import asign_label
from data_loader import get_loader
from retrieval_solver import Retrieval_Solver, GMM_Solver, DWC_Solver
from utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images_single, Timer


cudnn.benchmark = True
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces_retrieval.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", type=int, default=0)
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'Cat2Dog'])
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu list')
parser.add_argument('--use_pretrained_embed', type=int, default=1)
parser.add_argument('--mode', type=str, default='dwc', help='[dwc|gmm]')
parser.add_argument('--data_type', type=str, default='CelebA_retrieval_test')
parser.add_argument('--json_name', type=str, default='images_embeddings.json')
parser.add_argument('--usage_mode', type=str, default='test')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
# get device name: CPU or GPU
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

attr_path = config['attr_path'] if 'attr_path' in config else None

test_loader  = get_loader(
    config['data_root'], 
    config['crop_size'], 
    config['image_size'], 
    config['batch_size'], 
    config['train_list'], 
    config['test_list'],
    opts.data_type, 
    opts.usage_mode, 
    config['num_workers'])

model_name = os.path.splitext(os.path.basename(opts.config))[0]
ret_model_name = config['eval']['ret_path'].split('/')[-1].strip('.pt')
output_directory = os.path.join(opts.output_path + "/results", model_name, ret_model_name)
if not os.path.exists(output_directory):
    print("Creating directory: {}".format(output_directory))
    os.makedirs(output_directory)

# Setup model and data loader
trainer = Retrieval_Solver(config, device).to(device)
# Load Retrieval pretrained model
trainer.load_retnet(config['eval']['ret_path'])
trainer.eval()

if opts.mode == 'dwc':
    pretrained_embed =None
    if opts.use_pretrained_embed:
        with open(config['pretrained_embed'], 'rb') as fin:
            pretrained_embed = pickle.load(fin)
    dwc_gen = DWC_Solver(config, device, pretrained_embed).to(device)
else:
    dwc_gen = GMM_Solver(config, device).to(device)
#print(config['gen_path'])
dwc_gen.initial_network(config['gen_path'])
dwc_gen.eval()

results = {}

print("Computing test images embeddings ...")
for it, data_iter in enumerate(test_loader):
    i, filename, _, label = data_iter
    i = i.to(device)

    i_con, i_att = dwc_gen.gen_encode(i)
    embeddings = trainer.ret_embedding(i_con, i_att)
    for batch_idx, e in enumerate(embeddings):
        results[filename[batch_idx]] = {}
        results[filename[batch_idx]]['embedding'] = np.array(e.cpu()).tolist()
        results[filename[batch_idx]]['label'] = np.array(label[batch_idx]).tolist()
        results[filename[batch_idx]]['extracted_attributes'] = np.array(i_att[batch_idx,:].cpu()).tolist()

print("Saving results")
output_filename = os.path.join(output_directory, opts.json_name)
json.dump(results, open(output_filename,'w'))
print("Done")


