# For each test image, generate a query with a random attribute modification
# Compute its embedding and save results

import os
import sys
import json
import pickle
import shutil
import argparse
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import get_loader
from tools import asign_label, dist_sampling_split
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
    'CelebA_retrieval_test', 
    'test', 
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

print("Computing queries embeddings ...")
for it, data_iter in enumerate(test_loader):
    i, filename, trg_label, _ = data_iter

    i = i.to(device)
    #txt_src2trg = txt_src2trg.to(device)
    #txt_lens = txt_lens.to(device)

    i_con, i_att = dwc_gen.gen_encode(i)
    
    # -------------- Using LSTM -------------- #
    # target_att = trainer.gen_translate_attributes(i_att, txt_src2trg, txt_lens)
    # embeddings = trainer.ret_embedding(i_con, target_att)
    # -------------------------------------------- #

    # -------------- Using attribute labels -------------- #
    trg_label = trg_label.to(device)
    trg_style = dist_sampling_split(trg_label, config['c_dim'], config['stddev'], device)
    embeddings = trainer.ret_embedding(i_con, trg_style)
    trg_label = trg_label.cpu()
    # -------------------------------------------- #

    for batch_idx, e in enumerate(embeddings):
        results[filename[batch_idx]] = {}
        results[filename[batch_idx]]['embedding'] = np.array(e.cpu()).tolist()
        #results[filename[batch_idx]]['text_modifier'] = str(diff_txt_trg[batch_idx])
        results[filename[batch_idx]]['trg_label'] = np.array(trg_label[batch_idx]).tolist()
    # print(str(it) + ' / ' + str(len(test_loader)))

print("Saving results")
output_filename = output_directory + '/queries_embeddings.json'
json.dump(results, open(output_filename,'w'))
print("Done")


