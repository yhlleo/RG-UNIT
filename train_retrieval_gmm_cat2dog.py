import os
import sys
import shutil
import argparse
import pickle
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import tensorboardX

from data_loader import get_loader
from tools import asign_label, dist_sampling_split
from retrieval_solver import Retrieval_Solver, GMM_Solver
from utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images_single, Timer


cudnn.benchmark = True
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces_retrieval_gmm.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", type=int, default=0)
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'])
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu list')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
num_triplets = config['triplets']['num']
# get device name: CPU or GPU
print("GPU ID:", opts.gpu_ids)
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

attr_path = config['attr_path'] if 'attr_path' in config else None
num_class = config['gen']['num_cls']

train_loader = get_loader(
    config['data_root'], 
    config['crop_size'], 
    config['image_size'], 
    config['batch_size'], 
    config['train_list'], 
    config['test_list'],
    'Cat2Dog_retrieval', 
    'train', 
    config['num_workers'])
test_loader  = get_loader(
    config['data_root'], 
    config['crop_size'], 
    config['image_size'], 
    1, 
    config['train_list'], 
    config['test_list'],
    'Cat2Dog_retrieval', 
    'test', 
    config['num_workers'])

# Setup model and data loader
trainer = Retrieval_Solver(config, device).to(device)

# Load pretrained DWC-GAN model
gmm_gen = GMM_Solver(config, device).to(device)
#print(config['gen_path'])
gmm_gen.initial_network(config['gen_path'])
gmm_gen.eval()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Load Retrieval model checkpoint
iterations = trainer.resume(checkpoint_directory, config) if opts.resume else 0

# Plotting config
pp = int(max_iter/config['eval_and_plot_freq'])
train_loss = np.zeros(pp)
train_correct_triplets = np.zeros(pp)
val_loss = np.zeros(pp)
train_loss = np.zeros(pp)
val_correct_triplets = np.zeros(pp)
#it_axes = np.arange(config['eval_and_plot_freq'], max_iter+1, config['eval_and_plot_freq'])

eval_iters = 0
avg_loss_train = 0
avg_correct_train = 0
best_val_loss = 10

# Create triplets using labels for attribute translation
def create_triplets_labels(data_iter):
    # Get 2 real images
    # src_label: Anchor image domain
    # trg_label: Target domain
    i_anc, i_neg, src_label, trg_label = data_iter
    
    # Data to GPUs
    i_anc = i_anc.to(device)
    i_neg = i_neg.to(device)
    src_label = asign_label(src_label, num_class, mode=config['dataset']).to(device)
    trg_label = asign_label(trg_label, num_class, mode=config['dataset']).to(device)

    # Get real images att and content, extracted by frozen Ec, Ea
    i_anc_con, i_anc_att = gmm_gen.gen_encode(i_anc) # Ec(i_a), Ea(i_a)
    i_neg_con, i_neg_att = gmm_gen.gen_encode(i_neg) # Ec(i_n), Ea(i_n)

    # Get anchor image translated to the target domain
    trg_style = dist_sampling_split(trg_label, config['c_dim'], config['stddev'], device)
    i_anc_trans_trg = gmm_gen.gen_decode(i_anc, i_anc_con, trg_style) # T_t(i_a): Img translated to the target domain (positive image)
    # Get translated image content and attributes
    i_anc_trans_trg_con, i_anc_trans_trg_att = gmm_gen.gen_encode(i_anc_trans_trg) 

    triplets = [] # [anchor_content, anchor_attributes, positive_content, positive_attributes, negative_content, negative_attributes]

    # Create different types of triplets
    # Anchor [Content, Att]  ;  Positive [Content, Att]  ;  Negative  [Content, Att] 

    if config['triplets']['easy'] == 1:
    # 1. Easy: A [Ec(i_a), A_t]  ;  P [Ec(T_t(I_a)), Ea(T_t(I_a))]  ; N  [Ec(i_n), Ea(i_n)]
        triplets.append([i_anc_con, trg_style, 
                         i_anc_trans_trg_con, i_anc_trans_trg_att, 
                         i_neg_con, trg_style])

    if config['triplets']['medium'] == 1:
    # 2. Medium: A [Ec(i_a), A_t]  ;  P [Ec(T_t(I_a)), Ea(T_t(I_a))]  ; N  [Ec(T_t(i_n)), Ea(T_t(i_n))]
        # Translate negative image to the target domain and use it as negative
        i_neg_trans_trg = gmm_gen.gen_decode(i_neg, i_neg_con, trg_style)
        i_neg_trans_trg_con, i_neg_trans_trg_att = gmm_gen.gen_encode(i_neg_trans_trg) 
        triplets.append([i_anc_con, trg_style, 
                         i_anc_trans_trg_con, i_anc_trans_trg_att, 
                         i_neg_trans_trg_con, i_neg_trans_trg_att]) 

    if config['triplets']['hardest'] == 1:
    # 3. Hardest: A [Ec(i_a), A_t]  ;  P [Ec(T_t(I_a)), Ea(T_t(I_a))]  ; N  [Ec(T_a(i_a)), Ea(T_a(i_a))]
        # Sample anchor image from its GT domain and use it as negative
        i_anc_sampled = gmm_gen.gen_rec(i_anc) 
        i_anc_sampled_con, i_anc_sampled_att = gmm_gen.gen_encode(i_anc_sampled)
        triplets.append([i_anc_con, trg_style, 
                         i_anc_trans_trg_con, i_anc_trans_trg_att, 
                         i_anc_sampled_con, i_anc_sampled_att])

    return triplets


while True:
    for it, data_iter in enumerate(train_loader):

        triplets = create_triplets_labels(data_iter)
        with Timer("Elapsed time in update: %f"):
            for t in triplets:
                loss, correct = trainer.ret_update(t[0], t[1], t[2], t[3], t[4], t[5])
                avg_loss_train += loss.data.item()
                avg_correct_train += torch.sum(correct).data.item()
                torch.cuda.synchronize()
        trainer.update_learning_rate()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)
        
        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        # Evaluate and plot loss and correct triplets
        if config['plot'] == 'True' and (iterations + 1) % config['eval_and_plot_freq'] == 0:
            print("Evaluating and plotting. Eval iter: " + str(eval_iters) + " /  Train iters: " + str(it+1))
            avg_loss_val = 0
            avg_correct_val = 0

            # Forward evaluation set
            for val_it, val_data_iter in enumerate(test_loader):
                if val_it == config['max_eval_iter']: break

                val_triplets = create_triplets_labels(val_data_iter)
                for t in triplets:
                    loss, correct = trainer.ret_eval(t[0], t[1], t[2], t[3], t[4], t[5]) 
                    avg_loss_val += loss.data.item()
                    avg_correct_val += torch.sum(correct).data.item()
                    #torch.cuda.synchronize()

            avg_loss_train /= (config['eval_and_plot_freq'] * num_triplets)
            avg_correct_train /= config['eval_and_plot_freq']
            avg_loss_val /= (val_it * num_triplets)
            avg_correct_val /= val_it 

            print("Train Loss: " + str(avg_loss_train) + "; Val Loss:" + str(avg_loss_val) + "; Train C.:" + str(avg_correct_train) + "; Val C.:" + str(avg_correct_val))

            # Save model if it achieves best val loss
            if avg_loss_val < best_val_loss:
                best_val_loss = avg_loss_val
                print("Best val loss. Saving model")
                trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')


