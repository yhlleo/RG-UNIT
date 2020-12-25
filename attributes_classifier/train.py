import os
import sys
import shutil
import argparse
import pickle
import numpy as np

import torch
from torch.utils import data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import tensorboardX

from celeba_data import CelebA
from solver import Solver

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import prepare_sub_folder, write_loss, get_config


cudnn.benchmark = True
torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces_retrieval.yaml', help='Path to the config file.')
parser.add_argument("--resume", type=int, default=0)
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'])
parser.add_argument('--output_path', type=str, default='.')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu list')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
# get device name: CPU or GPU
print("GPU ID:", opts.gpu_ids)
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

attr_path = config['attr_path'] if 'attr_path' in config else None
train_dataset = CelebA(
    config['data_root'], 
    attr_path, 
    opts.selected_attrs, 
    config['crop_size'], 
    config['image_size'], 
    'train')
test_dataset  = CelebA(
    config['data_root'], 
    attr_path, 
    opts.selected_attrs, 
    config['crop_size'], 
    config['image_size'], 
    'test')

train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=config['batch_size'],
                               shuffle=True,
                               num_workers=config['num_workers'])
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              num_workers=config['num_workers'])

# Setup model and data loader
trainer = Solver(config, device).to(device)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Load  model checkpoint
if opts.resume:
    trainer.resume(config['checkpoint'])

iterations = 0
best_mean_acc = 0.0

while True:
    for it, data_iter in enumerate(train_loader):
        image, label = data_iter
        image = image.to(device)
        label = label.to(device)

        trainer(image, label)
        trainer.update_learning_rate()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            print("Train Loss: %.4f" % trainer.loss.data)
            write_loss(iterations, trainer, train_writer)
        
        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        # Validation
        if (iterations + 1) % config['valid_step'] == 0:
            mean_acc = 0.0
            #num_images = test_dataset.num_images
            for val_it, val_data_iter in enumerate(test_loader):
                val_image, val_label = val_data_iter
                val_image = val_image.to(device)
                val_label = val_label.to(device)
                cur_mean_acc = trainer.mean_acc(val_image, val_label)
                mean_acc += cur_mean_acc * val_image.size(0)
            mean_acc /= test_dataset.num_images

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                trainer.save_best(checkpoint_directory)
            print("Current Mean ACC: %.04f, Best ACC: %.4f" % (mean_acc.data, best_mean_acc.data))

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')


