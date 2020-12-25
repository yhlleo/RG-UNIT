import os
import sys
import shutil

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import tensorboardX

from tools import asign_label
from data_loader import get_loader
from utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images_single, Timer
from gmmunit_retrieval_solver import GMM_Solver

cudnn.benchmark = True
torch.manual_seed(1234)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba_faces_gmmunit_retrieval.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", type=int, default=0)
parser.add_argument('--dataset', type=str, default='CelebA')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu list')
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
# get device name: CPU or GPU
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
print("GPU ID:", opts.gpu_ids)
device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')

train_loader, test_loader = None, None
attr_path = config['attr_path'] if 'attr_path' in config else None
num_class = config['dis']['num_cls']

train_loader = get_loader(
    config['data_root'], 
    config['crop_size'], 
    config['image_size'], 
    config['batch_size'], 
    config['train_list'], 
    config['test_list'],
    config['dataset'], 
    'train', 
    config['num_workers'])
test_loader  = get_loader(
    config['data_root'], 
    config['crop_size'], 
    config['image_size'], 
    1, 
    config['train_list'], 
    config['test_list'],
    config['dataset'], 
    'test', 
    config['num_workers'])

train_display        = [train_loader.dataset[i] for i in range(display_size)]
train_display_images = torch.stack([item[0] for item in train_display]).to(device)
train_display_trg    = torch.stack([item[3] for item in train_display])
train_display_trg    = asign_label(train_display_trg, num_cls=num_class, mode=config['dataset']).to(device)

test_display        = [test_loader.dataset[i] for i in range(display_size)]
test_display_images = torch.stack([item[0] for item in test_display]).to(device)
test_display_trg    = torch.stack([item[3] for item in test_display])
test_display_trg    = asign_label(test_display_trg, num_cls=num_class, mode=config['dataset']).to(device)

# Setup model and data loader
trainer = GMM_Solver(config, device).to(device)
# Load Retrieval model checkpoint
trainer.load_ret_checkpoint(config['ret']['retrieval_checkpoint'])
if config['use_pretrain']:
    trainer.init_network(config['gen_pretrain'], config['dis_pretrain'])
trainer.freezen_networks()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, config) if opts.resume else 0

while True:
    for it, data_iter in enumerate(train_loader):
        x_src, label_src, x_trg, label_trg = data_iter
        c_src = asign_label(label_src, num_class, mode=config['dataset']).to(device)
        c_trg = asign_label(label_trg, num_class, mode=config['dataset']).to(device)

        x_src, x_trg = x_src.to(device), x_trg.to(device)
        label_src, label_trg = label_src.to(device), label_trg.to(device)
        
        with Timer("Elapsed time in update: %f"):
            trainer.dis_update(x_src, c_src, label_src, 
                               x_trg, c_trg, label_trg, 
                               config)
            trainer.gen_update(x_src, c_src, label_src,
                               x_trg, c_trg, label_trg,
                               config)
            torch.cuda.synchronize()
        trainer.update_learning_rate()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            print('Loss: gen %.04f, dis %.04f' % (trainer.loss_gen_total.data, trainer.loss_dis_all.data))
            write_loss(iterations, trainer, train_writer)
            print('Iter {}, lr {}'.format(it, trainer.gen_opt.param_groups[0]['lr']))

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(
                    test_display_images, 
                    test_display_trg)
                train_image_outputs = trainer.sample(
                    train_display_images, 
                    train_display_trg)
            write_2images_single(test_image_outputs, display_size, 
                image_directory, 'test_%08d' % (iterations + 1))
            write_2images_single(train_image_outputs, display_size, 
                image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", 
                iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(
                    train_display_images, 
                    train_display_trg)
            write_2images_single(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

