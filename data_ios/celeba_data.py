import os
import random
from PIL import Image
import numpy as np

import torch
from torch.utils import data

"""
Base on: https://github.com/yunjey/stargan
"""
random.seed(1234)

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, image_dir, train_list_path, test_list_path, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = self.load_files(train_list_path)
        self.test_dataset  = self.load_files(test_list_path)
        self.selected_attrs = 5
        #self.all_domains = self.collect_all_domains(self.selected_attrs)

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
        print(self.num_images)

    def load_files(self, list_path):
        lines = [line.rstrip() for line in open(list_path, 'r')]
        dataset = []
        for ll in lines:
            fname, label = ll.split()
            label = [int(v) for v in label]
            dataset.append([fname, label])
        print('Finished loading the CelebA dataset...')
        random.shuffle(dataset)
        return dataset

    def collect_all_domains(self, num_attr):
        domains = ['0', '1']
        for i in range(1, num_attr):
            current_domains = []
            for da in domains:
                current_domains.extend([da+'0', da+'1'])
            domains = current_domains

        domains_str2int = []
        for da in domains:
            domains_str2int.append([int(v) for v in da])
        return domains_str2int

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        fname_src, src_label = dataset[index]
        fname_trg, trg_label = random.choice(dataset)
        #trg_label = random.choice(self.all_domains)
        #fname_trg, _ = random.choice(dataset)
        src_image = Image.open(os.path.join(self.image_dir, fname_src))
        trg_image = Image.open(os.path.join(self.image_dir, fname_trg))
        src_image, trg_image = self.transform(src_image), self.transform(trg_image)
        src_label, trg_label = torch.tensor(src_label).float(), torch.tensor(trg_label).float()
        return  src_image, src_label, trg_image, trg_label 
            

    def __len__(self):
        """Return the number of images."""
        return self.num_images
