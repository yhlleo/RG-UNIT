"""
Base on: https://github.com/yunjey/stargan
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data

random.seed(1234)

class CelebA_retrieval(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, image_dir, train_list_path, test_list_path, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.selected_attrs = 5
        self.train_dataset = self.load_files(train_list_path)
        self.test_dataset  = self.load_files(test_list_path)

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

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

    def __getitem__(self, index):
        """Return an anchor image, a random negative image and random transformations."""

        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, src_label = dataset[index]
        while True:
            neg_filename, neg_src_label = random.choice(dataset)
            if filename != neg_filename: break

        # -------------- Easy target and negative labels only changing one element -------------- #
        ### Target label
        trg_label = src_label.copy()
        att_to_change_trg = random.randint(0,self.selected_attrs-1)
        new_value = 1
        if src_label[att_to_change_trg] == new_value:
            new_value = 0
        trg_label[att_to_change_trg] = new_value
        
        ### Negative label
        ## Simple Negative label
        while True:
            _, rnd_label = random.choice(dataset)
            if rnd_label != trg_label and rnd_label != src_label: break
        # -------------------------------------------- #

        image = Image.open(os.path.join(self.image_dir, filename))
        image = self.transform(image)
        if image.size(0) == 1: # convert grayscale to rgb
            image = torch.cat([image, image, image], dim=0)

        neg_image = Image.open(os.path.join(self.image_dir,neg_filename))
        neg_image = self.transform(neg_image)
        if neg_image.size(0) == 1: # convert grayscale to rgb
            neg_image = torch.cat([neg_image, neg_image, neg_image], dim=0)

        # If using labels
        src_label = torch.tensor(src_label).float()
        trg_label = torch.tensor(trg_label).float()
        rnd_label = torch.tensor(rnd_label).float()
        return image, neg_image, src_label, trg_label, rnd_label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


