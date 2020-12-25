"""
Base on: https://github.com/yunjey/stargan
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, image_dir, attr_path, selected_attrs, crop_size, image_size,  mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.mode = mode
        self.train_dataset = []
        self.test_dataset  = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

        self.transform = []
        if mode == 'train':
            self.transform.append(T.RandomHorizontalFlip())
        self.transform.append(T.CenterCrop(crop_size))
        self.transform.append(T.Resize(image_size))
        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(
            mean=[0.485, 0.456, 0.406], #(0.5, 0.5, 0.5), 
            std=[0.229, 0.224, 0.225])) #(0.5, 0.5, 0.5)))
        self.transform = T.Compose(self.transform)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(int(values[idx] == '1'))

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, src_label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        image = self.transform(image)
        if image.size(0) == 1: # convert grayscale to rgb
            image = torch.cat([image, image, image], dim=0)

        return image, torch.tensor(src_label).float()

    def __len__(self):
        """Return the number of images."""
        return self.num_images
