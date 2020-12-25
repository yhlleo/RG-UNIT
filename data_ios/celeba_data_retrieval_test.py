
import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils import data

random.seed(1234)

class CelebA_retrieval_test(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, image_dir, train_list_path, test_list_path, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.mode = mode
        self.train_dataset = self.load_files(train_list_path)
        self.test_dataset  = self.load_files(test_list_path)

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
        self.rand = np.random.RandomState(100)

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

        # -------------- Random target label from the dataset -------------- #
        trg_idx = self.rand.randint(0,self.num_images)
        _, trg_label = dataset[trg_idx]
        # -------------------------------------------- #

        image = Image.open(os.path.join(self.image_dir, filename))
        image = self.transform(image)
        if image.size(0) == 1: # convert grayscale to rgb
            image = torch.cat([image, image, image], dim=0)

        trg_label = torch.tensor(trg_label).float()
        src_label = torch.tensor(src_label).float()
        return image, filename, trg_label, src_label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

